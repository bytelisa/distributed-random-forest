import argparse
import csv
import os
import subprocess
import sys
import time

import grpc
import scipy.stats
import yaml

sys.path.append(os.getcwd())

from api.proto.worker.v1 import worker_pb2, worker_pb2_grpc  # noqa: E402
from scripts import baseline, benchmark, split  # noqa: E402
from services.worker.ml import model as ml_model  # noqa: E402

# Scalability analysis: n_estimators fixed, the changing configuration is
# dataset size x worker count, averaged on a configurable number of runs (eg. 10).
# Results are written to CSV after EACH configuration (not just at the end), so a crash
# partway through (e.g. at 8 workers) doesn't lose the configurations already done.
#
# Expects the synthetic datasets to already be on S3, named
# synthetic_<size>.csv (e.g. synthetic_10k.csv) - classification, produced by
# the (separate) synthetic dataset generator.
#
# Worker count is changed by this script itself, via the "worker_control"
# section of the config - two backends:
#   - "docker": starts/stops docker compose worker-N services locally.
#   - "aws_ec2": starts/stops the EC2 worker instances themselves via boto3
#     (NOT the process on them). Still need to test on AWS.


def wait_for_workers_ready(addresses: list, num_workers: int, health_check_timeout: float, max_wait_seconds: float, poll_interval: float = 1.0):
    """Polls the first num_workers addresses over gRPC Health until all answer
    healthy, or raises once max_wait_seconds elapses. A started
    container/instance isn't the same as a worker process ready to serve:
    the master itself won't error on a not-yet-ready worker, it silently
    routes trees around it, so an unverified sleep can quietly measure a
    lower worker count than the one being recorded."""
    pending = set(addresses[:num_workers])
    deadline = time.perf_counter() + max_wait_seconds
    while pending:
        for addr in list(pending):
            try:
                with grpc.insecure_channel(addr) as channel:
                    stub = worker_pb2_grpc.WorkerStub(channel)
                    if stub.Health(worker_pb2.HealthRequest(), timeout=health_check_timeout).healthy:
                        pending.discard(addr)
            except grpc.RpcError:
                pass
        if not pending:
            return
        if time.perf_counter() >= deadline:
            raise RuntimeError(f"worker(s) not ready after {max_wait_seconds}s: {sorted(pending)}")
        time.sleep(poll_interval)


def set_docker_workers(prefix: str, total_slots: int, num_workers: int, addresses: list, health_check_timeout: float, settle_seconds: float):
    to_start = [f"{prefix}{i}" for i in range(1, num_workers + 1)]
    to_stop = [f"{prefix}{i}" for i in range(num_workers + 1, total_slots + 1)]
    if to_start:
        # "up -d", not "start": worker-4..N may never have been created by
        # the default "docker compose up -d" (only worker-1..3 are), and
        # "start" only works on an already-created, stopped container.
        subprocess.run(["docker", "compose", "up", "-d"] + to_start, check=True, capture_output=True, text=True)
    if to_stop:
        subprocess.run(["docker", "compose", "stop"] + to_stop, check=True, capture_output=True, text=True)
    print(f"[Scalability] docker: {num_workers} worker(s) up ({', '.join(to_start) if to_start else 'none'}), rest stopped. Waiting for them to answer Health (up to {settle_seconds}s).")
    wait_for_workers_ready(addresses, num_workers, health_check_timeout, settle_seconds)


def set_aws_workers(instance_ids: list, region: str, num_workers: int, addresses: list, health_check_timeout: float, settle_seconds: float, probe: str = "public"):
    import boto3

    if num_workers > len(instance_ids):
        raise ValueError(f"requested {num_workers} workers but only {len(instance_ids)} instance_ids configured")

    ec2 = boto3.client("ec2", region_name=region)
    to_start = instance_ids[:num_workers]
    to_stop = instance_ids[num_workers:]

    if to_start:
        ec2.start_instances(InstanceIds=to_start)
        ec2.get_waiter("instance_running").wait(InstanceIds=to_start)
    if to_stop:
        ec2.stop_instances(InstanceIds=to_stop)

    if probe == "private":
        # Running inside the VPC (e.g. on a worker instance): the private
        # addresses the master dials are reachable directly.
        probe_addresses = addresses[:num_workers]
    else:
        # Running from outside the VPC: workers.addresses holds private IPs,
        # unreachable from here - and the public ones change on every
        # stop/start, so they can't live in the config. Read them fresh.
        public_ip = {}
        for reservation in ec2.describe_instances(InstanceIds=to_start)["Reservations"]:
            for inst in reservation["Instances"]:
                public_ip[inst["InstanceId"]] = inst.get("PublicIpAddress")
        probe_addresses = []
        for i, instance_id in enumerate(to_start):
            if not public_ip.get(instance_id):
                raise RuntimeError(f"instance {instance_id} is running but has no public IP to probe")
            port = addresses[i].rsplit(":", 1)[1]
            probe_addresses.append(f"{public_ip[instance_id]}:{port}")

    print(f"[Scalability] aws_ec2: {num_workers} instance(s) running, {len(to_stop)} stopped. Waiting for the worker process to answer Health (up to {settle_seconds}s).")
    wait_for_workers_ready(probe_addresses, num_workers, health_check_timeout, settle_seconds)


def set_worker_count(worker_control: dict, addresses: list, health_check_timeout: float, num_workers: int):
    backend = worker_control["backend"]
    settle = worker_control["settle_seconds"]
    if backend == "docker":
        set_docker_workers(worker_control["docker_service_prefix"], worker_control["total_slots"], num_workers, addresses, health_check_timeout, settle)
    elif backend == "aws_ec2":
        set_aws_workers(worker_control["instance_ids"], worker_control["region"], num_workers, addresses, health_check_timeout, settle, worker_control.get("probe", "public"))
    else:
        raise ValueError(f"unknown worker_control.backend {backend!r}")


def size_sort_key(label: str) -> int:
    return int(label[:-1]) * 1000 if label.endswith("k") else int(label)


def mean_and_ci95(values):
    """Sample mean and the +/- margin of a 95% confidence interval (Student's t,
    appropriate for the small sample sizes used here)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    margin = scipy.stats.t.ppf(0.975, df=n - 1) * std / (n ** 0.5)
    return mean, margin


def prepare_size(cfg, source_key, name, target_column):
    """Splits a source dataset once and extracts one real feature row to reuse
    as the fixed predict payload - both are deterministic (fixed split seed)
    and neither counts toward measured train/predict time, so this is done
    once per dataset size and reused across every worker-count configuration,
    instead of being redone for each of them."""
    client = baseline.s3_client(cfg["storage"])
    bucket = cfg["storage"]["bucket"]
    evaluation = cfg["evaluation"]

    train_key, _ = split.split_and_upload(
        client, bucket, source_key, name,
        evaluation["test_size"], evaluation["split_seed"], target_column,
    )

    # One real row, reused as the fixed predict payload for every repetition,
    # taken from the data itself so its shape always matches the training set.
    # Reason: comparing times and not accuracy, so averaging on the same requests
    # gives a more statistically valid and comparable result (i think?)
    # todo: check again this. What if it's a specific request that performs better in one of the two cases? But it's always a RF with the same Hps... mmh
    train_path = baseline.download_dataset(client, bucket, train_key)
    df = ml_model.load_dataset(train_path)
    X, _ = ml_model.prepare_features(df, target_column)
    sample_features = X.iloc[0].tolist()
    os.remove(train_path)

    return train_key, sample_features


def run_one_size(cfg, master_url, train_key, sample_features, n_estimators, target_column, runs):
    client = baseline.s3_client(cfg["storage"])
    bucket = cfg["storage"]["bucket"]
    evaluation = cfg["evaluation"]

    train_req = {
        "dataset_url": train_key,
        "task_type": "classification",
        "target_column": target_column,
        "n_estimators": n_estimators,
    }
    predict_req = {"features": sample_features, "task_type": "classification"}

    train_times, predict_times = [], []
    for run in range(runs):
        start = time.perf_counter()
        status, body = benchmark.http_json("POST", f"{master_url}/train", train_req)
        if status != 202:
            raise RuntimeError(f"POST /train rejected ({status}): {body}")
        model_id = body["model_id"]
        benchmark.wait_until_ready(master_url, model_id, evaluation["poll_interval_seconds"], cfg["system"]["timeout_training_seconds"])
        train_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        status, body = benchmark.http_json("POST", f"{master_url}/predict/{model_id}", predict_req)
        if status != 200:
            raise RuntimeError(f"POST /predict rejected ({status}): {body}")
        predict_times.append(time.perf_counter() - start)

        benchmark.delete_model(client, bucket, model_id)
        print(f"[Scalability] {train_key}, run {run + 1}/{runs}: train {train_times[-1]:.3f}s, predict {predict_times[-1] * 1000:.1f}ms")

    return train_times, predict_times


FIELDNAMES = ["dataset_size", "num_workers", "n_estimators", "runs",
              "train_mean_s", "train_ci95_s", "predict_mean_s", "predict_ci95_s"]

# One row per individual run, not per configuration - the aggregate CSV above
# only keeps mean/CI95, discarding the underlying distribution. Kept
# separately so a box plot (or any other per-run analysis) doesn't need to
# be decided on before running the campaign.
RAW_FIELDNAMES = ["dataset_size", "num_workers", "run", "train_s", "predict_s"]


def load_existing(output_csv):
    results = {}
    if os.path.exists(output_csv):
        with open(output_csv, newline="") as f:
            for row in csv.DictReader(f):
                results[(row["dataset_size"], int(row["num_workers"]))] = row
    return results


def write_csv(output_csv, results):
    rows_sorted = sorted(results.values(), key=lambda r: (size_sort_key(str(r["dataset_size"])), int(r["num_workers"])))
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_sorted)


def load_existing_raw(raw_csv):
    if os.path.exists(raw_csv):
        with open(raw_csv, newline="") as f:
            return list(csv.DictReader(f))
    return []


def write_raw_csv(raw_csv, raw_rows):
    rows_sorted = sorted(raw_rows, key=lambda r: (size_sort_key(str(r["dataset_size"])), int(r["num_workers"]), int(r["run"])))
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_sorted)


def main():
    parser = argparse.ArgumentParser(description="Sweep dataset size x worker count, measuring train/predict time for each configuration.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true", help="re-run configurations already present in the output CSV instead of skipping them")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    scal = cfg["scalability"]
    evaluation = cfg["evaluation"]
    master_url = evaluation["master_url"].rstrip("/")

    output_dir = scal["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "scalability_results.csv")
    results = load_existing(output_csv)
    raw_csv = os.path.join(output_dir, "scalability_runs.csv")
    raw_rows = load_existing_raw(raw_csv)

    addresses = cfg["workers"]["addresses"]
    health_check_timeout = cfg["system"]["timeout_health_check_seconds"]

    # Splitting is deterministic (fixed seed) and isn't part of what's timed,
    # so each size is split once here and reused across every worker count,
    # instead of being redone inside the loop below.
    prepared = {
        size_label: prepare_size(cfg, f"synthetic_{size_label}.csv", f"synthetic_{size_label}", scal["target_column"])
        for size_label in scal["dataset_sizes"]
    }

    total = len(scal["worker_counts"]) * len(scal["dataset_sizes"])
    done = 0
    for num_workers in scal["worker_counts"]:
        # Skip starting/stopping workers entirely when every size for this
        # worker count is already in the CSV - no point restarting
        # containers/instances (and waiting settle_seconds) just to discover
        # afterwards there's nothing left to run.
        needs_run = args.force or any((s, num_workers) not in results for s in scal["dataset_sizes"])
        if not needs_run:
            for size_label in scal["dataset_sizes"]:
                done += 1
                print(f"[Scalability] ({done}/{total}) {size_label}, {num_workers} workers: already in {output_csv}, skipping (use --force to redo)")
            continue

        set_worker_count(scal["worker_control"], addresses, health_check_timeout, num_workers)

        for size_label in scal["dataset_sizes"]:
            done += 1
            key = (size_label, num_workers)
            if key in results and not args.force:
                print(f"[Scalability] ({done}/{total}) {size_label}, {num_workers} workers: already in {output_csv}, skipping (use --force to redo)")
                continue

            print(f"[Scalability] ({done}/{total}) {size_label}, {num_workers} workers, {scal['n_estimators']} trees")
            train_key, sample_features = prepared[size_label]
            train_times, predict_times = run_one_size(
                cfg, master_url, train_key, sample_features,
                scal["n_estimators"], scal["target_column"], scal["runs"],
            )
            train_mean, train_ci95 = mean_and_ci95(train_times)
            predict_mean, predict_ci95 = mean_and_ci95(predict_times)
            results[key] = {
                "dataset_size": size_label,
                "num_workers": num_workers,
                "n_estimators": scal["n_estimators"],
                "runs": len(train_times),
                "train_mean_s": f"{train_mean:.6f}",
                "train_ci95_s": f"{train_ci95:.6f}",
                "predict_mean_s": f"{predict_mean:.6f}",
                "predict_ci95_s": f"{predict_ci95:.6f}",
            }

            # Drop this configuration's previous raw rows (only relevant on
            # --force) before appending the fresh ones, so a redo doesn't
            # leave stale runs mixed in with the new ones.
            raw_rows = [r for r in raw_rows if not (r["dataset_size"] == size_label and int(r["num_workers"]) == num_workers)]
            for run_index, (train_s, predict_s) in enumerate(zip(train_times, predict_times)):
                raw_rows.append({
                    "dataset_size": size_label,
                    "num_workers": num_workers,
                    "run": run_index,
                    "train_s": f"{train_s:.6f}",
                    "predict_s": f"{predict_s:.6f}",
                })

            # Written after every configuration, not just at the end: a crash
            # on a later configuration doesn't lose the ones already done.
            write_csv(output_csv, results)
            write_raw_csv(raw_csv, raw_rows)
            print(f"[Scalability] saved to {output_csv} and {raw_csv}")

    print(f"[Scalability] done. {output_csv}")


if __name__ == "__main__":
    main()
