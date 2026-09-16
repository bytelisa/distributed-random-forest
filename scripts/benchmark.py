import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

import yaml

sys.path.append(os.getcwd())

from scripts import baseline, split  # noqa: E402

# Timing comparison between the distributed system and the non-distributed
# baseline (scripts/baseline.py). Every run of both models is driven by the
# same two request files named in the "evaluation" section of the config:
# edit train_request.json / predict_request.json to change what is measured.
# dataset_url there is the source dataset: it is split into train/test
# once, before any timing starts, and both models train on the train part.
#
# Distributed training time is what the user experiences: from sending
# POST /train to GET /models/{id} first answering "ready". Baseline training
# time starts from the same place (the dataset on S3) and covers download,
# preprocessing and fit. Each distributed model is deleted from S3 right
# after its prediction, so a run leaves nothing behind.


def http_json(method: str, url: str, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_until_ready(master_url: str, model_id: str, poll_interval: float, timeout: float):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        status, body = http_json("GET", f"{master_url}/models/{model_id}")
        if status == 200 and body["status"] == "ready":
            return
        if status == 200 and body["status"] == "failed":
            raise RuntimeError(f"model {model_id} failed: {body.get('message')}")
        time.sleep(poll_interval)
    raise TimeoutError(f"model {model_id} not ready after {timeout}s")


def delete_model(client, bucket: str, model_id: str):
    prefix = f"models/{model_id}/"
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=bucket, Delete={"Objects": keys})


def run_distributed(cfg: dict, train_req: dict, predict_req: dict, client):
    evaluation = cfg["evaluation"]
    master_url = evaluation["master_url"].rstrip("/")
    bucket = cfg["storage"]["bucket"]
    train_timeout = cfg["system"]["timeout_training_seconds"]
    train_times, predict_times = [], []

    for run in range(evaluation["runs"]):
        start = time.perf_counter()
        status, body = http_json("POST", f"{master_url}/train", train_req)
        if status != 202:
            raise RuntimeError(f"POST /train rejected ({status}): {body}")
        model_id = body["model_id"]
        wait_until_ready(master_url, model_id, evaluation["poll_interval_seconds"], train_timeout)
        train_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        status, body = http_json("POST", f"{master_url}/predict/{model_id}", predict_req)
        if status != 200:
            raise RuntimeError(f"POST /predict rejected ({status}): {body}")
        predict_times.append(time.perf_counter() - start)
        if body.get("warning"):
            print(f"[Benchmark] run {run + 1}: partial prediction - {body['warning']}")

        delete_model(client, bucket, model_id)
        print(f"[Benchmark] distributed run {run + 1}/{evaluation['runs']}: train {train_times[-1]:.3f}s, predict {predict_times[-1] * 1000:.1f}ms")

    return train_times, predict_times


def run_baseline(cfg: dict, train_req: dict, predict_req: dict, client, n_estimators: int):
    bucket = cfg["storage"]["bucket"]
    runs = cfg["evaluation"]["runs"]
    defaults = cfg["model_defaults"][train_req["task_type"]]
    train_times, predict_times = [], []

    for run in range(runs):
        start = time.perf_counter()
        dataset_path = baseline.download_dataset(client, bucket, train_req["dataset_url"])
        model = baseline.load_and_train(
            dataset_path,
            train_req["task_type"],
            train_req["target_column"],
            n_estimators,
            defaults,
            train_req.get("hyperparameters", {}),
        )
        train_times.append(time.perf_counter() - start)
        os.remove(dataset_path)

        start = time.perf_counter()
        baseline.predict(model, predict_req["features"])
        predict_times.append(time.perf_counter() - start)
        print(f"[Benchmark] baseline run {run + 1}/{runs}: train {train_times[-1]:.3f}s, predict {predict_times[-1] * 1000:.1f}ms")

    return train_times, predict_times


def stats(values):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Collect train/predict timings for the distributed system and the baseline.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluation = cfg["evaluation"]
    with open(evaluation["train_request"]) as f:
        train_req = json.load(f)
    with open(evaluation["predict_request"]) as f:
        predict_req = json.load(f)

    if predict_req["task_type"] != train_req["task_type"]:
        sys.exit(f"task_type mismatch: train={train_req['task_type']} predict={predict_req['task_type']}")

    # The master applies this default too, so both models train the same forest size
    n_estimators = train_req.get("n_estimators") or cfg["system"]["default_n_estimators"]
    client = baseline.s3_client(cfg["storage"])

    # Split first, outside any timing; stratified only for classification
    # (there are no classes to balance in a regression target)
    source_key = train_req["dataset_url"]
    name = os.path.splitext(os.path.basename(source_key))[0]
    stratify_on = train_req["target_column"] if train_req["task_type"] == "classification" else None
    train_key, _ = split.split_and_upload(
        client, cfg["storage"]["bucket"], source_key, name,
        evaluation["test_size"], evaluation["split_seed"], stratify_on,
    )
    train_req = {**train_req, "dataset_url": train_key}

    print(f"[Benchmark] {evaluation['runs']} runs, {n_estimators} trees, {train_req['task_type']} on {train_key}")
    dist_train, dist_predict = run_distributed(cfg, train_req, predict_req, client)
    base_train, base_predict = run_baseline(cfg, train_req, predict_req, client, n_estimators)

    rows = []
    for model_name, train_times, predict_times in (("distributed", dist_train, dist_predict), ("baseline", base_train, base_predict)):
        train_mean, train_std = stats(train_times)
        predict_mean, predict_std = stats(predict_times)
        rows.append({
            "model": model_name,
            "task_type": train_req["task_type"],
            "dataset": train_req["dataset_url"],
            "n_estimators": n_estimators,
            "runs": len(train_times),
            "train_mean_s": f"{train_mean:.6f}",
            "train_std_s": f"{train_std:.6f}",
            "predict_mean_s": f"{predict_mean:.6f}",
            "predict_std_s": f"{predict_std:.6f}",
        })

    # One file per source dataset, so evaluating a second task never
    # overwrites a previous one's results.
    output_dir = evaluation["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"evaluation_results_{name}.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Benchmark] results written to {output_csv}")
    for row in rows:
        print(f"  {row['model']:>11}: train {row['train_mean_s']}s +/- {row['train_std_s']}, predict {row['predict_mean_s']}s +/- {row['predict_std_s']}")


if __name__ == "__main__":
    main()
