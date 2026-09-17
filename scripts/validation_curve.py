import argparse
import copy
import csv
import json
import os

import yaml

import sys
sys.path.append(os.getcwd())

from scripts.evaluation import run_evaluation  # noqa: E402

# OOB/accuracy vs n_estimators, fixed dataset and worker count - the
# convergence check for how many trees a Random Forest actually needs.

FIELDNAMES = ["n_estimators", "distributed_accuracy", "baseline_accuracy",
              "distributed_rmse", "baseline_rmse", "oob_score_name", "oob_score", "train_time_s"]


def load_existing(csv_path):
    results = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                results[int(row["n_estimators"])] = row
    return results


def write_csv(csv_path, results):
    rows_sorted = sorted(results.values(), key=lambda r: int(r["n_estimators"]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_sorted)


def main():
    parser = argparse.ArgumentParser(description="OOB/accuracy vs n_estimators, fixed dataset and worker count.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--train-request")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    vc = cfg["validation_curve"]
    request_path = args.train_request or vc["train_request"]
    with open(request_path) as f:
        base_req = json.load(f)
    name = os.path.splitext(os.path.basename(base_req["dataset_url"]))[0]

    output_dir = cfg["evaluation"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"validation_curve_{name}.csv")
    results = load_existing(csv_path)

    values = vc["n_estimators_values"]
    for i, n_estimators in enumerate(values, 1):
        if n_estimators in results and not args.force:
            print(f"[ValidationCurve] ({i}/{len(values)}) n_estimators={n_estimators}: already in {csv_path}, skipping")
            continue

        print(f"[ValidationCurve] ({i}/{len(values)}) n_estimators={n_estimators}")
        train_req = copy.deepcopy(base_req)
        train_req["n_estimators"] = n_estimators
        dist_metrics, base_metrics, oob_name, oob_score, train_time_s = run_evaluation(
            cfg, train_req, name, write_files=False, test_sample_size=vc.get("test_sample_size"),
        )

        results[n_estimators] = {
            "n_estimators": n_estimators,
            "distributed_accuracy": dist_metrics.get("accuracy", ""),
            "baseline_accuracy": base_metrics.get("accuracy", ""),
            "distributed_rmse": dist_metrics.get("rmse", ""),
            "baseline_rmse": base_metrics.get("rmse", ""),
            "oob_score_name": oob_name,
            "oob_score": f"{oob_score:.6f}",
            "train_time_s": f"{train_time_s:.6f}",
        }
        write_csv(csv_path, results)
        print(f"[ValidationCurve] saved {csv_path}")

    print(f"[ValidationCurve] done. {csv_path}")


if __name__ == "__main__":
    main()
