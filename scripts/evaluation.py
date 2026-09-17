import argparse
import csv
import json
import os
import sys

import yaml
from sklearn.metrics import mean_squared_error

sys.path.append(os.getcwd())

from scripts import baseline, benchmark, split  # noqa: E402
from services.worker.ml import model as ml_model  # noqa: E402

# Accuracy comparison between the distributed system and the non-distributed
# baseline, on the same held-out test set produced by scripts/split.py.
# One train per model (not repeated like in benchmark.py).
#
# Distributed accuracy is measured by calling the real POST /predict once per test row.
# Baseline accuracy uses scikit-learn's own predictions on the same rows.


def evaluate_distributed(master_url: str, model_id: str, X_test, y_test, task_type: str):
    correct = 0
    squared_errors = []
    partial = 0

    for features, true_value in zip(X_test.values.tolist(), y_test.tolist()):
        status, body = benchmark.http_json("POST", f"{master_url}/predict/{model_id}", {"features": features, "task_type": task_type})
        if status != 200:
            raise RuntimeError(f"POST /predict rejected ({status}): {body}")
        if body.get("warning"):
            partial += 1

        if task_type == "classification":
            if body["prediction"] == str(true_value):
                correct += 1
        else:
            squared_errors.append((float(body["prediction"]) - true_value) ** 2)

    if task_type == "classification":
        return {"accuracy": correct / len(y_test)}, partial
    mse = sum(squared_errors) / len(squared_errors)
    return {"mse": mse, "rmse": mse ** 0.5}, partial


def evaluate_baseline(model, X_test, y_test, task_type: str):
    if task_type == "classification":
        return {"accuracy": model.score(X_test, y_test)}
    mse = mean_squared_error(y_test, model.predict(X_test))
    return {"mse": mse, "rmse": mse ** 0.5}


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy/error of the distributed system and the baseline on the same held-out test set.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluation = cfg["evaluation"]
    master_url = evaluation["master_url"].rstrip("/")
    with open(evaluation["train_request"]) as f:
        train_req = json.load(f)

    task_type = train_req["task_type"]
    target_column = train_req["target_column"]
    n_estimators = train_req.get("n_estimators") or cfg["system"]["default_n_estimators"]
    client = baseline.s3_client(cfg["storage"])
    bucket = cfg["storage"]["bucket"]

    name = os.path.splitext(os.path.basename(train_req["dataset_url"]))[0]
    stratify_on = target_column if task_type == "classification" else None
    train_key, test_key = split.split_and_upload(
        client, bucket, train_req["dataset_url"], name,
        evaluation["test_size"], evaluation["split_seed"], stratify_on,
    )
    train_req = {**train_req, "dataset_url": train_key}

    test_path = baseline.download_dataset(client, bucket, test_key)
    test_df = ml_model.load_dataset(test_path)
    X_test, y_test = ml_model.prepare_features(test_df, target_column)
    os.remove(test_path)

    print(f"[Evaluation] {task_type} on {train_key}, {n_estimators} trees, {len(test_df)} test rows")

    status, body = benchmark.http_json("POST", f"{master_url}/train", train_req)
    if status != 202:
        raise RuntimeError(f"POST /train rejected ({status}): {body}")
    model_id = body["model_id"]
    benchmark.wait_until_ready(master_url, model_id, evaluation["poll_interval_seconds"], cfg["system"]["timeout_training_seconds"])

    dist_metrics, partial = evaluate_distributed(master_url, model_id, X_test, y_test, task_type)
    # Not deleted here on purpose: scripts/evaluate_oob.py needs this same
    # model_id alive to compute the OOB score on the same trained forest,
    # not a separately trained one. It deletes the model itself once done.
    if partial:
        print(f"[Evaluation] {partial}/{len(test_df)} distributed predictions were partial (a worker exhausted its retries)")

    dataset_path = baseline.download_dataset(client, bucket, train_key)
    defaults = cfg["model_defaults"][task_type]
    model = baseline.load_and_train(dataset_path, task_type, target_column, n_estimators, defaults, train_req.get("hyperparameters", {}))
    os.remove(dataset_path)
    base_metrics = evaluate_baseline(model, X_test, y_test, task_type)

    rows = []
    for model_name, metrics, partial_count in (("distributed", dist_metrics, partial), ("baseline", base_metrics, 0)):
        rows.append({
            "model": model_name,
            "task_type": task_type,
            "dataset": train_key,
            "n_estimators": n_estimators,
            "test_rows": len(test_df),
            "accuracy": metrics.get("accuracy", ""),
            "mse": metrics.get("mse", ""),
            "rmse": metrics.get("rmse", ""),
            "partial_predictions": partial_count,
        })

    # One file per source dataset, so evaluating a second task never
    # overwrites a previous one's results.
    output_dir = evaluation["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"accuracy_results_{name}.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Evaluation] results written to {output_csv}")
    for row in rows:
        if task_type == "classification":
            print(f"  {row['model']:>11}: accuracy {row['accuracy']:.4f}")
        else:
            print(f"  {row['model']:>11}: mse {row['mse']:.4f}, rmse {row['rmse']:.4f}")
    print(f"[Evaluation] distributed model {model_id} left on S3 - run scripts/evaluate_oob.py {model_id} to also score it on OOB, then delete it")


if __name__ == "__main__":
    main()
