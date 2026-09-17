import argparse
import csv
import json
import os
import sys
import time

import yaml
from sklearn.metrics import mean_squared_error

sys.path.append(os.getcwd())

from scripts import baseline, benchmark, split  # noqa: E402
from scripts.evaluate_oob import evaluate_oob  # noqa: E402
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
    predictions = []

    for features, true_value in zip(X_test.values.tolist(), y_test.tolist()):
        status, body = benchmark.http_json("POST", f"{master_url}/predict/{model_id}", {"features": features, "task_type": task_type})
        if status != 200:
            raise RuntimeError(f"POST /predict rejected ({status}): {body}")
        if body.get("warning"):
            partial += 1
        predictions.append(body["prediction"])

        if task_type == "classification":
            if body["prediction"] == str(true_value):
                correct += 1
        else:
            squared_errors.append((float(body["prediction"]) - true_value) ** 2)

    if task_type == "classification":
        return {"accuracy": correct / len(y_test)}, partial, predictions
    mse = sum(squared_errors) / len(squared_errors)
    return {"mse": mse, "rmse": mse ** 0.5}, partial, predictions


def evaluate_baseline(model, X_test, y_test, task_type: str):
    predictions = model.predict(X_test)
    if task_type == "classification":
        # String comparison, same convention as evaluate_distributed (whose
        # predictions always come back as strings over the wire) - so the two
        # accuracies are computed the exact same way, not just numerically
        # close by coincidence.
        correct = sum(1 for pred, true in zip(predictions, y_test) if str(pred) == str(true))
        return {"accuracy": correct / len(y_test)}, [str(p) for p in predictions]
    mse = mean_squared_error(y_test, predictions)
    return {"mse": mse, "rmse": mse ** 0.5}, list(predictions)


def run_evaluation(cfg, train_req, name, write_files=True, test_sample_size=None):
    """Trains one model, measures accuracy against the baseline and OOB error
    on the same held-out split, optionally writing accuracy_results_{name}.csv
    / predictions_{name}.csv / oob_results_{name}.csv. write_files=False skips
    all three (used by scripts/validation_curve.py, which sweeps n_estimators
    on the same name/dataset and writes its own aggregate CSV instead).
    test_sample_size caps how many test rows go through the live /predict
    loop (both models, same rows) - full test set if left unset.
    Returns (dist_metrics, base_metrics, oob_score_name, oob_score, train_time_s)."""
    evaluation = cfg["evaluation"]
    master_url = evaluation["master_url"].rstrip("/")
    task_type = train_req["task_type"]
    target_column = train_req["target_column"]
    n_estimators = train_req.get("n_estimators") or cfg["system"]["default_n_estimators"]
    client = baseline.s3_client(cfg["storage"])
    bucket = cfg["storage"]["bucket"]

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
    if test_sample_size and test_sample_size < len(X_test):
        X_test, y_test = X_test.iloc[:test_sample_size], y_test.iloc[:test_sample_size]

    print(f"[Evaluation] {task_type} on {train_key}, {n_estimators} trees, {len(X_test)} test rows")

    train_start = time.perf_counter()
    status, body = benchmark.http_json("POST", f"{master_url}/train", train_req)
    if status != 202:
        raise RuntimeError(f"POST /train rejected ({status}): {body}")
    model_id = body["model_id"]
    benchmark.wait_until_ready(master_url, model_id, evaluation["poll_interval_seconds"], cfg["system"]["timeout_training_seconds"])
    train_time_s = time.perf_counter() - train_start

    dist_metrics, partial, dist_predictions = evaluate_distributed(master_url, model_id, X_test, y_test, task_type)
    if partial:
        print(f"[Evaluation] {partial}/{len(X_test)} distributed predictions were partial (a worker exhausted its retries)")

    dataset_path = baseline.download_dataset(client, bucket, train_key)
    defaults = cfg["model_defaults"][task_type]
    model = baseline.load_and_train(dataset_path, task_type, target_column, n_estimators, defaults, train_req.get("hyperparameters", {}))
    os.remove(dataset_path)
    base_metrics, base_predictions = evaluate_baseline(model, X_test, y_test, task_type)

    if write_files:
        rows = []
        for model_name, metrics, partial_count in (("distributed", dist_metrics, partial), ("baseline", base_metrics, 0)):
            rows.append({
                "model": model_name,
                "task_type": task_type,
                "dataset": train_key,
                "n_estimators": n_estimators,
                "test_rows": len(X_test),
                "accuracy": metrics.get("accuracy", ""),
                "mse": metrics.get("mse", ""),
                "rmse": metrics.get("rmse", ""),
                "partial_predictions": partial_count,
            })

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

        if task_type == "classification":
            predictions_csv = os.path.join(output_dir, f"predictions_{name}.csv")
            with open(predictions_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["row", "true_label", "distributed_prediction", "baseline_prediction"])
                writer.writeheader()
                for row, (true_label, dist_pred, base_pred) in enumerate(zip(y_test.tolist(), dist_predictions, base_predictions)):
                    writer.writerow({"row": row, "true_label": true_label, "distributed_prediction": dist_pred, "baseline_prediction": base_pred})
            print(f"[Evaluation] per-row predictions written to {predictions_csv}")

    oob_name, oob_score = evaluate_oob(cfg, client, bucket, model_id, name, write_csv=write_files)
    return dist_metrics, base_metrics, oob_name, oob_score, train_time_s


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy/error of the distributed system and the baseline on the same held-out test set.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--train-request")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    request_path = args.train_request or cfg["evaluation"]["train_request"]
    with open(request_path) as f:
        train_req = json.load(f)
    name = os.path.splitext(os.path.basename(train_req["dataset_url"]))[0]

    run_evaluation(cfg, train_req, name)


if __name__ == "__main__":
    main()
