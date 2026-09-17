import argparse
import csv
import json
import os
import sys

import yaml

sys.path.append(os.getcwd())

from scripts import baseline, benchmark  # noqa: E402
from services.worker.ml import model as ml_model  # noqa: E402

# Aggregates the per-tree out-of-bag artifacts a training run already left on
# S3 (services/worker/worker_service.py: tree_{i}_oob.json, next to
# tree_{i}.joblib) into one OOB score for a given, already-trained model.
# Read-only and independent of scikit-learn/scripts/bootstrap.py: each
# artifact already names its own rows and, for classification, the tree's
# own class labels - no tree or bootstrap sample needs to be reloaded.
#
# Current limit: missing artifacts (a worker died between uploading a tree and its OOB
# file) are not regenerated: reported, and the affected trees are simply
# absent from the aggregation - with n_estimators in the dozens or more, a
# couple of missing trees barely move the final score. A row can end up with
# zero contributing trees; such rows are excluded from the score
# and reported, the same way partial predictions are reported elsewhere.
#
# Deletes the model from S3 once done: the point of leaving it alive after
# scripts/evaluation.py is exactly to run this against it right after.


def read_json(client, bucket, key):
    obj = client.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def list_oob_artifacts(client, bucket, model_id, n_estimators):
    """Returns (present: {tree_index: s3_key}, missing: [tree_index, ...])."""
    prefix = f"models/{model_id}/model_parts/"
    paginator = client.get_paginator("list_objects_v2")
    present = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = os.path.basename(key)
            if filename.endswith("_oob.json"):
                tree_index = int(filename[len("tree_"):-len("_oob.json")])
                present[tree_index] = key
    missing = [i for i in range(n_estimators) if i not in present]
    return present, missing


def aggregate_classification(entries_by_row):
    """entries_by_row: {row: [(classes, probabilities), ...]} -> {row: predicted_label}.
    Same soft voting as internal/orchestrator/pool.go's aggregateClassification:
    sum probabilities per label across contributing trees, argmax at the end,
    ties broken by sorted label order."""
    predictions = {}
    for row, contributions in entries_by_row.items():
        sums = {}
        for classes, probabilities in contributions:
            for cls, prob in zip(classes, probabilities):
                sums[cls] = sums.get(cls, 0.0) + prob
        predictions[row] = max(sorted(sums), key=lambda c: sums[c])
    return predictions


def aggregate_regression(entries_by_row):
    return {row: sum(values) / len(values) for row, values in entries_by_row.items()}


def main():
    parser = argparse.ArgumentParser(description="Compute the OOB error of an already-trained model from the per-tree artifacts left on S3.")
    parser.add_argument("model_id")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    client = baseline.s3_client(cfg["storage"])
    bucket = cfg["storage"]["bucket"]

    meta = read_json(client, bucket, f"models/{args.model_id}/train_request.json")
    task_type = "classification" if meta["task_type"] == 1 else "regression"
    target_column = meta["target_column"]
    n_estimators = meta["n_estimators"]
    train_key = meta["dataset_url"]

    present, missing = list_oob_artifacts(client, bucket, args.model_id, n_estimators)
    if missing:
        print(f"[OOB] {len(missing)}/{n_estimators} trees have no OOB artifact (worker died before uploading it): {missing}")

    # Gather every tree's OOB contributions per row, keyed by row position in
    # the training file - the same index space the bootstrap sampling uses.
    by_row = {}
    for tree_index, key in present.items():
        for entry in read_json(client, bucket, key):
            row = entry["row"]
            by_row.setdefault(row, [])
            if task_type == "classification":
                by_row[row].append((entry["classes"], entry["probabilities"]))
            else:
                by_row[row].append(entry["value"])

    dataset_path = baseline.download_dataset(client, bucket, train_key)
    df = ml_model.load_dataset(dataset_path)
    _, y = ml_model.prepare_features(df, target_column)
    os.remove(dataset_path)

    excluded_rows = [row for row in range(len(df)) if row not in by_row]
    if excluded_rows:
        print(f"[OOB] {len(excluded_rows)}/{len(df)} rows had zero contributing trees (excluded from the score): {excluded_rows}")

    if task_type == "classification":
        predictions = aggregate_classification(by_row)
        correct = sum(1 for row, pred in predictions.items() if pred == str(y.iloc[row]))
        score_name = "oob_accuracy"
        score = correct / len(predictions) if predictions else float("nan")
    else:
        predictions = aggregate_regression(by_row)
        squared_errors = [(pred - y.iloc[row]) ** 2 for row, pred in predictions.items()]
        score_name = "oob_rmse"
        score = (sum(squared_errors) / len(squared_errors)) ** 0.5 if squared_errors else float("nan")

    print(f"[OOB] {task_type} on {train_key}, {n_estimators} trees, {len(predictions)}/{len(df)} rows scored: {score_name} = {score:.4f}")

    name = os.path.splitext(os.path.basename(train_key))[0]
    output_dir = cfg["evaluation"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"oob_results_{name}.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_id", "task_type", "dataset", "n_estimators",
            "rows_scored", "rows_excluded", "trees_missing_oob", score_name,
        ])
        writer.writeheader()
        writer.writerow({
            "model_id": args.model_id,
            "task_type": task_type,
            "dataset": train_key,
            "n_estimators": n_estimators,
            "rows_scored": len(predictions),
            "rows_excluded": len(excluded_rows),
            "trees_missing_oob": len(missing),
            score_name: f"{score:.6f}",
        })
    print(f"[OOB] results written to {output_csv}")

    benchmark.delete_model(client, bucket, args.model_id)
    print(f"[OOB] model {args.model_id} deleted from S3.")


if __name__ == "__main__":
    main()
