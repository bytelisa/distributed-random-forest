import argparse
import json
import os
import sys
import tempfile
import time

import boto3
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sys.path.append(os.getcwd())

from services.worker.ml import model as ml_model  # noqa: E402

# Non-distributed reference model: a plain scikit-learn RandomForest trained
# on the same training set, with the same preprocessing, defaults and
# hyperparameters the distributed system applies - the same estimator, built
# on one machine. Single-threaded on purpose (n_jobs=1): the workers train
# each tree single-threaded too.


def s3_client(storage: dict):
    kwargs = {"endpoint_url": storage["endpoint"]}
    if storage.get("access_key") and storage.get("secret_key"):
        kwargs["aws_access_key_id"] = storage["access_key"]
        kwargs["aws_secret_access_key"] = storage["secret_key"]
    return boto3.client("s3", **kwargs)


def download_dataset(client, bucket: str, key: str) -> str:
    local_path = os.path.join(tempfile.gettempdir(), f"baseline_{os.path.basename(key)}")
    client.download_file(bucket, key, local_path)
    return local_path


def load_and_train(dataset_path: str, task_type: str, target_column: str, n_estimators: int, hyperparameters: dict, seed=None):
    """Loads the CSV, preprocesses it like the worker does, fits one RandomForest."""
    df = ml_model.load_dataset(dataset_path)
    X, y = ml_model.prepare_features(df, target_column)

    params = ml_model.default_hyperparameters(task_type, X.shape[1])
    params.update(hyperparameters)

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1, **params)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1, **params)

    model.fit(X, y)
    return model


def predict(model, features: list) -> str:
    return str(model.predict(np.array(features).reshape(1, -1))[0])


def main():
    parser = argparse.ArgumentParser(description="Train and query the non-distributed baseline once, printing timings.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluation = cfg["evaluation"]
    with open(evaluation["train_request"]) as f:
        train_req = json.load(f)
    with open(evaluation["predict_request"]) as f:
        predict_req = json.load(f)

    n_estimators = train_req.get("n_estimators") or cfg["system"]["default_n_estimators"]
    client = s3_client(cfg["storage"])

    start = time.perf_counter()
    dataset_path = download_dataset(client, cfg["storage"]["bucket"], train_req["dataset_url"])
    model = load_and_train(
        dataset_path,
        train_req["task_type"],
        train_req["target_column"],
        n_estimators,
        train_req.get("hyperparameters", {}),
    )
    train_seconds = time.perf_counter() - start
    os.remove(dataset_path)

    start = time.perf_counter()
    prediction = predict(model, predict_req["features"])
    predict_seconds = time.perf_counter() - start

    print(f"[Baseline] {n_estimators} trees on {train_req['dataset_url']}: trained in {train_seconds:.3f}s")
    print(f"[Baseline] prediction: {prediction} ({predict_seconds * 1000:.2f} ms)")


if __name__ == "__main__":
    main()
