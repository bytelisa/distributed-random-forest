import argparse
import os
import sys
import tempfile

import pandas as pd
import yaml
from sklearn.datasets import make_classification

sys.path.append(os.getcwd())

from scripts import baseline  # noqa: E402

# Generates synthetic_<size>.csv datasets for the scalability study
# (scripts/scalability.py), uploaded to S3 under the same key
# convention as iris/housing/sdss. n_samples per size come from
# scalability.dataset_sizes, so the two scripts can't drift out of sync.
#
# Column layout mirrors SDSS on purpose (see configs/config.yaml,
# "synthetic" section, for the exact mapping and the reason):
# a few informative features, a few redundant ones correlated with them,
# the rest  noise. shuffle=False keeps that column order intact so the
# configured column names follow the same order as in SDSS.


def parse_size(label: str) -> int:
    return int(label[:-1]) * 1000 if label.endswith("k") else int(label)


def generate(size_label: str, syn: dict, target_column: str) -> pd.DataFrame:
    columns = syn["columns"]
    all_columns = columns["informative"] + columns["redundant"] + columns["noise"]
    if len(all_columns) != syn["n_features"]:
        raise ValueError(f"synthetic.columns lists {len(all_columns)} names but n_features is {syn['n_features']}")

    X, y = make_classification(
        n_samples=parse_size(size_label),
        n_features=syn["n_features"],
        n_informative=syn["n_informative"],
        n_redundant=syn["n_redundant"],
        n_classes=syn["n_classes"],
        class_sep=syn["class_sep"],
        shuffle=False,
        random_state=syn["seed"],
    )
    df = pd.DataFrame(X, columns=all_columns)
    df[target_column] = y
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate and upload synthetic_<size>.csv datasets for the scalability study.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    syn = cfg["synthetic"]
    target_column = cfg["scalability"]["target_column"]

    client = baseline.s3_client(cfg["storage"])
    bucket = cfg["storage"]["bucket"]
    tmp_dir = tempfile.gettempdir()

    for size_label in cfg["scalability"]["dataset_sizes"]:
        df = generate(size_label, syn, target_column)
        key = f"synthetic_{size_label}.csv"
        local_path = os.path.join(tmp_dir, key)
        df.to_csv(local_path, index=False)
        client.upload_file(local_path, bucket, key)
        os.remove(local_path)
        print(f"[Synthetic] {size_label}: {df.shape[0]} rows, {df.shape[1] - 1} features -> s3://{bucket}/{key}")
        print(f"[Synthetic]   class distribution: {df[target_column].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
