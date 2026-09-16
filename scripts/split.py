import argparse
import os
import sys
import tempfile

import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

# Splits a source dataset already on S3 into train/test once, uploading both
# as distinct objects on the same bucket. Not invoked by the master -
# dataset_url in a /train request just points at whatever train file this
# produced, like any other dataset.
#
# The row order of the uploaded train file is the canonical index space the
# bootstrap sampling (scripts/bootstrap.py) relies on: shuffling happens
# exactly once, here, before the split and the upload. Nothing downstream
# (worker, evaluation script, baseline) may reorder it afterwards.


def split_and_upload(client, bucket: str, source_key: str, name: str, test_size: float, seed: int, target_column=None):
    """Returns the (train_key, test_key) uploaded. Stratified on target_column when given."""
    tmp_dir = tempfile.gettempdir()
    local_source = os.path.join(tmp_dir, f"split_source_{name}.csv")
    client.download_file(bucket, source_key, local_source)

    df = pd.read_csv(local_source)
    os.remove(local_source)

    stratify = df[target_column] if target_column else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_key = f"{name}_train.csv"
    test_key = f"{name}_test.csv"
    local_train = os.path.join(tmp_dir, train_key)
    local_test = os.path.join(tmp_dir, test_key)
    train_df.to_csv(local_train, index=False)
    test_df.to_csv(local_test, index=False)
    client.upload_file(local_train, bucket, train_key)
    client.upload_file(local_test, bucket, test_key)
    os.remove(local_train)
    os.remove(local_test)

    print(f"[Split] {len(df)} rows -> {len(train_df)} train / {len(test_df)} test")
    print(f"[Split] Uploaded s3://{bucket}/{train_key}")
    print(f"[Split] Uploaded s3://{bucket}/{test_key}")
    return train_key, test_key


def run_split(args):
    try:
        client_kwargs = {"endpoint_url": args.s3_endpoint}
        if args.s3_access_key and args.s3_secret_key:
            client_kwargs["aws_access_key_id"] = args.s3_access_key
            client_kwargs["aws_secret_access_key"] = args.s3_secret_key
        client = boto3.client("s3", **client_kwargs)

        split_and_upload(client, args.s3_bucket, args.source_key, args.name, args.test_size, args.seed, args.target_column)
        return 0

    except Exception as e:
        print(f"Split error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a dataset on S3 into train/test, once.")
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--s3-access-key", default=None)
    parser.add_argument("--s3-secret-key", default=None)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--source-key", required=True, help="S3 key of the source dataset, e.g. iris.csv")
    parser.add_argument("--name", required=True, help="Base name for the outputs: <name>_train.csv / <name>_test.csv")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows held out for test (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Single source of randomness for the shuffle/split")
    parser.add_argument("--target-column", default=None, help="If given, split is stratified on this column")

    sys.exit(run_split(parser.parse_args()))
