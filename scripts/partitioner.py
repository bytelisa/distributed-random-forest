import argparse
import json
import os
import sys
import boto3
import numpy as np
import pandas as pd

# Master's module responsible for dataset partitioning.
# Partitions a dataset and uploads the parts to a specific S3 folder.
# Fails silently with exit code 0 on success, or >0 on error.

def run_partitioning(args):
    try:
        # If access/secret key are not provided, omits them so boto3
        # falls back to its default credentials (e.g. the IAM role
        # attached to the EC2 instance).
        client_kwargs = {"endpoint_url": args.s3_endpoint}
        if args.s3_access_key and args.s3_secret_key:
            client_kwargs["aws_access_key_id"] = args.s3_access_key
            client_kwargs["aws_secret_access_key"] = args.s3_secret_key

        s3_client = boto3.client('s3', **client_kwargs)

        # Download source dataset
        local_source = f"/tmp/source_{args.model_id}.csv"
        s3_client.download_file(args.s3_bucket, args.source_key, local_source)

        # Read and optionally shuffle
        df = pd.read_csv(local_source)
        if args.shuffle:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into N parts
        df_parts = np.array_split(df, args.num_partitions)

        # Base folder for this model's dataset partitions
        base_prefix = f"models/{args.model_id}/dataset_partitions"

        # Upload each part
        for i, part_df in enumerate(df_parts):
            local_part = f"/tmp/part_{i}.csv"
            part_df.to_csv(local_part, index=False)

            s3_key = f"{base_prefix}/part_{i}.csv"
            s3_client.upload_file(local_part, args.s3_bucket, s3_key)
            os.remove(local_part)

        os.remove(local_source)

        # Train request metadata: writes a small metadata file with the original training parameters.
        # This is what lets a new master continue a training started with the previous master.
        train_request = {
            "task_type": args.task_type,
            "target_column": args.target_column,
            "n_estimators": args.n_estimators,
            "total_partitions": args.num_partitions,
        }
        local_meta = f"/tmp/train_request_{args.model_id}.json"
        with open(local_meta, "w") as f:
            json.dump(train_request, f)
        s3_client.upload_file(local_meta, args.s3_bucket, f"models/{args.model_id}/train_request.json")
        os.remove(local_meta)

        return 0

    except Exception as e:
        print(f"Partitioner Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--s3-access-key", default=None)
    parser.add_argument("--s3-secret-key", default=None)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--num-partitions", type=int, required=True)
    parser.add_argument("--shuffle", action='store_true', default=True)
    # Original training parameters, persisted alongside the partitions so a
    # missing model part can be reissued later without the original HTTP
    # request (see scripts/reconciler.py).
    parser.add_argument("--task-type", type=int, required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--n-estimators", type=int, required=True)

    sys.exit(run_partitioning(parser.parse_args()))