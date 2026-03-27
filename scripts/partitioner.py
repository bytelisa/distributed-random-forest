import argparse
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
        s3_client = boto3.client('s3', endpoint_url=args.s3_endpoint,
                                 aws_access_key_id=args.s3_access_key, aws_secret_access_key=args.s3_secret_key)

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
        return 0

    except Exception as e:
        print(f"Partitioner Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--s3-access-key", required=True)
    parser.add_argument("--s3-secret-key", required=True)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--num-partitions", type=int, required=True)
    parser.add_argument("--shuffle", action='store_true', default=True)

    sys.exit(run_partitioning(parser.parse_args()))