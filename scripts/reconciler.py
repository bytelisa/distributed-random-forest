import argparse
import json
import re
import sys
import boto3

# Master's module responsible for detecting trainings left incomplete by a
# master crash. Run at startup by the (possibly new) master instance.
#
# For every model_id under models/ on S3:
#   - if models/{model_id}/train_request.json is missing, partitioning
#     itself never finished for that model: in this case the user is
#     expected to send a new request.
#   - otherwise, compare how many dataset partitions were created against
#     how many model parts (forest_part_{index}.joblib) are already on S3,
#     and report which partition indices are still missing to reassign
#     training to healthy workers.
#
# Prints a JSON array to stdout, one entry per model with missing parts:
#   [{"model_id": ..., "task_type": ..., "target_column": ...,
#     "n_estimators": ..., "total_partitions": ..., "missing_indices": [...]}]
# Prints nothing (empty array) if every model is complete. Only reads from
# S3, never writes or dispatches training itself - that part is done by the
# Go master (internal/orchestrator/reconcile.go) using the output of this
# script.

PART_INDEX_RE = re.compile(r"forest_part_(\d+)\.joblib$")


def build_s3_client(args):
    client_kwargs = {"endpoint_url": args.s3_endpoint}
    if args.s3_access_key and args.s3_secret_key:
        client_kwargs["aws_access_key_id"] = args.s3_access_key
        client_kwargs["aws_secret_access_key"] = args.s3_secret_key
    return boto3.client('s3', **client_kwargs)


def list_model_ids(s3_client, bucket):
    """Top-level 'folders' directly under models/, i.e. one per model_id."""
    model_ids = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix="models/", Delimiter="/"):
        for common_prefix in page.get('CommonPrefixes', []):
            # common_prefix['Prefix'] looks like "models/<model_id>/"
            model_id = common_prefix['Prefix'].removeprefix("models/").rstrip("/")
            if model_id:
                model_ids.append(model_id)
    return model_ids


def list_keys(s3_client, bucket, prefix):
    keys = []
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            keys.append(obj['Key'])
    return keys


def read_json_object(s3_client, bucket, key):
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj['Body'].read())
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception:
        # Any other error (e.g. malformed JSON) is treated the same as
        # "not there": skip this model, don't crash the whole reconciliation.
        return None


def find_incomplete_models(s3_client, bucket):
    incomplete = []

    for model_id in list_model_ids(s3_client, bucket):
        train_request = read_json_object(s3_client, bucket, f"models/{model_id}/train_request.json")
        if train_request is None:
            # Partitioning never completed for this model ->  not auto-recovered
            continue

        total_partitions = train_request["total_partitions"]

        model_part_keys = list_keys(s3_client, bucket, f"models/{model_id}/model_parts/")
        present_indices = set()
        for key in model_part_keys:
            match = PART_INDEX_RE.search(key)
            if match:
                present_indices.add(int(match.group(1)))

        missing_indices = sorted(set(range(total_partitions)) - present_indices)
        if missing_indices:
            incomplete.append({
                "model_id": model_id,
                "task_type": train_request["task_type"],
                "target_column": train_request["target_column"],
                "n_estimators": train_request["n_estimators"],
                "total_partitions": total_partitions,
                "missing_indices": missing_indices,
            })

    return incomplete


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--s3-access-key", default=None)
    parser.add_argument("--s3-secret-key", default=None)
    parser.add_argument("--s3-bucket", required=True)

    args = parser.parse_args()

    try:
        client = build_s3_client(args)
        result = find_incomplete_models(client, args.s3_bucket)
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        print(f"Reconciler error: {e}", file=sys.stderr)
        sys.exit(1)
