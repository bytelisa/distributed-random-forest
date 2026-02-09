import boto3
import os
from urllib.parse import urlparse

class StorageManager:
    def __init__(self, cfg):
        """
        Initializes the StorageManager with a configuration object.
        """
        self.bucket_name = cfg.storage_bucket

        print(f"[Storage] Initializing S3 client at {cfg.storage_endpoint}...")

        self.s3_client = boto3.client(
            's3',
            endpoint_url=cfg.storage_endpoint,
            aws_access_key_id=cfg.storage_access_key,
            aws_secret_access_key=cfg.storage_secret_key
        )

    def download_file(self, s3_url: str, local_path: str):
        """
        Downloads a file from S3 to a local path.
        Handles both 's3://bucket/key' URLs and simple object keys.
        """
        try:
            if s3_url.startswith("s3://"):
                parsed = urlparse(s3_url)
                # if url contains bucket then use it, otherwise use fallback on config
                bucket = parsed.netloc if parsed.netloc else self.bucket_name
                key = parsed.path.lstrip('/')
            else:
                bucket = self.bucket_name
                key = s3_url

            print(f"[Storage] Downloading s3://{bucket}/{key} to {local_path}...")

            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            self.s3_client.download_file(bucket, key, local_path)
            print("[Storage] Download completed.")
        except Exception as e:
            raise Exception(f"Failed to download from S3: {e}")

    def upload_file(self, local_path: str, s3_key: str):
        """
        Uploads a local file to S3.
        """
        try:
            print(f"[Storage] Uploading {local_path} to s3://{self.bucket_name}/{s3_key}...")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print("[Storage] Upload completed.")
        except Exception as e:
            raise Exception(f"Failed to upload to S3: {e}")