
import os
import grpc
from api.proto.worker.v1 import worker_pb2
from api.proto.worker.v1 import worker_pb2_grpc
from services.worker.ml import model as ml_model
from services.worker.platform.storage import StorageManager
from services.worker.config import load_config
import uuid

# Note: all functions take and return objects defined in the two generated _pb2.py files,
# which implement the interface worker.proto in python.

# Temporary local directory for processing files
LOCAL_TEMP_DIR = "temp_worker_data"

class WorkerService(worker_pb2_grpc.WorkerServicer):
    def __init__(self, worker_id):
        # Load config
        self.cfg = load_config()

        # Initialize storage manager
        self.storage = StorageManager(self.cfg)

        base_temp_dir = self.cfg.local_temp_dir
        self.local_temp_dir = os.path.join(base_temp_dir, str(worker_id))

        os.makedirs(self.local_temp_dir, exist_ok=True)
        print(f"[Worker {worker_id}] Initialized with isolated temp dir: {self.local_temp_dir}")

    def _convert_type(self, task_type_enum):
        # convert enum
        if task_type_enum == worker_pb2.TaskType.CLASSIFICATION_TASK:
            return "classification"
        elif task_type_enum == worker_pb2.TaskType.REGRESSION_TASK:
            return "regression"
        else:
            raise ValueError("Unknown task type")

    def Train(self, request, context):
        print(f"[Worker] Received Train request for model {request.model_id}")

        try:
            # Use self.local_temp_dir
            dataset_filename = os.path.basename(request.dataset_url)
            local_dataset_path = os.path.join(self.local_temp_dir, dataset_filename)

            self.storage.download_file(request.dataset_url, local_dataset_path)

            # Load dataset and train
            ml_task_type = self._convert_type(request.task_type)
            df = ml_model.load_dataset(local_dataset_path)

            trained_model = ml_model.train_model(
                data=df,
                target_column=request.target_column,
                task_type=ml_task_type,
                n_estimators=request.n_estimators
            )

            # Save model locally

            # unique suffix for this worker (necessary to not overwrite models of other workers)
            part_id = str(uuid.uuid4())
            model_filename = f"part_{part_id}.joblib"
            local_model_path = os.path.join(self.local_temp_dir, model_filename)
            ml_model.save_model(trained_model, local_model_path)

            # Upload to S3
            #request's model id is now a directory where all workers upload their models
            s3_model_key = f"models/{request.model_id}/{model_filename}"
            self.storage.upload_file(local_model_path, s3_model_key)

            return worker_pb2.TrainResponse(
                success=True,
                message=f"Training part completed. Saved to s3://{self.cfg.storage_bucket}/{s3_model_key}"
            )

        except Exception as e:
            print(f"[Error Train] {e}")
            return worker_pb2.TrainResponse(success=False, message=str(e))

    def Predict(self, request, context):
        print(f"[Worker] Predict request for model {request.model_id}")

        try:
            # 1. PREPARE DIRECTORIES
            # We use a specific folder for this model inside the worker's temp dir
            local_model_dir = os.path.join(self.local_temp_dir, request.model_id)
            os.makedirs(local_model_dir, exist_ok=True)

            # 2. LIST AND DOWNLOAD FROM S3
            s3_prefix = f"models/{request.model_id}/"
            s3_keys = self.storage.list_files(s3_prefix)

            if not s3_keys:
                print(f"[Worker] No model files found on S3 for {request.model_id}")
                # Return empty to signal "I don't have this model"
                return worker_pb2.PredictResponse(prediction="")

            predictions = []

            for s3_key in s3_keys:
                # Extract filename (e.g., part_uuid.joblib)
                filename = os.path.basename(s3_key)
                local_path = os.path.join(local_model_dir, filename)

                # Download only if not exists (caching strategy) or always overwrite
                # For safety in distributed env, we download.
                self.storage.download_file(s3_key, local_path)

                # 3. PREDICT ON SINGLE TREE/FOREST PART
                # Returns a string (e.g. "setosa" or "150.5")
                pred_str = ml_model.load_and_predict(
                    model_path=local_path,
                    features=list(request.features)
                )
                predictions.append(pred_str)

            # 4. LOCAL AGGREGATION
            if not predictions:
                return worker_pb2.PredictResponse(prediction="")

            final_prediction = self._aggregate_predictions(predictions)

            print(f"[Worker] Local aggregation of {len(predictions)} parts: {final_prediction}")
            return worker_pb2.PredictResponse(prediction=final_prediction)

        except Exception as e:
            print(f"[Error Predict] {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return worker_pb2.PredictResponse()