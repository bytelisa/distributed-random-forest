
import os
import grpc
from api.proto.worker.v1 import worker_pb2
from api.proto.worker.v1 import worker_pb2_grpc
from services.worker.ml import model as ml_model
from services.worker.platform.storage import StorageManager
from services.worker.config import load_config

# Note: all functions take and return objects defined in the two generated _pb2.py files,
# which implement the interface worker.proto in python.

# Temporary local directory for processing files
LOCAL_TEMP_DIR = "temp_worker_data"

class WorkerService(worker_pb2_grpc.WorkerServicer):
    def __init__(self):
        # Load config
        self.cfg = load_config()

        # Initialize storage manager
        self.storage = StorageManager(self.cfg)

        # Use temp dir defined in yaml
        self.local_temp_dir = self.cfg.local_temp_dir
        os.makedirs(self.local_temp_dir, exist_ok=True)
        print(f"[Worker] Initialized with temp dir: {self.local_temp_dir}")

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
            model_filename = f"{request.model_id}.joblib"
            local_model_path = os.path.join(self.local_temp_dir, model_filename) # Usa config
            ml_model.save_model(trained_model, local_model_path)

            # Upload to S3
            s3_model_key = f"models/{model_filename}"
            self.storage.upload_file(local_model_path, s3_model_key)

            return worker_pb2.TrainResponse(
                success=True,
                message=f"Training completed. Saved to s3://{self.cfg.storage_bucket}/{s3_model_key}"
            )

        except Exception as e:
            print(f"[Error Train] {e}")
            return worker_pb2.TrainResponse(success=False, message=str(e))

    def Predict(self, request, context):
        print(f"[Worker] Predict request for model {request.model_id}")

        try:
            # DOWNLOAD MODEL FROM S3
            model_filename = f"{request.model_id}.joblib"
            local_model_path = os.path.join(self.local_temp_dir, model_filename)
            s3_model_key = f"models/{model_filename}"

            # Only download if we don't have it (caching optimization for later)
            # For now, let's always download to be safe (stateless)
            self.storage.download_file(s3_model_key, local_model_path)

            # PREDICT
            result_string = ml_model.load_and_predict(
                model_path=local_model_path,
                features=list(request.features)
            )

            return worker_pb2.PredictResponse(prediction=result_string)

        except Exception as e:
            print(f"[Error Predict] {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return worker_pb2.PredictResponse()