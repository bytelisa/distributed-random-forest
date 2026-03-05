
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
        print(f"         I am worker {request.worker_index} of {request.total_workers}")

        # Note: stateless behaviour
        # -> when prediction is asked, workers download trees from shared storage. No internal worker state.

        try:
            # 1. LIST FILES FROM S3
            s3_prefix = f"models/{request.model_id}/"
            s3_keys = self.storage.list_files(s3_prefix)

            if not s3_keys:
                # No file found
                return worker_pb2.PredictResponse(prediction="")

            if request.total_workers <= 0:
                print(f"[Error Predict] Invalid total_workers: {request.total_workers}")
                return worker_pb2.PredictResponse(prediction="") # O alza eccezione

            # 2. DETERMINISTIC PARTITIONING OF TRAINED TREES
            # Order files to have consistent partitioning across all workers
            s3_keys.sort()

            # [ROUND ROBIN] Only select files that correspond to this worker's index based on modular algebra
            # Example: 2 Workers.
            # Worker 0 takes 0, 2, 4...
            # Worker 1 takes 1, 3, 5...
            my_files = [k for i, k in enumerate(s3_keys) if i % request.total_workers == request.worker_index]

            if not my_files:
                print(f"[Worker] No files assigned to me (found {len(s3_keys)} total).")
                return worker_pb2.PredictResponse(prediction="")

            print(f"[Worker] Assigned {len(my_files)} files out of {len(s3_keys)} total.")

            # 3. DOWNLOAD & PREDICT
            prediction_results = [] # List to store raw predictions

            local_model_dir = os.path.join(self.local_temp_dir, request.model_id)
            os.makedirs(local_model_dir, exist_ok=True)

            for s3_key in my_files:
                filename = os.path.basename(s3_key)
                local_path = os.path.join(local_model_dir, filename)

                self.storage.download_file(s3_key, local_path)

                # Returns a single prediction (string) from one tree
                pred_str = ml_model.load_and_predict(
                    model_path=local_path,
                    features=list(request.features)
                )
                prediction_results.append(pred_str)

            print(f"[Worker] Returning {len(prediction_results)} partial predictions.")

            # 4. NO LOCAL AGGREGATION. Returns raw list (Master will aggregate).
            return worker_pb2.PredictResponse(predictions=prediction_results)

        except Exception as e:
            print(f"[Error Predict] {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return worker_pb2.PredictResponse()