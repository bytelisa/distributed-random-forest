
import json
import os
import grpc
from api.proto.worker.v1 import worker_pb2
from api.proto.worker.v1 import worker_pb2_grpc
from scripts.bootstrap import generate_bootstrap_indices, tree_seed
from services.worker.ml import model as ml_model
from services.worker.platform.storage import StorageManager
from services.worker.config import load_config

# Note: all functions take and return objects defined in the two generated _pb2.py files,
# which implement the interface worker.proto in python.


def _coerce_hyperparameter(value: str):
    """
    Converts a hyperparameter value back from its string form to the Python type scikit-learn
    expects. Generic on purpose: the master already validated which keys
    and values are allowed (internal/api/hyperparameters.go), so this
    only needs to undo the string conversion, not re-validate anything
    """
    if value == "None":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value  # plain string, e.g. "sqrt", "gini", "balanced"


def _tree_key(model_id: str, tree_index: int) -> str:
    return f"models/{model_id}/model_parts/tree_{tree_index}.joblib"


def _oob_key(model_id: str, tree_index: int) -> str:
    return f"models/{model_id}/model_parts/tree_{tree_index}_oob.json"


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


    def Health(self, request, context):
        # Ping
        return worker_pb2.HealthResponse(healthy=True)


    def Train(self, request, context):

        # DEBUG
        print(f"---[Worker] TRAIN JOB STARTED ---")
        print(f"[Worker] Model ID: {request.model_id}")
        print(f"[Worker] Trees to train: {list(request.tree_indices)}")
        print(f"[Worker] Training set: {request.dataset_url}")

        try:
            # 1. DOWNLOAD THE FULL TRAINING SET, ONCE
            dataset_filename = os.path.basename(request.dataset_url)
            local_dataset_path = os.path.join(self.local_temp_dir, dataset_filename)
            self.storage.download_file(request.dataset_url, local_dataset_path)

            df = ml_model.load_dataset(local_dataset_path)
            X, y = ml_model.prepare_features(df, request.target_column)
            n_rows = len(df)
            task_type = self._convert_type(request.task_type)
            defaults = self.cfg.model_defaults(task_type)

            # Extra hyperparameters arrive as strings (protobuf map values
            # can't be mixed types) and are coerced back to the right
            # Python type here - the master already validated the keys and
            # value shapes before sending, so this is a generic conversion,
            # not a second round of semantic checks.
            hyperparams = {k: _coerce_hyperparameter(v) for k, v in request.hyperparameters.items()}

            # 2. ONE TREE AT A TIME, EACH ON ITS OWN BOOTSTRAP SAMPLE
            for tree_index in request.tree_indices:
                seed = tree_seed(request.model_id, tree_index)
                indices = generate_bootstrap_indices(request.model_id, tree_index, n_rows)

                tree = ml_model.train_tree(
                    X.iloc[indices],
                    y.iloc[indices],
                    task_type,
                    seed,
                    defaults,
                    **hyperparams
                )

                # 3. UPLOAD IMMEDIATELY: if this worker dies later, only the
                # tree in progress is lost, not the whole batch.
                model_filename = f"tree_{tree_index}.joblib"
                local_model_path = os.path.join(self.local_temp_dir, model_filename)
                ml_model.save_model(tree, local_model_path)
                self.storage.upload_file(local_model_path, _tree_key(request.model_id, tree_index))

                # 4. OOB PREDICTIONS: this tree's own held-out estimate, on
                # the rows its bootstrap sample left out. Uploaded as a small
                # artifact next to the tree - if this upload is the one that
                # gets lost to a crash, scripts/evaluate_oob.py notices the
                # gap and skips this tree rather than blocking on it; the
                # tree itself is already safely on S3 either way.
                oob_entries = ml_model.compute_oob_predictions(tree, X, task_type, indices)
                oob_filename = f"tree_{tree_index}_oob.json"
                local_oob_path = os.path.join(self.local_temp_dir, oob_filename)
                with open(local_oob_path, "w") as f:
                    json.dump(oob_entries, f)
                self.storage.upload_file(local_oob_path, _oob_key(request.model_id, tree_index))

                # DEBUG
                print(f"[Worker] Tree {tree_index} of model {request.model_id} uploaded ({len(oob_entries)} OOB rows).")

            print(f"---[Worker] TRAIN JOB COMPLETED ---")

            return worker_pb2.TrainResponse(success=True, message="Training completed.")

        except Exception as e:
            return worker_pb2.TrainResponse(success=False, message=str(e))



    def Predict(self, request, context):
        print(f"[Worker] Predict request for model {request.model_id}, trees {list(request.tree_indices)}")
        # Note: stateless behaviour
        # -> when prediction is asked, workers download trees from shared storage. No internal worker state.

        try:
            local_model_dir = os.path.join(self.local_temp_dir, request.model_id)
            os.makedirs(local_model_dir, exist_ok=True)

            predictions = []
            for tree_index in request.tree_indices:
                s3_key = _tree_key(request.model_id, tree_index)
                local_path = os.path.join(local_model_dir, os.path.basename(s3_key))

                self.storage.download_file(s3_key, local_path)

                result = ml_model.load_and_predict(
                    model_path=local_path,
                    features=list(request.features)
                )
                predictions.append(worker_pb2.TreePrediction(
                    classes=result.get("classes", []),
                    probabilities=result.get("probabilities", []),
                    value=result.get("value", 0.0),
                ))

            print(f"[Worker] Returning {len(predictions)} tree predictions.")

            # NO LOCAL AGGREGATION. Returns raw list (Master will aggregate).
            return worker_pb2.PredictResponse(predictions=predictions)

        except Exception as e:
            print(f"[Error Predict] {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return worker_pb2.PredictResponse()
