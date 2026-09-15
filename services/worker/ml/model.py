# model.py
import math
import os
from typing import Union
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class ModelError(Exception):
    pass


def save_model(model, filepath: str):
    """Serialize the model with joblib."""
    try:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(model, filepath)
        print(f"[Model] Model saved in {filepath}")
    except Exception as e:
        raise ModelError(f"[Model] Error while saving model: {e}")


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Loads dataset from path.
    """
    if not os.path.exists(dataset_path):
        raise ModelError(f"[Model] Dataset not found at: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
        # Remove quotes and spaces from column names to avoid "Column not found" errors
        df.columns = df.columns.str.replace("'", "").str.replace('"', '').str.strip()

        print(f"[Model] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except Exception as e:
        raise ModelError(f"[Model] Could not load dataset from {dataset_path}: {e}")


def prepare_features(data: pd.DataFrame, target_column: str):
    """
    Splits the dataset into numeric features X and target y, keeping the row order.
    """
    # 1. VALIDATE TARGET
    if target_column not in data.columns:
        raise ModelError(f"Target column '{target_column}' not found. Available: {data.columns.tolist()}")

    # 2. SEPARATE TARGET (y) BEFORE PREPROCESSING X
    # This prevents accidentally deleting the target if it's a string (e.g. Iris Species)
    y = data[target_column]

    # 3. PREPARE FEATURES (X)
    X = data.drop(columns=[target_column])

    # Remove 'Id' column if it exists
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])

    # 4. HANDLE NON-NUMERIC FEATURES
    # Since gRPC currently sends repeated floats, we only train on numeric columns.
    # This automatically drops 'ocean_proximity' or other string features from X.
    X_numeric = X.select_dtypes(include=[np.number])

    # Check if we we still have columns to work with
    if X_numeric.shape[1] == 0:
        raise ModelError("Error: No numeric features remained after preprocessing.")

    # 5. HANDLE MISSING VALUES (NaN)
    # Quick fix: fill with 0. Necessary for housing.csv
    X_numeric = X_numeric.fillna(0)

    return X_numeric, y


def _default_hyperparameters(task_type: str, n_features: int) -> dict:
    # A lone DecisionTree defaults to considering every feature at each split,
    # which would make this bagging, not a random forest: the RandomForest
    # defaults are applied here, per task.
    if task_type == 'classification':
        return {"max_features": "sqrt", "criterion": "entropy"}
    if task_type == 'regression':
        return {"max_features": max(1, math.ceil(n_features / 3)), "criterion": "squared_error"}
    raise ModelError(f"Invalid task type '{task_type}'. Choose 'classification' or 'regression'.")


def train_tree(X: pd.DataFrame, y: pd.Series, task_type: str, seed: int, **hyperparameters) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
    """
    Trains one decision tree on the given (already bootstrapped) sample.
    Hyperparameters passed explicitly override the per-task defaults.
    """
    params = _default_hyperparameters(task_type, X.shape[1])
    params.update(hyperparameters)

    print(f"[Model] Training tree with random state {seed} and params {params}")

    if task_type == 'classification':
        model = DecisionTreeClassifier(random_state=seed, **params)
    else:
        model = DecisionTreeRegressor(random_state=seed, **params)

    try:
        model.fit(X, y)
        return model
    except Exception as e:
        raise ModelError(f"Scikit-learn training failed: {e}")


def load_and_predict(model_path: str, features: list) -> dict:
    """
    Loads a serialized tree and returns its raw output for one sample:
    {"classes": [...], "probabilities": [...]} for a classifier (the tree
    only knows the classes seen in its own bootstrap sample), {"value": v}
    for a regressor.
    """
    print("[Model] Starting prediction...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[Model] Model File not found in: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ModelError(f"[Model] Error during deserialization: {e}")

    # Prepare input: reshape list to 2D array (1 sample)
    new_data = np.array(features).reshape(1, -1)

    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(new_data)[0]
            return {
                "classes": [str(c) for c in model.classes_],
                "probabilities": [float(p) for p in probabilities],
            }
        return {"value": float(model.predict(new_data)[0])}
    except Exception as e:
        # Often happens if feature count doesn't match
        raise ModelError(f"[Model] Inference error (check feature count): {e}")
