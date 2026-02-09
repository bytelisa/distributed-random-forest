# model.py
import os
from typing import Union
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

def train_model(data: pd.DataFrame, target_column: str, task_type: str, n_estimators: int) -> Union[RandomForestClassifier, RandomForestRegressor]:
    """
    Trains a RandomForest model.
    """
    print(f"[Model] Starting training for target: {target_column}")

    # 1. VALIDATE TARGET
    if target_column not in data.columns:
        raise ModelError(f"Target column '{target_column}' not found. Available: {data.columns.tolist()}")

    # 2. SEPARATE TARGET (y) BEFORE PREPROCESSING X
    # This prevents accidentally deleting the target if it's a string (e.g. Iris Species)
    y = data[target_column]

    # 3. PREPARE FEATURES (X)
    X = data.drop(columns=[target_column])

    # Remove 'Id' column if it exists (it's noise)
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

    print(f"[Model] Training on features: {X_numeric.columns.tolist()}")

    # todo choose better initialization
    # 6. MODEL INITIALIZATION
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=1)
    elif task_type == 'regression':
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=1)
    else:
        raise ModelError(f"Invalid task type '{task_type}'. Choose 'classification' or 'regression'.")

    # 7. TRAIN
    try:
        model.fit(X_numeric, y)
        print("[Model] Training completed successfully.")
        return model
    except Exception as e:
        raise ModelError(f"Scikit-learn training failed: {e}")


def load_and_predict(model_path: str, features: list) -> str:
    """
    Loads a serialized model and uses it to make predictions.
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
        prediction = model.predict(new_data)
        result = str(prediction[0])
        print(f"[Model] Prediction result: {result}")
        return result
    except Exception as e:
        # Often happens if feature count doesn't match
        raise ModelError(f"[Model] Inference error (check feature count): {e}")