# model.py
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


def resolve_hyperparameters(defaults: dict, hyperparameters: dict) -> dict:
    """
    Merges the configured per-task defaults (model_defaults in config) with
    the ones from the request, which take precedence.
    """
    params = dict(defaults)
    params.update(hyperparameters)
    return params


def train_tree(X: pd.DataFrame, y: pd.Series, task_type: str, seed: int, defaults: dict, **hyperparameters) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
    """
    Trains one decision tree on the given (already bootstrapped) sample.
    """
    if task_type not in ('classification', 'regression'):
        raise ModelError(f"Invalid task type '{task_type}'. Choose 'classification' or 'regression'.")

    params = resolve_hyperparameters(defaults, hyperparameters)

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


def compute_oob_predictions(tree, X: pd.DataFrame, task_type: str, bootstrap_indices) -> list:
    """
    Runs the already-fitted tree on the rows its own bootstrap sample left
    out (out-of-bag) - X is the full training set, in the same row order the
    bootstrap indices refer to. One entry per OOB row, self-describing so
    scripts/evaluate_oob.py never has to re-derive anything:
    {"row": <position>, "classes": [...], "probabilities": [...]} for a
    classifier (only the classes this tree's bootstrap sample happened to
    contain), {"row": <position>, "value": v} for a regressor.
    """
    included = np.unique(bootstrap_indices)
    oob_positions = np.setdiff1d(np.arange(len(X)), included)
    if len(oob_positions) == 0:
        return []

    X_oob = X.iloc[oob_positions]
    if task_type == 'classification':
        probabilities = tree.predict_proba(X_oob)
        classes = [str(c) for c in tree.classes_]
        return [
            {"row": int(row), "classes": classes, "probabilities": [float(p) for p in probs]}
            for row, probs in zip(oob_positions, probabilities)
        ]
    values = tree.predict(X_oob)
    return [{"row": int(row), "value": float(v)} for row, v in zip(oob_positions, values)]


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

    # Wrap as a 1-row DataFrame with the training column names: the tree was
    # fit on one (prepare_features drops non-numeric columns), and predicting
    # with a bare array instead triggers a sklearn "no feature names" warning.
    new_data = pd.DataFrame([features], columns=model.feature_names_in_)

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
