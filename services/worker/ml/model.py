# model.py

# Implements the machine learning model (RandomForest)


import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split # Used for splitting data into features and target
import joblib
import io


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a specified file path (e.g., '/path/to/data.csv' or 's3://bucket/data.csv') into a pandas DataFrame.
    """

    print(f"Loading dataset from: {file_path}")

    # assumes the dataset is a CSV file (todo fix here in case it's not a csv)
    df = pd.read_csv(file_path)
    return df


def train_model(data: pd.DataFrame, target_column: str, model_type: str, n_estimators: int) -> bytes:
    """
    Trains a RandomForest model (Classifier or Regressor) on the dataset.
    TODO: extend this function to include more hyperparameters
    """

    print("Starting model training...")

    # --- PREPARE DATA ---
    # Separate features (X) from the target variable (y).
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    except KeyError:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        return None

    # TODO: more preprocessing, feature engineering, splitting step etc
    # add a data splitting step (train/test split). ((For now trains on the whole dataset)).
    # Add feature engineering and preprocessing steps. Like handling
    # categorical variables or scaling numerical features.


    # --- MODEL INITIALIZATION ---
    # todo: choose a better initialization
    if model_type == 'classifier':
        # Use a fixed random_state for reproducibility.
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=511)
        print(f"Initialized RandomForestClassifier with {n_estimators} estimators.")

    elif model_type == 'regressor':
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=511)
        print(f"Initialized RandomForestRegressor with {n_estimators} estimators.")

    else:
        print(f"Error: Invalid model_type '{model_type}'. Choose 'classifier' or 'regressor'.")
        return None

    # --- MODEL TRAINING ---
    model.fit(X, y)
    print("Model training completed successfully.")

    # --- MODEL SERIALIZATION ---
    # Serialize the trained model object into bytes (pickle) so that we can store it.
    with io.BytesIO() as buffer:
        joblib.dump(model, buffer)
        serialized_model = buffer.getvalue()

    print("Model serialized successfully.")
    return serialized_model


def predict_with_model(serialized_model: bytes, new_data: pd.DataFrame):
    """
    Loads a serialized model and uses it to make predictions on new data.
    """
    print("Starting prediction...")

    # --- MODEL DESERIALIZATION ---
    try:
        with io.BytesIO(serialized_model) as buffer:
            model = joblib.load(buffer)
        print("Model deserialized successfully.")

    except Exception as e:
        print(f"Error deserializing the model: {e}")
        return None

    # TODO: Apply the same data preprocessing on `new_data` as was done on the training data.

    # --- INFERENCE ---
    predictions = model.predict(new_data)
    print("Prediction completed successfully.")

    if predictions is None:
        print("Model prediction failed.")

    return predictions