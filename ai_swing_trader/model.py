"""Training, evaluation, and persistence helpers."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

from ai_swing_trader.config import TARGET_TO_SIGNAL


def time_series_train_test_split(frame: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe chronologically into train and test subsets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(frame) * (1 - test_size))
    if split_index <= 0 or split_index >= len(frame):
        raise ValueError("Not enough rows to create a time series train/test split.")

    train_df = frame.iloc[:split_index].copy()
    test_df = frame.iloc[split_index:].copy()
    return train_df, test_df


def train_random_forest(train_features: pd.DataFrame, train_target: pd.Series, random_state: int) -> RandomForestClassifier:
    """Fit a simple RandomForest classifier for multi-class swing signal prediction."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(train_features, train_target)
    return model


def train_random_forest_regressor(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    random_state: int,
) -> RandomForestRegressor:
    """Fit a RandomForest regressor for future-return estimation."""
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_features, train_target)
    return model


def evaluate_classifier(model: RandomForestClassifier, test_features: pd.DataFrame, test_target: pd.Series) -> dict:
    """Generate evaluation artifacts for the test period."""
    predictions = model.predict(test_features)
    report = classification_report(
        test_target,
        predictions,
        labels=[-1, 0, 1],
        target_names=[TARGET_TO_SIGNAL[-1], TARGET_TO_SIGNAL[0], TARGET_TO_SIGNAL[1]],
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(test_target, predictions, labels=[-1, 0, 1])
    return {
        "predictions": predictions,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_regressor(model: RandomForestRegressor, test_features: pd.DataFrame, test_target: pd.Series) -> dict:
    """Generate regression metrics for the future-return model."""
    predictions = model.predict(test_features)
    mae = mean_absolute_error(test_target, predictions)
    rmse = math.sqrt(mean_squared_error(test_target, predictions))
    r2 = r2_score(test_target, predictions)
    return {
        "predictions": predictions.tolist(),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def save_model_artifact(artifact_path: Path, payload: dict) -> None:
    """Persist the trained model and metadata."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, artifact_path)


def load_model_artifact(artifact_path: Path) -> dict:
    """Load the trained model artifact."""
    return joblib.load(artifact_path)
