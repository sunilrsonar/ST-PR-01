"""Helpers for generating the latest model-based trading signal."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ai_swing_trader.config import TARGET_TO_SIGNAL
from ai_swing_trader.data import download_price_history, model_artifact_path_for_ticker
from ai_swing_trader.features import add_technical_features, finalize_feature_dataset
from ai_swing_trader.model import load_model_artifact


def confidence_band(confidence: float) -> str:
    """Convert a numeric confidence score into a simple label."""
    if confidence >= 0.55:
        return "High"
    if confidence >= 0.40:
        return "Medium"
    return "Low"


def predict_latest_signal(
    ticker: str,
    period: str,
    market_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Load the trained model and predict the latest BUY/SELL/HOLD signal."""
    artifact_path = model_artifact_path_for_ticker(ticker)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {artifact_path}. Run scripts/train_model.py first."
        )

    artifact = load_model_artifact(artifact_path)
    frame = download_price_history(ticker=ticker, period=period)
    featured = finalize_feature_dataset(
        add_technical_features(frame, market_frame=market_frame)
    )
    if featured.empty:
        raise ValueError("No valid feature rows were generated for prediction.")

    latest_row = featured.iloc[[-1]].copy()
    feature_columns = artifact["feature_columns"]
    x_latest = latest_row[feature_columns]

    classifier = artifact.get("classification_model") or artifact.get("model")
    regressor = artifact.get("regression_model")
    if classifier is None:
        raise ValueError("Model artifact is missing the classification model. Re-train this ticker.")
    if regressor is None:
        raise ValueError("Model artifact is missing the regression model. Re-train this ticker.")

    prediction = int(classifier.predict(x_latest)[0])
    probabilities = classifier.predict_proba(x_latest)[0]
    probability_map = {
        TARGET_TO_SIGNAL[int(label)]: round(float(probability), 4)
        for label, probability in zip(classifier.classes_, probabilities)
    }
    confidence = float(max(probabilities))
    predicted_future_return = float(regressor.predict(x_latest)[0])
    latest_close = float(latest_row["Close"].iloc[0])
    predicted_price_change = latest_close * predicted_future_return
    predicted_future_close = latest_close + predicted_price_change

    return {
        "ticker": ticker.upper(),
        "signal_date": pd.Timestamp(latest_row["Date"].iloc[0]).date(),
        "latest_close": latest_close,
        "predicted_signal": TARGET_TO_SIGNAL[prediction],
        "prediction_code": prediction,
        "probabilities": probability_map,
        "confidence": confidence,
        "confidence_band": confidence_band(confidence),
        "predicted_future_return": predicted_future_return,
        "predicted_price_change": predicted_price_change,
        "predicted_future_close": predicted_future_close,
        "horizon_days": int(artifact.get("horizon_days", 0)),
        "artifact": artifact,
    }
