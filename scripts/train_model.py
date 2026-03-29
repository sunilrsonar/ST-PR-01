#!/usr/bin/env python3
"""Train a RandomForest classifier on daily swing trading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_swing_trader.config import (
    DEFAULT_BUY_THRESHOLD,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SELL_THRESHOLD,
    DEFAULT_TEST_SIZE,
)
from ai_swing_trader.data import (
    ensure_directories,
    load_price_history,
    model_artifact_path_for_ticker,
    processed_data_path_for_ticker,
    raw_data_path_for_ticker,
)
from ai_swing_trader.features import FEATURE_COLUMNS, add_technical_features, finalize_feature_dataset
from ai_swing_trader.labels import add_future_return_labels
from ai_swing_trader.model import (
    evaluate_classifier,
    evaluate_regressor,
    save_model_artifact,
    time_series_train_test_split,
    train_random_forest,
    train_random_forest_regressor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RandomForest model for swing trading signals.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, for example AAPL.")
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS, help="Label horizon in trading days.")
    parser.add_argument("--buy-threshold", type=float, default=DEFAULT_BUY_THRESHOLD, help="BUY threshold as decimal return.")
    parser.add_argument("--sell-threshold", type=float, default=DEFAULT_SELL_THRESHOLD, help="SELL threshold as decimal return.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Fraction of rows reserved for test data.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for reproducibility.")
    return parser.parse_args()


def _prepare_training_frame(
    raw_frame: pd.DataFrame,
    horizon_days: int,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    featured = add_technical_features(raw_frame)
    labeled = add_future_return_labels(
        featured,
        horizon_days=horizon_days,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    return finalize_feature_dataset(labeled)


def main() -> None:
    args = parse_args()
    ensure_directories()

    input_path = raw_data_path_for_ticker(args.ticker)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input data: {input_path}. Run scripts/fetch_data.py first."
        )

    raw_frame = load_price_history(input_path)
    dataset = _prepare_training_frame(
        raw_frame=raw_frame,
        horizon_days=args.horizon_days,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )

    if len(dataset) < 200:
        raise ValueError("Not enough processed rows to train reliably. Fetch a longer history.")

    processed_path = processed_data_path_for_ticker(args.ticker)
    dataset.to_csv(processed_path, index=False)

    train_df, test_df = time_series_train_test_split(dataset, test_size=args.test_size)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    y_train_reg = train_df["future_return"]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]
    y_test_reg = test_df["future_return"]

    model = train_random_forest(x_train, y_train, random_state=args.random_state)
    regressor = train_random_forest_regressor(x_train, y_train_reg, random_state=args.random_state)
    evaluation = evaluate_classifier(model, x_test, y_test)
    regression_evaluation = evaluate_regressor(regressor, x_test, y_test_reg)

    artifact_payload = {
        "ticker": args.ticker.upper(),
        "feature_columns": FEATURE_COLUMNS,
        "horizon_days": args.horizon_days,
        "buy_threshold": args.buy_threshold,
        "sell_threshold": args.sell_threshold,
        "test_size": args.test_size,
        "train_start_date": str(pd.Timestamp(train_df["Date"].iloc[0]).date()),
        "train_end_date": str(pd.Timestamp(train_df["Date"].iloc[-1]).date()),
        "test_start_date": str(pd.Timestamp(test_df["Date"].iloc[0]).date()),
        "test_end_date": str(pd.Timestamp(test_df["Date"].iloc[-1]).date()),
        "classification_model": model,
        "regression_model": regressor,
    }
    artifact_path = model_artifact_path_for_ticker(args.ticker)
    save_model_artifact(artifact_path, artifact_payload)

    feature_importance = (
        pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        .sort_values(ascending=False)
        .round(4)
    )

    print(f"Processed dataset saved to: {processed_path}")
    print(f"Model artifact saved to: {artifact_path}")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print("Class distribution:")
    print(dataset["signal"].value_counts(normalize=True).sort_index().round(4).to_string())
    print("Classification report:")
    print(json.dumps(evaluation["classification_report"], indent=2))
    print("Confusion matrix [SELL, HOLD, BUY]:")
    print(evaluation["confusion_matrix"])
    print("Regression metrics for future return:")
    print(json.dumps(
        {
            "mae": regression_evaluation["mae"],
            "rmse": regression_evaluation["rmse"],
            "r2": regression_evaluation["r2"],
        },
        indent=2,
    ))
    print("Top feature importances:")
    print(feature_importance.to_string())


if __name__ == "__main__":
    main()
