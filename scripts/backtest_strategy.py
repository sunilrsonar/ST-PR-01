#!/usr/bin/env python3
"""Backtest the trained swing trading model on the out-of-sample test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_swing_trader.backtest import backtest_path_for_ticker, run_signal_backtest
from ai_swing_trader.data import ensure_directories, load_price_history, model_artifact_path_for_ticker, processed_data_path_for_ticker, raw_data_path_for_ticker
from ai_swing_trader.features import add_technical_features, finalize_feature_dataset
from ai_swing_trader.labels import add_future_return_labels
from ai_swing_trader.model import load_model_artifact, time_series_train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the model on the chronological holdout period.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, for example AAPL.")
    parser.add_argument("--test-size", type=float, default=None, help="Override test split fraction if needed.")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0, help="One-way transaction cost in basis points.")
    parser.add_argument("--save-csv", action="store_true", help="Save detailed backtest rows to artifacts.")
    return parser.parse_args()


def _build_processed_dataset(ticker: str, artifact: dict) -> pd.DataFrame:
    processed_path = processed_data_path_for_ticker(ticker)
    if processed_path.exists():
        return pd.read_csv(processed_path, parse_dates=["Date"])

    raw_path = raw_data_path_for_ticker(ticker)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing processed and raw data for {ticker.upper()}. Run scripts/fetch_data.py and scripts/train_model.py first."
        )

    raw_frame = load_price_history(raw_path)
    featured = add_technical_features(raw_frame)
    labeled = add_future_return_labels(
        featured,
        horizon_days=int(artifact["horizon_days"]),
        buy_threshold=float(artifact["buy_threshold"]),
        sell_threshold=float(artifact["sell_threshold"]),
    )
    return finalize_feature_dataset(labeled)


def main() -> None:
    args = parse_args()
    ensure_directories()

    artifact_path = model_artifact_path_for_ticker(args.ticker)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {artifact_path}. Run scripts/train_model.py first."
        )

    artifact = load_model_artifact(artifact_path)
    dataset = _build_processed_dataset(args.ticker, artifact)

    test_size = args.test_size if args.test_size is not None else artifact.get("test_size")
    if test_size is None:
        raise ValueError("test_size is not available in the artifact. Re-train the model or pass --test-size.")

    _, test_df = time_series_train_test_split(dataset, test_size=float(test_size))
    feature_columns = artifact["feature_columns"]
    classifier = artifact.get("classification_model") or artifact.get("model")
    if classifier is None:
        raise ValueError("Model artifact is missing the classification model. Re-train this ticker.")
    predictions = classifier.predict(test_df[feature_columns])

    backtest_df, metrics = run_signal_backtest(
        frame=test_df[["Date", "Close", "target", "signal"]].copy(),
        predictions=pd.Series(predictions),
        transaction_cost_bps=args.transaction_cost_bps,
    )

    output_path = backtest_path_for_ticker(args.ticker)
    if args.save_csv:
        backtest_df.to_csv(output_path, index=False)
        print(f"Detailed backtest saved to: {output_path}")

    print(f"Ticker: {args.ticker.upper()}")
    print(f"Backtest window: {metrics['start_date']} -> {metrics['end_date']}")
    print("Performance summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
