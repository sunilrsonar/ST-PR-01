#!/usr/bin/env python3
"""Fetch and train all uncommented ticker symbols from a text file."""

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
    DEFAULT_TRAIN_START,
)
from ai_swing_trader.data import (
    download_price_history,
    ensure_directories,
    model_artifact_path_for_ticker,
    processed_data_path_for_ticker,
    raw_data_path_for_ticker,
    save_price_history,
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


def _load_tickers_from_file(file_path: Path) -> list[str]:
    """Load uncommented tickers from a plain-text file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Ticker file not found: {file_path}")

    tickers: list[str] = []
    seen: set[str] = set()
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for item in line.split(","):
            ticker = item.strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and train all uncommented stocks from a ticker file.")
    parser.add_argument("--ticker-file", default="stocks.txt", help="Path to the ticker list file. Defaults to stocks.txt.")
    parser.add_argument(
        "--failed-file",
        default="failed_stocks.txt",
        help="Path to write tickers that failed training. Defaults to failed_stocks.txt.",
    )
    parser.add_argument("--start", default=DEFAULT_TRAIN_START, help="Historical fetch start date in YYYY-MM-DD format.")
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS, help="Label horizon in trading days.")
    parser.add_argument("--buy-threshold", type=float, default=DEFAULT_BUY_THRESHOLD, help="BUY threshold as decimal return.")
    parser.add_argument("--sell-threshold", type=float, default=DEFAULT_SELL_THRESHOLD, help="SELL threshold as decimal return.")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Fraction of rows reserved for test data.")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    ticker_file = Path(args.ticker_file)
    failed_file = Path(args.failed_file)
    tickers = _load_tickers_from_file(ticker_file)
    if not tickers:
        raise ValueError(f"No active tickers found in {ticker_file}.")

    successes: list[dict] = []
    failures: list[dict] = []

    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{len(tickers)}] Processing {ticker}")
        try:
            raw_frame = download_price_history(ticker=ticker, start=args.start)
            save_price_history(raw_frame, raw_data_path_for_ticker(ticker))

            dataset = _prepare_training_frame(
                raw_frame=raw_frame,
                horizon_days=args.horizon_days,
                buy_threshold=args.buy_threshold,
                sell_threshold=args.sell_threshold,
            )
            if len(dataset) < 200:
                raise ValueError("Not enough processed rows to train reliably.")

            dataset.to_csv(processed_data_path_for_ticker(ticker), index=False)
            train_df, test_df = time_series_train_test_split(dataset, test_size=args.test_size)
            x_train = train_df[FEATURE_COLUMNS]
            y_train_cls = train_df["target"]
            y_train_reg = train_df["future_return"]
            x_test = test_df[FEATURE_COLUMNS]
            y_test_cls = test_df["target"]
            y_test_reg = test_df["future_return"]

            classifier = train_random_forest(x_train, y_train_cls, random_state=args.random_state)
            regressor = train_random_forest_regressor(x_train, y_train_reg, random_state=args.random_state)
            cls_eval = evaluate_classifier(classifier, x_test, y_test_cls)
            reg_eval = evaluate_regressor(regressor, x_test, y_test_reg)

            save_model_artifact(
                model_artifact_path_for_ticker(ticker),
                {
                    "ticker": ticker.upper(),
                    "feature_columns": FEATURE_COLUMNS,
                    "horizon_days": args.horizon_days,
                    "buy_threshold": args.buy_threshold,
                    "sell_threshold": args.sell_threshold,
                    "test_size": args.test_size,
                    "train_start_date": str(pd.Timestamp(train_df["Date"].iloc[0]).date()),
                    "train_end_date": str(pd.Timestamp(train_df["Date"].iloc[-1]).date()),
                    "test_start_date": str(pd.Timestamp(test_df["Date"].iloc[0]).date()),
                    "test_end_date": str(pd.Timestamp(test_df["Date"].iloc[-1]).date()),
                    "classification_model": classifier,
                    "regression_model": regressor,
                },
            )

            success = {
                "ticker": ticker,
                "rows": int(len(dataset)),
                "test_rows": int(len(test_df)),
                "accuracy": float(cls_eval["classification_report"]["accuracy"]),
                "regression_mae": float(reg_eval["mae"]),
            }
            successes.append(success)
            print(json.dumps(success, indent=2))
        except Exception as exc:
            failure = {"ticker": ticker, "error": str(exc)}
            failures.append(failure)
            print(json.dumps(failure, indent=2))

    print("\nSummary")
    print(f"Successful trainings: {len(successes)}")
    print(f"Failed trainings: {len(failures)}")
    if successes:
        print("Successful tickers:")
        print(", ".join(item["ticker"] for item in successes))
    if failures:
        print("Failed tickers:")
        for item in failures:
            print(f"- {item['ticker']}: {item['error']}")

    failed_lines = [
        "# Auto-generated by scripts/train_from_file.py",
        "# One failed ticker per line, with the error as a comment.",
        "",
    ]
    if failures:
        for item in failures:
            failed_lines.append(f"{item['ticker']}  # {item['error']}")
    else:
        failed_lines.append("# No failed tickers in the last bulk training run.")
    failed_file.write_text("\n".join(failed_lines) + "\n", encoding="utf-8")
    print(f"Failure report written to: {failed_file}")


if __name__ == "__main__":
    main()
