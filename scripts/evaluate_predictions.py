#!/usr/bin/env python3
"""Evaluate logged predictions against actual outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_swing_trader.tracking import (
    evaluate_prediction_log,
    prediction_log_path,
    summarize_prediction_outcomes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction log outcomes.")
    parser.add_argument(
        "--log-path",
        default=str(prediction_log_path()),
        help="Path to the prediction log CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent matured rows to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_path)
    outcomes = evaluate_prediction_log(log_path)
    summary = summarize_prediction_outcomes(outcomes)

    print("Prediction Accuracy Summary")
    print(f"Matured predictions: {summary['matured_predictions']}")
    print(f"Direction accuracy: {summary['direction_accuracy']:.2%}")
    print(f"Mean absolute % error: {summary['mean_abs_pct_error']:.2%}")
    print()

    matured = outcomes.dropna(subset=["actual_future_close"]).copy()
    if matured.empty:
        print("No matured predictions yet.")
        return

    recent = (
        matured.sort_values(["signal_date", "ticker"], ascending=[False, True])
        .head(args.limit)
        .copy()
    )
    recent_display = recent[
        [
            "ticker",
            "signal_date",
            "predicted_signal",
            "predicted_future_close",
            "actual_signal",
            "actual_future_close",
            "price_error",
            "abs_pct_error",
            "direction_correct",
        ]
    ]
    print("Recent matured predictions:")
    print(recent_display.to_string(index=False))


if __name__ == "__main__":
    main()
