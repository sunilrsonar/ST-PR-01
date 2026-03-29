#!/usr/bin/env python3
"""Generate a standalone HTML dashboard from saved model and backtest artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ai_swing_trader.backtest import backtest_path_for_ticker
from ai_swing_trader.dashboard import build_dashboard_html, dashboard_path_for_ticker, summarize_backtest_frame
from ai_swing_trader.data import model_artifact_path_for_ticker
from ai_swing_trader.model import load_model_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a standalone HTML dashboard from a saved backtest.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, for example AAPL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    backtest_path = backtest_path_for_ticker(args.ticker)
    if not backtest_path.exists():
        raise FileNotFoundError(
            f"Missing backtest CSV: {backtest_path}. Run scripts/backtest_strategy.py --ticker {args.ticker.upper()} --save-csv first."
        )

    artifact_path = model_artifact_path_for_ticker(args.ticker)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {artifact_path}. Run scripts/train_model.py first."
        )

    backtest_df = pd.read_csv(backtest_path, parse_dates=["Date"])
    artifact = load_model_artifact(artifact_path)
    transaction_cost_bps = 0.0
    if "transaction_cost" in backtest_df and "turnover" in backtest_df:
        nonzero_cost_rows = backtest_df[(backtest_df["transaction_cost"] > 0) & (backtest_df["turnover"] > 0)]
        if not nonzero_cost_rows.empty:
            transaction_cost_bps = float(
                (nonzero_cost_rows["transaction_cost"] / nonzero_cost_rows["turnover"]).median() * 10000
            )
    metrics = summarize_backtest_frame(backtest_df, transaction_cost_bps=transaction_cost_bps)

    dashboard_html = build_dashboard_html(
        ticker=args.ticker,
        backtest_df=backtest_df,
        artifact=artifact,
        metrics=metrics,
    )

    output_path = dashboard_path_for_ticker(args.ticker)
    output_path.write_text(dashboard_html, encoding="utf-8")
    print(f"Saved dashboard to: {output_path}")


if __name__ == "__main__":
    main()
