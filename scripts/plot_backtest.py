#!/usr/bin/env python3
"""Plot the saved backtest equity curve and drawdown."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd

from ai_swing_trader.backtest import backtest_path_for_ticker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a saved backtest CSV.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, for example AAPL.")
    return parser.parse_args()


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required. Install dependencies with: pip install -r requirements.txt") from exc

    args = parse_args()
    input_path = backtest_path_for_ticker(args.ticker)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing backtest CSV: {input_path}. Run scripts/backtest_strategy.py --ticker {args.ticker.upper()} --save-csv first."
        )

    df = pd.read_csv(input_path, parse_dates=["Date"])
    output_path = input_path.with_suffix(".png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[3, 1.5])

    axes[0].plot(df["Date"], df["equity_curve"], label="Strategy", linewidth=2.0)
    axes[0].plot(df["Date"], df["benchmark_curve"], label="Benchmark", linewidth=1.8, alpha=0.8)
    axes[0].set_title(f"{args.ticker.upper()} Backtest Equity Curve")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].fill_between(df["Date"], df["drawdown"], 0, alpha=0.35, color="firebrick")
    axes[1].plot(df["Date"], df["drawdown"], color="firebrick", linewidth=1.2)
    axes[1].set_title("Strategy Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
