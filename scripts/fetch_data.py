#!/usr/bin/env python3
"""Download daily OHLCV data and save it locally as CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_swing_trader.config import DEFAULT_TRAIN_START
from ai_swing_trader.data import download_price_history, ensure_directories, raw_data_path_for_ticker, save_price_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch daily stock OHLCV data and save it as CSV.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, for example AAPL.")
    parser.add_argument("--start", default=DEFAULT_TRAIN_START, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=None, help="Optional end date in YYYY-MM-DD format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    frame = download_price_history(ticker=args.ticker, start=args.start, end=args.end)
    output_path = raw_data_path_for_ticker(args.ticker)
    save_price_history(frame, output_path)

    print(f"Saved {len(frame)} rows for {args.ticker.upper()} to {output_path}")
    print(f"Date range: {frame['Date'].min().date()} -> {frame['Date'].max().date()}")


if __name__ == "__main__":
    main()
