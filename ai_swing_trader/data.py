"""Helpers for downloading, loading, and saving market data."""

from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import yfinance as yf

from ai_swing_trader.config import (
    ARTIFACTS_DIR,
    DEFAULT_MARKET_BENCHMARK_TICKER,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def ensure_directories() -> None:
    """Create local storage directories if they do not exist."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _standardize_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names returned by yfinance into a consistent schema."""
    if frame.empty:
        raise ValueError("No price data was returned.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            level_0 if level_0 else level_1
            for level_0, level_1 in frame.columns.to_flat_index()
        ]

    frame = frame.reset_index()
    frame.columns = [str(column).strip().replace(" ", "_") for column in frame.columns]

    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
    frame = frame.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=numeric_columns)
    return frame


def download_price_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    interval: str = "1d",
    retries: int = 3,
    retry_delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """Download daily OHLCV data using yfinance with cached fallback."""
    download_kwargs = {
        "tickers": ticker,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
        "threads": False,
    }

    if period:
        download_kwargs["period"] = period
    else:
        download_kwargs["start"] = start
        if end:
            download_kwargs["end"] = end

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            frame = yf.download(**download_kwargs)
            return _standardize_ohlcv_columns(frame)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay_seconds)

    fallback_paths = [
        raw_data_path_for_ticker(ticker),
        market_context_path_for_ticker(ticker),
    ]
    for fallback_path in fallback_paths:
        if fallback_path.exists():
            return load_price_history(fallback_path)

    if last_error is not None:
        raise RuntimeError(
            f"Failed to download {ticker} after {retries} attempts: {last_error}"
        ) from last_error
    raise RuntimeError(f"Failed to download {ticker}: unknown error")


def save_price_history(frame: pd.DataFrame, output_path: Path) -> None:
    """Persist OHLCV data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def load_price_history(csv_path: Path) -> pd.DataFrame:
    """Load OHLCV data from a local CSV file."""
    frame = pd.read_csv(csv_path, parse_dates=["Date"])
    return _standardize_ohlcv_columns(frame)


def raw_data_path_for_ticker(ticker: str) -> Path:
    """Return the default CSV path for a given ticker symbol."""
    safe_ticker = ticker.upper().replace("/", "_")
    return RAW_DATA_DIR / f"{safe_ticker}.csv"


def processed_data_path_for_ticker(ticker: str) -> Path:
    """Return the default processed dataset path for a given ticker symbol."""
    safe_ticker = ticker.upper().replace("/", "_")
    return PROCESSED_DATA_DIR / f"{safe_ticker}_features.csv"


def model_artifact_path_for_ticker(ticker: str) -> Path:
    """Return the default trained model path for a given ticker symbol."""
    safe_ticker = ticker.upper().replace("/", "_")
    return ARTIFACTS_DIR / f"{safe_ticker}_rf_model.joblib"


def market_context_path_for_ticker(ticker: str = DEFAULT_MARKET_BENCHMARK_TICKER) -> Path:
    """Return the default CSV path for a benchmark or market context ticker."""
    safe_ticker = ticker.upper().replace("/", "_").replace("^", "")
    return RAW_DATA_DIR / f"_benchmark_{safe_ticker}.csv"


def refresh_market_context(
    ticker: str = DEFAULT_MARKET_BENCHMARK_TICKER,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Download and cache benchmark data used as market context features."""
    frame = download_price_history(
        ticker=ticker,
        start=start,
        end=end,
        period=period,
    )
    save_price_history(frame, market_context_path_for_ticker(ticker))
    return frame
