"""Feature engineering for the swing trading model."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "daily_return",
    "intraday_range",
    "gap_return",
    "ma_20_ratio",
    "ma_50_ratio",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "volume_ratio_20",
    "obv",
    "volatility_20",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(average_loss != 0, 100)
    rsi = rsi.where(~((average_gain == 0) & (average_loss == 0)), 50)
    return rsi


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate an exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, signal, and histogram series."""
    ema_fast = _ema(close, span=12)
    ema_slow = _ema(close, span=26)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, span=9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create leak-safe technical indicators from historical OHLCV data."""
    df = frame.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["daily_return"] = df["Close"].pct_change()
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["gap_return"] = df["Open"] / df["Close"].shift(1) - 1

    df["ma_20"] = df["Close"].rolling(window=20).mean()
    df["ma_50"] = df["Close"].rolling(window=50).mean()
    df["ma_20_ratio"] = df["Close"] / df["ma_20"] - 1
    df["ma_50_ratio"] = df["Close"] / df["ma_50"] - 1

    df["rsi_14"] = _rsi(df["Close"], period=14)

    macd_line, macd_signal, macd_hist = _macd(df["Close"])
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    df["volume_sma_20"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio_20"] = df["Volume"] / df["volume_sma_20"]
    df["obv"] = _obv(df["Close"], df["Volume"])
    df["volatility_20"] = df["daily_return"].rolling(window=20).std()

    # Keep engineered features stable in scale for tree-based models.
    df["obv"] = df["obv"].pct_change().replace([np.inf, -np.inf], np.nan)

    return df


def finalize_feature_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where rolling indicators are not yet available."""
    required_columns = ["Date", "Close", "Volume", *FEATURE_COLUMNS]
    return frame.dropna(subset=required_columns).reset_index(drop=True)
