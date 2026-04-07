"""Feature engineering for the swing trading model."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "daily_return",
    "intraday_range",
    "gap_return",
    "return_20",
    "return_50",
    "ma_20_ratio",
    "ma_50_ratio",
    "rsi_14",
    "atr_14_pct",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_width_20",
    "bb_position_20",
    "volume_ratio_20",
    "obv",
    "volatility_20",
    "market_daily_return",
    "market_ma_20_ratio",
    "market_ma_50_ratio",
    "market_rsi_14",
    "market_macd_hist",
    "relative_strength_20",
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


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate true range."""
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate average true range."""
    return _true_range(high, low, close).rolling(window=period).mean()


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX and directional indicators."""
    up_move = high.diff()
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = _atr(high, low, close, period=period)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(window=period).mean()
    return adx, plus_di, minus_di


def _bollinger_bands(close: pd.Series, window: int = 20, width: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger band center, upper, and lower lines."""
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = middle + width * std
    lower = middle - width * std
    return middle, upper, lower


def _market_context_features(market_frame: pd.DataFrame) -> pd.DataFrame:
    """Build market trend features from a benchmark index."""
    market = market_frame.copy()
    market = market.sort_values("Date").reset_index(drop=True)

    market["market_daily_return"] = market["Close"].pct_change()
    market["market_return_20"] = market["Close"].pct_change(periods=20)
    market["market_ma_20"] = market["Close"].rolling(window=20).mean()
    market["market_ma_50"] = market["Close"].rolling(window=50).mean()
    market["market_ma_20_ratio"] = market["Close"] / market["market_ma_20"] - 1
    market["market_ma_50_ratio"] = market["Close"] / market["market_ma_50"] - 1
    market["market_rsi_14"] = _rsi(market["Close"], period=14)
    _, _, market_macd_hist = _macd(market["Close"])
    market["market_macd_hist"] = market_macd_hist

    return market[
        [
            "Date",
            "market_daily_return",
            "market_return_20",
            "market_ma_20_ratio",
            "market_ma_50_ratio",
            "market_rsi_14",
            "market_macd_hist",
        ]
    ]


def add_technical_features(frame: pd.DataFrame, market_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create leak-safe technical indicators from historical OHLCV data."""
    df = frame.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["daily_return"] = df["Close"].pct_change()
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["gap_return"] = df["Open"] / df["Close"].shift(1) - 1
    df["return_20"] = df["Close"].pct_change(periods=20)
    df["return_50"] = df["Close"].pct_change(periods=50)

    df["ma_20"] = df["Close"].rolling(window=20).mean()
    df["ma_50"] = df["Close"].rolling(window=50).mean()
    df["ma_20_ratio"] = df["Close"] / df["ma_20"] - 1
    df["ma_50_ratio"] = df["Close"] / df["ma_50"] - 1

    df["rsi_14"] = _rsi(df["Close"], period=14)
    df["atr_14_pct"] = _atr(df["High"], df["Low"], df["Close"], period=14) / df["Close"]
    adx_14, plus_di_14, minus_di_14 = _adx(df["High"], df["Low"], df["Close"], period=14)
    df["adx_14"] = adx_14
    df["plus_di_14"] = plus_di_14
    df["minus_di_14"] = minus_di_14

    macd_line, macd_signal, macd_hist = _macd(df["Close"])
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    bb_middle, bb_upper, bb_lower = _bollinger_bands(df["Close"], window=20, width=2)
    df["bb_width_20"] = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
    df["bb_position_20"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    df["volume_sma_20"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio_20"] = df["Volume"] / df["volume_sma_20"]
    df["obv"] = _obv(df["Close"], df["Volume"])
    df["volatility_20"] = df["daily_return"].rolling(window=20).std()

    # Keep engineered features stable in scale for tree-based models.
    df["obv"] = df["obv"].pct_change().replace([np.inf, -np.inf], np.nan)

    if market_frame is not None:
        market_features = _market_context_features(market_frame)
        df = df.merge(market_features, on="Date", how="left")
        df["relative_strength_20"] = df["return_20"] - df["market_return_20"]

    return df


def finalize_feature_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where rolling indicators are not yet available."""
    required_columns = ["Date", "Close", "Volume", *FEATURE_COLUMNS]
    return frame.dropna(subset=required_columns).reset_index(drop=True)
