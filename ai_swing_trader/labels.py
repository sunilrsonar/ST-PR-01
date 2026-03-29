"""Target generation for supervised swing trading signals."""

from __future__ import annotations

import pandas as pd

from ai_swing_trader.config import TARGET_TO_SIGNAL


def add_future_return_labels(
    frame: pd.DataFrame,
    horizon_days: int,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """Create BUY/SELL/HOLD labels from future returns without leaking future data."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be greater than 0.")
    if sell_threshold >= buy_threshold:
        raise ValueError("sell_threshold must be lower than buy_threshold.")

    df = frame.copy()
    df["future_close"] = df["Close"].shift(-horizon_days)
    df["future_return"] = df["future_close"] / df["Close"] - 1
    df["target"] = 0
    df.loc[df["future_return"] >= buy_threshold, "target"] = 1
    df.loc[df["future_return"] <= sell_threshold, "target"] = -1
    df["signal"] = df["target"].map(TARGET_TO_SIGNAL)

    # The last horizon rows do not have a valid future label.
    df = df.iloc[:-horizon_days].copy()
    return df.reset_index(drop=True)
