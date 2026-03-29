"""Simple backtesting utilities for model-driven swing trading signals."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ai_swing_trader.config import ARTIFACTS_DIR, TARGET_TO_SIGNAL


TRADING_DAYS_PER_YEAR = 252


def backtest_path_for_ticker(ticker: str) -> Path:
    """Return the default path for a backtest result CSV."""
    safe_ticker = ticker.upper().replace("/", "_")
    return ARTIFACTS_DIR / f"{safe_ticker}_backtest.csv"


def _annualized_return(total_return: float, periods: int) -> float:
    if periods <= 0:
        return 0.0
    growth = 1 + total_return
    if growth <= 0:
        return -1.0
    return growth ** (TRADING_DAYS_PER_YEAR / periods) - 1


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def run_signal_backtest(
    frame: pd.DataFrame,
    predictions: pd.Series,
    transaction_cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Backtest a daily flat/long/short strategy from predicted class labels."""
    if len(frame) != len(predictions):
        raise ValueError("frame and predictions must have the same number of rows.")

    df = frame.copy().reset_index(drop=True)
    df["predicted_target"] = pd.Series(predictions, index=df.index).astype(int)
    df["predicted_signal"] = df["predicted_target"].map(TARGET_TO_SIGNAL)

    df["market_return"] = df["Close"].pct_change().fillna(0.0)
    df["position"] = df["predicted_target"].shift(1).fillna(0).astype(int)
    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())

    cost_rate = transaction_cost_bps / 10000.0
    df["transaction_cost"] = df["turnover"] * cost_rate
    df["strategy_return_gross"] = df["position"] * df["market_return"]
    df["strategy_return_net"] = df["strategy_return_gross"] - df["transaction_cost"]

    df["equity_curve"] = (1 + df["strategy_return_net"]).cumprod()
    df["benchmark_curve"] = (1 + df["market_return"]).cumprod()
    df["drawdown"] = df["equity_curve"] / df["equity_curve"].cummax() - 1

    total_return = float(df["equity_curve"].iloc[-1] - 1)
    benchmark_return = float(df["benchmark_curve"].iloc[-1] - 1)
    strategy_volatility = float(df["strategy_return_net"].std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))
    benchmark_volatility = float(df["market_return"].std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))

    mean_daily_return = float(df["strategy_return_net"].mean())
    std_daily_return = float(df["strategy_return_net"].std(ddof=0))
    sharpe_ratio = 0.0 if std_daily_return == 0 else (mean_daily_return / std_daily_return) * math.sqrt(TRADING_DAYS_PER_YEAR)

    invested_mask = df["position"] != 0
    invested_days = int(invested_mask.sum())
    positive_days = int((df["strategy_return_net"] > 0).sum())
    invested_positive_days = int(((df["strategy_return_net"] > 0) & invested_mask).sum())

    trade_count = int((df["position"].ne(df["position"].shift(1))).sum() - 1)
    trade_count = max(trade_count, 0)

    metrics = {
        "rows": int(len(df)),
        "start_date": str(pd.Timestamp(df["Date"].iloc[0]).date()),
        "end_date": str(pd.Timestamp(df["Date"].iloc[-1]).date()),
        "transaction_cost_bps": float(transaction_cost_bps),
        "total_return": total_return,
        "annualized_return": _annualized_return(total_return, len(df)),
        "annualized_volatility": strategy_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": _max_drawdown(df["equity_curve"]),
        "benchmark_return": benchmark_return,
        "benchmark_annualized_return": _annualized_return(benchmark_return, len(df)),
        "benchmark_annualized_volatility": benchmark_volatility,
        "exposure_rate": float(invested_mask.mean()),
        "positive_day_rate": float(positive_days / len(df)),
        "invested_positive_day_rate": float(invested_positive_days / invested_days) if invested_days else 0.0,
        "trade_count": trade_count,
    }
    return df, metrics
