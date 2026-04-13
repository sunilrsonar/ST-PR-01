"""Track live predictions and evaluate them after the forecast horizon."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_swing_trader.config import ARTIFACTS_DIR
from ai_swing_trader.data import download_price_history, raw_data_path_for_ticker, save_price_history


PREDICTION_LOG_COLUMNS = [
    "ticker",
    "signal_date",
    "signal_close",
    "predicted_signal",
    "predicted_future_close",
    "predicted_future_return",
    "predicted_price_change",
    "confidence",
    "confidence_band",
    "horizon_days",
    "buy_threshold",
    "sell_threshold",
    "logged_at",
    "actual_future_date",
    "actual_future_close",
    "actual_future_return",
    "actual_signal",
    "price_error",
    "abs_pct_error",
    "direction_correct",
]


def prediction_log_path() -> Path:
    """Return the CSV path used to store logged live predictions."""
    return ARTIFACTS_DIR / "prediction_log.csv"


def load_prediction_log(log_path: Path | None = None) -> pd.DataFrame:
    """Load the prediction log if it exists, otherwise return an empty frame."""
    path = log_path or prediction_log_path()
    if not path.exists():
        return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)

    frame = pd.read_csv(
        path,
        parse_dates=["signal_date", "logged_at", "actual_future_date"],
    )
    for column in PREDICTION_LOG_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[PREDICTION_LOG_COLUMNS].copy()
    object_columns = [
        "ticker",
        "predicted_signal",
        "confidence_band",
        "actual_signal",
        "direction_correct",
    ]
    for column in object_columns:
        frame[column] = frame[column].astype("object")

    numeric_columns = [
        "signal_close",
        "predicted_future_close",
        "predicted_future_return",
        "predicted_price_change",
        "confidence",
        "horizon_days",
        "buy_threshold",
        "sell_threshold",
        "actual_future_close",
        "actual_future_return",
        "price_error",
        "abs_pct_error",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _prediction_record(prediction_payload: dict) -> dict[str, object]:
    artifact = prediction_payload.get("artifact", {})
    return {
        "ticker": prediction_payload["ticker"],
        "signal_date": pd.Timestamp(prediction_payload["signal_date"]),
        "signal_close": float(prediction_payload["latest_close"]),
        "predicted_signal": prediction_payload["predicted_signal"],
        "predicted_future_close": float(prediction_payload["predicted_future_close"]),
        "predicted_future_return": float(prediction_payload["predicted_future_return"]),
        "predicted_price_change": float(prediction_payload["predicted_price_change"]),
        "confidence": float(prediction_payload["confidence"]),
        "confidence_band": prediction_payload["confidence_band"],
        "horizon_days": int(prediction_payload["horizon_days"]),
        "buy_threshold": float(artifact.get("buy_threshold", 0.03)),
        "sell_threshold": float(artifact.get("sell_threshold", -0.03)),
        "logged_at": pd.Timestamp.now("UTC"),
        "actual_future_date": pd.NaT,
        "actual_future_close": pd.NA,
        "actual_future_return": pd.NA,
        "actual_signal": pd.NA,
        "price_error": pd.NA,
        "abs_pct_error": pd.NA,
        "direction_correct": pd.NA,
    }


def upsert_prediction_log(
    prediction_payloads: list[dict],
    log_path: Path | None = None,
) -> pd.DataFrame:
    """Insert or update daily predictions in the log."""
    path = log_path or prediction_log_path()
    existing = load_prediction_log(path)

    if not prediction_payloads:
        return existing

    incoming = pd.DataFrame([_prediction_record(payload) for payload in prediction_payloads])
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = (
        combined.sort_values(["ticker", "signal_date", "logged_at"])
        .drop_duplicates(subset=["ticker", "signal_date", "horizon_days"], keep="last")
        .sort_values(["signal_date", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def _actual_signal(actual_return: float, buy_threshold: float, sell_threshold: float) -> str:
    if actual_return >= buy_threshold:
        return "BUY"
    if actual_return <= sell_threshold:
        return "SELL"
    return "HOLD"


def evaluate_prediction_log(log_path: Path | None = None) -> pd.DataFrame:
    """Evaluate matured predictions against actual closes after the forecast horizon."""
    path = log_path or prediction_log_path()
    log_frame = load_prediction_log(path)
    if log_frame.empty:
        return log_frame

    pending = log_frame[log_frame["actual_future_close"].isna()].copy()
    if pending.empty:
        return log_frame

    history_cache: dict[str, pd.DataFrame] = {}
    min_dates = (
        pending.groupby("ticker")["signal_date"]
        .min()
        .apply(lambda value: (pd.Timestamp(value) - pd.Timedelta(days=20)).date().isoformat())
        .to_dict()
    )

    for ticker, start_date in min_dates.items():
        try:
            history = download_price_history(ticker=ticker, start=start_date)
            save_price_history(history, raw_data_path_for_ticker(ticker))
            history["Date"] = pd.to_datetime(history["Date"]).dt.normalize()
            history_cache[ticker] = history.sort_values("Date").reset_index(drop=True)
        except Exception:
            continue

    for index, row in pending.iterrows():
        ticker = row["ticker"]
        history = history_cache.get(ticker)
        if history is None or history.empty:
            continue

        signal_date = pd.Timestamp(row["signal_date"]).normalize()
        matched = history.index[history["Date"] == signal_date].tolist()
        if not matched:
            continue

        signal_index = matched[0]
        target_index = signal_index + int(row["horizon_days"])
        if target_index >= len(history):
            continue

        actual_future_row = history.iloc[target_index]
        actual_future_close = float(actual_future_row["Close"])
        actual_future_date = pd.Timestamp(actual_future_row["Date"])
        actual_future_return = actual_future_close / float(row["signal_close"]) - 1
        actual_signal = _actual_signal(
            actual_future_return,
            buy_threshold=float(row["buy_threshold"]),
            sell_threshold=float(row["sell_threshold"]),
        )
        price_error = float(row["predicted_future_close"]) - actual_future_close
        abs_pct_error = abs(price_error) / actual_future_close if actual_future_close else pd.NA

        mask = (
            (log_frame["ticker"] == ticker)
            & (pd.to_datetime(log_frame["signal_date"]).dt.normalize() == signal_date)
            & (log_frame["horizon_days"] == row["horizon_days"])
        )
        log_frame.loc[mask, "actual_future_date"] = actual_future_date
        log_frame.loc[mask, "actual_future_close"] = actual_future_close
        log_frame.loc[mask, "actual_future_return"] = actual_future_return
        log_frame.loc[mask, "actual_signal"] = actual_signal
        log_frame.loc[mask, "price_error"] = price_error
        log_frame.loc[mask, "abs_pct_error"] = abs_pct_error
        log_frame.loc[mask, "direction_correct"] = actual_signal == row["predicted_signal"]

    log_frame = log_frame.sort_values(["signal_date", "ticker"], ascending=[False, True]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_frame.to_csv(path, index=False)
    return log_frame


def summarize_prediction_outcomes(log_frame: pd.DataFrame) -> dict[str, float]:
    """Summarize evaluated prediction quality for quick display."""
    matured = log_frame.dropna(subset=["actual_future_close"]).copy()
    if matured.empty:
        return {
            "matured_predictions": 0,
            "direction_accuracy": 0.0,
            "mean_abs_pct_error": 0.0,
        }

    direction_correct = matured["direction_correct"].astype(str).str.lower().eq("true")
    abs_pct_error = pd.to_numeric(matured["abs_pct_error"], errors="coerce")
    return {
        "matured_predictions": int(len(matured)),
        "direction_accuracy": float(direction_correct.mean()),
        "mean_abs_pct_error": float(abs_pct_error.mean()),
    }
