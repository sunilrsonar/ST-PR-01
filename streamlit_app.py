#!/usr/bin/env python3
"""Streamlit UI for live swing-trading predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_swing_trader.data import ensure_directories, refresh_market_context
from ai_swing_trader.prediction import predict_latest_signal
from ai_swing_trader.tracking import (
    evaluate_prediction_log,
    summarize_prediction_outcomes,
    upsert_prediction_log,
)
from ai_swing_trader.watchlist import load_watchlist

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TICKER_FILE = PROJECT_ROOT / "stocks.txt"


def _format_currency(value: float) -> str:
    return f"{value:,.2f}"


def _load_predictions(
    ticker_file: str,
    period: str,
    min_confidence: float,
) -> tuple[pd.DataFrame, list[dict[str, str]], pd.DataFrame, dict[str, float]]:
    ensure_directories()
    watchlist = load_watchlist(ticker_file)
    market_frame = refresh_market_context(period=period)

    rows: list[dict[str, object]] = []
    prediction_payloads: list[dict] = []
    failures: list[dict[str, str]] = []

    for item in watchlist:
        ticker = item["ticker"]
        try:
            payload = predict_latest_signal(
                ticker=ticker,
                period=period,
                market_frame=market_frame,
            )
            if payload["confidence"] < min_confidence:
                continue
            prediction_payloads.append(payload)

            rows.append(
                {
                    "Ticker": ticker,
                    "Name": item["label"],
                    "Signal": payload["predicted_signal"],
                    "Confidence Band": payload["confidence_band"],
                    "Confidence": payload["confidence"],
                    "Signal Date": str(payload["signal_date"]),
                    "Current Close": payload["latest_close"],
                    "Predicted Price": payload["predicted_future_close"],
                    "Expected Move": payload["predicted_price_change"],
                    "Expected Return": payload["predicted_future_return"],
                    "BUY Prob": payload["probabilities"].get("BUY", 0.0),
                    "HOLD Prob": payload["probabilities"].get("HOLD", 0.0),
                    "SELL Prob": payload["probabilities"].get("SELL", 0.0),
                }
            )
        except Exception as exc:
            failures.append({"ticker": ticker, "reason": str(exc)})

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            by=["Signal", "Confidence", "Expected Return"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        upsert_prediction_log(prediction_payloads)
    outcomes = evaluate_prediction_log()
    summary = summarize_prediction_outcomes(outcomes)
    return frame, failures, outcomes, summary


def _signal_color(signal: str) -> str:
    if signal == "BUY":
        return "background-color: #dff3e4; color: #166534; font-weight: 700;"
    if signal == "SELL":
        return "background-color: #fde2e1; color: #b42318; font-weight: 700;"
    return "background-color: #fff1d6; color: #9a6700; font-weight: 700;"


def main() -> None:
    st.set_page_config(
        page_title="AI Swing Trader",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

    st.markdown(
        """
        <style>
          .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
          .metric-card {
            background: linear-gradient(135deg, #fffaf1, #f7efe1);
            border: 1px solid #ead8bd;
            border-radius: 18px;
            padding: 16px 18px;
          }
          .metric-label {
            color: #6b5b45;
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
          }
          .metric-value {
            color: #1f2933;
            font-size: 1.7rem;
            font-weight: 700;
            margin-top: 6px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI Swing Trader")
    st.caption("Live watchlist view for signals, confidence, and predicted future prices.")

    with st.sidebar:
        st.header("Controls")
        ticker_file = st.text_input("Ticker file", value=str(DEFAULT_TICKER_FILE))
        period = st.selectbox("Prediction lookback", ["6mo", "1y", "2y", "5y"], index=2)
        min_confidence = st.slider("Minimum confidence", min_value=0.0, max_value=0.9, value=0.45, step=0.05)
        allowed_signals = st.multiselect(
            "Signals to show",
            options=["BUY", "HOLD", "SELL"],
            default=["BUY", "HOLD", "SELL"],
        )
        refresh_now = st.button("Refresh Predictions", type="primary")

    if "last_frame" not in st.session_state:
        st.session_state.last_frame = None
        st.session_state.last_failures = []
        st.session_state.last_outcomes = None
        st.session_state.last_outcome_summary = None
        st.session_state.last_params = None

    params = {
        "ticker_file": ticker_file,
        "period": period,
        "min_confidence": min_confidence,
    }
    if refresh_now or st.session_state.last_frame is None or st.session_state.last_params != params:
        with st.spinner("Fetching market context and generating predictions..."):
            frame, failures, outcomes, outcome_summary = _load_predictions(
                ticker_file=ticker_file,
                period=period,
                min_confidence=min_confidence,
            )
        st.session_state.last_frame = frame
        st.session_state.last_failures = failures
        st.session_state.last_outcomes = outcomes
        st.session_state.last_outcome_summary = outcome_summary
        st.session_state.last_params = params

    frame = st.session_state.last_frame
    failures = st.session_state.last_failures
    outcomes = st.session_state.last_outcomes
    outcome_summary = st.session_state.last_outcome_summary or {
        "matured_predictions": 0,
        "direction_accuracy": 0.0,
        "mean_abs_pct_error": 0.0,
    }

    if frame is None or frame.empty:
        st.warning("No predictions available for the current filters. Try lowering the confidence threshold.")
        if failures:
            st.error(f"{len(failures)} tickers failed. Check the details section below.")
    else:
        filtered = frame[frame["Signal"].isin(allowed_signals)].copy()

        buy_count = int((filtered["Signal"] == "BUY").sum())
        hold_count = int((filtered["Signal"] == "HOLD").sum())
        sell_count = int((filtered["Signal"] == "SELL").sum())
        avg_confidence = float(filtered["Confidence"].mean()) if not filtered.empty else 0.0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Visible Stocks</div><div class="metric-value">{len(filtered)}</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">BUY / HOLD / SELL</div><div class="metric-value">{buy_count} / {hold_count} / {sell_count}</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Average Confidence</div><div class="metric-value">{avg_confidence:.2%}</div></div>',
                unsafe_allow_html=True,
            )
        with col4:
            best_row = filtered.sort_values("Confidence", ascending=False).iloc[0]
            st.markdown(
                (
                    '<div class="metric-card"><div class="metric-label">Top Setup</div>'
                    f'<div class="metric-value">{best_row["Ticker"]}</div></div>'
                ),
                unsafe_allow_html=True,
            )

        display_frame = filtered.copy()
        display_frame["Current Close"] = display_frame["Current Close"].map(_format_currency)
        display_frame["Predicted Price"] = display_frame["Predicted Price"].map(_format_currency)
        display_frame["Expected Move"] = display_frame["Expected Move"].map(lambda value: f"{value:+.2f}")
        display_frame["Expected Return"] = display_frame["Expected Return"].map(lambda value: f"{value:+.2%}")
        display_frame["Confidence"] = display_frame["Confidence"].map(lambda value: f"{value:.2%}")
        display_frame["BUY Prob"] = display_frame["BUY Prob"].map(lambda value: f"{value:.2%}")
        display_frame["HOLD Prob"] = display_frame["HOLD Prob"].map(lambda value: f"{value:.2%}")
        display_frame["SELL Prob"] = display_frame["SELL Prob"].map(lambda value: f"{value:.2%}")

        st.subheader("Predictions")
        st.dataframe(
            display_frame.style.map(_signal_color, subset=["Signal"]),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Predicted Price vs Current Close")
        chart_frame = filtered.set_index("Ticker")[["Current Close", "Predicted Price"]]
        st.bar_chart(chart_frame)

        st.subheader("Prediction Accuracy Tracker")
        out1, out2, out3 = st.columns(3)
        with out1:
            st.metric("Matured Predictions", int(outcome_summary["matured_predictions"]))
        with out2:
            st.metric("Direction Accuracy", f"{outcome_summary['direction_accuracy']:.2%}")
        with out3:
            st.metric("Mean Absolute % Error", f"{outcome_summary['mean_abs_pct_error']:.2%}")

        matured = pd.DataFrame()
        if outcomes is not None and not outcomes.empty:
            matured = outcomes.dropna(subset=["actual_future_close"]).copy()

        if matured.empty:
            st.info("No matured 5-day predictions yet. After a few trading days, this section will compare predicted vs actual prices.")
        else:
            matured = matured.sort_values(["signal_date", "ticker"], ascending=[False, True]).head(25)
            matured_display = matured[
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
            ].copy()
            matured_display.columns = [
                "Ticker",
                "Prediction Date",
                "Predicted Signal",
                "Predicted Price",
                "Actual Signal",
                "Actual Price",
                "Price Error",
                "Abs % Error",
                "Direction Correct",
            ]
            matured_display["Predicted Price"] = matured_display["Predicted Price"].map(_format_currency)
            matured_display["Actual Price"] = matured_display["Actual Price"].map(_format_currency)
            matured_display["Price Error"] = matured_display["Price Error"].map(lambda value: f"{float(value):+.2f}")
            matured_display["Abs % Error"] = matured_display["Abs % Error"].map(lambda value: f"{float(value):.2%}")
            matured_display["Direction Correct"] = matured_display["Direction Correct"].map(
                lambda value: "Yes" if str(value).lower() == "true" else "No"
            )
            st.dataframe(matured_display, width="stretch", hide_index=True)

    with st.expander("Ticker Failures", expanded=False):
        if failures:
            st.dataframe(pd.DataFrame(failures), width="stretch", hide_index=True)
        else:
            st.write("No ticker failures in the latest refresh.")


if __name__ == "__main__":
    main()
