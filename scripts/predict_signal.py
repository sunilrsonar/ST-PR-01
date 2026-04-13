#!/usr/bin/env python3
"""Generate the latest BUY, SELL, or HOLD signal from the trained model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_swing_trader.config import DEFAULT_LOOKBACK_PERIOD
from ai_swing_trader.data import ensure_directories, refresh_market_context
from ai_swing_trader.notifications import (
    build_multi_signal_messages,
    build_signal_message,
    resolve_telegram_credentials,
    send_telegram_messages,
)
from ai_swing_trader.prediction import predict_latest_signal
from ai_swing_trader.tracking import evaluate_prediction_log, upsert_prediction_log


def _load_tickers_from_file(file_path: Path) -> list[str]:
    """Load ticker symbols from a plain-text file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Ticker file not found: {file_path}")

    tickers: list[str] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        tickers.extend(part.strip() for part in line.split(",") if part.strip())
    return tickers


def _normalize_tickers(raw_values: list[str]) -> list[str]:
    """Accept repeated or comma-separated tickers and return a clean unique list."""
    tickers: list[str] = []
    seen: set[str] = set()

    for raw_value in raw_values:
        for item in raw_value.split(","):
            ticker = item.strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)

    if not tickers:
        raise ValueError("At least one valid ticker is required.")
    return tickers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the latest swing trading signal.")
    parser.add_argument(
        "--ticker",
        nargs="+",
        help="One or more stock tickers, for example AAPL MSFT or AAPL,MSFT.",
    )
    parser.add_argument(
        "--ticker-file",
        default="stocks.txt",
        help="Path to a file containing one or more ticker symbols. Defaults to stocks.txt.",
    )
    parser.add_argument("--period", default=DEFAULT_LOOKBACK_PERIOD, help="Recent history window for prediction, for example 2y.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Only keep predictions whose top class probability is at least this value, for example 0.45.",
    )
    parser.add_argument("--send-telegram", action="store_true", help="Send the latest prediction to Telegram.")
    parser.add_argument("--bot-token", default=None, help="Telegram bot token. Defaults to TELEGRAM_BOT_TOKEN.")
    parser.add_argument("--chat-id", default=None, help="Telegram chat ID. Defaults to TELEGRAM_CHAT_ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    market_frame = refresh_market_context(period=args.period)

    raw_tickers: list[str] = []
    if args.ticker:
        raw_tickers.extend(args.ticker)

    ticker_file = Path(args.ticker_file)
    if ticker_file.exists():
        raw_tickers.extend(_load_tickers_from_file(ticker_file))

    if not raw_tickers:
        raise ValueError(
            "No tickers provided. Use --ticker or add ticker symbols to stocks.txt."
        )

    tickers = _normalize_tickers(raw_tickers)
    prediction_payloads = []
    skipped: list[tuple[str, str]] = []
    filtered_out: list[tuple[str, float]] = []
    for ticker in tickers:
        try:
            payload = predict_latest_signal(
                ticker=ticker,
                period=args.period,
                market_frame=market_frame,
            )
            if payload["confidence"] < args.min_confidence:
                filtered_out.append((ticker, payload["confidence"]))
                continue
            prediction_payloads.append(payload)
        except Exception as exc:
            skipped.append((ticker, str(exc)))

    if not prediction_payloads:
        raise RuntimeError("No predictions could be generated after filtering the provided tickers.")

    upsert_prediction_log(prediction_payloads)
    evaluate_prediction_log()

    for prediction_payload in prediction_payloads:
        print(f"Ticker: {prediction_payload['ticker']}")
        print(f"Signal date: {prediction_payload['signal_date']}")
        print(f"Latest close: {prediction_payload['latest_close']:.2f}")
        print(f"Predicted signal: {prediction_payload['predicted_signal']}")
        print(
            f"Confidence: {prediction_payload['confidence_band']} "
            f"({prediction_payload['confidence']:.2%})"
        )
        print(
            "Expected move over "
            f"{prediction_payload['horizon_days']} days: "
            f"{prediction_payload['predicted_price_change']:+.2f} "
            f"({prediction_payload['predicted_future_return']:+.2%})"
        )
        print(f"Expected future close: {prediction_payload['predicted_future_close']:.2f}")
        print(f"Class probabilities: {prediction_payload['probabilities']}")
        print()

    if filtered_out:
        print("Filtered out by confidence:")
        for ticker, confidence in filtered_out:
            print(f"- {ticker}: {confidence:.2%} < {args.min_confidence:.2%}")
        print()

    if skipped:
        print("Skipped tickers:")
        for ticker, reason in skipped:
            print(f"- {ticker}: {reason}")
        print()

    if args.send_telegram:
        bot_token, chat_id = resolve_telegram_credentials(
            bot_token=args.bot_token,
            chat_id=args.chat_id,
        )
        if len(prediction_payloads) == 1:
            messages = [build_signal_message(prediction_payloads[0])]
        else:
            messages = build_multi_signal_messages(prediction_payloads)
        send_telegram_messages(messages=messages, bot_token=bot_token, chat_id=chat_id)
        print(f"Telegram notification sent successfully in {len(messages)} message(s).")


if __name__ == "__main__":
    main()
