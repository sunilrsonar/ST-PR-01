"""Notification helpers for Telegram delivery."""

from __future__ import annotations

import json
import os
from urllib import error, parse, request


TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAM_MESSAGE_LIMIT = 3800


def resolve_telegram_credentials(
    bot_token: str | None = None,
    chat_id: str | None = None,
) -> tuple[str, str]:
    """Resolve Telegram credentials from CLI arguments or environment variables."""
    resolved_bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    resolved_chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not resolved_bot_token:
        raise ValueError("Missing Telegram bot token. Set TELEGRAM_BOT_TOKEN or pass --bot-token.")
    if not resolved_chat_id:
        raise ValueError("Missing Telegram chat ID. Set TELEGRAM_CHAT_ID or pass --chat-id.")

    return resolved_bot_token, resolved_chat_id


def send_telegram_message(
    message: str,
    bot_token: str,
    chat_id: str,
    timeout_seconds: int = 20,
) -> dict:
    """Send a plain-text Telegram bot message."""
    endpoint = f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"
    payload = parse.urlencode(
        {
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")

    request_obj = request.Request(endpoint, data=payload, method="POST")
    try:
        with request.urlopen(request_obj, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram API returned HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Telegram request failed: {exc.reason}") from exc

    result = json.loads(response_body)
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")
    return result


def send_telegram_messages(
    messages: list[str],
    bot_token: str,
    chat_id: str,
    timeout_seconds: int = 20,
) -> list[dict]:
    """Send one or more Telegram messages in sequence."""
    results = []
    for message in messages:
        results.append(
            send_telegram_message(
                message=message,
                bot_token=bot_token,
                chat_id=chat_id,
                timeout_seconds=timeout_seconds,
            )
        )
    return results


def build_signal_message(prediction_payload: dict) -> str:
    """Create a concise Telegram message from the latest prediction payload."""
    probabilities = prediction_payload["probabilities"]
    return "\n".join(
        [
            "AI Swing Trading Signal",
            f"Ticker: {prediction_payload['ticker']}",
            f"Date: {prediction_payload['signal_date']}",
            f"Close: {prediction_payload['latest_close']:.2f}",
            f"Signal: {prediction_payload['predicted_signal']}",
            (
                f"Expected {prediction_payload['horizon_days']}-day move: "
                f"{prediction_payload['predicted_price_change']:+.2f} "
                f"({prediction_payload['predicted_future_return']:+.2%})"
            ),
            f"Expected future close: {prediction_payload['predicted_future_close']:.2f}",
            (
                "Probabilities: "
                f"BUY {probabilities.get('BUY', 0.0):.2%}, "
                f"HOLD {probabilities.get('HOLD', 0.0):.2%}, "
                f"SELL {probabilities.get('SELL', 0.0):.2%}"
            ),
        ]
    )


def build_multi_signal_message(prediction_payloads: list[dict]) -> str:
    """Create a Telegram message covering multiple ticker predictions."""
    if not prediction_payloads:
        raise ValueError("prediction_payloads must not be empty.")

    lines = ["AI Swing Trading Signals", ""]
    for payload in prediction_payloads:
        probabilities = payload["probabilities"]
        lines.extend(
            [
                (
                    f"{payload['ticker']} | {payload['predicted_signal']} | "
                    f"Close {payload['latest_close']:.2f} | Date {payload['signal_date']}"
                ),
                (
                    f"Expected {payload['horizon_days']}-day move: "
                    f"{payload['predicted_price_change']:+.2f} "
                    f"({payload['predicted_future_return']:+.2%}), "
                    f"future close {payload['predicted_future_close']:.2f}"
                ),
                (
                    "Probabilities: "
                    f"BUY {probabilities.get('BUY', 0.0):.2%}, "
                    f"HOLD {probabilities.get('HOLD', 0.0):.2%}, "
                    f"SELL {probabilities.get('SELL', 0.0):.2%}"
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def build_multi_signal_messages(
    prediction_payloads: list[dict],
    max_length: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:
    """Split multi-stock alerts into Telegram-safe message chunks."""
    if not prediction_payloads:
        raise ValueError("prediction_payloads must not be empty.")

    blocks = []
    for payload in prediction_payloads:
        probabilities = payload["probabilities"]
        block = "\n".join(
            [
                (
                    f"{payload['ticker']} | {payload['predicted_signal']} | "
                    f"Close {payload['latest_close']:.2f} | Date {payload['signal_date']}"
                ),
                (
                    f"Expected {payload['horizon_days']}-day move: "
                    f"{payload['predicted_price_change']:+.2f} "
                    f"({payload['predicted_future_return']:+.2%}), "
                    f"future close {payload['predicted_future_close']:.2f}"
                ),
                (
                    "Probabilities: "
                    f"BUY {probabilities.get('BUY', 0.0):.2%}, "
                    f"HOLD {probabilities.get('HOLD', 0.0):.2%}, "
                    f"SELL {probabilities.get('SELL', 0.0):.2%}"
                ),
            ]
        )
        blocks.append(block)

    header = "AI Swing Trading Signals"
    chunks: list[str] = []
    current_blocks: list[str] = []

    def build_chunk(block_items: list[str]) -> str:
        return "\n\n".join([header, *block_items]).rstrip()

    for block in blocks:
        candidate_blocks = [*current_blocks, block]
        candidate = build_chunk(candidate_blocks)
        if len(candidate) <= max_length:
            current_blocks = candidate_blocks
            continue

        if current_blocks:
            chunks.append(build_chunk(current_blocks))
            current_blocks = [block]
        else:
            chunks.append(build_chunk([block]))
            current_blocks = []

    if current_blocks:
        chunks.append(build_chunk(current_blocks))

    if len(chunks) == 1:
        return chunks

    labeled_chunks = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        prefix = f"{header} ({index}/{total})"
        body = chunk.split("\n\n", 1)[1] if "\n\n" in chunk else ""
        labeled_chunks.append("\n\n".join([prefix, body]).rstrip())
    return labeled_chunks
