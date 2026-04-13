"""Helpers for reading ticker symbols and labels from watchlist files."""

from __future__ import annotations

from pathlib import Path


def load_watchlist(file_path: str | Path = "stocks.txt") -> list[dict[str, str]]:
    """Load a deduplicated watchlist from a plain-text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    items: list[dict[str, str]] = []
    seen: set[str] = set()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        ticker_part, _, label_part = raw_line.partition("#")
        label = label_part.strip()
        for raw_ticker in ticker_part.split(","):
            ticker = raw_ticker.strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            items.append(
                {
                    "ticker": ticker,
                    "label": label or ticker,
                }
            )

    return items
