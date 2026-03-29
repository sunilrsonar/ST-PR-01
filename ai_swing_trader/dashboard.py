"""Standalone HTML dashboard generation for model and backtest artifacts."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

from ai_swing_trader.backtest import TRADING_DAYS_PER_YEAR
from ai_swing_trader.config import ARTIFACTS_DIR


def dashboard_path_for_ticker(ticker: str) -> Path:
    """Return the default output path for a dashboard HTML file."""
    safe_ticker = ticker.upper().replace("/", "_")
    return ARTIFACTS_DIR / f"{safe_ticker}_dashboard.html"


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_float(value: float) -> str:
    return f"{value:.2f}"


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def summarize_backtest_frame(frame: pd.DataFrame, transaction_cost_bps: float | None = None) -> dict:
    """Create dashboard-friendly performance metrics from a saved backtest frame."""
    if frame.empty:
        raise ValueError("Backtest frame is empty.")

    total_return = float(frame["equity_curve"].iloc[-1] - 1)
    benchmark_return = float(frame["benchmark_curve"].iloc[-1] - 1)
    daily_returns = frame["strategy_return_net"].astype(float)
    market_returns = frame["market_return"].astype(float)

    daily_std = float(daily_returns.std(ddof=0))
    sharpe_ratio = 0.0 if daily_std == 0 else float(daily_returns.mean() / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR))

    metrics = {
        "rows": int(len(frame)),
        "start_date": str(pd.Timestamp(frame["Date"].iloc[0]).date()),
        "end_date": str(pd.Timestamp(frame["Date"].iloc[-1]).date()),
        "transaction_cost_bps": float(transaction_cost_bps if transaction_cost_bps is not None else 0.0),
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "annualized_return": (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(frame)) - 1 if (1 + total_return) > 0 else -1.0,
        "annualized_volatility": float(daily_returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)),
        "benchmark_annualized_return": (1 + benchmark_return) ** (TRADING_DAYS_PER_YEAR / len(frame)) - 1 if (1 + benchmark_return) > 0 else -1.0,
        "benchmark_annualized_volatility": float(market_returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": float(frame["drawdown"].min()),
        "trade_count": int((frame["position"].ne(frame["position"].shift(1))).sum() - 1),
        "exposure_rate": float((frame["position"] != 0).mean()),
        "win_rate_in_market": float(((frame["strategy_return_net"] > 0) & (frame["position"] != 0)).sum() / max((frame["position"] != 0).sum(), 1)),
    }
    return metrics


def _make_chart_points(frame: pd.DataFrame, column: str) -> list[dict]:
    values: list[dict] = []
    for row in frame[["Date", column]].itertuples(index=False):
        values.append(
            {
                "date": str(pd.Timestamp(row[0]).date()),
                "value": round(float(row[1]), 6),
            }
        )
    return values


def _render_metric_card(label: str, value: str, tone: str = "neutral") -> str:
    return (
        f'<div class="metric-card {tone}">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        "</div>"
    )


def _render_distribution_bars(title: str, counts: dict[str, int], palette: dict[str, str]) -> str:
    total = max(sum(counts.values()), 1)
    bars = []
    for key, value in counts.items():
        width = (value / total) * 100
        bars.append(
            '<div class="dist-row">'
            f'<div class="dist-label">{html.escape(key)}</div>'
            '<div class="dist-bar-wrap">'
            f'<div class="dist-bar" style="width:{width:.2f}%; background:{palette.get(key, "#4f46e5")};"></div>'
            "</div>"
            f'<div class="dist-value">{value}</div>'
            "</div>"
        )
    return (
        '<section class="panel half">'
        f"<h2>{html.escape(title)}</h2>"
        + "".join(bars)
        + "</section>"
    )


def _render_feature_importance(feature_importances: pd.Series) -> str:
    if feature_importances.empty:
        return '<section class="panel"><h2>Feature Importance</h2><p>No feature importance data available.</p></section>'

    max_value = max(float(feature_importances.max()), 1e-9)
    rows = []
    for name, value in feature_importances.items():
        width = (float(value) / max_value) * 100
        rows.append(
            '<div class="importance-row">'
            f'<div class="importance-name">{html.escape(str(name))}</div>'
            '<div class="importance-bar-wrap">'
            f'<div class="importance-bar" style="width:{width:.2f}%;"></div>'
            "</div>"
            f'<div class="importance-value">{float(value):.4f}</div>'
            "</div>"
        )
    return '<section class="panel"><h2>Feature Importance</h2>' + "".join(rows) + "</section>"


def _render_recent_rows(frame: pd.DataFrame) -> str:
    recent = frame.tail(12).copy()
    rows = []
    for row in recent.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(pd.Timestamp(row.Date).date()))}</td>"
            f"<td>{_format_currency(float(row.Close))}</td>"
            f'<td><span class="signal-pill {str(row.predicted_signal).lower()}">{html.escape(str(row.predicted_signal))}</span></td>'
            f"<td>{int(row.position)}</td>"
            f"<td>{_format_pct(float(row.strategy_return_net))}</td>"
            f"<td>{_format_float(float(row.equity_curve))}</td>"
            "</tr>"
        )
    return (
        '<section class="panel wide">'
        "<h2>Recent Backtest Rows</h2>"
        "<table>"
        "<thead><tr><th>Date</th><th>Close</th><th>Predicted</th><th>Position</th><th>Net Return</th><th>Equity</th></tr></thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def build_dashboard_html(
    ticker: str,
    backtest_df: pd.DataFrame,
    artifact: dict,
    metrics: dict,
) -> str:
    """Build a standalone HTML dashboard string."""
    strategy_points = _make_chart_points(backtest_df, "equity_curve")
    benchmark_points = _make_chart_points(backtest_df, "benchmark_curve")
    drawdown_points = _make_chart_points(backtest_df, "drawdown")
    strategy_points_json = json.dumps(strategy_points, separators=(",", ":"))
    benchmark_points_json = json.dumps(benchmark_points, separators=(",", ":"))
    drawdown_points_json = json.dumps(drawdown_points, separators=(",", ":"))

    prediction_counts = backtest_df["predicted_signal"].value_counts().reindex(["BUY", "HOLD", "SELL"], fill_value=0).to_dict()
    actual_counts = backtest_df["signal"].value_counts().reindex(["BUY", "HOLD", "SELL"], fill_value=0).to_dict() if "signal" in backtest_df else {}

    model = artifact.get("classification_model") or artifact.get("model")
    feature_importances = pd.Series(dtype=float)
    if model is not None and hasattr(model, "feature_importances_"):
        feature_importances = (
            pd.Series(model.feature_importances_, index=artifact.get("feature_columns", []))
            .sort_values(ascending=False)
            .head(10)
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(ticker.upper())} Trading Dashboard</title>
  <style>
    :root {{
      --bg: #f5efe5;
      --ink: #1f2933;
      --muted: #5f6c7b;
      --card: #fffdf8;
      --line: #d8cdbd;
      --accent: #0f766e;
      --accent-2: #b45309;
      --danger: #b42318;
      --good: #166534;
      --shadow: 0 20px 40px rgba(45, 38, 27, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.12), transparent 25%),
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 28%),
        var(--bg);
      color: var(--ink);
    }}
    .shell {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,253,248,0.95), rgba(249,240,224,0.92));
      border: 1px solid rgba(113, 92, 62, 0.12);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px;
      margin-bottom: 24px;
    }}
    .eyebrow {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 12px;
      margin-bottom: 10px;
      font-weight: 700;
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-size: clamp(2rem, 3vw, 3.4rem);
      line-height: 1;
      margin-bottom: 10px;
    }}
    .subhead {{
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
      max-width: 820px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    .meta-chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255,255,255,0.75);
      font-size: 0.92rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 18px;
    }}
    .metrics {{
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }}
    .metric-card, .panel {{
      background: var(--card);
      border: 1px solid rgba(113, 92, 62, 0.12);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .metric-card {{
      padding: 18px;
    }}
    .metric-card.positive .metric-value {{ color: var(--good); }}
    .metric-card.negative .metric-value {{ color: var(--danger); }}
    .metric-label {{
      color: var(--muted);
      font-size: 0.88rem;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric-value {{
      font-size: 1.75rem;
      font-weight: 700;
    }}
    .panel {{
      padding: 20px;
    }}
    .panel h2 {{
      font-size: 1.2rem;
      margin-bottom: 14px;
    }}
    .wide {{ grid-column: span 12; }}
    .half {{ grid-column: span 6; }}
    @media (max-width: 920px) {{
      .half {{ grid-column: span 12; }}
    }}
    .chart-frame {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: linear-gradient(180deg, #fffdfa, #faf4ea);
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .dist-row, .importance-row {{
      display: grid;
      grid-template-columns: 78px 1fr 64px;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .dist-bar-wrap, .importance-bar-wrap {{
      width: 100%;
      height: 12px;
      border-radius: 999px;
      background: #efe4d4;
      overflow: hidden;
    }}
    .dist-bar, .importance-bar {{
      height: 100%;
      border-radius: 999px;
    }}
    .importance-bar {{
      background: linear-gradient(90deg, var(--accent), #14b8a6);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid #efe4d4;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
    }}
    .signal-pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.03em;
    }}
    .signal-pill.buy {{ background: rgba(22, 101, 52, 0.12); color: var(--good); }}
    .signal-pill.hold {{ background: rgba(180, 83, 9, 0.12); color: var(--accent-2); }}
    .signal-pill.sell {{ background: rgba(180, 35, 24, 0.12); color: var(--danger); }}
    .footer-note {{
      color: var(--muted);
      margin-top: 20px;
      font-size: 0.92rem;
      line-height: 1.6;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">AI Swing Trading Dashboard</div>
      <h1>{html.escape(ticker.upper())}</h1>
      <p class="subhead">
        Standalone report built from the saved model artifact and holdout backtest.
        This dashboard summarizes out-of-sample performance, predicted signal behavior,
        feature importance, and the latest backtest rows in one place.
      </p>
      <div class="meta">
        <div class="meta-chip">Window: {html.escape(metrics["start_date"])} to {html.escape(metrics["end_date"])}</div>
        <div class="meta-chip">Horizon: {artifact.get("horizon_days", "n/a")} trading days</div>
        <div class="meta-chip">Thresholds: BUY {artifact.get("buy_threshold", "n/a")} / SELL {artifact.get("sell_threshold", "n/a")}</div>
        <div class="meta-chip">Test split: {artifact.get("test_size", "n/a")}</div>
      </div>
    </section>

    <div class="grid">
      <section class="metrics">
        {_render_metric_card("Strategy Return", _format_pct(metrics["total_return"]), "positive" if metrics["total_return"] >= 0 else "negative")}
        {_render_metric_card("Benchmark Return", _format_pct(metrics["benchmark_return"]), "positive" if metrics["benchmark_return"] >= 0 else "negative")}
        {_render_metric_card("Sharpe Ratio", _format_float(metrics["sharpe_ratio"]), "positive" if metrics["sharpe_ratio"] >= 0 else "negative")}
        {_render_metric_card("Max Drawdown", _format_pct(metrics["max_drawdown"]), "negative")}
        {_render_metric_card("Trade Count", str(metrics["trade_count"]))}
        {_render_metric_card("Exposure Rate", _format_pct(metrics["exposure_rate"]))}
      </section>

      <section class="panel wide">
        <h2>Equity Curve vs Benchmark</h2>
        <div class="chart-frame">
          <svg id="equity-chart" viewBox="0 0 900 360" role="img" aria-label="Equity curve chart"></svg>
          <div class="legend">
            <div class="legend-item"><span class="legend-swatch" style="background:#0f766e;"></span>Strategy</div>
            <div class="legend-item"><span class="legend-swatch" style="background:#b45309;"></span>Benchmark</div>
          </div>
        </div>
      </section>

      <section class="panel half">
        <h2>Drawdown</h2>
        <div class="chart-frame">
          <svg id="drawdown-chart" viewBox="0 0 900 280" role="img" aria-label="Drawdown chart"></svg>
        </div>
      </section>

      {_render_distribution_bars("Predicted Signal Mix", prediction_counts, {"BUY": "#166534", "HOLD": "#b45309", "SELL": "#b42318"})}
      {_render_distribution_bars("Actual Signal Mix", actual_counts, {"BUY": "#166534", "HOLD": "#b45309", "SELL": "#b42318"})}

      <section class="panel half">
        <h2>Model Snapshot</h2>
        <div class="dist-row"><div class="dist-label">Rows</div><div class="dist-bar-wrap"><div class="dist-bar" style="width:100%; background:#0f766e;"></div></div><div class="dist-value">{metrics["rows"]}</div></div>
        <div class="dist-row"><div class="dist-label">Ann Ret</div><div class="dist-bar-wrap"><div class="dist-bar" style="width:{min(max((metrics["annualized_return"] + 0.5) * 60, 4), 100):.2f}%; background:#0f766e;"></div></div><div class="dist-value">{_format_pct(metrics["annualized_return"])}</div></div>
        <div class="dist-row"><div class="dist-label">Ann Vol</div><div class="dist-bar-wrap"><div class="dist-bar" style="width:{min(metrics["annualized_volatility"] * 100, 100):.2f}%; background:#b45309;"></div></div><div class="dist-value">{_format_pct(metrics["annualized_volatility"])}</div></div>
        <div class="dist-row"><div class="dist-label">Win Rate</div><div class="dist-bar-wrap"><div class="dist-bar" style="width:{metrics["win_rate_in_market"] * 100:.2f}%; background:#166534;"></div></div><div class="dist-value">{_format_pct(metrics["win_rate_in_market"])}</div></div>
      </section>

      <div class="half">
        {_render_feature_importance(feature_importances)}
      </div>

      {_render_recent_rows(backtest_df)}
    </div>

    <p class="footer-note">
      This is a simple analytical dashboard for model review, not a guarantee of future returns.
      The backtest is based on a single chronological holdout period and should be treated as an initial diagnostic.
    </p>
  </div>

  <script>
    const strategyPoints = {strategy_points_json};
    const benchmarkPoints = {benchmark_points_json};
    const drawdownPoints = {drawdown_points_json};

    function buildPath(points, width, height, minValue, maxValue, pad) {{
      if (!points.length || maxValue === minValue) return "";
      const xSpan = Math.max(points.length - 1, 1);
      return points.map((point, index) => {{
        const x = pad + (index / xSpan) * (width - pad * 2);
        const yRatio = (point.value - minValue) / (maxValue - minValue);
        const y = height - pad - yRatio * (height - pad * 2);
        return `${{index === 0 ? "M" : "L"}}${{x.toFixed(2)}},${{y.toFixed(2)}}`;
      }}).join(" ");
    }}

    function renderLineChart(targetId, primaryPoints, secondaryPoints, options) {{
      const svg = document.getElementById(targetId);
      const width = 900;
      const height = options.height;
      const pad = 36;
      const allValues = primaryPoints.map(p => p.value).concat(secondaryPoints.map(p => p.value));
      const minValue = Math.min(...allValues);
      const maxValue = Math.max(...allValues);

      const primaryPath = buildPath(primaryPoints, width, height, minValue, maxValue, pad);
      const secondaryPath = secondaryPoints.length ? buildPath(secondaryPoints, width, height, minValue, maxValue, pad) : "";

      svg.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" rx="18" fill="#fffdfa"></rect>
        <line x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad}}" y2="${{height - pad}}" stroke="#d8cdbd" stroke-width="1"></line>
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height - pad}}" stroke="#d8cdbd" stroke-width="1"></line>
        <path d="${{secondaryPath}}" fill="none" stroke="#b45309" stroke-width="3" stroke-linecap="round"></path>
        <path d="${{primaryPath}}" fill="none" stroke="#0f766e" stroke-width="3.5" stroke-linecap="round"></path>
      `;
    }}

    function renderDrawdownChart(targetId, points) {{
      const svg = document.getElementById(targetId);
      const width = 900;
      const height = 280;
      const pad = 36;
      const minValue = Math.min(...points.map(p => p.value), -0.0001);
      const maxValue = 0;
      const xSpan = Math.max(points.length - 1, 1);

      const line = points.map((point, index) => {{
        const x = pad + (index / xSpan) * (width - pad * 2);
        const yRatio = (point.value - minValue) / (maxValue - minValue);
        const y = height - pad - yRatio * (height - pad * 2);
        return `${{index === 0 ? "M" : "L"}}${{x.toFixed(2)}},${{y.toFixed(2)}}`;
      }}).join(" ");

      const area = line + ` L${{width - pad}},${{height - pad}} L${{pad}},${{height - pad}} Z`;

      svg.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" rx="18" fill="#fffdfa"></rect>
        <line x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad}}" y2="${{height - pad}}" stroke="#d8cdbd" stroke-width="1"></line>
        <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height - pad}}" stroke="#d8cdbd" stroke-width="1"></line>
        <path d="${{area}}" fill="rgba(180,35,24,0.18)"></path>
        <path d="${{line}}" fill="none" stroke="#b42318" stroke-width="3"></path>
      `;
    }}

    renderLineChart("equity-chart", strategyPoints, benchmarkPoints, {{ height: 360 }});
    renderDrawdownChart("drawdown-chart", drawdownPoints);
  </script>
</body>
</html>"""
