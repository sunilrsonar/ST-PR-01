#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

print_help() {
  cat <<'EOF'
Usage:
  ./run_trading_bot.sh setup
  ./run_trading_bot.sh send [--bot-token TOKEN --chat-id CHAT_ID --min-confidence VALUE]
  ./run_trading_bot.sh train
  ./run_trading_bot.sh fetch
  ./run_trading_bot.sh refresh
  ./run_trading_bot.sh evaluate [--log-path PATH --limit N]
  ./run_trading_bot.sh ui [streamlit args]

Commands:
  setup   Create the virtual environment, install dependencies, and fetch/train all active stocks from stocks.txt.
  send    Generate predictions for active stocks from stocks.txt and send Telegram alerts.
  train   Re-run bulk fetch/training for active stocks from stocks.txt.
  fetch   Alias of train for convenience.
  refresh Remove stale data/artifacts for stocks no longer in stocks.txt, then fetch/train the active list.
  evaluate Evaluate matured predictions against actual outcomes.
  ui      Launch the local Streamlit interface for predictions.

Notes:
  - Edit stocks.txt to control which stocks are active.
  - For Telegram, either export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
    or pass --bot-token and --chat-id to the send command.
  - Use --min-confidence 0.45 or similar to send only stronger signals.
EOF
}

ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
}

install_requirements() {
  echo "Installing Python dependencies"
  "$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"
}

run_bulk_training() {
  echo "Fetching historical data and training active stocks from stocks.txt"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_from_file.py"
}

cleanup_stale_stock_files() {
  echo "Removing stale data and model artifacts not present in stocks.txt"

  mapfile -t active_tickers < <(
    awk 'NF && $1 !~ /^#/ {print $1}' "$ROOT_DIR/stocks.txt" | sort -u
  )

  local file
  local base
  local ticker
  local keep_file

  while IFS= read -r file; do
    base="$(basename "$file")"
    case "$base" in
      .gitkeep)
        continue
        ;;
    esac

    ticker="$base"
    ticker="${ticker%.csv}"
    ticker="${ticker%_features.csv}"
    ticker="${ticker%_rf_model.joblib}"

    keep_file=0
    local active
    for active in "${active_tickers[@]}"; do
      if [[ "$ticker" == "$active" ]]; then
        keep_file=1
        break
      fi
    done

    if [[ "$keep_file" -eq 0 ]]; then
      rm -f "$file"
    fi
  done < <(find "$ROOT_DIR/data/raw" "$ROOT_DIR/data/processed" "$ROOT_DIR/artifacts" -maxdepth 1 -type f)
}

run_send() {
  echo "Generating predictions and sending Telegram alert"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/predict_signal.py" --send-telegram "$@"
}

run_evaluate() {
  echo "Evaluating prediction log"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluate_predictions.py" "$@"
}

run_ui() {
  echo "Launching local UI"
  "$VENV_DIR/bin/streamlit" run "$ROOT_DIR/streamlit_app.py" "$@"
}

main() {
  if [[ $# -lt 1 ]]; then
    print_help
    exit 1
  fi

  local command="$1"
  shift

  case "$command" in
    setup)
      ensure_venv
      install_requirements
      cleanup_stale_stock_files
      run_bulk_training
      echo
      echo "Setup complete."
      echo "Next time, send alerts with:"
      echo "  ./run_trading_bot.sh send --bot-token YOUR_BOT_TOKEN --chat-id YOUR_CHAT_ID --min-confidence 0.45"
      ;;
    train|fetch)
      ensure_venv
      install_requirements
      run_bulk_training
      ;;
    refresh)
      ensure_venv
      install_requirements
      cleanup_stale_stock_files
      run_bulk_training
      ;;
    evaluate)
      ensure_venv
      install_requirements
      run_evaluate "$@"
      ;;
    send)
      ensure_venv
      run_send "$@"
      ;;
    ui)
      ensure_venv
      install_requirements
      run_ui "$@"
      ;;
    help|-h|--help)
      print_help
      ;;
    *)
      echo "Unknown command: $command" >&2
      echo >&2
      print_help
      exit 1
      ;;
  esac
}

main "$@"
