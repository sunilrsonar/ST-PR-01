# AI Swing Trading System

This project builds a simple machine learning swing trading workflow for daily stock data.
It downloads OHLCV data, engineers technical indicators, labels future price movement,
trains a `RandomForestClassifier`, predicts the latest `BUY`, `SELL`, or `HOLD` signal,
and can send the signal to Telegram.

## Project Structure

```text
ST-PR-01/
├── ai_swing_trader/
│   ├── __init__.py
│   ├── backtest.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── labels.py
│   └── model.py
├── artifacts/
├── data/
│   ├── processed/
│   └── raw/
├── scripts/
│   ├── backtest_strategy.py
│   ├── fetch_data.py
│   ├── generate_dashboard.py
│   ├── predict_signal.py
│   ├── plot_backtest.py
│   └── train_model.py
└── requirements.txt
```

## Setup

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## One-Command Workflow

Use the shell helper for the simplest flow:

```bash
./run_trading_bot.sh setup
./run_trading_bot.sh send --bot-token "YOUR_BOT_TOKEN" --chat-id "YOUR_CHAT_ID" --min-confidence 0.45
```

What it does:

- `setup`: creates `.venv`, installs all requirements, fetches data, and trains all active stocks from `stocks.txt`
- `send`: loads the trained models, filters weaker predictions if needed, and sends the Telegram alert

## Step 1: Download Historical Data

Example:

```bash
python3 scripts/fetch_data.py --ticker AAPL --start 2015-01-01
```

This saves daily OHLCV data to `data/raw/AAPL.csv`.

## Step 2: Train the Model

Example:

```bash
python3 scripts/train_model.py --ticker AAPL
```

This script:

- Loads `data/raw/AAPL.csv`
- Adds technical indicators
- Creates labels from the 5-day future return
- Splits the data chronologically into train and test sets
- Trains a random forest classifier
- Prints classification metrics
- Saves the trained artifact to `artifacts/AAPL_rf_model.joblib`

## Step 3: Generate the Latest Signal

Example:

```bash
python3 scripts/predict_signal.py --ticker AAPL
```

Multiple stocks:

```bash
python3 scripts/predict_signal.py --ticker AAPL MSFT TSLA
```

Or use the stock list file:

```bash
python3 scripts/predict_signal.py
```

This downloads recent data, applies the same features, loads the saved model,
and prints the latest trading signal, class probabilities, and the model's estimated
future price move over the forecast horizon.

## Step 4: Send the Signal to Telegram

Set your Telegram credentials first:

```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

Then run:

```bash
python3 scripts/predict_signal.py --send-telegram
```

This script will:

- Fetch the latest market data
- Generate the newest `BUY`, `SELL`, or `HOLD` signal
- Estimate the expected future return and price change over the horizon
- Optionally filter out low-confidence signals
- Print the signal locally
- Send the same signals to your Telegram chat in one message

By default, it reads tickers from `stocks.txt`. You can edit that file and keep one symbol per line.

## Bulk Training

Example:

```bash
python3 scripts/train_from_file.py
```

This script will:

- Read all uncommented tickers from `stocks.txt`
- Download fresh historical data for each symbol
- Train the classification and regression models
- Write any failures to `failed_stocks.txt`

## Step 5: Backtest the Strategy

Example:

```bash
python3 scripts/backtest_strategy.py --ticker AAPL --save-csv
```

This script:

- Loads the trained model artifact
- Reuses the chronological test split from training
- Generates model predictions on the holdout period
- Simulates a daily flat/long/short strategy
- Applies transaction costs
- Prints return, Sharpe ratio, drawdown, and benchmark comparison
- Optionally saves the detailed equity curve to `artifacts/AAPL_backtest.csv`

## Step 6: Plot the Backtest

Example:

```bash
python3 scripts/plot_backtest.py --ticker AAPL
```

This script:

- Loads `artifacts/AAPL_backtest.csv`
- Creates an equity curve chart versus buy-and-hold
- Plots the strategy drawdown
- Saves the image to `artifacts/AAPL_backtest.png`

## Step 7: Generate the HTML Dashboard

Example:

```bash
python3 scripts/generate_dashboard.py --ticker AAPL
```

This script:

- Loads the saved backtest CSV and model artifact
- Builds a standalone HTML report
- Includes summary metrics, signal mix, feature importance, and recent rows
- Saves the result to `artifacts/AAPL_dashboard.html`

## Labeling Rules

The default labeling horizon is 5 trading days:

- `BUY`: future 5-day return is greater than or equal to `+3%`
- `SELL`: future 5-day return is less than or equal to `-3%`
- `HOLD`: everything in between

You can change these thresholds from the training script arguments.

## Notes

- This project is educational and should not be treated as financial advice.
- The model uses only historical price/volume information and technical indicators.
- Accuracy will vary by ticker, time period, and market regime.
- The included backtest is a simple holdout backtest, not a full walk-forward portfolio simulator.
- Telegram delivery requires a bot token and chat ID from your own Telegram bot setup.
