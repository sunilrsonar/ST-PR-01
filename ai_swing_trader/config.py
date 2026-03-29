"""Project-wide configuration values."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_TRAIN_START = "2015-01-01"
DEFAULT_LOOKBACK_PERIOD = "2y"
DEFAULT_HORIZON_DAYS = 5
DEFAULT_BUY_THRESHOLD = 0.03
DEFAULT_SELL_THRESHOLD = -0.03
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

SIGNAL_TO_TARGET = {"SELL": -1, "HOLD": 0, "BUY": 1}
TARGET_TO_SIGNAL = {value: key for key, value in SIGNAL_TO_TARGET.items()}
