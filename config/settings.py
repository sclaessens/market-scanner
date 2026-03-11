from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"
FEATURES_DIR = DATA_DIR / "features"
LOGS_DIR = DATA_DIR / "logs"

REPORTS_DIR = PROJECT_ROOT / "reports"

TICKERS_FILE = PROJECT_ROOT / "tickers.txt"
SCANS_LOG_FILE = DATA_DIR / "scans_log.csv"

INDEX_TICKERS = ["QQQ", "SPY"]

MIN_VOLUME = 1_000_000
PROBABILITY_THRESHOLD = 0.60
MIN_RR = 1.8
