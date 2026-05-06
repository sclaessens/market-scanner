from pathlib import Path


# =========================
# Project root
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =========================
# Data directories
# =========================
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"
WATCHLIST_DIR = DATA_DIR / "watchlist"
PORTFOLIO_DIR = DATA_DIR / "portfolio"


# =========================
# Reports
# =========================
REPORTS_DIR = PROJECT_ROOT / "reports"


# =========================
# Files
# =========================
TICKERS_FILE = PROJECT_ROOT / "tickers.txt"

SCANS_LOG_FILE = LOGS_DIR / "scans_log.csv"


# =========================
# Index tickers
# =========================
INDEX_TICKERS = ["QQQ", "SPY"]


# =========================
# Global filters
# =========================
MIN_VOLUME = 1_000_000
MIN_AVG_VOLUME = 1_000_000
MIN_PRICE = 10


# =========================
# Risk / model thresholds
# =========================
PROBABILITY_THRESHOLD = 0.60
MIN_RR = 2.0


# =========================
# Entry quality validation
# =========================
ENTRY_QUALITY_CONFIG = {
    "max_distance_breakout_pct": 3.0,
    "max_breakout_extension_atr": 2.0,
    "max_extension_atr": 2.5,
    "min_volume_ratio": 1.2,
    "max_volume_ratio": 4.0,
    "max_range_atr": 2.5,
}

assert (
    0
    < ENTRY_QUALITY_CONFIG["min_volume_ratio"]
    < ENTRY_QUALITY_CONFIG["max_volume_ratio"]
)
assert ENTRY_QUALITY_CONFIG["max_extension_atr"] > 0
assert ENTRY_QUALITY_CONFIG["max_distance_breakout_pct"] > 0


# =========================
# Report settings
# =========================
REPORT_EMPTY_TEXT = "(none)"
TOP_SETUPS_PER_SECTION = 10


# =========================
# VCP settings
# =========================
VCP_LOOKBACK_DAYS = 60
VCP_NEAR_HIGH_THRESHOLD = 0.95
VCP_CONTRACTION_THRESHOLD = 0.12