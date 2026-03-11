import csv
from config.settings import (
    DATA_DIR,
    PRICES_DIR,
    FEATURES_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    TICKERS_FILE,
    SCANS_LOG_FILE,
)

def ensure_directories():

    for path in [
        DATA_DIR,
        PRICES_DIR,
        FEATURES_DIR,
        LOGS_DIR,
        REPORTS_DIR
    ]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_tickers_file():

    if not TICKERS_FILE.exists():
        raise FileNotFoundError(
            f"Missing required file: {TICKERS_FILE}"
        )


def ensure_scans_log():

    if not SCANS_LOG_FILE.exists():

        with open(SCANS_LOG_FILE, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "ticker",
                "setup",
                "score",
                "regime",
                "notes"
            ])


def validate_inputs():

    ensure_directories()
    ensure_tickers_file()
    ensure_scans_log()
