import pandas as pd
from pathlib import Path
from datetime import datetime

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RANKED_PATH = PROJECT_ROOT / "data/processed/scanner_ranked.csv"
LOG_PATH = PROJECT_ROOT / "data/logs/scans_log.csv"
REGIME_PATH = PROJECT_ROOT / "data/processed/market_regime.csv"

# =========================
# HELPERS
# =========================

def load_ranked():
    if not RANKED_PATH.exists():
        print("⚠ scanner_ranked.csv not found")
        return pd.DataFrame()
    return pd.read_csv(RANKED_PATH)


def load_regime():
    if not REGIME_PATH.exists():
        return None

    df = pd.read_csv(REGIME_PATH)

    if df.empty:
        return None

    latest = df.iloc[-1]

    return {
        "regime": latest.get("regime", "UNKNOWN"),
        "risk_state": latest.get("risk_state", "UNKNOWN")
    }


def create_scan_id(row, scan_date):
    return f"{row['ticker']}_{scan_date}_{row['setup_type']}"


def ensure_log_file():
    if not LOG_PATH.exists():
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "scan_id", "scan_timestamp", "scan_date",
            "ticker", "setup_type", "setup_grade", "candidate_status",
            "regime", "risk_state",
            "score_total", "score_trend", "score_momentum", "score_position",
            "entry", "stop", "target", "rr",
            "close", "ma20", "ma50", "ma200", "high_20d", "atr14", "volume",
            "source_run_id", "validation_status"
        ]

        pd.DataFrame(columns=columns).to_csv(LOG_PATH, index=False)


def load_existing_ids():
    if not LOG_PATH.exists():
        return set()

    df = pd.read_csv(LOG_PATH, usecols=["scan_id"])
    return set(df["scan_id"].astype(str).values)


# =========================
# MAIN LOGIC
# =========================

def log_scans():

    df = load_ranked()

    if df.empty:
        print("⚠ No ranked setups to log")
        return

    ensure_log_file()

    existing_ids = load_existing_ids()

    regime_info = load_regime()

    now = datetime.utcnow()
    scan_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    scan_date = now.strftime("%Y-%m-%d")

    rows = []

    for _, row in df.iterrows():

        scan_id = create_scan_id(row, scan_date)

        if scan_id in existing_ids:
            continue

        log_row = {
            "scan_id": scan_id,
            "scan_timestamp": scan_timestamp,
            "scan_date": scan_date,

            "ticker": row.get("ticker"),
            "setup_type": row.get("setup_type"),
            "setup_grade": row.get("setup_grade"),
            "candidate_status": row.get("candidate_status"),

            "regime": regime_info["regime"] if regime_info else "UNKNOWN",
            "risk_state": regime_info["risk_state"] if regime_info else "UNKNOWN",

            "score_total": row.get("score_total"),
            "score_trend": row.get("score_trend"),
            "score_momentum": row.get("score_momentum"),
            "score_position": row.get("score_position"),

            "entry": row.get("entry"),
            "stop": row.get("stop"),
            "target": row.get("target"),
            "rr": row.get("rr"),

            "close": row.get("close"),
            "ma20": row.get("ma20"),
            "ma50": row.get("ma50"),
            "ma200": row.get("ma200"),
            "high_20d": row.get("high_20d"),
            "atr14": row.get("atr14"),
            "volume": row.get("volume"),

            "source_run_id": scan_timestamp,
            "validation_status": "PENDING"
        }

        rows.append(log_row)

    if not rows:
        print("ℹ No new scans to log")
        return

    new_df = pd.DataFrame(rows)

    new_df.to_csv(LOG_PATH, mode="a", header=False, index=False)

    print(f"✅ Logged {len(new_df)} new scans")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    log_scans()