from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/scanner_ranked.csv")
OUTPUT_PATH = Path("data/processed/validation_layer.csv")

REQUIRED_COLUMNS = [
    "ticker",
    "setup",
    "primary_setup",
    "close",
    "high_20d",
    "ma20",
    "ma50",
    "entry",
    "stop",
    "target",
    "rr",
    "grade",
]


def _is_missing(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _get_setup(row: pd.Series) -> str:
    primary_setup = row.get("primary_setup")

    if not _is_missing(primary_setup):
        return str(primary_setup).upper().strip()

    setup = row.get("setup")

    if not _is_missing(setup):
        return str(setup).upper().strip().split(",")[0].strip()

    return ""


def _has_required_price_fields(row: pd.Series) -> bool:
    required = ["close", "entry", "stop", "target", "rr"]

    for column in required:
        if column not in row.index or _is_missing(row.get(column)):
            return False

    return True


def evaluate_valid_setup(row: pd.Series) -> tuple[bool, str]:
    """
    Sprint 1 rule:
    VALID_SETUP = technische setup-validatie op basis van scanner-output.

    Minimale edge logic:
    - Alleen A-grade setups zijn valid.
    - Alleen BREAKOUT en PULLBACK worden voorlopig toegelaten.
    - RR moet minimaal 2.0 zijn.

    Geen fundamentals.
    Geen confidence.
    Geen decision logic.
    Geen context layer.
    """

    if not _has_required_price_fields(row):
        return False, "missing_data"

    setup = _get_setup(row)

    if not setup:
        return False, "no_setup"

    grade = str(row.get("grade", "")).upper().strip()

    try:
        rr = float(row.get("rr", 0))
    except (TypeError, ValueError):
        return False, "invalid_rr"

    if grade != "A":
        return False, "filtered_non_A"

    if setup not in {"BREAKOUT", "PULLBACK"}:
        return False, "filtered_setup_type"

    if rr < 2.0:
        return False, "filtered_rr"

    if setup == "BREAKOUT":
        return True, "valid_breakout"

    if setup == "PULLBACK":
        return True, "valid_pullback"

    return False, "unknown_setup"


def build_validation_layer() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Scanner ranked file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"scanner_ranked.csv is missing required columns: {missing_columns}"
        )

    rows: list[dict] = []
    validation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in df.iterrows():
        valid_setup, validation_reason = evaluate_valid_setup(row)

        rows.append(
            {
                "ticker": str(row["ticker"]).upper().strip(),
                "date": validation_date,
                "valid_setup": bool(valid_setup),
                "tradeable_setup": bool(valid_setup),  # Sprint 1 rule
                "validation_reason": validation_reason,
            }
        )

    validation_df = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "date",
            "valid_setup",
            "tradeable_setup",
            "validation_reason",
        ],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Validation layer written to: {OUTPUT_PATH}")

    return validation_df


if __name__ == "__main__":
    build_validation_layer()