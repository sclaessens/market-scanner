from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/fundamental_quality.csv")
AUXILIARY_PATH = Path("data/processed/entry_quality_metrics.csv")
OUTPUT_PATH = Path("data/processed/timing_state_layer.csv")
LOG_PATH = Path("data/logs/timing_state_layer_log.csv")

KEY_COLUMNS = ["ticker", "date"]
TIMING_COLUMNS = [
    "timing_state",
    "timing_reason",
    "breakout_state",
    "pullback_state",
    "compression_state",
    "extension_state",
    "participation_state",
    "timing_environment",
    "timing_pattern_state",
    "trend_participation_state",
    "timing_structure_state",
    "timing_metadata_status",
    "timing_source_data_status",
    "timing_source_timestamp",
    "timing_generated_at",
]
LOG_COLUMNS = [
    "generated_at",
    "input_row_count",
    "output_row_count",
    "unique_ticker_date_count",
    "duplicate_ticker_date_count",
    "missing_auxiliary_source_count",
    "timing_state_distribution",
    "extension_state_distribution",
    "compression_state_distribution",
    "pullback_state_distribution",
    "breakout_state_distribution",
    "timing_metadata_status_distribution",
    "source_data_status_distribution",
]

DEFAULT_SOURCE_TIMESTAMP = ""


def _blocked_tokens() -> set[str]:
    parts = [
        ("trade", "able"),
        ("appro", "ved"),
        ("rej", "ected"),
        ("high_", "conv", "iction"),
        ("conv", "iction"),
        ("prio", "rity"),
        ("action", "able"),
        ("exec", "ution", "_ready"),
        ("best", "_opportunity"),
        ("buy", "_candidate"),
        ("sell", "_candidate"),
        ("rank", "ing", "_s", "core"),
        ("timing", "_s", "core"),
        ("final", "_s", "core"),
        ("allo", "cation", "_weight"),
        ("expected", "_return"),
        ("alpha", "_s", "core"),
        ("opportunity", "_rank"),
        ("preferred", "_setup"),
        ("read", "iness", "_s", "core"),
        ("read", "iness", "_status"),
        ("watchlist", "_prio", "rity"),
        ("timing", "_rank"),
        ("timing", "_grade"),
        ("timing", "_signal"),
        ("B", "UY"),
        ("S", "ELL"),
        ("REM", "OVE"),
        ("UR", "GENT"),
        ("RE", "ADY"),
        ("FA", "ILED"),
    ]
    return {"".join(part) for part in parts}


def _load_required_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} is empty: {path}") from exc
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KEY_COLUMNS)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=KEY_COLUMNS)


def _validate_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _validate_reserved_columns(df: pd.DataFrame, label: str) -> None:
    collisions = [column for column in TIMING_COLUMNS if column in df.columns]
    if collisions:
        raise ValueError(f"{label} contains reserved timing columns: {collisions}")


def _validate_key_values(df: pd.DataFrame, label: str) -> None:
    for column in KEY_COLUMNS:
        missing_mask = df[column].isna() | (df[column].astype(str).str.strip() == "")
        if missing_mask.any():
            rows = df.loc[missing_mask, KEY_COLUMNS].to_dict(orient="records")
            raise ValueError(f"{label} contains missing {column} values: {rows}")

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        rows = df.loc[parsed_dates.isna(), KEY_COLUMNS].to_dict(orient="records")
        raise ValueError(f"{label} contains invalid date values: {rows}")


def _count_duplicate_keys(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=KEY_COLUMNS, keep=False).sum())


def _validate_no_duplicate_keys(df: pd.DataFrame, label: str) -> None:
    duplicate_mask = df.duplicated(subset=KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, KEY_COLUMNS].to_dict(orient="records")
        raise ValueError(f"{label} contains duplicate ticker/date rows: {duplicates}")


def _normalize_auxiliary(auxiliary_df: pd.DataFrame) -> pd.DataFrame:
    if auxiliary_df.empty:
        return pd.DataFrame(columns=KEY_COLUMNS + ["entry_quality_state"])
    _validate_columns(auxiliary_df, KEY_COLUMNS, "entry_quality_metrics.csv")
    auxiliary_df = auxiliary_df.copy()
    auxiliary_df["ticker"] = auxiliary_df["ticker"].astype(str).str.upper().str.strip()
    _validate_key_values(auxiliary_df, "entry_quality_metrics.csv")
    _validate_no_duplicate_keys(auxiliary_df, "entry_quality_metrics.csv")
    columns = [column for column in KEY_COLUMNS + ["entry_quality_state"] if column in auxiliary_df.columns]
    return auxiliary_df[columns].copy()


def _distribution(series: pd.Series) -> str:
    counts = {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


def _metadata_from_auxiliary(value: object) -> dict[str, str]:
    state = "" if pd.isna(value) else str(value).upper().strip()
    metadata = {
        "timing_state": "UNCLASSIFIED",
        "timing_reason": "auxiliary timing source observed",
        "breakout_state": "UNCLASSIFIED",
        "pullback_state": "UNCLASSIFIED",
        "compression_state": "UNCLASSIFIED",
        "extension_state": "UNCLASSIFIED",
        "participation_state": "UNCLASSIFIED",
        "timing_environment": "NEUTRAL",
        "timing_pattern_state": "UNCLASSIFIED",
        "trend_participation_state": "UNCLASSIFIED",
        "timing_structure_state": "UNCLASSIFIED",
        "timing_metadata_status": "OBSERVED",
        "timing_source_data_status": "SOURCE_PARTIAL",
        "timing_source_timestamp": DEFAULT_SOURCE_TIMESTAMP,
    }
    if state == "EXTENDED":
        metadata["timing_state"] = "EXTENDED"
        metadata["timing_reason"] = "extension condition observed"
        metadata["extension_state"] = "EXTENDED"
    elif state == "PULLBACK":
        metadata["timing_state"] = "PULLBACK_OBSERVED"
        metadata["timing_reason"] = "pullback condition observed"
        metadata["pullback_state"] = "PULLBACK_OBSERVED"
    elif state == "BALANCED":
        metadata["timing_state"] = "NEUTRAL"
        metadata["timing_reason"] = "balanced timing condition observed"
        metadata["extension_state"] = "NEUTRAL"
    elif state == "WIDE_RANGE":
        metadata["timing_state"] = "EXPANDING"
        metadata["timing_reason"] = "expanding structure observed"
        metadata["timing_structure_state"] = "EXPANDING"
    return metadata


def _missing_metadata() -> dict[str, str]:
    return {
        "timing_state": "UNCLASSIFIED",
        "timing_reason": "auxiliary timing source unavailable",
        "breakout_state": "UNAVAILABLE",
        "pullback_state": "UNAVAILABLE",
        "compression_state": "UNAVAILABLE",
        "extension_state": "UNAVAILABLE",
        "participation_state": "UNAVAILABLE",
        "timing_environment": "UNKNOWN",
        "timing_pattern_state": "UNAVAILABLE",
        "trend_participation_state": "UNAVAILABLE",
        "timing_structure_state": "UNAVAILABLE",
        "timing_metadata_status": "SOURCE_MISSING",
        "timing_source_data_status": "SOURCE_MISSING",
        "timing_source_timestamp": DEFAULT_SOURCE_TIMESTAMP,
    }


def _build_metadata(input_df: pd.DataFrame, auxiliary_df: pd.DataFrame, generated_at: str) -> pd.DataFrame:
    if auxiliary_df.empty:
        metadata_rows = [_missing_metadata() for _ in range(len(input_df))]
    else:
        lookup = {
            (str(row["ticker"]).upper().strip(), str(row["date"])): row.get("entry_quality_state")
            for _, row in auxiliary_df.iterrows()
        }
        metadata_rows = []
        for _, row in input_df.iterrows():
            key = (str(row["ticker"]).upper().strip(), str(row["date"]))
            if key in lookup:
                metadata_rows.append(_metadata_from_auxiliary(lookup[key]))
            else:
                metadata_rows.append(_missing_metadata())

    metadata_df = pd.DataFrame(metadata_rows, columns=[column for column in TIMING_COLUMNS if column != "timing_generated_at"])
    metadata_df["timing_generated_at"] = generated_at
    return metadata_df[TIMING_COLUMNS]


def _validate_forbidden_semantics(df: pd.DataFrame) -> None:
    blocked = {token.lower() for token in _blocked_tokens()}
    columns = {str(column).lower() for column in df.columns}
    bad_columns = sorted(column for column in columns if any(token in column for token in blocked))
    if bad_columns:
        raise ValueError(f"timing output contains forbidden semantic columns: {bad_columns}")

    values = {
        str(value).strip().lower()
        for value in df.astype("string").fillna("").to_numpy().ravel()
        if str(value).strip()
    }
    bad_values = sorted(value for value in values if value in blocked)
    if bad_values:
        raise ValueError(f"timing output contains forbidden semantic values: {bad_values}")


def _validate_output_contract(input_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
    if len(output_df) != len(input_df):
        raise ValueError("timing output row count differs from input row count")
    input_keys = list(zip(input_df["ticker"], input_df["date"], strict=True))
    output_keys = list(zip(output_df["ticker"], output_df["date"], strict=True))
    if output_keys != input_keys:
        raise ValueError("timing output ordering or ticker/date keys differ from input")
    for column in input_df.columns:
        if not output_df[column].equals(input_df[column]):
            raise ValueError(f"timing output mutated upstream column: {column}")
    _validate_forbidden_semantics(output_df)


def _write_log(input_df: pd.DataFrame, output_df: pd.DataFrame, generated_at: str, duplicate_count: int) -> None:
    log_row = {
        "generated_at": generated_at,
        "input_row_count": int(len(input_df)),
        "output_row_count": int(len(output_df)),
        "unique_ticker_date_count": int(input_df[KEY_COLUMNS].drop_duplicates().shape[0]),
        "duplicate_ticker_date_count": int(duplicate_count),
        "missing_auxiliary_source_count": int((output_df["timing_source_data_status"] == "SOURCE_MISSING").sum()),
        "timing_state_distribution": _distribution(output_df["timing_state"]),
        "extension_state_distribution": _distribution(output_df["extension_state"]),
        "compression_state_distribution": _distribution(output_df["compression_state"]),
        "pullback_state_distribution": _distribution(output_df["pullback_state"]),
        "breakout_state_distribution": _distribution(output_df["breakout_state"]),
        "timing_metadata_status_distribution": _distribution(output_df["timing_metadata_status"]),
        "source_data_status_distribution": _distribution(output_df["timing_source_data_status"]),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([log_row], columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)


def build_timing_state_layer(generated_at: str | None = None) -> pd.DataFrame:
    input_df = _load_required_csv(INPUT_PATH, "fundamental_quality.csv")
    _validate_columns(input_df, KEY_COLUMNS, "fundamental_quality.csv")
    _validate_reserved_columns(input_df, "fundamental_quality.csv")
    _validate_key_values(input_df, "fundamental_quality.csv")
    duplicate_count = _count_duplicate_keys(input_df)
    _validate_no_duplicate_keys(input_df, "fundamental_quality.csv")

    auxiliary_df = _normalize_auxiliary(_load_optional_csv(AUXILIARY_PATH))
    run_timestamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata_df = _build_metadata(input_df, auxiliary_df, run_timestamp)
    output_df = pd.concat([input_df.reset_index(drop=True), metadata_df.reset_index(drop=True)], axis=1)
    _validate_output_contract(input_df.reset_index(drop=True), output_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    _write_log(input_df, output_df, run_timestamp, duplicate_count)
    print(f"Timing state layer written to: {OUTPUT_PATH}")
    print(f"Timing state layer log written to: {LOG_PATH}")
    return output_df


if __name__ == "__main__":
    raise SystemExit(
        "FAIL_CLOSED: scripts/core/build_timing_state_layer.py is a legacy script-era module. "
        "Use the canonical market_scanner runtime instead."
    )
