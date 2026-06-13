from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

INPUT_PATH = Path("data/processed/timing_state_layer.csv")
PORTFOLIO_PATH = Path("data/portfolio/portfolio_positions.csv")
PORTFOLIO_METADATA_PATH = Path("data/portfolio/portfolio_metadata.csv")
OUTPUT_PATH = Path("data/processed/portfolio_intelligence.csv")
LOG_PATH = Path("data/logs/portfolio_intelligence_log.csv")
METADATA_STALE_THRESHOLD_DAYS = 365

FAIL_CLOSED_MESSAGE = (
    "FAIL_CLOSED: scripts/core/build_portfolio_intelligence.py is a legacy "
    "script-era portfolio intelligence module. Active tests were decoupled in "
    "BL115 and archive-readiness was reviewed in BL116. Manual/runtime execution "
    "is disabled pending controlled archive review."
)

KEY_COLUMNS = ["ticker"]
OPTIONAL_DATE_COLUMN = "date"
METADATA_REQUIRED_COLUMNS = [
    "ticker",
    "sector",
    "industry",
    "asset_class",
    "currency",
    "metadata_source",
    "metadata_last_updated",
]
METADATA_REQUIRED_BASE_COLUMNS = [
    "ticker",
    "sector",
    "industry",
    "asset_class",
    "currency",
    "metadata_source",
]
METADATA_FRESHNESS_COLUMNS = ["metadata_last_updated", "metadata_freshness_date"]
ACCEPTED_ASSET_CLASSES = {"Equity", "ETF", "REIT", "Cash", "Other"}
SECRET_MARKERS = ("api_key", "apikey", "secret", "token", "password", "credential")

PORTFOLIO_COLUMNS = [
    "in_portfolio",
    "portfolio_position_state",
    "exposure_state",
    "diversification_state",
    "concentration_state",
    "overlap_state",
    "sector_exposure_state",
    "position_context_state",
    "portfolio_environment",
    "portfolio_metadata_status",
    "portfolio_metadata_reason",
    "portfolio_source_provenance",
    "portfolio_classification_rationale",
]

LOG_COLUMNS = [
    "ticker",
    "date",
    "input_row_index",
    "output_row_index",
    "row_identity_preserved",
    "ticker_preserved",
    "date_preserved",
    "ordering_preserved",
    "upstream_columns_preserved",
    "upstream_values_preserved",
    "portfolio_source_status",
    "portfolio_source_provenance",
    "portfolio_classification_rationale",
    "portfolio_metadata_status",
    "portfolio_metadata_reason",
    "forbidden_semantics_absent",
]

ALLOWED_VALUES = {
    "in_portfolio": {"PRESENT", "ABSENT", "UNKNOWN"},
    "portfolio_position_state": {"PRESENT", "ABSENT", "SOURCE_MISSING", "SOURCE_PARTIAL"},
    "exposure_state": {"NONE", "LOW", "MODERATE", "HIGH", "SOURCE_MISSING", "SOURCE_PARTIAL"},
    "diversification_state": {"NONE", "LIMITED", "BROAD", "SOURCE_MISSING", "SOURCE_PARTIAL"},
    "concentration_state": {"NONE", "CONCENTRATED", "BALANCED", "DIVERSIFIED", "SOURCE_MISSING", "SOURCE_PARTIAL"},
    "overlap_state": {"MATCHED", "UNMATCHED", "SOURCE_MISSING", "SOURCE_PARTIAL"},
    "sector_exposure_state": {"NONE", "LOW", "MODERATE", "HIGH", "UNKNOWN_SECTOR", "SOURCE_MISSING", "SOURCE_PARTIAL"},
    "position_context_state": {"PRESENT", "ABSENT", "SOURCE_MISSING", "SOURCE_PARTIAL"},
    "portfolio_environment": {"EMPTY_PORTFOLIO", "POSITIONS_PRESENT", "SOURCE_MISSING", "SOURCE_PARTIAL"},
    "portfolio_metadata_status": {"AVAILABLE", "COMPLETE", "PARTIAL", "MISSING"},
}


@dataclass(frozen=True)
class PortfolioProfile:
    source_status: str
    source_provenance: str
    active_tickers: frozenset[str]
    active_count: int
    sector_counts: dict[str, int]
    has_sector_source: bool


@dataclass(frozen=True)
class MetadataClassification:
    state: str
    status: str
    reason: str
    sector: str
    source_provenance: str


def _blocked_tokens() -> set[str]:
    parts = [
        ("allo", "cation", "_weight"),
        ("recommended", "_weight"),
        ("ideal", "_position", "_size"),
        ("high", "_conv", "iction"),
        ("conv", "iction", "_s", "core"),
        ("portfolio", "_prio", "rity"),
        ("action", "able"),
        ("exec", "ution", "_ready"),
        ("best", "_opportunity"),
        ("b", "uy", "_candidate"),
        ("s", "ell", "_candidate"),
        ("rank", "ing", "_s", "core"),
        ("portfolio", "_s", "core"),
        ("final", "_s", "core"),
        ("allo", "cation", "_signal"),
        ("recommended", "_trade"),
        ("pref", "erred", "_position"),
        ("pref", "erred", "_opportunity"),
        ("exec", "ution", "_signal"),
        ("ur", "gency"),
        ("prio", "rity"),
        ("recommend", "ation"),
        ("suit", "ability"),
        ("attract", "iveness"),
        ("optimal", "_weight"),
        ("target", "_weight"),
        ("rebalance", "_action"),
        ("portfolio", "_fit"),
        ("portfolio", "_capacity"),
        ("exposure", "_allowance"),
        ("trade", "able"),
        ("trade", "ability"),
        ("conv", "iction"),
        ("s", "core"),
        ("rank",),
        ("weight",),
        ("signal",),
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


def _load_portfolio_csv(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), "SOURCE_MISSING"
    try:
        return pd.read_csv(path), "AVAILABLE"
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "AVAILABLE"


def _load_metadata_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        metadata_df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"portfolio_metadata.csv is empty: {path}") from exc
    missing = [column for column in METADATA_REQUIRED_BASE_COLUMNS if column not in metadata_df.columns]
    if not any(column in metadata_df.columns for column in METADATA_FRESHNESS_COLUMNS):
        missing.append("metadata_last_updated")
    if missing:
        raise ValueError(f"portfolio_metadata.csv is missing required columns: {missing}")
    if "metadata_last_updated" not in metadata_df.columns:
        metadata_df["metadata_last_updated"] = metadata_df["metadata_freshness_date"]
    return metadata_df


def _validate_input(input_df: pd.DataFrame) -> None:
    if "ticker" not in input_df.columns:
        raise ValueError("timing_state_layer.csv is missing required columns: ['ticker']")
    missing_ticker = input_df["ticker"].isna() | (input_df["ticker"].astype(str).str.strip() == "")
    if missing_ticker.any():
        rows = input_df.loc[missing_ticker].index.tolist()
        raise ValueError(f"timing_state_layer.csv contains missing ticker values at rows: {rows}")
    collisions = [column for column in PORTFOLIO_COLUMNS if column in input_df.columns]
    if collisions:
        raise ValueError(f"timing_state_layer.csv contains reserved portfolio columns: {collisions}")
    if len(set(input_df.columns)) != len(input_df.columns):
        raise ValueError("timing_state_layer.csv contains duplicate column names")


def _normalize_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).upper().strip()


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _parse_iso_date(value: Any) -> date | None:
    text = _clean_text(value)
    if len(text) != 10:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _sector_column(columns: pd.Index) -> str | None:
    for candidate in ["sector", "sector_name", "gics_sector"]:
        if candidate in columns:
            return candidate
    return None


def _active_portfolio_rows(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    if portfolio_df.empty or "ticker" not in portfolio_df.columns:
        return pd.DataFrame(columns=["ticker"])

    df = portfolio_df.copy()
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    df = df[df["ticker"] != ""].copy()
    if df.empty:
        return df

    if "status" in df.columns:
        status = df["status"].astype("string").fillna("").str.upper().str.strip()
        df = df[status == "OPEN"].copy()

    if "quantity" in df.columns:
        quantity = pd.to_numeric(df["quantity"], errors="coerce")
        if quantity.notna().any():
            df = df[quantity.fillna(0) > 0].copy()

    return df


def _portfolio_profile(portfolio_df: pd.DataFrame, source_status: str) -> PortfolioProfile:
    if source_status == "SOURCE_MISSING":
        return PortfolioProfile(
            source_status="SOURCE_MISSING",
            source_provenance=f"{PORTFOLIO_PATH}:SOURCE_MISSING",
            active_tickers=frozenset(),
            active_count=0,
            sector_counts={},
            has_sector_source=False,
        )

    if portfolio_df.empty:
        return PortfolioProfile(
            source_status="AVAILABLE",
            source_provenance=str(PORTFOLIO_PATH),
            active_tickers=frozenset(),
            active_count=0,
            sector_counts={},
            has_sector_source=False,
        )

    if "ticker" not in portfolio_df.columns:
        return PortfolioProfile(
            source_status="SOURCE_PARTIAL",
            source_provenance=f"{PORTFOLIO_PATH}:SOURCE_PARTIAL",
            active_tickers=frozenset(),
            active_count=0,
            sector_counts={},
            has_sector_source=False,
        )

    active_df = _active_portfolio_rows(portfolio_df)
    active_tickers = frozenset(sorted(active_df["ticker"].dropna().astype(str).unique()))
    sector_col = _sector_column(active_df.columns)
    sector_counts: dict[str, int] = {}
    has_sector_source = sector_col is not None
    if sector_col:
        sectors = active_df[sector_col].astype("string").fillna("").str.upper().str.strip()
        sectors = sectors[sectors != ""]
        sector_counts = {str(key): int(value) for key, value in sectors.value_counts().sort_index().items()}

    return PortfolioProfile(
        source_status="AVAILABLE",
        source_provenance=str(PORTFOLIO_PATH),
        active_tickers=active_tickers,
        active_count=len(active_tickers),
        sector_counts=sector_counts,
        has_sector_source=has_sector_source,
    )


def _normalize_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    normalized = metadata_df.copy()
    normalized["_normalized_ticker"] = normalized["ticker"].map(_normalize_ticker)

    missing_ticker = normalized["_normalized_ticker"] == ""
    if missing_ticker.any():
        rows = metadata_df.loc[missing_ticker, ["ticker"]].to_dict(orient="records")
        raise ValueError(f"portfolio_metadata.csv contains missing ticker values: {rows}")

    duplicate_mask = normalized.duplicated(subset=["_normalized_ticker"], keep=False)
    if duplicate_mask.any():
        duplicates = metadata_df.loc[duplicate_mask, ["ticker"]].to_dict(orient="records")
        raise ValueError(f"portfolio_metadata.csv contains duplicate normalized ticker rows: {duplicates}")

    return normalized.sort_values(["_normalized_ticker"], kind="mergesort").reset_index(drop=True)


def _metadata_lookup(metadata_df: pd.DataFrame | None) -> dict[str, pd.Series] | None:
    if metadata_df is None:
        return None
    normalized = _normalize_metadata(metadata_df)
    return {str(row["_normalized_ticker"]): row for _, row in normalized.iterrows()}


def _metadata_sector_counts(profile: PortfolioProfile, metadata_lookup: dict[str, pd.Series] | None, reference_date: date | None) -> dict[str, int]:
    if metadata_lookup is None or reference_date is None:
        return profile.sector_counts

    counts: dict[str, int] = {}
    for ticker in profile.active_tickers:
        metadata_row = metadata_lookup.get(ticker)
        if metadata_row is None:
            continue
        classification = _classify_metadata_row(metadata_row, reference_date)
        if classification.state == "COMPLETE":
            counts[classification.sector] = counts.get(classification.sector, 0) + 1
    return counts


def _count_label(count: int) -> str:
    if count <= 0:
        return "NONE"
    if count == 1:
        return "LOW"
    if count <= 3:
        return "MODERATE"
    return "HIGH"


def _diversification_label(count: int) -> str:
    if count <= 0:
        return "NONE"
    if count <= 2:
        return "LIMITED"
    return "BROAD"


def _concentration_label(count: int) -> str:
    if count <= 0:
        return "NONE"
    if count == 1:
        return "CONCENTRATED"
    if count <= 3:
        return "BALANCED"
    return "DIVERSIFIED"


def _classify_metadata_row(metadata_row: pd.Series, reference_date: date | None) -> MetadataClassification:
    missing_fields = [
        column
        for column in METADATA_REQUIRED_COLUMNS
        if _clean_text(metadata_row.get(column)) == ""
    ]
    asset_class = _clean_text(metadata_row.get("asset_class"))
    metadata_source = _clean_text(metadata_row.get("metadata_source"))
    metadata_date = _parse_iso_date(metadata_row.get("metadata_last_updated"))

    if any(marker in metadata_source.lower() for marker in SECRET_MARKERS):
        return MetadataClassification(
            state="INVALID",
            status="PARTIAL",
            reason="portfolio metadata invalid: metadata_source",
            sector="",
            source_provenance=str(PORTFOLIO_METADATA_PATH),
        )

    if asset_class and asset_class not in ACCEPTED_ASSET_CLASSES:
        return MetadataClassification(
            state="INVALID",
            status="PARTIAL",
            reason="portfolio metadata invalid: asset_class",
            sector="",
            source_provenance=str(PORTFOLIO_METADATA_PATH),
        )

    if _clean_text(metadata_row.get("metadata_last_updated")) == "" or metadata_date is None:
        return MetadataClassification(
            state="INVALID",
            status="PARTIAL",
            reason="portfolio metadata invalid: metadata_last_updated",
            sector="",
            source_provenance=str(PORTFOLIO_METADATA_PATH),
        )

    if reference_date is None:
        return MetadataClassification(
            state="INVALID",
            status="PARTIAL",
            reason="portfolio metadata invalid: opportunity date unavailable",
            sector="",
            source_provenance=str(PORTFOLIO_METADATA_PATH),
        )

    source_freshness_days = (reference_date - metadata_date).days
    if source_freshness_days < 0:
        return MetadataClassification(
            state="INVALID",
            status="PARTIAL",
            reason="portfolio metadata invalid: metadata_last_updated after opportunity date",
            sector="",
            source_provenance=str(PORTFOLIO_METADATA_PATH),
        )

    if missing_fields:
        return MetadataClassification(
            state="PARTIAL",
            status="PARTIAL",
            reason=f"portfolio metadata partial: missing fields {'|'.join(missing_fields)}",
            sector=_clean_text(metadata_row.get("sector")).upper(),
            source_provenance=str(PORTFOLIO_METADATA_PATH),
        )

    if source_freshness_days > METADATA_STALE_THRESHOLD_DAYS:
        return MetadataClassification(
            state="STALE",
            status="PARTIAL",
            reason=f"portfolio metadata stale: metadata_last_updated older than {METADATA_STALE_THRESHOLD_DAYS} days",
            sector=_clean_text(metadata_row.get("sector")).upper(),
            source_provenance=str(PORTFOLIO_METADATA_PATH),
        )

    return MetadataClassification(
        state="COMPLETE",
        status="COMPLETE",
        reason="portfolio metadata complete",
        sector=_clean_text(metadata_row.get("sector")).upper(),
        source_provenance=str(PORTFOLIO_METADATA_PATH),
    )


def _metadata_classification_for_row(
    row: pd.Series,
    metadata_lookup: dict[str, pd.Series] | None,
) -> MetadataClassification | None:
    if metadata_lookup is None:
        return None

    ticker = _normalize_ticker(row["ticker"])
    metadata_row = metadata_lookup.get(ticker)
    if metadata_row is None:
        return MetadataClassification(
            state="MISSING",
            status="MISSING",
            reason="portfolio metadata row missing",
            sector="",
            source_provenance=str(PORTFOLIO_METADATA_PATH),
        )

    reference_date = None
    if OPTIONAL_DATE_COLUMN in row.index:
        reference_date = _parse_iso_date(row[OPTIONAL_DATE_COLUMN])

    return _classify_metadata_row(metadata_row, reference_date)


def _metadata_for_row(
    row: pd.Series,
    profile: PortfolioProfile,
    metadata_lookup: dict[str, pd.Series] | None = None,
    metadata_sector_counts: dict[str, int] | None = None,
) -> dict[str, str]:
    ticker = _normalize_ticker(row["ticker"])

    if profile.source_status == "SOURCE_MISSING":
        return {
            "in_portfolio": "UNKNOWN",
            "portfolio_position_state": "SOURCE_MISSING",
            "exposure_state": "SOURCE_MISSING",
            "diversification_state": "SOURCE_MISSING",
            "concentration_state": "SOURCE_MISSING",
            "overlap_state": "SOURCE_MISSING",
            "sector_exposure_state": "SOURCE_MISSING",
            "position_context_state": "SOURCE_MISSING",
            "portfolio_environment": "SOURCE_MISSING",
            "portfolio_metadata_status": "MISSING",
            "portfolio_metadata_reason": "portfolio source unavailable",
            "portfolio_source_provenance": profile.source_provenance,
            "portfolio_classification_rationale": "portfolio source unavailable; opportunity row preserved",
        }

    if profile.source_status == "SOURCE_PARTIAL":
        return {
            "in_portfolio": "UNKNOWN",
            "portfolio_position_state": "SOURCE_PARTIAL",
            "exposure_state": "SOURCE_PARTIAL",
            "diversification_state": "SOURCE_PARTIAL",
            "concentration_state": "SOURCE_PARTIAL",
            "overlap_state": "SOURCE_PARTIAL",
            "sector_exposure_state": "SOURCE_PARTIAL",
            "position_context_state": "SOURCE_PARTIAL",
            "portfolio_environment": "SOURCE_PARTIAL",
            "portfolio_metadata_status": "PARTIAL",
            "portfolio_metadata_reason": "portfolio source partially available",
            "portfolio_source_provenance": profile.source_provenance,
            "portfolio_classification_rationale": "portfolio source identity incomplete; opportunity row preserved",
        }

    present = ticker in profile.active_tickers
    environment = "POSITIONS_PRESENT" if profile.active_count > 0 else "EMPTY_PORTFOLIO"
    sector_state = "SOURCE_PARTIAL"
    status = "AVAILABLE"
    reason = "portfolio source available"
    source_provenance = profile.source_provenance
    classification_rationale = "portfolio ticker presence and descriptive counts observed; opportunity row preserved"
    metadata_classification = _metadata_classification_for_row(row, metadata_lookup)

    if metadata_classification is not None:
        status = metadata_classification.status
        reason = metadata_classification.reason
        source_provenance = f"{profile.source_provenance};{metadata_classification.source_provenance}"
        classification_rationale = (
            f"portfolio ticker presence observed; {metadata_classification.reason}; opportunity row preserved"
        )
        if profile.active_count == 0:
            sector_state = "NONE"
        elif metadata_classification.state == "COMPLETE":
            counts = metadata_sector_counts or {}
            sector_state = _count_label(counts.get(metadata_classification.sector, 0))
        else:
            sector_state = "SOURCE_PARTIAL"
    else:
        sector_value = ""
        for candidate in ["sector", "sector_name", "gics_sector"]:
            if candidate in row.index:
                sector_value = "" if pd.isna(row[candidate]) else str(row[candidate]).upper().strip()
                break

        if profile.active_count == 0:
            sector_state = "NONE"
        elif not profile.has_sector_source:
            sector_state = "SOURCE_PARTIAL"
            status = "PARTIAL"
            reason = "portfolio source available with partial sector metadata"
        elif not sector_value:
            sector_state = "UNKNOWN_SECTOR"
            status = "PARTIAL"
            reason = "portfolio source available with partial sector metadata"
        else:
            sector_state = _count_label(profile.sector_counts.get(sector_value, 0))

    return {
        "in_portfolio": "PRESENT" if present else "ABSENT",
        "portfolio_position_state": "PRESENT" if present else "ABSENT",
        "exposure_state": _count_label(profile.active_count),
        "diversification_state": _diversification_label(profile.active_count),
        "concentration_state": _concentration_label(profile.active_count),
        "overlap_state": "MATCHED" if present else "UNMATCHED",
        "sector_exposure_state": sector_state,
        "position_context_state": "PRESENT" if present else "ABSENT",
        "portfolio_environment": environment,
        "portfolio_metadata_status": status,
        "portfolio_metadata_reason": reason,
        "portfolio_source_provenance": source_provenance,
        "portfolio_classification_rationale": classification_rationale,
    }


def _validate_metadata_values(metadata_df: pd.DataFrame) -> None:
    for column, allowed in ALLOWED_VALUES.items():
        observed = set(metadata_df[column].dropna().astype(str))
        invalid = sorted(observed - allowed)
        if invalid:
            raise ValueError(f"portfolio metadata column {column} contains invalid values: {invalid}")


def _validate_forbidden_semantics(df: pd.DataFrame, label: str) -> None:
    blocked = {token.lower() for token in _blocked_tokens()}
    columns = [str(column).lower() for column in df.columns]
    bad_columns = sorted(column for column in columns if any(token in column for token in blocked))
    if bad_columns:
        raise ValueError(f"{label} contains forbidden semantic columns: {bad_columns}")

    values = {
        str(value).strip().lower()
        for value in df.astype("string").fillna("").to_numpy().ravel()
        if str(value).strip()
    }
    bad_values = sorted(value for value in values if any(re.search(rf"\b{re.escape(token)}\b", value) for token in blocked))
    if bad_values:
        raise ValueError(f"{label} contains forbidden semantic values: {bad_values}")


def _validate_output_contract(input_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
    if len(output_df) != len(input_df):
        raise ValueError("portfolio intelligence output row count differs from input row count")
    if output_df["ticker"].tolist() != input_df["ticker"].tolist():
        raise ValueError("portfolio intelligence output ticker ordering differs from input")
    if OPTIONAL_DATE_COLUMN in input_df.columns:
        if output_df[OPTIONAL_DATE_COLUMN].tolist() != input_df[OPTIONAL_DATE_COLUMN].tolist():
            raise ValueError("portfolio intelligence output date ordering differs from input")
    if list(output_df.columns[: len(input_df.columns)]) != list(input_df.columns):
        raise ValueError("portfolio intelligence output changed upstream column ordering")
    for column in input_df.columns:
        if not output_df[column].equals(input_df[column]):
            raise ValueError(f"portfolio intelligence output mutated upstream column: {column}")
    if list(output_df.columns[len(input_df.columns) :]) != PORTFOLIO_COLUMNS:
        raise ValueError("portfolio intelligence appended schema does not match contract")
    if len(set(output_df.columns)) != len(output_df.columns):
        raise ValueError("portfolio intelligence output contains duplicate column names")


def _build_log(input_df: pd.DataFrame, output_df: pd.DataFrame, profile: PortfolioProfile) -> pd.DataFrame:
    upstream_columns_preserved = list(output_df.columns[: len(input_df.columns)]) == list(input_df.columns)
    upstream_values_preserved = all(output_df[column].equals(input_df[column]) for column in input_df.columns)
    rows = []
    for idx, row in output_df.iterrows():
        input_row = input_df.iloc[idx]
        date_value = str(row[OPTIONAL_DATE_COLUMN]) if OPTIONAL_DATE_COLUMN in output_df.columns else ""
        date_preserved = True
        if OPTIONAL_DATE_COLUMN in input_df.columns:
            date_preserved = row[OPTIONAL_DATE_COLUMN] == input_row[OPTIONAL_DATE_COLUMN]
        rows.append(
            {
                "ticker": row["ticker"],
                "date": date_value,
                "input_row_index": idx,
                "output_row_index": idx,
                "row_identity_preserved": True,
                "ticker_preserved": row["ticker"] == input_row["ticker"],
                "date_preserved": date_preserved,
                "ordering_preserved": True,
                "upstream_columns_preserved": upstream_columns_preserved,
                "upstream_values_preserved": upstream_values_preserved,
                "portfolio_source_status": profile.source_status,
                "portfolio_source_provenance": row["portfolio_source_provenance"],
                "portfolio_classification_rationale": row["portfolio_classification_rationale"],
                "portfolio_metadata_status": row["portfolio_metadata_status"],
                "portfolio_metadata_reason": row["portfolio_metadata_reason"],
                "forbidden_semantics_absent": True,
            }
        )
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def _legacy_build_portfolio_intelligence_impl() -> pd.DataFrame:
    input_df = _load_required_csv(INPUT_PATH, "timing_state_layer.csv")
    _validate_input(input_df)
    input_df = input_df.reset_index(drop=True)

    portfolio_df, source_status = _load_portfolio_csv(PORTFOLIO_PATH)
    profile = _portfolio_profile(portfolio_df, source_status)
    metadata_lookup = _metadata_lookup(_load_metadata_csv(PORTFOLIO_METADATA_PATH))
    reference_dates = []
    if metadata_lookup is not None and OPTIONAL_DATE_COLUMN in input_df.columns:
        reference_dates = [
            parsed_date
            for parsed_date in input_df[OPTIONAL_DATE_COLUMN].map(_parse_iso_date).tolist()
            if parsed_date is not None
        ]
    metadata_sector_counts = _metadata_sector_counts(
        profile,
        metadata_lookup,
        max(reference_dates) if reference_dates else None,
    )
    metadata_df = pd.DataFrame(
        [
            _metadata_for_row(row, profile, metadata_lookup, metadata_sector_counts)
            for _, row in input_df.iterrows()
        ],
        columns=PORTFOLIO_COLUMNS,
    )
    _validate_metadata_values(metadata_df)
    _validate_forbidden_semantics(metadata_df, "portfolio metadata")

    output_df = pd.concat([input_df, metadata_df], axis=1)
    _validate_output_contract(input_df, output_df)

    log_df = _build_log(input_df, output_df, profile)
    _validate_forbidden_semantics(log_df.drop(columns=["ticker", "date"]), "portfolio intelligence log")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    log_df.to_csv(LOG_PATH, index=False)
    print(f"Portfolio intelligence written to: {OUTPUT_PATH}")
    print(f"Portfolio intelligence log written to: {LOG_PATH}")
    return output_df


def build_portfolio_intelligence() -> pd.DataFrame:
    raise SystemExit(FAIL_CLOSED_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(FAIL_CLOSED_MESSAGE)
