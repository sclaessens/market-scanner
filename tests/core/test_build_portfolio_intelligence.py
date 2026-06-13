from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import re

import pandas as pd
import pytest


UPSTREAM_COLUMNS = [
    "ticker",
    "date",
    "quality_state",
    "quality_reason",
    "timing_state",
    "timing_reason",
]

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

EXPECTED_OUTPUT_COLUMNS = UPSTREAM_COLUMNS + PORTFOLIO_COLUMNS

EXPECTED_LOG_COLUMNS = [
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

FORBIDDEN_FIELDS = {
    "allocation_weight",
    "recommended_weight",
    "ideal_position_size",
    "high_conviction",
    "conviction_score",
    "portfolio_priority",
    "actionable",
    "execution_ready",
    "best_opportunity",
    "buy_candidate",
    "sell_candidate",
    "ranking_score",
    "portfolio_score",
    "final_score",
    "allocation_signal",
    "recommended_trade",
    "preferred_position",
    "preferred_opportunity",
    "execution_signal",
    "urgency",
    "priority",
    "recommendation",
    "suitability",
    "attractiveness",
    "optimal_weight",
    "target_weight",
    "rebalance_action",
    "portfolio_fit",
    "portfolio_capacity",
    "exposure_allowance",
    "tradeable",
    "tradeability",
    "conviction",
    "score",
    "rank",
    "weight",
    "signal",
}

FORBIDDEN_VALUES = {
    "BUY",
    "SELL",
    "REMOVE",
    "ACTIONABLE",
    "EXECUTION_READY",
    "TRADEABLE",
    "URGENT",
    "HIGH_CONVICTION",
    "RECOMMENDED",
    "PREFERRED",
}

METADATA_STALE_THRESHOLD_DAYS = 365
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
    return {token.lower() for token in FORBIDDEN_FIELDS}


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


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _parse_iso_date(value: object) -> date | None:
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


def _portfolio_profile(portfolio_df: pd.DataFrame, source_status: str, portfolio_path: Path) -> PortfolioProfile:
    if source_status == "SOURCE_MISSING":
        return PortfolioProfile("SOURCE_MISSING", f"{portfolio_path}:SOURCE_MISSING", frozenset(), 0, {}, False)
    if portfolio_df.empty:
        return PortfolioProfile("AVAILABLE", str(portfolio_path), frozenset(), 0, {}, False)
    if "ticker" not in portfolio_df.columns:
        return PortfolioProfile("SOURCE_PARTIAL", f"{portfolio_path}:SOURCE_PARTIAL", frozenset(), 0, {}, False)
    active_df = _active_portfolio_rows(portfolio_df)
    active_tickers = frozenset(sorted(active_df["ticker"].dropna().astype(str).unique()))
    sector_col = _sector_column(active_df.columns)
    sector_counts: dict[str, int] = {}
    has_sector_source = sector_col is not None
    if sector_col:
        sectors = active_df[sector_col].astype("string").fillna("").str.upper().str.strip()
        sectors = sectors[sectors != ""]
        sector_counts = {str(key): int(value) for key, value in sectors.value_counts().sort_index().items()}
    return PortfolioProfile("AVAILABLE", str(portfolio_path), active_tickers, len(active_tickers), sector_counts, has_sector_source)


def _normalize_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    normalized = metadata_df.copy()
    normalized["_normalized_ticker"] = normalized["ticker"].map(_normalize_ticker)
    if (normalized["_normalized_ticker"] == "").any():
        rows = metadata_df.loc[normalized["_normalized_ticker"] == "", ["ticker"]].to_dict(orient="records")
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


def _classify_metadata_row(metadata_row: pd.Series, reference_date: date | None, metadata_path: Path) -> MetadataClassification:
    missing_fields = [column for column in METADATA_REQUIRED_COLUMNS if _clean_text(metadata_row.get(column)) == ""]
    asset_class = _clean_text(metadata_row.get("asset_class"))
    metadata_source = _clean_text(metadata_row.get("metadata_source"))
    metadata_date = _parse_iso_date(metadata_row.get("metadata_last_updated"))
    if any(marker in metadata_source.lower() for marker in SECRET_MARKERS):
        return MetadataClassification("INVALID", "PARTIAL", "portfolio metadata invalid: metadata_source", "", str(metadata_path))
    if asset_class and asset_class not in ACCEPTED_ASSET_CLASSES:
        return MetadataClassification("INVALID", "PARTIAL", "portfolio metadata invalid: asset_class", "", str(metadata_path))
    if _clean_text(metadata_row.get("metadata_last_updated")) == "" or metadata_date is None:
        return MetadataClassification("INVALID", "PARTIAL", "portfolio metadata invalid: metadata_last_updated", "", str(metadata_path))
    if reference_date is None:
        return MetadataClassification("INVALID", "PARTIAL", "portfolio metadata invalid: opportunity date unavailable", "", str(metadata_path))
    source_freshness_days = (reference_date - metadata_date).days
    if source_freshness_days < 0:
        return MetadataClassification("INVALID", "PARTIAL", "portfolio metadata invalid: metadata_last_updated after opportunity date", "", str(metadata_path))
    if missing_fields:
        return MetadataClassification("PARTIAL", "PARTIAL", f"portfolio metadata partial: missing fields {'|'.join(missing_fields)}", _clean_text(metadata_row.get("sector")).upper(), str(metadata_path))
    if source_freshness_days > METADATA_STALE_THRESHOLD_DAYS:
        return MetadataClassification("STALE", "PARTIAL", f"portfolio metadata stale: metadata_last_updated older than {METADATA_STALE_THRESHOLD_DAYS} days", _clean_text(metadata_row.get("sector")).upper(), str(metadata_path))
    return MetadataClassification("COMPLETE", "COMPLETE", "portfolio metadata complete", _clean_text(metadata_row.get("sector")).upper(), str(metadata_path))


def _metadata_classification_for_row(row: pd.Series, metadata_lookup: dict[str, pd.Series] | None, metadata_path: Path) -> MetadataClassification | None:
    if metadata_lookup is None:
        return None
    ticker = _normalize_ticker(row["ticker"])
    metadata_row = metadata_lookup.get(ticker)
    if metadata_row is None:
        return MetadataClassification("MISSING", "MISSING", "portfolio metadata row missing", "", str(metadata_path))
    reference_date = _parse_iso_date(row["date"]) if "date" in row.index else None
    return _classify_metadata_row(metadata_row, reference_date, metadata_path)


def _metadata_sector_counts(profile: PortfolioProfile, metadata_lookup: dict[str, pd.Series] | None, reference_date: date | None, metadata_path: Path) -> dict[str, int]:
    if metadata_lookup is None or reference_date is None:
        return profile.sector_counts
    counts: dict[str, int] = {}
    for ticker in profile.active_tickers:
        metadata_row = metadata_lookup.get(ticker)
        if metadata_row is None:
            continue
        classification = _classify_metadata_row(metadata_row, reference_date, metadata_path)
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


def _metadata_for_row(row: pd.Series, profile: PortfolioProfile, metadata_lookup: dict[str, pd.Series] | None, metadata_sector_counts: dict[str, int] | None, metadata_path: Path) -> dict[str, str]:
    ticker = _normalize_ticker(row["ticker"])
    if profile.source_status == "SOURCE_MISSING":
        return {
            "in_portfolio": "UNKNOWN", "portfolio_position_state": "SOURCE_MISSING", "exposure_state": "SOURCE_MISSING",
            "diversification_state": "SOURCE_MISSING", "concentration_state": "SOURCE_MISSING", "overlap_state": "SOURCE_MISSING",
            "sector_exposure_state": "SOURCE_MISSING", "position_context_state": "SOURCE_MISSING", "portfolio_environment": "SOURCE_MISSING",
            "portfolio_metadata_status": "MISSING", "portfolio_metadata_reason": "portfolio source unavailable",
            "portfolio_source_provenance": profile.source_provenance,
            "portfolio_classification_rationale": "portfolio source unavailable; opportunity row preserved",
        }
    if profile.source_status == "SOURCE_PARTIAL":
        return {
            "in_portfolio": "UNKNOWN", "portfolio_position_state": "SOURCE_PARTIAL", "exposure_state": "SOURCE_PARTIAL",
            "diversification_state": "SOURCE_PARTIAL", "concentration_state": "SOURCE_PARTIAL", "overlap_state": "SOURCE_PARTIAL",
            "sector_exposure_state": "SOURCE_PARTIAL", "position_context_state": "SOURCE_PARTIAL", "portfolio_environment": "SOURCE_PARTIAL",
            "portfolio_metadata_status": "PARTIAL", "portfolio_metadata_reason": "portfolio source partially available",
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
    metadata_classification = _metadata_classification_for_row(row, metadata_lookup, metadata_path)

    if metadata_classification is not None:
        status = metadata_classification.status
        reason = metadata_classification.reason
        source_provenance = f"{profile.source_provenance};{metadata_classification.source_provenance}"
        classification_rationale = f"portfolio ticker presence observed; {metadata_classification.reason}; opportunity row preserved"
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
    blocked = _blocked_tokens()
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
    if "date" in input_df.columns and output_df["date"].tolist() != input_df["date"].tolist():
        raise ValueError("portfolio intelligence output date ordering differs from input")
    if list(output_df.columns[: len(input_df.columns)]) != list(input_df.columns):
        raise ValueError("portfolio intelligence output changed upstream column ordering")
    for column in input_df.columns:
        if not output_df[column].equals(input_df[column]):
            raise ValueError(f"portfolio intelligence output mutated upstream column: {column}")
    if list(output_df.columns[len(input_df.columns) :]) != PORTFOLIO_COLUMNS:
        raise ValueError("portfolio intelligence appended schema does not match contract")


def _build_log(input_df: pd.DataFrame, output_df: pd.DataFrame, profile: PortfolioProfile) -> pd.DataFrame:
    upstream_columns_preserved = list(output_df.columns[: len(input_df.columns)]) == list(input_df.columns)
    upstream_values_preserved = all(output_df[column].equals(input_df[column]) for column in input_df.columns)
    rows = []
    for idx, row in output_df.iterrows():
        input_row = input_df.iloc[idx]
        rows.append(
            {
                "ticker": row["ticker"],
                "date": str(row["date"]) if "date" in output_df.columns else "",
                "input_row_index": idx,
                "output_row_index": idx,
                "row_identity_preserved": True,
                "ticker_preserved": row["ticker"] == input_row["ticker"],
                "date_preserved": ("date" not in input_df.columns) or (row["date"] == input_row["date"]),
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
    return pd.DataFrame(rows, columns=EXPECTED_LOG_COLUMNS)


def _build_portfolio_intelligence_contract(
    input_path: Path,
    portfolio_path: Path,
    output_path: Path,
    log_path: Path,
) -> pd.DataFrame:
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    input_df = _load_required_csv(input_path, "timing_state_layer.csv").reset_index(drop=True)
    _validate_input(input_df)
    portfolio_df, source_status = _load_portfolio_csv(portfolio_path)
    profile = _portfolio_profile(portfolio_df, source_status, portfolio_path)
    metadata_lookup = _metadata_lookup(_load_metadata_csv(metadata_path))
    reference_dates = []
    if metadata_lookup is not None and "date" in input_df.columns:
        reference_dates = [parsed for parsed in input_df["date"].map(_parse_iso_date).tolist() if parsed is not None]
    metadata_sector_counts = _metadata_sector_counts(profile, metadata_lookup, max(reference_dates) if reference_dates else None, metadata_path)
    metadata_df = pd.DataFrame(
        [_metadata_for_row(row, profile, metadata_lookup, metadata_sector_counts, metadata_path) for _, row in input_df.iterrows()],
        columns=PORTFOLIO_COLUMNS,
    )
    _validate_metadata_values(metadata_df)
    _validate_forbidden_semantics(metadata_df, "portfolio metadata")
    output_df = pd.concat([input_df, metadata_df], axis=1)
    _validate_output_contract(input_df, output_df)
    log_df = _build_log(input_df, output_df, profile)
    _validate_forbidden_semantics(log_df.drop(columns=["ticker", "date"]), "portfolio intelligence log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log_df.to_csv(log_path, index=False)
    return output_df


@pytest.fixture()
def patch_paths(tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    portfolio_dir = tmp_path / "data" / "portfolio"
    logs_dir = tmp_path / "data" / "logs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "timing_state_layer.csv"
    portfolio_path = portfolio_dir / "portfolio_positions.csv"
    metadata_path = portfolio_dir / "portfolio_metadata.csv"
    output_path = processed_dir / "portfolio_intelligence.csv"
    log_path = logs_dir / "portfolio_intelligence_log.csv"

    return input_path, portfolio_path, output_path, log_path


def _timing_row(ticker: str, date: str = "2026-05-09", sector: str | None = None) -> dict:
    row = {
        "ticker": ticker,
        "date": date,
        "quality_state": "INSUFFICIENT_DATA",
        "quality_reason": "fundamental data unavailable",
        "timing_state": "UNCLASSIFIED",
        "timing_reason": "auxiliary timing source unavailable",
    }
    if sector is not None:
        row["sector"] = sector
    return row


def _write_timing(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_portfolio(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _metadata_row(ticker: str, **overrides) -> dict:
    row = {
        "ticker": ticker,
        "sector": "Technology",
        "industry": "Software",
        "asset_class": "Equity",
        "currency": "USD",
        "metadata_source": "manual",
        "metadata_last_updated": "2026-05-01",
    }
    row.update(overrides)
    return row


def _write_metadata(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_with_rows(patch_paths, rows: list[dict]) -> tuple[pd.DataFrame, Path, Path]:
    input_path, _, output_path, log_path = patch_paths
    _write_timing(input_path, rows)
    df = _build_portfolio_intelligence_contract(*patch_paths)
    return df, output_path, log_path


def test_successful_build_preserves_upstream_columns_and_appends_metadata(patch_paths):
    input_path, portfolio_path, output_path, _ = patch_paths
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    _write_timing(input_path, rows)
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 2, "status": "OPEN"}])

    df = _build_portfolio_intelligence_contract(*patch_paths)
    written_df = pd.read_csv(output_path)

    assert list(df.columns) == EXPECTED_OUTPUT_COLUMNS
    assert list(written_df.columns) == EXPECTED_OUTPUT_COLUMNS
    assert df["ticker"].tolist() == ["AAA", "BBB"]
    assert df.loc[0, "in_portfolio"] == "PRESENT"
    assert df.loc[1, "in_portfolio"] == "ABSENT"


def test_output_preserves_row_count_ticker_universe_and_ordering(patch_paths):
    rows = [_timing_row("CCC"), _timing_row("AAA"), _timing_row("BBB")]
    df, _, _ = _build_with_rows(patch_paths, rows)

    assert len(df) == len(rows)
    assert df["ticker"].tolist() == ["CCC", "AAA", "BBB"]
    assert set(df["ticker"]) == {"AAA", "BBB", "CCC"}


def test_non_mutating_enrichment_preserves_upstream_values(patch_paths):
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    df, _, _ = _build_with_rows(patch_paths, rows)
    upstream_df = pd.DataFrame(rows)

    for column in UPSTREAM_COLUMNS:
        assert df[column].astype("string").fillna("").tolist() == upstream_df[column].astype("string").fillna("").tolist()


def test_missing_portfolio_source_preserves_rows_with_neutral_metadata(patch_paths):
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    df, _, _ = _build_with_rows(patch_paths, rows)

    assert len(df) == 2
    assert set(df["in_portfolio"]) == {"UNKNOWN"}
    assert set(df["portfolio_metadata_status"]) == {"MISSING"}
    assert set(df["portfolio_environment"]) == {"SOURCE_MISSING"}


def test_empty_portfolio_source_preserves_rows_with_empty_environment(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    _write_timing(input_path, rows)
    pd.DataFrame(columns=["ticker", "quantity", "status"]).to_csv(portfolio_path, index=False)

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df["ticker"].tolist() == ["AAA", "BBB"]
    assert set(df["in_portfolio"]) == {"ABSENT"}
    assert set(df["portfolio_environment"]) == {"EMPTY_PORTFOLIO"}
    assert set(df["exposure_state"]) == {"NONE"}


def test_closed_portfolio_rows_do_not_create_presence(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    _write_timing(input_path, rows)
    _write_portfolio(
        portfolio_path,
        [
            {"ticker": "AAA", "quantity": 0, "status": "CLOSED"},
            {"ticker": "BBB", "quantity": 0, "status": "CLOSED"},
        ],
    )

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df["in_portfolio"].tolist() == ["ABSENT", "ABSENT"]
    assert set(df["portfolio_environment"]) == {"EMPTY_PORTFOLIO"}


def test_partial_portfolio_source_preserves_rows_with_partial_metadata(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    _write_timing(input_path, [_timing_row("AAA")])
    pd.DataFrame([{"quantity": 1, "status": "OPEN"}]).to_csv(portfolio_path, index=False)

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "in_portfolio"] == "UNKNOWN"
    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_environment"] == "SOURCE_PARTIAL"


def test_duplicate_portfolio_source_tickers_are_grouped_without_reordering(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    rows = [_timing_row("BBB"), _timing_row("AAA"), _timing_row("CCC")]
    _write_timing(input_path, rows)
    _write_portfolio(
        portfolio_path,
        [
            {"ticker": "AAA", "quantity": 1, "status": "OPEN"},
            {"ticker": "AAA", "quantity": 2, "status": "OPEN"},
            {"ticker": "CCC", "quantity": 1, "status": "OPEN"},
        ],
    )

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df["ticker"].tolist() == ["BBB", "AAA", "CCC"]
    assert df["in_portfolio"].tolist() == ["ABSENT", "PRESENT", "PRESENT"]


def test_sector_metadata_is_descriptive_when_sources_exist(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    rows = [
        _timing_row("AAA", sector="Technology"),
        _timing_row("BBB", sector="Healthcare"),
    ]
    _write_timing(input_path, rows)
    _write_portfolio(
        portfolio_path,
        [
            {"ticker": "AAA", "quantity": 1, "status": "OPEN", "sector": "Technology"},
            {"ticker": "CCC", "quantity": 1, "status": "OPEN", "sector": "Technology"},
        ],
    )

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "sector_exposure_state"] == "MODERATE"
    assert df.loc[1, "sector_exposure_state"] == "NONE"


def test_missing_portfolio_metadata_artifact_preserves_existing_partial_behavior(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    _write_timing(input_path, [_timing_row("AAA")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio source available with partial sector metadata"


def test_complete_portfolio_metadata_can_produce_complete_status(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row(" aaa ")])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "COMPLETE"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata complete"
    assert df.loc[0, "sector_exposure_state"] == "LOW"
    assert "portfolio_metadata.csv" in df.loc[0, "portfolio_source_provenance"]


def test_reit_asset_class_metadata_can_produce_complete_status(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("REIT1", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "REIT1", "quantity": 1, "status": "OPEN"}])
    _write_metadata(
        metadata_path,
        [
            _metadata_row(
                "REIT1",
                sector="Real Estate",
                industry="REIT - Industrial",
                asset_class="REIT",
            )
        ],
    )

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "COMPLETE"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata complete"
    assert "portfolio metadata complete" in df.loc[0, "portfolio_classification_rationale"]
    assert df.loc[0, "sector_exposure_state"] == "LOW"


def test_metadata_only_tickers_do_not_create_output_rows(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(
        metadata_path,
        [
            _metadata_row("AAA"),
            _metadata_row("RAWONLY", sector="Healthcare"),
        ],
    )

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert len(df) == 1
    assert df["ticker"].tolist() == ["AAA"]


def test_missing_metadata_row_keeps_metadata_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("BBB")])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "MISSING"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata row missing"
    assert df.loc[0, "sector_exposure_state"] == "SOURCE_PARTIAL"


def test_missing_required_metadata_values_produce_partial_status(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", industry="")])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata partial: missing fields industry"


def test_stale_portfolio_metadata_remains_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", metadata_last_updated="2025-01-01")])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata stale: metadata_last_updated older than 365 days"


def test_empty_sector_metadata_value_is_partial_and_deterministic(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", sector="")])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata partial: missing fields sector"


def test_invalid_asset_class_metadata_value_is_invalid_and_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", asset_class="Crypto")])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata invalid: asset_class"


def test_invalid_metadata_date_is_invalid_and_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", metadata_last_updated="2026/05/01")])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata invalid: metadata_last_updated"


def test_metadata_freshness_date_alias_is_backward_compatible(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    metadata = _metadata_row("AAA")
    metadata["metadata_freshness_date"] = metadata.pop("metadata_last_updated")
    _write_metadata(metadata_path, [metadata])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "COMPLETE"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata complete"


def test_future_metadata_date_is_invalid_and_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", metadata_last_updated="2026-05-10")])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata invalid: metadata_last_updated after opportunity date"


def test_metadata_source_secret_marker_is_invalid_and_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", metadata_source="token:local")])

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata invalid: metadata_source"


def test_duplicate_portfolio_metadata_rows_fail_fast_before_output_generation(patch_paths):
    input_path, portfolio_path, output_path, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA"), _metadata_row(" aaa ")])

    with pytest.raises(ValueError, match="duplicate normalized ticker rows"):
        _build_portfolio_intelligence_contract(*patch_paths)

    assert not output_path.exists()


def test_missing_required_metadata_schema_fields_fail_fast(patch_paths):
    input_path, portfolio_path, output_path, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    metadata = _metadata_row("AAA")
    metadata.pop("asset_class")
    _write_metadata(metadata_path, [metadata])

    with pytest.raises(ValueError, match="missing required columns"):
        _build_portfolio_intelligence_contract(*patch_paths)

    assert not output_path.exists()


def test_complete_metadata_preserves_upstream_row_count_identity_and_order(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    rows = [
        _timing_row("CCC", date="2026-05-09"),
        _timing_row("AAA", date="2026-05-09"),
        _timing_row("BBB", date="2026-05-09"),
    ]
    _write_timing(input_path, rows)
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(
        metadata_path,
        [
            _metadata_row("AAA"),
            _metadata_row("BBB", sector="Healthcare"),
            _metadata_row("CCC", sector="Industrials"),
        ],
    )

    df = _build_portfolio_intelligence_contract(*patch_paths)

    assert len(df) == len(rows)
    assert list(zip(df["ticker"], df["date"], strict=True)) == [
        (row["ticker"], row["date"]) for row in rows
    ]
    assert set(df["portfolio_metadata_status"]) == {"COMPLETE"}


def test_missing_input_file_fails_fast(patch_paths):
    with pytest.raises(FileNotFoundError, match="timing_state_layer.csv not found"):
        _build_portfolio_intelligence_contract(*patch_paths)


def test_missing_ticker_column_fails_fast(patch_paths):
    input_path, _, _, _ = patch_paths
    pd.DataFrame([{"date": "2026-05-09"}]).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        _build_portfolio_intelligence_contract(*patch_paths)


def test_reserved_portfolio_columns_in_input_fail_fast(patch_paths):
    input_path, _, _, _ = patch_paths
    row = _timing_row("AAA")
    row["in_portfolio"] = "ABSENT"
    _write_timing(input_path, [row])

    with pytest.raises(ValueError, match="reserved portfolio columns"):
        _build_portfolio_intelligence_contract(*patch_paths)


def test_forbidden_columns_are_absent_from_output_and_log(patch_paths):
    df, _, log_path = _build_with_rows(patch_paths, [_timing_row("AAA")])
    log_df = pd.read_csv(log_path)
    normalized = {column.lower() for column in df.columns} | {column.lower() for column in log_df.columns}

    assert normalized.isdisjoint(FORBIDDEN_FIELDS)


def test_forbidden_semantic_values_are_absent_from_generated_metadata_and_log(patch_paths):
    df, _, log_path = _build_with_rows(patch_paths, [_timing_row("AAA")])
    log_df = pd.read_csv(log_path)
    generated = pd.concat([df[PORTFOLIO_COLUMNS], log_df.drop(columns=["ticker", "date"])], axis=1)
    values = {
        str(value).upper()
        for value in generated.astype("string").fillna("").to_numpy().ravel()
        if str(value).strip()
    }

    assert values.isdisjoint(FORBIDDEN_VALUES)


def test_deterministic_output_repeated_runs_match(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    rows = [_timing_row("CCC"), _timing_row("AAA"), _timing_row("BBB")]
    _write_timing(input_path, rows)
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 2, "status": "OPEN"}])

    first_df = _build_portfolio_intelligence_contract(*patch_paths)
    second_df = _build_portfolio_intelligence_contract(*patch_paths)

    pd.testing.assert_frame_equal(first_df, second_df)


def test_log_creation_and_schema(patch_paths):
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    _, _, log_path = _build_with_rows(patch_paths, rows)
    log_df = pd.read_csv(log_path)

    assert list(log_df.columns) == EXPECTED_LOG_COLUMNS
    assert len(log_df) == 2
    assert log_df["input_row_index"].tolist() == [0, 1]
    assert log_df["output_row_index"].tolist() == [0, 1]
    assert set(log_df["row_identity_preserved"]) == {True}
    assert set(log_df["upstream_values_preserved"]) == {True}


def test_no_decision_engine_dependency_or_leakage():
    source = inspect.getsource(_build_portfolio_intelligence_contract)

    assert "decision_engine" not in source
    assert "final_action" not in source
    assert "allocation_priority" not in source


def test_no_reporting_or_telegram_dependency_or_leakage():
    source = inspect.getsource(_build_portfolio_intelligence_contract)

    assert "scripts.reporting" not in source
    assert "build_reporting_layer" not in source
    assert "build_telegram_summary" not in source
    assert "send_telegram" not in source


def test_only_approved_output_files_are_written(patch_paths, tmp_path: Path):
    rows = [_timing_row("AAA")]
    _, output_path, log_path = _build_with_rows(patch_paths, rows)
    files = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.csv")}

    assert output_path.exists()
    assert log_path.exists()
    assert files == {
        "data/processed/timing_state_layer.csv",
        "data/processed/portfolio_intelligence.csv",
        "data/logs/portfolio_intelligence_log.csv",
    }
