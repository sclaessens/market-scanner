from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PORTFOLIO_POSITIONS_PATH = Path("data/portfolio/portfolio_positions.csv")
WATCHLIST_ACTIVE_PATH = Path("data/watchlist/watchlist_active.csv")
SCANNER_RANKED_PATH = Path("data/processed/scanner_ranked.csv")
PORTFOLIO_METADATA_PATH = Path("data/portfolio/portfolio_metadata.csv")
FUNDAMENTALS_PATH = Path("data/raw/fundamentals.csv")

METADATA_REQUIRED_COLUMNS = [
    "ticker",
    "sector",
    "industry",
    "asset_class",
    "currency",
    "metadata_source",
    "metadata_last_updated",
]
FUNDAMENTAL_REQUIRED_COLUMNS = [
    "ticker",
    "as_of_date",
    "source_name",
    "source_last_updated",
    "report_period",
    "currency",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "gross_margin",
    "operating_margin",
    "debt_to_equity",
    "free_cash_flow_positive",
]
FUNDAMENTAL_METRIC_COLUMNS = [
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "gross_margin",
    "operating_margin",
    "debt_to_equity",
    "free_cash_flow_positive",
]
TARGET_MODES = {"portfolio", "portfolio-watchlist", "scanner", "scanner-ab", "explicit", "full-scanner"}
FUNDAMENTAL_STALE_THRESHOLD_DAYS = 120
METADATA_STALE_THRESHOLD_DAYS = 365


@dataclass(frozen=True)
class AuditPaths:
    portfolio_positions: Path = PORTFOLIO_POSITIONS_PATH
    watchlist_active: Path = WATCHLIST_ACTIVE_PATH
    scanner_ranked: Path = SCANNER_RANKED_PATH
    portfolio_metadata: Path = PORTFOLIO_METADATA_PATH
    fundamentals: Path = FUNDAMENTALS_PATH


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_ticker(value: Any) -> str:
    return _clean_text(value).upper()


def _parse_iso_date(value: Any) -> date | None:
    text = _clean_text(value)
    if len(text) != 10:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} is empty or malformed: {path}") from exc


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _target_frame(rows: list[dict[str, str]], default_date: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ticker", "date"])
    if df.empty:
        raise ValueError("target universe is empty")
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    df["date"] = df["date"].map(lambda value: _clean_text(value) or default_date)
    df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker", "date"], keep="first").reset_index(drop=True)
    if df.empty:
        raise ValueError("target universe is empty")
    return df


def _date_from_row(row: pd.Series, default_date: str) -> str:
    for column in ["date", "scan_date", "added_at", "last_action_at"]:
        if column in row.index:
            text = _clean_text(row[column])
            if len(text) >= 10:
                return text[:10]
    return default_date


def load_target_universe(
    target_mode: str,
    tickers: str | None = None,
    as_of_date: str | None = None,
    paths: AuditPaths | None = None,
) -> pd.DataFrame:
    paths = paths or AuditPaths()
    default_date = as_of_date or date.today().isoformat()
    if _parse_iso_date(default_date) is None:
        raise ValueError(f"as_of_date must be an ISO date in YYYY-MM-DD form: {default_date}")
    if target_mode not in TARGET_MODES:
        raise ValueError(f"unknown target universe mode: {target_mode}")

    if target_mode == "explicit":
        if not tickers or not tickers.strip():
            raise ValueError("explicit target mode requires --tickers")
        rows = [{"ticker": ticker, "date": default_date} for ticker in tickers.split(",")]
        return _target_frame(rows, default_date)

    if target_mode == "portfolio":
        portfolio_df = _read_csv(paths.portfolio_positions, "portfolio_positions.csv")
        _require_columns(portfolio_df, ["ticker"], "portfolio_positions.csv")
        if "status" in portfolio_df.columns:
            portfolio_df = portfolio_df[portfolio_df["status"].astype(str).str.upper().str.strip() == "OPEN"].copy()
        if "quantity" in portfolio_df.columns:
            quantity = pd.to_numeric(portfolio_df["quantity"], errors="coerce")
            if quantity.notna().any():
                portfolio_df = portfolio_df[quantity.fillna(0) > 0].copy()
        rows = [
            {"ticker": row["ticker"], "date": _date_from_row(row, default_date)}
            for _, row in portfolio_df.iterrows()
        ]
        return _target_frame(rows, default_date)

    if target_mode == "portfolio-watchlist":
        portfolio_target = load_target_universe("portfolio", as_of_date=default_date, paths=paths)
        if not paths.watchlist_active.exists():
            raise FileNotFoundError(f"watchlist_active.csv not found: {paths.watchlist_active}")
        watchlist_df = _read_csv(paths.watchlist_active, "watchlist_active.csv")
        _require_columns(watchlist_df, ["ticker"], "watchlist_active.csv")
        if "is_active" in watchlist_df.columns:
            active = watchlist_df["is_active"].astype(str).str.upper().str.strip()
            watchlist_df = watchlist_df[active.isin({"TRUE", "1", "YES", "ACTIVE"})].copy()
        rows = portfolio_target.to_dict(orient="records")
        rows.extend(
            {"ticker": row["ticker"], "date": _date_from_row(row, default_date)}
            for _, row in watchlist_df.iterrows()
        )
        return _target_frame(rows, default_date)

    scanner_df = _read_csv(paths.scanner_ranked, "scanner_ranked.csv")
    _require_columns(scanner_df, ["ticker"], "scanner_ranked.csv")
    if target_mode == "scanner-ab":
        grade_column = "grade" if "grade" in scanner_df.columns else "setup_grade"
        if grade_column not in scanner_df.columns:
            raise ValueError("scanner-ab target mode requires scanner grade column")
        scanner_df = scanner_df[scanner_df[grade_column].astype(str).str.upper().str.strip().isin({"A", "B"})].copy()
    rows = [
        {"ticker": row["ticker"], "date": _date_from_row(row, default_date)}
        for _, row in scanner_df.iterrows()
    ]
    return _target_frame(rows, default_date)


def _freshness_bucket(days: int | None, stale_threshold: int) -> str:
    if days is None:
        return "invalid"
    if days < 0:
        return "invalid"
    if days > stale_threshold:
        return "stale"
    return "fresh"


def _coverage_percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def audit_portfolio_metadata(target_df: pd.DataFrame, metadata_path: Path = PORTFOLIO_METADATA_PATH) -> dict[str, Any]:
    target_tickers = sorted(set(target_df["ticker"].map(_normalize_ticker)))
    total = len(target_tickers)
    base = {
        "total_target_tickers": total,
        "metadata_complete_count": 0,
        "metadata_partial_count": 0,
        "metadata_missing_count": 0,
        "metadata_invalid_count": 0,
        "duplicate_metadata_ticker_count": 0,
        "missing_sector_count": 0,
        "missing_industry_count": 0,
        "missing_asset_class_count": 0,
        "missing_currency_count": 0,
        "missing_metadata_source_count": 0,
        "missing_metadata_last_updated_count": 0,
        "metadata_freshness_distribution": {},
        "metadata_coverage_percentage": 0.0,
        "source_status": "unavailable",
    }
    if not metadata_path.exists():
        base["metadata_missing_count"] = total
        base["source_status"] = "missing"
        base["metadata_freshness_distribution"] = {"missing": total}
        return base

    metadata_df = _read_csv(metadata_path, "portfolio_metadata.csv")
    _require_columns(metadata_df, METADATA_REQUIRED_COLUMNS, "portfolio_metadata.csv")
    metadata_df = metadata_df.copy()
    metadata_df["_normalized_ticker"] = metadata_df["ticker"].map(_normalize_ticker)
    blank_identity = metadata_df["_normalized_ticker"] == ""
    duplicate_mask = metadata_df.duplicated(subset=["_normalized_ticker"], keep=False) & ~blank_identity
    duplicate_tickers = set(metadata_df.loc[duplicate_mask, "_normalized_ticker"])
    lookup = {
        row["_normalized_ticker"]: row
        for _, row in metadata_df[~duplicate_mask & ~blank_identity].iterrows()
    }
    base["duplicate_metadata_ticker_count"] = int(len(duplicate_tickers))
    base["source_status"] = "available"
    freshness_distribution: dict[str, int] = {}

    for ticker in target_tickers:
        row = lookup.get(ticker)
        if ticker in duplicate_tickers:
            base["metadata_invalid_count"] += 1
            freshness_distribution["invalid"] = freshness_distribution.get("invalid", 0) + 1
            continue
        if row is None:
            base["metadata_missing_count"] += 1
            freshness_distribution["missing"] = freshness_distribution.get("missing", 0) + 1
            continue

        missing_fields = [column for column in METADATA_REQUIRED_COLUMNS if _clean_text(row[column]) == ""]
        if "sector" in missing_fields:
            base["missing_sector_count"] += 1
        if "industry" in missing_fields:
            base["missing_industry_count"] += 1
        if "asset_class" in missing_fields:
            base["missing_asset_class_count"] += 1
        if "currency" in missing_fields:
            base["missing_currency_count"] += 1
        if "metadata_source" in missing_fields:
            base["missing_metadata_source_count"] += 1
        if "metadata_last_updated" in missing_fields:
            base["missing_metadata_last_updated_count"] += 1

        updated_at = _parse_iso_date(row["metadata_last_updated"])
        target_dates = [
            _parse_iso_date(value)
            for value in target_df.loc[target_df["ticker"] == ticker, "date"].tolist()
        ]
        reference_dates = [value for value in target_dates if value is not None]
        reference_date = max(reference_dates) if reference_dates else None
        freshness_days = (reference_date - updated_at).days if updated_at and reference_date else None
        bucket = _freshness_bucket(freshness_days, METADATA_STALE_THRESHOLD_DAYS)
        freshness_distribution[bucket] = freshness_distribution.get(bucket, 0) + 1

        if updated_at is None or reference_date is None or (freshness_days is not None and freshness_days < 0):
            base["metadata_invalid_count"] += 1
        elif missing_fields:
            base["metadata_partial_count"] += 1
        else:
            base["metadata_complete_count"] += 1

    base["metadata_freshness_distribution"] = freshness_distribution
    base["metadata_coverage_percentage"] = _coverage_percentage(base["metadata_complete_count"], total)
    return base


def _fundamental_lookup(fundamentals_df: pd.DataFrame) -> tuple[dict[tuple[str, str], pd.Series], set[tuple[str, str]], int]:
    fundamentals_df = fundamentals_df.copy()
    fundamentals_df["_normalized_ticker"] = fundamentals_df["ticker"].map(_normalize_ticker)
    fundamentals_df["_identity_as_of_date"] = fundamentals_df["as_of_date"].map(_clean_text)
    duplicate_mask = fundamentals_df.duplicated(subset=["_normalized_ticker", "_identity_as_of_date"], keep=False)
    duplicate_keys = {
        (row["_normalized_ticker"], row["_identity_as_of_date"])
        for _, row in fundamentals_df.loc[duplicate_mask].iterrows()
    }
    lookup = {
        (row["_normalized_ticker"], row["_identity_as_of_date"]): row
        for _, row in fundamentals_df.loc[~duplicate_mask].iterrows()
    }
    return lookup, duplicate_keys, int(len(duplicate_keys))


def _select_fundamental_row(
    ticker: str,
    opportunity_date: date | None,
    lookup: dict[tuple[str, str], pd.Series],
    duplicate_keys: set[tuple[str, str]],
) -> tuple[pd.Series | None, bool, str]:
    if opportunity_date is None:
        return None, False, "malformed opportunity date"
    candidates: list[tuple[date, pd.Series | None, bool]] = []
    for (row_ticker, as_of_text), row in lookup.items():
        if row_ticker != ticker:
            continue
        as_of_date = _parse_iso_date(as_of_text)
        if as_of_date and as_of_date <= opportunity_date:
            candidates.append((as_of_date, row, False))
    for row_ticker, as_of_text in duplicate_keys:
        if row_ticker != ticker:
            continue
        as_of_date = _parse_iso_date(as_of_text)
        if as_of_date and as_of_date <= opportunity_date:
            candidates.append((as_of_date, None, True))
    if not candidates:
        return None, False, "no source row on or before opportunity date"
    candidates.sort(key=lambda item: item[0])
    _, selected_row, is_duplicate = candidates[-1]
    if is_duplicate:
        return None, True, "duplicate ticker/date source identity"
    return selected_row, False, ""


def audit_fundamentals(target_df: pd.DataFrame, fundamentals_path: Path = FUNDAMENTALS_PATH) -> dict[str, Any]:
    total = int(len(target_df))
    base = {
        "total_target_ticker_date_rows": total,
        "fundamentals_sufficient_count": 0,
        "fundamentals_partial_count": 0,
        "fundamentals_stale_count": 0,
        "fundamentals_insufficient_count": 0,
        "fundamentals_invalid_count": 0,
        "source_row_missing_count": 0,
        "duplicate_ticker_date_identity_count": 0,
        "ticker_date_match_success_count": 0,
        "ticker_date_match_failure_count": 0,
        "missing_revenue_growth_yoy_count": 0,
        "missing_eps_growth_yoy_count": 0,
        "missing_gross_margin_count": 0,
        "missing_operating_margin_count": 0,
        "missing_debt_to_equity_count": 0,
        "missing_free_cash_flow_positive_count": 0,
        "source_freshness_distribution": {},
        "fundamentals_coverage_percentage": 0.0,
        "source_status": "unavailable",
        "diagnostics": [],
    }
    if not fundamentals_path.exists():
        base["fundamentals_insufficient_count"] = total
        base["source_row_missing_count"] = total
        base["ticker_date_match_failure_count"] = total
        base["source_status"] = "missing"
        base["source_freshness_distribution"] = {"missing": total}
        base["diagnostics"] = [
            {
                "ticker": row["ticker"],
                "date": row["date"],
                "fundamentals_source_row_matched": False,
                "match_failure_reason": "fundamentals source artifact missing",
            }
            for _, row in target_df.iterrows()
        ]
        return base

    fundamentals_df = _read_csv(fundamentals_path, "fundamentals.csv")
    _require_columns(fundamentals_df, FUNDAMENTAL_REQUIRED_COLUMNS, "fundamentals.csv")
    lookup, duplicate_keys, duplicate_count = _fundamental_lookup(fundamentals_df)
    base["duplicate_ticker_date_identity_count"] = duplicate_count
    base["source_status"] = "available"
    freshness_distribution: dict[str, int] = {}
    diagnostics: list[dict[str, Any]] = []

    for _, target_row in target_df.iterrows():
        ticker = _normalize_ticker(target_row["ticker"])
        opportunity_date = _parse_iso_date(target_row["date"])
        row, duplicate_selected, failure_reason = _select_fundamental_row(ticker, opportunity_date, lookup, duplicate_keys)
        diagnostic = {
            "ticker": ticker,
            "date": target_row["date"],
            "fundamentals_source_row_matched": row is not None,
            "match_failure_reason": failure_reason,
        }
        if duplicate_selected:
            base["fundamentals_invalid_count"] += 1
            base["ticker_date_match_failure_count"] += 1
            freshness_distribution["invalid"] = freshness_distribution.get("invalid", 0) + 1
            diagnostics.append(diagnostic)
            continue
        if row is None:
            base["fundamentals_insufficient_count"] += 1
            base["source_row_missing_count"] += 1
            base["ticker_date_match_failure_count"] += 1
            freshness_distribution["missing"] = freshness_distribution.get("missing", 0) + 1
            diagnostics.append(diagnostic)
            continue

        base["ticker_date_match_success_count"] += 1
        source_last_updated = _parse_iso_date(row["source_last_updated"])
        freshness_days = (opportunity_date - source_last_updated).days if opportunity_date and source_last_updated else None
        if freshness_days is not None and freshness_days < 0:
            base["fundamentals_invalid_count"] += 1
            freshness_distribution["invalid"] = freshness_distribution.get("invalid", 0) + 1
            diagnostic["match_failure_reason"] = "source_last_updated after opportunity date"
            diagnostics.append(diagnostic)
            continue
        if source_last_updated is None or opportunity_date is None:
            base["fundamentals_invalid_count"] += 1
            freshness_distribution["invalid"] = freshness_distribution.get("invalid", 0) + 1
            diagnostic["match_failure_reason"] = "malformed source or opportunity date"
            diagnostics.append(diagnostic)
            continue

        missing_metrics = [column for column in FUNDAMENTAL_METRIC_COLUMNS if _clean_text(row[column]) == ""]
        for column in missing_metrics:
            base[f"missing_{column}_count"] += 1
        bucket = _freshness_bucket(freshness_days, FUNDAMENTAL_STALE_THRESHOLD_DAYS)
        freshness_distribution[bucket] = freshness_distribution.get(bucket, 0) + 1
        if bucket == "stale":
            base["fundamentals_stale_count"] += 1
        elif missing_metrics:
            base["fundamentals_partial_count"] += 1
        else:
            base["fundamentals_sufficient_count"] += 1
        diagnostics.append(diagnostic)

    base["source_freshness_distribution"] = freshness_distribution
    base["fundamentals_coverage_percentage"] = _coverage_percentage(base["fundamentals_sufficient_count"], total)
    base["diagnostics"] = diagnostics
    return base


def run_coverage_audit(
    target_mode: str,
    tickers: str | None = None,
    as_of_date: str | None = None,
    paths: AuditPaths | None = None,
) -> dict[str, Any]:
    paths = paths or AuditPaths()
    target_df = load_target_universe(target_mode=target_mode, tickers=tickers, as_of_date=as_of_date, paths=paths)
    metadata = audit_portfolio_metadata(target_df, paths.portfolio_metadata)
    fundamentals = audit_fundamentals(target_df, paths.fundamentals)
    return {
        "generated_at": _now(),
        "target_mode": target_mode,
        "target_total_tickers": int(target_df["ticker"].nunique()),
        "target_total_ticker_date_rows": int(len(target_df)),
        "portfolio_metadata": metadata,
        "fundamentals": fundamentals,
        "diagnostics_only": True,
        "runtime_consumption": "none",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit source data coverage for an explicit target universe.")
    parser.add_argument("--target-mode", required=True, choices=sorted(TARGET_MODES))
    parser.add_argument("--tickers", help="Comma-separated ticker list for explicit target mode.")
    parser.add_argument("--as-of-date", default=date.today().isoformat(), help="Target opportunity date for modes without dates.")
    parser.add_argument("--portfolio-positions", type=Path, default=PORTFOLIO_POSITIONS_PATH)
    parser.add_argument("--watchlist-active", type=Path, default=WATCHLIST_ACTIVE_PATH)
    parser.add_argument("--scanner-ranked", type=Path, default=SCANNER_RANKED_PATH)
    parser.add_argument("--portfolio-metadata", type=Path, default=PORTFOLIO_METADATA_PATH)
    parser.add_argument("--fundamentals", type=Path, default=FUNDAMENTALS_PATH)
    args = parser.parse_args(argv)
    paths = AuditPaths(
        portfolio_positions=args.portfolio_positions,
        watchlist_active=args.watchlist_active,
        scanner_ranked=args.scanner_ranked,
        portfolio_metadata=args.portfolio_metadata,
        fundamentals=args.fundamentals,
    )
    try:
        result = run_coverage_audit(
            target_mode=args.target_mode,
            tickers=args.tickers,
            as_of_date=args.as_of_date,
            paths=paths,
        )
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
