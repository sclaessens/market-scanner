from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.data.complete_local_market_dataset import DEFAULT_CANONICAL_CONFIG
from market_engine.data.local_market_data_universe import DEFAULT_PRICE_HISTORY_ROOT, build_universe_snapshot
from market_engine.run.broad_non_price_evidence_advice_readiness import (
    DEFAULT_COMPACT_EVIDENCE_ROOT,
    DEFAULT_MARKET_CONTEXT_PATH,
    DEFAULT_OUTPUT_ROOT as DEFAULT_RUN31_OUTPUT_ROOT,
    run_broad_non_price_evidence_advice_readiness,
)
from market_engine.run.local_portfolio_context_fixture import DEFAULT_PORTFOLIO_CONTEXT_PATH
from market_engine.source_context.sec_companyfacts_context import (
    SecCompanyFactsContextBuildError,
    build_sec_companyfacts_source_context_from_snapshot_path,
)


SPRINT_ID = "ME-DATA06"
SCHEMA_VERSION = "market-engine-data06-fundamental-evidence-coverage-v1"
DEFAULT_OUTPUT_ROOT = Path("artifacts/market_engine/fundamental_evidence_coverage_runs")
DEFAULT_EXISTING_FUNDAMENTAL_PATH = Path("data/processed/fundamental_quality.csv")
DEFAULT_RAW_FUNDAMENTALS_PATH = Path("data/raw/fundamentals.csv")
DEFAULT_INTAKE_FUNDAMENTALS_PATH = Path("data/intake/os5_scanner_ab_fundamentals_intake_pilot.csv")
DEFAULT_SEC_COMPANYFACTS_ROOT = Path("data/market_engine/source_snapshots/sec_companyfacts")
DEFAULT_COMPANY_PROFILE_ROOT = Path("artifacts/market_engine")
DEFAULT_TECHNICAL_SCREENING_ARTIFACT = Path("artifacts/market_engine/universe_analysis_runs/me-run30-full-canonical-universe-analysis-ranking-20260714T143209Z")
DEFAULT_BASELINE_RUN_EVIDENCE = Path(
    "artifacts/market_engine/run_evidence/"
    "me-run31-broad-non-price-evidence-full-advice-readiness-20260715T154103Z"
)
DEFAULT_AS_OF_DATE = "2026-07-10"
FRESHNESS_MAX_AGE_DAYS = 120
BASELINE_MANIFEST_ARTIFACT_TYPE = "market-engine-run31-broad-non-price-evidence-advice-readiness"
BASELINE_MANIFEST_SCHEMA_VERSION = "market-engine-run31-broad-non-price-evidence-advice-readiness-v1"
BASELINE_INDEX_ARTIFACT_TYPE = "market-engine-run31-evidence-coverage-index"
BASELINE_INDEX_SCHEMA_VERSION = "market-engine-run31-evidence-coverage-index-v1"
BASELINE_COMPACT_SCHEMA_VERSION = "market-engine-run31-compact-run-evidence-v1"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_COMPLETED_WITH_REGRESSIONS = "completed_with_regressions"
RUN_STATUS_FAILED_VALIDATION = "failed_validation"

FUNDAMENTAL_QUALITY_COLUMNS = (
    "ticker",
    "date",
    "quality_state",
    "quality_reason",
    "profitability_profile",
    "balance_sheet_profile",
    "earnings_quality_profile",
    "capital_efficiency_profile",
    "cashflow_profile",
    "stability_profile",
    "quality_metadata_status",
    "source_data_status",
    "source_timestamp",
    "source_name",
    "source_last_updated",
    "source_freshness_days",
    "missing_required_fields",
    "partial_data_reason",
    "stale_data_reason",
    "invalid_data_reason",
    "generated_at",
)
MVP_METRIC_FIELDS = (
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "gross_margin",
    "operating_margin",
    "debt_to_equity",
)
SEC_CONTEXT_FIELDS = (
    "revenue",
    "net_income",
    "operating_cash_flow",
    "capital_expenditures",
)
SOURCE_PRIORITY = {
    "manual_mvp_fundamentals": 100,
    "existing_processed_fundamental_quality": 80,
    "sec_companyfacts_source_context": 60,
}


class FundamentalEvidenceCoverageError(ValueError):
    pass


class FundamentalEvidenceCoverageValidationError(FundamentalEvidenceCoverageError):
    run_status = RUN_STATUS_FAILED_VALIDATION


def run_fundamental_evidence_coverage(
    *,
    run_id: str,
    canonical_universe: str | Path = DEFAULT_CANONICAL_CONFIG,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    existing_fundamental_path: str | Path = DEFAULT_EXISTING_FUNDAMENTAL_PATH,
    raw_fundamentals_path: str | Path = DEFAULT_RAW_FUNDAMENTALS_PATH,
    intake_fundamentals_path: str | Path = DEFAULT_INTAKE_FUNDAMENTALS_PATH,
    sec_companyfacts_root: str | Path = DEFAULT_SEC_COMPANYFACTS_ROOT,
    company_profile_root: str | Path = DEFAULT_COMPANY_PROFILE_ROOT,
    as_of_date: str = DEFAULT_AS_OF_DATE,
    baseline_run_evidence: str | Path | None = None,
    technical_screening_artifact: str | Path = DEFAULT_TECHNICAL_SCREENING_ARTIFACT,
    market_context_path: str | Path = DEFAULT_MARKET_CONTEXT_PATH,
    portfolio_context_path: str | Path | None = DEFAULT_PORTFOLIO_CONTEXT_PATH,
    run31_output_root: str | Path = DEFAULT_RUN31_OUTPUT_ROOT,
    run31_compact_evidence_root: str | Path = DEFAULT_COMPACT_EVIDENCE_ROOT,
    run31_run_id: str | None = None,
    allow_overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    started_at = _utc_now()
    parsed_as_of = _parse_date(as_of_date, field_name="as_of_date")
    output_dir = Path(output_root) / run_id
    if output_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"ME-DATA06 output directory already exists: {output_dir}")

    universe = build_universe_snapshot(canonical_universe, price_history_root=price_history_root)
    instruments = sorted(universe["instruments"], key=lambda row: str(row["instrument_id"]))
    canonical_symbols = [str(row["symbol"]).upper() for row in instruments]
    canonical_set = set(canonical_symbols)
    baseline_input = Path(baseline_run_evidence or DEFAULT_BASELINE_RUN_EVIDENCE)
    baseline = _load_before_baseline(
        baseline_input,
        canonical_symbols=canonical_symbols,
        canonical_universe_version=universe.get("universe_version"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_rows = _load_existing_fundamental_quality(existing_fundamental_path)
    manual_sources = _load_manual_fundamentals(raw_fundamentals_path, parsed_as_of)
    sec_sources = _load_sec_companyfacts_sources(sec_companyfacts_root, parsed_as_of)
    intake_sources = _load_intake_inventory(intake_fundamentals_path)
    company_profile_inventory = _load_company_profile_inventory(company_profile_root, canonical_set)

    candidates_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for source_row in manual_sources["candidates"] + _existing_rows_to_candidates(existing_rows, parsed_as_of) + sec_sources["candidates"]:
        ticker = str(source_row["ticker"]).upper()
        if ticker in canonical_set:
            candidates_by_ticker.setdefault(ticker, []).append(source_row)

    per_ticker = [
        _classify_ticker(
            instrument,
            candidates_by_ticker.get(str(instrument["symbol"]).upper(), []),
            parsed_as_of=parsed_as_of,
            generated_at=started_at,
        )
        for instrument in instruments
    ]
    normalized_rows = [row["fundamental_quality_row"] for row in per_ticker]
    normalized_path = output_dir / "normalized_fundamental_quality.csv"
    _write_csv(normalized_path, normalized_rows, FUNDAMENTAL_QUALITY_COLUMNS)

    before = dict(baseline["counts"])
    summary = _coverage_summary(per_ticker, before=before, universe=universe)
    inventory = _evidence_source_inventory(
        canonical_set=canonical_set,
        existing_path=existing_fundamental_path,
        existing_rows=existing_rows,
        manual=manual_sources,
        sec=sec_sources,
        intake=intake_sources,
        company_profiles=company_profile_inventory,
        as_of=parsed_as_of,
    )
    run31_id = run31_run_id or f"{run_id}-me-run31-rerun"
    run31_artifacts, run31_output_dir = run_broad_non_price_evidence_advice_readiness(
        run_id=run31_id,
        canonical_universe=canonical_universe,
        price_history_root=price_history_root,
        technical_screening_artifact=technical_screening_artifact,
        output_root=run31_output_root,
        compact_evidence_root=run31_compact_evidence_root,
        fundamental_evidence_path=normalized_path,
        market_context_path=market_context_path,
        portfolio_context_path=portfolio_context_path,
        freshness_reference_date=as_of_date,
        allow_overwrite=allow_overwrite,
    )
    downstream = _downstream_summary(run31_artifacts, run31_output_dir)
    after = _after_summary(per_ticker, downstream)
    comparison, transitions = _before_after_comparison(baseline, after, per_ticker, run31_artifacts)
    run_status = comparison["run_status"]
    summary["after"] = after
    summary["comparison"] = comparison
    summary["downstream_me_run31"] = downstream

    artifacts = {
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "market-engine-fundamental-evidence-coverage-run",
            "sprint_id": SPRINT_ID,
            "run_id": run_id,
            "run_status": run_status,
            "command": "python -m market_engine.data.fundamental_evidence_coverage",
            "started_at": started_at,
            "completed_at": _utc_now(),
            "git_commit": _git_commit(),
            "input_paths": {
                "canonical_universe": Path(canonical_universe).as_posix(),
                "price_history_root": Path(price_history_root).as_posix(),
                "existing_fundamental_path": Path(existing_fundamental_path).as_posix(),
                "raw_fundamentals_path": Path(raw_fundamentals_path).as_posix(),
                "intake_fundamentals_path": Path(intake_fundamentals_path).as_posix(),
                "sec_companyfacts_root": Path(sec_companyfacts_root).as_posix(),
                "company_profile_root": Path(company_profile_root).as_posix(),
                "baseline_run_evidence": baseline_input.as_posix(),
                "baseline_artifact_path": baseline["artifact_path"],
                "technical_screening_artifact": Path(technical_screening_artifact).as_posix(),
                "market_context_path": Path(market_context_path).as_posix(),
                "portfolio_context_path": Path(portfolio_context_path).as_posix() if portfolio_context_path else None,
            },
            "input_checksums": _input_checksums(
                canonical_universe,
                existing_fundamental_path,
                raw_fundamentals_path,
                intake_fundamentals_path,
                market_context_path,
                portfolio_context_path,
                baseline["artifact_path"],
            ),
            "baseline_run_id": baseline["run_id"],
            "baseline_artifact_path": baseline["artifact_path"],
            "baseline_artifact_type": baseline["artifact_type"],
            "baseline_schema_version": baseline["schema_version"],
            "baseline_checksum": baseline["checksum"],
            "baseline_canonical_universe_version": baseline["canonical_universe_version"],
            "baseline_canonical_universe_size": baseline["canonical_universe_size"],
            "canonical_universe": {
                "universe_version": universe.get("universe_version"),
                "snapshot_date": universe.get("snapshot_date"),
                "total_instruments": len(instruments),
            },
            "as_of_date": as_of_date,
            "schema_versions": {
                "me_data06": SCHEMA_VERSION,
                "normalized_fundamental_quality": "market-engine-fundamental-quality-csv-v1",
            },
            "attempted_tickers": len(instruments),
            "coverage_counts": summary["overall_counts"],
            "validation_result": "passed",
            "downstream_rerun_result": downstream,
            "blocker_summary": summary["blocker_summary"],
            "side_effects": {
                "network_access_performed": False,
                "provider_api_call_performed": False,
                "model_invocation_performed": False,
                "broker_order_execution_performed": False,
                "allocation_performed": False,
                "portfolio_watchlist_mutation_performed": False,
                "telegram_delivery_performed": False,
                "decision_engine_authority_changed": False,
                "recommendation_rules_changed": False,
            },
            "outputs": {
                "normalized_fundamental_quality": normalized_path.as_posix(),
                "downstream_me_run31_output_dir": run31_output_dir.as_posix(),
                "downstream_me_run31_compact_evidence_dir": run31_artifacts.get("compact_evidence_dir"),
                "per_ticker_fundamental_transitions": (output_dir / "per_ticker_fundamental_transitions.json").as_posix(),
            },
        },
        "evidence_source_inventory": inventory,
        "fundamental_coverage_summary": summary,
        "per_ticker_fundamental_status": {
            "schema_version": "market-engine-data06-per-ticker-fundamental-status-v1",
            "run_id": run_id,
            "tickers": [_strip_quality_row(row) for row in per_ticker],
        },
        "missing_fundamental_evidence": {
            "schema_version": "market-engine-data06-missing-fundamental-evidence-v1",
            "run_id": run_id,
            "tickers": sorted(
                row["ticker"] for row in per_ticker if row["overall_fundamental_status"] == "missing"
            ),
        },
        "partial_fundamental_evidence": {
            "schema_version": "market-engine-data06-partial-fundamental-evidence-v1",
            "run_id": run_id,
            "tickers": sorted(
                row["ticker"] for row in per_ticker if row["overall_fundamental_status"] == "partial"
            ),
        },
        "invalid_or_stale_evidence": {
            "schema_version": "market-engine-data06-invalid-or-stale-evidence-v1",
            "run_id": run_id,
            "tickers": sorted(
                row["ticker"]
                for row in per_ticker
                if row["overall_fundamental_status"] in {"invalid", "stale", "conflicting"}
            ),
        },
        "before_after_comparison": comparison,
        "per_ticker_fundamental_transitions": transitions,
    }
    _write_artifacts(output_dir, artifacts)
    (output_dir / "coverage_report.md").write_text(_render_report(artifacts), encoding="utf-8")
    return artifacts, output_dir


def _load_existing_fundamental_quality(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_manual_fundamentals(path: str | Path, as_of: date) -> dict[str, Any]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {"path": csv_path.as_posix(), "status": "missing", "candidates": [], "rows": []}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"ticker", "as_of_date", "source_name", "source_reference", "source_freshness_date", *MVP_METRIC_FIELDS}
        missing = sorted(required - fieldnames)
        rows = [dict(row) for row in reader]
    if missing:
        return {"path": csv_path.as_posix(), "status": "unsupported_schema", "missing_columns": missing, "candidates": [], "rows": rows}
    candidates = []
    for row in rows:
        candidates.append(_manual_row_to_candidate(row, csv_path, as_of))
    return {"path": csv_path.as_posix(), "status": "consumeable", "candidates": candidates, "rows": rows}


def _manual_row_to_candidate(row: Mapping[str, str], path: Path, as_of: date) -> dict[str, Any]:
    ticker = str(row.get("ticker") or "").upper()
    source_date_text = row.get("source_freshness_date") or row.get("as_of_date")
    invalid_reasons: list[str] = []
    source_date: date | None = None
    try:
        source_date = _parse_date(source_date_text, field_name="source_freshness_date")
        if source_date > as_of:
            invalid_reasons.append("future_dated_evidence")
    except FundamentalEvidenceCoverageError:
        invalid_reasons.append("invalid_source_date")
    metrics: dict[str, Any] = {}
    missing_metrics = []
    invalid_metrics = []
    for field in MVP_METRIC_FIELDS:
        raw = row.get(field)
        if raw is None or str(raw).strip() == "":
            metrics[field] = None
            missing_metrics.append(field)
            continue
        try:
            metrics[field] = float(raw)
        except ValueError:
            metrics[field] = None
            invalid_metrics.append(field)
    if invalid_metrics:
        invalid_reasons.append("invalid_numeric_metric")
    if invalid_reasons:
        status = "invalid"
    elif not missing_metrics:
        status = "complete"
    elif len(missing_metrics) < len(MVP_METRIC_FIELDS):
        status = "partial"
    else:
        status = "partial"
    if source_date and _is_stale(source_date, as_of=as_of):
        status = "stale"
    return {
        "ticker": ticker,
        "source_family": "manual_mvp_fundamentals",
        "source_name": row.get("source_name") or "local_manual_fundamentals",
        "source_path": path.as_posix(),
        "source_reference": row.get("source_reference") or "",
        "source_date": source_date.isoformat() if source_date else "",
        "as_of_date": row.get("as_of_date") or "",
        "currency": row.get("currency") or "",
        "priority": SOURCE_PRIORITY["manual_mvp_fundamentals"],
        "coverage_status": status,
        "metrics": metrics,
        "units": {field: "ratio" for field in MVP_METRIC_FIELDS},
        "missing_metrics": missing_metrics,
        "invalid_metrics": invalid_metrics,
        "blockers": invalid_reasons,
        "notes": row.get("fundamental_notes") or "",
    }


def _existing_rows_to_candidates(rows: Sequence[Mapping[str, str]], as_of: date) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        source_date_text = row.get("source_last_updated") or row.get("source_timestamp") or row.get("date")
        try:
            source_date = _parse_date(source_date_text, field_name="source_date")
        except FundamentalEvidenceCoverageError:
            continue
        state = str(row.get("quality_state") or "").upper()
        if state == "SUFFICIENT_DATA":
            status = "complete"
        elif state == "PARTIAL_DATA" or str(row.get("source_data_status") or "") == "partial_data":
            status = "partial"
        else:
            status = "missing"
        if status != "missing" and _is_stale(source_date, as_of=as_of):
            status = "stale"
        candidates.append(
            {
                "ticker": ticker,
                "source_family": "existing_processed_fundamental_quality",
                "source_name": row.get("source_name") or "existing_processed_fundamental_quality",
                "source_path": DEFAULT_EXISTING_FUNDAMENTAL_PATH.as_posix(),
                "source_reference": "data/processed/fundamental_quality.csv",
                "source_date": source_date.isoformat(),
                "as_of_date": row.get("date") or source_date.isoformat(),
                "currency": "",
                "priority": SOURCE_PRIORITY["existing_processed_fundamental_quality"],
                "coverage_status": status,
                "metrics": {},
                "units": {},
                "missing_metrics": [item for item in str(row.get("missing_required_fields") or "").split("|") if item],
                "invalid_metrics": [],
                "blockers": [],
                "existing_quality_row": dict(row),
            }
        )
    return candidates


def _load_sec_companyfacts_sources(root: str | Path, as_of: date) -> dict[str, Any]:
    source_root = Path(root)
    candidates = []
    errors = []
    for path in sorted(source_root.glob("*/raw/*.json")):
        try:
            context = build_sec_companyfacts_source_context_from_snapshot_path(path)
        except SecCompanyFactsContextBuildError as exc:
            errors.append({"path": path.as_posix(), "error": str(exc)})
            continue
        source_date = _parse_date(context.source_refresh_fetched_at[:10], field_name="source_refresh_fetched_at")
        present = [
            field
            for field in SEC_CONTEXT_FIELDS
            if context.field_states[field].value == "PRESENT"
        ]
        missing = [field for field in SEC_CONTEXT_FIELDS if field not in present]
        status = "partial" if present else "missing"
        if present and not missing:
            status = "partial"
        if _is_stale(source_date, as_of=as_of):
            status = "stale"
        candidates.append(
            {
                "ticker": context.ticker,
                "source_family": "sec_companyfacts_source_context",
                "source_name": context.provider_name,
                "source_path": path.as_posix(),
                "source_reference": context.source_refresh_snapshot_id,
                "source_date": source_date.isoformat(),
                "as_of_date": source_date.isoformat(),
                "currency": "",
                "priority": SOURCE_PRIORITY["sec_companyfacts_source_context"],
                "coverage_status": status,
                "metrics": {field: context.canonical_fields[field] for field in SEC_CONTEXT_FIELDS},
                "units": {
                    field: (context.fields[field].unit if context.fields[field].state.value == "PRESENT" else None)
                    for field in SEC_CONTEXT_FIELDS
                },
                "missing_metrics": list(MVP_METRIC_FIELDS),
                "missing_source_context_fields": missing,
                "invalid_metrics": [],
                "blockers": ["sec_companyfacts_not_sufficient_for_mvp_quality_contract"],
                "context_state": context.source_context_state.value,
                "field_states": {field: context.field_states[field].value for field in SEC_CONTEXT_FIELDS},
            }
        )
    return {"path": source_root.as_posix(), "status": "consumeable" if source_root.exists() else "missing", "candidates": candidates, "errors": errors}


def _load_intake_inventory(path: str | Path) -> dict[str, Any]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {"path": csv_path.as_posix(), "status": "missing", "rows": []}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return {
        "path": csv_path.as_posix(),
        "status": "not_consumeable",
        "reason": "intake placeholder rows require source acquisition and are not runtime evidence",
        "rows": rows,
    }


def _load_company_profile_inventory(root: str | Path, canonical_set: set[str]) -> dict[str, Any]:
    paths = sorted(Path(root).glob("**/company_profile/company_profile.json"))
    tickers = []
    for path in paths:
        try:
            payload = _read_json(path)
        except json.JSONDecodeError:
            continue
        ticker = str(payload.get("ticker") or path.parts[-3]).upper()
        tickers.append(ticker)
    return {
        "path": Path(root).as_posix(),
        "status": "not_consumeable_for_fundamental_quality",
        "reason": "company profile evidence is descriptive identity/profile context, not canonical fundamental quality metrics",
        "total_records": len(paths),
        "tickers": sorted(set(tickers)),
        "canonical_matches": sorted(set(tickers) & canonical_set),
        "unsupported_tickers": sorted(set(tickers) - canonical_set),
    }


def _classify_ticker(
    instrument: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    parsed_as_of: date,
    generated_at: str,
) -> dict[str, Any]:
    ticker = str(instrument["symbol"]).upper()
    selected, conflict = _select_candidate(candidates)
    family_status = _family_status(selected)
    if conflict:
        status = "conflicting"
        selected = None
        blockers = ["conflicting_fundamental_evidence"]
    elif selected:
        status = str(selected["coverage_status"])
        blockers = list(selected.get("blockers") or [])
    else:
        status = "missing"
        blockers = ["missing_fundamental_context"]
    row = _quality_row(
        ticker=ticker,
        status=status,
        selected=selected,
        as_of=parsed_as_of,
        generated_at=generated_at,
        blockers=blockers,
    )
    return {
        "instrument_id": instrument["instrument_id"],
        "ticker": ticker,
        "asset_type": instrument.get("asset_type"),
        "name": instrument.get("name"),
        "overall_fundamental_status": status,
        "coverage_families": family_status,
        "selected_source_family": selected.get("source_family") if selected else None,
        "selected_source_name": selected.get("source_name") if selected else None,
        "selected_source_path": selected.get("source_path") if selected else None,
        "source_date": selected.get("source_date") if selected else None,
        "missing_metrics": list(selected.get("missing_metrics") or []) if selected else list(MVP_METRIC_FIELDS),
        "blockers": blockers,
        "candidate_sources": [
            {
                "source_family": item.get("source_family"),
                "source_name": item.get("source_name"),
                "source_date": item.get("source_date"),
                "coverage_status": item.get("coverage_status"),
                "priority": item.get("priority"),
            }
            for item in candidates
        ],
        "fundamental_quality_row": row,
    }


def _select_candidate(candidates: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, bool]:
    if not candidates:
        return None, False
    usable = [item for item in candidates if item.get("coverage_status") != "missing"]
    if not usable:
        return None, False
    best_priority = max(int(item.get("priority") or 0) for item in usable)
    priority_rows = [item for item in usable if int(item.get("priority") or 0) == best_priority]
    best_date = max(str(item.get("source_date") or "") for item in priority_rows)
    newest = [item for item in priority_rows if str(item.get("source_date") or "") == best_date]
    unique_payloads = {_canonical_candidate(item) for item in newest}
    if len(unique_payloads) > 1:
        return None, True
    return sorted(newest, key=lambda item: (str(item.get("source_family")), str(item.get("source_name")), str(item.get("source_reference"))))[0], False


def _canonical_candidate(item: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "source_family": item.get("source_family"),
            "source_name": item.get("source_name"),
            "source_date": item.get("source_date"),
            "coverage_status": item.get("coverage_status"),
            "metrics": item.get("metrics"),
            "missing_metrics": item.get("missing_metrics"),
        },
        sort_keys=True,
        default=str,
    )


def _family_status(selected: Mapping[str, Any] | None) -> dict[str, str]:
    if selected is None:
        return {
            "company_identity_profile": "missing",
            "revenue_growth": "missing",
            "profitability": "missing",
            "balance_sheet_strength": "missing",
            "cash_flow": "missing",
            "valuation": "unsupported",
            "overall_canonical_fundamental_context": "missing",
        }
    if selected.get("source_family") == "sec_companyfacts_source_context":
        field_states = selected.get("field_states") or {}
        return {
            "company_identity_profile": "partial",
            "revenue_growth": "partial" if field_states.get("revenue") == "PRESENT" else "missing",
            "profitability": "partial" if field_states.get("net_income") == "PRESENT" else "missing",
            "balance_sheet_strength": "missing",
            "cash_flow": "partial" if field_states.get("operating_cash_flow") == "PRESENT" else "missing",
            "valuation": "unsupported",
            "overall_canonical_fundamental_context": str(selected.get("coverage_status") or "partial"),
        }
    metrics = selected.get("metrics") or {}
    complete = str(selected.get("coverage_status")) == "complete"
    return {
        "company_identity_profile": "complete",
        "revenue_growth": "complete" if metrics.get("revenue_growth_yoy") is not None else "missing",
        "profitability": "complete" if metrics.get("gross_margin") is not None and metrics.get("operating_margin") is not None else "partial",
        "balance_sheet_strength": "complete" if metrics.get("debt_to_equity") is not None else "missing",
        "cash_flow": "unsupported",
        "valuation": "unsupported",
        "overall_canonical_fundamental_context": "complete" if complete else str(selected.get("coverage_status") or "partial"),
    }


def _quality_row(
    *,
    ticker: str,
    status: str,
    selected: Mapping[str, Any] | None,
    as_of: date,
    generated_at: str,
    blockers: Sequence[str],
) -> dict[str, str]:
    source_date = selected.get("source_date") if selected else ""
    freshness = ""
    if source_date:
        freshness = str((as_of - _parse_date(str(source_date), field_name="source_date")).days)
    missing = list(selected.get("missing_metrics") or []) if selected else list(MVP_METRIC_FIELDS)
    if status == "complete":
        quality_state = "SUFFICIENT_DATA"
        quality_reason = "fundamental source data available"
        profile = "OBSERVED"
        metadata_status = "complete"
        source_status = "source_available"
        partial_reason = ""
        invalid_reason = ""
        stale_reason = ""
    elif status == "partial":
        quality_state = "PARTIAL_DATA"
        quality_reason = "fundamental source data partially available"
        profile = "PARTIAL"
        metadata_status = "partial"
        source_status = "partial_data"
        partial_reason = "one or more required fundamental fields are missing"
        invalid_reason = ""
        stale_reason = ""
    elif status == "stale":
        quality_state = "INSUFFICIENT_DATA"
        quality_reason = "fundamental source data stale"
        profile = "UNAVAILABLE"
        metadata_status = "stale"
        source_status = "stale_data"
        partial_reason = ""
        invalid_reason = ""
        stale_reason = "source date is older than freshness threshold"
    elif status in {"invalid", "conflicting"}:
        quality_state = "INSUFFICIENT_DATA"
        quality_reason = "fundamental source data invalid"
        profile = "UNAVAILABLE"
        metadata_status = "invalid"
        source_status = "invalid_data"
        partial_reason = ""
        invalid_reason = "|".join(blockers)
        stale_reason = ""
    else:
        quality_state = "INSUFFICIENT_DATA"
        quality_reason = "fundamental source row unavailable"
        profile = "UNAVAILABLE"
        metadata_status = "row_missing"
        source_status = "row_missing"
        partial_reason = ""
        invalid_reason = ""
        stale_reason = ""
    return {
        "ticker": ticker,
        "date": as_of.isoformat(),
        "quality_state": quality_state,
        "quality_reason": quality_reason,
        "profitability_profile": profile,
        "balance_sheet_profile": profile,
        "earnings_quality_profile": profile,
        "capital_efficiency_profile": profile,
        "cashflow_profile": profile,
        "stability_profile": profile,
        "quality_metadata_status": metadata_status,
        "source_data_status": source_status,
        "source_timestamp": source_date,
        "source_name": str(selected.get("source_name") or "") if selected else "",
        "source_last_updated": source_date,
        "source_freshness_days": freshness,
        "missing_required_fields": "|".join(missing),
        "partial_data_reason": partial_reason,
        "stale_data_reason": stale_reason,
        "invalid_data_reason": invalid_reason,
        "generated_at": generated_at,
    }


def _coverage_summary(per_ticker: Sequence[Mapping[str, Any]], *, before: Mapping[str, Any], universe: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(str(row["overall_fundamental_status"]) for row in per_ticker)
    family_counts = {
        family: dict(sorted(Counter(str((row.get("coverage_families") or {}).get(family, "missing")) for row in per_ticker).items()))
        for family in (
            "company_identity_profile",
            "revenue_growth",
            "profitability",
            "balance_sheet_strength",
            "cash_flow",
            "valuation",
            "overall_canonical_fundamental_context",
        )
    }
    blocker_counts = Counter(blocker for row in per_ticker for blocker in row.get("blockers") or [])
    return {
        "schema_version": "market-engine-data06-fundamental-coverage-summary-v1",
        "canonical_universe_version": universe.get("universe_version"),
        "canonical_universe_size": len(per_ticker),
        "before": dict(before),
        "overall_counts": {status: counts.get(status, 0) for status in ("complete", "partial", "missing", "stale", "invalid", "conflicting", "unsupported")},
        "family_counts": family_counts,
        "blocker_summary": dict(sorted(blocker_counts.items())),
        "source_priority": SOURCE_PRIORITY,
        "classification_rules": {
            "complete": "all existing MVP fundamental quality metrics are present, numeric, current, and non-conflicting",
            "partial": "at least one approved local fundamental evidence family is present but the MVP quality contract is incomplete",
            "missing": "no approved local fundamental evidence source matched the canonical ticker",
            "stale": f"source date is more than {FRESHNESS_MAX_AGE_DAYS} days before the as-of date",
            "invalid": "source schema, source date, provenance, or numeric metric validation failed",
            "conflicting": "same-priority same-date evidence rows disagree materially",
            "unsupported": "local evidence exists but is not a supported fundamental-quality contract input",
        },
    }


def _load_before_baseline(
    baseline_run_evidence: str | Path,
    *,
    canonical_symbols: Sequence[str],
    canonical_universe_version: Any,
) -> dict[str, Any]:
    operator_path = Path(baseline_run_evidence)
    if not operator_path.exists():
        raise FundamentalEvidenceCoverageValidationError(f"baseline path does not exist: {operator_path}")
    index_path, compact_payload = _resolve_baseline_index(operator_path)
    manifest_path = index_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FundamentalEvidenceCoverageValidationError(f"baseline manifest missing: {manifest_path}")
    index = _read_baseline_json(index_path)
    manifest = _read_baseline_json(manifest_path)
    if not isinstance(index, Mapping) or not isinstance(manifest, Mapping):
        raise FundamentalEvidenceCoverageValidationError("baseline JSON artifacts must be objects")
    if manifest.get("artifact_type") != BASELINE_MANIFEST_ARTIFACT_TYPE:
        raise FundamentalEvidenceCoverageValidationError(
            f"unsupported baseline artifact type: {manifest.get('artifact_type')!r}"
        )
    if manifest.get("schema_version") != BASELINE_MANIFEST_SCHEMA_VERSION:
        raise FundamentalEvidenceCoverageValidationError(
            f"unsupported baseline manifest schema version: {manifest.get('schema_version')!r}"
        )
    if index.get("schema_version") != BASELINE_INDEX_SCHEMA_VERSION:
        raise FundamentalEvidenceCoverageValidationError(
            f"unsupported baseline index schema version: {index.get('schema_version')!r}"
        )
    run_id = str(index.get("run_id") or "").strip()
    if not run_id or run_id != str(manifest.get("run_id") or "").strip():
        raise FundamentalEvidenceCoverageValidationError("baseline run ID is missing or inconsistent")
    if compact_payload is not None and run_id != str(compact_payload.get("run_id") or "").strip():
        raise FundamentalEvidenceCoverageValidationError("compact and full baseline run IDs do not match")
    fundamental_input = str((manifest.get("input_artifacts") or {}).get("fundamental_evidence_path") or "")
    if "after-me-data06" in run_id.lower() or "fundamental_evidence_coverage_runs" in fundamental_input:
        raise FundamentalEvidenceCoverageValidationError("ME-DATA06 after-output cannot be used as the before baseline")
    baseline_version = manifest.get("canonical_universe_version")
    if baseline_version is None:
        raise FundamentalEvidenceCoverageValidationError("baseline canonical universe version is missing")
    if canonical_universe_version is None:
        raise FundamentalEvidenceCoverageValidationError("current canonical universe version is missing")
    if str(baseline_version) != str(canonical_universe_version):
        raise FundamentalEvidenceCoverageValidationError(
            f"baseline canonical universe version mismatch: {baseline_version!r} != {canonical_universe_version!r}"
        )
    instruments = index.get("instruments")
    if not isinstance(instruments, list):
        raise FundamentalEvidenceCoverageValidationError("baseline per-ticker instruments are missing")
    canonical = [str(symbol).upper() for symbol in canonical_symbols]
    if len(canonical) != len(set(canonical)):
        raise FundamentalEvidenceCoverageValidationError("current canonical universe contains duplicate tickers")
    if len(instruments) != len(canonical):
        raise FundamentalEvidenceCoverageValidationError(
            f"baseline canonical universe size mismatch: {len(instruments)} != {len(canonical)}"
        )
    records = []
    raw_status_counts: Counter[str] = Counter()
    seen: set[str] = set()
    for item in instruments:
        if not isinstance(item, Mapping):
            raise FundamentalEvidenceCoverageValidationError("baseline per-ticker record is not an object")
        ticker = str(item.get("symbol") or "").upper()
        if not ticker:
            raise FundamentalEvidenceCoverageValidationError("baseline ticker is missing")
        if ticker in seen:
            raise FundamentalEvidenceCoverageValidationError(f"duplicate baseline ticker: {ticker}")
        seen.add(ticker)
        fundamental = item.get("fundamental_context")
        if not isinstance(fundamental, Mapping):
            raise FundamentalEvidenceCoverageValidationError(f"baseline fundamental context missing for {ticker}")
        raw_status = str(fundamental.get("status") or "")
        raw_status_counts[raw_status] += 1
        status = _normalize_run31_fundamental_status(fundamental, ticker=ticker)
        canonical_ready = item.get("canonical_advice_input_ready")
        full_ready = item.get("full_advice_ready")
        if not isinstance(canonical_ready, bool) or not isinstance(full_ready, bool):
            raise FundamentalEvidenceCoverageValidationError(f"baseline readiness flags missing for {ticker}")
        advice_label = item.get("canonical_advice_label")
        if not isinstance(advice_label, str) or not advice_label:
            raise FundamentalEvidenceCoverageValidationError(f"baseline advice status missing for {ticker}")
        records.append(
            {
                "ticker": ticker,
                "fundamental_status": status,
                "advice_input_ready": canonical_ready,
                "full_advice_ready": full_ready,
                "unable_to_advise": advice_label == "unable_to_advise",
            }
        )
    if seen != set(canonical):
        missing = sorted(set(canonical) - seen)
        extra = sorted(seen - set(canonical))
        raise FundamentalEvidenceCoverageValidationError(
            f"baseline canonical ticker identity mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    coverage = manifest.get("coverage_summary")
    if not isinstance(coverage, Mapping):
        raise FundamentalEvidenceCoverageValidationError("baseline coverage summary is missing")
    declared_fundamental = coverage.get("fundamental_counts")
    if not isinstance(declared_fundamental, Mapping):
        raise FundamentalEvidenceCoverageValidationError("baseline fundamental counts are missing")
    for status in ("available", "partial", "missing", "stale", "invalid", "blocked", "not_applicable"):
        if _strict_count(declared_fundamental.get(status), f"baseline fundamental {status}") != raw_status_counts.get(status, 0):
            raise FundamentalEvidenceCoverageValidationError(f"baseline fundamental count mismatch for {status}")
    if sum(raw_status_counts.values()) != len(records):
        raise FundamentalEvidenceCoverageValidationError("baseline fundamental counts do not reconcile")
    _validate_declared_count(coverage, "canonical_instruments", len(records), "baseline canonical instruments")
    _validate_declared_count(coverage, "attempted_instruments", len(records), "baseline attempted instruments")
    _validate_declared_count(
        coverage,
        "canonical_advice_input_ready",
        sum(1 for row in records if row["advice_input_ready"]),
        "baseline advice-input-ready",
    )
    _validate_declared_count(
        coverage,
        "full_advice_ready",
        sum(1 for row in records if row["full_advice_ready"]),
        "baseline full-advice-ready",
    )
    advice_counts = coverage.get("advice_counts")
    if not isinstance(advice_counts, Mapping):
        raise FundamentalEvidenceCoverageValidationError("baseline advice counts are missing")
    _validate_declared_count(
        advice_counts,
        "unable_to_advise",
        sum(1 for row in records if row["unable_to_advise"]),
        "baseline unable-to-advise",
    )
    normalized_counts = Counter(str(row["fundamental_status"]) for row in records)
    checksum = _sha256(index_path)
    if compact_payload is not None:
        expected_checksum = (
            ((compact_payload.get("full_artifact") or {}).get("top_level_file_checksums") or {}).get(index_path.name)
        )
        if not expected_checksum or expected_checksum != checksum:
            raise FundamentalEvidenceCoverageValidationError("baseline index checksum does not match compact evidence metadata")
    records.sort(key=lambda row: row["ticker"])
    return {
        "run_id": run_id,
        "artifact_path": index_path.as_posix(),
        "artifact_type": BASELINE_INDEX_ARTIFACT_TYPE,
        "schema_version": str(index["schema_version"]),
        "checksum": checksum,
        "canonical_universe_version": str(baseline_version),
        "canonical_universe_size": len(records),
        "records": records,
        "counts": {
            "fundamental_complete": normalized_counts.get("complete", 0),
            "fundamental_partial": normalized_counts.get("partial", 0),
            "fundamental_missing": normalized_counts.get("missing", 0),
            "invalid_stale_conflicting": sum(
                normalized_counts.get(status, 0) for status in ("invalid", "stale", "conflicting")
            ),
            "canonical_advice_input_ready": sum(1 for row in records if row["advice_input_ready"]),
            "full_advice_ready": sum(1 for row in records if row["full_advice_ready"]),
            "unable_to_advise": sum(1 for row in records if row["unable_to_advise"]),
            "source_run_id": run_id,
        },
    }


def _resolve_baseline_index(path: Path) -> tuple[Path, Mapping[str, Any] | None]:
    if path.is_dir() and (path / "evidence_coverage_index.json").exists():
        return path / "evidence_coverage_index.json", None
    compact_index = path / "run_evidence_index.json" if path.is_dir() else path
    if compact_index.name == "evidence_coverage_index.json":
        return compact_index, None
    if compact_index.name != "run_evidence_index.json" or not compact_index.exists():
        raise FundamentalEvidenceCoverageValidationError(
            "baseline must be a full ME-RUN31 evidence directory, evidence_coverage_index.json, "
            "or compact ME-RUN31 evidence directory"
        )
    compact = _read_baseline_json(compact_index)
    if not isinstance(compact, Mapping):
        raise FundamentalEvidenceCoverageValidationError("compact baseline index must be an object")
    if compact.get("schema_version") != BASELINE_COMPACT_SCHEMA_VERSION:
        raise FundamentalEvidenceCoverageValidationError(
            f"unsupported compact baseline schema version: {compact.get('schema_version')!r}"
        )
    full_path = (compact.get("full_artifact") or {}).get("local_path")
    if not full_path:
        raise FundamentalEvidenceCoverageValidationError("compact baseline does not identify its full artifact")
    index_path = Path(str(full_path)) / "evidence_coverage_index.json"
    if not index_path.exists():
        raise FundamentalEvidenceCoverageValidationError(
            f"compact baseline lacks required per-ticker full artifact: {index_path}"
        )
    return index_path, compact


def _read_baseline_json(path: Path) -> Any:
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise FundamentalEvidenceCoverageValidationError(f"baseline JSON is not parseable: {path}") from exc


def _normalize_run31_fundamental_status(context: Mapping[str, Any], *, ticker: str) -> str:
    raw_status = str(context.get("status") or "")
    if raw_status == "available":
        return "complete"
    if raw_status in {"partial", "missing", "stale"}:
        return raw_status
    if raw_status in {"invalid", "blocked"}:
        blockers = {str(item) for item in context.get("blockers") or []}
        if "duplicate_fundamental_rows_conflict" in blockers or any("conflict" in item for item in blockers):
            return "conflicting"
        return "invalid"
    raise FundamentalEvidenceCoverageValidationError(
        f"unsupported baseline fundamental status for {ticker}: {raw_status!r}"
    )


def _strict_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FundamentalEvidenceCoverageValidationError(f"{label} must be a non-negative integer")
    return value


def _validate_declared_count(payload: Mapping[str, Any], key: str, actual: int, label: str) -> None:
    if _strict_count(payload.get(key), label) != actual:
        raise FundamentalEvidenceCoverageValidationError(f"{label} count mismatch")


def _after_summary(per_ticker: Sequence[Mapping[str, Any]], downstream: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(str(row["overall_fundamental_status"]) for row in per_ticker)
    return {
        "fundamental_complete": counts.get("complete", 0),
        "fundamental_partial": counts.get("partial", 0),
        "fundamental_missing": counts.get("missing", 0),
        "invalid_stale_conflicting": counts.get("invalid", 0) + counts.get("stale", 0) + counts.get("conflicting", 0),
        "canonical_advice_input_ready": downstream.get("canonical_advice_input_ready"),
        "full_advice_ready": downstream.get("full_advice_ready"),
        "unable_to_advise": downstream.get("unable_to_advise"),
    }


def _before_after_comparison(
    baseline: Mapping[str, Any],
    after: Mapping[str, Any],
    per_ticker: Sequence[Mapping[str, Any]],
    run31_artifacts: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    before_records = {str(row["ticker"]): row for row in baseline["records"]}
    after_status = {str(row["ticker"]): str(row["overall_fundamental_status"]) for row in per_ticker}
    if len(after_status) != len(per_ticker):
        raise FundamentalEvidenceCoverageValidationError("duplicate after fundamental tickers")
    run31_index = run31_artifacts.get("evidence_coverage_index")
    if not isinstance(run31_index, Mapping) or run31_index.get("schema_version") != BASELINE_INDEX_SCHEMA_VERSION:
        raise FundamentalEvidenceCoverageValidationError("after ME-RUN31 evidence index is missing or unsupported")
    after_instruments = run31_index.get("instruments")
    if not isinstance(after_instruments, list):
        raise FundamentalEvidenceCoverageValidationError("after ME-RUN31 per-ticker evidence is missing")
    after_records: dict[str, dict[str, Any]] = {}
    after_raw_status_counts: Counter[str] = Counter()
    for item in after_instruments:
        if not isinstance(item, Mapping):
            raise FundamentalEvidenceCoverageValidationError("after ME-RUN31 per-ticker record is not an object")
        ticker = str(item.get("symbol") or "").upper()
        if not ticker or ticker in after_records:
            raise FundamentalEvidenceCoverageValidationError(f"duplicate or missing after ticker: {ticker!r}")
        fundamental = item.get("fundamental_context")
        if not isinstance(fundamental, Mapping):
            raise FundamentalEvidenceCoverageValidationError(f"after fundamental context missing for {ticker}")
        after_raw_status_counts[str(fundamental.get("status") or "")] += 1
        status = _normalize_run31_fundamental_status(fundamental, ticker=ticker)
        if after_status.get(ticker) != status:
            raise FundamentalEvidenceCoverageValidationError(f"after fundamental status mismatch for {ticker}")
        advice_ready = item.get("canonical_advice_input_ready")
        full_ready = item.get("full_advice_ready")
        if not isinstance(advice_ready, bool) or not isinstance(full_ready, bool):
            raise FundamentalEvidenceCoverageValidationError(f"after readiness flags missing for {ticker}")
        advice_label = item.get("canonical_advice_label")
        if not isinstance(advice_label, str) or not advice_label:
            raise FundamentalEvidenceCoverageValidationError(f"after advice status missing for {ticker}")
        after_records[ticker] = {
            "ticker": ticker,
            "fundamental_status": status,
            "advice_input_ready": advice_ready,
            "full_advice_ready": full_ready,
            "unable_to_advise": advice_label == "unable_to_advise",
        }
    if set(before_records) != set(after_records) or set(after_records) != set(after_status):
        raise FundamentalEvidenceCoverageValidationError("before and after canonical ticker identities do not match")
    expected_after = {
        "fundamental_complete": sum(1 for row in after_records.values() if row["fundamental_status"] == "complete"),
        "fundamental_partial": sum(1 for row in after_records.values() if row["fundamental_status"] == "partial"),
        "fundamental_missing": sum(1 for row in after_records.values() if row["fundamental_status"] == "missing"),
        "invalid_stale_conflicting": sum(
            1 for row in after_records.values() if row["fundamental_status"] in {"invalid", "stale", "conflicting"}
        ),
        "canonical_advice_input_ready": sum(1 for row in after_records.values() if row["advice_input_ready"]),
        "full_advice_ready": sum(1 for row in after_records.values() if row["full_advice_ready"]),
        "unable_to_advise": sum(1 for row in after_records.values() if row["unable_to_advise"]),
    }
    for key, value in expected_after.items():
        if after.get(key) != value:
            raise FundamentalEvidenceCoverageValidationError(f"after count mismatch for {key}")
    after_summary = (run31_artifacts.get("evidence_coverage_summary") or {}).get("summary")
    if not isinstance(after_summary, Mapping):
        raise FundamentalEvidenceCoverageValidationError("after ME-RUN31 coverage summary is missing")
    _validate_declared_count(after_summary, "attempted_instruments", len(after_records), "after attempted instruments")
    _validate_declared_count(after_summary, "canonical_instruments", len(after_records), "after canonical instruments")
    _validate_declared_count(
        after_summary,
        "canonical_advice_input_ready",
        expected_after["canonical_advice_input_ready"],
        "after advice-input-ready",
    )
    _validate_declared_count(
        after_summary,
        "full_advice_ready",
        expected_after["full_advice_ready"],
        "after full-advice-ready",
    )
    after_advice_counts = after_summary.get("advice_counts")
    if not isinstance(after_advice_counts, Mapping):
        raise FundamentalEvidenceCoverageValidationError("after ME-RUN31 advice counts are missing")
    _validate_declared_count(
        after_advice_counts,
        "unable_to_advise",
        expected_after["unable_to_advise"],
        "after unable-to-advise",
    )
    after_fundamental_counts = after_summary.get("fundamental_counts")
    if not isinstance(after_fundamental_counts, Mapping):
        raise FundamentalEvidenceCoverageValidationError("after ME-RUN31 fundamental counts are missing")
    for status in ("available", "partial", "missing", "stale", "invalid", "blocked", "not_applicable"):
        if _strict_count(after_fundamental_counts.get(status), f"after fundamental {status}") != after_raw_status_counts.get(status, 0):
            raise FundamentalEvidenceCoverageValidationError(f"after fundamental count mismatch for {status}")
    transitions = []
    improvement_tickers: dict[str, list[str]] = {}
    regression_tickers: dict[str, list[str]] = {}
    transition_tickers: dict[str, list[str]] = {}
    for ticker in sorted(before_records):
        before_row = before_records[ticker]
        after_row = after_records[ticker]
        transition_types = _fundamental_transition_types(before_row, after_row)
        for transition_type in transition_types:
            transition_tickers.setdefault(transition_type, []).append(ticker)
            if transition_type in _improvement_transition_types():
                improvement_tickers.setdefault(transition_type, []).append(ticker)
            if transition_type in _regression_transition_types():
                regression_tickers.setdefault(transition_type, []).append(ticker)
        transitions.append(
            {
                "ticker": ticker,
                "before_fundamental_status": before_row["fundamental_status"],
                "after_fundamental_status": after_row["fundamental_status"],
                "before_advice_input_ready": before_row["advice_input_ready"],
                "after_advice_input_ready": after_row["advice_input_ready"],
                "before_full_advice_ready": before_row["full_advice_ready"],
                "after_full_advice_ready": after_row["full_advice_ready"],
                "transition_types": transition_types,
            }
        )
    for transition_type in sorted(_improvement_transition_types()):
        improvement_tickers.setdefault(transition_type, [])
    for transition_type in sorted(_regression_transition_types()):
        regression_tickers.setdefault(transition_type, [])
    regression_ticker_set = {ticker for tickers in regression_tickers.values() for ticker in tickers}
    improvement_ticker_set = {ticker for tickers in improvement_tickers.values() for ticker in tickers}
    before = baseline["counts"]
    run_status = RUN_STATUS_COMPLETED_WITH_REGRESSIONS if regression_ticker_set else RUN_STATUS_COMPLETED
    comparison = {
        "schema_version": "market-engine-data06-before-after-comparison-v2",
        "run_status": run_status,
        "baseline": {
            "run_id": baseline["run_id"],
            "artifact_path": baseline["artifact_path"],
            "artifact_type": baseline["artifact_type"],
            "schema_version": baseline["schema_version"],
            "checksum": baseline["checksum"],
            "canonical_universe_version": baseline["canonical_universe_version"],
            "canonical_universe_size": baseline["canonical_universe_size"],
        },
        "before_counts": dict(before),
        "after_counts": dict(after),
        "absolute_improvement": {
            key: (after.get(key, 0) or 0) - (before.get(key, 0) or 0)
            for key in ("fundamental_complete", "fundamental_partial", "fundamental_missing", "canonical_advice_input_ready", "full_advice_ready", "unable_to_advise")
        },
        "percent_improvement": {
            key: _percent_change(before.get(key, 0) or 0, after.get(key, 0) or 0)
            for key in ("fundamental_complete", "fundamental_partial", "canonical_advice_input_ready")
        },
        "improvement_counts": {key: len(value) for key, value in sorted(improvement_tickers.items())},
        "regression_counts": {key: len(value) for key, value in sorted(regression_tickers.items())},
        "improvement_tickers": dict(sorted(improvement_tickers.items())),
        "regression_tickers": dict(sorted(regression_tickers.items())),
        "improvement_ticker_count": len(improvement_ticker_set),
        "regression_ticker_count": len(regression_ticker_set),
        "transition_counts": {key: len(value) for key, value in sorted(transition_tickers.items())},
        "transition_detail_artifact": "per_ticker_fundamental_transitions.json",
        "transition_reconciliation": {
            "ticker_count": len(transitions),
            "before_unique_tickers": len(before_records),
            "after_unique_tickers": len(after_records),
            "counts_reconciled": True,
        },
    }
    detail = {
        "schema_version": "market-engine-data06-per-ticker-fundamental-transitions-v1",
        "artifact_type": "market-engine-data06-per-ticker-fundamental-transitions",
        "baseline_run_id": baseline["run_id"],
        "after_run_id": run31_index.get("run_id"),
        "ticker_count": len(transitions),
        "tickers": transitions,
    }
    return comparison, detail


def _fundamental_transition_types(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    before_status = str(before["fundamental_status"])
    after_status = str(after["fundamental_status"])
    transitions = []
    direct = f"{before_status}_to_{after_status}"
    if direct in {
        "complete_to_partial",
        "complete_to_missing",
        "complete_to_stale",
        "complete_to_invalid",
        "complete_to_conflicting",
        "partial_to_missing",
        "partial_to_stale",
        "partial_to_invalid",
        "partial_to_conflicting",
        "missing_to_partial",
        "missing_to_complete",
        "partial_to_complete",
    }:
        transitions.append(direct)
    if before_status != "stale" and after_status == "stale":
        transitions.append("became_stale")
    if before_status != "invalid" and after_status == "invalid":
        transitions.append("became_invalid")
    if before_status != "conflicting" and after_status == "conflicting":
        transitions.append("became_conflicting")
    if before_status == "stale" and after_status in {"complete", "partial"}:
        transitions.append("resolved_stale")
    if before_status == "invalid" and after_status in {"complete", "partial"}:
        transitions.append("resolved_invalid")
    if before_status == "conflicting" and after_status in {"complete", "partial"}:
        transitions.append("resolved_conflicting")
    if before["advice_input_ready"] and not after["advice_input_ready"]:
        transitions.append("lost_advice_input_readiness")
    if not before["advice_input_ready"] and after["advice_input_ready"]:
        transitions.append("newly_advice_input_ready")
    if before["full_advice_ready"] and not after["full_advice_ready"]:
        transitions.append("lost_full_advice_readiness")
    if not before["full_advice_ready"] and after["full_advice_ready"]:
        transitions.append("newly_full_advice_ready")
    return transitions


def _improvement_transition_types() -> set[str]:
    return {
        "missing_to_partial",
        "missing_to_complete",
        "partial_to_complete",
        "resolved_stale",
        "resolved_invalid",
        "resolved_conflicting",
        "newly_advice_input_ready",
        "newly_full_advice_ready",
    }


def _regression_transition_types() -> set[str]:
    return {
        "complete_to_partial",
        "complete_to_missing",
        "complete_to_stale",
        "complete_to_invalid",
        "complete_to_conflicting",
        "partial_to_missing",
        "partial_to_stale",
        "partial_to_invalid",
        "partial_to_conflicting",
        "became_stale",
        "became_invalid",
        "became_conflicting",
        "lost_advice_input_readiness",
        "lost_full_advice_readiness",
    }


def _downstream_summary(artifacts: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    summary = (artifacts.get("evidence_coverage_summary") or {}).get("summary") or {}
    advice_counts = summary.get("advice_counts") or {}
    return {
        "run_id": artifacts.get("manifest", {}).get("run_id"),
        "output_dir": output_dir.as_posix(),
        "compact_evidence_dir": artifacts.get("compact_evidence_dir"),
        "attempted": summary.get("attempted_instruments"),
        "canonical_advice_input_ready": summary.get("canonical_advice_input_ready"),
        "full_advice_ready": summary.get("full_advice_ready"),
        "unable_to_advise": advice_counts.get("unable_to_advise"),
        "advice_engine_completed": summary.get("advice_engine_completed"),
    }


def _evidence_source_inventory(
    *,
    canonical_set: set[str],
    existing_path: str | Path,
    existing_rows: Sequence[Mapping[str, str]],
    manual: Mapping[str, Any],
    sec: Mapping[str, Any],
    intake: Mapping[str, Any],
    company_profiles: Mapping[str, Any],
    as_of: date,
) -> dict[str, Any]:
    sources = []
    sources.append(
        _inventory_row(
            "existing_processed_fundamental_quality",
            "processed fundamental quality CSV",
            existing_path,
            existing_rows,
            canonical_set,
            "consumeable_baseline",
            "existing ME-RUN31 contract input",
            as_of=as_of,
            date_fields=("source_last_updated", "source_timestamp", "date"),
        )
    )
    sources.append(
        _inventory_row(
            "manual_mvp_fundamentals",
            "raw manual MVP fundamentals CSV",
            manual["path"],
            manual.get("rows") or [],
            canonical_set,
            manual["status"],
            "consumed for normalized MVP fundamental quality where rows validate",
            as_of=as_of,
            date_fields=("source_freshness_date", "as_of_date"),
            validation_failed=manual.get("status") == "unsupported_schema"
            and bool(set(manual.get("missing_columns") or []) & {"source_freshness_date", "as_of_date"}),
        )
    )
    sec_tickers = [row["ticker"] for row in sec.get("candidates") or []]
    sec_freshness = _classify_inventory_freshness(
        sec.get("candidates") or [],
        as_of=as_of,
        date_fields=("source_date",),
        validation_failed=bool(sec.get("errors")),
    )
    sources.append(
        {
            "evidence_family": "sec_companyfacts_source_context",
            "dataset_name": "local SEC CompanyFacts raw snapshots",
            "path": sec["path"],
            "format": "sec-companyfacts-raw-v1",
            "schema_version": "sec-companyfacts-source-context-v1",
            "tickers_found": len(set(sec_tickers)),
            "canonical_matches": len(set(sec_tickers) & canonical_set),
            "unsupported_tickers": sorted(set(sec_tickers) - canonical_set),
            "source_date": sec_freshness["latest_source_date"],
            "acquired_at": sec_freshness["latest_source_date"],
            "provenance_status": "available" if sec.get("candidates") else "missing",
            "freshness_status": sec_freshness["freshness_status"],
            "freshness_counts": sec_freshness["freshness_counts"],
            "validation_status": "valid" if not sec.get("errors") else "partial_with_errors",
            "market_engine_consumeable": True,
            "consumeability_reason": "source-context observations are consumed as partial evidence only; they do not satisfy the MVP quality metric contract",
        }
    )
    sources.append(
        _inventory_row(
            "intake_placeholder_fundamentals",
            "OS5 AB fundamentals intake pilot",
            intake["path"],
            intake.get("rows") or [],
            canonical_set,
            "not_consumeable",
            intake.get("reason") or "not runtime evidence",
            as_of=as_of,
            date_fields=("fundamentals_last_updated",),
        )
    )
    company_profile_freshness = _classify_inventory_freshness(
        [{} for _ in range(int(company_profiles["total_records"]))],
        as_of=as_of,
        date_fields=(),
        applicable=False,
    )
    sources.append(
        {
            "evidence_family": "company_profile",
            "dataset_name": "local company profile snapshots",
            "path": company_profiles["path"],
            "format": "company_profile.json",
            "schema_version": "market-engine-company-profile-source-context-v1",
            "tickers_found": company_profiles["total_records"],
            "canonical_matches": len(company_profiles["canonical_matches"]),
            "unsupported_tickers": company_profiles["unsupported_tickers"],
            "source_date": "",
            "acquired_at": "",
            "provenance_status": "available" if company_profiles["total_records"] else "missing",
            "freshness_status": company_profile_freshness["freshness_status"],
            "freshness_counts": company_profile_freshness["freshness_counts"],
            "validation_status": "not_consumeable_for_fundamental_quality",
            "market_engine_consumeable": False,
            "consumeability_reason": company_profiles["reason"],
        }
    )
    return {
        "schema_version": "market-engine-data06-evidence-source-inventory-v1",
        "sources": sources,
        "summary": {
            "sources_discovered": len(sources),
            "sources_consumed": sum(1 for row in sources if row["market_engine_consumeable"]),
            "sources_rejected": sum(1 for row in sources if not row["market_engine_consumeable"]),
            "freshness_counts": dict(
                sorted(Counter(str(row["freshness_status"]) for row in sources).items())
            ),
        },
    }


def _inventory_row(
    family: str,
    name: str,
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    canonical_set: set[str],
    status: str,
    reason: str,
    *,
    as_of: date,
    date_fields: Sequence[str],
    validation_failed: bool = False,
) -> dict[str, Any]:
    tickers = {str(row.get("ticker") or "").upper() for row in rows if row.get("ticker")}
    freshness = _classify_inventory_freshness(
        rows,
        as_of=as_of,
        date_fields=date_fields,
        validation_failed=validation_failed,
    )
    return {
        "evidence_family": family,
        "dataset_name": name,
        "path": Path(path).as_posix(),
        "format": "csv",
        "schema_version": "market-engine-fundamental-quality-csv-v1" if family == "existing_processed_fundamental_quality" else "local-manual-mvp-fundamentals-csv-v1",
        "tickers_found": len(tickers),
        "canonical_matches": len(tickers & canonical_set),
        "unsupported_tickers": sorted(tickers - canonical_set),
        "source_date": freshness["latest_source_date"],
        "acquired_at": freshness["latest_source_date"],
        "provenance_status": "available" if rows else "missing",
        "freshness_status": freshness["freshness_status"],
        "freshness_counts": freshness["freshness_counts"],
        "validation_status": status,
        "market_engine_consumeable": status in {"consumeable", "consumeable_baseline"},
        "consumeability_reason": reason,
    }


def _classify_inventory_freshness(
    records: Sequence[Mapping[str, Any]],
    *,
    as_of: date,
    date_fields: Sequence[str],
    applicable: bool = True,
    validation_failed: bool = False,
) -> dict[str, Any]:
    if not applicable:
        return {
            "freshness_status": "not_assessed",
            "freshness_counts": {"not_assessed": len(records)},
            "latest_source_date": "",
        }
    counts: Counter[str] = Counter()
    valid_dates: list[date] = []
    for record in records:
        raw = next((record.get(field) for field in date_fields if record.get(field) not in (None, "")), None)
        if raw is None:
            counts["unknown"] += 1
            continue
        try:
            source_date = _parse_date(raw, field_name="inventory_source_date")
        except FundamentalEvidenceCoverageError:
            counts["invalid"] += 1
            continue
        if source_date > as_of:
            counts["invalid"] += 1
            continue
        valid_dates.append(source_date)
        counts["stale" if _is_stale(source_date, as_of=as_of) else "current"] += 1
    if validation_failed:
        counts["invalid"] += 1
    if counts["invalid"]:
        status = "invalid"
    elif counts["unknown"]:
        status = "unknown"
    elif counts["current"] and counts["stale"]:
        status = "mixed"
    elif counts["current"]:
        status = "current"
    elif counts["stale"]:
        status = "stale"
    else:
        status = "unknown"
    freshness_counts = {
        key: counts.get(key, 0)
        for key in ("current", "stale", "invalid", "unknown")
        if counts.get(key, 0)
    }
    if not freshness_counts and status == "unknown":
        freshness_counts = {"unknown": 0}
    return {
        "freshness_status": status,
        "freshness_counts": freshness_counts,
        "latest_source_date": max(valid_dates).isoformat() if valid_dates else "",
    }


def _write_artifacts(output_dir: Path, artifacts: Mapping[str, Any]) -> None:
    _write_json(output_dir / "manifest.json", artifacts["manifest"])
    _write_json(output_dir / "evidence_source_inventory.json", artifacts["evidence_source_inventory"])
    _write_json(output_dir / "fundamental_coverage_summary.json", artifacts["fundamental_coverage_summary"])
    _write_json(output_dir / "per_ticker_fundamental_status.json", artifacts["per_ticker_fundamental_status"])
    _write_json(output_dir / "missing_fundamental_evidence.json", artifacts["missing_fundamental_evidence"])
    _write_json(output_dir / "partial_fundamental_evidence.json", artifacts["partial_fundamental_evidence"])
    _write_json(output_dir / "invalid_or_stale_evidence.json", artifacts["invalid_or_stale_evidence"])
    _write_json(output_dir / "before_after_comparison.json", artifacts["before_after_comparison"])
    _write_json(
        output_dir / "per_ticker_fundamental_transitions.json",
        artifacts["per_ticker_fundamental_transitions"],
    )


def _render_report(artifacts: Mapping[str, Any]) -> str:
    manifest = artifacts["manifest"]
    summary = artifacts["fundamental_coverage_summary"]
    before = summary["before"]
    after = summary["after"]
    comparison = summary["comparison"]
    inventory = artifacts["evidence_source_inventory"]["summary"]
    return "\n".join(
        [
            f"# ME-DATA06 Fundamental Evidence Coverage Report",
            "",
            f"Run ID: `{manifest['run_id']}`",
            f"Run status: `{manifest['run_status']}`",
            f"As-of date: `{manifest['as_of_date']}`",
            f"Baseline run ID: `{manifest['baseline_run_id']}`",
            "",
            "## Evidence Sources",
            "",
            f"- discovered: {inventory['sources_discovered']}",
            f"- consumed: {inventory['sources_consumed']}",
            f"- rejected: {inventory['sources_rejected']}",
            "",
            "## Before",
            "",
            f"- fundamental complete: {before.get('fundamental_complete', 0)}",
            f"- fundamental partial: {before.get('fundamental_partial', 0)}",
            f"- fundamental missing: {before.get('fundamental_missing', 0)}",
            f"- canonical advice-input-ready: {before.get('canonical_advice_input_ready', 0)}",
            f"- full-advice-ready: {before.get('full_advice_ready', 0)}",
            f"- unable-to-advise: {before.get('unable_to_advise', 0)}",
            "",
            "## After",
            "",
            f"- fundamental complete: {after.get('fundamental_complete', 0)}",
            f"- fundamental partial: {after.get('fundamental_partial', 0)}",
            f"- fundamental missing: {after.get('fundamental_missing', 0)}",
            f"- canonical advice-input-ready: {after.get('canonical_advice_input_ready', 0)}",
            f"- full-advice-ready: {after.get('full_advice_ready', 0)}",
            f"- unable-to-advise: {after.get('unable_to_advise', 0)}",
            "",
            "## Improvements",
            "",
            f"- missing to partial: {comparison['improvement_counts']['missing_to_partial']}",
            f"- missing to complete: {comparison['improvement_counts']['missing_to_complete']}",
            f"- partial to complete: {comparison['improvement_counts']['partial_to_complete']}",
            f"- newly advice-input-ready: {comparison['improvement_counts']['newly_advice_input_ready']}",
            f"- regression tickers: {comparison['regression_ticker_count']}",
            "",
            "## Side Effects",
            "",
            "No network access, provider calls, model calls, broker actions, allocation, portfolio/watchlist mutation, Telegram delivery, scheduler behavior, Decision Engine changes, or recommendation-rule changes were performed.",
            "",
        ]
    )


def _strip_quality_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "fundamental_quality_row"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_date(value: Any, *, field_name: str) -> date:
    if value is None or str(value).strip() == "":
        raise FundamentalEvidenceCoverageError(f"missing {field_name}")
    text = str(value).strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise FundamentalEvidenceCoverageError(f"invalid {field_name}: {value}") from exc


def _is_stale(source_date: date, *, as_of: date) -> bool:
    return (as_of - source_date).days > FRESHNESS_MAX_AGE_DAYS


def _percent_change(before: int | float, after: int | float) -> float | None:
    if before == 0:
        return None if after == 0 else 100.0
    return round(((after - before) / before) * 100, 4)


def _input_checksums(*paths: str | Path | None) -> dict[str, str | None]:
    checksums = {}
    for path in paths:
        if path is None:
            continue
        resolved = Path(path)
        checksums[resolved.as_posix()] = _sha256(resolved) if resolved.exists() and resolved.is_file() else None
    return checksums


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ME-DATA06 local fundamental evidence coverage expansion.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--canonical-universe", default=DEFAULT_CANONICAL_CONFIG)
    parser.add_argument("--price-history-root", default=DEFAULT_PRICE_HISTORY_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--existing-fundamental-path", default=DEFAULT_EXISTING_FUNDAMENTAL_PATH)
    parser.add_argument("--raw-fundamentals-path", default=DEFAULT_RAW_FUNDAMENTALS_PATH)
    parser.add_argument("--intake-fundamentals-path", default=DEFAULT_INTAKE_FUNDAMENTALS_PATH)
    parser.add_argument("--sec-companyfacts-root", default=DEFAULT_SEC_COMPANYFACTS_ROOT)
    parser.add_argument("--company-profile-root", default=DEFAULT_COMPANY_PROFILE_ROOT)
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    parser.add_argument(
        "--baseline-run-evidence",
        help=(
            "ME-RUN31 compact/full evidence directory or evidence_coverage_index.json "
            f"(default: {DEFAULT_BASELINE_RUN_EVIDENCE})"
        ),
    )
    parser.add_argument("--technical-screening-artifact", default=DEFAULT_TECHNICAL_SCREENING_ARTIFACT)
    parser.add_argument("--market-context-path", default=DEFAULT_MARKET_CONTEXT_PATH)
    parser.add_argument("--portfolio-context-path", default=DEFAULT_PORTFOLIO_CONTEXT_PATH)
    parser.add_argument("--run31-output-root", default=DEFAULT_RUN31_OUTPUT_ROOT)
    parser.add_argument("--run31-compact-evidence-root", default=DEFAULT_COMPACT_EVIDENCE_ROOT)
    parser.add_argument("--run31-run-id")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO, stderr: TextIO) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        artifacts, output_dir = run_fundamental_evidence_coverage(
            run_id=args.run_id,
            canonical_universe=args.canonical_universe,
            price_history_root=args.price_history_root,
            output_root=args.output_root,
            existing_fundamental_path=args.existing_fundamental_path,
            raw_fundamentals_path=args.raw_fundamentals_path,
            intake_fundamentals_path=args.intake_fundamentals_path,
            sec_companyfacts_root=args.sec_companyfacts_root,
            company_profile_root=args.company_profile_root,
            as_of_date=args.as_of_date,
            baseline_run_evidence=args.baseline_run_evidence,
            technical_screening_artifact=args.technical_screening_artifact,
            market_context_path=args.market_context_path,
            portfolio_context_path=args.portfolio_context_path,
            run31_output_root=args.run31_output_root,
            run31_compact_evidence_root=args.run31_compact_evidence_root,
            run31_run_id=args.run31_run_id,
            allow_overwrite=args.allow_overwrite,
        )
    except Exception as exc:
        print(f"ME-DATA06 failed: {exc}", file=stderr)
        return 1
    print(json.dumps({"run_id": args.run_id, "output_dir": output_dir.as_posix(), "summary": artifacts["fundamental_coverage_summary"]}, indent=2, sort_keys=True), file=stdout)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
