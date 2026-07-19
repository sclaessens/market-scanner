from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.data.fundamental_evidence_coverage import (
    DEFAULT_AS_OF_DATE,
    DEFAULT_TECHNICAL_SCREENING_ARTIFACT,
    FRESHNESS_MAX_AGE_DAYS,
    MVP_METRIC_FIELDS,
    run_fundamental_evidence_coverage,
)
from market_engine.data.complete_local_market_dataset import DEFAULT_CANONICAL_CONFIG
from market_engine.data.local_market_data_universe import (
    DEFAULT_PRICE_HISTORY_ROOT,
    build_universe_snapshot,
)
from market_engine.data.operator_source_approval import validate_source_approval_decision


SCHEMA_VERSION = "market-engine-data07-validated-fundamental-metric-sourcing-v1"
OPERATOR_IMPORT_SCHEMA_VERSION = "market-engine-data07-operator-fundamental-metrics-v1"
DEFAULT_OUTPUT_ROOT = Path("artifacts/market_engine/fundamental_metric_sourcing_runs")
DEFAULT_BASELINE_DATA06_RUN = Path(
    "artifacts/market_engine/fundamental_evidence_coverage_runs/"
    "me-data06-fundamental-evidence-coverage-review-fix-20260718T113254Z"
)
DEFAULT_BASELINE_RUN31_EVIDENCE = Path(
    "artifacts/market_engine/run_evidence/me-run31-after-me-data06-review-fix-20260718T113254Z"
)
DEFAULT_OPERATOR_IMPORT_PATH = Path(
    "operator_input/market_engine/me-data07/fundamental_metrics.json"
)
DEFAULT_RAW_SNAPSHOT_ROOT = Path(
    "data/market_engine/source_snapshots/fundamental_metrics"
)
SUPPORTED_SOURCE_MODES = frozenset({"inventory_only", "operator_import", "approved_acquisition"})
SUPPORTED_BATCH_TIERS = frozenset({"pilot", "expanded", "full"})
ALLOWED_METRIC_UNITS = frozenset({"ratio", "percent"})
SUCCESS_SOURCING_STATUSES = frozenset({"complete", "partial"})
BLOCKED_SOURCING_STATUSES = frozenset(
    {
        "blocked_no_source",
        "blocked_missing_credentials",
        "blocked_mapping",
        "blocked_provider_coverage",
        "blocked_missing_metric",
        "blocked_invalid_payload",
        "blocked_stale",
        "blocked_period_mismatch",
        "blocked_conflict",
        "unsupported",
    }
)
FAILED_SOURCING_STATUSES = frozenset({"failed_request", "failed_validation"})
PENDING_SOURCING_STATUSES = frozenset({"selected"})
TERMINAL_RUN_STATUSES = frozenset(
    {
        "blocked_external_source_requirement",
        "failed_validation",
        "completed_import_without_downstream",
        "completed_with_coverage_measurement",
    }
)
OUTPUT_NAMES = (
    "manifest",
    "source_approval_decision",
    "concrete_source_approval_validation",
    "metric_gap_analysis",
    "sourcing_plan",
    "fundamental_source_symbol_mapping",
    "batch_execution_summary",
    "per_ticker_sourcing_status",
    "metric_validation_summary",
    "normalized_metric_evidence",
    "coverage_before_after",
    "blocker_report",
)


class ValidatedFundamentalMetricSourcingError(ValueError):
    pass


def run_validated_fundamental_metric_sourcing(
    *,
    run_id: str,
    source_mode: str,
    batch_tier: str,
    as_of_date: str = DEFAULT_AS_OF_DATE,
    canonical_universe: str | Path = DEFAULT_CANONICAL_CONFIG,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
    baseline_data06_run: str | Path = DEFAULT_BASELINE_DATA06_RUN,
    baseline_run31_evidence: str | Path = DEFAULT_BASELINE_RUN31_EVIDENCE,
    operator_import_path: str | Path | None = DEFAULT_OPERATOR_IMPORT_PATH,
    source_approval_decision_path: str | Path | None = None,
    source_document_root: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    execute_downstream: bool = False,
    data06_run_id: str | None = None,
    run31_run_id: str | None = None,
    data06_output_root: str | Path | None = None,
    run31_output_root: str | Path | None = None,
    allow_overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    if source_mode not in SUPPORTED_SOURCE_MODES:
        raise ValidatedFundamentalMetricSourcingError(f"unsupported source mode: {source_mode}")
    if batch_tier not in SUPPORTED_BATCH_TIERS:
        raise ValidatedFundamentalMetricSourcingError(f"unsupported batch tier: {batch_tier}")
    as_of = _parse_date(as_of_date, "as_of_date")
    output_dir = Path(output_root) / run_id
    if output_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"ME-DATA07 output directory already exists: {output_dir}")

    universe = build_universe_snapshot(canonical_universe, price_history_root=price_history_root)
    instruments = sorted(
        universe.get("instruments") or [],
        key=lambda row: (str(row.get("symbol") or ""), str(row.get("instrument_id") or "")),
    )
    baseline = _load_baseline_data06(Path(baseline_data06_run), instruments, universe)
    run31 = _load_run31_per_ticker(Path(baseline_run31_evidence), instruments, universe)
    mappings = _build_symbol_mappings(instruments, canonical_universe)
    approval = _source_approval_decision(source_mode, operator_import_path)
    per_ticker, gap = _build_gap_analysis(instruments, baseline, run31, mappings)
    plan = _build_sourcing_plan(per_ticker, batch_tier)
    selected = _selected_tickers(per_ticker, batch_tier)

    normalized_records: list[dict[str, Any]] = []
    validation = _empty_validation_summary()
    snapshot: dict[str, Any] | None = None
    execution_status = "not_executed"
    execution_reason = "inventory_only_mode"
    import_attempted = False
    concrete_approval = {
        "schema_version": "market-engine-data09-source-approval-validation-v1",
        "validation_status": "not_applicable",
        "concrete_package_source_approved": False,
        "reason_codes": [],
        "issues": [],
    }
    if source_mode == "operator_import":
        import_path = Path(operator_import_path) if operator_import_path else None
        if import_path is None or not import_path.exists():
            execution_status = "blocked"
            execution_reason = "operator_import_package_missing"
        elif approval["approval_status"] != "approved_operator_supplied_import":
            execution_status = "blocked"
            execution_reason = "source_approval_failed"
        else:
            concrete_approval = validate_source_approval_decision(
                source_approval_decision_path,
                import_path,
                source_document_root=source_document_root,
            )
            if not concrete_approval["concrete_package_source_approved"]:
                execution_status = "blocked"
                execution_reason = "concrete_source_approval_failed"
            else:
                import_attempted = True
                normalized_records, validation = _load_and_validate_operator_import(
                    import_path,
                    mappings={row["ticker"]: row for row in mappings},
                    instruments={str(row["symbol"]).upper(): row for row in instruments},
                    as_of=as_of,
                    allowed_tickers=set(selected),
                )
                if validation["validation_status"] == "passed":
                    snapshot = _persist_operator_snapshot(
                        import_path,
                        run_id=run_id,
                        raw_snapshot_root=Path(raw_snapshot_root),
                        normalized_records=normalized_records,
                        allow_overwrite=allow_overwrite,
                    )
                    execution_status = "completed"
                    execution_reason = "operator_import_validated"
                else:
                    execution_status = "failed_validation"
                    execution_reason = "operator_import_validation_failed"
    elif source_mode == "approved_acquisition":
        execution_status = "blocked"
        execution_reason = "no_approved_mvp_fundamental_provider"

    imported_by_ticker = {row["ticker"]: row for row in normalized_records}
    _apply_execution_status(
        per_ticker,
        selected,
        imported_by_ticker,
        execution_status,
        execution_reason,
        failed_tickers=set(validation.get("failed_tickers") or []),
    )
    preliminary_run_status = _run_status(source_mode, execution_status, None)
    status_reconciliation = _reconcile_sourcing_status_counts(
        per_ticker,
        selected,
        run_status=preliminary_run_status,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_csv_path: Path | None = None
    if normalized_records:
        normalized_csv_path = output_dir / "normalized_fundamental_metrics.csv"
        _write_normalized_csv(normalized_csv_path, normalized_records)

    downstream: dict[str, Any] | None = None
    if execute_downstream and normalized_csv_path is not None and validation["validation_status"] == "passed":
        data06_id = data06_run_id or f"{run_id}-me-data06"
        run31_id = run31_run_id or f"{run_id}-me-run31"
        downstream_baseline = (
            (baseline["manifest"].get("input_paths") or {}).get("baseline_run_evidence")
            or baseline_run31_evidence
        )
        downstream_artifacts, downstream_dir = run_fundamental_evidence_coverage(
            run_id=data06_id,
            existing_fundamental_path=Path(baseline_data06_run) / "normalized_fundamental_quality.csv",
            raw_fundamentals_path=normalized_csv_path,
            baseline_run_evidence=downstream_baseline,
            canonical_universe=canonical_universe,
            price_history_root=price_history_root,
            technical_screening_artifact=DEFAULT_TECHNICAL_SCREENING_ARTIFACT,
            output_root=data06_output_root or Path("artifacts/market_engine/fundamental_evidence_coverage_runs"),
            run31_output_root=run31_output_root or Path("artifacts/market_engine/full_advice_readiness_runs"),
            run31_run_id=run31_id,
            allow_overwrite=allow_overwrite,
        )
        downstream = {
            "data06_run_id": data06_id,
            "run31_run_id": run31_id,
            "data06_output_dir": downstream_dir.as_posix(),
            "data06_run_status": downstream_artifacts["manifest"]["run_status"],
            "after": downstream_artifacts["fundamental_coverage_summary"]["after"],
            "historical_origin_comparison": {
                **downstream_artifacts["before_after_comparison"],
                "attributable_to_current_sprint": False,
            },
        }

    before_counts = dict(baseline["counts"])
    after_counts = dict(downstream["after"]) if downstream else dict(before_counts)
    run_status = _run_status(source_mode, execution_status, downstream)
    blocker_report = _blocker_report(per_ticker, execution_reason, run_status, status_reconciliation)
    batch_summary = _batch_execution_summary(
        source_mode=source_mode,
        batch_tier=batch_tier,
        selected=selected,
        normalized_records=normalized_records,
        validation=validation,
        execution_status=execution_status,
        execution_reason=execution_reason,
        import_attempted=import_attempted,
        status_reconciliation=status_reconciliation,
    )
    absolute_delta = {
        key: after_counts.get(key, 0) - before_counts.get(key, 0)
        for key in (
            "fundamental_complete",
            "fundamental_partial",
            "fundamental_missing",
            "invalid_stale_conflicting",
            "canonical_advice_input_ready",
            "full_advice_ready",
            "unable_to_advise",
        )
    }
    imported_ticker = normalized_records[0]["ticker"] if len(normalized_records) == 1 else None
    before_ticker = baseline["by_ticker"].get(imported_ticker, {}) if imported_ticker else {}
    after_ticker = None
    if imported_ticker and downstream:
        after_rows = downstream_artifacts["per_ticker_fundamental_status"]["tickers"]
        after_ticker = next((row for row in after_rows if row.get("ticker") == imported_ticker), None)
    ticker_delta = None
    if imported_ticker:
        imported_record = normalized_records[0]
        ticker_delta = {
            "ticker": imported_ticker,
            "before_status": before_ticker.get("overall_fundamental_status"),
            "after_status": (after_ticker or {}).get("overall_fundamental_status", before_ticker.get("overall_fundamental_status")),
            "new_metrics": sorted(
                metric
                for metric, value in (imported_record.get("metrics") or {}).items()
                if value is not None
            ),
            "remaining_missing_metrics": sorted(imported_record.get("missing_metrics") or []),
            "advice_input_ready_before": bool((run31["by_ticker"].get(imported_ticker) or {}).get("canonical_advice_input_ready")),
            "advice_input_ready_after": bool((after_ticker or {}).get("canonical_advice_input_ready", False)),
        }
    coverage = {
        "schema_version": "market-engine-data07-coverage-before-after-v1",
        "run_status": run_status,
        "before": before_counts,
        "after": after_counts,
        "current_sprint_comparison": {
            "before": before_counts,
            "after": after_counts,
            "absolute_delta": absolute_delta,
        },
        "ticker_delta": ticker_delta,
        "historical_origin_comparison": downstream.get("historical_origin_comparison") if downstream else None,
        "downstream_executed": downstream is not None,
        "downstream_run_identity": {
            key: downstream[key]
            for key in ("data06_run_id", "run31_run_id", "data06_output_dir", "data06_run_status")
        } if downstream else None,
        "coverage_claim": "measured_downstream_result" if downstream else "no_coverage_change_claimed",
    }
    artifacts = {
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "market-engine-data07-validated-fundamental-metric-sourcing-run",
            "sprint_id": "ME-DATA07",
            "run_id": run_id,
            "run_status": run_status,
            "generated_at": _utc_now(),
            "git_commit": _git_commit(),
            "source_mode": source_mode,
            "batch_tier": batch_tier,
            "as_of_date": as_of_date,
            "mvp_metric_contract": {
                "required_metrics": list(MVP_METRIC_FIELDS),
                "freshness_max_age_days": FRESHNESS_MAX_AGE_DAYS,
                "completeness_rule": "all required metrics valid, current, same-period, non-conflicting, and provenance-backed",
                "normalized_unit": "ratio",
                "growth_period_rule": "source-reported comparable year-over-year periods only; no derivation or annualization",
                "debt_to_equity_rule": "source-reported total-debt-to-total-equity ratio only; no synthetic derivation",
            },
            "canonical_universe_version": universe.get("universe_version"),
            "canonical_universe_size": len(instruments),
            "input_paths": {
                "canonical_universe": Path(canonical_universe).as_posix(),
                "price_history_root": Path(price_history_root).as_posix(),
                "baseline_data06_run": Path(baseline_data06_run).as_posix(),
                "baseline_run31_evidence": Path(baseline_run31_evidence).as_posix(),
                "operator_import_path": Path(operator_import_path).as_posix() if operator_import_path else None,
                "source_approval_decision_path": Path(source_approval_decision_path).as_posix() if source_approval_decision_path else None,
                "source_document_root_supplied": source_document_root is not None,
            },
            "input_checksums": _input_checksums(
                Path(canonical_universe),
                Path(baseline_data06_run) / "per_ticker_fundamental_status.json",
                Path(baseline_data06_run) / "manifest.json",
                _run31_input_index_path(Path(baseline_run31_evidence)),
                Path(run31["index_path"]),
                Path(operator_import_path) if operator_import_path else None,
                Path(source_approval_decision_path) if source_approval_decision_path else None,
            ),
            "raw_snapshot": snapshot,
            "downstream": downstream,
            "sourcing_status_reconciliation": status_reconciliation,
            "guardrails": {
                "provider_calls_performed": False,
                "network_access_performed": False,
                "credentials_logged": False,
                "model_invocation_performed": False,
                "broker_order_execution_performed": False,
                "allocation_performed": False,
                "portfolio_watchlist_mutation_performed": False,
                "telegram_delivery_performed": False,
                "decision_engine_changed": False,
                "recommendation_rules_changed": False,
            },
            "outputs": {
                **{name: f"{name}.json" for name in OUTPUT_NAMES},
                "coverage_report": "coverage_report.md",
                "normalized_metric_csv": "normalized_fundamental_metrics.csv" if normalized_csv_path else None,
            },
        },
        "source_approval_decision": approval,
        "concrete_source_approval_validation": concrete_approval,
        "metric_gap_analysis": gap,
        "sourcing_plan": plan,
        "fundamental_source_symbol_mapping": {
            "schema_version": "market-engine-data07-fundamental-source-symbol-mapping-v1",
            "ticker_count": len(mappings),
            "mappings": mappings,
        },
        "batch_execution_summary": batch_summary,
        "per_ticker_sourcing_status": {
            "schema_version": "market-engine-data07-per-ticker-sourcing-status-v1",
            "ticker_count": len(per_ticker),
            "tickers": per_ticker,
        },
        "metric_validation_summary": validation,
        "normalized_metric_evidence": {
            "schema_version": "market-engine-data07-normalized-metric-evidence-v1",
            "record_count": len(normalized_records),
            "records": normalized_records,
        },
        "coverage_before_after": coverage,
        "blocker_report": blocker_report,
    }
    _write_artifacts(output_dir, artifacts)
    (output_dir / "coverage_report.md").write_text(_render_report(artifacts), encoding="utf-8")
    return artifacts, output_dir


def _load_baseline_data06(root: Path, instruments: Sequence[Mapping[str, Any]], universe: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_json_required(root / "manifest.json")
    payload = _read_json_required(root / "per_ticker_fundamental_status.json")
    rows = payload.get("tickers")
    if not isinstance(rows, list):
        raise ValidatedFundamentalMetricSourcingError("baseline ME-DATA06 per-ticker status is missing")
    expected = {str(row["symbol"]).upper() for row in instruments}
    by_ticker = _unique_by_ticker(rows, key="ticker", label="baseline ME-DATA06")
    if set(by_ticker) != expected:
        raise ValidatedFundamentalMetricSourcingError("baseline ME-DATA06 ticker identity mismatch")
    canonical = manifest.get("canonical_universe") or {}
    if canonical.get("universe_version") != universe.get("universe_version") or canonical.get("total_instruments") != len(instruments):
        raise ValidatedFundamentalMetricSourcingError("baseline ME-DATA06 universe mismatch")
    counts = Counter(str(row.get("overall_fundamental_status") or "missing") for row in rows)
    after = ((root / "fundamental_coverage_summary.json"))
    summary = _read_json_required(after)
    declared = summary.get("after") or {}
    result_counts = {
        "fundamental_complete": counts["complete"],
        "fundamental_partial": counts["partial"],
        "fundamental_missing": counts["missing"],
        "invalid_stale_conflicting": sum(counts[key] for key in ("invalid", "stale", "conflicting")),
        "canonical_advice_input_ready": declared.get("canonical_advice_input_ready"),
        "full_advice_ready": declared.get("full_advice_ready"),
        "unable_to_advise": declared.get("unable_to_advise"),
    }
    if sum(counts.values()) != len(instruments):
        raise ValidatedFundamentalMetricSourcingError("baseline ME-DATA06 counts do not reconcile")
    return {"manifest": manifest, "by_ticker": by_ticker, "counts": result_counts}


def _load_run31_per_ticker(path: Path, instruments: Sequence[Mapping[str, Any]], universe: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_dir() and (path / "evidence_coverage_index.json").exists():
        index_path = path / "evidence_coverage_index.json"
        manifest_path = path / "manifest.json"
    else:
        compact_path = path / "run_evidence_index.json" if path.is_dir() else path
        compact = _read_json_required(compact_path)
        full_path = Path(str((compact.get("full_artifact") or {}).get("local_path") or ""))
        index_path = full_path / "evidence_coverage_index.json"
        manifest_path = full_path / "manifest.json"
    index = _read_json_required(index_path)
    manifest = _read_json_required(manifest_path)
    rows = index.get("instruments")
    if not isinstance(rows, list):
        raise ValidatedFundamentalMetricSourcingError("ME-RUN31 per-ticker evidence is missing")
    by_ticker = _unique_by_ticker(rows, key="symbol", label="baseline ME-RUN31")
    expected = {str(row["symbol"]).upper() for row in instruments}
    if set(by_ticker) != expected or manifest.get("canonical_universe_version") != universe.get("universe_version"):
        raise ValidatedFundamentalMetricSourcingError("baseline ME-RUN31 universe identity mismatch")
    return {"manifest": manifest, "index_path": index_path.as_posix(), "by_ticker": by_ticker}


def _build_symbol_mappings(instruments: Sequence[Mapping[str, Any]], canonical_path: str | Path) -> list[dict[str, Any]]:
    config = _read_json_required(Path(canonical_path))
    override_config = config.get("symbol_overrides") or []
    if isinstance(override_config, str):
        overrides_payload = _read_json_required(Path(override_config))
        override_config = overrides_payload.get("overrides") or []
    if not isinstance(override_config, list):
        raise ValidatedFundamentalMetricSourcingError("canonical symbol overrides must be a list or JSON path")
    overrides = {str(row["canonical_symbol"]).upper(): row for row in override_config}
    rows = []
    for instrument in instruments:
        ticker = str(instrument["symbol"]).upper()
        asset_type = str(instrument.get("asset_type") or "unknown")
        override = overrides.get(ticker)
        candidates = [str(value).strip() for value in instrument.get("provider_symbol_candidates") or [] if str(value).strip()]
        if asset_type != "equity":
            status, provider_symbol, reason = "unsupported_asset_type", None, "MVP company fundamentals do not apply to ETFs"
        elif instrument.get("duplicate_listing"):
            status, provider_symbol, reason = "rejected_duplicate_listing", None, "duplicate listing is not a canonical sourcing target"
        elif len(set(candidates)) > 1:
            status, provider_symbol, reason = "ambiguous", None, "multiple provider symbols require operator validation"
        elif instrument.get("provider_symbol_required") and not candidates and not override:
            status, provider_symbol, reason = "missing_provider_symbol", None, "required provider symbol is absent"
        elif override and str(override.get("mapping_status") or override.get("source_mapping_status") or "") == "unsupported":
            status, provider_symbol, reason = "unsupported_exchange", override.get("source_symbol"), str(override.get("reason") or "unsupported mapping")
        elif override:
            status, provider_symbol, reason = "mapped_with_explicit_alias", override.get("source_symbol"), str(override.get("reason") or "explicit alias")
        elif candidates:
            status, provider_symbol, reason = "mapped_with_explicit_alias", candidates[0], "explicit instrument provider symbol"
        else:
            status, provider_symbol, reason = "mapped", ticker, "exact canonical ticker mapping"
        rows.append(
            {
                "ticker": ticker,
                "instrument_id": instrument.get("instrument_id"),
                "asset_type": asset_type,
                "exchange": instrument.get("exchange"),
                "region": instrument.get("country"),
                "provider_symbol": provider_symbol,
                "mapping_status": status,
                "share_class": "class_share" if "." in ticker or "-" in ticker else "standard",
                "listing_form": "adr_or_cross_listed" if ticker in {"ASML", "TSM"} else "primary_or_unspecified",
                "reason": reason,
            }
        )
    return rows


def _source_approval_decision(source_mode: str, operator_import_path: str | Path | None) -> dict[str, Any]:
    credentials = [name for name in ("SEC_API_KEY", "FUNDAMENTAL_PROVIDER_API_KEY") if os.environ.get(name)]
    routes = [
        {
            "route": "existing_local_me_data06_evidence",
            "repository_support": "operational",
            "decision": "approved_existing_local_acquisition",
            "limitation": "already consumed; does not expand the remaining coverage",
        },
        {
            "route": "sec_companyfacts_bounded_smoke_adapter",
            "repository_support": "bounded smoke only",
            "decision": "blocked_unsupported_metric_contract",
            "limitation": "not approved as broad provider and does not supply all five MVP ratios",
        },
        {
            "route": "automated_cached_source_acquisition",
            "repository_support": "deterministic company_profile adapter only",
            "decision": "blocked_unsupported_metric_contract",
            "limitation": "company profiles are not fundamental-quality evidence",
        },
        {
            "route": "operator_supplied_import",
            "repository_support": "local import/staging precedent exists; ME-DATA07 adds metric validator",
            "decision": "approved_operator_supplied_import",
            "limitation": "requires an operator package with primary-source provenance",
        },
        {
            "route": "third_party_fundamental_provider",
            "repository_support": "no concrete approved adapter or provider contract",
            "decision": "blocked_no_approved_source",
            "limitation": "provider identity, licensing, credentials, mappings, and metric definitions are absent",
        },
        {
            "route": "credentialed_provider_acquisition",
            "repository_support": "no approved full-contract adapter or credential boundary",
            "decision": "blocked_missing_credentials",
            "limitation": "credentials cannot authorize a provider route that has not first passed source approval",
        },
        {
            "route": "unsupported_region_or_exchange",
            "repository_support": "canonical mappings explicitly fail closed where source identity is unsupported",
            "decision": "blocked_unsupported_region",
            "limitation": "no automatic cross-listing substitution is permitted",
        },
    ]
    approval_status = {
        "operator_import": "approved_operator_supplied_import",
        "inventory_only": "blocked_no_approved_source",
        "approved_acquisition": "blocked_no_approved_source",
    }[source_mode]
    return {
        "schema_version": "market-engine-data07-source-approval-decision-v1",
        "source_mode": source_mode,
        "routes_inspected": routes,
        "provider_identity": "operator-supplied primary-source package" if source_mode == "operator_import" else None,
        "use_purpose": "MVP fundamental metric evidence only",
        "allowed_metric_families": list(MVP_METRIC_FIELDS),
        "authentication_method": "none for local operator import",
        "credentials_status": "not_required_for_operator_import" if source_mode == "operator_import" else ("configured" if credentials else "not_configured"),
        "rate_limit": {"requests": 0, "retries": 0, "timeout_seconds": 0, "budget": 0},
        "provenance_quality": "primary-source reference and checksum required",
        "freshness_capability": f"source_date validated against {FRESHNESS_MAX_AGE_DAYS}-day threshold",
        "ticker_coverage": "canonical mapped equities supplied by operator",
        "region_exchange_support": "only unambiguous canonical mappings",
        "licensing_constraints": "no broad provider license is documented; operator must supply evidence with permitted local use",
        "local_persistence": "immutable run-scoped raw snapshot plus checksum manifest after validation",
        "approval_status": approval_status,
        "rejection_reason": None if approval_status.startswith("approved_") else "no approved provider route for the complete MVP metric contract",
        "chosen_source_path": Path(operator_import_path).as_posix() if operator_import_path else None,
        "secret_names_detected": [],
        "secret_values_persisted": False,
    }


def _build_gap_analysis(
    instruments: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    run31: Mapping[str, Any],
    mappings: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping_by_ticker = {row["ticker"]: row for row in mappings}
    rows = []
    for instrument in instruments:
        ticker = str(instrument["symbol"]).upper()
        fundamental = baseline["by_ticker"][ticker]
        readiness = run31["by_ticker"][ticker]
        missing = sorted(set(str(item) for item in fundamental.get("missing_metrics") or MVP_METRIC_FIELDS))
        status = str(fundamental.get("overall_fundamental_status") or "missing")
        if status == "complete":
            missing = []
        mapping = mapping_by_ticker[ticker]
        technical_eligible = bool((readiness.get("technical_screening") or {}).get("ranking_eligible"))
        other_context_ready = (
            (readiness.get("technical_screening") or {}).get("status") == "available"
            and (readiness.get("market_context") or {}).get("status") == "available"
            and (readiness.get("portfolio_context") or {}).get("status") in {"available", "not_applicable"}
            and ((readiness.get("technical_screening") or {}).get("setup_price_market_context") or {}).get("context_status") in {"available", "partial"}
        )
        source_eligible = mapping["mapping_status"] in {"mapped", "mapped_with_explicit_alias"} and status != "complete"
        if source_eligible and technical_eligible and other_context_ready:
            tier = "tier_1"
            reason = "technical candidate with fundamental context as the remaining evidence-readiness gap"
        elif source_eligible and status == "partial":
            tier = "tier_2"
            reason = "existing partial fundamental context"
        elif source_eligible:
            tier = "tier_3"
            reason = "broad canonical equity coverage"
        else:
            tier = "not_selected"
            reason = mapping["reason"] if status != "complete" else "fundamental context already complete"
        rows.append(
            {
                "ticker": ticker,
                "instrument_id": instrument.get("instrument_id"),
                "name": instrument.get("name"),
                "asset_type": instrument.get("asset_type"),
                "sector": instrument.get("sector"),
                "market_region": instrument.get("country"),
                "exchange": instrument.get("exchange"),
                "overall_fundamental_status": status,
                "selected_evidence_source": fundamental.get("selected_source_family"),
                "missing_mvp_metrics": missing,
                "source_date": fundamental.get("source_date"),
                "freshness": _freshness_label(fundamental.get("source_date"), status),
                "canonical_advice_input_ready": bool(readiness.get("canonical_advice_input_ready")),
                "technical_eligible": technical_eligible,
                "canonical_advice_input_blocker": "fundamental_context" if status != "complete" else None,
                "mapping_status": mapping["mapping_status"],
                "provider_symbol": mapping["provider_symbol"],
                "share_class": mapping["share_class"],
                "listing_form": mapping["listing_form"],
                "sourcing_eligible": source_eligible,
                "sourcing_tier": tier,
                "sourcing_reason": reason,
                "sourcing_status": "not_selected",
                "execution_issues": [],
            }
        )
    metric_counts = Counter(metric for row in rows for metric in row["missing_mvp_metrics"])
    combinations = Counter("|".join(row["missing_mvp_metrics"]) or "none" for row in rows)
    gap = {
        "schema_version": "market-engine-data07-metric-gap-analysis-v1",
        "canonical_universe_size": len(rows),
        "detail_artifact": "per_ticker_sourcing_status.json",
        "status_counts": dict(sorted(Counter(row["overall_fundamental_status"] for row in rows).items())),
        "missing_metric_counts": {metric: metric_counts[metric] for metric in MVP_METRIC_FIELDS},
        "missing_metric_combinations": dict(sorted(combinations.items())),
        "aggregations": {
            "market_region": dict(sorted(Counter(str(row["market_region"] or "unknown") for row in rows).items())),
            "asset_type": dict(sorted(Counter(str(row["asset_type"] or "unknown") for row in rows).items())),
            "source_family": dict(sorted(Counter(str(row["selected_evidence_source"] or "none") for row in rows).items())),
            "mapping_status": dict(sorted(Counter(row["mapping_status"] for row in rows).items())),
            "sourcing_eligibility": dict(sorted(Counter("eligible" if row["sourcing_eligible"] else "ineligible" for row in rows).items())),
            "technical_candidate_status": dict(sorted(Counter("eligible" if row["technical_eligible"] else "not_eligible" for row in rows).items())),
            "canonical_advice_input_blocker": dict(sorted(Counter(str(row["canonical_advice_input_blocker"] or "none") for row in rows).items())),
            "sourcing_tier": dict(sorted(Counter(row["sourcing_tier"] for row in rows).items())),
        },
    }
    return rows, gap


def _build_sourcing_plan(rows: Sequence[Mapping[str, Any]], batch_tier: str) -> dict[str, Any]:
    tiers = {
        tier: sorted(row["ticker"] for row in rows if row["sourcing_tier"] == tier)
        for tier in ("tier_1", "tier_2", "tier_3")
    }
    pilot_candidates = _pilot_tickers(rows)
    return {
        "schema_version": "market-engine-data07-sourcing-plan-v1",
        "priority_rules": {
            "tier_1": "technical candidates whose remaining evidence-readiness gap is fundamental context",
            "tier_2": "remaining partial fundamental contexts",
            "tier_3": "remaining mapped canonical equities",
        },
        "batch_tier": batch_tier,
        "tier_counts": {key: len(value) for key, value in tiers.items()},
        "pilot_tickers": pilot_candidates,
        "selection_count": len(_selected_tickers(rows, batch_tier)),
        "request_limits": {"pilot": 12, "expanded": 100, "full": len(rows), "retries": 0, "timeout_seconds": 0, "budget": 0},
    }


def _selected_tickers(rows: Sequence[Mapping[str, Any]], batch_tier: str) -> list[str]:
    if batch_tier == "pilot":
        return _pilot_tickers(rows)
    ordered = [row["ticker"] for tier in ("tier_1", "tier_2", "tier_3") for row in rows if row["sourcing_tier"] == tier]
    limit = 100 if batch_tier == "expanded" else len(ordered)
    return ordered[:limit]


def _pilot_tickers(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    eligible = sorted(
        (row for row in rows if row["sourcing_tier"] in {"tier_1", "tier_2", "tier_3"}),
        key=lambda row: ({"tier_1": 1, "tier_2": 2, "tier_3": 3}[row["sourcing_tier"]], row["ticker"]),
    )
    selected: list[str] = []

    def take(candidates: Sequence[Mapping[str, Any]], limit: int) -> None:
        for row in candidates:
            if row["ticker"] not in selected:
                selected.append(row["ticker"])
            if len(selected) >= limit:
                return

    partial = [row for row in eligible if row["overall_fundamental_status"] == "partial"]
    missing = [row for row in eligible if row["overall_fundamental_status"] == "missing"]
    take(partial, 3)
    take(missing, 7)
    take([row for row in eligible if row["market_region"] not in {None, "US"}], 8)
    take([row for row in eligible if row["mapping_status"] == "mapped_with_explicit_alias"], 9)
    take(eligible, 12)
    return selected


def _load_and_validate_operator_import(
    path: Path,
    *,
    mappings: Mapping[str, Mapping[str, Any]],
    instruments: Mapping[str, Mapping[str, Any]],
    as_of: date,
    allowed_tickers: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_json_required(path)
    issues = []
    if payload.get("schema_version") != OPERATOR_IMPORT_SCHEMA_VERSION:
        issues.append("unsupported_import_schema")
    records = payload.get("records")
    if not isinstance(records, list):
        records = []
        issues.append("records_missing")
    normalized = []
    seen: dict[tuple[str, str], str] = {}
    metric_counts: Counter[str] = Counter()
    duplicate_records_deduplicated = 0
    excluded_unselected_count = 0
    source_package_checksum = _sha256(path)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            issues.append(f"record_{index}_invalid")
            continue
        ticker = str(record.get("ticker") or "").upper()
        if ticker not in instruments:
            issues.append(f"record_{index}_unknown_ticker")
            continue
        if allowed_tickers is not None and ticker not in allowed_tickers:
            excluded_unselected_count += 1
            continue
        mapping = mappings[ticker]
        if mapping["mapping_status"] not in {"mapped", "mapped_with_explicit_alias"}:
            issues.append(f"{ticker}_mapping_blocked")
            continue
        if str(record.get("instrument_id") or "") != str(instruments[ticker].get("instrument_id") or ""):
            issues.append(f"{ticker}_instrument_identity_mismatch")
            continue
        if str(record.get("provider_symbol") or "") != str(mapping.get("provider_symbol") or ""):
            issues.append(f"{ticker}_provider_symbol_mismatch")
            continue
        try:
            source_date = _parse_date(record.get("source_date"), f"{ticker}.source_date")
        except ValidatedFundamentalMetricSourcingError:
            issues.append(f"{ticker}_invalid_source_date")
            continue
        if source_date > as_of:
            issues.append(f"{ticker}_future_dated")
            continue
        if (as_of - source_date).days > FRESHNESS_MAX_AGE_DAYS:
            issues.append(f"{ticker}_stale")
            continue
        period = str(record.get("reporting_period") or "").strip()
        provider = str(record.get("provider") or "").strip()
        reference = str(record.get("source_reference") or "").strip()
        parser_version = str(record.get("parser_version") or "").strip()
        if not period or not provider or not reference or not parser_version:
            issues.append(f"{ticker}_provenance_incomplete")
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            issues.append(f"{ticker}_metrics_missing")
            continue
        normalized_metrics: dict[str, float | None] = {}
        lineage = []
        record_invalid = False
        for metric in MVP_METRIC_FIELDS:
            evidence = metrics.get(metric)
            if evidence is None:
                normalized_metrics[metric] = None
                continue
            if not isinstance(evidence, Mapping):
                issues.append(f"{ticker}_{metric}_invalid_evidence")
                record_invalid = True
                continue
            raw_value = evidence.get("value")
            if raw_value is None:
                normalized_metrics[metric] = None
                continue
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                issues.append(f"{ticker}_{metric}_invalid_numeric")
                record_invalid = True
                continue
            unit = str(evidence.get("unit") or "")
            if unit not in ALLOWED_METRIC_UNITS:
                issues.append(f"{ticker}_{metric}_invalid_unit")
                record_invalid = True
                continue
            metric_period = str(evidence.get("reporting_period") or period).strip()
            if metric_period != period:
                issues.append(f"{ticker}_{metric}_reporting_period_mismatch")
                record_invalid = True
                continue
            normalized_value = float(raw_value) / 100.0 if unit == "percent" else float(raw_value)
            normalized_metrics[metric] = normalized_value
            metric_counts[metric] += 1
            lineage.append(
                {
                    "canonical_metric": metric,
                    "raw_source_field": evidence.get("raw_source_field") or metric,
                    "raw_value": raw_value,
                    "raw_unit": unit,
                    "normalized_value": normalized_value,
                    "normalized_unit": "ratio",
                    "reporting_period": period,
                    "source_date": source_date.isoformat(),
                    "provider": provider,
                    "snapshot_id": record.get("snapshot_id"),
                    "source_package_checksum": source_package_checksum,
                    "transformation": "percent_to_ratio" if unit == "percent" else "identity",
                    "validation_status": "valid",
                }
            )
        if record_invalid:
            continue
        fingerprint = json.dumps(normalized_metrics, sort_keys=True)
        key = (ticker, period)
        if key in seen and seen[key] != fingerprint:
            issues.append(f"{ticker}_conflicting_duplicate_period")
            continue
        if key in seen:
            duplicate_records_deduplicated += 1
            continue
        seen[key] = fingerprint
        missing = [metric for metric in MVP_METRIC_FIELDS if normalized_metrics.get(metric) is None]
        normalized_row = {
                "ticker": ticker,
                "instrument_id": instruments[ticker]["instrument_id"],
                "provider": provider,
                "provider_symbol": mapping["provider_symbol"],
                "exchange": record.get("exchange") or instruments[ticker].get("exchange"),
                "region": record.get("region") or instruments[ticker].get("country"),
                "acquired_at": record.get("acquired_at"),
                "source_date": source_date.isoformat(),
                "reporting_period": period,
                "source_reference": reference,
                "parser_version": parser_version,
                "source_package_checksum": source_package_checksum,
                "snapshot_id": record.get("snapshot_id"),
                "metrics": normalized_metrics,
                "metric_lineage": lineage,
                "missing_metrics": missing,
                "coverage_status": "complete" if not missing else "partial",
                "validation_status": "valid",
            }
        normalized_row["normalized_record_checksum"] = hashlib.sha256(
            json.dumps(normalized_row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        normalized.append(normalized_row)
    normalized.sort(key=lambda row: row["ticker"])
    failed_tickers = sorted(
        ticker
        for ticker in instruments
        if any(issue.startswith(f"{ticker}_") for issue in issues)
    )
    return normalized, {
        "schema_version": "market-engine-data07-metric-validation-summary-v1",
        "validation_status": "passed" if not issues else "failed",
        "input_record_count": len(records),
        "normalized_record_count": len(normalized),
        "complete_count": sum(row["coverage_status"] == "complete" for row in normalized),
        "partial_count": sum(row["coverage_status"] == "partial" for row in normalized),
        "duplicate_records_deduplicated": duplicate_records_deduplicated,
        "excluded_unselected_count": excluded_unselected_count,
        "failed_tickers": failed_tickers,
        "validation_failed_record_count": len(failed_tickers),
        "package_validation_failed": bool(issues),
        "metric_counts": {metric: metric_counts[metric] for metric in MVP_METRIC_FIELDS},
        "issues": sorted(set(issues)),
        "null_preserved": True,
        "negative_values_allowed": True,
        "deterministic_ordering": True,
    }


def _persist_operator_snapshot(
    source_path: Path,
    *,
    run_id: str,
    raw_snapshot_root: Path,
    normalized_records: Sequence[Mapping[str, Any]],
    allow_overwrite: bool,
) -> dict[str, Any]:
    destination = raw_snapshot_root / run_id
    if destination.exists() and not allow_overwrite:
        raise FileExistsError(f"ME-DATA07 raw snapshot directory already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "operator_import.json"
    shutil.copyfile(source_path, raw_path)
    checksum = _sha256(raw_path)
    manifest = {
        "schema_version": "market-engine-data07-raw-fundamental-metric-snapshot-v1",
        "run_id": run_id,
        "source_mode": "operator_import",
        "raw_path": raw_path.as_posix(),
        "raw_checksum": checksum,
        "persisted_at": _utc_now(),
        "request_identity": {"mode": "operator_import", "request_count": 0},
        "record_count": len(normalized_records),
        "tickers": sorted(row["ticker"] for row in normalized_records),
        "parser_version": "market-engine-data07-operator-import-parser-v1",
        "status": "validated",
        "provenance": "operator-supplied package; source_reference retained per record",
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _apply_execution_status(
    rows: list[dict[str, Any]],
    selected: Sequence[str],
    imported: Mapping[str, Mapping[str, Any]],
    execution_status: str,
    execution_reason: str,
    *,
    failed_tickers: set[str] | None = None,
) -> None:
    selected_set = set(selected)
    failed_ticker_set = failed_tickers or set()
    for row in rows:
        ticker = row["ticker"]
        if ticker not in selected_set:
            row["sourcing_status"] = "not_selected"
        elif execution_status == "failed_validation" and ticker in failed_ticker_set:
            row["sourcing_status"] = "failed_validation"
            row["execution_issues"] = [execution_reason]
        elif execution_status == "failed_validation":
            row["sourcing_status"] = "blocked_invalid_payload"
            row["execution_issues"] = ["operator_import_package_validation_failed"]
        elif ticker in imported:
            row["sourcing_status"] = imported[ticker]["coverage_status"]
        elif execution_reason in {
            "operator_import_package_missing",
            "source_approval_failed",
            "concrete_source_approval_failed",
        }:
            row["sourcing_status"] = "blocked_no_source"
            row["execution_issues"] = [execution_reason]
        elif execution_reason == "no_approved_mvp_fundamental_provider":
            row["sourcing_status"] = "blocked_no_source"
            row["execution_issues"] = [execution_reason]
        elif execution_status == "completed":
            row["sourcing_status"] = "blocked_provider_coverage"
            row["execution_issues"] = ["operator_package_did_not_include_ticker"]
        else:
            row["sourcing_status"] = "selected"


def _reconcile_sourcing_status_counts(
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[str],
    *,
    run_status: str,
) -> dict[str, Any]:
    selected_set = set(selected)
    if len(selected_set) != len(selected):
        raise ValidatedFundamentalMetricSourcingError("selected sourcing tickers are not unique")
    rows_by_ticker = _unique_by_ticker(rows, key="ticker", label="ME-DATA07 sourcing status")
    if not selected_set <= set(rows_by_ticker):
        raise ValidatedFundamentalMetricSourcingError("selected sourcing tickers are missing from per-ticker status")

    selected_status_counts = Counter(str(rows_by_ticker[ticker].get("sourcing_status") or "") for ticker in selected_set)
    global_status_counts = Counter(str(row.get("sourcing_status") or "") for row in rows)
    classified = SUCCESS_SOURCING_STATUSES | BLOCKED_SOURCING_STATUSES | FAILED_SOURCING_STATUSES | PENDING_SOURCING_STATUSES
    unknown = sorted(status for status in selected_status_counts if status not in classified)
    if unknown:
        raise ValidatedFundamentalMetricSourcingError(
            f"selected sourcing statuses are not classified: {', '.join(unknown)}"
        )

    success_count = sum(selected_status_counts[status] for status in SUCCESS_SOURCING_STATUSES)
    blocked_count = sum(selected_status_counts[status] for status in BLOCKED_SOURCING_STATUSES)
    failed_count = sum(selected_status_counts[status] for status in FAILED_SOURCING_STATUSES)
    pending_count = sum(selected_status_counts[status] for status in PENDING_SOURCING_STATUSES)
    selected_count = len(selected_set)
    terminal_total = success_count + blocked_count + failed_count + pending_count
    not_selected_count = global_status_counts["not_selected"]
    if terminal_total != selected_count:
        raise ValidatedFundamentalMetricSourcingError(
            "selected sourcing status reconciliation failed: "
            f"selected={selected_count}, success={success_count}, blocked={blocked_count}, "
            f"failed={failed_count}, pending={pending_count}"
        )
    if not_selected_count != len(rows) - selected_count:
        raise ValidatedFundamentalMetricSourcingError(
            "not-selected sourcing status reconciliation failed: "
            f"not_selected={not_selected_count}, expected={len(rows) - selected_count}"
        )
    if run_status in TERMINAL_RUN_STATUSES and pending_count:
        raise ValidatedFundamentalMetricSourcingError(
            f"terminal run status {run_status} contains {pending_count} pending selected tickers"
        )

    return {
        "selected_count": selected_count,
        "success_count": success_count,
        "blocked_count": blocked_count,
        "failed_count": failed_count,
        "pending_count": pending_count,
        "not_selected_count": not_selected_count,
        "selected_status_counts": dict(sorted(selected_status_counts.items())),
        "global_status_counts": dict(sorted(global_status_counts.items())),
        "reconciliation": {
            "selected_count": selected_count,
            "terminal_success_count": success_count,
            "terminal_blocked_count": blocked_count,
            "terminal_failed_count": failed_count,
            "pending_count": pending_count,
            "reconciled": True,
        },
    }


def _batch_execution_summary(**values: Any) -> dict[str, Any]:
    records = values["normalized_records"]
    status = values["status_reconciliation"]
    imported_count = len(records) if values["execution_status"] == "completed" else 0
    return {
        "schema_version": "market-engine-data07-batch-execution-summary-v1",
        "source_mode": values["source_mode"],
        "batch_tier": values["batch_tier"],
        "execution_status": values["execution_status"],
        "execution_reason": values["execution_reason"],
        "selected_count": status["selected_count"],
        "success_count": status["success_count"],
        "blocked_count": status["blocked_count"],
        "failed_count": status["failed_count"],
        "pending_count": status["pending_count"],
        "not_selected_count": status["not_selected_count"],
        "selected_status_counts": status["selected_status_counts"],
        "reconciliation": status["reconciliation"],
        "requests_attempted": 0,
        "provider_calls_performed": 0,
        "input_presence_checks": 1 if values["source_mode"] == "operator_import" else 0,
        "imports_attempted": 1 if values["import_attempted"] else 0,
        "imported_count": imported_count,
        "normalized_count": len(records),
        "complete_count": status["selected_status_counts"].get("complete", 0),
        "partial_count": status["selected_status_counts"].get("partial", 0),
        "validation_issue_count": len(values["validation"].get("issues") or []),
        "validation_failed_record_count": values["validation"].get("validation_failed_record_count", 0),
        "package_validation_failed": bool(values["validation"].get("package_validation_failed")),
    }


def _blocker_report(
    rows: Sequence[Mapping[str, Any]],
    execution_reason: str,
    run_status: str,
    status_reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "market-engine-data07-blocker-report-v1",
        "run_status": run_status,
        "primary_blocker": execution_reason,
        "status_counts": status_reconciliation["global_status_counts"],
        "selected_status_counts": status_reconciliation["selected_status_counts"],
        "selected_blocked_count": status_reconciliation["blocked_count"],
        "selected_failed_count": status_reconciliation["failed_count"],
        "remaining_fundamental_missing": sum(row["overall_fundamental_status"] == "missing" for row in rows),
        "remaining_fundamental_partial": sum(row["overall_fundamental_status"] == "partial" for row in rows),
        "next_action": "supply a governance-approved operator package with primary-source metric lineage",
    }


def _run_status(source_mode: str, execution_status: str, downstream: Mapping[str, Any] | None) -> str:
    if downstream:
        return "completed_with_coverage_measurement"
    if execution_status == "completed":
        return "completed_import_without_downstream"
    if source_mode == "inventory_only":
        return "inventory_completed_no_acquisition"
    if execution_status == "failed_validation":
        return "failed_validation"
    return "blocked_external_source_requirement"


def _empty_validation_summary() -> dict[str, Any]:
    return {
        "schema_version": "market-engine-data07-metric-validation-summary-v1",
        "validation_status": "not_run",
        "input_record_count": 0,
        "normalized_record_count": 0,
        "complete_count": 0,
        "partial_count": 0,
        "duplicate_records_deduplicated": 0,
        "excluded_unselected_count": 0,
        "failed_tickers": [],
        "validation_failed_record_count": 0,
        "package_validation_failed": False,
        "metric_counts": {metric: 0 for metric in MVP_METRIC_FIELDS},
        "issues": [],
        "null_preserved": True,
        "negative_values_allowed": True,
        "deterministic_ordering": True,
    }


def _freshness_label(value: Any, status: str) -> str:
    if status == "stale":
        return "stale"
    if status in {"invalid", "conflicting"}:
        return "invalid"
    return "current" if value else "unknown"


def _unique_by_ticker(rows: Sequence[Mapping[str, Any]], *, key: str, label: str) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        ticker = str(row.get(key) or "").upper()
        if not ticker or ticker in result:
            raise ValidatedFundamentalMetricSourcingError(f"{label} contains missing or duplicate tickers")
        result[ticker] = row
    return result


def _write_normalized_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    columns = ["ticker", "as_of_date", "source_name", "source_reference", "source_freshness_date", "currency", *MVP_METRIC_FIELDS, "fundamental_notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            metrics = record["metrics"]
            writer.writerow(
                {
                    "ticker": record["ticker"],
                    "as_of_date": record["source_date"],
                    "source_name": record["provider"],
                    "source_reference": record["source_reference"],
                    "source_freshness_date": record["source_date"],
                    "currency": "",
                    **{metric: "" if metrics.get(metric) is None else metrics[metric] for metric in MVP_METRIC_FIELDS},
                    "fundamental_notes": f"validated ME-DATA07 operator import; snapshot={record.get('snapshot_id') or 'unspecified'}",
                }
            )


def _write_artifacts(output_dir: Path, artifacts: Mapping[str, Any]) -> None:
    for name in OUTPUT_NAMES:
        _write_json(output_dir / f"{name}.json", artifacts[name])


def _render_report(artifacts: Mapping[str, Any]) -> str:
    manifest = artifacts["manifest"]
    gap = artifacts["metric_gap_analysis"]
    batch = artifacts["batch_execution_summary"]
    coverage = artifacts["coverage_before_after"]
    return "\n".join(
        [
            "# ME-DATA07 Validated Fundamental Metric Sourcing Report",
            "",
            f"Run ID: `{manifest['run_id']}`",
            f"Run status: `{manifest['run_status']}`",
            f"Source mode: `{manifest['source_mode']}`",
            "",
            "## Metric Gap",
            "",
            f"- canonical instruments: {gap['canonical_universe_size']}",
            f"- complete: {gap['status_counts'].get('complete', 0)}",
            f"- partial: {gap['status_counts'].get('partial', 0)}",
            f"- missing: {gap['status_counts'].get('missing', 0)}",
            "",
            "## Execution",
            "",
            f"- selected: {batch['selected_count']}",
            f"- successful: {batch['success_count']}",
            f"- blocked: {batch['blocked_count']}",
            f"- failed: {batch['failed_count']}",
            f"- pending: {batch['pending_count']}",
            f"- not selected: {batch['not_selected_count']}",
            f"- reconciled: {str(batch['reconciliation']['reconciled']).lower()}",
            f"- input presence checks: {batch['input_presence_checks']}",
            f"- imports attempted: {batch['imports_attempted']}",
            f"- imported: {batch['imported_count']}",
            f"- normalized: {batch['normalized_count']}",
            f"- reason: {batch['execution_reason']}",
            "",
            "## Coverage",
            "",
            f"- claim: {coverage['coverage_claim']}",
            f"- downstream executed: {str(coverage['downstream_executed']).lower()}",
            f"- current sprint before: {json.dumps(coverage['current_sprint_comparison']['before'], sort_keys=True)}",
            f"- current sprint after: {json.dumps(coverage['current_sprint_comparison']['after'], sort_keys=True)}",
            f"- current sprint absolute delta: {json.dumps(coverage['current_sprint_comparison']['absolute_delta'], sort_keys=True)}",
            f"- historical origin comparison attributable to current sprint: false" if coverage["historical_origin_comparison"] else "- historical origin comparison: not executed",
            "",
            "No provider/network call, secret persistence, broker/order action, allocation, portfolio/watchlist mutation, Telegram delivery, Decision Engine change, or recommendation-rule change was performed.",
            "",
        ]
    )


def _read_json_required(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ValidatedFundamentalMetricSourcingError(f"required JSON artifact missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedFundamentalMetricSourcingError(f"required JSON artifact malformed: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValidatedFundamentalMetricSourcingError(f"required JSON artifact must be an object: {path}")
    return payload


def _parse_date(value: Any, label: str) -> date:
    if value is None or not str(value).strip():
        raise ValidatedFundamentalMetricSourcingError(f"{label} is required")
    text = str(value).strip().split("T", 1)[0]
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValidatedFundamentalMetricSourcingError(f"{label} is invalid") from exc


def _input_checksums(*paths: Path | None) -> dict[str, str | None]:
    return {
        path.as_posix(): _sha256(path) if path.exists() and path.is_file() else None
        for path in paths
        if path is not None
    }


def _run31_input_index_path(path: Path) -> Path:
    if path.is_file():
        return path
    compact = path / "run_evidence_index.json"
    return compact if compact.exists() else path / "evidence_coverage_index.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ME-DATA07 validated MVP fundamental metric sourcing.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-mode", required=True, choices=sorted(SUPPORTED_SOURCE_MODES))
    parser.add_argument("--batch-tier", required=True, choices=sorted(SUPPORTED_BATCH_TIERS))
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    parser.add_argument("--canonical-universe", default=DEFAULT_CANONICAL_CONFIG)
    parser.add_argument("--price-history-root", default=DEFAULT_PRICE_HISTORY_ROOT)
    parser.add_argument("--baseline-data06-run", default=DEFAULT_BASELINE_DATA06_RUN)
    parser.add_argument("--baseline-run-evidence", default=DEFAULT_BASELINE_RUN31_EVIDENCE)
    parser.add_argument("--operator-import-path", default=DEFAULT_OPERATOR_IMPORT_PATH)
    parser.add_argument("--source-approval-decision")
    parser.add_argument("--source-document-root")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-snapshot-root", default=DEFAULT_RAW_SNAPSHOT_ROOT)
    parser.add_argument("--execute-downstream", action="store_true")
    parser.add_argument("--data06-run-id")
    parser.add_argument("--run31-run-id")
    parser.add_argument("--data06-output-root")
    parser.add_argument("--run31-output-root")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO, stderr: TextIO) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        artifacts, output_dir = run_validated_fundamental_metric_sourcing(
            run_id=args.run_id,
            source_mode=args.source_mode,
            batch_tier=args.batch_tier,
            as_of_date=args.as_of_date,
            canonical_universe=args.canonical_universe,
            price_history_root=args.price_history_root,
            baseline_data06_run=args.baseline_data06_run,
            baseline_run31_evidence=args.baseline_run_evidence,
            operator_import_path=args.operator_import_path,
            source_approval_decision_path=args.source_approval_decision,
            source_document_root=args.source_document_root,
            output_root=args.output_root,
            raw_snapshot_root=args.raw_snapshot_root,
            execute_downstream=args.execute_downstream,
            data06_run_id=args.data06_run_id,
            run31_run_id=args.run31_run_id,
            data06_output_root=args.data06_output_root,
            run31_output_root=args.run31_output_root,
            allow_overwrite=args.allow_overwrite,
        )
    except Exception as exc:
        print(f"ME-DATA07 failed: {exc}", file=stderr)
        return 1
    print(json.dumps({"run_id": args.run_id, "run_status": artifacts["manifest"]["run_status"], "output_dir": output_dir.as_posix()}, indent=2, sort_keys=True), file=stdout)
    return 0 if artifacts["manifest"]["run_status"].startswith("completed") else 2


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
