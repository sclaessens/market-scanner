from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.advice.deterministic_advice import ADVICE_LABELS, build_advice_index
from market_engine.advice.setup_price_market_context import extract_setup_price_market_context
from market_engine.data.complete_local_market_dataset import DEFAULT_CANONICAL_CONFIG
from market_engine.data.local_market_data_universe import DEFAULT_PRICE_HISTORY_ROOT, build_universe_snapshot
from market_engine.run.local_portfolio_context_fixture import (
    DEFAULT_PORTFOLIO_CONTEXT_PATH,
    LOCAL_PORTFOLIO_CONTEXT_BATCH_FORMAT_VERSION,
)


SCHEMA_VERSION = "market-engine-run31-broad-non-price-evidence-advice-readiness-v1"
DEFAULT_OUTPUT_ROOT = Path("artifacts/market_engine/full_advice_readiness_runs")
DEFAULT_COMPACT_EVIDENCE_ROOT = Path("artifacts/market_engine/run_evidence")
DEFAULT_FUNDAMENTAL_EVIDENCE_PATH = Path("data/processed/fundamental_quality.csv")
DEFAULT_MARKET_CONTEXT_PATH = Path("data/processed/market_regime.csv")
FUNDAMENTAL_STALE_AFTER_DAYS = 120
MARKET_STALE_AFTER_DAYS = 120
PORTFOLIO_STALE_AFTER_DAYS = 45
FULL_ADVICE_RANKING_SCOPE = "canonical_deterministic_advice_ready"
TECHNICAL_RANKING_SCOPE = "technical_setup_screening"
EVIDENCE_STATUSES = ("available", "partial", "missing", "stale", "invalid", "not_applicable", "blocked")


class BroadAdviceReadinessError(ValueError):
    pass


def run_broad_non_price_evidence_advice_readiness(
    *,
    run_id: str,
    technical_screening_artifact: str | Path,
    canonical_universe: str | Path = DEFAULT_CANONICAL_CONFIG,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    compact_evidence_root: str | Path = DEFAULT_COMPACT_EVIDENCE_ROOT,
    fundamental_evidence_path: str | Path = DEFAULT_FUNDAMENTAL_EVIDENCE_PATH,
    market_context_path: str | Path = DEFAULT_MARKET_CONTEXT_PATH,
    portfolio_context_path: str | Path | None = DEFAULT_PORTFOLIO_CONTEXT_PATH,
    freshness_reference_date: str | None = None,
    tickers: Sequence[str] | None = None,
    ticker_limit: int | None = None,
    top_limit: int = 25,
    allow_overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    started_at = _utc_now()
    wall_start = time.perf_counter()
    phase_timings: dict[str, float] = {}

    phase = time.perf_counter()
    universe = build_universe_snapshot(canonical_universe, price_history_root=price_history_root)
    instruments = _select_instruments(universe["instruments"], tickers=tickers, ticker_limit=ticker_limit)
    technical = _load_technical_screening(
        technical_screening_artifact,
        universe=universe,
        selected_instruments=instruments,
        freshness_reference_date=freshness_reference_date,
    )
    freshness = _resolve_freshness_reference_date(freshness_reference_date, technical)
    phase_timings["technical_input_loading"] = _phase_elapsed(phase)

    phase = time.perf_counter()
    fundamentals = _load_fundamental_rows(fundamental_evidence_path, reference_date=freshness["reference"])
    phase_timings["fundamental_evidence_resolution"] = _phase_elapsed(phase)

    phase = time.perf_counter()
    market = _load_market_context(market_context_path, reference_date=freshness["reference"])
    phase_timings["market_context_resolution"] = _phase_elapsed(phase)

    phase = time.perf_counter()
    portfolio = _load_portfolio_context(portfolio_context_path)
    phase_timings["portfolio_context_resolution"] = _phase_elapsed(phase)

    entries: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    dry_runs: dict[str, dict[str, Any]] = {}
    for instrument in instruments:
        ticker_start = time.perf_counter()
        try:
            entry, status_row, dry_run = _instrument_entry(
                instrument,
                technical_by_instrument=technical["by_instrument"],
                fundamental_rows=fundamentals,
                fundamental_evidence_path=fundamental_evidence_path,
                market_context=market,
                portfolio_context=portfolio,
                generated_at=started_at,
                reference_date=freshness["reference"],
                run_id=run_id,
            )
        except Exception as exc:
            entry = _failed_entry(instrument, exc, ticker_start)
            status_row = _status_row_for_failed_entry(entry)
            dry_run = _dry_run_payload_for_entry(entry, setup_context=None)
        entries.append(entry)
        status_rows.append(status_row)
        dry_runs[entry["symbol"]] = dry_run

    phase = time.perf_counter()
    output_dir = _prepare_output_dir(output_root=output_root, run_id=run_id, allow_overwrite=allow_overwrite)
    input_dir = output_dir / "canonical_advice_inputs"
    status_index_path = _write_canonical_advice_inputs(
        input_dir=input_dir,
        run_id=run_id,
        generated_at=started_at,
        status_rows=status_rows,
        dry_runs=dry_runs,
    )
    advice_index = build_advice_index(status_index_path, run_id=run_id, generated_at=started_at)
    phase_timings["canonical_advice_execution"] = _phase_elapsed(phase)

    advice_by_ticker = {str(row["ticker"]).upper(): row for row in advice_index.get("tickers") or []}
    enriched_entries = [_attach_advice(row, advice_by_ticker.get(row["symbol"], {})) for row in entries]
    technical_ranking = _technical_ranking(enriched_entries)
    full_advice_ranking = _full_advice_ranking(enriched_entries)
    summary = _coverage_summary(universe, enriched_entries, advice_index)
    throughput = _throughput_report(
        enriched_entries,
        total_runtime_seconds=time.perf_counter() - wall_start,
        start_time=started_at,
        end_time=_utc_now(),
        phase_timings={**phase_timings, "artifact_writing": 0.0},
    )
    artifacts = _artifacts(
        run_id=run_id,
        generated_at=started_at,
        universe=universe,
        canonical_universe=canonical_universe,
        price_history_root=price_history_root,
        technical_screening_artifact=technical["artifact_dir"],
        technical_input=technical,
        freshness_policy=freshness,
        fundamental_evidence_path=fundamental_evidence_path,
        market_context_path=market_context_path,
        portfolio_context_path=portfolio_context_path,
        entries=enriched_entries,
        status_index_path=status_index_path,
        advice_index=advice_index,
        technical_ranking=technical_ranking,
        full_advice_ranking=full_advice_ranking,
        summary=summary,
        throughput=throughput,
        top_limit=top_limit,
    )
    phase = time.perf_counter()
    _write_artifacts(output_dir, artifacts)
    artifacts["throughput_report"]["phase_runtime_seconds"]["artifact_writing"] = _phase_elapsed(phase)
    _write_json(output_dir / "throughput_report.json", artifacts["throughput_report"])
    final_output_dir = _validate_required_outputs(output_dir)
    _rewrite_final_artifact_paths(final_output_dir=final_output_dir, temp_output_dir=output_dir)
    compact_dir = _write_compact_evidence_package(
        run_id=run_id,
        generated_at=started_at,
        full_output_dir=final_output_dir,
        compact_evidence_root=compact_evidence_root,
        artifacts=artifacts,
        freshness_policy=freshness,
        technical_input=technical,
        allow_overwrite=allow_overwrite,
    )
    artifacts = _refresh_final_manifest_and_compact_index(final_output_dir=final_output_dir, compact_dir=compact_dir, artifacts=artifacts)
    return artifacts, final_output_dir


def _select_instruments(
    instruments: Sequence[Mapping[str, Any]],
    *,
    tickers: Sequence[str] | None,
    ticker_limit: int | None,
) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    selected: list[Mapping[str, Any]] = []
    allowed = {ticker.upper() for ticker in tickers or ()}
    for instrument in sorted(instruments, key=lambda row: str(row["instrument_id"])):
        instrument_id = str(instrument["instrument_id"])
        if instrument_id in seen:
            raise BroadAdviceReadinessError(f"duplicate instrument_id: {instrument_id}")
        seen.add(instrument_id)
        symbol = str(instrument["symbol"]).upper()
        source = str(instrument.get("source_symbol") or symbol).upper()
        if allowed and symbol not in allowed and source not in allowed:
            continue
        selected.append(instrument)
    return selected[:ticker_limit] if ticker_limit else selected


def _load_technical_screening(
    path: str | Path,
    *,
    universe: Mapping[str, Any],
    selected_instruments: Sequence[Mapping[str, Any]],
    freshness_reference_date: str | None,
) -> dict[str, Any]:
    artifact_dir = Path(path)
    index_path = artifact_dir / "universe_analysis_index.json" if artifact_dir.is_dir() else artifact_dir
    if not index_path.exists():
        raise BroadAdviceReadinessError(f"technical input missing: {index_path}")
    payload = _read_json(index_path)
    if payload.get("schema_version") != "market-engine-run30-universe-analysis-index-v1":
        raise BroadAdviceReadinessError("technical screening artifact must be a ME-RUN30 universe_analysis_index.json")
    manifest_path = index_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise BroadAdviceReadinessError("technical input manifest missing")
    manifest = _read_json(manifest_path)
    manifest_run_id = str(manifest.get("run_id") or "")
    index_run_id = str(payload.get("run_id") or "")
    if not manifest_run_id:
        raise BroadAdviceReadinessError("technical input manifest missing run_id")
    if not index_run_id:
        raise BroadAdviceReadinessError("technical input index missing run_id")
    if manifest_run_id != index_run_id:
        raise BroadAdviceReadinessError("technical input manifest/index run_id mismatch")
    rows = payload.get("instruments")
    if not isinstance(rows, list):
        raise BroadAdviceReadinessError("technical screening artifact missing instruments")
    by_instrument: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("instrument_id"):
            raise BroadAdviceReadinessError("technical screening row missing instrument_id")
        key = str(row["instrument_id"])
        if key in by_instrument:
            raise BroadAdviceReadinessError(f"duplicate technical screening instrument_id: {key}")
        by_instrument[key] = row
    current_universe_version = universe.get("universe_version")
    technical_universe_version = payload.get("universe_version") or manifest.get("canonical_universe_version")
    current_ids = {str(row["instrument_id"]) for row in universe.get("instruments") or []}
    technical_ids = set(by_instrument)
    universe_compatibility = "explicit_version_match"
    if technical_universe_version is None:
        universe_compatibility = "inferred_from_exact_instrument_ids"
    elif technical_universe_version != current_universe_version:
        raise BroadAdviceReadinessError("technical input universe version mismatch")
    selected_ids = {str(row["instrument_id"]) for row in selected_instruments}
    missing_ids = sorted(selected_ids - technical_ids)
    extra_ids = sorted(technical_ids - current_ids)
    if missing_ids:
        raise BroadAdviceReadinessError("technical input incomplete: missing instrument_id " + ", ".join(missing_ids[:10]))
    if extra_ids:
        raise BroadAdviceReadinessError("technical input contains unknown instrument_id " + ", ".join(extra_ids[:10]))
    manifest_cutoff = manifest.get("cutoff_date") or manifest.get("as_of_date") or manifest.get("reference_date")
    if manifest_cutoff is not None:
        _parse_iso_date(str(manifest_cutoff), blocker="invalid_technical_input_cutoff_date")
    elif freshness_reference_date is None:
        raise BroadAdviceReadinessError("freshness reference date missing and technical input manifest has no cutoff/as-of date")
    return {
        "status": "valid",
        "artifact_dir": artifact_dir if artifact_dir.is_dir() else index_path.parent,
        "index_path": index_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "payload": payload,
        "run_id": index_run_id,
        "technical_input_universe_version": technical_universe_version,
        "current_canonical_universe_version": current_universe_version,
        "universe_compatibility": universe_compatibility,
        "technical_input_status": "valid",
        "manifest_cutoff_date": manifest_cutoff,
        "by_instrument": by_instrument,
    }


def _resolve_freshness_reference_date(explicit: str | None, technical: Mapping[str, Any]) -> dict[str, Any]:
    if explicit:
        reference = _parse_iso_date(explicit, blocker="invalid_freshness_reference_date")
        source = "explicit_cli"
    else:
        manifest_cutoff = technical.get("manifest_cutoff_date")
        if not manifest_cutoff:
            raise BroadAdviceReadinessError("freshness reference date missing and technical input manifest has no cutoff/as-of date")
        reference = _parse_iso_date(str(manifest_cutoff), blocker="invalid_technical_input_cutoff_date")
        source = "technical_input_manifest"
    return {
        "reference_date": reference.isoformat(),
        "reference": reference,
        "resolution_source": source,
        "fundamental_stale_after_days": FUNDAMENTAL_STALE_AFTER_DAYS,
        "market_stale_after_days": MARKET_STALE_AFTER_DAYS,
        "portfolio_stale_after_days": PORTFOLIO_STALE_AFTER_DAYS,
    }


def _load_fundamental_rows(path: str | Path, *, reference_date: date) -> dict[str, Mapping[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ticker", "quality_state", "source_data_status", "source_timestamp", "source_name", "generated_at"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise BroadAdviceReadinessError("fundamental evidence CSV missing columns: " + ", ".join(missing))
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            ticker = str(row.get("ticker") or "").upper()
            if not ticker:
                continue
            grouped.setdefault(ticker, []).append(dict(row))
    return {ticker: _select_fundamental_row(ticker, rows, reference_date=reference_date) for ticker, rows in grouped.items()}


def _select_fundamental_row(ticker: str, rows: Sequence[Mapping[str, str]], *, reference_date: date) -> dict[str, Any]:
    candidates: list[tuple[date, dict[str, str]]] = []
    invalid: list[str] = []
    for row in rows:
        raw_date = _first_present(row, ("source_last_updated", "source_timestamp", "date", "generated_at"))
        if raw_date is None:
            invalid.append("missing_fundamental_source_date")
            continue
        try:
            source_date = _parse_source_date(raw_date, reference_date=reference_date, missing_blocker="missing_fundamental_source_date", invalid_blocker="invalid_fundamental_source_date")
        except BroadAdviceReadinessError:
            invalid.append("invalid_fundamental_source_date")
            continue
        candidates.append((source_date, dict(row)))
    metadata = {
        "selection_policy": "latest_valid_source_date",
        "candidate_row_count": len(rows),
        "duplicate_identical_rows": 0,
        "conflicting_rows": 0,
        "invalid_candidate_rows": len(invalid),
        "selected_source_date": None,
    }
    if not candidates:
        return {
            "ticker": ticker,
            "_selection_metadata": metadata,
            "_selection_status": "invalid",
            "_selection_blockers": sorted(set(invalid or ["missing_fundamental_source_date"])),
        }
    newest = max(source_date for source_date, _ in candidates)
    newest_rows = [row for source_date, row in candidates if source_date == newest]
    unique = {_canonical_row(row) for row in newest_rows}
    metadata["selected_source_date"] = newest.isoformat()
    metadata["duplicate_identical_rows"] = max(0, len(newest_rows) - len(unique))
    if len(unique) > 1:
        metadata["conflicting_rows"] = len(unique)
        selected = dict(newest_rows[0])
        selected["_selection_metadata"] = metadata
        selected["_selection_status"] = "invalid"
        selected["_selection_blockers"] = ["duplicate_fundamental_rows_conflict"]
        return selected
    selected = dict(newest_rows[0])
    selected["_selection_metadata"] = metadata
    selected["_selection_status"] = "selected"
    selected["_selection_blockers"] = []
    return selected


def _load_market_context(path: str | Path, *, reference_date: date) -> dict[str, Any]:
    csv_path = Path(path)
    if not csv_path.exists():
        return _evidence("missing", "market_context", blockers=["missing_market_context"], source_path=csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return _evidence("missing", "market_context", blockers=["missing_market_context"], source_path=csv_path)
    required = {"date", "regime"}
    if required - set(rows[0]):
        return _evidence("invalid", "market_context", blockers=["invalid_market_context"], source_path=csv_path)
    selected = _select_market_context_row(rows, reference_date=reference_date)
    if selected["status"] == "invalid":
        return _evidence(
            "invalid",
            "market_context",
            source_path=csv_path,
            values={"selection_metadata": selected["selection_metadata"]},
            blockers=selected["blockers"],
        )
    row = selected["row"]
    status = "available"
    blockers: list[str] = []
    source_date = selected["source_date"]
    if _is_stale(source_date, reference_date=reference_date, max_age_days=MARKET_STALE_AFTER_DAYS):
        status = "stale"
        blockers.append("stale_market_context")
    return _evidence(
        status,
        "market_context",
        source_path=csv_path,
        source_date=source_date.isoformat(),
        values={
            "regime": row.get("regime"),
            "spy_close": row.get("spy_close"),
            "qqq_close": row.get("qqq_close"),
            "selection_metadata": selected["selection_metadata"],
        },
        blockers=blockers,
    )


def _select_market_context_row(rows: Sequence[Mapping[str, str]], *, reference_date: date) -> dict[str, Any]:
    parsed: list[tuple[date, dict[str, str]]] = []
    blockers: list[str] = []
    for row in rows:
        raw_date = row.get("date")
        if not str(row.get("regime") or "").strip():
            blockers.append("invalid_market_context")
        try:
            parsed_date = _parse_source_date(raw_date, reference_date=reference_date, missing_blocker="missing_market_context_date", invalid_blocker="invalid_market_context_date")
        except BroadAdviceReadinessError as exc:
            blockers.append(str(exc))
            continue
        parsed.append((parsed_date, dict(row)))
    metadata = {
        "selection_policy": "latest_valid_market_context_date",
        "row_count": len(rows),
        "selected_date": None,
        "duplicate_dates": 0,
        "conflicting_dates": 0,
        "invalid_rows": len(blockers),
    }
    if blockers or not parsed:
        return {"status": "invalid", "blockers": sorted(set(blockers or ["missing_market_context_date"])), "selection_metadata": metadata}
    counts = Counter(source_date for source_date, _ in parsed)
    metadata["duplicate_dates"] = sum(1 for count in counts.values() if count > 1)
    newest = max(counts)
    newest_rows = [row for source_date, row in parsed if source_date == newest]
    unique = {_canonical_row(row) for row in newest_rows}
    metadata["selected_date"] = newest.isoformat()
    if len(unique) > 1:
        metadata["conflicting_dates"] = len(unique)
        return {"status": "invalid", "blockers": ["duplicate_market_context_date_conflict"], "selection_metadata": metadata}
    return {"status": "selected", "row": newest_rows[0], "source_date": newest, "selection_metadata": metadata, "blockers": []}


def _load_portfolio_context(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "enabled": False,
            "source_path": None,
            "positions_by_ticker": {},
            "metadata": {"portfolio_context_status": "missing"},
            "source_checksum": None,
        }
    context_path = Path(path)
    if not context_path.exists():
        return {
            "enabled": False,
            "source_path": context_path.as_posix(),
            "positions_by_ticker": {},
            "metadata": {"portfolio_context_status": "missing"},
            "source_checksum": None,
        }
    payload = _read_json(context_path)
    if payload.get("portfolio_context_batch_format_version") != LOCAL_PORTFOLIO_CONTEXT_BATCH_FORMAT_VERSION:
        return {
            "enabled": False,
            "source_path": context_path.as_posix(),
            "positions_by_ticker": {},
            "metadata": {"portfolio_context_status": "invalid", "blocker": "invalid_portfolio_context_contract"},
            "source_checksum": _sha256(context_path),
        }
    positions = payload.get("positions_by_ticker")
    if not isinstance(positions, Mapping):
        positions = {}
    return {
        "enabled": True,
        "source_path": context_path.as_posix(),
        "positions_by_ticker": {str(ticker).upper(): dict(value) for ticker, value in positions.items() if isinstance(value, Mapping)},
        "metadata": payload,
        "source_checksum": _sha256(context_path),
    }


def _instrument_entry(
    instrument: Mapping[str, Any],
    *,
    technical_by_instrument: Mapping[str, Mapping[str, Any]],
    fundamental_rows: Mapping[str, Mapping[str, Any]],
    fundamental_evidence_path: str | Path,
    market_context: Mapping[str, Any],
    portfolio_context: Mapping[str, Any],
    generated_at: str,
    reference_date: date,
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    symbol = str(instrument["symbol"]).upper()
    technical = _technical_context(instrument, technical_by_instrument)
    fundamental = _fundamental_context(symbol, fundamental_rows, source_path=fundamental_evidence_path, reference_date=reference_date)
    market = _market_context_for_instrument(instrument, market_context)
    portfolio = _portfolio_context_for_instrument(symbol, portfolio_context, generated_at=generated_at, reference_date=reference_date, run_id=run_id)
    setup_context = _setup_context_from_technical(symbol, technical, market)
    evidence = _evidence_readiness(technical, fundamental, market, portfolio, setup_context)
    dry_run = _dry_run_payload_for_entry(
        {
            "symbol": symbol,
            "technical_context": technical,
            "fundamental_context": fundamental,
            "market_context": market,
            "portfolio_context": portfolio,
            "evidence_readiness": evidence,
        },
        setup_context=setup_context,
    )
    status_row = _status_row_for_entry(
        instrument=instrument,
        dry_run_path=f"canonical_advice_inputs/{symbol}/dry_run.json",
        dry_run_payload=dry_run,
        evidence=evidence,
    )
    entry = {
        "instrument_id": instrument["instrument_id"],
        "symbol": symbol,
        "asset_type": instrument["asset_type"],
        "name": instrument.get("name"),
        "exchange": instrument.get("exchange"),
        "country": instrument.get("country"),
        "currency": instrument.get("currency"),
        "sector": instrument.get("sector"),
        "industry": instrument.get("industry"),
        "universe_memberships": list(instrument.get("universe_memberships") or []),
        "technical_screening": technical,
        "technical_screening_label": technical.get("label") or "unable_to_analyse",
        "technical_candidate_score": technical.get("candidate_score"),
        "fundamental_context": fundamental,
        "market_context": market,
        "portfolio_context": portfolio,
        "evidence_readiness": evidence,
        "canonical_advice_input_status": status_row["status"],
        "technical_screening_ready": "ready" if technical["status"] == "available" else "blocked",
        "canonical_advice_input_ready": evidence["canonical_advice_input_ready"],
        "advice_generation_attempted": True,
        "advice_generation_status": "pending",
        "full_advice_ready": False,
        "full_advice_blockers": evidence["blockers"],
        "missing_evidence": evidence["missing_evidence"],
        "stale_evidence": evidence["stale_evidence"],
        "invalid_evidence": evidence["invalid_evidence"],
        "not_applicable_evidence": evidence["not_applicable_evidence"],
        "source_lineage": _lineage(technical, fundamental, market, portfolio),
        "timings": {"runtime_seconds": _elapsed(started)},
    }
    return entry, status_row, dry_run


def _technical_context(
    instrument: Mapping[str, Any],
    technical_by_instrument: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row = technical_by_instrument.get(str(instrument["instrument_id"]))
    if not row:
        return _evidence("missing", "technical_context", blockers=["missing_technical_screening"])
    if row.get("final_processing_status") != "eligible_analyzed":
        return _evidence(
            "blocked",
            "technical_context",
            blockers=list(row.get("blockers") or ["technical_context_blocked"]),
            values={"final_processing_status": row.get("final_processing_status"), "label": row.get("output_label")},
        )
    return {
        "family": "technical_context",
        "status": "available",
        "label": row.get("output_label"),
        "candidate_score": row.get("candidate_score"),
        "ranking_eligible": bool(row.get("ranking_eligible")),
        "ranking_scope": row.get("ranking_scope") or TECHNICAL_RANKING_SCOPE,
        "setup_detection": row.get("setup_detection") or {},
        "setup_price_market_context": row.get("setup_price_market_context") or {},
        "technical_rank": None,
        "source_artifact": "universe_analysis_index.json",
        "source_checksum": None,
        "source_date": (row.get("price_history") or {}).get("end_date"),
        "blockers": list(row.get("blockers") or []),
        "missing": list(row.get("missing_evidence") or []),
    }


def _fundamental_context(
    symbol: str,
    rows: Mapping[str, Mapping[str, Any]],
    *,
    source_path: str | Path = DEFAULT_FUNDAMENTAL_EVIDENCE_PATH,
    reference_date: date,
) -> dict[str, Any]:
    row = rows.get(symbol)
    if row is None:
        return _evidence("missing", "fundamental_context", blockers=["missing_fundamental_context"], source_path=source_path)
    selection_metadata = dict(row.get("_selection_metadata") or {})
    if row.get("_selection_status") == "invalid":
        return _evidence(
            "invalid",
            "fundamental_context",
            blockers=list(row.get("_selection_blockers") or ["invalid_fundamental_context"]),
            source_path=source_path,
            values={"selection_metadata": selection_metadata},
        )
    state = str(row.get("quality_state") or "").upper()
    if state == "SUFFICIENT_DATA":
        status = "available"
    elif state == "PARTIAL_DATA" or str(row.get("source_data_status")) == "partial_data":
        status = "partial"
    elif state == "INSUFFICIENT_DATA":
        status = "missing" if str(row.get("quality_metadata_status")) == "row_missing" else "partial"
    else:
        status = "invalid"
    blockers: list[str] = []
    if status == "partial":
        blockers.append("partial_fundamental_context")
    if status == "missing":
        blockers.append("missing_fundamental_context")
    if status == "invalid":
        blockers.append("invalid_fundamental_context")
    source_date_raw = _first_present(row, ("source_last_updated", "source_timestamp", "date", "generated_at"))
    try:
        parsed_source_date = _parse_source_date(source_date_raw, reference_date=reference_date, missing_blocker="missing_fundamental_source_date", invalid_blocker="invalid_fundamental_source_date")
    except BroadAdviceReadinessError as exc:
        return _evidence(
            "invalid",
            "fundamental_context",
            blockers=[str(exc)],
            source_path=source_path,
            values={"selection_metadata": selection_metadata},
        )
    if status in {"available", "partial"} and _is_stale(parsed_source_date, reference_date=reference_date, max_age_days=FUNDAMENTAL_STALE_AFTER_DAYS):
        status = "stale"
        blockers = ["stale_fundamental_context"]
    missing_fields = [field for field in str(row.get("missing_required_fields") or "").split("|") if field]
    return _evidence(
        status,
        "fundamental_context",
        source_path=source_path,
        source_date=parsed_source_date.isoformat(),
        values={
            "quality_state": row.get("quality_state"),
            "quality_reason": row.get("quality_reason"),
            "source_name": row.get("source_name"),
            "missing_required_fields": missing_fields,
            "selection_metadata": selection_metadata,
        },
        blockers=blockers,
        missing=missing_fields,
    )


def _market_context_for_instrument(
    instrument: Mapping[str, Any],
    market_context: Mapping[str, Any],
) -> dict[str, Any]:
    if "market_context" in instrument.get("universe_memberships", ()):
        applicability = "context_instrument"
    else:
        applicability = "instrument_market_context"
    return {**dict(market_context), "applicability": applicability}


def _portfolio_context_for_instrument(
    symbol: str,
    portfolio: Mapping[str, Any],
    *,
    generated_at: str,
    reference_date: date,
    run_id: str,
) -> dict[str, Any]:
    positions = portfolio.get("positions_by_ticker") or {}
    metadata = portfolio.get("metadata") or {}
    source_path = portfolio.get("source_path")
    if metadata.get("portfolio_context_status") == "invalid":
        return _evidence("invalid", "portfolio_context", blockers=["invalid_portfolio_context"], source_path=source_path)
    if symbol not in positions:
        return _evidence(
            "not_applicable",
            "portfolio_context",
            source_path=source_path,
            source_date=metadata.get("portfolio_snapshot_timestamp"),
            values={"applicability": "not_applicable_no_existing_position"},
            not_applicable=["portfolio_context"],
        )
    position = positions[symbol]
    required = ("position_state", "current_quantity", "current_market_value", "current_ticker_exposure_pct")
    missing = [field for field in required if position.get(field) in (None, "")]
    if missing:
        return _evidence("missing", "portfolio_context", blockers=["missing_applicable_portfolio_context"], missing=missing, source_path=source_path)
    status = "available"
    blockers: list[str] = []
    try:
        parsed_source_date = _parse_source_date(
            metadata.get("portfolio_snapshot_timestamp"),
            reference_date=reference_date,
            missing_blocker="missing_portfolio_snapshot_date",
            invalid_blocker="invalid_portfolio_snapshot_date",
        )
    except BroadAdviceReadinessError as exc:
        return _evidence("invalid", "portfolio_context", blockers=[str(exc)], source_path=source_path)
    source_date = str(metadata.get("portfolio_snapshot_timestamp") or generated_at)
    if _is_stale(parsed_source_date, reference_date=reference_date, max_age_days=PORTFOLIO_STALE_AFTER_DAYS):
        status = "stale"
        blockers.append("stale_portfolio_context")
    payload = {
        "portfolio_context_format_version": metadata.get("portfolio_context_format_version") or "market-engine-portfolio-context-v1",
        "portfolio_context_run_id": f"{run_id}-{symbol.lower()}-portfolio-context",
        "portfolio_snapshot_timestamp": metadata.get("portfolio_snapshot_timestamp") or generated_at,
        "portfolio_base_currency": metadata.get("portfolio_base_currency") or "EUR",
        "ticker": symbol,
        "position_state": position.get("position_state"),
        "existing_position": position.get("position_state") == "held",
        "current_quantity": position.get("current_quantity"),
        "current_market_value": position.get("current_market_value"),
        "portfolio_total_value": metadata.get("portfolio_total_value"),
        "current_ticker_exposure_pct": position.get("current_ticker_exposure_pct"),
        "concentration_thresholds": metadata.get("concentration_thresholds") or {},
        "policy_constraints": metadata.get("policy_constraints") or {},
        "context_provenance": {
            "source": "non_production_fixture",
            "source_path": source_path,
            "portfolio_write_authority": False,
            "no_broker_or_live_portfolio_access": True,
            "no_portfolio_or_watchlist_mutation": True,
        },
    }
    return _evidence(status, "portfolio_context", source_path=source_path, source_date=parsed_source_date.isoformat(), values=payload, blockers=blockers)


def _setup_context_from_technical(
    symbol: str,
    technical: Mapping[str, Any],
    market: Mapping[str, Any],
) -> dict[str, Any]:
    if technical.get("status") != "available":
        return extract_setup_price_market_context(
            {"ticker": symbol},
            {
                "ticker": symbol,
                "setup_price_market_context": {
                    "context_status": "missing",
                    "missing": ["valid_setup_price_market_context"],
                    "blocked_reasons": ["technical_context_not_available"],
                },
            },
        ).to_payload()
    base = dict(technical.get("setup_price_market_context") or {})
    setup = technical.get("setup_detection") or {}
    base.update(
        {
            "schema_version": "market-engine-setup-price-market-context-v1",
            "ticker": symbol,
            "context_status": "available" if market.get("status") == "available" else "partial",
            "price_context_available": True,
            "setup_context_available": True,
            "market_context_available": market.get("status") == "available",
            "trend_state": setup.get("trend_state") or base.get("trend_state") or "unknown",
            "setup_state": setup.get("setup_state") or base.get("setup_state") or "unknown",
            "price_position": setup.get("price_position") or base.get("price_position") or "unknown",
            "risk_state": setup.get("risk_state") or base.get("risk_state") or "unknown",
            "missing": [] if market.get("status") == "available" else ["market_context"],
            "blocked_reasons": [] if market.get("status") == "available" else list(market.get("blockers") or ["missing_market_context"]),
            "evidence": list(base.get("evidence") or [])
            + [
                {
                    "field": "market_context",
                    "source_path": market.get("source_path"),
                    "source_family": "local_market_regime",
                    "as_of_date": market.get("source_date"),
                }
            ],
        }
    )
    return extract_setup_price_market_context(
        {"ticker": symbol},
        {"ticker": symbol, "setup_price_market_context": base},
    ).to_payload()


def _evidence_readiness(
    technical: Mapping[str, Any],
    fundamental: Mapping[str, Any],
    market: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    setup_context: Mapping[str, Any],
) -> dict[str, Any]:
    contexts = {
        "technical_context": technical,
        "fundamental_context": fundamental,
        "market_context": market,
        "portfolio_context": portfolio,
        "setup_price_market_context": {"status": setup_context.get("context_status"), "blockers": setup_context.get("blocked_reasons") or []},
    }
    missing = []
    stale = []
    invalid = []
    not_applicable = []
    blockers = []
    for family, context in contexts.items():
        status = str(context.get("status") or "missing")
        if status == "missing":
            missing.append(family)
        elif status == "stale":
            stale.append(family)
        elif status == "invalid":
            invalid.append(family)
        elif status == "not_applicable":
            not_applicable.append(family)
        blockers.extend(str(item) for item in context.get("blockers") or [])
    canonical_ready = (
        technical.get("status") == "available"
        and fundamental.get("status") == "available"
        and market.get("status") == "available"
        and portfolio.get("status") in {"available", "not_applicable"}
        and setup_context.get("context_status") == "available"
        and not stale
        and not invalid
    )
    return {
        "overall_evidence_status": "ready" if canonical_ready else ("blocked" if invalid or stale else "partial"),
        "technical_screening_ready": technical.get("status") == "available",
        "canonical_advice_input_ready": canonical_ready,
        "full_advice_ready": False,
        "blockers": sorted(set(blockers)),
        "missing_evidence": sorted(set(missing)),
        "stale_evidence": sorted(set(stale)),
        "invalid_evidence": sorted(set(invalid)),
        "not_applicable_evidence": sorted(set(not_applicable)),
    }


def _dry_run_payload_for_entry(entry: Mapping[str, Any], *, setup_context: Mapping[str, Any] | None) -> dict[str, Any]:
    symbol = str(entry["symbol"]).upper()
    fundamental = entry.get("fundamental_context") or {}
    market = entry.get("market_context") or {}
    portfolio = entry.get("portfolio_context") or {}
    stage_results = []
    provenance: dict[str, Any] = {}
    if fundamental.get("status") == "available":
        stage_results.append({"stage_name": "fundamental_observations", "status": "completed"})
        provenance["fundamental_observations"] = {
            "source_path": fundamental.get("source_path"),
            "source_date": fundamental.get("source_date"),
            "source_checksum": fundamental.get("source_checksum"),
        }
    payload: dict[str, Any] = {
        "ticker": symbol,
        "stage_results": stage_results,
        "provenance_summary": provenance,
        "available_context_families": ["setup_price_context"] if setup_context and setup_context.get("context_status") in {"available", "partial"} else [],
        "fundamental_context_present": fundamental.get("status") == "available",
        "fundamental_context_required": True,
        "market_context": market if market.get("status") == "available" else None,
    }
    if setup_context is not None:
        payload["setup_price_market_context"] = setup_context
    if portfolio.get("status") == "available":
        payload["portfolio_context"] = portfolio.get("values") or {}
    return {"payload": payload}


def _status_row_for_entry(
    *,
    instrument: Mapping[str, Any],
    dry_run_path: str,
    dry_run_payload: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    missing = list(evidence.get("missing_evidence") or [])
    stale = list(evidence.get("stale_evidence") or [])
    invalid = list(evidence.get("invalid_evidence") or [])
    blockers = list(evidence.get("blockers") or [])
    ready = bool(evidence.get("canonical_advice_input_ready"))
    status = "review_ready" if ready else "blocked"
    readiness_level = "ready" if ready else "partial_analysis"
    return {
        "ticker": str(instrument["symbol"]).upper(),
        "status": status,
        "readiness_level": readiness_level,
        "context_stale": bool(stale),
        "actionable_review_allowed": ready,
        "decision_engine_ready": False,
        "blocked_stage": None if ready else "evidence_readiness",
        "blocked_reasons": blockers,
        "readiness_blocked_reasons": blockers,
        "missing_data_summary": missing,
        "evidence_families_missing": [item for item in missing if item != "setup_price_market_context"],
        "artifact_path": dry_run_path,
        "artifact_sha256": _sha256_payload(dry_run_payload),
    }


def _failed_entry(instrument: Mapping[str, Any], exc: Exception, started: float) -> dict[str, Any]:
    return {
        "instrument_id": instrument.get("instrument_id"),
        "symbol": str(instrument.get("symbol") or "UNKNOWN").upper(),
        "asset_type": instrument.get("asset_type"),
        "name": instrument.get("name"),
        "exchange": instrument.get("exchange"),
        "country": instrument.get("country"),
        "currency": instrument.get("currency"),
        "sector": instrument.get("sector"),
        "industry": instrument.get("industry"),
        "universe_memberships": list(instrument.get("universe_memberships") or []),
        "technical_screening": _evidence("blocked", "technical_context", blockers=["ticker_failure"]),
        "technical_screening_label": "unable_to_analyse",
        "technical_candidate_score": None,
        "fundamental_context": _evidence("blocked", "fundamental_context", blockers=["ticker_failure"]),
        "market_context": _evidence("blocked", "market_context", blockers=["ticker_failure"]),
        "portfolio_context": _evidence("blocked", "portfolio_context", blockers=["ticker_failure"]),
        "evidence_readiness": {
            "overall_evidence_status": "failed",
            "canonical_advice_input_ready": False,
            "full_advice_ready": False,
            "blockers": [f"{type(exc).__name__}: {exc}"],
            "missing_evidence": ["valid_evidence_inventory"],
            "stale_evidence": [],
            "invalid_evidence": [],
            "not_applicable_evidence": [],
        },
        "canonical_advice_input_status": "failed",
        "canonical_advice_input_ready": False,
        "advice_generation_attempted": True,
        "advice_generation_status": "failed",
        "full_advice_ready": False,
        "full_advice_blockers": [f"{type(exc).__name__}: {exc}"],
        "missing_evidence": ["valid_evidence_inventory"],
        "stale_evidence": [],
        "invalid_evidence": [],
        "not_applicable_evidence": [],
        "source_lineage": [],
        "timings": {"runtime_seconds": _elapsed(started)},
    }


def _status_row_for_failed_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ticker": entry["symbol"],
        "status": "invalid_artifact",
        "readiness_level": "not_ready",
        "context_stale": False,
        "actionable_review_allowed": False,
        "decision_engine_ready": False,
        "blocked_stage": "evidence_readiness",
        "blocked_reasons": entry.get("full_advice_blockers") or ["ticker_failure"],
        "readiness_blocked_reasons": entry.get("full_advice_blockers") or ["ticker_failure"],
        "missing_data_summary": ["valid_evidence_inventory"],
        "evidence_families_missing": ["valid_evidence_inventory"],
        "artifact_path": None,
        "artifact_sha256": "",
    }


def _write_canonical_advice_inputs(
    *,
    input_dir: Path,
    run_id: str,
    generated_at: str,
    status_rows: Sequence[Mapping[str, Any]],
    dry_runs: Mapping[str, Mapping[str, Any]],
) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in status_rows:
        symbol = str(row["ticker"]).upper()
        dry_run_path = input_dir / symbol / "dry_run.json"
        dry_run_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dry_runs[symbol]
        _write_json(dry_run_path, payload)
        rows.append({**dict(row), "artifact_path": dry_run_path.as_posix(), "artifact_sha256": _sha256(dry_run_path)})
    status_index = {
        "schema_version": "market-engine-ticker-status-index-v1",
        "artifact_type": "market-engine-ticker-status-index",
        "run_id": f"{run_id}-canonical-advice-input",
        "generated_at": generated_at,
        "summary": {"tickers_total": len(rows)},
        "tickers": rows,
    }
    status_index_path = input_dir / "ticker_status_index.json"
    _write_json(status_index_path, status_index)
    return status_index_path


def _attach_advice(entry: Mapping[str, Any], advice: Mapping[str, Any]) -> dict[str, Any]:
    label = str(advice.get("advice") or "unable_to_advise")
    readiness = str(advice.get("advice_readiness") or "not_ready")
    full_ready = readiness == "ready" and label != "unable_to_advise" and not advice.get("missing_for_buy_candidate")
    blockers = sorted(set(list(entry.get("full_advice_blockers") or []) + list(advice.get("blockers") or [])))
    return {
        **dict(entry),
        "canonical_advice_output": advice,
        "canonical_advice_label": label,
        "advice_confidence": advice.get("confidence") or "low",
        "advice_readiness": readiness,
        "advice_generation_status": "completed" if advice else "failed",
        "full_advice_ready": full_ready,
        "full_advice_blockers": blockers,
        "missing_evidence": sorted(set(list(entry.get("missing_evidence") or []) + list(advice.get("missing_for_buy_candidate") or []))),
    }


def _technical_ranking(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = [
        {
            "rank": 0,
            "instrument_id": row["instrument_id"],
            "symbol": row["symbol"],
            "technical_candidate_score": row.get("technical_candidate_score"),
            "technical_screening_label": row.get("technical_screening_label"),
            "canonical_advice_label": row.get("canonical_advice_label"),
            "advice_readiness": row.get("advice_readiness"),
            "full_advice_ready": row.get("full_advice_ready"),
            "missing_evidence": row.get("missing_evidence") or [],
            "blockers": row.get("full_advice_blockers") or [],
        }
        for row in entries
        if row.get("technical_candidate_score") is not None and bool((row.get("technical_screening") or {}).get("ranking_eligible"))
    ]
    ranked.sort(key=lambda row: (-int(row["technical_candidate_score"]), str(row["symbol"]), str(row["instrument_id"])))
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _full_advice_ranking(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = [
        {
            "rank": 0,
            "instrument_id": row["instrument_id"],
            "symbol": row["symbol"],
            "canonical_advice_label": row.get("canonical_advice_label"),
            "advice_confidence": row.get("advice_confidence"),
            "advice_readiness": row.get("advice_readiness"),
            "technical_candidate_score": row.get("technical_candidate_score") or 0,
            "full_advice_ready": row.get("full_advice_ready"),
            "missing_evidence": row.get("missing_evidence") or [],
            "blockers": row.get("full_advice_blockers") or [],
        }
        for row in entries
        if row.get("full_advice_ready")
    ]
    label_priority = {"buy_candidate": 0, "wait_for_price": 1, "hold_existing": 2, "watchlist": 3, "avoid_for_now": 4, "take_loss_review": 5}
    ranked.sort(key=lambda row: (label_priority.get(str(row["canonical_advice_label"]), 99), -int(row["technical_candidate_score"]), str(row["symbol"]), str(row["instrument_id"])))
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _coverage_summary(
    universe: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    advice_index: Mapping[str, Any],
) -> dict[str, Any]:
    fundamental = Counter(str(row["fundamental_context"]["status"]) for row in entries)
    market = Counter(str(row["market_context"]["status"]) for row in entries)
    portfolio = Counter(str(row["portfolio_context"]["status"]) for row in entries)
    advice = Counter(str((row.get("canonical_advice_output") or {}).get("advice") or "unable_to_advise") for row in entries)
    readiness = Counter(str(row.get("advice_readiness") or "not_ready") for row in entries)
    technical = Counter(str(row.get("technical_screening", {}).get("status")) for row in entries)
    blockers = Counter(blocker for row in entries for blocker in row.get("full_advice_blockers") or [])
    total = len(entries)
    failed = sum(1 for row in entries if row.get("evidence_readiness", {}).get("overall_evidence_status") == "failed")
    return {
        "canonical_instruments": universe["summary"]["total_instruments"],
        "attempted_instruments": total,
        "technical_analysed": technical.get("available", 0),
        "blocked": sum(1 for row in entries if row.get("evidence_readiness", {}).get("overall_evidence_status") in {"blocked", "failed"}),
        "failed": failed,
        "fundamental_counts": {status: fundamental.get(status, 0) for status in EVIDENCE_STATUSES},
        "market_counts": {status: market.get(status, 0) for status in EVIDENCE_STATUSES},
        "portfolio_counts": {status: portfolio.get(status, 0) for status in EVIDENCE_STATUSES},
        "portfolio_applicable": total - portfolio.get("not_applicable", 0),
        "canonical_advice_input_ready": sum(1 for row in entries if row.get("canonical_advice_input_ready")),
        "advice_generation_attempted": total,
        "advice_engine_completed": len(advice_index.get("tickers") or []),
        "advice_input_ready": sum(1 for row in entries if row.get("canonical_advice_input_ready")),
        "advice_output_actionable": sum(1 for row in entries if row.get("canonical_advice_label") in {"buy_candidate", "hold_existing", "take_loss_review"}),
        "non_unable_advice_outputs": sum(1 for row in entries if row.get("canonical_advice_label") != "unable_to_advise"),
        "advice_attempted": total,
        "advice_completed": len(advice_index.get("tickers") or []),
        "full_advice_ready": sum(1 for row in entries if row.get("full_advice_ready")),
        "partial_advice": readiness.get("partial", 0),
        "unable_to_advise": advice.get("unable_to_advise", 0),
        "advice_counts": {label: advice.get(label, 0) for label in ADVICE_LABELS},
        "full_advice_ranking_eligible": sum(1 for row in entries if row.get("full_advice_ready")),
        "technical_ranking_eligible": sum(1 for row in entries if row.get("technical_candidate_score") is not None and bool((row.get("technical_screening") or {}).get("ranking_eligible"))),
        "top_blockers": dict(blockers.most_common(20)),
    }


def _throughput_report(
    entries: Sequence[Mapping[str, Any]],
    *,
    total_runtime_seconds: float,
    start_time: str,
    end_time: str,
    phase_timings: Mapping[str, float],
) -> dict[str, Any]:
    runtimes = sorted(float((row.get("timings") or {}).get("runtime_seconds") or 0.0) for row in entries)
    attempted = len(entries)
    return {
        "schema_version": "market-engine-run31-throughput-report-v1",
        "start_time": start_time,
        "end_time": end_time,
        "total_runtime_seconds": round(total_runtime_seconds, 6),
        "attempted": attempted,
        "technical_context_loaded": sum(1 for row in entries if row.get("technical_screening", {}).get("status") == "available"),
        "fundamental_context_resolved": sum(1 for row in entries if row.get("fundamental_context", {}).get("status") in {"available", "partial"}),
        "market_context_resolved": sum(1 for row in entries if row.get("market_context", {}).get("status") in {"available", "partial"}),
        "portfolio_context_resolved": sum(1 for row in entries if row.get("portfolio_context", {}).get("status") == "available"),
        "canonical_advice_attempted": attempted,
        "canonical_advice_completed": attempted,
        "advice_engine_completed": sum(1 for row in entries if row.get("advice_generation_status") == "completed"),
        "canonical_advice_input_ready": sum(1 for row in entries if row.get("canonical_advice_input_ready")),
        "non_unable_advice_outputs": sum(1 for row in entries if row.get("canonical_advice_label") != "unable_to_advise"),
        "full_advice_ready": sum(1 for row in entries if row.get("full_advice_ready")),
        "partial_advice": sum(1 for row in entries if row.get("advice_readiness") == "partial"),
        "unable_to_advise": sum(1 for row in entries if row.get("canonical_advice_label") == "unable_to_advise"),
        "blocked": sum(1 for row in entries if row.get("evidence_readiness", {}).get("overall_evidence_status") == "blocked"),
        "failed": sum(1 for row in entries if row.get("evidence_readiness", {}).get("overall_evidence_status") == "failed"),
        "mean_ticker_runtime_seconds": round(sum(runtimes) / len(runtimes), 8) if runtimes else 0.0,
        "median_ticker_runtime_seconds": round(_percentile_nearest_rank(runtimes, 50), 8),
        "p95_ticker_runtime_seconds": round(_percentile_nearest_rank(runtimes, 95), 8),
        "min_ticker_runtime_seconds": round(runtimes[0], 8) if runtimes else 0.0,
        "max_ticker_runtime_seconds": round(runtimes[-1], 8) if runtimes else 0.0,
        "tickers_per_second": round(attempted / total_runtime_seconds, 6) if total_runtime_seconds else 0.0,
        "slowest_tickers": sorted(
            (
                {"symbol": row["symbol"], "instrument_id": row["instrument_id"], "runtime_seconds": (row.get("timings") or {}).get("runtime_seconds") or 0.0}
                for row in entries
            ),
            key=lambda row: (-float(row["runtime_seconds"]), str(row["symbol"])),
        )[:10],
        "phase_runtime_seconds": {key: round(value, 6) for key, value in phase_timings.items()},
    }


def _artifacts(
    *,
    run_id: str,
    generated_at: str,
    universe: Mapping[str, Any],
    canonical_universe: str | Path,
    price_history_root: str | Path,
    technical_screening_artifact: Path,
    technical_input: Mapping[str, Any],
    freshness_policy: Mapping[str, Any],
    fundamental_evidence_path: str | Path,
    market_context_path: str | Path,
    portfolio_context_path: str | Path | None,
    entries: Sequence[Mapping[str, Any]],
    status_index_path: Path,
    advice_index: Mapping[str, Any],
    technical_ranking: Sequence[Mapping[str, Any]],
    full_advice_ranking: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    throughput: Mapping[str, Any],
    top_limit: int,
) -> dict[str, Any]:
    blocker_report = _blocker_report(entries)
    source_lineage = _source_lineage(entries)
    transition = _technical_to_advice_transition(entries)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "market-engine-run31-broad-non-price-evidence-advice-readiness",
        "run_id": run_id,
        "generated_at": generated_at,
        "git_commit": _git_commit(),
        "input_artifacts": {
            "canonical_universe": Path(canonical_universe).as_posix(),
            "price_history_root": Path(price_history_root).as_posix(),
            "technical_screening_artifact": technical_screening_artifact.as_posix(),
            "fundamental_evidence_path": Path(fundamental_evidence_path).as_posix(),
            "market_context_path": Path(market_context_path).as_posix(),
            "portfolio_context_path": Path(portfolio_context_path).as_posix() if portfolio_context_path else None,
        },
        "input_checksums": _input_checksums(canonical_universe, technical_screening_artifact, fundamental_evidence_path, market_context_path, portfolio_context_path),
        "canonical_universe_version": universe.get("universe_version"),
        "technical_screening_run_id": technical_input.get("run_id"),
        "freshness_policy": _freshness_policy_for_artifact(freshness_policy),
        "technical_input_validation": {
            "technical_input_status": technical_input.get("technical_input_status"),
            "technical_screening_run_id": technical_input.get("run_id"),
            "technical_input_universe_version": technical_input.get("technical_input_universe_version"),
            "current_canonical_universe_version": technical_input.get("current_canonical_universe_version"),
            "universe_compatibility": technical_input.get("universe_compatibility"),
            "manifest_cutoff_date": technical_input.get("manifest_cutoff_date"),
            "manifest_path": Path(technical_input.get("manifest_path")).as_posix() if technical_input.get("manifest_path") else None,
            "index_path": Path(technical_input.get("index_path")).as_posix() if technical_input.get("index_path") else None,
        },
        "fundamental_selection_policy": "latest_valid_source_date",
        "market_context_selection_policy": "latest_valid_market_context_date",
        "compact_evidence_package": None,
        "full_artifact_tree_digest": None,
        "full_artifact_file_count": None,
        "full_artifact_total_size_bytes": None,
        "evidence_roots": {
            "fundamental": Path(fundamental_evidence_path).parent.as_posix(),
            "market": Path(market_context_path).parent.as_posix(),
        },
        "portfolio_context_path": Path(portfolio_context_path).as_posix() if portfolio_context_path else None,
        "coverage_summary": summary,
        "advice_summary": {"advice_counts": summary["advice_counts"]},
        "ranking_summary": {
            "technical_ranking_count": len(technical_ranking),
            "full_advice_ranking_count": len(full_advice_ranking),
        },
        "run_status": "completed_with_blockers" if summary["top_blockers"] else "completed_successfully",
        "guardrails": {
            "openai_api_invocation_performed": False,
            "model_invocation_performed": False,
            "live_provider_call_performed": False,
            "yfinance_download_performed": False,
            "broker_order_execution_performed": False,
            "allocation_performed": False,
            "portfolio_watchlist_mutation_performed": False,
            "telegram_delivery_performed": False,
            "scheduler_or_worker_started": False,
            "decision_engine_authority_changed": False,
            "parallel_advice_rules_added": False,
        },
        "outputs": _required_outputs(),
    }
    return {
        "manifest": manifest,
        "evidence_coverage_index": {"schema_version": "market-engine-run31-evidence-coverage-index-v1", "run_id": run_id, "instruments": list(entries)},
        "evidence_coverage_summary": {
            "schema_version": "market-engine-run31-evidence-coverage-summary-v1",
            "run_id": run_id,
            "freshness_policy": _freshness_policy_for_artifact(freshness_policy),
            "technical_input_validation": manifest["technical_input_validation"],
            "summary": summary,
        },
        "evidence_coverage_summary_md": _render_coverage_summary(run_id, summary),
        "canonical_advice_input_index": _read_json(status_index_path),
        "canonical_advice_output_index": advice_index,
        "advice_readiness_report": {
            "schema_version": "market-engine-run31-advice-readiness-report-v1",
            "run_id": run_id,
            "freshness_policy": _freshness_policy_for_artifact(freshness_policy),
            "technical_input_validation": manifest["technical_input_validation"],
            "summary": summary,
            "transitions": transition,
        },
        "advice_readiness_report_md": _render_advice_readiness(run_id, summary, transition),
        "technical_to_advice_transition": transition,
        "technical_ranking": {"schema_version": "market-engine-run31-technical-ranking-v1", "ranking_scope": TECHNICAL_RANKING_SCOPE, "candidates": list(technical_ranking)},
        "full_advice_ranking": {"schema_version": "market-engine-run31-full-advice-ranking-v1", "ranking_scope": FULL_ADVICE_RANKING_SCOPE, "candidates": list(full_advice_ranking)},
        "full_advice_ranking_md": _render_full_advice_ranking(full_advice_ranking),
        "top_full_advice_candidates_md": _render_top_full_advice_candidates(full_advice_ranking[:top_limit]),
        "unable_to_advise": {"schema_version": "market-engine-run31-unable-to-advise-v1", "tickers": [row for row in entries if row.get("canonical_advice_label") == "unable_to_advise"]},
        "unable_to_advise_md": _render_unable(entries),
        "blocker_report": blocker_report,
        "source_lineage": source_lineage,
        "throughput_report": throughput,
    }


def _write_artifacts(output_dir: Path, artifacts: Mapping[str, Any]) -> None:
    _write_json(output_dir / "manifest.json", artifacts["manifest"])
    _write_json(output_dir / "evidence_coverage_index.json", artifacts["evidence_coverage_index"])
    _write_json(output_dir / "evidence_coverage_summary.json", artifacts["evidence_coverage_summary"])
    (output_dir / "evidence_coverage_summary.md").write_text(artifacts["evidence_coverage_summary_md"], encoding="utf-8")
    _write_json(output_dir / "canonical_advice_input_index.json", artifacts["canonical_advice_input_index"])
    _write_json(output_dir / "canonical_advice_output_index.json", artifacts["canonical_advice_output_index"])
    _write_json(output_dir / "advice_readiness_report.json", artifacts["advice_readiness_report"])
    (output_dir / "advice_readiness_report.md").write_text(artifacts["advice_readiness_report_md"], encoding="utf-8")
    _write_json(output_dir / "technical_to_advice_transition.json", artifacts["technical_to_advice_transition"])
    _write_json(output_dir / "technical_ranking.json", artifacts["technical_ranking"])
    _write_json(output_dir / "full_advice_ranking.json", artifacts["full_advice_ranking"])
    (output_dir / "full_advice_ranking.md").write_text(artifacts["full_advice_ranking_md"], encoding="utf-8")
    (output_dir / "top_full_advice_candidates.md").write_text(artifacts["top_full_advice_candidates_md"], encoding="utf-8")
    _write_json(output_dir / "unable_to_advise.json", artifacts["unable_to_advise"])
    (output_dir / "unable_to_advise.md").write_text(artifacts["unable_to_advise_md"], encoding="utf-8")
    _write_json(output_dir / "blocker_report.json", artifacts["blocker_report"])
    _write_json(output_dir / "source_lineage.json", artifacts["source_lineage"])
    _write_json(output_dir / "throughput_report.json", artifacts["throughput_report"])


def _prepare_output_dir(*, output_root: str | Path, run_id: str, allow_overwrite: bool) -> Path:
    output_dir = Path(output_root) / run_id
    if output_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    temp_dir = Path(output_root) / f".{run_id}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _validate_required_outputs(temp_dir: Path) -> Path:
    expected = set(_required_outputs().values())
    actual = {path.name for path in temp_dir.iterdir() if path.is_file()}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError("ME-RUN31 artifact set incomplete: " + ", ".join(missing))
    final_dir = temp_dir.with_name(temp_dir.name.removeprefix(".").removesuffix(".tmp"))
    if final_dir.exists():
        shutil.rmtree(final_dir)
    temp_dir.rename(final_dir)
    return final_dir


def _rewrite_final_artifact_paths(*, final_output_dir: Path, temp_output_dir: Path) -> None:
    temp_text = temp_output_dir.as_posix()
    final_text = final_output_dir.as_posix()
    for path in final_output_dir.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rewritten = _replace_string_values(payload, temp_text, final_text)
        if rewritten != payload:
            _write_json(path, rewritten)


def _replace_string_values(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_string_values(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_string_values(item, old, new) for key, item in value.items()}
    return value


def _write_compact_evidence_package(
    *,
    run_id: str,
    generated_at: str,
    full_output_dir: Path,
    compact_evidence_root: str | Path,
    artifacts: Mapping[str, Any],
    freshness_policy: Mapping[str, Any],
    technical_input: Mapping[str, Any],
    allow_overwrite: bool,
) -> Path:
    compact_dir = Path(compact_evidence_root) / run_id
    if compact_dir.exists() and not allow_overwrite:
        raise FileExistsError(f"compact evidence directory already exists: {compact_dir}")
    temp_dir = Path(compact_evidence_root) / f".{run_id}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if compact_dir.exists():
        shutil.rmtree(compact_dir)
    temp_dir.mkdir(parents=True, exist_ok=False)

    copied = {
        "manifest": "manifest.json",
        "evidence_coverage_summary": "evidence_coverage_summary.json",
        "advice_readiness_report": "advice_readiness_report.json",
        "technical_to_advice_transition": "technical_to_advice_transition.json",
        "blocker_report": "blocker_report.json",
        "throughput_report": "throughput_report.json",
        "full_advice_ranking": "full_advice_ranking.json",
    }
    for filename in copied.values():
        shutil.copyfile(full_output_dir / filename, temp_dir / filename)
    top_level_checksums = _top_level_file_checksums(full_output_dir)
    _write_json(temp_dir / "top_level_checksums.json", top_level_checksums)
    stats = _artifact_tree_stats(full_output_dir)
    summary = artifacts["evidence_coverage_summary"]["summary"]
    run_evidence_index = {
        "schema_version": "market-engine-run31-compact-run-evidence-v1",
        "run_id": run_id,
        "generated_at": generated_at,
        "git_commit": _git_commit(),
        "canonical_universe_version": artifacts["manifest"].get("canonical_universe_version"),
        "freshness_reference_date": freshness_policy["reference_date"],
        "technical_screening_run_id": technical_input.get("run_id"),
        "input_artifacts": artifacts["manifest"].get("input_artifacts"),
        "input_checksums": artifacts["manifest"].get("input_checksums"),
        "full_artifact": {
            "local_path": full_output_dir.as_posix(),
            "file_count": stats["file_count"],
            "total_size_bytes": stats["total_size_bytes"],
            "top_level_file_checksums": top_level_checksums,
            "full_tree_digest": stats["tree_digest"],
        },
        "metrics": {
            "attempted_instruments": summary["attempted_instruments"],
            "technical_analysed": summary["technical_analysed"],
            "technical_ranking_eligible": summary["technical_ranking_eligible"],
            "canonical_advice_input_ready": summary["canonical_advice_input_ready"],
            "advice_engine_completed": summary["advice_engine_completed"],
            "full_advice_ready": summary["full_advice_ready"],
            "failed": summary["failed"],
        },
        "advice_counts": summary["advice_counts"],
        "evidence_status_counts": {
            "fundamental": summary["fundamental_counts"],
            "market": summary["market_counts"],
            "portfolio": summary["portfolio_counts"],
        },
        "top_blockers": summary["top_blockers"],
        "compact_outputs": {key: value for key, value in copied.items()} | {"top_level_checksums": "top_level_checksums.json"},
    }
    _write_json(temp_dir / "run_evidence_index.json", run_evidence_index)
    temp_dir.rename(compact_dir)
    return compact_dir


def _refresh_final_manifest_and_compact_index(*, final_output_dir: Path, compact_dir: Path, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    stats = _artifact_tree_stats(final_output_dir)
    manifest_path = final_output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "compact_evidence_package": compact_dir.as_posix(),
            "full_artifact_tree_digest": stats["tree_digest"],
            "full_artifact_file_count": stats["file_count"],
            "full_artifact_total_size_bytes": stats["total_size_bytes"],
        }
    )
    _write_json(manifest_path, manifest)
    stats = _artifact_tree_stats(final_output_dir)
    manifest["full_artifact_tree_digest"] = stats["tree_digest"]
    manifest["full_artifact_file_count"] = stats["file_count"]
    manifest["full_artifact_total_size_bytes"] = stats["total_size_bytes"]
    _write_json(manifest_path, manifest)
    top_level_checksums = _top_level_file_checksums(final_output_dir)
    _write_json(compact_dir / "manifest.json", manifest)
    _write_json(compact_dir / "top_level_checksums.json", top_level_checksums)
    index_path = compact_dir / "run_evidence_index.json"
    index = _read_json(index_path)
    index["full_artifact"]["file_count"] = stats["file_count"]
    index["full_artifact"]["total_size_bytes"] = stats["total_size_bytes"]
    index["full_artifact"]["top_level_file_checksums"] = top_level_checksums
    index["full_artifact"]["full_tree_digest"] = stats["tree_digest"]
    index["compact_outputs"]["manifest"] = "manifest.json"
    index["compact_outputs"]["run_evidence_index"] = "run_evidence_index.json"
    _write_json(index_path, index)
    return {**dict(artifacts), "manifest": manifest, "compact_evidence_dir": compact_dir.as_posix(), "run_evidence_index": index}


def _artifact_tree_stats(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file() and "/." not in path.relative_to(root).as_posix()]
    return {
        "file_count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
        "tree_digest": _artifact_tree_digest(root),
    }


def _artifact_tree_digest(root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted((path for path in root.rglob("*") if path.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("."):
            continue
        if relative == "manifest.json":
            continue
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_sha256(path).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _top_level_file_checksums(root: Path) -> dict[str, str]:
    return {path.name: _sha256(path) for path in sorted(root.iterdir(), key=lambda item: item.name) if path.is_file()}


def _required_outputs() -> dict[str, str]:
    return {
        "manifest": "manifest.json",
        "evidence_coverage_index": "evidence_coverage_index.json",
        "evidence_coverage_summary": "evidence_coverage_summary.json",
        "evidence_coverage_summary_md": "evidence_coverage_summary.md",
        "canonical_advice_input_index": "canonical_advice_input_index.json",
        "canonical_advice_output_index": "canonical_advice_output_index.json",
        "advice_readiness_report": "advice_readiness_report.json",
        "advice_readiness_report_md": "advice_readiness_report.md",
        "technical_to_advice_transition": "technical_to_advice_transition.json",
        "technical_ranking": "technical_ranking.json",
        "full_advice_ranking": "full_advice_ranking.json",
        "full_advice_ranking_md": "full_advice_ranking.md",
        "top_full_advice_candidates": "top_full_advice_candidates.md",
        "unable_to_advise": "unable_to_advise.json",
        "unable_to_advise_md": "unable_to_advise.md",
        "blocker_report": "blocker_report.json",
        "source_lineage": "source_lineage.json",
        "throughput_report": "throughput_report.json",
    }


def _blocker_report(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blockers = Counter(blocker for row in entries for blocker in row.get("full_advice_blockers") or [])
    return {
        "schema_version": "market-engine-run31-blocker-report-v1",
        "blocker_counts": dict(blockers.most_common()),
        "blocked_instruments": [
            {"symbol": row["symbol"], "blockers": row.get("full_advice_blockers") or [], "missing_evidence": row.get("missing_evidence") or []}
            for row in entries
            if row.get("full_advice_blockers") or row.get("missing_evidence")
        ],
    }


def _source_lineage(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "market-engine-run31-source-lineage-v1",
        "entries": [
            {"instrument_id": row["instrument_id"], "symbol": row["symbol"], "source_lineage": row.get("source_lineage") or []}
            for row in entries
        ],
    }


def _technical_to_advice_transition(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        f"{row.get('technical_screening_label')} -> {row.get('canonical_advice_label')}"
        for row in entries
    )
    return {
        "schema_version": "market-engine-run31-technical-to-advice-transition-v1",
        "transition_counts": dict(sorted(counts.items())),
        "examples": [
            {
                "symbol": row["symbol"],
                "technical_screening_label": row.get("technical_screening_label"),
                "canonical_advice_label": row.get("canonical_advice_label"),
                "advice_readiness": row.get("advice_readiness"),
                "full_advice_ready": row.get("full_advice_ready"),
                "missing_evidence": row.get("missing_evidence") or [],
            }
            for row in entries[:50]
        ],
    }


def _lineage(*contexts: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for context in contexts:
        source_path = context.get("source_path") or context.get("source_artifact")
        if source_path:
            rows.append(
                {
                    "family": context.get("family"),
                    "source_path": source_path,
                    "source_checksum": context.get("source_checksum"),
                    "source_date": context.get("source_date"),
                    "status": context.get("status"),
                }
            )
    return rows


def _render_coverage_summary(run_id: str, summary: Mapping[str, Any]) -> str:
    rows = ["# ME-RUN31 Evidence Coverage Summary", "", f"Run ID: `{run_id}`", "", "| Metric | Value |", "|---|---:|"]
    for key in (
        "attempted_instruments",
        "technical_analysed",
        "canonical_advice_input_ready",
        "advice_engine_completed",
        "non_unable_advice_outputs",
        "full_advice_ready",
        "partial_advice",
        "unable_to_advise",
        "full_advice_ranking_eligible",
    ):
        rows.append(f"| {key} | {summary.get(key, 0)} |")
    rows.append("")
    return "\n".join(rows)


def _render_advice_readiness(run_id: str, summary: Mapping[str, Any], transition: Mapping[str, Any]) -> str:
    rows = ["# ME-RUN31 Advice Readiness Report", "", f"Run ID: `{run_id}`", "", "## Advice Counts", "", "| Label | Count |", "|---|---:|"]
    for label, count in summary["advice_counts"].items():
        rows.append(f"| {label} | {count} |")
    rows.extend(["", "## Technical To Advice Transitions", "", "| Transition | Count |", "|---|---:|"])
    for transition_name, count in transition["transition_counts"].items():
        rows.append(f"| {transition_name} | {count} |")
    rows.append("")
    return "\n".join(rows)


def _freshness_policy_for_artifact(freshness_policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reference_date": freshness_policy["reference_date"],
        "resolution_source": freshness_policy["resolution_source"],
        "fundamental_stale_after_days": FUNDAMENTAL_STALE_AFTER_DAYS,
        "market_stale_after_days": MARKET_STALE_AFTER_DAYS,
        "portfolio_stale_after_days": PORTFOLIO_STALE_AFTER_DAYS,
    }


def _render_full_advice_ranking(ranking: Sequence[Mapping[str, Any]]) -> str:
    rows = ["# ME-RUN31 Full Advice Ranking", "", "| Rank | Symbol | Advice | Readiness | Technical score |", "|---:|---|---|---|---:|"]
    for row in ranking:
        rows.append(f"| {row['rank']} | {row['symbol']} | {row['canonical_advice_label']} | {row['advice_readiness']} | {row['technical_candidate_score']} |")
    if not ranking:
        rows.append("|  | No full-advice-ready candidates |  |  |  |")
    rows.append("")
    return "\n".join(rows)


def _render_top_full_advice_candidates(ranking: Sequence[Mapping[str, Any]]) -> str:
    if not ranking:
        return "# ME-RUN31 Top Full Advice Candidates\n\nNo full-advice-ready candidates were produced in this run.\n"
    return _render_full_advice_ranking(ranking)


def _render_unable(entries: Sequence[Mapping[str, Any]]) -> str:
    rows = ["# ME-RUN31 Unable To Advise", "", "| Symbol | Missing evidence | Blockers |", "|---|---|---|"]
    for row in entries:
        if row.get("canonical_advice_label") == "unable_to_advise":
            rows.append(f"| {row['symbol']} | {', '.join(row.get('missing_evidence') or [])} | {', '.join(row.get('full_advice_blockers') or [])} |")
    rows.append("")
    return "\n".join(rows)


def _input_checksums(*paths: str | Path | None) -> dict[str, str | None]:
    checksums = {}
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw)
        if path.is_dir():
            manifest = path / "manifest.json"
            checksums[path.as_posix()] = _sha256(manifest) if manifest.exists() else None
        else:
            checksums[path.as_posix()] = _sha256(path) if path.exists() else None
    return checksums


def _evidence(
    status: str,
    family: str,
    *,
    source_path: str | Path | None = None,
    source_date: str | None = None,
    values: Mapping[str, Any] | None = None,
    blockers: Sequence[str] = (),
    missing: Sequence[str] = (),
    not_applicable: Sequence[str] = (),
) -> dict[str, Any]:
    path = Path(source_path) if source_path else None
    return {
        "family": family,
        "status": status,
        "source_path": path.as_posix() if path else None,
        "source_checksum": _sha256(path) if path and path.exists() and path.is_file() else None,
        "source_date": source_date,
        "values": dict(values or {}),
        "blockers": list(blockers),
        "missing": list(missing),
        "not_applicable": list(not_applicable),
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BroadAdviceReadinessError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _first_present(row: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def _canonical_row(row: Mapping[str, Any]) -> str:
    comparable = {key: value for key, value in row.items() if not str(key).startswith("_")}
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def _parse_source_date(value: Any, *, reference_date: date, missing_blocker: str, invalid_blocker: str) -> date:
    if value in (None, ""):
        raise BroadAdviceReadinessError(missing_blocker)
    parsed = _parse_iso_date(str(value)[:10], blocker=invalid_blocker)
    if parsed > reference_date:
        raise BroadAdviceReadinessError(invalid_blocker)
    return parsed


def _parse_iso_date(value: str, *, blocker: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise BroadAdviceReadinessError(blocker) from exc
    if parsed.isoformat() != value:
        raise BroadAdviceReadinessError(blocker)
    return parsed


def _is_stale(source_date: date, *, reference_date: date, max_age_days: int) -> bool:
    return (reference_date - source_date).days > max_age_days


def _percentile_nearest_rank(sorted_values: Sequence[float], percentile: int) -> float:
    if not sorted_values:
        return 0.0
    rank = math.ceil((percentile / 100) * len(sorted_values))
    return sorted_values[max(0, min(len(sorted_values) - 1, rank - 1))]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _phase_elapsed(started: float) -> float:
    return time.perf_counter() - started


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 8)


def _git_commit() -> str | None:
    head = Path(".git/HEAD")
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = Path(".git") / text.removeprefix("ref: ")
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    return text


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO, stderr: TextIO) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        artifacts, output_dir = run_broad_non_price_evidence_advice_readiness(
            run_id=args.run_id,
            technical_screening_artifact=args.technical_screening_artifact,
            canonical_universe=args.canonical_universe,
            price_history_root=args.price_history_root,
            output_root=args.output_root,
            compact_evidence_root=args.compact_evidence_root,
            fundamental_evidence_path=args.fundamental_evidence_path,
            market_context_path=args.market_context_path,
            portfolio_context_path=args.portfolio_context_path,
            freshness_reference_date=args.freshness_reference_date,
            tickers=_split(args.tickers),
            ticker_limit=args.ticker_limit,
            top_limit=args.top_limit,
            allow_overwrite=args.allow_overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=stderr)
        return 2
    print(json.dumps({"run_id": args.run_id, "output_dir": output_dir.as_posix(), "summary": artifacts["evidence_coverage_summary"]["summary"]}, indent=2, sort_keys=True), file=stdout)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ME-RUN31 broad non-price evidence advice readiness.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--canonical-universe", default=DEFAULT_CANONICAL_CONFIG.as_posix())
    parser.add_argument("--price-history-root", default=DEFAULT_PRICE_HISTORY_ROOT.as_posix())
    parser.add_argument("--technical-screening-artifact", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    parser.add_argument("--compact-evidence-root", default=DEFAULT_COMPACT_EVIDENCE_ROOT.as_posix())
    parser.add_argument("--fundamental-evidence-path", default=DEFAULT_FUNDAMENTAL_EVIDENCE_PATH.as_posix())
    parser.add_argument("--market-context-path", default=DEFAULT_MARKET_CONTEXT_PATH.as_posix())
    parser.add_argument("--portfolio-context-path", default=DEFAULT_PORTFOLIO_CONTEXT_PATH)
    parser.add_argument("--freshness-reference-date", default=None)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--ticker-limit", type=int, default=None)
    parser.add_argument("--top-limit", type=int, default=25)
    parser.add_argument("--allow-overwrite", action="store_true", default=False)
    return parser


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip().upper() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
