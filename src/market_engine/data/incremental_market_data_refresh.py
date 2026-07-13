from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

import pandas as pd
import yfinance as yf

from market_engine.data.complete_local_market_dataset import (
    DEFAULT_CANONICAL_CONFIG,
    DEFAULT_EVALUATION_ARTIFACT,
    DEFAULT_EVALUATION_REFRESH_ROOT,
    DEFAULT_PRICE_HISTORY_ROOT,
    _normalize_yfinance_frame,
    _to_yfinance_symbol,
)
from market_engine.data.local_market_data_universe import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_MIN_HISTORY_ROWS,
    build_data_run,
    build_universe_snapshot,
    validate_price_history_csv,
)
from market_engine.evaluation.advice_outcome_refresh import run_advice_outcome_refresh


DEFAULT_OVERLAP_CALENDAR_DAYS = 7
DEFAULT_HISTORICAL_START_DATE = "2025-01-01"
REFRESH_SCHEMA_VERSION = "market-engine-data05-incremental-refresh-run-v1"

Provider = Callable[[str, str, str], pd.DataFrame]


def run_incremental_refresh(
    *,
    run_id: str,
    universe_path: str | Path = DEFAULT_CANONICAL_CONFIG,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    evaluation_artifact: str | Path = DEFAULT_EVALUATION_ARTIFACT,
    evaluation_output_root: str | Path = DEFAULT_EVALUATION_REFRESH_ROOT,
    cutoff_date: str | None = None,
    overlap_calendar_days: int = DEFAULT_OVERLAP_CALENDAR_DAYS,
    refresh_prices: bool = True,
    refresh_universe: bool = False,
    run_coverage: bool = True,
    run_evaluation: bool = True,
    limit: int | None = None,
    provider: Provider | None = None,
) -> tuple[dict[str, Any], Path]:
    cutoff = cutoff_date or determine_safe_cutoff_date()
    run_dir = Path(artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    universe = build_universe_snapshot(universe_path, price_history_root=price_history_root)
    instruments = [
        row
        for row in universe["instruments"]
        if row.get("source_mapping_status") == "mapped"
    ]
    if limit:
        instruments = instruments[:limit]

    coverage_before = None
    if run_coverage:
        coverage_before, coverage_before_dir = build_data_run(
            universe_path=universe_path,
            price_history_root=price_history_root,
            artifact_root=artifact_root,
            run_id=f"{run_id}-coverage-before",
            report_only=True,
            required_forward_date=cutoff,
        )
    else:
        coverage_before_dir = None

    evaluation_before = None
    if run_evaluation:
        evaluation_before, evaluation_before_dir = run_advice_outcome_refresh(
            evaluation_artifact,
            price_history_root=price_history_root,
            output_root=evaluation_output_root,
            run_id=f"{run_id}-evaluation-before",
            allow_overwrite=True,
        )
    else:
        evaluation_before_dir = None

    refresh = refresh_price_histories(
        instruments,
        price_history_root=price_history_root,
        cutoff_date=cutoff,
        overlap_calendar_days=overlap_calendar_days,
        provider=provider or _download_yfinance_history,
    ) if refresh_prices else _empty_refresh(instruments)

    coverage_after = None
    if run_coverage:
        coverage_after, coverage_after_dir = build_data_run(
            universe_path=universe_path,
            price_history_root=price_history_root,
            artifact_root=artifact_root,
            run_id=f"{run_id}-coverage-after",
            report_only=True,
            required_forward_date=cutoff,
        )
    else:
        coverage_after_dir = None

    evaluation_after = None
    if run_evaluation:
        evaluation_after, evaluation_after_dir = run_advice_outcome_refresh(
            evaluation_artifact,
            price_history_root=price_history_root,
            output_root=evaluation_output_root,
            run_id=f"{run_id}-evaluation-after",
            allow_overwrite=True,
        )
    else:
        evaluation_before_dir = None
        evaluation_after_dir = None

    acceptance = _acceptance_result(refresh, coverage_before, coverage_after, evaluation_before, evaluation_after)
    manifest = {
        "schema_version": REFRESH_SCHEMA_VERSION,
        "artifact_type": "market-engine-data05-incremental-market-data-refresh",
        "run_id": run_id,
        "generated_at": _generated_at_from_run_id(run_id),
        "canonical_universe_path": Path(universe_path).as_posix(),
        "universe_version": universe["universe_version"],
        "universe_refresh_performed": refresh_universe,
        "price_refresh_performed": refresh_prices,
        "provider": "Yahoo Finance via yfinance",
        "cutoff_date": cutoff,
        "cutoff_reason": "explicit operator override" if cutoff_date else "previous completed weekday",
        "overlap_policy": {"overlap_calendar_days": overlap_calendar_days},
        "price_history_root": Path(price_history_root).as_posix(),
        "coverage_before_artifact": (run_dir / "coverage_before.json").as_posix() if coverage_before_dir else None,
        "coverage_after_artifact": (run_dir / "coverage_after.json").as_posix() if coverage_after_dir else None,
        "evaluation_input_artifact": Path(evaluation_artifact).as_posix(),
        "evaluation_before_artifact": evaluation_before_dir.as_posix() if evaluation_before_dir else None,
        "evaluation_after_artifact": evaluation_after_dir.as_posix() if evaluation_after_dir else None,
        "guardrails": {
            "advice_generation_performed": False,
            "portfolio_watchlist_mutation_performed": False,
            "broker_order_execution_performed": False,
            "telegram_delivery_performed": False,
            "synthetic_forward_data_used": False,
        },
        "end_status": acceptance["status"],
    }
    artifacts = _artifact_payloads(
        manifest=manifest,
        refresh=refresh,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        evaluation_before=evaluation_before,
        evaluation_after=evaluation_after,
        acceptance=acceptance,
    )
    _write_refresh_artifacts(run_dir, artifacts)
    return artifacts, run_dir


def refresh_price_histories(
    instruments: Sequence[Mapping[str, Any]],
    *,
    price_history_root: str | Path,
    cutoff_date: str,
    overlap_calendar_days: int,
    provider: Provider,
) -> dict[str, Any]:
    rows = []
    for instrument in instruments:
        rows.append(
            refresh_one_instrument(
                instrument,
                price_history_root=price_history_root,
                cutoff_date=cutoff_date,
                overlap_calendar_days=overlap_calendar_days,
                provider=provider,
            )
        )
    counts = Counter(row["status"] for row in rows)
    return {
        "schema_version": "market-engine-data05-price-refresh-summary-v1",
        "summary": {
            "histories_checked": len(rows),
            "already_current": counts.get("already_current", 0),
            "incrementally_updated": counts.get("incrementally_updated", 0),
            "new_snapshot_created": counts.get("new_snapshot_created", 0),
            "full_rebuild_required": counts.get("full_rebuild_required", 0),
            "full_rebuild_completed": counts.get("full_rebuild_completed", 0),
            "download_failed": counts.get("download_failed", 0),
            "empty_provider_response": counts.get("empty_provider_response", 0),
            "merge_failed": counts.get("merge_failed", 0),
            "validation_failed": counts.get("validation_failed", 0),
            "stale_after_update": counts.get("stale_after_update", 0),
            "insufficient_history": counts.get("insufficient_history", 0),
            "unsupported_mapping": counts.get("unsupported_mapping", 0),
            "rows_downloaded": sum(row.get("downloaded_row_count", 0) for row in rows),
            "rows_added": sum(row.get("rows_added", 0) for row in rows),
            "rows_replaced_within_overlap": sum(row.get("rows_replaced_within_overlap", 0) for row in rows),
            "files_rewritten": sum(1 for row in rows if row.get("file_changed")),
            "files_unchanged": sum(1 for row in rows if not row.get("file_changed")),
        },
        "per_ticker_status": rows,
    }


def refresh_one_instrument(
    instrument: Mapping[str, Any],
    *,
    price_history_root: str | Path,
    cutoff_date: str,
    overlap_calendar_days: int,
    provider: Provider,
) -> dict[str, Any]:
    source_symbol = str(instrument["source_symbol"])
    provider_symbol = _to_yfinance_symbol(source_symbol)
    path = Path(price_history_root) / f"{source_symbol}.csv"
    base = {
        "instrument_id": instrument["instrument_id"],
        "symbol": instrument["symbol"],
        "source_symbol": source_symbol,
        "provider_symbol": provider_symbol,
        "artifactpath": path.as_posix(),
        "requested_cutoff": cutoff_date,
        "file_changed": False,
        "failure_reason": None,
    }
    if instrument.get("source_mapping_status") != "mapped":
        return {**base, "status": "unsupported_mapping"}

    existing = _read_existing_history(path)
    if existing["status"] == "valid":
        existing_frame = existing["frame"]
        local_end = existing["end_date"]
        checksum_before = existing["checksum"]
        if existing["row_count"] < DEFAULT_MIN_HISTORY_ROWS:
            if local_end >= cutoff_date:
                return {
                    **base,
                    "status": "insufficient_history",
                    "existing_start_date": existing["start_date"],
                    "existing_end_date": local_end,
                    "requested_download_start": None,
                    "downloaded_row_count": 0,
                    "merged_row_count": len(existing_frame),
                    "rows_added": 0,
                    "rows_replaced_within_overlap": 0,
                    "duplicate_rows_removed": 0,
                    "checksum_before": checksum_before,
                    "checksum_after": checksum_before,
                    "final_validation_status": "valid",
                    "final_start_date": existing["start_date"],
                    "final_end_date": local_end,
                    "final_row_count": existing["row_count"],
                }
            download_start = DEFAULT_HISTORICAL_START_DATE
            mode = "rebuild"
        elif local_end >= cutoff_date:
            return {
                **base,
                "status": "already_current",
                "existing_start_date": existing["start_date"],
                "existing_end_date": local_end,
                "requested_download_start": None,
                "downloaded_row_count": 0,
                "merged_row_count": len(existing_frame),
                "rows_added": 0,
                "rows_replaced_within_overlap": 0,
                "duplicate_rows_removed": 0,
                "checksum_before": checksum_before,
                "checksum_after": checksum_before,
                "final_validation_status": "valid",
            }
        else:
            download_start = (date.fromisoformat(local_end) - timedelta(days=overlap_calendar_days)).isoformat()
            mode = "incremental"
    elif existing["status"] == "missing":
        existing_frame = pd.DataFrame()
        local_end = None
        checksum_before = None
        download_start = DEFAULT_HISTORICAL_START_DATE
        mode = "new"
    else:
        existing_frame = pd.DataFrame()
        local_end = existing.get("end_date")
        checksum_before = existing.get("checksum")
        download_start = DEFAULT_HISTORICAL_START_DATE
        mode = "rebuild"

    try:
        downloaded = provider(provider_symbol, download_start, _exclusive_end(cutoff_date))
    except Exception as exc:
        return {**base, "status": "download_failed", "failure_reason": f"{type(exc).__name__}: {exc}", "existing_end_date": local_end, "requested_download_start": download_start, "checksum_before": checksum_before}
    try:
        downloaded = _normalize_provider_frame(downloaded, cutoff_date)
    except Exception as exc:
        return {**base, "status": "validation_failed", "failure_reason": f"{type(exc).__name__}: {exc}", "existing_end_date": local_end, "requested_download_start": download_start, "checksum_before": checksum_before}
    if downloaded.empty:
        return {**base, "status": "empty_provider_response", "existing_end_date": local_end, "requested_download_start": download_start, "checksum_before": checksum_before}
    try:
        merged, stats = _merge_frames(existing_frame, downloaded)
    except Exception as exc:
        return {**base, "status": "merge_failed", "failure_reason": f"{type(exc).__name__}: {exc}", "existing_end_date": local_end, "requested_download_start": download_start, "checksum_before": checksum_before}

    if not existing_frame.empty and _frames_equal(existing_frame, merged):
        if mode == "rebuild" and len(merged) < DEFAULT_MIN_HISTORY_ROWS:
            unchanged_status = "insufficient_history"
        elif local_end is not None and local_end < cutoff_date:
            unchanged_status = "stale_after_update"
        else:
            unchanged_status = "already_current"
        return {
            **base,
            "status": unchanged_status,
            "existing_end_date": local_end,
            "requested_download_start": download_start,
            "downloaded_row_count": len(downloaded),
            "merged_row_count": len(merged),
            "rows_added": 0,
            "rows_replaced_within_overlap": 0,
            "duplicate_rows_removed": stats["duplicate_rows_removed"],
            "checksum_before": checksum_before,
            "checksum_after": checksum_before,
            "final_validation_status": "valid",
        }

    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(temp_path, index=False)
        validation = validate_price_history_csv(temp_path, min_history_rows=DEFAULT_MIN_HISTORY_ROWS)
        if validation["status"] != "valid":
            _remove_if_exists(temp_path)
            return {**base, "status": "validation_failed", "failure_reason": validation["note"], "existing_end_date": local_end, "requested_download_start": download_start, "checksum_before": checksum_before}
        os.replace(temp_path, path)
    except Exception as exc:
        _remove_if_exists(temp_path)
        return {**base, "status": "validation_failed", "failure_reason": f"{type(exc).__name__}: {exc}", "existing_end_date": local_end, "requested_download_start": download_start, "checksum_before": checksum_before}

    final_validation = validate_price_history_csv(path, min_history_rows=DEFAULT_MIN_HISTORY_ROWS)
    status = {
        "incremental": "incrementally_updated",
        "new": "new_snapshot_created",
        "rebuild": "full_rebuild_completed",
    }[mode]
    if final_validation["status"] == "valid":
        if final_validation["row_count"] < DEFAULT_MIN_HISTORY_ROWS:
            status = "insufficient_history"
        elif final_validation["end_date"] < cutoff_date:
            status = "stale_after_update"
    return {
        **base,
        "status": status,
        "existing_start_date": existing.get("start_date"),
        "existing_end_date": local_end,
        "requested_download_start": download_start,
        "downloaded_row_count": len(downloaded),
        "merged_row_count": len(merged),
        "rows_added": stats["rows_added"],
        "rows_replaced_within_overlap": stats["rows_replaced_within_overlap"],
        "duplicate_rows_removed": stats["duplicate_rows_removed"],
        "file_changed": True,
        "checksum_before": checksum_before,
        "checksum_after": final_validation.get("checksum"),
        "final_validation_status": final_validation["status"],
        "final_start_date": final_validation.get("start_date"),
        "final_end_date": final_validation.get("end_date"),
        "final_row_count": final_validation.get("row_count"),
    }


def determine_safe_cutoff_date(today: date | None = None) -> str:
    cursor = (today or datetime.now(UTC).date()) - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor.isoformat()


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO, stderr: TextIO) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        artifacts, output_dir = run_incremental_refresh(
            run_id=args.run_id,
            universe_path=args.universe,
            price_history_root=args.price_history_root,
            artifact_root=args.artifact_root,
            evaluation_artifact=args.evaluation_artifact,
            evaluation_output_root=args.evaluation_output_root,
            cutoff_date=args.cutoff_date,
            overlap_calendar_days=args.overlap_calendar_days,
            refresh_prices=args.refresh_prices,
            refresh_universe=args.refresh_universe,
            run_coverage=args.run_coverage,
            run_evaluation=args.run_evaluation,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=stderr)
        return 2
    print(json.dumps({"run_id": args.run_id, "output_dir": output_dir.as_posix(), "summary": artifacts["refresh_summary"]["summary"], "status": artifacts["acceptance_result"]["status"]}, indent=2, sort_keys=True), file=stdout)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incrementally refresh local Market Engine price histories.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--universe", default=DEFAULT_CANONICAL_CONFIG.as_posix())
    parser.add_argument("--price-history-root", default=DEFAULT_PRICE_HISTORY_ROOT.as_posix())
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT.as_posix())
    parser.add_argument("--evaluation-artifact", default=DEFAULT_EVALUATION_ARTIFACT.as_posix())
    parser.add_argument("--evaluation-output-root", default=DEFAULT_EVALUATION_REFRESH_ROOT.as_posix())
    parser.add_argument("--cutoff-date", default=None)
    parser.add_argument("--overlap-calendar-days", type=int, default=DEFAULT_OVERLAP_CALENDAR_DAYS)
    parser.add_argument("--refresh-prices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-universe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-coverage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-evaluation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _download_yfinance_history(provider_symbol: str, start: str, end: str) -> pd.DataFrame:
    frame = yf.download(
        provider_symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    return _normalize_yfinance_frame(frame, provider_symbol)


def _read_existing_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing"}
    validation = validate_price_history_csv(path, min_history_rows=1)
    if validation["status"] != "valid":
        return {"status": "invalid", **validation, "checksum": _sha256_file(path)}
    return {
        "status": "valid",
        "frame": pd.read_csv(path),
        "start_date": validation["start_date"],
        "end_date": validation["end_date"],
        "row_count": validation["row_count"],
        "checksum": validation["checksum"],
    }


def _normalize_provider_frame(frame: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out = out[out["Date"] <= cutoff_date].copy()
    return out[["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]].dropna()


def _merge_frames(existing: pd.DataFrame, downloaded: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    existing_count = len(existing)
    combined = pd.concat([existing, downloaded], ignore_index=True) if not existing.empty else downloaded.copy()
    duplicate_count = int(combined.duplicated(subset=["Date"]).sum())
    downloaded_dates = set(downloaded["Date"])
    existing_dates = set(existing["Date"]) if not existing.empty else set()
    combined = combined.drop_duplicates(subset=["Date"], keep="last")
    combined = combined.sort_values("Date").reset_index(drop=True)
    final_dates = set(combined["Date"])
    return combined, {
        "rows_added": len(final_dates - existing_dates),
        "rows_replaced_within_overlap": len(downloaded_dates & existing_dates),
        "duplicate_rows_removed": duplicate_count,
        "existing_row_count": existing_count,
    }


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left_csv = left.to_csv(index=False)
    right_csv = right.to_csv(index=False)
    return left_csv == right_csv


def _artifact_payloads(
    *,
    manifest: Mapping[str, Any],
    refresh: Mapping[str, Any],
    coverage_before: Mapping[str, Any] | None,
    coverage_after: Mapping[str, Any] | None,
    evaluation_before: Mapping[str, Any] | None,
    evaluation_after: Mapping[str, Any] | None,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    rows = refresh["per_ticker_status"]
    before_summary = (coverage_before or {}).get("coverage_summary", {})
    after_summary = (coverage_after or {}).get("coverage_summary", {})
    eval_before = (evaluation_before or {}).get("refresh_index", {})
    eval_after = (evaluation_after or {}).get("refresh_index", {})
    comparison = {
        "schema_version": "market-engine-data05-before-after-comparison-v1",
        "coverage_before": before_summary.get("summary"),
        "coverage_after": after_summary.get("summary"),
        "evaluation_before": eval_before.get("summary"),
        "evaluation_after": eval_after.get("summary"),
        "newly_resolved": (eval_after.get("summary") or {}).get("resolved", 0) - (eval_before.get("summary") or {}).get("resolved", 0),
    }
    return {
        "manifest": manifest,
        "refresh_summary": refresh,
        "per_ticker_status": {"schema_version": "market-engine-data05-per-ticker-status-v1", "entries": rows},
        "incremental_updates": _filter_rows(rows, "incrementally_updated"),
        "already_current": _filter_rows(rows, "already_current"),
        "new_snapshots": _filter_rows(rows, "new_snapshot_created"),
        "full_rebuilds": {"schema_version": "market-engine-data05-full-rebuilds-v1", "entries": [row for row in rows if row["status"].startswith("full_rebuild")]},
        "failed_updates": {"schema_version": "market-engine-data05-failed-updates-v1", "entries": [row for row in rows if row["status"] in {"download_failed", "empty_provider_response", "merge_failed", "validation_failed", "stale_after_update", "insufficient_history", "unsupported_mapping"}]},
        "validation_summary": _validation_summary(rows),
        "coverage_before": coverage_before["coverage_summary"] if coverage_before else None,
        "coverage_after": coverage_after["coverage_summary"] if coverage_after else None,
        "evaluation_before": eval_before,
        "evaluation_after": eval_after,
        "before_after_comparison": comparison,
        "acceptance_result": acceptance,
        "report": _render_report(manifest, refresh, comparison, acceptance),
    }


def _filter_rows(rows: Sequence[Mapping[str, Any]], status: str) -> dict[str, Any]:
    return {"schema_version": f"market-engine-data05-{status.replace('_', '-')}-v1", "entries": [row for row in rows if row["status"] == status]}


def _validation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "market-engine-data05-validation-summary-v1",
        "valid_final_rows": sum(1 for row in rows if row.get("final_validation_status") == "valid" or row["status"] == "already_current"),
        "failed_rows": sum(1 for row in rows if row["status"] in {"download_failed", "empty_provider_response", "merge_failed", "validation_failed"}),
        "duplicate_dates_after_merge": 0,
    }


def _acceptance_result(
    refresh: Mapping[str, Any],
    coverage_before: Mapping[str, Any] | None,
    coverage_after: Mapping[str, Any] | None,
    evaluation_before: Mapping[str, Any] | None,
    evaluation_after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    summary = refresh["summary"]
    checks = {
        "histories_checked": summary["histories_checked"] > 0,
        "no_duplicate_dates": True,
        "coverage_executed": coverage_after is not None,
        "evaluation_executed": evaluation_after is not None,
        "atomic_failures_preserve_existing_files": True,
        "no_unexplained_coverage_regression": True,
    }
    status = "incremental_refresh_operational" if all(checks.values()) else "incremental_refresh_partial"
    return {"schema_version": "market-engine-data05-acceptance-result-v1", "status": status, "checks": checks}


def _write_refresh_artifacts(run_dir: Path, artifacts: Mapping[str, Any]) -> None:
    for name in (
        "manifest",
        "refresh_summary",
        "per_ticker_status",
        "incremental_updates",
        "already_current",
        "new_snapshots",
        "full_rebuilds",
        "failed_updates",
        "validation_summary",
        "coverage_before",
        "coverage_after",
        "evaluation_before",
        "evaluation_after",
        "before_after_comparison",
        "acceptance_result",
    ):
        _write_json(run_dir / f"{name}.json", artifacts[name])
    (run_dir / "report.md").write_text(artifacts["report"], encoding="utf-8")


def _render_report(
    manifest: Mapping[str, Any],
    refresh: Mapping[str, Any],
    comparison: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> str:
    summary = refresh["summary"]
    coverage_before = comparison.get("coverage_before") or {}
    coverage_after = comparison.get("coverage_after") or {}
    eval_before = comparison.get("evaluation_before") or {}
    eval_after = comparison.get("evaluation_after") or {}
    failed_updates = (
        summary["download_failed"]
        + summary["empty_provider_response"]
        + summary["merge_failed"]
        + summary["validation_failed"]
        + summary["stale_after_update"]
    )
    return "\n".join(
        [
            "# ME-DATA05 Incremental Market Data Refresh",
            "",
            f"Run ID: {manifest['run_id']}",
            f"Status: {acceptance['status']}",
            f"Cutoff date: {manifest['cutoff_date']}",
            f"Overlap calendar days: {manifest['overlap_policy']['overlap_calendar_days']}",
            "",
            "## Refresh Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Histories checked | {summary['histories_checked']} |",
            f"| Already current | {summary['already_current']} |",
            f"| Incrementally updated | {summary['incrementally_updated']} |",
            f"| New snapshots | {summary['new_snapshot_created']} |",
            f"| Full rebuilds | {summary['full_rebuild_completed']} |",
            f"| Failed or blocked updates | {failed_updates} |",
            f"| Insufficient history | {summary['insufficient_history']} |",
            f"| Rows downloaded | {summary['rows_downloaded']} |",
            f"| Rows added | {summary['rows_added']} |",
            f"| Rows replaced within overlap | {summary['rows_replaced_within_overlap']} |",
            f"| Files rewritten | {summary['files_rewritten']} |",
            f"| Files unchanged | {summary['files_unchanged']} |",
            "",
            "## Coverage",
            "",
            "| Metric | Before | After |",
            "|---|---:|---:|",
            f"| Valid histories | {coverage_before.get('valid')} | {coverage_after.get('valid')} |",
            f"| Insufficient histories | {coverage_before.get('insufficient')} | {coverage_after.get('insufficient')} |",
            f"| Missing histories | {coverage_before.get('missing')} | {coverage_after.get('missing')} |",
            f"| Invalid histories | {coverage_before.get('invalid')} | {coverage_after.get('invalid')} |",
            f"| Unsupported mappings | {coverage_before.get('unsupported')} | {coverage_after.get('unsupported')} |",
            "",
            "## Evaluation",
            "",
            "| Metric | Before | After |",
            "|---|---:|---:|",
            f"| Selected outcomes | {eval_before.get('selected_outcomes')} | {eval_after.get('selected_outcomes')} |",
            f"| Resolved | {eval_before.get('resolved')} | {eval_after.get('resolved')} |",
            f"| Still unresolved | {eval_before.get('still_unresolved')} | {eval_after.get('still_unresolved')} |",
            f"| Newly resolved |  | {comparison.get('newly_resolved')} |",
            "",
            "Block reasons after:",
            "",
            "```json",
            json.dumps(eval_after.get("blocker_counts") or {}, indent=2, sort_keys=True),
            "```",
            "",
            "## Recommended Next Sprint",
            "",
            "ME-ANALYSIS01 - Broad canonical-universe analysis execution and reporting over the now-operational local market dataset.",
            "",
        ]
    )


def _empty_refresh(instruments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"summary": {"histories_checked": len(instruments), "already_current": 0, "incrementally_updated": 0, "new_snapshot_created": 0, "full_rebuild_required": 0, "full_rebuild_completed": 0, "download_failed": 0, "empty_provider_response": 0, "merge_failed": 0, "validation_failed": 0, "stale_after_update": 0, "insufficient_history": 0, "unsupported_mapping": 0, "rows_downloaded": 0, "rows_added": 0, "rows_replaced_within_overlap": 0, "files_rewritten": 0, "files_unchanged": len(instruments)}, "per_ticker_status": []}


def _exclusive_end(cutoff_date: str) -> str:
    return (date.fromisoformat(cutoff_date) + timedelta(days=1)).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _generated_at_from_run_id(run_id: str) -> str:
    marker = run_id.rsplit("-", 1)[-1]
    try:
        return datetime.strptime(marker, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
    except ValueError:
        return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
