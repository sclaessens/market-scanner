from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


CONFIG_SCHEMA_VERSION = "market-engine-canonical-local-market-data-universe-config-v1"
UNIVERSE_SNAPSHOT_SCHEMA_VERSION = "market-engine-canonical-local-market-data-universe-v1"
DATA_RUN_MANIFEST_SCHEMA_VERSION = "market-engine-local-market-data-run-manifest-v1"

DEFAULT_CONFIG_PATH = Path("config/market_engine/universes/canonical_universe.json")
DEFAULT_PRICE_HISTORY_ROOT = Path("data/processed")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/market_engine/data_runs")
DEFAULT_MIN_HISTORY_ROWS = 252
DEFAULT_REQUIRED_FORWARD_DATE = "2026-07-11"
REQUIRED_PRICE_COLUMNS = ("date", "open", "high", "low", "close")
KNOWN_ETF_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "SOXX"}


class MarketDataUniverseError(ValueError):
    pass


def build_universe_snapshot(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
) -> dict[str, Any]:
    config = _load_config(config_path)
    instruments: dict[str, dict[str, Any]] = {}
    duplicate_attempts = 0
    for layer in config["layers"]:
        layer_id = layer["layer_id"]
        membership = layer.get("membership", layer_id)
        if layer["source_type"] == "local_price_history_directory":
            for path in sorted(Path(price_history_root).glob("*.csv")):
                symbol = path.stem.upper()
                if _non_instrument_csv(symbol):
                    continue
                entry = _base_entry(
                    symbol=symbol,
                    name=symbol,
                    layer=layer,
                    membership=membership,
                )
                duplicate_attempts += _merge_entry(instruments, entry, membership)
        elif layer["source_type"] == "explicit_instruments":
            for raw in layer.get("instruments") or ():
                entry = _base_entry(
                    symbol=str(raw["symbol"]).upper(),
                    name=raw.get("name") or str(raw["symbol"]).upper(),
                    layer={**layer, **raw},
                    membership=membership,
                )
                duplicate_attempts += _merge_entry(instruments, entry, membership)
        else:
            raise MarketDataUniverseError(f"unsupported layer source_type: {layer['source_type']}")

    _apply_symbol_overrides(instruments, config.get("symbol_overrides") or ())
    _validate_universe(instruments)
    entries = sorted(instruments.values(), key=lambda item: item["instrument_id"])
    layer_counts = Counter(
        membership
        for entry in entries
        for membership in entry["universe_memberships"]
    )
    return {
        "schema_version": UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
        "artifact_type": "market-engine-canonical-local-market-data-universe",
        "universe_version": config["universe_version"],
        "snapshot_date": config["snapshot_date"],
        "source_config_path": Path(config_path).as_posix(),
        "point_in_time_note": config["point_in_time_note"],
        "provenance": config["provenance"],
        "blocked_layers": config.get("blocked_layers") or [],
        "summary": {
            "total_instruments": len(entries),
            "layer_counts": dict(sorted(layer_counts.items())),
            "unique_equities": sum(1 for entry in entries if entry["asset_type"] == "equity"),
            "etf_count": sum(1 for entry in entries if entry["asset_type"] == "etf"),
            "context_count": sum(1 for entry in entries if "market_context" in entry["universe_memberships"]),
            "duplicate_attempts_before_deduplication": duplicate_attempts,
            "duplicates_after_deduplication": 0,
        },
        "instruments": entries,
    }


def build_data_run(
    *,
    universe_path: str | Path,
    price_history_root: str | Path,
    artifact_root: str | Path,
    run_id: str,
    import_root: str | Path | None = None,
    layer: str | None = None,
    tickers: Sequence[str] | None = None,
    limit: int | None = None,
    report_only: bool = False,
    skip_valid: bool = False,
    force_refresh: bool = False,
    required_forward_date: str = DEFAULT_REQUIRED_FORWARD_DATE,
    min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS,
) -> tuple[dict[str, Any], Path]:
    universe = _load_universe_or_config(universe_path, price_history_root=price_history_root)
    selected = _select_instruments(universe["instruments"], layer=layer, tickers=tickers, limit=limit)
    results = []
    output_root = Path(price_history_root)
    for instrument in selected:
        before = inspect_price_history(
            instrument,
            price_history_root=output_root,
            required_forward_date=required_forward_date,
            min_history_rows=min_history_rows,
        )
        result = dict(before)
        if _should_import(result, report_only=report_only, skip_valid=skip_valid, force_refresh=force_refresh):
            result = _try_import(
                instrument,
                current=result,
                import_root=Path(import_root) if import_root else None,
                output_root=output_root,
                required_forward_date=required_forward_date,
                min_history_rows=min_history_rows,
                force_refresh=force_refresh,
            )
        elif result["snapshotstatus"] == "valid_current_snapshot" and skip_valid:
            result["snapshotstatus"] = "skipped"
            result["note"] = "Valid local snapshot skipped by operator option."
        results.append(result)

    unresolved_readiness = _unresolved_outcome_readiness(results)
    coverage = _coverage_summary(universe, results)
    run = {
        "manifest": _manifest(run_id, universe, universe_path, price_history_root, artifact_root),
        "universe_snapshot": universe,
        "coverage_summary": coverage,
        "instrument_results": {
            "schema_version": "market-engine-local-market-data-instrument-results-v1",
            "run_id": run_id,
            "results": results,
        },
        "missing_price_history": {
            "schema_version": "market-engine-local-market-data-missing-price-history-v1",
            "run_id": run_id,
            "tickers": [row["symbol"] for row in results if row["snapshotstatus"] == "missing_price_history"],
        },
        "unsupported_symbol_mappings": {
            "schema_version": "market-engine-local-market-data-unsupported-symbol-mappings-v1",
            "run_id": run_id,
            "entries": [row for row in results if row["snapshotstatus"] == "unsupported_symbol_mapping"],
        },
        "unresolved_outcome_readiness": unresolved_readiness,
        "report": _render_report(run_id, universe, coverage, unresolved_readiness),
    }
    output_dir = write_data_run(run, artifact_root=artifact_root, run_id=run_id)
    return run, output_dir


def inspect_price_history(
    instrument: Mapping[str, Any],
    *,
    price_history_root: str | Path,
    required_forward_date: str = DEFAULT_REQUIRED_FORWARD_DATE,
    min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS,
) -> dict[str, Any]:
    symbol = instrument["symbol"]
    source_symbol = instrument["source_symbol"]
    path = Path(price_history_root) / f"{source_symbol}.csv"
    base = {
        "instrument_id": instrument["instrument_id"],
        "symbol": symbol,
        "source_symbol": source_symbol,
        "universe_memberships": instrument["universe_memberships"],
        "asset_type": instrument["asset_type"],
        "snapshotstatus": "missing_price_history",
        "start_date": None,
        "end_date": None,
        "row_count": 0,
        "validation_result": "missing",
        "artifactpath": path.as_posix(),
        "blocker": "missing_price_history",
        "note": "No local price-history CSV found.",
    }
    if instrument.get("source_mapping_status") != "mapped":
        return {**base, "snapshotstatus": "unsupported_symbol_mapping", "blocker": "unsupported_symbol_mapping", "note": "Source symbol mapping is not supported."}
    if not path.exists():
        return base
    validation = validate_price_history_csv(path, min_history_rows=min_history_rows)
    if validation["status"] != "valid":
        return {
            **base,
            "snapshotstatus": validation["status"],
            "start_date": validation.get("start_date"),
            "end_date": validation.get("end_date"),
            "row_count": validation.get("row_count", 0),
            "validation_result": "failed",
            "blocker": validation["status"],
            "note": validation["note"],
        }
    status = "valid_current_snapshot"
    blocker = None
    note = "Valid local price-history snapshot."
    if validation["row_count"] < min_history_rows:
        status = "insufficient_history"
        blocker = "insufficient_history"
        note = "Local price history has fewer rows than the configured minimum."
    elif validation["end_date"] < required_forward_date:
        status = "insufficient_forward_data"
        blocker = "insufficient_forward_data"
        note = "Local price history does not reach the configured forward evaluation date."
    return {
        **base,
        "snapshotstatus": status,
        "start_date": validation["start_date"],
        "end_date": validation["end_date"],
        "row_count": validation["row_count"],
        "validation_result": "valid" if blocker is None else "limited",
        "blocker": blocker,
        "note": note,
    }


def validate_price_history_csv(path: str | Path, *, min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS) -> dict[str, Any]:
    csv_path = Path(path)
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers = {_normalize_column(header): header for header in (reader.fieldnames or [])}
            missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in headers]
            if missing:
                return {"status": "validation_failed", "note": "Missing required OHLC columns: " + ", ".join(missing), "row_count": 0}
            rows = []
            for row in reader:
                raw_date = row.get(headers["date"])
                if not raw_date:
                    return {"status": "validation_failed", "note": "Missing date value.", "row_count": len(rows)}
                rows.append(date.fromisoformat(raw_date[:10]))
    except (OSError, ValueError, csv.Error) as exc:
        return {"status": "validation_failed", "note": f"Invalid price-history CSV: {exc}", "row_count": 0}
    if not rows:
        return {"status": "validation_failed", "note": "Price-history CSV is empty.", "row_count": 0}
    if len(set(rows)) != len(rows):
        return {"status": "validation_failed", "note": "Duplicate price-history dates.", "row_count": len(rows)}
    if rows != sorted(rows):
        return {"status": "validation_failed", "note": "Price-history dates are not monotonic.", "row_count": len(rows)}
    return {
        "status": "valid",
        "note": "valid",
        "start_date": rows[0].isoformat(),
        "end_date": rows[-1].isoformat(),
        "row_count": len(rows),
    }


def write_universe_snapshot(
    snapshot: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, snapshot)
    return output


def write_data_run(run: Mapping[str, Any], *, artifact_root: str | Path, run_id: str) -> Path:
    output_dir = Path(artifact_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", run["manifest"])
    _write_json(output_dir / "universe_snapshot.json", run["universe_snapshot"])
    _write_json(output_dir / "coverage_summary.json", run["coverage_summary"])
    _write_json(output_dir / "instrument_results.json", run["instrument_results"])
    _write_json(output_dir / "missing_price_history.json", run["missing_price_history"])
    _write_json(output_dir / "unsupported_symbol_mappings.json", run["unsupported_symbol_mappings"])
    _write_json(output_dir / "unresolved_outcome_readiness.json", run["unresolved_outcome_readiness"])
    (output_dir / "report.md").write_text(run["report"], encoding="utf-8")
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO, stderr: TextIO) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.build_universe_only:
            snapshot = build_universe_snapshot(args.universe, price_history_root=args.output_root)
            output_path = Path(args.artifact_root) / args.run_id / "universe_snapshot.json"
            write_universe_snapshot(snapshot, output_path)
            print(json.dumps({"run_id": args.run_id, "universe_snapshot": output_path.as_posix(), "summary": snapshot["summary"]}, indent=2, sort_keys=True), file=stdout)
            return 0
        run, output_dir = build_data_run(
            universe_path=args.universe,
            price_history_root=args.output_root,
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            import_root=args.import_root,
            layer=args.layer,
            tickers=_split_tickers(args.tickers),
            limit=args.limit,
            report_only=args.report_only,
            skip_valid=args.skip_valid,
            force_refresh=args.force_refresh,
            required_forward_date=args.required_forward_date,
            min_history_rows=args.min_history_rows,
        )
    except (OSError, MarketDataUniverseError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    print(json.dumps({"run_id": args.run_id, "output_dir": output_dir.as_posix(), "summary": run["coverage_summary"]["summary"]}, indent=2, sort_keys=True), file=stdout)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and inspect the local Market Engine market-data universe.")
    parser.add_argument("--universe", default=DEFAULT_CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=DEFAULT_PRICE_HISTORY_ROOT.as_posix())
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT.as_posix())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--import-root", default=None)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-valid", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--build-universe-only", action="store_true")
    parser.add_argument("--required-forward-date", default=DEFAULT_REQUIRED_FORWARD_DATE)
    parser.add_argument("--min-history-rows", type=int, default=DEFAULT_MIN_HISTORY_ROWS)
    return parser


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise MarketDataUniverseError("unsupported canonical universe config schema")
    if not payload.get("provenance"):
        raise MarketDataUniverseError("canonical universe config missing provenance")
    return payload


def _load_universe_or_config(path: str | Path, *, price_history_root: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") == UNIVERSE_SNAPSHOT_SCHEMA_VERSION:
        return payload
    return build_universe_snapshot(path, price_history_root=price_history_root)


def _base_entry(*, symbol: str, name: str, layer: Mapping[str, Any], membership: str) -> dict[str, Any]:
    source_symbol = str(layer.get("source_symbol") or symbol).upper()
    source_mapping_status = "mapped"
    if layer.get("source_notes") and "required" in str(layer.get("source_notes")).lower():
        source_mapping_status = "unsupported"
    asset_type = str(layer.get("asset_type") or "equity").lower()
    if layer.get("source_type") == "local_price_history_directory" and symbol in KNOWN_ETF_SYMBOLS:
        asset_type = "etf"
    context_only = bool(layer.get("context_only", False))
    advice_eligible = bool(layer.get("advice_eligible", not context_only and asset_type == "equity"))
    return {
        "instrument_id": f"{asset_type}:{symbol.lower()}",
        "symbol": symbol,
        "asset_type": asset_type,
        "name": name,
        "exchange": layer.get("exchange") or "UNKNOWN",
        "country": layer.get("country") or "US",
        "currency": layer.get("currency") or "USD",
        "sector": layer.get("sector"),
        "industry": layer.get("industry"),
        "universe_memberships": [membership],
        "analysis_eligible": bool(layer.get("analysis_eligible", True)),
        "advice_eligible": advice_eligible,
        "context_only": context_only,
        "active": bool(layer.get("active", True)),
        "source_symbol": source_symbol,
        "source_notes": layer.get("source_notes"),
        "source_mapping_status": source_mapping_status,
    }


def _merge_entry(instruments: dict[str, dict[str, Any]], entry: dict[str, Any], membership: str) -> int:
    key = entry["instrument_id"]
    if key in instruments:
        existing = instruments[key]
        if existing["source_symbol"] != entry["source_symbol"]:
            raise MarketDataUniverseError(f"ambiguous source mapping for {entry['symbol']}")
        existing["universe_memberships"] = sorted(set(existing["universe_memberships"]) | {membership})
        for field in ("name", "exchange", "country", "currency", "sector", "industry", "source_notes"):
            if not existing.get(field) and entry.get(field):
                existing[field] = entry[field]
        existing["context_only"] = existing["context_only"] and entry["context_only"]
        existing["advice_eligible"] = existing["advice_eligible"] or entry["advice_eligible"]
        return 1
    instruments[key] = entry
    return 0


def _apply_symbol_overrides(instruments: dict[str, dict[str, Any]], overrides: Sequence[Mapping[str, Any]]) -> None:
    for override in overrides:
        symbol = str(override["canonical_symbol"]).upper()
        key = f"equity:{symbol.lower()}"
        if key in instruments:
            instruments[key]["source_symbol"] = str(override["source_symbol"]).upper()
            instruments[key]["source_notes"] = override.get("reason")


def _validate_universe(instruments: Mapping[str, Mapping[str, Any]]) -> None:
    symbols = [entry["symbol"] for entry in instruments.values()]
    if len(symbols) != len(set(symbols)):
        raise MarketDataUniverseError("duplicate canonical symbol")
    mapped_sources: dict[str, str] = {}
    for entry in instruments.values():
        if not entry.get("instrument_id") or not entry.get("symbol"):
            raise MarketDataUniverseError("invalid canonical instrument identity")
        source_symbol = entry["source_symbol"]
        if source_symbol in mapped_sources and mapped_sources[source_symbol] != entry["instrument_id"]:
            raise MarketDataUniverseError(f"duplicate acquisition symbol: {source_symbol}")
        mapped_sources[source_symbol] = entry["instrument_id"]


def _select_instruments(entries: Sequence[Mapping[str, Any]], *, layer: str | None, tickers: Sequence[str] | None, limit: int | None) -> list[Mapping[str, Any]]:
    allowed = {ticker.upper() for ticker in tickers or ()}
    selected = [
        entry
        for entry in entries
        if (not layer or layer in entry["universe_memberships"])
        and (not allowed or entry["symbol"].upper() in allowed or entry["source_symbol"].upper() in allowed)
    ]
    return selected[:limit] if limit else selected


def _should_import(result: Mapping[str, Any], *, report_only: bool, skip_valid: bool, force_refresh: bool) -> bool:
    if report_only:
        return False
    if result["snapshotstatus"] == "valid_current_snapshot" and not force_refresh:
        return False
    if result["snapshotstatus"] == "valid_current_snapshot" and skip_valid:
        return False
    return result["snapshotstatus"] in {"missing_price_history", "stale_snapshot", "insufficient_history", "insufficient_forward_data", "validation_failed", "valid_current_snapshot"}


def _try_import(instrument: Mapping[str, Any], *, current: Mapping[str, Any], import_root: Path | None, output_root: Path, required_forward_date: str, min_history_rows: int, force_refresh: bool) -> dict[str, Any]:
    if import_root is None:
        return {**current, "note": current["note"] + " No import root supplied.", "blocker": current["blocker"] or "import_root_missing"}
    source = import_root / f"{instrument['source_symbol']}.csv"
    if not source.exists():
        return {**current, "snapshotstatus": "acquisition_failed", "blocker": "operator_snapshot_missing", "note": "Operator import root does not contain a CSV for this source symbol."}
    destination = output_root / f"{instrument['source_symbol']}.csv"
    if destination.exists() and not force_refresh:
        return {**current, "snapshotstatus": "skipped", "blocker": None, "note": "Destination exists and force-refresh was not requested."}
    validation = validate_price_history_csv(source, min_history_rows=min_history_rows)
    if validation["status"] != "valid":
        return {**current, "snapshotstatus": "validation_failed", "blocker": "validation_failed", "note": validation["note"]}
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    after = inspect_price_history(instrument, price_history_root=output_root, required_forward_date=required_forward_date, min_history_rows=min_history_rows)
    status = "refreshed" if force_refresh else "imported"
    if after["snapshotstatus"] != "valid_current_snapshot":
        return {**after, "note": f"Imported local CSV but snapshot remains {after['snapshotstatus']}."}
    return {**after, "snapshotstatus": status, "note": "Imported valid operator-supplied local CSV snapshot."}


def _coverage_summary(universe: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["snapshotstatus"] for row in results)
    summary = {
        "total_canonical_instruments": universe["summary"]["total_instruments"],
        "selected_instruments": len(results),
        "layer_counts": universe["summary"]["layer_counts"],
        "unique_equities": universe["summary"]["unique_equities"],
        "etf_count": universe["summary"]["etf_count"],
        "context_count": universe["summary"]["context_count"],
        "valid": counts.get("valid_current_snapshot", 0),
        "imported": counts.get("imported", 0),
        "refreshed": counts.get("refreshed", 0),
        "skipped": counts.get("skipped", 0),
        "missing": counts.get("missing_price_history", 0),
        "stale": counts.get("stale_snapshot", 0),
        "insufficient": counts.get("insufficient_history", 0) + counts.get("insufficient_forward_data", 0),
        "invalid": counts.get("validation_failed", 0),
        "unsupported": counts.get("unsupported_symbol_mapping", 0),
        "failed": counts.get("acquisition_failed", 0),
        "completion_status": "completed_with_blockers" if any(counts.values()) and (counts.get("missing_price_history") or counts.get("acquisition_failed") or counts.get("validation_failed") or counts.get("insufficient_forward_data")) else "completed_successfully",
    }
    return {"schema_version": "market-engine-local-market-data-coverage-summary-v1", "summary": summary, "status_counts": dict(sorted(counts.items()))}


def _unresolved_outcome_readiness(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    critical = {"AMD", "ASML", "AVGO", "CLS", "COST", "CRDO", "IREN", "META", "MSFT", "NVDA", "TSM", "VRT"}
    rows = []
    for result in results:
        if result["symbol"] in critical:
            readiness = "ready_for_me_eval02_refresh" if result["snapshotstatus"] in {"valid_current_snapshot", "imported", "refreshed"} else "not_ready"
            rows.append({**result, "previous_blocker_context": "ME-EVAL02 unresolved outcome", "expected_refresh_readiness": readiness})
    return {
        "schema_version": "market-engine-unresolved-outcome-readiness-v1",
        "critical_tickers_total": len(critical),
        "covered_critical_tickers": sorted(row["symbol"] for row in rows),
        "ready_count": sum(1 for row in rows if row["expected_refresh_readiness"] == "ready_for_me_eval02_refresh"),
        "not_ready_count": sum(1 for row in rows if row["expected_refresh_readiness"] != "ready_for_me_eval02_refresh"),
        "outcomes": rows,
    }


def _manifest(run_id: str, universe: Mapping[str, Any], universe_path: str | Path, price_history_root: str | Path, artifact_root: str | Path) -> dict[str, Any]:
    return {
        "schema_version": DATA_RUN_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "market-engine-local-market-data-run-manifest",
        "run_id": run_id,
        "universe_version": universe["universe_version"],
        "snapshot_date": universe["snapshot_date"],
        "input": {"universe_path": Path(universe_path).as_posix(), "price_history_root": Path(price_history_root).as_posix()},
        "artifact_root": Path(artifact_root).as_posix(),
        "baseline_guardrail": {
            "provider_invocation_allowed": False,
            "live_source_acquisition_performed": False,
            "broker_order_execution_performed": False,
            "portfolio_watchlist_mutation_performed": False,
            "advice_generation_performed": False,
            "scheduler_implemented": False,
        },
    }


def _render_report(run_id: str, universe: Mapping[str, Any], coverage: Mapping[str, Any], readiness: Mapping[str, Any]) -> str:
    summary = coverage["summary"]
    rows = [
        "# Local Market Data Universe Run Report",
        "",
        f"Run ID: {run_id}",
        f"Universe version: {universe['universe_version']}",
        f"Snapshot date: {universe['snapshot_date']}",
        "",
        "## Coverage",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ("total_canonical_instruments", "selected_instruments", "valid", "imported", "refreshed", "skipped", "missing", "insufficient", "invalid", "unsupported", "failed"):
        rows.append(f"| {key} | {summary.get(key, 0)} |")
    rows.extend(["", "## ME-EVAL02 Critical Readiness", "", f"Ready: {readiness['ready_count']}", f"Not ready: {readiness['not_ready_count']}", ""])
    return "\n".join(rows)


def _split_tickers(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    return tuple(part.strip().upper() for part in raw.split(",") if part.strip())


def _non_instrument_csv(symbol: str) -> bool:
    return symbol.lower() in {"portfolio_intelligence", "entry_quality_metrics", "entry_quality_metrics_historical", "context_strength", "context_strength_historical", "market_regime", "reporting_dashboard_data", "validation_results", "validation_summary", "stability_state", "final_decisions", "scanner_ranked", "fundamental_quality", "validation_layer", "timing_state_layer"}


def _normalize_column(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
