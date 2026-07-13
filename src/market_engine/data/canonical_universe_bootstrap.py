from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.data.local_market_data_universe import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG_PATH,
    DEFAULT_PRICE_HISTORY_ROOT,
    MarketDataUniverseError,
    build_universe_snapshot_from_config,
)


SOURCE_SCHEMA_VERSION = "market-engine-canonical-universe-source-v1"
UNIVERSE_RUN_MANIFEST_SCHEMA_VERSION = "market-engine-canonical-universe-bootstrap-run-v1"
DEFAULT_SOURCE_ROOT = Path("config/market_engine/universes/sources")
DEFAULT_SYMBOL_OVERRIDES_PATH = Path("config/market_engine/universes/symbol_overrides.json")
DEFAULT_UNIVERSE_ARTIFACT_ROOT = Path("artifacts/market_engine/universe_runs")


def build_universe_bootstrap_run(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    base_config_path: str | Path = DEFAULT_CONFIG_PATH,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
    run_id: str,
    universe_version: str = "me-boot03-authoritative-universe-local-price-history-bootstrap-v1",
    snapshot_date: str = "2026-07-13",
    symbol_overrides_path: str | Path = DEFAULT_SYMBOL_OVERRIDES_PATH,
) -> dict[str, Any]:
    source_inventory = build_source_inventory(source_root)
    config = build_canonical_config(
        source_inventory,
        base_config_path=base_config_path,
        universe_version=universe_version,
        snapshot_date=snapshot_date,
        symbol_overrides_path=symbol_overrides_path,
    )
    snapshot = build_universe_snapshot_from_config(
        config,
        config_path="generated:me-boot03-canonical-universe-bootstrap",
        price_history_root=price_history_root,
    )
    layer_summary = _layer_summary(source_inventory, snapshot)
    overlap_report = _overlap_report(snapshot)
    symbol_mapping_report = _symbol_mapping_report(snapshot)
    excluded = _excluded_instruments(source_inventory)
    unsupported = {
        "schema_version": "market-engine-canonical-universe-unsupported-symbol-mappings-v1",
        "run_id": run_id,
        "entries": [
            entry
            for entry in symbol_mapping_report["entries"]
            if entry["source_mapping_status"] != "mapped"
        ],
    }
    manifest = {
        "schema_version": UNIVERSE_RUN_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "market-engine-canonical-universe-bootstrap-run",
        "run_id": run_id,
        "generated_at": _generated_at_from_run_id(run_id),
        "source_root": Path(source_root).as_posix(),
        "base_config_path": Path(base_config_path).as_posix(),
        "price_history_root": Path(price_history_root).as_posix(),
        "source_count": len(source_inventory["sources"]),
        "canonical_instrument_count": snapshot["summary"]["total_instruments"],
        "guardrails": {
            "live_network_access_performed": False,
            "provider_fallback_performed": False,
            "advice_generation_performed": False,
            "portfolio_watchlist_mutation_performed": False,
            "broker_order_execution_performed": False,
            "historical_membership_reconstruction_performed": False,
        },
    }
    return {
        "manifest": manifest,
        "source_inventory": source_inventory,
        "canonical_config": config,
        "canonical_universe": snapshot,
        "layer_summary": layer_summary,
        "overlap_report": overlap_report,
        "symbol_mapping_report": symbol_mapping_report,
        "excluded_instruments": excluded,
        "unsupported_symbol_mappings": unsupported,
        "report": render_universe_bootstrap_report(
            run_id=run_id,
            manifest=manifest,
            source_inventory=source_inventory,
            snapshot=snapshot,
            layer_summary=layer_summary,
            unsupported=unsupported,
        ),
    }


def build_source_inventory(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root)
    sources = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "symbol_overrides.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(_validate_source(path, payload))
    return {
        "schema_version": "market-engine-canonical-universe-source-inventory-v1",
        "source_root": root.as_posix(),
        "source_count": len(sources),
        "sources": sources,
    }


def build_canonical_config(
    source_inventory: Mapping[str, Any],
    *,
    base_config_path: str | Path,
    universe_version: str,
    snapshot_date: str,
    symbol_overrides_path: str | Path,
) -> dict[str, Any]:
    base = json.loads(Path(base_config_path).read_text(encoding="utf-8"))
    if base.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise MarketDataUniverseError("unsupported base canonical universe config schema")
    local_layers = [
        layer
        for layer in base.get("layers") or []
        if layer.get("source_type") == "local_price_history_directory"
    ]
    source_layers = []
    blocked_layers = []
    provenance = []
    for source in source_inventory["sources"]:
        provenance.append(source["path"])
        if source["status"] == "blocked":
            blocked_layers.append({"layer_id": source["universe_id"], "reason": source["known_limitations"]})
            continue
        source_layers.append(_source_to_layer(source))

    overrides = _load_symbol_overrides(symbol_overrides_path, base.get("symbol_overrides") or [])
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "universe_version": universe_version,
        "snapshot_date": snapshot_date,
        "provenance": sorted(set([*base.get("provenance", []), *provenance])),
        "point_in_time_note": (
            "This is a current canonical universe snapshot assembled from versioned local source "
            "artifacts. It is not a historical point-in-time membership database."
        ),
        "layers": [*local_layers, *source_layers],
        "blocked_layers": blocked_layers,
        "symbol_overrides": overrides,
    }


def write_universe_bootstrap_run(
    run: Mapping[str, Any],
    *,
    artifact_root: str | Path = DEFAULT_UNIVERSE_ARTIFACT_ROOT,
    run_id: str,
    canonical_config_output: str | Path | None = None,
) -> Path:
    output_dir = Path(artifact_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", run["manifest"])
    _write_json(output_dir / "source_inventory.json", run["source_inventory"])
    _write_json(output_dir / "canonical_universe.json", run["canonical_universe"])
    _write_json(output_dir / "layer_summary.json", run["layer_summary"])
    _write_json(output_dir / "overlap_report.json", run["overlap_report"])
    _write_json(output_dir / "symbol_mapping_report.json", run["symbol_mapping_report"])
    _write_json(output_dir / "excluded_instruments.json", run["excluded_instruments"])
    _write_json(output_dir / "unsupported_symbol_mappings.json", run["unsupported_symbol_mappings"])
    (output_dir / "report.md").write_text(run["report"], encoding="utf-8")
    if canonical_config_output is not None:
        _write_json(Path(canonical_config_output), run["canonical_config"])
    return output_dir


def run_universe_bootstrap(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    base_config_path: str | Path = DEFAULT_CONFIG_PATH,
    price_history_root: str | Path = DEFAULT_PRICE_HISTORY_ROOT,
    artifact_root: str | Path = DEFAULT_UNIVERSE_ARTIFACT_ROOT,
    run_id: str,
    universe_version: str = "me-boot03-authoritative-universe-local-price-history-bootstrap-v1",
    snapshot_date: str = "2026-07-13",
    symbol_overrides_path: str | Path = DEFAULT_SYMBOL_OVERRIDES_PATH,
    canonical_config_output: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    run = build_universe_bootstrap_run(
        source_root=source_root,
        base_config_path=base_config_path,
        price_history_root=price_history_root,
        run_id=run_id,
        universe_version=universe_version,
        snapshot_date=snapshot_date,
        symbol_overrides_path=symbol_overrides_path,
    )
    output_dir = write_universe_bootstrap_run(
        run,
        artifact_root=artifact_root,
        run_id=run_id,
        canonical_config_output=canonical_config_output,
    )
    return run, output_dir


def render_universe_bootstrap_report(
    *,
    run_id: str,
    manifest: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    layer_summary: Mapping[str, Any],
    unsupported: Mapping[str, Any],
) -> str:
    rows = [
        "# Canonical Universe Bootstrap Report",
        "",
        f"Run ID: {run_id}",
        f"Generated at: {manifest['generated_at']}",
        f"Source count: {source_inventory['source_count']}",
        f"Canonical instruments: {snapshot['summary']['total_instruments']}",
        "",
        "## Layer Summary",
        "",
        "| Layer | Raw constituents | Unique instruments | Status |",
        "|---|---:|---:|---|",
    ]
    for layer in layer_summary["layers"]:
        rows.append(
            f"| {layer['layer_id']} | {layer['raw_constituents']} | "
            f"{layer['unique_instruments']} | {layer['status']} |"
        )
    rows.extend(
        [
            "",
            "## Unsupported Mappings",
            "",
            ", ".join(entry["symbol"] for entry in unsupported["entries"]) or "None",
            "",
            "## Limitations",
            "",
            "This run uses versioned local membership snapshots only. Partial layers remain partial and do not claim full index membership.",
            "No provider fallback, broker access, advice generation, or portfolio/watchlist mutation was performed.",
            "",
        ]
    )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO, stderr: TextIO) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        run, output_dir = run_universe_bootstrap(
            source_root=args.source_root,
            base_config_path=args.base_config,
            price_history_root=args.price_history_root,
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            universe_version=args.universe_version,
            snapshot_date=args.snapshot_date,
            symbol_overrides_path=args.symbol_overrides,
            canonical_config_output=args.write_canonical_config,
        )
    except (OSError, MarketDataUniverseError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "output_dir": output_dir.as_posix(),
                "canonical_instruments": run["canonical_universe"]["summary"]["total_instruments"],
                "source_count": run["source_inventory"]["source_count"],
            },
            indent=2,
            sort_keys=True,
        ),
        file=stdout,
    )
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Market Engine canonical universe from local source snapshots.")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT.as_posix())
    parser.add_argument("--base-config", default=DEFAULT_CONFIG_PATH.as_posix())
    parser.add_argument("--price-history-root", default=DEFAULT_PRICE_HISTORY_ROOT.as_posix())
    parser.add_argument("--artifact-root", default=DEFAULT_UNIVERSE_ARTIFACT_ROOT.as_posix())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--universe-version", default="me-boot03-authoritative-universe-local-price-history-bootstrap-v1")
    parser.add_argument("--snapshot-date", default="2026-07-13")
    parser.add_argument("--symbol-overrides", default=DEFAULT_SYMBOL_OVERRIDES_PATH.as_posix())
    parser.add_argument("--write-canonical-config", default=None)
    return parser


def _validate_source(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise MarketDataUniverseError(f"unsupported universe source schema: {path}")
    required = ("universe_id", "source_name", "snapshot_date", "retrieval_date", "provenance", "status")
    missing = [field for field in required if not payload.get(field)]
    if missing:
        raise MarketDataUniverseError(f"universe source missing required fields: {path}: {', '.join(missing)}")
    if payload["status"] not in {"active", "partial", "blocked"}:
        raise MarketDataUniverseError(f"unsupported universe source status: {path}: {payload['status']}")
    constituents = payload.get("constituents") or []
    seen = set()
    deduped = []
    duplicate_count = 0
    for raw in constituents:
        symbol = str(raw.get("symbol") or "").upper()
        if not symbol:
            raise MarketDataUniverseError(f"universe source constituent missing symbol: {path}")
        if symbol in seen:
            duplicate_count += 1
            continue
        seen.add(symbol)
        deduped.append(dict(raw, symbol=symbol))
    return {
        "path": path.as_posix(),
        "file_hash": _sha256_file(path),
        "universe_id": payload["universe_id"],
        "source_name": payload["source_name"],
        "snapshot_date": payload["snapshot_date"],
        "retrieval_date": payload["retrieval_date"],
        "provenance": payload["provenance"],
        "raw_constituent_count": len(constituents),
        "deduplicated_constituent_count": len(deduped),
        "duplicate_raw_entries": duplicate_count,
        "status": payload["status"],
        "asset_type": payload.get("asset_type") or "equity",
        "analysis_eligible": payload.get("analysis_eligible", True),
        "advice_eligible": payload.get("advice_eligible", True),
        "context_only": payload.get("context_only", False),
        "known_limitations": payload.get("known_limitations") or [],
        "constituents": deduped,
    }


def _source_to_layer(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "layer_id": source["universe_id"],
        "description": source["source_name"],
        "source_type": "explicit_instruments",
        "membership": source["universe_id"],
        "asset_type": source["asset_type"],
        "analysis_eligible": source["analysis_eligible"],
        "advice_eligible": source["advice_eligible"],
        "context_only": source["context_only"],
        "provenance": source["path"],
        "instruments": [
            {
                **constituent,
                "source_provenance": [
                    {
                        "source": source["source_name"],
                        "snapshot_date": source["snapshot_date"],
                        "path": source["path"],
                        "file_hash": source["file_hash"],
                    }
                ],
            }
            for constituent in source["constituents"]
        ],
    }


def _load_symbol_overrides(path: str | Path, fallback: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    override_path = Path(path)
    if not override_path.exists():
        return list(fallback)
    payload = json.loads(override_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "market-engine-symbol-overrides-v1":
        raise MarketDataUniverseError("unsupported symbol overrides schema")
    return list(payload.get("overrides") or [])


def _layer_summary(source_inventory: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    unique_by_layer = Counter(
        membership
        for entry in snapshot["instruments"]
        for membership in entry["universe_memberships"]
    )
    layers = []
    for source in source_inventory["sources"]:
        layers.append(
            {
                "layer_id": source["universe_id"],
                "source_name": source["source_name"],
                "raw_constituents": source["raw_constituent_count"],
                "deduplicated_constituents": source["deduplicated_constituent_count"],
                "duplicate_raw_entries": source["duplicate_raw_entries"],
                "unique_instruments": unique_by_layer.get(source["universe_id"], 0),
                "status": source["status"],
                "known_limitations": source["known_limitations"],
            }
        )
    return {"schema_version": "market-engine-canonical-universe-layer-summary-v1", "layers": layers}


def _overlap_report(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    for entry in snapshot["instruments"]:
        memberships = sorted(entry["universe_memberships"])
        for left_index, left in enumerate(memberships):
            for right in memberships[left_index + 1 :]:
                pair_counts[(left, right)] += 1
    return {
        "schema_version": "market-engine-canonical-universe-overlap-report-v1",
        "overlaps": [
            {"left_layer": left, "right_layer": right, "instrument_count": count}
            for (left, right), count in sorted(pair_counts.items())
        ],
    }


def _symbol_mapping_report(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    source_to_instruments: dict[str, list[str]] = defaultdict(list)
    for entry in snapshot["instruments"]:
        source_to_instruments[entry["source_symbol"]].append(entry["instrument_id"])
        entries.append(
            {
                "instrument_id": entry["instrument_id"],
                "symbol": entry["symbol"],
                "source_symbol": entry["source_symbol"],
                "asset_type": entry["asset_type"],
                "exchange": entry["exchange"],
                "country": entry["country"],
                "currency": entry["currency"],
                "source_mapping_status": entry.get("source_mapping_status"),
                "mapping_status": entry.get("mapping_status"),
                "memberships": entry["universe_memberships"],
                "notes": entry.get("source_notes"),
            }
        )
    duplicate_sources = {
        source: sorted(ids)
        for source, ids in source_to_instruments.items()
        if len(set(ids)) > 1
    }
    return {
        "schema_version": "market-engine-canonical-universe-symbol-mapping-report-v1",
        "entries": sorted(entries, key=lambda row: (row["source_mapping_status"] != "mapped", row["symbol"])),
        "duplicate_source_symbols": duplicate_sources,
    }


def _excluded_instruments(source_inventory: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for source in source_inventory["sources"]:
        if source["status"] == "blocked":
            for constituent in source["constituents"]:
                entries.append(
                    {
                        "symbol": constituent["symbol"],
                        "source": source["universe_id"],
                        "reason": "source_blocked",
                        "known_limitations": source["known_limitations"],
                    }
                )
    return {"schema_version": "market-engine-canonical-universe-excluded-instruments-v1", "entries": entries}


def _generated_at_from_run_id(run_id: str) -> str:
    marker = run_id.rsplit("-", 1)[-1]
    try:
        return datetime.strptime(marker, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
    except ValueError:
        return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
