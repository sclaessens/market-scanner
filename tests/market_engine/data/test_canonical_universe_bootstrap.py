from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from market_engine.data.canonical_universe_bootstrap import (
    SOURCE_SCHEMA_VERSION,
    build_source_inventory,
    build_universe_bootstrap_run,
    run_command,
)
from market_engine.data.local_market_data_universe import MarketDataUniverseError


def test_membership_sources_are_loaded_and_require_provenance(tmp_path: Path) -> None:
    source_root = _write_sources(tmp_path)
    inventory = build_source_inventory(source_root)

    assert inventory["source_count"] == 3
    assert {source["universe_id"] for source in inventory["sources"]} == {
        "nasdaq100",
        "sp400",
        "sp500",
    }
    assert all(source["snapshot_date"] == "2026-07-12" for source in inventory["sources"])
    assert all(source["file_hash"] for source in inventory["sources"])

    broken = source_root / "broken.json"
    broken.write_text(
        json.dumps(
            {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "universe_id": "broken",
                "source_name": "Broken source",
                "snapshot_date": "2026-07-12",
                "retrieval_date": "2026-07-13",
                "status": "active",
                "constituents": [{"symbol": "BAD"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MarketDataUniverseError, match="provenance"):
        build_source_inventory(source_root)


def test_bootstrap_run_preserves_overlap_and_symbol_mapping(tmp_path: Path) -> None:
    source_root = _write_sources(tmp_path)
    base_config = _write_base_config(tmp_path)
    overrides = _write_overrides(tmp_path)

    run = build_universe_bootstrap_run(
        source_root=source_root,
        base_config_path=base_config,
        price_history_root=tmp_path / "prices",
        run_id="me-boot03-test-20260713T120000Z",
        symbol_overrides_path=overrides,
    )

    snapshot = run["canonical_universe"]
    nvda = _instrument(snapshot, "NVDA")
    brk = _instrument(snapshot, "BRK.B")
    rhm = _instrument(snapshot, "RHM")

    assert {"sp500", "nasdaq100"} <= set(nvda["universe_memberships"])
    assert brk["source_symbol"] == "BRK-B"
    assert rhm["source_mapping_status"] == "unsupported"
    assert run["overlap_report"]["overlaps"]
    assert run["unsupported_symbol_mappings"]["entries"]
    assert run["manifest"]["guardrails"]["advice_generation_performed"] is False


def test_bootstrap_command_writes_universe_artifacts_and_config(tmp_path: Path) -> None:
    source_root = _write_sources(tmp_path)
    base_config = _write_base_config(tmp_path)
    overrides = _write_overrides(tmp_path)
    output_config = tmp_path / "canonical_universe.json"
    stdout = io.StringIO()

    exit_code = run_command(
        [
            "--source-root",
            source_root.as_posix(),
            "--base-config",
            base_config.as_posix(),
            "--price-history-root",
            (tmp_path / "prices").as_posix(),
            "--artifact-root",
            (tmp_path / "runs").as_posix(),
            "--run-id",
            "me-boot03-test-20260713T120000Z",
            "--symbol-overrides",
            overrides.as_posix(),
            "--write-canonical-config",
            output_config.as_posix(),
        ],
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    run_dir = tmp_path / "runs" / "me-boot03-test-20260713T120000Z"
    for filename in (
        "manifest.json",
        "source_inventory.json",
        "canonical_universe.json",
        "layer_summary.json",
        "overlap_report.json",
        "symbol_mapping_report.json",
        "excluded_instruments.json",
        "unsupported_symbol_mappings.json",
        "report.md",
    ):
        assert (run_dir / filename).exists()
    assert output_config.exists()
    assert json.loads(stdout.getvalue())["source_count"] == 3


def _instrument(snapshot: dict[str, object], symbol: str) -> dict[str, object]:
    matches = [entry for entry in snapshot["instruments"] if entry["symbol"] == symbol]
    assert len(matches) == 1
    return matches[0]


def _write_sources(tmp_path: Path) -> Path:
    root = tmp_path / "sources"
    root.mkdir()
    _write_source(
        root / "sp500.json",
        "sp500",
        [
            {"symbol": "NVDA", "name": "NVIDIA", "exchange": "NASDAQ"},
            {"symbol": "BRK.B", "name": "Berkshire Hathaway", "exchange": "NYSE"},
        ],
    )
    _write_source(
        root / "nasdaq100.json",
        "nasdaq100",
        [
            {"symbol": "NVDA", "name": "NVIDIA", "exchange": "NASDAQ"},
            {"symbol": "AMD", "name": "AMD", "exchange": "NASDAQ"},
        ],
    )
    _write_source(
        root / "sp400.json",
        "sp400",
        [
            {"symbol": "CLS", "name": "Celestica", "exchange": "NYSE"},
            {
                "symbol": "RHM",
                "source_symbol": "RHM.DE",
                "name": "Rheinmetall",
                "exchange": "XETRA",
                "country": "DE",
                "currency": "EUR",
                "source_mapping_status": "unsupported",
                "source_notes": "Primary-listing mapping requires operator validation.",
            },
        ],
    )
    return root


def _write_source(path: Path, universe_id: str, constituents: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "universe_id": universe_id,
                "source_name": f"{universe_id} test source",
                "snapshot_date": "2026-07-12",
                "retrieval_date": "2026-07-13",
                "provenance": "unit test source",
                "status": "partial",
                "known_limitations": ["partial test source"],
                "constituents": constituents,
            }
        ),
        encoding="utf-8",
    )


def _write_base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "market-engine-canonical-local-market-data-universe-config-v1",
                "universe_version": "base",
                "snapshot_date": "2026-07-12",
                "provenance": ["base"],
                "point_in_time_note": "base",
                "layers": [
                    {
                        "layer_id": "local_price_history_covered",
                        "source_type": "local_price_history_directory",
                        "membership": "local_price_history_covered",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_overrides(tmp_path: Path) -> Path:
    path = tmp_path / "symbol_overrides.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "market-engine-symbol-overrides-v1",
                "overrides": [
                    {"canonical_symbol": "BRK.B", "source_symbol": "BRK-B", "reason": "test"},
                    {
                        "canonical_symbol": "RHM",
                        "source_symbol": "RHM.DE",
                        "source_mapping_status": "unsupported",
                        "mapping_status": "unsupported",
                        "reason": "test",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path
