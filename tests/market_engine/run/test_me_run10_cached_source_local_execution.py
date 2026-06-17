from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_engine.run.cached_source_execution import (
    CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
    MARKET_ENGINE_CACHED_SOURCE_LOCAL_EXECUTION_INPUT_FORMAT_VERSION,
    CachedSourceLocalExecutionError,
    build_cached_source_local_execution_stage_payloads,
    load_cached_source_local_execution_stage_payloads,
)
from market_engine.run.end_to_end_dry_run_command import (
    run_market_engine_end_to_end_dry_run_command,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    persist_sec_companyfacts_raw_snapshot,
)


def test_cached_source_local_execution_builds_dry_run_payload(tmp_path: Path, capsys) -> None:
    source_root = tmp_path / "source_snapshots"
    snapshot_path = _persist_snapshot(source_root)
    portfolio_context_path = _portfolio_context_path(tmp_path)

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
            "--source-snapshot-json",
            str(snapshot_path),
            "--source-snapshot-root",
            str(source_root),
            "--portfolio-context-json",
            str(portfolio_context_path),
            "--dry-run-id",
            "cached-run-001",
            "--generated-at",
            "2026-06-17T15:00:00Z",
            "--compact",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["dry_run_format_version"] == "market-engine-end-to-end-dry-run-v1"
    assert payload["input_mode"] == CACHED_SOURCE_SNAPSHOT_INPUT_MODE
    assert payload["run_state"] == "dry_run_completed"
    assert payload["ticker"] == "NVDA"
    assert payload["cik"] == "0001045810"
    assert payload["provenance_summary"]["source_context"]["cached_source_reference"][
        "source_snapshot_path"
    ] == snapshot_path.resolve().as_posix()
    assert payload["provenance_summary"]["source_context"]["source_refresh_snapshot_id"] == (
        "NVDA_companyfacts"
    )
    assert payload["numeric_zero_evidence_summary"][
        "portfolio_review.portfolio_context_provenance.current_quantity"
    ] == 0
    assert payload["delivery_report_reference"]["source_handoff_run_id"] == (
        "cached-run-001-decision-engine-handoff"
    )


def test_missing_cached_source_fails_closed(tmp_path: Path, capsys) -> None:
    missing_snapshot_path = tmp_path / "source_snapshots" / "missing.json"

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
            "--source-snapshot-json",
            str(missing_snapshot_path),
            "--source-snapshot-root",
            str(tmp_path / "source_snapshots"),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "cannot build SEC CompanyFacts Source Context from snapshot" in captured.err


def test_malformed_cached_source_fails_closed(tmp_path: Path, capsys) -> None:
    source_root = tmp_path / "source_snapshots"
    snapshot_path = source_root / "bad.json"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_text("{not-json", encoding="utf-8")

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
            "--source-snapshot-json",
            str(snapshot_path),
            "--source-snapshot-root",
            str(source_root),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "invalid JSON" in captured.err


def test_live_provider_input_mode_is_rejected() -> None:
    with pytest.raises(SystemExit):
        run_market_engine_end_to_end_dry_run_command(
            ["--input-mode", "live_provider_fetch"]
        )


def test_cached_source_artifact_writing_is_explicit(tmp_path: Path, capsys) -> None:
    source_root = tmp_path / "source_snapshots"
    artifact_root = tmp_path / "artifacts"
    snapshot_path = _persist_snapshot(source_root)
    portfolio_context_path = _portfolio_context_path(tmp_path)

    base_args = [
        "--input-mode",
        CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
        "--source-snapshot-json",
        str(snapshot_path),
        "--source-snapshot-root",
        str(source_root),
        "--portfolio-context-json",
        str(portfolio_context_path),
        "--dry-run-id",
        "cached-artifact-run-001",
        "--generated-at",
        "2026-06-17T15:00:00Z",
        "--artifact-output-root",
        str(artifact_root),
        "--artifact-created-at",
        "2026-06-17T15:01:00Z",
    ]

    assert run_market_engine_end_to_end_dry_run_command(base_args) == 0
    assert not (artifact_root / "cached-artifact-run-001").exists()

    capsys.readouterr()

    assert run_market_engine_end_to_end_dry_run_command(
        [*base_args, "--write-local-artifact"]
    ) == 0
    captured = capsys.readouterr()

    manifest_path = artifact_root / "cached-artifact-run-001" / "manifest.json"
    artifact_path = (
        artifact_root
        / "cached-artifact-run-001"
        / "artifacts"
        / "market_engine_dry_run_cached-artifact-run-001_2026-06-17.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert captured.err == ""
    assert manifest["source_dry_run_id"] == "cached-artifact-run-001"
    assert artifact["payload"]["input_mode"] == CACHED_SOURCE_SNAPSHOT_INPUT_MODE
    assert artifact["payload"]["provenance_summary"]["source_context"][
        "cached_source_reference"
    ]["source_snapshot_path"] == snapshot_path.resolve().as_posix()


def test_cached_source_wrapper_input_is_supported(tmp_path: Path) -> None:
    source_root = tmp_path / "source_snapshots"
    snapshot_path = _persist_snapshot(source_root)
    wrapper_path = tmp_path / "cached_source_input.json"
    wrapper_path.write_text(
        json.dumps(
            {
                "cached_source_local_execution_input_format_version": (
                    MARKET_ENGINE_CACHED_SOURCE_LOCAL_EXECUTION_INPUT_FORMAT_VERSION
                ),
                "input_mode": CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
                "non_production_local_execution": True,
                "source_snapshot_path": snapshot_path.as_posix(),
                "source_snapshot_root": source_root.as_posix(),
                "portfolio_context": _portfolio_context_payload(),
            }
        ),
        encoding="utf-8",
    )

    stage_payloads = load_cached_source_local_execution_stage_payloads(
        wrapper_path,
        dry_run_id="cached-wrapper-run-001",
        generated_at="2026-06-17T15:00:00Z",
    )

    assert stage_payloads["source_context"]["source_context_format_version"] == (
        "sec-companyfacts-source-context-v1"
    )
    assert stage_payloads["delivery_reporting"]["report_format_version"] == (
        "market-engine-delivery-report-v1"
    )


def test_cached_source_path_must_stay_inside_configured_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source_snapshots"
    other_root = tmp_path / "other"
    snapshot_path = _persist_snapshot(other_root)

    with pytest.raises(CachedSourceLocalExecutionError, match="must stay under"):
        build_cached_source_local_execution_stage_payloads(
            source_snapshot_path=snapshot_path,
            source_snapshot_root=source_root,
            dry_run_id="cached-run-001",
            generated_at="2026-06-17T15:00:00Z",
            portfolio_context_payload=_portfolio_context_payload(),
        )


def test_cached_source_execution_module_has_no_side_effect_dependencies() -> None:
    module_source = Path("src/market_engine/run/cached_source_execution.py").read_text(
        encoding="utf-8"
    )

    forbidden_terms = (
        "from scripts",
        "import scripts",
        "from market_scanner",
        "import market_scanner",
        "telegram",
        "smtplib",
        "yfinance",
        "urllib",
        "requests",
        "socket",
        "subprocess",
    )

    assert not any(term in module_source for term in forbidden_terms)


def _persist_snapshot(source_root: Path) -> Path:
    return persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root,
        run_id="20260617T150000Z",
        ticker="NVDA",
        cik="0001045810",
        raw_payload=_companyfacts_payload(),
        fetched_at="2026-06-17T15:00:00Z",
    )


def _portfolio_context_path(tmp_path: Path) -> Path:
    path = tmp_path / "portfolio_context.json"
    path.write_text(json.dumps(_portfolio_context_payload()), encoding="utf-8")
    return path


def _portfolio_context_payload() -> dict[str, object]:
    return {
        "portfolio_context_format_version": "market-engine-portfolio-context-v1",
        "portfolio_context_run_id": "portfolio-context-run-001",
        "portfolio_snapshot_timestamp": "2026-06-17T14:55:00Z",
        "portfolio_base_currency": "USD",
        "ticker": "NVDA",
        "position_state": "not_held",
        "current_quantity": 0,
        "current_market_value": 0.0,
        "portfolio_total_value": 100000.0,
        "current_ticker_exposure_pct": 0,
        "exposure_buckets": {"technology": 0},
        "concentration_thresholds": {"single_ticker_review_pct": 10},
        "policy_constraints": {},
        "missing_portfolio_context_fields": [],
        "stale_portfolio_context_fields": [],
        "context_provenance": {
            "portfolio_context_source": "local_non_production_fixture"
        },
    }


def _companyfacts_payload() -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_fact(100, "2025-12-31")]}},
                "NetIncomeLoss": {"units": {"USD": [_fact(20, "2025-12-31")]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_fact(30, "2025-12-31")]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_fact(5, "2025-12-31")]}
                },
            }
        }
    }


def _fact(value: int | None, end: str) -> dict[str, object]:
    return {
        "val": value,
        "fy": int(end[:4]),
        "fp": "FY",
        "form": "10-K",
        "filed": f"{int(end[:4]) + 1}-02-15",
        "start": f"{end[:4]}-01-01",
        "end": end,
        "accn": f"0000000000-{end[:4]}-000001",
        "frame": f"CY{end[:4]}",
    }
