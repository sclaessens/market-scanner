from __future__ import annotations

import hashlib
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


def test_company_profile_cached_source_is_consumed_into_source_context(
    tmp_path: Path,
    capsys,
) -> None:
    source_root = tmp_path / "source_snapshots"
    snapshot_path = _write_company_profile_package(source_root)

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
            "--source-snapshot-json",
            str(snapshot_path),
            "--source-snapshot-root",
            str(source_root),
            "--dry-run-id",
            "company-profile-gate-run-001",
        ]
    )

    captured = capsys.readouterr()

    dry_run = json.loads(captured.out)
    company_profile = dry_run["provenance_summary"]["source_context"][
        "company_profile"
    ]
    profile_observations = dry_run["provenance_summary"][
        "fundamental_observations"
    ]["company_profile"]
    analysis_context = dry_run["provenance_summary"]["analysis_review"][
        "company_profile"
    ]
    stage_results = {
        stage["stage_name"]: stage for stage in dry_run["stage_results"]
    }

    assert exit_code == 0
    assert captured.err == ""
    assert dry_run["ticker"] == "NVDA"
    assert dry_run["run_state"] == "dry_run_blocked"
    assert dry_run["blocked_stage"] == "recommendation_review"
    assert stage_results["source_context"]["status"] == "completed"
    assert stage_results["fundamental_observations"]["status"] == "completed"
    assert stage_results["derived_observations"]["status"] == "completed"
    assert stage_results["setup_detection"]["status"] == "completed"
    assert stage_results["analysis_review"]["status"] == "completed"
    assert stage_results["recommendation_review"]["status"] == "blocked"
    assert company_profile["input_family"] == "company_profile"
    assert company_profile["consumption_state"] == "consumed"
    assert company_profile["symbol"] == "NVDA"
    assert company_profile["profile"]["business_summary"].startswith("Deterministic")
    assert company_profile["compatibility_gate"]["allowed"] is True
    assert company_profile["compatibility_gate"]["result"] == (
        "company_profile_consumption_allowed"
    )
    assert profile_observations["input_family"] == "company_profile"
    assert profile_observations["source_context_state"] == "consumed"
    assert profile_observations["observation_format_version"] == (
        "market-engine-company-profile-fundamental-observations-v1"
    )
    assert {
        observation["observation_code"]
        for observation in profile_observations["observations"]
    } >= {
        "company_profile_identity_observed",
        "company_profile_symbol_observed",
        "company_profile_exchange_observed",
        "company_profile_sector_observed",
        "company_profile_industry_observed",
        "company_profile_country_observed",
        "company_profile_currency_observed",
        "company_profile_description_available",
        "company_profile_provenance_retained",
        "company_profile_as_of_retained",
    }
    assert analysis_context["context_state"] == "descriptive_context_available"
    assert analysis_context["symbol"] == "NVDA"
    assert {
        item["context_type"] for item in analysis_context["descriptive_context"]
    } >= {
        "company_identity_context",
        "symbol_context",
        "exchange_context",
        "sector_context",
        "industry_context",
        "country_context",
        "currency_context",
        "description_availability_context",
        "website_context",
        "provenance_context",
        "as_of_context",
    }
    assert "blocked_company_profile_consumption_not_implemented" not in captured.out


def test_company_profile_cached_source_missing_manifest_is_traceably_blocked(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    snapshot_dir = source_root / "NVDA" / "company_profile"
    snapshot_dir.mkdir(parents=True)
    snapshot_path = snapshot_dir / "company_profile.json"
    snapshot_path.write_text(json.dumps(_company_profile_payload()), encoding="utf-8")

    stage_payloads = build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=source_root,
        dry_run_id="company-profile-gate-run-002",
        generated_at="2026-06-26T12:00:00Z",
    )

    source_context = stage_payloads["source_context"]
    assert source_context["consumption_state"] == "blocked"
    assert "profile" not in source_context["company_profile"]
    assert "company_profile" not in stage_payloads["fundamental_observations"]
    assert "company_profile" not in stage_payloads["analysis_review"]
    assert "blocked_missing_company_profile_manifest" in source_context[
        "blocked_reasons"
    ]


def test_company_profile_cached_source_provider_call_provenance_is_blocked(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    payload = _company_profile_payload()
    payload["provenance"]["request_metadata"]["provider_calls_performed"] = True
    snapshot_path = _write_company_profile_package(source_root, payload=payload)

    stage_payloads = build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=source_root,
        dry_run_id="company-profile-gate-run-003",
        generated_at="2026-06-26T12:00:00Z",
    )

    assert (
        "blocked_company_profile_network_dependency: provider_calls_performed"
        in stage_payloads["source_context"]["blocked_reasons"]
    )


def test_company_profile_cached_source_manifest_payload_ticker_mismatch_is_blocked(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    snapshot_path = _write_company_profile_package(
        source_root,
        manifest_overrides={"ticker": "AMD"},
    )

    stage_payloads = build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=source_root,
        dry_run_id="company-profile-gate-run-004",
        generated_at="2026-06-26T12:00:00Z",
    )

    source_context = stage_payloads["source_context"]
    assert source_context["consumption_state"] == "blocked"
    assert (
        "blocked_ambiguous_company_profile_identity: ticker_mismatch"
        in source_context["blocked_reasons"]
    )
    assert "company_profile" not in stage_payloads["fundamental_observations"]


def test_company_profile_cached_source_package_ticker_mismatch_is_blocked(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    payload = _company_profile_payload()
    payload["ticker"] = "AMD"
    payload["provenance"]["canonical_source_identity"] = "fake://company_profile/AMD"
    snapshot_path = _write_company_profile_package(
        source_root,
        payload=payload,
        manifest_overrides={
            "ticker": "AMD",
            "source_url": "fake://company_profile/AMD",
        },
    )

    stage_payloads = build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=source_root,
        dry_run_id="company-profile-gate-run-package-mismatch",
        generated_at="2026-06-26T12:00:00Z",
    )

    assert (
        "blocked_ambiguous_company_profile_identity: package_ticker_mismatch"
        in stage_payloads["source_context"]["blocked_reasons"]
    )


def test_company_profile_invalid_format_is_not_consumed(tmp_path: Path) -> None:
    source_root = tmp_path / "source_snapshots"
    payload = _company_profile_payload()
    payload["payload_format"] = "unknown-company-profile-format"
    snapshot_path = _write_company_profile_package(source_root, payload=payload)

    stage_payloads = build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=source_root,
        dry_run_id="company-profile-gate-run-005",
        generated_at="2026-06-26T12:00:00Z",
    )

    source_context = stage_payloads["source_context"]
    assert source_context["consumption_state"] == "blocked"
    assert "profile" not in source_context["company_profile"]
    assert "company_profile" not in stage_payloads["fundamental_observations"]
    assert "company_profile" not in stage_payloads["analysis_review"]
    assert (
        "blocked_malformed_company_profile_payload: payload_format"
        in source_context["blocked_reasons"]
    )


def test_company_profile_missing_provenance_is_not_consumed(tmp_path: Path) -> None:
    source_root = tmp_path / "source_snapshots"
    payload = _company_profile_payload()
    payload.pop("provenance")
    snapshot_path = _write_company_profile_package(source_root, payload=payload)

    stage_payloads = build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=source_root,
        dry_run_id="company-profile-gate-run-006",
        generated_at="2026-06-26T12:00:00Z",
    )

    assert "blocked_malformed_company_profile_payload: provenance" in stage_payloads[
        "source_context"
    ]["blocked_reasons"]
    assert "company_profile" not in stage_payloads["fundamental_observations"]
    assert "company_profile" not in stage_payloads["analysis_review"]


def test_company_profile_unsupported_profile_field_is_not_consumed(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    payload = _company_profile_payload()
    payload["profile"]["target_price"] = 200
    snapshot_path = _write_company_profile_package(source_root, payload=payload)

    stage_payloads = build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=source_root,
        dry_run_id="company-profile-gate-run-unsupported-field",
        generated_at="2026-06-26T12:00:00Z",
    )

    assert (
        "blocked_malformed_company_profile_payload: "
        "unsupported_profile_fields=target_price"
    ) in stage_payloads["source_context"]["blocked_reasons"]


def test_malformed_company_profile_payload_returns_blocked_state(
    tmp_path: Path,
    capsys,
) -> None:
    source_root = tmp_path / "source_snapshots"
    snapshot_path = source_root / "NVDA" / "company_profile" / "company_profile.json"
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
    dry_run = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert dry_run["blocked_stage"] == "source_context"
    assert "blocked_malformed_company_profile_payload" in dry_run["blocked_reasons"]


def test_stale_company_profile_manifest_is_not_consumed(tmp_path: Path) -> None:
    source_root = tmp_path / "source_snapshots"
    snapshot_path = _write_company_profile_package(
        source_root,
        manifest_overrides={"staleness_status": "stale"},
    )

    stage_payloads = build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=source_root,
        dry_run_id="company-profile-gate-run-007",
        generated_at="2026-06-26T12:00:00Z",
    )

    assert (
        "blocked_company_profile_non_consumable_timestamp: staleness_status"
        in stage_payloads["source_context"]["blocked_reasons"]
    )


def test_unknown_company_profile_source_timestamp_is_consumed_with_note(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    payload = _company_profile_payload()
    payload["provenance"]["source_timestamp"] = None
    snapshot_path = _write_company_profile_package(source_root, payload=payload)

    stage_payloads = build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=source_root,
        dry_run_id="company-profile-gate-run-008",
        generated_at="2026-06-26T12:00:00Z",
    )

    source_context = stage_payloads["source_context"]
    assert source_context["consumption_state"] == "consumed"
    assert source_context["stale_data_markers"] == (
        "source_context.company_profile.source_timestamp_unknown",
    )


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


def test_company_profile_is_visible_in_written_dry_run_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    source_root = tmp_path / "source_snapshots"
    artifact_root = tmp_path / "artifacts"
    snapshot_path = _write_company_profile_package(source_root)

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
            "--source-snapshot-json",
            str(snapshot_path),
            "--source-snapshot-root",
            str(source_root),
            "--dry-run-id",
            "company-profile-artifact-run",
            "--generated-at",
            "2026-06-26T12:00:00Z",
            "--write-local-artifact",
            "--artifact-output-root",
            str(artifact_root),
            "--artifact-created-at",
            "2026-06-26T12:01:00Z",
        ]
    )

    capsys.readouterr()
    artifact_path = (
        artifact_root
        / "company-profile-artifact-run"
        / "artifacts"
        / "market_engine_dry_run_company-profile-artifact-run_2026-06-26.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    company_profile = artifact["payload"]["provenance_summary"]["source_context"][
        "company_profile"
    ]
    profile_observations = artifact["payload"]["provenance_summary"][
        "fundamental_observations"
    ]["company_profile"]
    analysis_context = artifact["payload"]["provenance_summary"]["analysis_review"][
        "company_profile"
    ]

    assert exit_code == 0
    assert company_profile["consumption_state"] == "consumed"
    assert company_profile["profile"]["business_summary"].startswith("Deterministic")
    assert profile_observations["source_context_state"] == "consumed"
    assert profile_observations["observations"]
    assert analysis_context["context_state"] == "descriptive_context_available"
    assert analysis_context["descriptive_context"]


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


def test_sec_companyfacts_source_context_marks_company_profile_absent_optional(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    snapshot_path = _persist_snapshot(source_root)

    stage_payloads = build_cached_source_local_execution_stage_payloads(
        source_snapshot_path=snapshot_path,
        source_snapshot_root=source_root,
        dry_run_id="cached-run-without-profile",
        generated_at="2026-06-17T15:00:00Z",
    )

    assert stage_payloads["source_context"]["company_profile"] == {
        "input_family": "company_profile",
        "consumption_state": "absent_optional",
        "consumption_reason_codes": ("company_profile_absent_optional",),
    }
    assert "company_profile" not in stage_payloads["fundamental_observations"]
    assert "company_profile" not in stage_payloads["analysis_review"]


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


def _write_company_profile_package(
    source_root: Path,
    *,
    payload: dict[str, object] | None = None,
    manifest_overrides: dict[str, object] | None = None,
) -> Path:
    snapshot_dir = source_root / "NVDA" / "company_profile"
    snapshot_dir.mkdir(parents=True)
    snapshot_path = snapshot_dir / "company_profile.json"
    profile_payload = payload or _company_profile_payload()
    snapshot_path.write_text(
        json.dumps(profile_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "manifest_format_version": "market-engine-cached-source-acquisition-manifest-v1",
        "snapshot_id": "NVDA-company_profile-company-profile-gate-run",
        "batch_id": "company-profile-gate-run",
        "created_at_utc": "2026-06-26T12:00:00Z",
        "acquired_at_utc": "2026-06-26T12:00:00Z",
        "acquisition_mode": "automated_dry_run",
        "source_family": "company_profile",
        "source_name": "deterministic_fake_provider",
        "source_url": "fake://company_profile/NVDA",
        "local_use_allowed": True,
        "commit_allowed": False,
        "ticker": "NVDA",
        "local_snapshot_path": "company_profile.json",
        "local_manifest_path": "manifest.json",
        "local_payload_sha256": _file_sha256(snapshot_path),
        "local_payload_size_bytes": snapshot_path.stat().st_size,
        "validation_status": "passed",
        "validation_errors": [],
        "validation_warnings": [],
        "staleness_status": "fresh",
        "staleness_reason": "Deterministic test fixture.",
        "usable_for_cached_source_dry_run": True,
        "blocked_reason": None,
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot_path


def _company_profile_payload() -> dict[str, object]:
    return {
        "payload_format": "market-engine-company-profile-snapshot-v1",
        "ticker": "NVDA",
        "entity_name": "NVIDIA Corporation",
        "entity_country": "US",
        "entity_exchange": "NASDAQ",
        "source_family": "company_profile",
        "profile": {
            "business_summary": "Deterministic non-production company profile for NVDA.",
            "sector": "Technology",
            "industry": "Semiconductors",
            "currency": "USD",
            "website": "https://example.invalid/nvda",
            "missing_data": [],
        },
        "provenance": {
            "adapter_id": "fake_company_profile_adapter",
            "adapter_version": "test-v1",
            "provider_name": "deterministic_fake_provider",
            "canonical_source_identity": "fake://company_profile/NVDA",
            "retrieved_at": "2026-06-26T12:00:00Z",
            "source_timestamp": "2026-06-26T11:59:00Z",
            "request_metadata": {
                "network_used": False,
                "provider_calls_performed": False,
                "deterministic_fake_adapter": True,
            },
        },
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
