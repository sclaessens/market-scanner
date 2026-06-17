from __future__ import annotations

import json
from pathlib import Path

from market_engine.run.end_to_end_dry_run_command import (
    run_market_engine_end_to_end_dry_run_command,
)
from market_engine.run.local_dry_run_artifacts import (
    MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION,
    MARKET_ENGINE_LOCAL_DRY_RUN_MANIFEST_FORMAT_VERSION,
)
from market_engine.run.local_dry_run_inputs import (
    MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION,
    load_market_engine_local_dry_run_input,
)


FIXTURE_PATH = Path(
    "tests/fixtures/market_engine/run/"
    "me_run07_realistic_local_snapshot_fixture.json"
)


EXPECTED_MISSING_MARKERS = {
    "fundamental_observations.segment_revenue_breakdown",
    "setup_detection.forward_guidance_not_in_fixture",
}

EXPECTED_STALE_MARKERS = {
    "sec_companyfacts.snapshot_age.review_required",
    "analysis_review.market_price_snapshot_absent",
    "delivery_reporting.local_fixture_review_timestamp",
}

EXPECTED_BLOCKED_REASON = (
    "ME-RUN07 fixture intentionally blocks before any delivery channel."
)


def test_me_run07_fixture_wrapper_is_accepted() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert fixture["dry_run_input_fixture_format_version"] == (
        MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION
    )
    assert fixture["input_mode"] == "local_snapshot_fixture"
    assert fixture["non_production_fixture"] is True

    stage_payloads = load_market_engine_local_dry_run_input(
        FIXTURE_PATH,
        input_mode="local_snapshot_fixture",
    )

    assert stage_payloads["source_context"]["ticker"] == "MSFT"
    assert stage_payloads["source_context"]["cik"] == "0000789019"
    assert stage_payloads["delivery_reporting"]["blocked_reasons"] == [
        EXPECTED_BLOCKED_REASON
    ]


def test_me_run07_fixture_runs_end_to_end_and_preserves_review_evidence(capsys) -> None:
    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            "local_snapshot_fixture",
            "--stage-payloads-json",
            str(FIXTURE_PATH),
            "--dry-run-id",
            "me-run07-realistic-local-fixture-dry-run",
            "--generated-at",
            "2026-06-17T14:15:00Z",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["dry_run_id"] == "me-run07-realistic-local-fixture-dry-run"
    assert payload["input_mode"] == "local_snapshot_fixture"
    assert payload["ticker"] == "MSFT"
    assert payload["cik"] == "0000789019"
    assert payload["run_state"] == "dry_run_blocked"
    assert payload["blocked_stage"] == "delivery_reporting"
    assert payload["blocked_reasons"] == [EXPECTED_BLOCKED_REASON]
    assert EXPECTED_MISSING_MARKERS.issubset(set(payload["missing_data_summary"]))
    assert EXPECTED_STALE_MARKERS.issubset(set(payload["stale_data_summary"]))
    assert payload["numeric_zero_evidence_summary"][
        "portfolio_review.portfolio_context_reference.current_quantity"
    ] == 0
    assert payload["numeric_zero_evidence_summary"][
        "portfolio_review.portfolio_context_reference.current_market_value"
    ] == 0.0
    assert payload["numeric_zero_evidence_summary"][
        "portfolio_review.portfolio_context_reference.cash_available_for_review"
    ] == 0
    assert payload["provenance_summary"]["delivery_reporting"]["report_id"] == (
        "me-run07-delivery-report-msft-001"
    )
    assert payload["delivery_report_reference"]["report_id"] == (
        "me-run07-delivery-report-msft-001"
    )
    assert "No provider" in payload["forbidden_side_effect_confirmation"]
    assert "Decision Engine remains" in payload["authority_boundary_confirmation"]


def test_me_run07_artifact_is_written_only_when_explicitly_requested(
    tmp_path: Path,
    capsys,
) -> None:
    no_artifact_run_id = "me-run07-no-artifact"
    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            "local_snapshot_fixture",
            "--stage-payloads-json",
            str(FIXTURE_PATH),
            "--dry-run-id",
            no_artifact_run_id,
            "--generated-at",
            "2026-06-17T14:15:00Z",
            "--artifact-output-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert not (tmp_path / no_artifact_run_id).exists()

    artifact_run_id = "me-run07-realistic-local-fixture-artifact"
    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            "local_snapshot_fixture",
            "--stage-payloads-json",
            str(FIXTURE_PATH),
            "--dry-run-id",
            artifact_run_id,
            "--generated-at",
            "2026-06-17T14:15:00Z",
            "--write-local-artifact",
            "--artifact-output-root",
            str(tmp_path),
            "--artifact-created-at",
            "2026-06-17T14:30:00Z",
        ]
    )

    captured = capsys.readouterr()
    emitted_payload = json.loads(captured.out)
    run_directory = tmp_path / artifact_run_id
    manifest_path = run_directory / "manifest.json"
    artifact_path = (
        run_directory
        / "artifacts"
        / "market_engine_dry_run_"
        "me-run07-realistic-local-fixture-artifact_2026-06-17.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured.err == ""
    assert emitted_payload["dry_run_id"] == artifact_run_id
    assert manifest["manifest_format_version"] == (
        MARKET_ENGINE_LOCAL_DRY_RUN_MANIFEST_FORMAT_VERSION
    )
    assert manifest["artifact_count"] == 1
    assert manifest["non_production_artifact"] is True
    assert artifact["artifact_format_version"] == (
        MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION
    )
    assert artifact["non_production_artifact"] is True
    assert artifact["source_dry_run_id"] == artifact_run_id
    assert artifact["source_input_mode"] == "local_snapshot_fixture"
    assert artifact["source_run_state"] == "dry_run_blocked"
    assert artifact["payload"]["blocked_stage"] == "delivery_reporting"
    assert artifact["payload"]["blocked_reasons"] == [EXPECTED_BLOCKED_REASON]
    assert EXPECTED_MISSING_MARKERS.issubset(
        set(artifact["payload"]["missing_data_summary"])
    )
    assert EXPECTED_STALE_MARKERS.issubset(
        set(artifact["payload"]["stale_data_summary"])
    )
    assert artifact["payload"]["numeric_zero_evidence_summary"][
        "portfolio_review.portfolio_context_reference.current_market_value"
    ] == 0.0
    assert artifact["payload"]["provenance_summary"]["source_context"][
        "source_refresh_snapshot_id"
    ] == "me-run07-source-snapshot-msft-001"
