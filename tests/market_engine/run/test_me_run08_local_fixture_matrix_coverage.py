from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from market_engine.run.end_to_end_dry_run_command import (
    build_synthetic_dry_run_stage_payloads,
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


MATRIX_CASES = (
    {
        "case_id": "completed",
        "ticker": "NVDA",
        "expected_run_state": "dry_run_completed",
        "expected_blocked_stage": None,
        "expected_blocked_reasons": [],
        "expected_missing": [],
        "expected_stale": [],
    },
    {
        "case_id": "completed-with-limitations",
        "ticker": "AMD",
        "expected_run_state": "dry_run_completed_with_limitations",
        "expected_blocked_stage": None,
        "expected_blocked_reasons": [],
        "expected_missing": [
            "fundamental_observations.revenue_segment_breakdown",
        ],
        "expected_stale": [
            "source_context.sec_companyfacts.snapshot_review_required",
        ],
    },
    {
        "case_id": "blocked",
        "ticker": "META",
        "expected_run_state": "dry_run_blocked",
        "expected_blocked_stage": "setup_detection",
        "expected_blocked_reasons": [
            "ME-RUN08 blocked fixture stops before review layers.",
        ],
        "expected_missing": [],
        "expected_stale": [],
    },
    {
        "case_id": "stale-data",
        "ticker": "COST",
        "expected_run_state": "dry_run_completed_with_limitations",
        "expected_blocked_stage": None,
        "expected_blocked_reasons": [],
        "expected_missing": [],
        "expected_stale": [
            "analysis_review.local_price_snapshot_absent",
            "delivery_reporting.local_review_timestamp_stale",
        ],
    },
    {
        "case_id": "missing-data",
        "ticker": "ASML",
        "expected_run_state": "dry_run_completed_with_limitations",
        "expected_blocked_stage": None,
        "expected_blocked_reasons": [],
        "expected_missing": [
            "source_context.companyfacts.capital_expenditures_unavailable",
            "setup_detection.forward_guidance_not_in_fixture",
        ],
        "expected_stale": [],
    },
    {
        "case_id": "numeric-zero",
        "ticker": "TSLA",
        "expected_run_state": "dry_run_completed",
        "expected_blocked_stage": None,
        "expected_blocked_reasons": [],
        "expected_missing": [],
        "expected_stale": [],
        "expected_zero_paths": {
            "fundamental_observations.net_income": 0,
            "derived_observations.free_cash_flow": 0,
            "portfolio_review.portfolio_context_reference.current_quantity": 0,
            "portfolio_review.portfolio_context_reference.current_market_value": 0.0,
            "portfolio_review.portfolio_context_reference.cash_available_for_review": 0,
        },
    },
    {
        "case_id": "unsupported-input",
        "ticker": "AVGO",
        "expected_run_state": "dry_run_unsupported_input",
        "expected_blocked_stage": "analysis_review",
        "expected_blocked_reasons": [
            "Analysis Review contract version is unsupported.",
        ],
        "expected_missing": [],
        "expected_stale": [],
    },
    {
        "case_id": "provenance-heavy",
        "ticker": "MSFT",
        "expected_run_state": "dry_run_completed",
        "expected_blocked_stage": None,
        "expected_blocked_reasons": [],
        "expected_missing": [],
        "expected_stale": [],
        "expected_provenance": {
            "source_context": "source_provenance_reference",
            "recommendation_review": "input_provenance",
            "delivery_reporting": "upstream_provenance_summary",
        },
    },
)


@pytest.mark.parametrize("case", MATRIX_CASES, ids=[case["case_id"] for case in MATRIX_CASES])
def test_me_run08_local_snapshot_fixture_matrix_covers_dry_run_states(
    case: dict[str, Any],
    tmp_path: Path,
    capsys,
) -> None:
    fixture_path = _write_fixture_case(tmp_path, case)

    loaded = load_market_engine_local_dry_run_input(
        fixture_path,
        input_mode="local_snapshot_fixture",
    )
    assert loaded["source_context"]["ticker"] == case["ticker"]

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            "local_snapshot_fixture",
            "--stage-payloads-json",
            str(fixture_path),
            "--dry-run-id",
            f"me-run08-{case['case_id']}",
            "--generated-at",
            "2026-06-17T15:00:00Z",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload["dry_run_id"] == f"me-run08-{case['case_id']}"
    assert payload["input_mode"] == "local_snapshot_fixture"
    assert payload["ticker"] == case["ticker"]
    assert payload["run_state"] == case["expected_run_state"]
    assert payload["blocked_stage"] == case["expected_blocked_stage"]
    assert payload["blocked_reasons"] == case["expected_blocked_reasons"]
    assert set(case["expected_missing"]).issubset(set(payload["missing_data_summary"]))
    assert set(case["expected_stale"]).issubset(set(payload["stale_data_summary"]))
    assert "No provider" in payload["forbidden_side_effect_confirmation"]
    assert "Decision Engine remains" in payload["authority_boundary_confirmation"]

    for evidence_path, expected_value in case.get("expected_zero_paths", {}).items():
        assert payload["numeric_zero_evidence_summary"][evidence_path] == expected_value

    for stage_name, provenance_key in case.get("expected_provenance", {}).items():
        assert provenance_key in payload["provenance_summary"][stage_name]


def test_me_run08_matrix_fixture_artifact_writing_remains_explicit_only(
    tmp_path: Path,
    capsys,
) -> None:
    case = next(case for case in MATRIX_CASES if case["case_id"] == "numeric-zero")
    fixture_path = _write_fixture_case(tmp_path, case)

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            "local_snapshot_fixture",
            "--stage-payloads-json",
            str(fixture_path),
            "--dry-run-id",
            "me-run08-no-artifact",
            "--generated-at",
            "2026-06-17T15:00:00Z",
            "--artifact-output-root",
            str(tmp_path / "artifacts"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert not (tmp_path / "artifacts" / "me-run08-no-artifact").exists()

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            "local_snapshot_fixture",
            "--stage-payloads-json",
            str(fixture_path),
            "--dry-run-id",
            "me-run08-artifact",
            "--generated-at",
            "2026-06-17T15:00:00Z",
            "--write-local-artifact",
            "--artifact-output-root",
            str(tmp_path / "artifacts"),
            "--artifact-created-at",
            "2026-06-17T15:30:00Z",
        ]
    )

    captured = capsys.readouterr()
    emitted_payload = json.loads(captured.out)
    run_directory = tmp_path / "artifacts" / "me-run08-artifact"
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    artifact = json.loads(
        (
            run_directory
            / "artifacts"
            / "market_engine_dry_run_me-run08-artifact_2026-06-17.json"
        ).read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert captured.err == ""
    assert emitted_payload["dry_run_id"] == "me-run08-artifact"
    assert manifest["manifest_format_version"] == (
        MARKET_ENGINE_LOCAL_DRY_RUN_MANIFEST_FORMAT_VERSION
    )
    assert manifest["artifact_count"] == 1
    assert manifest["non_production_artifact"] is True
    assert artifact["artifact_format_version"] == (
        MARKET_ENGINE_LOCAL_DRY_RUN_ARTIFACT_FORMAT_VERSION
    )
    assert artifact["non_production_artifact"] is True
    assert artifact["source_input_mode"] == "local_snapshot_fixture"
    assert artifact["source_run_state"] == "dry_run_completed"
    assert artifact["payload"]["numeric_zero_evidence_summary"][
        "portfolio_review.portfolio_context_reference.current_market_value"
    ] == 0.0


def _write_fixture_case(tmp_path: Path, case: dict[str, Any]) -> Path:
    fixture_path = tmp_path / f"me_run08_{case['case_id']}_fixture.json"
    fixture_path.write_text(
        json.dumps(_fixture_wrapper(case), sort_keys=True),
        encoding="utf-8",
    )
    return fixture_path


def _fixture_wrapper(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "dry_run_input_fixture_format_version": (
            MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION
        ),
        "fixture_generated_at": "2026-06-17T15:00:00Z",
        "fixture_id": f"me-run08-{case['case_id']}-fixture",
        "fixture_scope": "ME-RUN08 local non-production fixture matrix",
        "input_mode": "local_snapshot_fixture",
        "non_production_fixture": True,
        "stage_payloads": _stage_payloads_for_case(case),
    }


def _stage_payloads_for_case(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stage_payloads = deepcopy(build_synthetic_dry_run_stage_payloads())
    ticker = case["ticker"]
    cik = _cik_for_ticker(ticker)

    for stage_payload in stage_payloads.values():
        stage_payload["ticker"] = ticker
        stage_payload["cik"] = cik
        stage_payload["fixture_backed"] = True

    case_id = case["case_id"]
    if case_id == "completed-with-limitations":
        stage_payloads["fundamental_observations"]["missing_data_markers"] = [
            "fundamental_observations.revenue_segment_breakdown",
        ]
        stage_payloads["source_context"]["stale_data_markers"] = [
            "source_context.sec_companyfacts.snapshot_review_required",
        ]
    elif case_id == "blocked":
        stage_payloads["setup_detection"]["setup_detection_state"] = "blocked_for_review"
        stage_payloads["setup_detection"]["blocked_reasons"] = [
            "ME-RUN08 blocked fixture stops before review layers.",
        ]
    elif case_id == "stale-data":
        stage_payloads["analysis_review"]["stale_data_markers"] = [
            "analysis_review.local_price_snapshot_absent",
        ]
        stage_payloads["delivery_reporting"]["stale_data_markers"] = [
            "delivery_reporting.local_review_timestamp_stale",
        ]
    elif case_id == "missing-data":
        stage_payloads["source_context"]["missing_data_markers"] = [
            "source_context.companyfacts.capital_expenditures_unavailable",
        ]
        stage_payloads["setup_detection"]["missing_data_markers"] = [
            "setup_detection.forward_guidance_not_in_fixture",
        ]
    elif case_id == "numeric-zero":
        stage_payloads["fundamental_observations"]["net_income"] = 0
        stage_payloads["derived_observations"]["free_cash_flow"] = 0
        stage_payloads["portfolio_review"]["portfolio_context_reference"][
            "cash_available_for_review"
        ] = 0
    elif case_id == "unsupported-input":
        stage_payloads["analysis_review"]["analysis_review_format_version"] = (
            "sec-companyfacts-analysis-review-v999"
        )
    elif case_id == "provenance-heavy":
        stage_payloads["source_context"]["source_provenance_reference"] = {
            "fixture_id": "me-run08-provenance-heavy-fixture",
            "source_refresh_snapshot_id": "me-run08-source-snapshot-msft-001",
            "snapshot_source": "non_production_local_fixture",
        }
        stage_payloads["fundamental_observations"]["source_context_reference"] = {
            "source_refresh_snapshot_id": "me-run08-source-snapshot-msft-001",
            "source_context_trace_id": "me-run08-source-context-trace-msft-001",
        }
        stage_payloads["recommendation_review"]["input_provenance"] = {
            "analysis_review_run_id": "analysis-review-run-001",
            "setup_detection_run_id": "setup-run-001",
            "fixture_matrix_case_id": "provenance-heavy",
        }
        stage_payloads["delivery_reporting"]["upstream_provenance_summary"] = {
            "decision_engine_handoff": {"handoff_run_id": "handoff-run-001"},
            "fixture_matrix_case_id": "provenance-heavy",
            "local_fixture_id": "me-run08-provenance-heavy-fixture",
        }

    return stage_payloads


def _cik_for_ticker(ticker: str) -> str:
    return {
        "NVDA": "0001045810",
        "AMD": "0000002488",
        "META": "0001326801",
        "COST": "0000909832",
        "ASML": "0000937966",
        "TSLA": "0001318605",
        "AVGO": "0001730168",
        "MSFT": "0000789019",
    }[ticker]
