from __future__ import annotations

import io
import json
from copy import deepcopy
from pathlib import Path

import pytest

from market_engine.advisory.advisory_artifact import (
    ChatGPTReadyAdvisoryArtifactError,
    assemble_chatgpt_ready_advisory_artifact,
    persist_chatgpt_ready_advisory_artifact,
)
from market_engine.advisory.advisory_artifact_validation import (
    ADVISORY_ARTIFACT_VALIDATOR_VERSION,
    validate_chatgpt_ready_advisory_artifact,
)
from market_engine.advisory.daily_artifact import (
    run_chatgpt_ready_advisory_artifact_command,
)


GENERATED_AT = "2026-07-08T08:00:00Z"


def test_minimal_valid_ci05_advisory_artifact_validates() -> None:
    result = validate_chatgpt_ready_advisory_artifact(_minimal_artifact())

    assert result.valid is True
    assert result.status == "valid"
    assert result.issues == ()


def test_full_valid_artifact_with_supported_context_families_validates() -> None:
    result = validate_chatgpt_ready_advisory_artifact(_full_artifact())

    assert result.valid is True
    assert result.to_payload()["validator_version"] == ADVISORY_ARTIFACT_VALIDATOR_VERSION


def test_allowed_optional_context_absence_is_valid_and_not_empty_portfolio() -> None:
    artifact = _minimal_artifact()

    result = validate_chatgpt_ready_advisory_artifact(artifact)

    assert result.valid is True
    assert artifact["portfolio_intelligence_context"]["payload"] is None
    assert "holdings" not in artifact["portfolio_intelligence_context"]


def test_referenced_context_shape_is_valid_without_claiming_embedded_validation() -> None:
    artifact = _minimal_artifact()
    artifact["portfolio_intelligence_context"] = {
        "include_mode": "referenced_context",
        "availability_state": "available",
        "semantic_override_allowed": False,
        "payload": None,
        "reference": {
            "artifact_ref": "artifact:portfolio:NVDA:run-001",
            "schema_version": "chatgpt-portfolio-intelligence-context-v1",
            "artifact_type": "market-engine-chatgpt-portfolio-intelligence-context",
            "run_id": "portfolio-run-001",
            "ticker": "NVDA",
        },
    }

    result = validate_chatgpt_ready_advisory_artifact(artifact)

    assert result.valid is True


def test_validation_results_are_deterministic_for_identical_input() -> None:
    artifact = _full_artifact()

    assert validate_chatgpt_ready_advisory_artifact(artifact).to_payload() == (
        validate_chatgpt_ready_advisory_artifact(deepcopy(artifact)).to_payload()
    )


def test_missing_required_top_level_field_fails() -> None:
    artifact = _minimal_artifact()
    artifact.pop("contract_identity")

    assert _codes(artifact) == {"invalid_field_type", "missing_required_field"}


def test_wrong_primitive_type_fails() -> None:
    artifact = _minimal_artifact()
    artifact["blockers"] = "none"

    assert "invalid_field_type" in _codes(artifact)


def test_unknown_enum_value_fails() -> None:
    artifact = _minimal_artifact()
    artifact["composition_status"]["state"] = "maybe_ready"

    assert "invalid_enum_value" in _codes(artifact)


def test_unsupported_schema_version_fails() -> None:
    artifact = _minimal_artifact()
    artifact["contract_identity"]["schema_version"] = "market-engine-chatgpt-ready-advisory-artifact-v0"

    assert "unsupported_schema_version" in _codes(artifact)


def test_artifact_type_mismatch_fails() -> None:
    artifact = _minimal_artifact()
    artifact["artifact_identity"]["artifact_type"] = "not-the-advisory-artifact"

    assert "artifact_type_mismatch" in _codes(artifact)


def test_malformed_nested_context_fails() -> None:
    artifact = _minimal_artifact()
    artifact["portfolio_intelligence_context"] = {
        "include_mode": "embedded_preserved_context",
        "availability_state": "available",
        "semantic_override_allowed": False,
        "payload": "not-an-object",
    }

    assert "invalid_field_type" in _codes(artifact)


def test_ticker_mismatch_fails() -> None:
    artifact = _minimal_artifact()
    artifact["structured_decision_context"]["payload"]["ticker"] = "AMD"

    assert "ticker_identity_mismatch" in _codes(artifact)


def test_run_identity_mismatch_fails() -> None:
    artifact = _minimal_artifact()
    artifact["structured_decision_context"]["payload"]["run_id"] = "other-run"

    assert "run_identity_mismatch" in _codes(artifact)


def test_structured_decision_source_identity_conflict_fails() -> None:
    artifact = _minimal_artifact()
    artifact["run_identity"]["source_structured_decision_run_id"] = "other-run"

    assert "run_identity_mismatch" in _codes(artifact)


def test_governor_context_mismatch_fails() -> None:
    artifact = _full_artifact()
    artifact["governor_context"]["payload"]["ticker"] = "AMD"

    assert "context_identity_mismatch" in _codes(artifact)


def test_dispatch_context_mismatch_fails() -> None:
    artifact = _full_artifact()
    artifact["dispatch_context"]["payload"]["subject"]["ticker"] = "AMD"

    assert "context_identity_mismatch" in _codes(artifact)


def test_portfolio_context_identity_conflict_fails() -> None:
    artifact = _full_artifact()
    artifact["portfolio_intelligence_context"]["payload"]["holdings"][0]["ticker"] = "AMD"

    assert "context_identity_mismatch" in _codes(artifact)


def test_explainability_current_run_inconsistency_fails() -> None:
    artifact = _full_artifact()
    artifact["explainability_change_rationale_context"]["payload"][
        "current_run_identity"
    ]["run_id"] = "other-run"

    assert "context_identity_mismatch" in _codes(artifact)


def test_missing_explainability_context_does_not_imply_unchanged() -> None:
    artifact = _minimal_artifact()

    result = validate_chatgpt_ready_advisory_artifact(artifact)

    assert result.valid is True
    assert "unchanged" not in json.dumps(artifact["explainability_change_rationale_context"])


def test_unknown_freshness_is_valid_but_not_converted_to_fresh() -> None:
    artifact = _minimal_artifact()
    artifact["freshness_context"]["global_freshness_status"] = "unknown"
    artifact["freshness_context"]["family_freshness"][0]["status"] = "unknown"

    result = validate_chatgpt_ready_advisory_artifact(artifact)

    assert result.valid is True
    assert artifact["freshness_context"]["global_freshness_status"] == "unknown"


def test_invalid_referenced_context_shape_fails() -> None:
    artifact = _minimal_artifact()
    artifact["portfolio_intelligence_context"] = {
        "include_mode": "referenced_context",
        "availability_state": "available",
        "semantic_override_allowed": False,
        "payload": None,
        "reference": {"artifact_ref": "artifact:portfolio:NVDA:run-001"},
    }

    assert "invalid_field_type" in _codes(artifact)


def test_embedded_reference_conflict_fails() -> None:
    artifact = _minimal_artifact()
    artifact["portfolio_intelligence_context"] = {
        "include_mode": "referenced_context",
        "availability_state": "available",
        "semantic_override_allowed": False,
        "payload": _portfolio_context(),
        "reference": {
            "artifact_ref": "artifact:portfolio:NVDA:run-001",
            "schema_version": "chatgpt-portfolio-intelligence-context-v1",
            "artifact_type": "market-engine-chatgpt-portfolio-intelligence-context",
        },
    }

    assert "embedded_reference_conflict" in _codes(artifact)


def test_unauthorized_semantic_override_fails() -> None:
    artifact = _minimal_artifact()
    artifact["composition_status"]["semantic_override_performed"] = True

    assert "forbidden_field_present" in _codes(artifact)


def test_forbidden_authority_field_in_forbidden_location_fails() -> None:
    artifact = _minimal_artifact()
    artifact["advisory_eligibility"]["order_quantity"] = 12

    assert "forbidden_field_present" in _codes(artifact)


def test_invalid_artifact_is_not_persisted_as_valid(tmp_path: Path) -> None:
    artifact = _minimal_artifact()
    artifact["composition_status"]["state"] = "maybe_ready"

    with pytest.raises(ChatGPTReadyAdvisoryArtifactError, match="contract validation"):
        persist_chatgpt_ready_advisory_artifact(artifact, output_root=tmp_path)

    assert not (tmp_path / "run-001" / "NVDA" / "chatgpt_ready_advisory.json").exists()


def test_valid_artifact_persists_with_validation_evidence(tmp_path: Path) -> None:
    result = persist_chatgpt_ready_advisory_artifact(
        _minimal_artifact(),
        output_root=tmp_path,
        validated_at=GENERATED_AT,
    )

    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert artifact["validation_summary"]["contract_validation"]["validation_status"] == "valid"
    assert manifest["validation_status"] == "valid"
    assert manifest["validator_version"] == ADVISORY_ARTIFACT_VALIDATOR_VERSION
    assert manifest["validation_issue_count"] == 0


def test_overwrite_protection_remains_active_after_validation(tmp_path: Path) -> None:
    persist_chatgpt_ready_advisory_artifact(_minimal_artifact(), output_root=tmp_path)

    with pytest.raises(ChatGPTReadyAdvisoryArtifactError, match="already exists"):
        persist_chatgpt_ready_advisory_artifact(_minimal_artifact(), output_root=tmp_path)


def test_cli_returns_nonzero_and_writes_no_valid_artifact_on_validation_failure(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    sdo = _sdo()
    advisory = _advisory_context(run_id="other-run")
    _write_json(input_dir / "structured_decision_output.json", sdo)
    _write_json(input_dir / "chatgpt_advisory_context.json", advisory)
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_chatgpt_ready_advisory_artifact_command(
        [
            "--input-artifact-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            GENERATED_AT,
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 2
    assert "contract validation" in stderr.getvalue()
    assert not (output_dir / "run-001" / "NVDA" / "chatgpt_ready_advisory.json").exists()


def test_cli_is_deterministic_for_valid_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_a = tmp_path / "output-a"
    output_b = tmp_path / "output-b"
    input_dir.mkdir()
    _write_json(input_dir / "structured_decision_output.json", _sdo())

    for output_dir in (output_a, output_b):
        assert (
            run_chatgpt_ready_advisory_artifact_command(
                [
                    "--input-artifact-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--generated-at",
                    GENERATED_AT,
                    "--emit-json",
                ],
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
            == 0
        )

    artifact_a = json.loads(
        (output_a / "run-001" / "NVDA" / "chatgpt_ready_advisory.json").read_text(
            encoding="utf-8"
        )
    )
    artifact_b = json.loads(
        (output_b / "run-001" / "NVDA" / "chatgpt_ready_advisory.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact_a == artifact_b


def _codes(artifact: dict[str, object]) -> set[str]:
    return {
        issue.code
        for issue in validate_chatgpt_ready_advisory_artifact(artifact).issues
    }


def _minimal_artifact() -> dict[str, object]:
    return assemble_chatgpt_ready_advisory_artifact(
        structured_decision_output=_sdo(),
        generated_at=GENERATED_AT,
        chatgpt_advisory_context=None,
        portfolio_intelligence_context=None,
        explainability_change_rationale_context=None,
        governor_context=None,
        dispatch_context=None,
    )


def _full_artifact() -> dict[str, object]:
    return assemble_chatgpt_ready_advisory_artifact(
        structured_decision_output=_sdo(),
        generated_at=GENERATED_AT,
        chatgpt_advisory_context=_advisory_context(),
        portfolio_intelligence_context=_portfolio_context(),
        explainability_change_rationale_context=_explainability_context(),
        governor_context=_governor_context(),
        dispatch_context=_dispatch_context(),
    )


def _sdo() -> dict[str, object]:
    return {
        "schema_version": "structured-decision-output-v1",
        "artifact_type": "market-engine-structured-decision-output",
        "generated_at": "2026-07-08T07:00:00Z",
        "run_id": "run-001",
        "ticker": "NVDA",
        "instrument": {
            "ticker": "NVDA",
            "name": "NVIDIA Corporation",
            "asset_type": "equity",
            "exchange": "NASDAQ",
            "currency": "USD",
        },
        "data_coverage": {
            "coverage_status": "ready",
            "coverage_score": 90,
            "freshness_status": "fresh",
            "missing_families": [],
            "stale_families": [],
            "blocked_reason": None,
        },
        "decision": {
            "action": "add_candidate",
            "action_strength": "medium",
            "time_horizon": "swing",
            "is_actionable": True,
            "actionability_blockers": [],
            "review_required": True,
        },
        "scores": {
            "confidence": {
                "value": 82,
                "scale": "0_100",
                "status": "available",
                "reason_codes": [],
            }
        },
        "portfolio_context": {
            "position_status": "unknown",
            "current_weight": None,
            "target_weight": None,
            "max_weight": None,
            "exposure_flags": [],
            "concentration_risk": "unknown",
            "cash_dependency": "not_available",
            "position_sizing_available": False,
        },
        "risk": {},
        "levels": {},
        "thesis": {},
        "evidence": {"artifact_refs": ["artifact:sdo:NVDA:run-001"]},
        "explainability": {
            "primary_reason_codes": [],
            "blocking_reasons": [],
            "human_summary_allowed": True,
        },
        "consumer_guidance": {},
        "validation": {
            "contract_status": "valid",
            "required_fields_present": True,
            "semantic_warnings": [],
            "fail_closed_reason": None,
        },
    }


def _advisory_context(run_id: str = "run-001") -> dict[str, object]:
    return {
        "schema_version": "chatgpt-advisory-context-v1",
        "artifact_type": "market-engine-chatgpt-advisory-context",
        "generated_at": "2026-07-08T07:10:00Z",
        "run_id": run_id,
        "ticker": "NVDA",
        "instrument": {"ticker": "NVDA", "asset_type": "equity"},
        "source_artifact_refs": ["artifact:sdo:NVDA:run-001"],
        "advisory_eligibility": {
            "state": "eligible",
            "reason_codes": ["structured_decision_output_valid"],
            "allowed_scope": ["explain_decision_state"],
            "required_disclosures": ["human_review_required"],
            "blocking_reasons": [],
        },
        "freshness_context": {
            "global_freshness_status": "fresh",
            "family_freshness": [],
            "stale_markers": [],
            "stale_reasons": [],
            "unknown_freshness": [],
        },
        "uncertainty_context": {
            "confidence": 82,
            "uncertainty_level": "medium",
            "missing_evidence": [],
            "limitations": ["human_review_required"],
        },
    }


def _portfolio_context() -> dict[str, object]:
    return {
        "schema_version": "chatgpt-portfolio-intelligence-context-v1",
        "artifact_type": "market-engine-chatgpt-portfolio-intelligence-context",
        "generated_at": "2026-07-08T07:15:00Z",
        "run_id": "portfolio-run-001",
        "portfolio_identity": {"portfolio_id": "synthetic"},
        "portfolio_snapshot_identity": {"snapshot_id": "snapshot-001"},
        "source_artifact_refs": ["artifact:portfolio:run-001"],
        "availability": {"state": "available", "reason_codes": []},
        "holdings": [{"ticker": "NVDA", "position_state": "held"}],
        "cash_context": {
            "state": "not_provided",
            "amount": None,
            "currency": None,
            "deployable_cash_state": "unknown",
        },
        "recommendation_to_position_relationship": {"ticker": "NVDA"},
        "freshness": {"portfolio_review_freshness": "fresh"},
    }


def _explainability_context() -> dict[str, object]:
    return {
        "schema_version": "chatgpt-explainability-change-rationale-context-v1",
        "artifact_type": "market-engine-chatgpt-explainability-change-rationale-context",
        "generated_at": "2026-07-08T07:20:00Z",
        "run_id": "explainability-run-001",
        "instrument": {"ticker": "NVDA", "asset_type": "equity"},
        "current_run_identity": {"run_id": "run-001"},
        "reference_run_identity": None,
        "comparison_window": {"mode": "current_state_only"},
        "source_artifact_refs": ["artifact:explainability:run-001"],
        "availability": {"state": "available", "reason_codes": []},
        "current_state_rationale": {"current_state": "eligible"},
        "validation": {"contract_valid": True, "blocked_reasons": []},
    }


def _governor_context() -> dict[str, object]:
    return {
        "schema_version": "market-engine-governor-context-v1",
        "artifact_type": "market-engine-governor-context",
        "run_id": "run-001",
        "ticker": "NVDA",
        "state": "evaluation_completed_non_actionable",
        "blockers": [],
        "freshness_context": {"global_freshness_status": "fresh"},
    }


def _dispatch_context() -> dict[str, object]:
    return {
        "report_contract_version": "market-engine-dispatch-station-governor-report-v1",
        "subject": {"ticker": "NVDA"},
        "sections": [],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
