from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_engine.advisory.grounded_advisory_output import (
    ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION,
    ModelInvocationResult,
    generate_grounded_advisory_output,
)


def test_grounded_advisory_output_happy_path_writes_artifacts(tmp_path: Path) -> None:
    source_path = _write_source(tmp_path)
    result = generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "outputs",
        run_id="ci11-test-happy",
        generated_at="2026-07-09T12:00:00Z",
        invoker=_FakeInvoker(_valid_model_response()),
    )

    structured = json.loads(result.structured_output_path.read_text(encoding="utf-8"))
    report = result.report_path.read_text(encoding="utf-8")

    assert structured["advisory_status"] == "grounded_advisory_generated"
    assert structured["validation_result"]["status"] == "valid"
    assert structured["source_readiness"]["actionability_allowed"] is False
    assert result.invocation_request_path.exists()
    assert result.raw_response_path.exists()
    assert result.parser_result_path.exists()
    assert result.validation_result_path.exists()
    assert result.manifest_path.exists()
    assert "Grounded Advisory Report - NVDA" in report
    assert "based only on the referenced Market Engine artifact" in report
    assert "No immediate action" in report


def test_model_input_contains_only_allowed_evidence(tmp_path: Path) -> None:
    source_path = _write_source(tmp_path)
    invoker = _RecordingInvoker(_valid_model_response())

    generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "outputs",
        run_id="ci11-test-grounding",
        generated_at="2026-07-09T12:00:00Z",
        invoker=invoker,
    )

    allowed_refs = invoker.request["grounding_handoff_contract"][
        "allowed_evidence_references"
    ]
    ref_ids = {ref["evidence_ref_id"] for ref in allowed_refs}
    assert {"readiness:analysis_context", "blocked:stage", "recommendation:review"}.issubset(
        ref_ids
    )
    assert invoker.request["input_context"]["forbidden_inferences"]
    assert invoker.request["model_capability_profile"]["external_browsing_allowed"] is False


def test_blocked_readiness_cannot_accept_actionable_model_language(tmp_path: Path) -> None:
    source_path = _write_source(tmp_path)
    response = _valid_model_response()
    response["executive_conclusion"] = "Buy now because the setup is ready."

    result = generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "outputs",
        run_id="ci11-test-blocked-action",
        generated_at="2026-07-09T12:00:00Z",
        invoker=_FakeInvoker(response),
    )
    structured = json.loads(result.structured_output_path.read_text(encoding="utf-8"))

    assert structured["advisory_status"] == "blocked_validation_failed"
    assert "actionable_language_not_allowed" in _codes(structured)


def test_stale_and_missing_data_are_preserved(tmp_path: Path) -> None:
    source = _source_payload()
    source["payload"]["analysis_context_readiness"]["context_stale"] = True
    source["payload"]["missing_data_summary"].append("source_timestamp")
    source_path = _write_source(tmp_path, source=source)

    result = generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "outputs",
        run_id="ci11-test-stale-missing",
        generated_at="2026-07-09T12:00:00Z",
        invoker=_FakeInvoker(_valid_model_response()),
    )
    structured = json.loads(result.structured_output_path.read_text(encoding="utf-8"))
    report = result.report_path.read_text(encoding="utf-8")

    assert "context_stale" in structured["source_readiness"]["stale_data"]
    assert "source_timestamp" in structured["source_readiness"]["missing_data"]
    assert "source_timestamp" in report


def test_malformed_model_output_fails_closed(tmp_path: Path) -> None:
    source_path = _write_source(tmp_path)

    result = generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "outputs",
        run_id="ci11-test-malformed",
        generated_at="2026-07-09T12:00:00Z",
        invoker=_RawTextInvoker("{not-json"),
    )
    structured = json.loads(result.structured_output_path.read_text(encoding="utf-8"))

    assert structured["advisory_status"] == "blocked_model_output_invalid"
    assert structured["parser_result"]["parser_state"] == "malformed_json"
    assert structured["validation_result"]["status"] == "invalid"


def test_unknown_artifact_format_fails_closed(tmp_path: Path) -> None:
    source = _source_payload()
    source["artifact_format_version"] = "unknown"
    source_path = _write_source(tmp_path, source=source)

    result = generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "outputs",
        run_id="ci11-test-unsupported",
        generated_at="2026-07-09T12:00:00Z",
        invoker=_FakeInvoker(_valid_model_response()),
    )
    structured = json.loads(result.structured_output_path.read_text(encoding="utf-8"))

    assert structured["advisory_status"] == "blocked_source_not_supported"
    assert "unsupported_artifact_format" in {
        issue["code"] for issue in structured["validation_result"]["issues"]
    }


def test_traceability_preserves_source_and_invocation_metadata(tmp_path: Path) -> None:
    source_path = _write_source(tmp_path)
    result = generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "outputs",
        run_id="ci11-test-trace",
        generated_at="2026-07-09T12:00:00Z",
        invoker=_FakeInvoker(_valid_model_response()),
    )
    structured = json.loads(result.structured_output_path.read_text(encoding="utf-8"))
    raw = json.loads(result.raw_response_path.read_text(encoding="utf-8"))

    assert structured["source_artifact"]["path"] == source_path.as_posix()
    assert structured["source_artifact"]["run_id"] == "run-nvda"
    assert structured["invocation_boundary"]["provider_name"] == "fake-provider"
    assert raw["delivery_eligible"] is False
    assert structured["provenance_trace"]["invocation_request_hash"]
    assert structured["provenance_trace"]["raw_output_hash"]


class _FakeInvoker:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response

    def invoke(self, request):
        return ModelInvocationResult(
            invocation_state="response_received",
            provider_name="fake-provider",
            model_name="fake-model",
            raw_output=json.dumps(self.response),
            provider_request_id="fake-request",
            finish_reason="stop",
            usage_metadata={"input_tokens": 10, "output_tokens": 10},
        )


class _RecordingInvoker(_FakeInvoker):
    def invoke(self, request):
        self.request = request
        return super().invoke(request)


class _RawTextInvoker:
    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text

    def invoke(self, request):
        return ModelInvocationResult(
            invocation_state="response_received",
            provider_name="fake-provider",
            model_name="fake-model",
            raw_output=self.raw_text,
        )


def _valid_model_response() -> dict[str, object]:
    return {
        "schema_version": ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION,
        "advisory_status": "non_actionable_grounded_interpretation",
        "executive_conclusion": (
            "No immediate action is supported. The artifact shows partial analysis with fundamental evidence, but portfolio review remains blocked."
        ),
        "supporting_evidence_references": [
            "readiness:analysis_context",
            "recommendation:review",
        ],
        "risk_evidence_references": [
            "blocked:stage",
            "blocked:reason:0",
            "missing:data:0",
        ],
        "limitations": [
            "Portfolio review is blocked.",
            "Setup or price context is missing.",
        ],
        "practical_interpretation": (
            "Treat the stock as watch-only until missing setup and portfolio context are resolved."
        ),
        "confidence_and_evidence_quality": (
            "Evidence quality is partial: SEC fundamental provenance exists, but actionability is not allowed."
        ),
    }


def _write_source(tmp_path: Path, *, source: dict[str, object] | None = None) -> Path:
    path = tmp_path / "dry_run.json"
    path.write_text(json.dumps(source or _source_payload()), encoding="utf-8")
    return path


def _source_payload() -> dict[str, object]:
    return {
        "artifact_format_version": "market-engine-local-dry-run-artifact-v1",
        "artifact_type": "market_engine_end_to_end_dry_run",
        "artifact_created_at": "2026-07-02T11:56:52Z",
        "payload": {
            "dry_run_id": "run-nvda",
            "ticker": "NVDA",
            "generated_at": "2026-07-02T11:56:52Z",
            "input_mode": "cached_source_snapshot",
            "blocked_stage": "portfolio_review",
            "blocked_reasons": ["Stage preserves an upstream blocked state."],
            "missing_data_summary": ["portfolio_context"],
            "analysis_context_readiness": {
                "readiness_level": "partial_analysis",
                "actionable_review_allowed": False,
                "decision_engine_ready": False,
                "context_stale": False,
                "blocked_reasons": ["missing_setup_or_price_context"],
                "evidence_families_missing": ["setup_price_market"],
            },
            "provenance_summary": {
                "fundamental_observations": {
                    "source_refresh_snapshot_id": "NVDA_companyfacts"
                },
                "portfolio_review": {
                    "recommendation_review_provenance": {
                        "review_state": "human_review_required",
                        "review_category": "analysis_mixed_or_conflicted",
                        "input_provenance": {"ticker": "NVDA"},
                    }
                },
            },
        },
    }


def _codes(structured: dict[str, object]) -> set[str]:
    return {issue["code"] for issue in structured["validation_result"]["issues"]}
