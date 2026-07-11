from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from market_engine.advisory.grounded_advisory_output import (
    ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION,
    ModelInvocationResult,
    OpenAIResponsesInvoker,
    generate_grounded_advisory_output,
)
from market_engine.advisory import grounded_advisory_runtime as runtime
from market_engine.advisory.grounded_advisory_runtime import (
    MAX_OUTPUT_TOKENS,
    _openai_request_payload,
)


def test_grounded_advisory_output_uses_ci09_and_writes_artifacts(tmp_path: Path) -> None:
    result = _generate(tmp_path, "happy", _valid_model_response())
    structured = _read(result.structured_output_path)
    report = result.report_path.read_text(encoding="utf-8")

    assert structured["advisory_status"] == "grounded_advisory_generated"
    assert structured["validation_result"]["status"] == "valid"
    assert structured["validation_result"]["grounding_status"] == (
        "grounded_with_mandatory_caveats"
    )
    assert structured["validation_result"]["validator"] == (
        "ME-CI09.validate_advisory_response_grounding"
    )
    assert structured["source_readiness"]["actionability_allowed"] is False
    assert "Grounded Advisory Report - NVDA" in report
    assert "CI09 grounding validation" in report
    assert "No immediate action" in report


def test_model_input_contains_explicit_evidence_catalog(tmp_path: Path) -> None:
    source_path = _write_source(tmp_path)
    invoker = _RecordingInvoker(_valid_model_response())
    generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "outputs",
        run_id="ci11-grounding",
        generated_at="2026-07-09T12:00:00Z",
        invoker=invoker,
    )

    refs = invoker.request["grounding_handoff_contract"]["allowed_evidence_references"]
    ref_ids = {ref["evidence_ref_id"] for ref in refs}
    assert {
        "readiness:analysis_context",
        "blocked:stage",
        "recommendation:review",
    }.issubset(ref_ids)
    assert all(ref["source_path"].startswith("$.payload") for ref in refs)
    assert all("evidence_catalog[" in ref["projection_path"] for ref in refs)
    assert invoker.request["grounding_handoff_contract"]["validator"] == (
        "ME-CI09.validate_advisory_response_grounding"
    )


def test_blocked_readiness_rejects_advisory_interpretation_status(tmp_path: Path) -> None:
    response = _valid_model_response()
    response["advisory_status"] = "grounded_interpretation"
    result = _generate(tmp_path, "actionability", response)
    structured = _read(result.structured_output_path)

    assert structured["advisory_status"] == "blocked_validation_failed"
    assert "actionability_ceiling_exceeded" in _codes(structured)


def test_unknown_evidence_reference_fails_closed(tmp_path: Path) -> None:
    response = _valid_model_response()
    response["evidence_references"][0]["evidence_ref_id"] = "invented:evidence"
    result = _generate(tmp_path, "unknown-ref", response)
    structured = _read(result.structured_output_path)

    assert structured["validation_result"]["status"] == "invalid"
    assert "unknown_evidence_reference" in _codes(structured)


def test_material_claim_requires_evidence_reference(tmp_path: Path) -> None:
    response = _valid_model_response()
    response["evidence_references"] = [
        ref
        for ref in response["evidence_references"]
        if ref["claim_id"] != "claim-assessment"
    ]
    result = _generate(tmp_path, "claim-ref", response)
    structured = _read(result.structured_output_path)

    assert "material_claim_without_evidence_reference" in _codes(structured)


def test_wrong_field_type_fails_strict_parser(tmp_path: Path) -> None:
    response = _valid_model_response()
    response["limitations"] = "not-a-list"
    result = _generate(tmp_path, "types", response)
    structured = _read(result.structured_output_path)

    assert structured["advisory_status"] == "blocked_model_output_invalid"
    assert structured["parser_result"]["parser_state"] == "schema_invalid"


def test_stale_and_missing_data_are_preserved(tmp_path: Path) -> None:
    source = _source_payload()
    source["payload"]["analysis_context_readiness"]["context_stale"] = True
    source["payload"]["missing_data_summary"].append("source_timestamp")
    response = _valid_model_response()
    response["required_disclosures"].append("staleness_disclosure")
    result = _generate(tmp_path, "stale", response, source=source)
    structured = _read(result.structured_output_path)
    report = result.report_path.read_text(encoding="utf-8")

    assert "context_stale" in structured["source_readiness"]["stale_data"]
    assert "source_timestamp" in structured["source_readiness"]["missing_data"]
    assert "source_timestamp" in report


def test_malformed_model_output_fails_closed(tmp_path: Path) -> None:
    source_path = _write_source(tmp_path)
    result = generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "outputs",
        run_id="ci11-malformed",
        generated_at="2026-07-09T12:00:00Z",
        invoker=_RawTextInvoker("{not-json"),
    )
    structured = _read(result.structured_output_path)

    assert structured["advisory_status"] == "blocked_model_output_invalid"
    assert structured["parser_result"]["parser_state"] == "malformed_json"
    assert structured["validation_result"]["status"] == "invalid"


def test_traceability_preserves_raw_provider_metadata(tmp_path: Path) -> None:
    result = _generate(tmp_path, "trace", _valid_model_response())
    structured = _read(result.structured_output_path)
    raw = _read(result.raw_response_path)

    assert structured["source_artifact"]["run_id"] == "run-nvda"
    assert structured["source_artifact"]["sha256"]
    assert structured["invocation_boundary"]["provider_name"] == "fake-provider"
    assert raw["delivery_eligible"] is False
    assert raw["raw_provider_response"]["id"] == "fake-request"
    assert raw["raw_provider_response_hash"]
    assert structured["provenance_trace"]["invocation_request_hash"]
    assert structured["provenance_trace"]["idempotency_key"]


def test_idempotency_changes_when_model_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = _write_source(tmp_path)
    monkeypatch.setenv("MARKET_ENGINE_ADVISORY_MODEL", "model-a")
    invoker_a = _RecordingInvoker(_valid_model_response())
    generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "a",
        run_id="ci11-model-a",
        generated_at="2026-07-09T12:00:00Z",
        invoker=invoker_a,
    )

    monkeypatch.setenv("MARKET_ENGINE_ADVISORY_MODEL", "model-b")
    invoker_b = _RecordingInvoker(_valid_model_response())
    generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "b",
        run_id="ci11-model-b",
        generated_at="2026-07-09T12:00:00Z",
        invoker=invoker_b,
    )

    assert invoker_a.request["idempotency_policy"]["idempotency_key"] != (
        invoker_b.request["idempotency_policy"]["idempotency_key"]
    )


def test_openai_payload_enforces_schema_and_output_budget(tmp_path: Path) -> None:
    source_path = _write_source(tmp_path)
    invoker = _RecordingInvoker(_valid_model_response())
    generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / "outputs",
        run_id="ci11-provider-shape",
        generated_at="2026-07-09T12:00:00Z",
        invoker=invoker,
    )

    payload = _openai_request_payload(invoker.request, "test-model")
    assert payload["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["additionalProperties"] is False


def test_openai_invoker_captures_provider_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _captured_invocation_request(tmp_path, "provider-refusal")
    monkeypatch.setattr(
        runtime.urllib.request,
        "urlopen",
        lambda http_request, timeout: _FakeHttpResponse(
            {
                "id": "resp-refusal",
                "status": "completed",
                "output": [
                    {
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "Request refused by provider policy.",
                            }
                        ]
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        ),
    )

    result = OpenAIResponsesInvoker(
        api_key="test-key", model="test-model", base_url="https://example.invalid/v1"
    ).invoke(request)

    assert result.invocation_state == "provider_refusal"
    assert result.provider_name == "openai"
    assert result.model_name == "test-model"
    assert result.provider_request_id == "resp-refusal"
    assert result.error_message == "Request refused by provider policy."
    assert result.raw_output is None
    assert result.raw_provider_response["status"] == "completed"
    assert result.received_at


def test_openai_invoker_captures_truncated_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _captured_invocation_request(tmp_path, "provider-truncated")
    monkeypatch.setattr(
        runtime.urllib.request,
        "urlopen",
        lambda http_request, timeout: _FakeHttpResponse(
            {
                "id": "resp-truncated",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": "{}",
                "usage": {"input_tokens": 10, "output_tokens": MAX_OUTPUT_TOKENS},
            }
        ),
    )

    result = OpenAIResponsesInvoker(
        api_key="test-key", model="test-model", base_url="https://example.invalid/v1"
    ).invoke(request)

    assert result.invocation_state == "truncated_response"
    assert result.provider_request_id == "resp-truncated"
    assert result.finish_reason == "max_output_tokens"
    assert result.error_message == "Provider response incomplete: max_output_tokens"
    assert result.raw_output == "{}"
    assert result.raw_provider_response["status"] == "incomplete"
    assert result.received_at


def test_openai_invoker_captures_provider_network_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _captured_invocation_request(tmp_path, "provider-network")

    def raise_network_error(http_request, timeout):
        raise urllib.error.URLError("network unavailable")

    monkeypatch.setattr(runtime.urllib.request, "urlopen", raise_network_error)

    result = OpenAIResponsesInvoker(
        api_key="test-key", model="test-model", base_url="https://example.invalid/v1"
    ).invoke(request)

    assert result.invocation_state == "provider_failure"
    assert result.provider_name == "openai"
    assert result.model_name == "test-model"
    assert "network unavailable" in str(result.error_message)
    assert result.raw_output is None
    assert result.raw_provider_response is None
    assert result.received_at is None


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
            raw_provider_response={"id": "fake-request", "status": "completed"},
            received_at="2026-07-09T12:00:01Z",
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


class _FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _valid_model_response() -> dict[str, object]:
    return {
        "schema_version": ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION,
        "advisory_status": "descriptive_only",
        "executive_conclusion": (
            "No immediate action is supported. The artifact remains partial and blocked."
        ),
        "claims": [
            {
                "role": "assessment",
                "claim_id": "claim-assessment",
                "claim_type": "supported_interpretation",
                "text": "The source remains partial and subject to an upstream blocked state.",
            },
            {
                "role": "supporting",
                "claim_id": "claim-review",
                "claim_type": "evidence_summary",
                "text": "Recommendation Review requires human review.",
            },
            {
                "role": "blocker",
                "claim_id": "claim-blocker-1",
                "claim_type": "missingness_statement",
                "text": "Stage preserves an upstream blocked state.",
            },
            {
                "role": "blocker",
                "claim_id": "claim-blocker-2",
                "claim_type": "missingness_statement",
                "text": "missing_setup_or_price_context",
            },
        ],
        "evidence_references": [
            {
                "claim_id": "claim-assessment",
                "evidence_ref_id": "readiness:analysis_context",
                "support_type": "interpreted",
            },
            {
                "claim_id": "claim-review",
                "evidence_ref_id": "recommendation:review",
                "support_type": "summarized",
            },
        ],
        "limitations": [
            "Portfolio review is blocked.",
            "Setup or price context is missing.",
        ],
        "practical_interpretation": (
            "Treat the stock as watch-only until missing setup and portfolio context are resolved."
        ),
        "confidence_and_evidence_quality": (
            "Evidence quality is partial and the source readiness ceiling remains restrictive."
        ),
        "required_disclosures": [
            "based_only_on_supplied_market_engine_artifact",
            "not_broker_or_order_instruction",
            "blocked_or_partial_states_remain_non_actionable",
            "descriptive_only_disclosure",
        ],
    }


def _generate(
    tmp_path: Path,
    suffix: str,
    response: dict[str, object],
    *,
    source: dict[str, object] | None = None,
):
    source_path = _write_source(tmp_path, source=source)
    return generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / f"outputs-{suffix}",
        run_id=f"ci11-{suffix}",
        generated_at="2026-07-09T12:00:00Z",
        invoker=_FakeInvoker(response),
    )


def _captured_invocation_request(tmp_path: Path, suffix: str):
    source_path = _write_source(tmp_path)
    invoker = _RecordingInvoker(_valid_model_response())
    generate_grounded_advisory_output(
        source_artifact_path=source_path,
        output_root=tmp_path / f"outputs-{suffix}",
        run_id=f"ci11-{suffix}",
        generated_at="2026-07-09T12:00:00Z",
        invoker=invoker,
    )
    return invoker.request


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


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _codes(structured: dict[str, object]) -> set[str]:
    return {
        issue["code"] for issue in structured["validation_result"]["issues"]
    }
