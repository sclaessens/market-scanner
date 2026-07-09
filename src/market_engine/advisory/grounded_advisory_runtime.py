from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TextIO

from market_engine.advisory.advisory_response_grounding import (
    RESPONSE_SCHEMA_VERSION as CI09_RESPONSE_SCHEMA_VERSION,
    validate_advisory_response_grounding,
)


CI10_CONTRACT_NAME = "controlled_model_invocation_boundary"
CI10_CONTRACT_VERSION = "v1"
CI10_SCHEMA_VERSION = "market-engine-controlled-model-invocation-boundary-v1"
ADVISORY_OUTPUT_SCHEMA_VERSION = "market-engine-grounded-advisory-output-v1"
ADVISORY_OUTPUT_ARTIFACT_TYPE = "market-engine-grounded-advisory-output"
ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION = (
    "market-engine-grounded-advisory-model-response-v2"
)
DEFAULT_OUTPUT_ROOT = "artifacts/market_engine/grounded_advisory_outputs"
SUPPORTED_SOURCE_ARTIFACT_VERSION = "market-engine-local-dry-run-artifact-v1"
SUPPORTED_SOURCE_ARTIFACT_TYPE = "market_engine_end_to_end_dry_run"
CI11_GROUNDING_PROJECTION_SCHEMA_VERSION = "market-engine-ci11-grounding-projection-v1"
CI11_GROUNDING_PROJECTION_ARTIFACT_TYPE = "market-engine-ci11-grounding-projection"
MAX_OUTPUT_TOKENS = 1200
MAX_INPUT_CHARACTERS = 120_000
ALLOWED_ADVISORY_STATUSES = {
    "grounded_interpretation",
    "descriptive_only",
    "partial_answer",
    "unable_to_determine",
}
CLAIM_ROLES = {
    "assessment",
    "supporting",
    "opposing",
    "blocker",
    "uncertainty",
    "freshness",
    "unable_to_determine",
}
CLAIM_TYPES = {
    "direct_artifact_fact",
    "upstream_state_description",
    "evidence_summary",
    "explicit_upstream_reason",
    "supported_interpretation",
    "conditional_interpretation",
    "associated_change",
    "uncertainty_statement",
    "missingness_statement",
    "authority_boundary_statement",
}
SUPPORT_TYPES = {"direct", "summarized", "interpreted", "conditional", "associated_only"}
NON_MATERIAL_CLAIM_TYPES = {
    "uncertainty_statement",
    "missingness_statement",
    "authority_boundary_statement",
}


class GroundedAdvisoryOutputError(ValueError):
    """Raised when the grounded advisory output flow cannot run safely."""


@dataclass(frozen=True)
class ModelInvocationResult:
    invocation_state: str
    provider_name: str
    model_name: str
    raw_output: str | None
    provider_request_id: str | None = None
    finish_reason: str | None = None
    usage_metadata: Mapping[str, Any] | None = None
    error_message: str | None = None
    raw_provider_response: Mapping[str, Any] | None = None
    received_at: str | None = None


class ModelInvoker(Protocol):
    def invoke(self, request: Mapping[str, Any]) -> ModelInvocationResult:
        ...


@dataclass(frozen=True)
class GroundedAdvisoryGenerationResult:
    output_directory: Path
    structured_output_path: Path
    report_path: Path
    invocation_request_path: Path
    raw_response_path: Path
    parser_result_path: Path
    validation_result_path: Path
    manifest_path: Path
    summary: dict[str, Any]


def generate_grounded_advisory_output(
    *,
    source_artifact_path: Path | str,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    run_id: str,
    generated_at: str,
    invoker: ModelInvoker | None = None,
    allow_overwrite: bool = False,
) -> GroundedAdvisoryGenerationResult:
    source_path = Path(source_artifact_path)
    source_artifact = _read_json_object(source_path)
    source_validation = _validate_source_artifact(source_artifact)
    source_summary = _source_summary(source_artifact, source_path)
    ticker = _safe_path_segment(source_summary["ticker"], "ticker")
    safe_run_id = _safe_path_segment(run_id, "run_id")
    output_root_path = _validated_output_root(output_root)
    output_directory = _resolved_child(
        _resolved_child(output_root_path.resolve(), safe_run_id), ticker
    )
    if output_directory.exists() and not allow_overwrite:
        raise GroundedAdvisoryOutputError(
            f"Grounded advisory output directory already exists: {output_directory}"
        )
    if output_directory.exists() and allow_overwrite:
        _remove_tree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)

    invocation_request = _build_invocation_request(
        source_artifact=source_artifact,
        source_path=source_path,
        source_validation=source_validation,
        source_summary=source_summary,
        run_id=safe_run_id,
        generated_at=generated_at,
    )
    request_validation = _validate_invocation_request(invocation_request)
    if not source_validation["valid"]:
        invocation_result = _blocked_invocation_result(
            "Source artifact failed pre-invocation validation."
        )
    elif not request_validation["valid"]:
        invocation_result = _blocked_invocation_result(
            "Invocation request failed deterministic CI10 pre-invocation validation."
        )
    else:
        invocation_result = (invoker or OpenAIResponsesInvoker.from_environment()).invoke(
            invocation_request
        )

    parser_result = _parse_model_response(invocation_result.raw_output)
    validation_result = _validate_model_response(
        parsed_response=parser_result.get("parsed_response"),
        invocation_result=invocation_result,
        source_summary=source_summary,
        source_validation=source_validation,
        invocation_request=invocation_request,
        request_validation=request_validation,
        run_id=safe_run_id,
    )
    structured_output = _structured_output(
        source_summary=source_summary,
        invocation_request=invocation_request,
        invocation_result=invocation_result,
        parser_result=parser_result,
        validation_result=validation_result,
        run_id=safe_run_id,
        generated_at=generated_at,
    )
    report = _render_report(structured_output)
    manifest = _manifest(
        output_directory=output_directory,
        structured_output=structured_output,
        run_id=safe_run_id,
        ticker=ticker,
    )

    invocation_request_path = output_directory / "invocation_request.json"
    raw_response_path = output_directory / "raw_model_response.json"
    parser_result_path = output_directory / "parser_result.json"
    validation_result_path = output_directory / "validation_result.json"
    structured_output_path = output_directory / "grounded_advisory_output.json"
    report_path = output_directory / "advisory_report.md"
    manifest_path = output_directory / "manifest.json"
    _write_json(invocation_request_path, invocation_request)
    _write_json(raw_response_path, _raw_response_payload(invocation_result))
    _write_json(parser_result_path, parser_result)
    _write_json(validation_result_path, validation_result)
    _write_json(structured_output_path, structured_output)
    report_path.write_text(report, encoding="utf-8")
    _write_json(manifest_path, manifest)

    return GroundedAdvisoryGenerationResult(
        output_directory=output_directory,
        structured_output_path=structured_output_path,
        report_path=report_path,
        invocation_request_path=invocation_request_path,
        raw_response_path=raw_response_path,
        parser_result_path=parser_result_path,
        validation_result_path=validation_result_path,
        manifest_path=manifest_path,
        summary={
            "run_id": safe_run_id,
            "ticker": ticker,
            "advisory_status": structured_output["advisory_status"],
            "validation_status": validation_result["status"],
            "grounding_status": validation_result.get("grounding_status"),
            "invocation_state": invocation_result.invocation_state,
            "structured_output_path": structured_output_path.as_posix(),
            "report_path": report_path.as_posix(),
        },
    )


class OpenAIResponsesInvoker:
    """Single-provider non-streaming CI10 boundary with structured output."""

    def __init__(self, *, api_key: str, model: str, base_url: str) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

    @classmethod
    def from_environment(cls) -> "OpenAIResponsesInvoker | MissingConfigInvoker":
        api_key = os.environ.get("OPENAI_API_KEY")
        model = os.environ.get("MARKET_ENGINE_ADVISORY_MODEL") or os.environ.get(
            "OPENAI_MODEL"
        )
        if not api_key or not model:
            return MissingConfigInvoker(
                "OPENAI_API_KEY and MARKET_ENGINE_ADVISORY_MODEL or OPENAI_MODEL are required for real invocation."
            )
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(api_key=api_key, model=model, base_url=base_url)

    def invoke(self, request: Mapping[str, Any]) -> ModelInvocationResult:
        payload = _openai_request_payload(request, self._model)
        data = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self._base_url}/responses",
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(http_request, timeout=60) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return ModelInvocationResult(
                invocation_state="provider_failure",
                provider_name="openai",
                model_name=self._model,
                raw_output=None,
                error_message=str(exc),
            )

        received_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        refusal = _extract_openai_refusal(response_payload)
        incomplete_reason = _extract_incomplete_reason(response_payload)
        output_text = _extract_openai_output_text(response_payload)
        if refusal:
            state = "provider_refusal"
            error_message = refusal
        elif incomplete_reason:
            state = "truncated_response" if incomplete_reason == "max_output_tokens" else "provider_failure"
            error_message = f"Provider response incomplete: {incomplete_reason}"
        else:
            state = "response_received"
            error_message = None

        return ModelInvocationResult(
            invocation_state=state,
            provider_name="openai",
            model_name=self._model,
            raw_output=output_text,
            provider_request_id=(
                str(response_payload.get("id")) if response_payload.get("id") else None
            ),
            finish_reason=_find_finish_reason(response_payload) or incomplete_reason,
            usage_metadata={
                "provider_usage": response_payload.get("usage"),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
            error_message=error_message,
            raw_provider_response=dict(response_payload),
            received_at=received_at,
        )


class MissingConfigInvoker:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    def invoke(self, request: Mapping[str, Any]) -> ModelInvocationResult:
        return _blocked_invocation_result(self._reason)


def run_grounded_advisory_output_command(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    args = _argument_parser().parse_args(argv)
    try:
        result = generate_grounded_advisory_output(
            source_artifact_path=args.artifact,
            output_root=args.output_root,
            run_id=args.run_id,
            generated_at=args.generated_at,
            allow_overwrite=args.allow_overwrite,
        )
    except GroundedAdvisoryOutputError as exc:
        print(str(exc), file=error_stream)
        return 2
    json.dump(result.summary, output_stream, indent=2, sort_keys=True)
    output_stream.write("\n")
    return 0 if result.summary["validation_status"] == "valid" else 2


def main() -> None:
    raise SystemExit(run_grounded_advisory_output_command())


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-engine-grounded-advisory-output",
        description=(
            "Generate a local grounded advisory output from one existing Market Engine artifact. "
            "The command preserves the CI10 invocation boundary and executes CI09 grounding validation."
        ),
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def _validate_source_artifact(source_artifact: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if source_artifact.get("artifact_format_version") != SUPPORTED_SOURCE_ARTIFACT_VERSION:
        issues.append(
            _issue(
                "unsupported_artifact_format",
                "$.artifact_format_version",
                SUPPORTED_SOURCE_ARTIFACT_VERSION,
                source_artifact.get("artifact_format_version"),
            )
        )
    if source_artifact.get("artifact_type") != SUPPORTED_SOURCE_ARTIFACT_TYPE:
        issues.append(
            _issue(
                "unsupported_artifact_type",
                "$.artifact_type",
                SUPPORTED_SOURCE_ARTIFACT_TYPE,
                source_artifact.get("artifact_type"),
            )
        )
    payload = _object(source_artifact.get("payload"))
    if not payload:
        issues.append(_issue("missing_payload", "$.payload", "object", None))
    readiness = _object(payload.get("analysis_context_readiness"))
    if not readiness:
        issues.append(
            _issue(
                "missing_analysis_context_readiness",
                "$.payload.analysis_context_readiness",
                "object",
                None,
            )
        )
    if not _nested_text(
        payload,
        ("provenance_summary", "fundamental_observations", "source_refresh_snapshot_id"),
    ):
        issues.append(
            _issue(
                "missing_fundamental_provenance",
                "$.payload.provenance_summary.fundamental_observations",
                "source_refresh_snapshot_id",
                None,
            )
        )
    return {
        "status": "valid" if not issues else "invalid",
        "valid": not issues,
        "issues": issues,
    }


def _source_summary(
    source_artifact: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    payload = _object(source_artifact.get("payload"))
    readiness = _object(payload.get("analysis_context_readiness"))
    provenance = _object(payload.get("provenance_summary"))
    recommendation = _object(
        _nested(provenance, ("portfolio_review", "recommendation_review_provenance"))
    )
    ticker = str(
        payload.get("ticker")
        or _nested_text(recommendation, ("input_provenance", "ticker"))
        or "UNKNOWN"
    )
    evidence_refs = _evidence_refs(payload)
    source_hash = _sha256_json(source_artifact)
    return {
        "ticker": ticker,
        "instrument": ticker,
        "company_name": _nested_text(payload, ("company_identity", "company_name")),
        "source_artifact_path": source_path.as_posix(),
        "source_artifact_hash": source_hash,
        "source_artifact_ref": f"artifact:ci11:{source_hash}",
        "source_artifact_format": source_artifact.get("artifact_format_version"),
        "source_artifact_type": source_artifact.get("artifact_type"),
        "source_run_id": str(payload.get("dry_run_id") or "unknown-run"),
        "source_generated_at": str(
            payload.get("generated_at")
            or source_artifact.get("artifact_created_at")
            or "unknown"
        ),
        "input_mode": str(payload.get("input_mode") or "unknown"),
        "blocked_stage": payload.get("blocked_stage"),
        "blocked_reasons": _stable_unique(
            _list_of_text(payload.get("blocked_reasons"))
            + _list_of_text(readiness.get("blocked_reasons"))
        ),
        "missing_data": _stable_unique(
            _list_of_text(payload.get("missing_data_summary"))
            + _list_of_text(readiness.get("evidence_families_missing"))
        ),
        "stale_data": ["context_stale"] if readiness.get("context_stale") else [],
        "readiness_level": str(readiness.get("readiness_level") or "unknown"),
        "actionable_review_allowed": bool(readiness.get("actionable_review_allowed")),
        "decision_engine_ready": bool(readiness.get("decision_engine_ready")),
        "actionability_allowed": bool(readiness.get("actionable_review_allowed"))
        and bool(readiness.get("decision_engine_ready")),
        "recommendation_review_state": str(recommendation.get("review_state") or "unknown"),
        "recommendation_review_category": str(
            recommendation.get("review_category") or "unknown"
        ),
        "evidence_refs": evidence_refs,
    }


def _evidence_refs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    readiness = _object(payload.get("analysis_context_readiness"))
    raw: list[tuple[str, str, str, str]] = [
        (
            "readiness:analysis_context",
            "readiness",
            "$.payload.analysis_context_readiness",
            (
                f"Readiness is {readiness.get('readiness_level')} with "
                f"actionable_review_allowed={readiness.get('actionable_review_allowed')} "
                f"and decision_engine_ready={readiness.get('decision_engine_ready')}."
            ),
        )
    ]
    if payload.get("blocked_stage"):
        raw.append(
            (
                "blocked:stage",
                "blocked_state",
                "$.payload.blocked_stage",
                f"Dry-run is blocked at {payload.get('blocked_stage')}.",
            )
        )
    for index, reason in enumerate(_list_of_text(payload.get("blocked_reasons"))):
        raw.append(
            (
                f"blocked:payload:{index}",
                "blocked_state",
                f"$.payload.blocked_reasons[{index}]",
                reason,
            )
        )
    for index, reason in enumerate(_list_of_text(readiness.get("blocked_reasons"))):
        raw.append(
            (
                f"blocked:readiness:{index}",
                "blocked_state",
                f"$.payload.analysis_context_readiness.blocked_reasons[{index}]",
                reason,
            )
        )
    for index, item in enumerate(_list_of_text(payload.get("missing_data_summary"))):
        raw.append(
            (
                f"missing:summary:{index}",
                "missing_data",
                f"$.payload.missing_data_summary[{index}]",
                item,
            )
        )
    for index, item in enumerate(_list_of_text(readiness.get("evidence_families_missing"))):
        raw.append(
            (
                f"missing:readiness:{index}",
                "missing_data",
                f"$.payload.analysis_context_readiness.evidence_families_missing[{index}]",
                item,
            )
        )
    recommendation = _object(
        _nested(
            _object(payload.get("provenance_summary")),
            ("portfolio_review", "recommendation_review_provenance"),
        )
    )
    if recommendation:
        raw.append(
            (
                "recommendation:review",
                "recommendation_review",
                "$.payload.provenance_summary.portfolio_review.recommendation_review_provenance",
                (
                    f"Recommendation Review state is {recommendation.get('review_state')} "
                    f"and category is {recommendation.get('review_category')}."
                ),
            )
        )
    refs = []
    for index, (ref_id, family, source_path, summary) in enumerate(raw):
        refs.append(
            {
                "evidence_ref_id": ref_id,
                "evidence_family": family,
                "source_path": source_path,
                "projection_path": (
                    f"$.structured_decision_context.payload.evidence_catalog[{index}].summary"
                ),
                "summary": summary,
            }
        )
    return refs


def _build_invocation_request(
    *,
    source_artifact: Mapping[str, Any],
    source_path: Path,
    source_validation: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    run_id: str,
    generated_at: str,
) -> dict[str, Any]:
    model_name = (
        os.environ.get("MARKET_ENGINE_ADVISORY_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "not_configured"
    )
    prompt_package = {
        "schema_version": "market-engine-grounded-advisory-input-package-v2",
        "artifact_type": "market-engine-grounded-advisory-input-package",
        "source_artifact_identity": {
            "path": source_path.as_posix(),
            "sha256": source_summary["source_artifact_hash"],
            "format": source_summary["source_artifact_format"],
            "type": source_summary["source_artifact_type"],
            "run_id": source_summary["source_run_id"],
        },
        "instrument_identity": {
            "ticker": source_summary["ticker"],
            "instrument": source_summary["instrument"],
        },
        "question": (
            "What does the Market Engine artifact say about this stock, and what cautious conclusion is supported by the available evidence?"
        ),
        "question_classification": {
            "question_class": "current_state_explanation",
            "permitted_use_case": "bounded_current_state_interpretation",
        },
        "selected_context": {
            "readiness": {
                "readiness_level": source_summary["readiness_level"],
                "actionable_review_allowed": source_summary["actionable_review_allowed"],
                "decision_engine_ready": source_summary["decision_engine_ready"],
                "blocked_stage": source_summary["blocked_stage"],
                "blocked_reasons": source_summary["blocked_reasons"],
            },
            "recommendation_review": {
                "state": source_summary["recommendation_review_state"],
                "category": source_summary["recommendation_review_category"],
            },
            "missing_data": source_summary["missing_data"],
            "stale_data": source_summary["stale_data"],
        },
        "allowed_evidence_references": source_summary["evidence_refs"],
        "mandatory_disclosures": [
            "based_only_on_supplied_market_engine_artifact",
            "not_broker_or_order_instruction",
            "blocked_or_partial_states_remain_non_actionable",
        ],
        "forbidden_inferences": [
            "invented_current_price",
            "invented_target_price",
            "invented_entry_level",
            "invented_stop_loss",
            "invented_news",
            "stronger_than_upstream_recommendation",
            "blocked_artifact_as_actionable_advice",
        ],
        "required_response_schema": ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION,
    }
    prompt_hash = _sha256_json(prompt_package)
    idempotency_material = {
        "source_artifact_hash": source_summary["source_artifact_hash"],
        "prompt_package_hash": prompt_hash,
        "question_class": "current_state_explanation",
        "permitted_use_case": "bounded_current_state_interpretation",
        "provider": "openai",
        "model": model_name,
        "configuration_profile": "ci11-nonproduction-structured-v1",
        "response_schema": ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION,
        "budget_profile": {"max_output_tokens": MAX_OUTPUT_TOKENS},
    }
    return {
        "invocation_request_identity": {
            "schema_version": CI10_SCHEMA_VERSION,
            "artifact_type": "market-engine-controlled-model-invocation-request",
            "contract_name": CI10_CONTRACT_NAME,
            "contract_version": CI10_CONTRACT_VERSION,
            "request_id": run_id,
            "generated_at": generated_at,
        },
        "source_prompt_package_identity": {
            "schema_version": prompt_package["schema_version"],
            "artifact_type": prompt_package["artifact_type"],
            "sha256": prompt_hash,
        },
        "source_artifact_identity": prompt_package["source_artifact_identity"],
        "instrument_identity": prompt_package["instrument_identity"],
        "run_identity": {"run_id": source_summary["source_run_id"]},
        "invocation_identity": {"invocation_id": run_id},
        "attempt_identity": {
            "attempt_id": f"{run_id}-attempt-1",
            "attempt_sequence": 1,
        },
        "question_classification": prompt_package["question_classification"],
        "permitted_use_case": "bounded_current_state_interpretation",
        "provider_identity": {
            "provider_name": "openai",
            "provider_api_family": "responses",
        },
        "model_identity": {
            "model_name": model_name,
            "configuration_profile_id": "ci11-nonproduction-structured-v1",
        },
        "model_capability_profile": {
            "structured_json_output_required": True,
            "non_streaming_required": True,
            "tool_use_allowed": False,
            "external_browsing_allowed": False,
        },
        "instruction_contract": {
            "system_boundary": (
                "Use only supplied Market Engine context. Do not invent facts, prices, news, "
                "targets, sizing, allocation, or execution guidance. Every material claim must "
                "carry a claim_id and at least one evidence reference."
            ),
            "user_question_cannot_override_boundary": True,
        },
        "input_context": prompt_package,
        "response_schema_contract": {
            "schema_version": ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION,
            "artifact_type": "market-engine-grounded-advisory-model-response",
            "strict": True,
        },
        "budget_limits": {
            "max_input_characters": MAX_INPUT_CHARACTERS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        "timeout_policy": {"total_invocation_timeout_seconds": 60},
        "retry_policy": {"max_attempts": 1, "retry_enabled": False},
        "idempotency_policy": {
            "idempotency_key": _sha256_json(idempotency_material),
            "material": idempotency_material,
        },
        "data_handling_policy": {
            "persist_secrets": False,
            "external_browsing_allowed": False,
        },
        "grounding_handoff_contract": {
            "validator": "ME-CI09.validate_advisory_response_grounding",
            "allowed_evidence_references": source_summary["evidence_refs"],
            "fail_closed_on_invalid_reference": True,
        },
        "audit_context": {"source_validation": source_validation},
    }


def _validate_invocation_request(request: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    required_paths = {
        "question_class": _nested_text(
            request, ("question_classification", "question_class")
        ),
        "permitted_use_case": request.get("permitted_use_case"),
        "provider_name": _nested_text(request, ("provider_identity", "provider_name")),
        "model_name": _nested_text(request, ("model_identity", "model_name")),
        "response_schema": _nested_text(
            request, ("response_schema_contract", "schema_version")
        ),
        "idempotency_key": _nested_text(
            request, ("idempotency_policy", "idempotency_key")
        ),
    }
    for name, value in required_paths.items():
        if not isinstance(value, str) or not value:
            issues.append({"code": "missing_invocation_field", "path": name})
    if required_paths["question_class"] != "current_state_explanation":
        issues.append({"code": "unsupported_question_class", "path": "question_class"})
    if required_paths["permitted_use_case"] != "bounded_current_state_interpretation":
        issues.append({"code": "unsupported_permitted_use_case", "path": "permitted_use_case"})
    if required_paths["response_schema"] != ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION:
        issues.append({"code": "response_schema_mismatch", "path": "response_schema"})
    max_output = _nested(request, ("budget_limits", "max_output_tokens"))
    if not isinstance(max_output, int) or max_output <= 0:
        issues.append({"code": "invalid_output_budget", "path": "budget_limits.max_output_tokens"})
    input_size = len(json.dumps(request.get("input_context", {}), sort_keys=True))
    max_input = _nested(request, ("budget_limits", "max_input_characters"))
    if not isinstance(max_input, int) or input_size > max_input:
        issues.append({"code": "input_budget_exceeded", "path": "budget_limits.max_input_characters"})
    capability = _object(request.get("model_capability_profile"))
    if capability.get("tool_use_allowed") is not False:
        issues.append({"code": "tool_use_not_disabled", "path": "model_capability_profile.tool_use_allowed"})
    if capability.get("external_browsing_allowed") is not False:
        issues.append({"code": "browsing_not_disabled", "path": "model_capability_profile.external_browsing_allowed"})
    return {"status": "valid" if not issues else "invalid", "valid": not issues, "issues": issues}


def _prompt_text(request: Mapping[str, Any]) -> str:
    return (
        "Use only the supplied request JSON. Every material claim must have a claim_id and at "
        "least one evidence reference to an allowed evidence_ref_id. Preserve blockers, missing "
        "data, uncertainty, freshness and the source readiness ceiling.\n\n"
        + json.dumps(request, indent=2, sort_keys=True)
    )


def _openai_request_payload(request: Mapping[str, Any], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": request["instruction_contract"]["system_boundary"],
            },
            {"role": "user", "content": _prompt_text(request)},
        ],
        "max_output_tokens": request["budget_limits"]["max_output_tokens"],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "market_engine_grounded_advisory",
                "strict": True,
                "schema": _model_response_json_schema(),
            }
        },
    }


def _model_response_json_schema() -> dict[str, Any]:
    claim_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "role": {"type": "string", "enum": sorted(CLAIM_ROLES)},
            "claim_id": {"type": "string"},
            "claim_type": {"type": "string", "enum": sorted(CLAIM_TYPES)},
            "text": {"type": "string"},
        },
        "required": ["role", "claim_id", "claim_type", "text"],
    }
    reference_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claim_id": {"type": "string"},
            "evidence_ref_id": {"type": "string"},
            "support_type": {"type": "string", "enum": sorted(SUPPORT_TYPES)},
        },
        "required": ["claim_id", "evidence_ref_id", "support_type"],
    }
    properties = {
        "schema_version": {"type": "string", "enum": [ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION]},
        "advisory_status": {"type": "string", "enum": sorted(ALLOWED_ADVISORY_STATUSES)},
        "executive_conclusion": {"type": "string"},
        "claims": {"type": "array", "items": claim_schema},
        "evidence_references": {"type": "array", "items": reference_schema},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "practical_interpretation": {"type": "string"},
        "confidence_and_evidence_quality": {"type": "string"},
        "required_disclosures": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _parse_model_response(raw_output: str | None) -> dict[str, Any]:
    if not raw_output:
        return {
            "status": "invalid",
            "parser_state": "empty_response",
            "parsed_response": None,
            "issues": [{"code": "empty_response", "path": "$"}],
        }
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid",
            "parser_state": "malformed_json",
            "parsed_response": None,
            "issues": [{"code": "malformed_json", "path": "$", "message": str(exc)}],
        }
    issues = _response_schema_issues(parsed)
    return {
        "status": "valid" if not issues else "invalid",
        "parser_state": "parsed" if not issues else "schema_invalid",
        "parsed_response": parsed if isinstance(parsed, dict) else None,
        "issues": issues,
    }


def _response_schema_issues(parsed: Any) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        return [{"code": "response_not_object", "path": "$"}]
    required = {
        "schema_version",
        "advisory_status",
        "executive_conclusion",
        "claims",
        "evidence_references",
        "limitations",
        "practical_interpretation",
        "confidence_and_evidence_quality",
        "required_disclosures",
    }
    issues: list[dict[str, Any]] = []
    for field in sorted(required - set(parsed)):
        issues.append({"code": "missing_required_field", "path": f"$.{field}"})
    for field in sorted(set(parsed) - required):
        issues.append({"code": "unknown_field", "path": f"$.{field}"})
    if parsed.get("schema_version") != ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION:
        issues.append({"code": "schema_version_mismatch", "path": "$.schema_version"})
    if parsed.get("advisory_status") not in ALLOWED_ADVISORY_STATUSES:
        issues.append({"code": "invalid_advisory_status", "path": "$.advisory_status"})
    for field in (
        "executive_conclusion",
        "practical_interpretation",
        "confidence_and_evidence_quality",
    ):
        if not isinstance(parsed.get(field), str) or not parsed.get(field, "").strip():
            issues.append({"code": "invalid_string_field", "path": f"$.{field}"})
    for field in ("claims", "evidence_references", "limitations", "required_disclosures"):
        if not isinstance(parsed.get(field), list):
            issues.append({"code": "invalid_list_field", "path": f"$.{field}"})
    if not isinstance(parsed.get("claims"), list) or not isinstance(parsed.get("evidence_references"), list):
        return issues
    for index, claim in enumerate(parsed["claims"]):
        path = f"$.claims[{index}]"
        if not isinstance(claim, Mapping):
            issues.append({"code": "claim_not_object", "path": path})
            continue
        if set(claim) != {"role", "claim_id", "claim_type", "text"}:
            issues.append({"code": "invalid_claim_shape", "path": path})
        if claim.get("role") not in CLAIM_ROLES:
            issues.append({"code": "invalid_claim_role", "path": f"{path}.role"})
        if claim.get("claim_type") not in CLAIM_TYPES:
            issues.append({"code": "invalid_claim_type", "path": f"{path}.claim_type"})
        if not isinstance(claim.get("claim_id"), str) or not claim.get("claim_id"):
            issues.append({"code": "invalid_claim_id", "path": f"{path}.claim_id"})
        if not isinstance(claim.get("text"), str) or not claim.get("text", "").strip():
            issues.append({"code": "invalid_claim_text", "path": f"{path}.text"})
    for index, ref in enumerate(parsed["evidence_references"]):
        path = f"$.evidence_references[{index}]"
        if not isinstance(ref, Mapping):
            issues.append({"code": "reference_not_object", "path": path})
            continue
        if set(ref) != {"claim_id", "evidence_ref_id", "support_type"}:
            issues.append({"code": "invalid_reference_shape", "path": path})
        if ref.get("support_type") not in SUPPORT_TYPES:
            issues.append({"code": "invalid_support_type", "path": f"{path}.support_type"})
    for field in ("limitations", "required_disclosures"):
        values = parsed.get(field)
        if isinstance(values, list) and any(not isinstance(item, str) for item in values):
            issues.append({"code": "invalid_string_list", "path": f"$.{field}"})
    return issues


def _validate_model_response(
    *,
    parsed_response: Any,
    invocation_result: ModelInvocationResult,
    source_summary: Mapping[str, Any],
    source_validation: Mapping[str, Any],
    invocation_request: Mapping[str, Any],
    request_validation: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not source_validation.get("valid"):
        issues.append({"code": "source_validation_failed", "path": "$.source"})
    if not request_validation.get("valid"):
        issues.append({"code": "invocation_request_invalid", "path": "$.invocation_request"})
        issues.extend(dict(issue) for issue in request_validation.get("issues", []))
    if invocation_result.invocation_state != "response_received":
        issues.append(
            {
                "code": "invocation_not_completed",
                "path": "$.invocation",
                "message": invocation_result.error_message,
            }
        )
    if not isinstance(parsed_response, Mapping):
        issues.append({"code": "missing_parsed_response", "path": "$.model_response"})
        return {"status": "invalid", "valid": False, "grounding_status": None, "issues": issues}

    allowed_by_id = {
        ref["evidence_ref_id"]: ref for ref in source_summary["evidence_refs"]
    }
    claim_ids = {
        claim.get("claim_id")
        for claim in parsed_response.get("claims", [])
        if isinstance(claim, Mapping)
    }
    for index, ref in enumerate(parsed_response.get("evidence_references", [])):
        if ref.get("claim_id") not in claim_ids:
            issues.append(
                {
                    "code": "unknown_claim_reference",
                    "path": f"$.evidence_references[{index}].claim_id",
                }
            )
        if ref.get("evidence_ref_id") not in allowed_by_id:
            issues.append(
                {
                    "code": "unknown_evidence_reference",
                    "path": f"$.evidence_references[{index}].evidence_ref_id",
                    "actual": ref.get("evidence_ref_id"),
                }
            )
    referenced_claim_ids = {
        ref.get("claim_id")
        for ref in parsed_response.get("evidence_references", [])
        if isinstance(ref, Mapping)
    }
    for index, claim in enumerate(parsed_response.get("claims", [])):
        if (
            claim.get("claim_type") not in NON_MATERIAL_CLAIM_TYPES
            and claim.get("claim_id") not in referenced_claim_ids
        ):
            issues.append(
                {
                    "code": "material_claim_without_evidence_reference",
                    "path": f"$.claims[{index}]",
                }
            )
    mandatory = set(invocation_request["input_context"]["mandatory_disclosures"])
    present = set(parsed_response.get("required_disclosures", []))
    for disclosure in sorted(mandatory - present):
        issues.append(
            {
                "code": "mandatory_disclosure_missing",
                "path": "$.required_disclosures",
                "expected": disclosure,
            }
        )
    if (
        not source_summary.get("actionability_allowed")
        and parsed_response.get("advisory_status") == "grounded_interpretation"
    ):
        issues.append(
            {
                "code": "actionability_ceiling_exceeded",
                "path": "$.advisory_status",
            }
        )
    if issues:
        return {"status": "invalid", "valid": False, "grounding_status": None, "issues": issues}

    ci09_source = _ci09_source_projection(source_summary)
    ci09_prompt = _ci09_prompt_projection(source_summary, invocation_request)
    ci09_response = _ci09_response_projection(
        parsed_response=parsed_response,
        source_summary=source_summary,
        ci09_prompt=ci09_prompt,
        run_id=run_id,
    )
    grounding = validate_advisory_response_grounding(
        source_artifact=ci09_source,
        prompt_package=ci09_prompt,
        response=ci09_response,
    )
    grounding_issues = [issue.to_payload() for issue in grounding.issues]
    if not grounding.valid:
        issues.append(
            {
                "code": "ci09_grounding_validation_failed",
                "path": "$.grounding",
                "grounding_status": grounding.status,
            }
        )
        issues.extend(grounding_issues)
    return {
        "status": "valid" if not issues else "invalid",
        "valid": not issues,
        "grounding_status": grounding.status,
        "validator": "ME-CI09.validate_advisory_response_grounding",
        "issues": issues,
        "grounding_result": grounding.to_payload(),
    }


def _ci09_source_projection(source_summary: Mapping[str, Any]) -> dict[str, Any]:
    evidence_catalog = [
        {
            "evidence_ref_id": ref["evidence_ref_id"],
            "evidence_family": ref["evidence_family"],
            "source_path": ref["source_path"],
            "summary": ref["summary"],
        }
        for ref in source_summary["evidence_refs"]
    ]
    return {
        "run_identity": {"run_id": source_summary["source_run_id"]},
        "instrument_identity": {"ticker": source_summary["ticker"]},
        "artifact_identity": {
            "artifact_type": CI11_GROUNDING_PROJECTION_ARTIFACT_TYPE,
            "schema_version": CI11_GROUNDING_PROJECTION_SCHEMA_VERSION,
        },
        "source_artifact_references": [source_summary["source_artifact_ref"]],
        "structured_decision_context": {
            "include_mode": "embedded_canonical_context",
            "payload": {"evidence_catalog": evidence_catalog},
        },
        "portfolio_intelligence_context": {
            "include_mode": "absent",
            "availability_state": "unavailable",
            "payload": None,
        },
        "explainability_change_rationale_context": {
            "include_mode": "absent",
            "availability_state": "unavailable",
            "payload": None,
        },
        "governor_context": {"include_mode": "absent", "payload": None},
        "dispatch_context": {"include_mode": "absent", "payload": None},
        "freshness_context": {
            "global_freshness_status": (
                "stale" if source_summary["stale_data"] else "current"
            ),
            "family_freshness": [],
        },
        "blockers": list(source_summary["blocked_reasons"]),
    }


def _ci09_prompt_projection(
    source_summary: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "prompt_package_identity": {
            "artifact_type": "market-engine-grounded-advisory-input-package",
        },
        "source_artifact_identity": {
            "schema_version": CI11_GROUNDING_PROJECTION_SCHEMA_VERSION,
            "artifact_type": CI11_GROUNDING_PROJECTION_ARTIFACT_TYPE,
            "run_id": source_summary["source_run_id"],
        },
        "instrument_identity": {"ticker": source_summary["ticker"]},
        "question_classification": {
            "question_class": "current_state_explanation",
            "required_context_families": ["structured_decision_output"],
            "missing_required_context_families": [],
        },
        "selected_context": {
            "structured_decision_output": _ci09_source_projection(source_summary)[
                "structured_decision_context"
            ],
            "freshness_context": _ci09_source_projection(source_summary)[
                "freshness_context"
            ],
            "blockers": list(source_summary["blocked_reasons"]),
        },
        "mandatory_disclosures": list(
            request["input_context"]["mandatory_disclosures"]
        ),
        "authority_boundary": {
            "allocation_authority": False,
            "position_sizing_authority": False,
            "execution_authority": False,
            "broker_authority": False,
            "portfolio_write_authority": False,
            "watchlist_write_authority": False,
            "question_class_requires_refusal": False,
        },
    }


def _ci09_response_projection(
    *,
    parsed_response: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    ci09_prompt: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    role_fields = {
        "assessment": "assessment",
        "supporting": "evidence_supporting",
        "opposing": "evidence_opposing",
        "blocker": "blockers",
        "uncertainty": "uncertainty",
        "freshness": "freshness_caveats",
        "unable_to_determine": "unable_to_determine",
    }
    sections: dict[str, list[dict[str, Any]]] = {
        field: [] for field in role_fields.values()
    }
    for claim in parsed_response["claims"]:
        sections[role_fields[claim["role"]]].append(
            {
                "claim_id": claim["claim_id"],
                "claim_type": claim["claim_type"],
                "text": claim["text"],
            }
        )
    allowed_by_id = {
        ref["evidence_ref_id"]: ref for ref in source_summary["evidence_refs"]
    }
    claim_types = {
        claim["claim_id"]: claim["claim_type"] for claim in parsed_response["claims"]
    }
    evidence_references = []
    for index, ref in enumerate(parsed_response["evidence_references"]):
        allowed = allowed_by_id[ref["evidence_ref_id"]]
        evidence_references.append(
            {
                "ref_id": f"ci11-ref-{index}",
                "claim_id": ref["claim_id"],
                "claim_type": claim_types[ref["claim_id"]],
                "source_context_family": "structured_decision_output",
                "artifact_ref": source_summary["source_artifact_ref"],
                "run_id": source_summary["source_run_id"],
                "path": allowed["projection_path"],
                "support_type": ref["support_type"],
            }
        )
    status_to_mode = {
        "grounded_interpretation": "advisory_interpretation",
        "descriptive_only": "descriptive_only",
        "partial_answer": "partial_answer",
        "unable_to_determine": "unable_to_determine",
    }
    response_mode = status_to_mode[parsed_response["advisory_status"]]
    disclosures = list(parsed_response["required_disclosures"])
    if response_mode == "partial_answer":
        declared_grounding = "partially_grounded"
    elif disclosures or response_mode == "unable_to_determine":
        declared_grounding = "grounded_with_mandatory_caveats"
    else:
        declared_grounding = "grounded"
    return {
        "schema_version": CI09_RESPONSE_SCHEMA_VERSION,
        "artifact_type": "market-engine-chatgpt-advisory-response-grounding-example",
        "response_identity": {
            "response_id": run_id,
            "response_mode": response_mode,
            "non_production_example": True,
        },
        "source_artifact_identity": dict(ci09_prompt["source_artifact_identity"]),
        "instrument_identity": {"ticker": source_summary["ticker"], "asset_type": "equity"},
        "question_classification": {
            "question_class": "current_state_explanation",
            "requested_scope": "ci11_real_invocation",
            "required_context_families": ["structured_decision_output"],
            "unavailable_context_families": [],
        },
        "response_mode": response_mode,
        "summary": parsed_response["executive_conclusion"],
        "assessment": sections["assessment"],
        "evidence_supporting": sections["evidence_supporting"],
        "evidence_opposing": sections["evidence_opposing"],
        "blockers": sections["blockers"],
        "uncertainty": sections["uncertainty"],
        "freshness_caveats": sections["freshness_caveats"],
        "portfolio_context": {
            "availability": "not_requested",
            "disclosure_required": False,
            "claims": [],
        },
        "change_rationale": {
            "availability": "not_requested",
            "attribution_level": "not_applicable",
            "claims": [],
        },
        "required_disclosures": disclosures,
        "unable_to_determine": sections["unable_to_determine"],
        "evidence_references": evidence_references,
        "grounding_summary": {
            "status": declared_grounding,
            "issue_count": 0,
            "issues": [],
        },
        "authority_boundary": {
            "allocation_authority": False,
            "position_sizing_authority": False,
            "execution_authority": False,
            "broker_authority": False,
            "portfolio_write_authority": False,
            "watchlist_write_authority": False,
        },
    }


def _structured_output(
    *,
    source_summary: Mapping[str, Any],
    invocation_request: Mapping[str, Any],
    invocation_result: ModelInvocationResult,
    parser_result: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    run_id: str,
    generated_at: str,
) -> dict[str, Any]:
    parsed = parser_result.get("parsed_response") if validation_result.get("valid") else None
    if parsed:
        advisory_status = "grounded_advisory_generated"
        executive = parsed["executive_conclusion"]
        claim_role_by_id = {claim["claim_id"]: claim["role"] for claim in parsed["claims"]}
        supporting_refs = sorted(
            {
                ref["evidence_ref_id"]
                for ref in parsed["evidence_references"]
                if claim_role_by_id.get(ref["claim_id"]) in {"assessment", "supporting"}
            }
        )
        risk_refs = sorted(
            {
                ref["evidence_ref_id"]
                for ref in parsed["evidence_references"]
                if claim_role_by_id.get(ref["claim_id"])
                in {"opposing", "blocker", "uncertainty", "freshness", "unable_to_determine"}
            }
        )
        limitations = list(parsed["limitations"])
        practical = parsed["practical_interpretation"]
        confidence = parsed["confidence_and_evidence_quality"]
    else:
        advisory_status = _blocked_status(invocation_result, parser_result)
        executive = _blocked_executive_conclusion(source_summary, invocation_result)
        supporting_refs = []
        risk_refs = [
            ref["evidence_ref_id"]
            for ref in source_summary["evidence_refs"]
            if ref["evidence_family"] in {"blocked_state", "missing_data", "readiness"}
        ]
        limitations = list(source_summary["blocked_reasons"]) + list(source_summary["missing_data"])
        practical = (
            "Treat this as a blocked/non-actionable advisory generation result. "
            "Review the source artifact and invocation blocker before using any advisory output."
        )
        confidence = "No successful CI09-grounded model response was accepted."
    return {
        "schema_version": ADVISORY_OUTPUT_SCHEMA_VERSION,
        "artifact_type": ADVISORY_OUTPUT_ARTIFACT_TYPE,
        "generated_at": generated_at,
        "run_id": run_id,
        "ticker": source_summary["ticker"],
        "instrument_identity": {
            "ticker": source_summary["ticker"],
            "instrument": source_summary["instrument"],
            "company_name": source_summary.get("company_name"),
        },
        "source_artifact": {
            "path": source_summary["source_artifact_path"],
            "sha256": source_summary["source_artifact_hash"],
            "format": source_summary["source_artifact_format"],
            "type": source_summary["source_artifact_type"],
            "run_id": source_summary["source_run_id"],
            "generated_at": source_summary["source_generated_at"],
            "input_mode": source_summary["input_mode"],
        },
        "invocation_boundary": {
            "contract_name": CI10_CONTRACT_NAME,
            "contract_version": CI10_CONTRACT_VERSION,
            "schema_version": CI10_SCHEMA_VERSION,
            "invocation_state": invocation_result.invocation_state,
            "provider_name": invocation_result.provider_name,
            "model_name": invocation_result.model_name,
            "provider_request_id": invocation_result.provider_request_id,
            "finish_reason": invocation_result.finish_reason,
        },
        "source_readiness": {
            "readiness_level": source_summary["readiness_level"],
            "actionable_review_allowed": source_summary["actionable_review_allowed"],
            "decision_engine_ready": source_summary["decision_engine_ready"],
            "actionability_allowed": source_summary["actionability_allowed"],
            "blocked_stage": source_summary["blocked_stage"],
            "blocked_reasons": source_summary["blocked_reasons"],
            "missing_data": source_summary["missing_data"],
            "stale_data": source_summary["stale_data"],
        },
        "advisory_status": advisory_status,
        "executive_conclusion": executive,
        "supporting_evidence_references": supporting_refs,
        "risk_evidence_references": risk_refs,
        "limitations": limitations,
        "practical_interpretation": practical,
        "confidence_and_evidence_quality": confidence,
        "allowed_evidence_references": source_summary["evidence_refs"],
        "validation_result": validation_result,
        "parser_result": {
            "status": parser_result.get("status"),
            "parser_state": parser_result.get("parser_state"),
            "issues": parser_result.get("issues", []),
        },
        "provenance_trace": {
            "prompt_package_hash": invocation_request["source_prompt_package_identity"]["sha256"],
            "invocation_request_hash": _sha256_json(invocation_request),
            "raw_output_hash": _sha256_text(invocation_result.raw_output or ""),
            "idempotency_key": invocation_request["idempotency_policy"]["idempotency_key"],
            "grounding_validator": "ME-CI09.validate_advisory_response_grounding",
        },
    }


def _render_report(output: Mapping[str, Any]) -> str:
    readiness = _object(output.get("source_readiness"))
    return "\n".join(
        [
            f"# Grounded Advisory Report - {output['ticker']}",
            "",
            "This report is based only on the referenced Market Engine artifact. Successful advisory conclusions are accepted only after CI09 grounding validation.",
            "",
            "## Instrument and context",
            "",
            f"- Ticker: {output['ticker']}",
            f"- Company/instrument: {output['instrument_identity'].get('company_name') or output['instrument_identity'].get('instrument')}",
            f"- Source artifact: `{output['source_artifact']['path']}`",
            f"- Source run: `{output['source_artifact']['run_id']}`",
            f"- Source generated at: {output['source_artifact']['generated_at']}",
            f"- Readiness: {readiness.get('readiness_level')}",
            f"- Actionability allowed: {readiness.get('actionability_allowed')}",
            f"- Advisory status: {output['advisory_status']}",
            f"- Missing data: {', '.join(readiness.get('missing_data') or []) or 'None'}",
            f"- Stale data: {', '.join(readiness.get('stale_data') or []) or 'None'}",
            f"- Blocked stage: {readiness.get('blocked_stage') or 'None'}",
            "",
            "## Executive conclusion",
            "",
            str(output["executive_conclusion"]),
            "",
            "## What supports this conclusion?",
            "",
            _format_refs(output, "supporting_evidence_references"),
            "",
            "## Main risks and limitations",
            "",
            _format_refs(output, "risk_evidence_references"),
            "",
            _format_list("Limitations", output.get("limitations", [])),
            "",
            "## Practical interpretation",
            "",
            str(output["practical_interpretation"]),
            "",
            "## Confidence / evidence quality",
            "",
            str(output["confidence_and_evidence_quality"]),
            "",
            "## Grounding and validation",
            "",
            f"- Parser status: {output['parser_result']['status']} / {output['parser_result']['parser_state']}",
            f"- Validation status: {output['validation_result']['status']}",
            f"- CI09 grounding status: {output['validation_result'].get('grounding_status')}",
            f"- Invocation state: {output['invocation_boundary']['invocation_state']}",
            f"- Provider/model: {output['invocation_boundary']['provider_name']} / {output['invocation_boundary']['model_name']}",
            "",
        ]
    )


def _format_refs(output: Mapping[str, Any], field: str) -> str:
    refs = set(output.get(field, []))
    if not refs:
        return "- None accepted in the grounded output."
    by_id = {
        item["evidence_ref_id"]: item
        for item in output.get("allowed_evidence_references", [])
    }
    lines = []
    for ref in sorted(refs):
        item = by_id.get(ref, {"summary": ref, "evidence_family": "unknown"})
        lines.append(
            f"- `{ref}` ({item.get('evidence_family')}): {item.get('summary')}"
        )
    return "\n".join(lines)


def _format_list(title: str, values: Sequence[Any]) -> str:
    if not values:
        return f"### {title}\n\n- None."
    return f"### {title}\n\n" + "\n".join(f"- {value}" for value in values)


def _manifest(
    *,
    output_directory: Path,
    structured_output: Mapping[str, Any],
    run_id: str,
    ticker: str,
) -> dict[str, Any]:
    return {
        "manifest_format_version": "market-engine-grounded-advisory-output-manifest-v1",
        "run_id": run_id,
        "ticker": ticker,
        "advisory_status": structured_output["advisory_status"],
        "artifact_count": 6,
        "non_production_artifact": True,
        "local_only": True,
        "grounding_validator": "ME-CI09.validate_advisory_response_grounding",
        "delivery_performed": False,
        "portfolio_write_performed": False,
        "watchlist_write_performed": False,
        "broker_action_performed": False,
        "artifact_directory": output_directory.as_posix(),
    }


def _raw_response_payload(invocation_result: ModelInvocationResult) -> dict[str, Any]:
    raw_provider_response = (
        dict(invocation_result.raw_provider_response)
        if isinstance(invocation_result.raw_provider_response, Mapping)
        else None
    )
    return {
        "schema_version": "market-engine-raw-model-response-v2",
        "artifact_type": "market-engine-raw-model-response",
        "invocation_state": invocation_result.invocation_state,
        "provider_name": invocation_result.provider_name,
        "model_name": invocation_result.model_name,
        "provider_request_id": invocation_result.provider_request_id,
        "received_at": invocation_result.received_at,
        "finish_reason": invocation_result.finish_reason,
        "usage_metadata": dict(invocation_result.usage_metadata or {}),
        "raw_provider_response": raw_provider_response,
        "raw_output": invocation_result.raw_output,
        "raw_provider_response_hash": (
            _sha256_json(raw_provider_response) if raw_provider_response else None
        ),
        "raw_output_hash": _sha256_text(invocation_result.raw_output or ""),
        "error_message": invocation_result.error_message,
        "raw_output_is_grounded": False,
        "delivery_eligible": False,
    }


def _blocked_invocation_result(reason: str) -> ModelInvocationResult:
    return ModelInvocationResult(
        invocation_state="request_blocked",
        provider_name="not_configured",
        model_name="not_configured",
        raw_output=None,
        error_message=reason,
    )


def _blocked_status(
    invocation_result: ModelInvocationResult,
    parser_result: Mapping[str, Any],
) -> str:
    if invocation_result.invocation_state == "request_blocked":
        return "blocked_invocation_not_configured"
    if invocation_result.invocation_state != "response_received":
        return "blocked_invocation_failed"
    if parser_result.get("status") != "valid":
        return "blocked_model_output_invalid"
    return "blocked_validation_failed"


def _blocked_executive_conclusion(
    source_summary: Mapping[str, Any],
    invocation_result: ModelInvocationResult,
) -> str:
    if invocation_result.invocation_state == "request_blocked":
        return (
            "No grounded advisory conclusion was generated because the model invocation boundary is blocked: "
            f"{invocation_result.error_message}"
        )
    return (
        "No successful CI09-grounded advisory conclusion was accepted. The source remains "
        f"{source_summary.get('readiness_level')} with blocked stage {source_summary.get('blocked_stage')}."
    )


def _extract_openai_output_text(payload: Mapping[str, Any]) -> str | None:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
        if chunks:
            return "\n".join(chunks)
    return None


def _extract_openai_refusal(payload: Mapping[str, Any]) -> str | None:
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "refusal":
                    refusal = part.get("refusal")
                    return str(refusal) if refusal else "provider_refusal"
    return None


def _extract_incomplete_reason(payload: Mapping[str, Any]) -> str | None:
    if payload.get("status") != "incomplete":
        return None
    details = _object(payload.get("incomplete_details"))
    reason = details.get("reason")
    return str(reason) if reason else "unknown"


def _find_finish_reason(payload: Mapping[str, Any]) -> str | None:
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, Mapping) and item.get("finish_reason"):
                return str(item.get("finish_reason"))
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GroundedAdvisoryOutputError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GroundedAdvisoryOutputError(f"JSON file is malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise GroundedAdvisoryOutputError(f"JSON file must contain an object: {path}")
    return payload


def _validated_output_root(output_root: Path | str) -> Path:
    root = Path(output_root)
    if any(part == ".." for part in root.parts):
        raise GroundedAdvisoryOutputError(
            f"Output root must not contain parent traversal: {root}"
        )
    return root


def _resolved_child(parent: Path, child_name: str) -> Path:
    child = parent / child_name
    resolved = child.resolve()
    if parent not in resolved.parents and resolved != parent:
        raise GroundedAdvisoryOutputError(
            f"Resolved output path escapes artifact root: {child}"
        )
    return resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for segment in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current


def _nested_text(payload: Mapping[str, Any], path: tuple[str, ...]) -> str | None:
    value = _nested(payload, path)
    return value if isinstance(value, str) else None


def _list_of_text(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _issue(code: str, path: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {"code": code, "path": path, "expected": expected, "actual": actual}


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_path_segment(value: str, field_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise GroundedAdvisoryOutputError(
            f"{field_name} must be a safe path segment: {value}"
        )
    return value


def _remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()


if __name__ == "__main__":
    main()
