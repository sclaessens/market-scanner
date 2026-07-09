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
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TextIO


CI10_CONTRACT_NAME = "controlled_model_invocation_boundary"
CI10_CONTRACT_VERSION = "v1"
CI10_SCHEMA_VERSION = "market-engine-controlled-model-invocation-boundary-v1"
ADVISORY_OUTPUT_SCHEMA_VERSION = "market-engine-grounded-advisory-output-v1"
ADVISORY_OUTPUT_ARTIFACT_TYPE = "market-engine-grounded-advisory-output"
ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION = (
    "market-engine-grounded-advisory-model-response-v1"
)
DEFAULT_OUTPUT_ROOT = "artifacts/market_engine/grounded_advisory_outputs"
SUPPORTED_SOURCE_ARTIFACT_VERSION = "market-engine-local-dry-run-artifact-v1"
SUPPORTED_SOURCE_ARTIFACT_TYPE = "market_engine_end_to_end_dry_run"
SUCCESS_STATUSES = {"grounded_advisory_generated"}
BLOCKED_STATUSES = {
    "blocked_source_not_supported",
    "blocked_source_not_readable",
    "blocked_invocation_not_configured",
    "blocked_invocation_failed",
    "blocked_model_output_invalid",
    "blocked_validation_failed",
}
FORBIDDEN_ACTION_TERMS = (
    "buy now",
    "sell now",
    "place an order",
    "set a stop",
    "target weight",
    "position size",
    "exact shares",
    "exact cash amount",
)


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
    output_base = Path(output_root)
    source_artifact = _read_json_object(source_path)
    source_validation = _validate_source_artifact(source_artifact)
    source_summary = _source_summary(source_artifact, source_path)
    ticker = _safe_path_segment(source_summary["ticker"], "ticker")
    safe_run_id = _safe_path_segment(run_id, "run_id")
    output_directory = output_base.resolve() / safe_run_id / ticker
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
    invocation_result = (
        _blocked_invocation_result("Source artifact failed pre-invocation validation.")
        if not source_validation["valid"]
        else (invoker or OpenAIResponsesInvoker.from_environment()).invoke(
            invocation_request
        )
    )
    parser_result = _parse_model_response(invocation_result.raw_output)
    validation_result = _validate_model_response(
        parsed_response=parser_result.get("parsed_response"),
        invocation_result=invocation_result,
        source_summary=source_summary,
        source_validation=source_validation,
    )
    structured_output = _structured_output(
        source_summary=source_summary,
        source_validation=source_validation,
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
            "invocation_state": invocation_result.invocation_state,
            "structured_output_path": structured_output_path.as_posix(),
            "report_path": report_path.as_posix(),
        },
    )


class OpenAIResponsesInvoker:
    """Minimal non-streaming CI10-boundary invoker using explicit environment config."""

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
        payload = {
            "model": self._model,
            "input": _prompt_text(request),
            "temperature": 0,
        }
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
        output_text = _extract_openai_output_text(response_payload)
        return ModelInvocationResult(
            invocation_state="response_received",
            provider_name="openai",
            model_name=self._model,
            raw_output=output_text,
            provider_request_id=str(response_payload.get("id"))
            if response_payload.get("id")
            else None,
            finish_reason=_find_finish_reason(response_payload),
            usage_metadata={
                "provider_usage": response_payload.get("usage"),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
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
            "The command preserves the CI10 invocation boundary, validates allowed evidence "
            "references, and writes only local artifacts."
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
            _issue("missing_analysis_context_readiness", "$.payload.analysis_context_readiness", "object", None)
        )
    if not _nested_text(payload, ("provenance_summary", "fundamental_observations", "source_refresh_snapshot_id")):
        issues.append(
            _issue(
                "missing_fundamental_provenance",
                "$.payload.provenance_summary.fundamental_observations",
                "source_refresh_snapshot_id",
                None,
            )
        )
    return {"status": "valid" if not issues else "invalid", "valid": not issues, "issues": issues}


def _source_summary(
    source_artifact: Mapping[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    payload = _object(source_artifact.get("payload"))
    readiness = _object(payload.get("analysis_context_readiness"))
    provenance = _object(payload.get("provenance_summary"))
    recommendation = _object(_nested(provenance, ("portfolio_review", "recommendation_review_provenance")))
    ticker = str(payload.get("ticker") or _nested_text(recommendation, ("input_provenance", "ticker")) or "UNKNOWN")
    evidence_refs = _evidence_refs(payload)
    actionability = bool(readiness.get("actionable_review_allowed")) and bool(
        readiness.get("decision_engine_ready")
    )
    return {
        "ticker": ticker,
        "instrument": ticker,
        "company_name": _nested_text(payload, ("company_identity", "company_name")),
        "source_artifact_path": source_path.as_posix(),
        "source_artifact_format": source_artifact.get("artifact_format_version"),
        "source_artifact_type": source_artifact.get("artifact_type"),
        "source_run_id": str(payload.get("dry_run_id") or "unknown-run"),
        "source_generated_at": str(payload.get("generated_at") or source_artifact.get("artifact_created_at") or "unknown"),
        "input_mode": str(payload.get("input_mode") or "unknown"),
        "dry_run_state": "dry_run_blocked" if payload.get("blocked_stage") else "dry_run_completed",
        "blocked_stage": payload.get("blocked_stage"),
        "blocked_reasons": _list(payload.get("blocked_reasons")) + _list(readiness.get("blocked_reasons")),
        "missing_data": _list(payload.get("missing_data_summary")) + _list(readiness.get("evidence_families_missing")),
        "stale_data": ["context_stale"] if readiness.get("context_stale") else [],
        "readiness_level": str(readiness.get("readiness_level") or "unknown"),
        "actionable_review_allowed": bool(readiness.get("actionable_review_allowed")),
        "decision_engine_ready": bool(readiness.get("decision_engine_ready")),
        "actionability_allowed": actionability,
        "recommendation_review_state": str(recommendation.get("review_state") or "unknown"),
        "recommendation_review_category": str(recommendation.get("review_category") or "unknown"),
        "portfolio_review_state": "blocked" if payload.get("blocked_stage") == "portfolio_review" else "unknown",
        "provenance_summary": provenance,
        "evidence_refs": evidence_refs,
        "supporting_evidence": _supporting_evidence(evidence_refs),
        "risk_evidence": _risk_evidence(payload, readiness, recommendation),
    }


def _build_invocation_request(
    *,
    source_artifact: Mapping[str, Any],
    source_path: Path,
    source_validation: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    run_id: str,
    generated_at: str,
) -> dict[str, Any]:
    prompt_package = {
        "schema_version": "market-engine-grounded-advisory-input-package-v1",
        "artifact_type": "market-engine-grounded-advisory-input-package",
        "source_artifact_identity": {
            "path": source_path.as_posix(),
            "sha256": _sha256_json(source_artifact),
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
            "supporting_evidence": source_summary["supporting_evidence"],
            "risk_evidence": source_summary["risk_evidence"],
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
            "sha256": _sha256_json(prompt_package),
        },
        "source_artifact_identity": prompt_package["source_artifact_identity"],
        "instrument_identity": prompt_package["instrument_identity"],
        "run_identity": {"run_id": source_summary["source_run_id"]},
        "invocation_identity": {"invocation_id": run_id},
        "attempt_identity": {"attempt_id": f"{run_id}-attempt-1", "attempt_sequence": 1},
        "question_classification": prompt_package["question_classification"],
        "permitted_use_case": prompt_package["question_classification"]["permitted_use_case"],
        "provider_identity": {"provider_name": "openai", "provider_api_family": "responses"},
        "model_identity": {
            "model_name": os.environ.get("MARKET_ENGINE_ADVISORY_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "not_configured"
        },
        "model_capability_profile": {
            "structured_json_output_required": True,
            "non_streaming_required": True,
            "tool_use_allowed": False,
            "external_browsing_allowed": False,
        },
        "instruction_contract": {
            "system_boundary": "Use only supplied Market Engine context. Do not invent facts, prices, news, targets, sizing, allocation, or execution guidance.",
            "user_question_cannot_override_boundary": True,
        },
        "input_context": prompt_package,
        "response_schema_contract": {
            "schema_version": ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION,
            "artifact_type": "market-engine-grounded-advisory-model-response",
        },
        "budget_limits": {"max_input_tokens": None, "max_output_tokens": None},
        "timeout_policy": {"total_invocation_timeout_seconds": 60},
        "retry_policy": {"max_attempts": 1, "retry_enabled": False},
        "idempotency_policy": {"idempotency_key": _sha256_json(prompt_package)},
        "data_handling_policy": {"persist_secrets": False, "external_browsing_allowed": False},
        "grounding_handoff_contract": {
            "allowed_evidence_references": source_summary["evidence_refs"],
            "fail_closed_on_invalid_reference": True,
        },
        "audit_context": {"source_validation": source_validation},
    }


def _prompt_text(request: Mapping[str, Any]) -> str:
    return (
        "Return only JSON matching schema_version "
        f"{ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION}. "
        "Do not use external knowledge. Use only this request JSON:\n"
        + json.dumps(request, indent=2, sort_keys=True)
    )


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
    if not isinstance(parsed, dict):
        return {
            "status": "invalid",
            "parser_state": "schema_invalid",
            "parsed_response": None,
            "issues": [{"code": "response_not_object", "path": "$"}],
        }
    return {"status": "valid", "parser_state": "parsed", "parsed_response": parsed, "issues": []}


def _validate_model_response(
    *,
    parsed_response: Any,
    invocation_result: ModelInvocationResult,
    source_summary: Mapping[str, Any],
    source_validation: Mapping[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not source_validation.get("valid"):
        issues.append({"code": "source_validation_failed", "path": "$.source"})
        issues.extend(dict(issue) for issue in source_validation.get("issues", []))
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
        return {"status": "invalid", "valid": False, "issues": issues}
    required = (
        "schema_version",
        "advisory_status",
        "executive_conclusion",
        "supporting_evidence_references",
        "risk_evidence_references",
        "limitations",
        "practical_interpretation",
        "confidence_and_evidence_quality",
    )
    for field in required:
        if field not in parsed_response:
            issues.append({"code": "missing_required_field", "path": f"$.{field}"})
    if parsed_response.get("schema_version") != ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION:
        issues.append(
            {
                "code": "schema_version_mismatch",
                "path": "$.schema_version",
                "expected": ADVISORY_MODEL_RESPONSE_SCHEMA_VERSION,
                "actual": parsed_response.get("schema_version"),
            }
        )
    allowed_refs = {ref["evidence_ref_id"] for ref in source_summary["evidence_refs"]}
    for field in ("supporting_evidence_references", "risk_evidence_references"):
        refs = parsed_response.get(field, [])
        if not isinstance(refs, list):
            issues.append({"code": "invalid_reference_list", "path": f"$.{field}"})
            continue
        for index, ref in enumerate(refs):
            if ref not in allowed_refs:
                issues.append(
                    {
                        "code": "unknown_evidence_reference",
                        "path": f"$.{field}[{index}]",
                        "actual": ref,
                    }
                )
    if not str(parsed_response.get("executive_conclusion", "")).strip():
        issues.append({"code": "empty_executive_conclusion", "path": "$.executive_conclusion"})
    response_text = json.dumps(parsed_response, sort_keys=True).lower()
    if not source_summary.get("actionability_allowed") and any(
        term in response_text for term in FORBIDDEN_ACTION_TERMS
    ):
        issues.append(
            {
                "code": "actionable_language_not_allowed",
                "path": "$",
                "message": "Source artifact is not actionable or decision ready.",
            }
        )
    return {"status": "valid" if not issues else "invalid", "valid": not issues, "issues": issues}


def _structured_output(
    *,
    source_summary: Mapping[str, Any],
    source_validation: Mapping[str, Any],
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
        executive = str(parsed["executive_conclusion"])
        supporting_refs = list(parsed.get("supporting_evidence_references", []))
        risk_refs = list(parsed.get("risk_evidence_references", []))
        limitations = list(parsed.get("limitations", []))
        practical = str(parsed.get("practical_interpretation", ""))
        confidence = str(parsed.get("confidence_and_evidence_quality", ""))
    else:
        advisory_status = _blocked_status(invocation_result, source_validation, parser_result)
        executive = _blocked_executive_conclusion(source_summary, validation_result, invocation_result)
        supporting_refs = []
        risk_refs = [
            ref["evidence_ref_id"]
            for ref in source_summary["evidence_refs"]
            if ref["evidence_family"] in {"blocked_state", "missing_data", "readiness"}
        ]
        limitations = _list(source_summary.get("blocked_reasons")) + _list(
            source_summary.get("missing_data")
        )
        practical = "Treat this as a blocked/non-actionable advisory generation result. Review the source artifact and invocation blocker before using any advisory output."
        confidence = "No successful grounded model response was accepted."
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
            "invocation_request_hash": _sha256_json(invocation_request),
            "raw_output_hash": _sha256_text(invocation_result.raw_output or ""),
        },
    }


def _render_report(output: Mapping[str, Any]) -> str:
    readiness = _object(output.get("source_readiness"))
    return "\n".join(
        [
            f"# Grounded Advisory Report - {output['ticker']}",
            "",
            "This report is based only on the referenced Market Engine artifact. It does not use external model knowledge, live prices, news, broker data, portfolio mutation, or delivery authority.",
            "",
            "## Instrument and context",
            "",
            f"- Ticker: {output['ticker']}",
            f"- Company/instrument: {output['instrument_identity'].get('company_name') or output['instrument_identity'].get('instrument')}",
            f"- Source artifact: `{output['source_artifact']['path']}`",
            f"- Source run: `{output['source_artifact']['run_id']}`",
            f"- Source generated at: {output['source_artifact']['generated_at']}",
            f"- Input mode: {output['source_artifact']['input_mode']}",
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
            f"- Invocation state: {output['invocation_boundary']['invocation_state']}",
            f"- Provider/model: {output['invocation_boundary']['provider_name']} / {output['invocation_boundary']['model_name']}",
            "",
        ]
    )


def _format_refs(output: Mapping[str, Any], field: str) -> str:
    refs = set(output.get(field, []))
    if not refs:
        return "- None accepted in the grounded output."
    by_id = {item["evidence_ref_id"]: item for item in output.get("allowed_evidence_references", [])}
    lines = []
    for ref in sorted(refs):
        item = by_id.get(ref, {"summary": ref, "evidence_family": "unknown"})
        lines.append(f"- `{ref}` ({item.get('evidence_family')}): {item.get('summary')}")
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
        "delivery_performed": False,
        "portfolio_write_performed": False,
        "watchlist_write_performed": False,
        "broker_action_performed": False,
        "artifact_directory": output_directory.as_posix(),
    }


def _raw_response_payload(invocation_result: ModelInvocationResult) -> dict[str, Any]:
    return {
        "schema_version": "market-engine-raw-model-response-v1",
        "artifact_type": "market-engine-raw-model-response",
        "invocation_state": invocation_result.invocation_state,
        "provider_name": invocation_result.provider_name,
        "model_name": invocation_result.model_name,
        "provider_request_id": invocation_result.provider_request_id,
        "finish_reason": invocation_result.finish_reason,
        "usage_metadata": dict(invocation_result.usage_metadata or {}),
        "raw_output": invocation_result.raw_output,
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
    source_validation: Mapping[str, Any],
    parser_result: Mapping[str, Any],
) -> str:
    if not source_validation.get("valid"):
        return "blocked_source_not_supported"
    if invocation_result.invocation_state == "request_blocked":
        return "blocked_invocation_not_configured"
    if invocation_result.invocation_state != "response_received":
        return "blocked_invocation_failed"
    if parser_result.get("status") != "valid":
        return "blocked_model_output_invalid"
    return "blocked_validation_failed"


def _blocked_executive_conclusion(
    source_summary: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    invocation_result: ModelInvocationResult,
) -> str:
    if invocation_result.invocation_state == "request_blocked":
        return (
            "No grounded advisory conclusion was generated because the model invocation boundary is blocked: "
            f"{invocation_result.error_message}"
        )
    return (
        "No successful grounded advisory conclusion was accepted. The source remains "
        f"{source_summary.get('readiness_level')} with blocked stage {source_summary.get('blocked_stage')}."
    )


def _evidence_refs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    readiness = _object(payload.get("analysis_context_readiness"))
    refs.append(
        {
            "evidence_ref_id": "readiness:analysis_context",
            "evidence_family": "readiness",
            "path": "$.payload.analysis_context_readiness",
            "summary": f"Readiness is {readiness.get('readiness_level')} with actionable_review_allowed={readiness.get('actionable_review_allowed')} and decision_engine_ready={readiness.get('decision_engine_ready')}.",
        }
    )
    if payload.get("blocked_stage"):
        refs.append(
            {
                "evidence_ref_id": "blocked:stage",
                "evidence_family": "blocked_state",
                "path": "$.payload.blocked_stage",
                "summary": f"Dry-run is blocked at {payload.get('blocked_stage')}.",
            }
        )
    for index, reason in enumerate(_list(payload.get("blocked_reasons")) + _list(readiness.get("blocked_reasons"))):
        refs.append(
            {
                "evidence_ref_id": f"blocked:reason:{index}",
                "evidence_family": "blocked_state",
                "path": "$.payload.blocked_reasons",
                "summary": str(reason),
            }
        )
    for index, missing in enumerate(_list(payload.get("missing_data_summary")) + _list(readiness.get("evidence_families_missing"))):
        refs.append(
            {
                "evidence_ref_id": f"missing:data:{index}",
                "evidence_family": "missing_data",
                "path": "$.payload.missing_data_summary",
                "summary": str(missing),
            }
        )
    recommendation = _object(
        _nested(_object(payload.get("provenance_summary")), ("portfolio_review", "recommendation_review_provenance"))
    )
    if recommendation:
        refs.append(
            {
                "evidence_ref_id": "recommendation:review",
                "evidence_family": "recommendation_review",
                "path": "$.payload.provenance_summary.portfolio_review.recommendation_review_provenance",
                "summary": f"Recommendation Review state is {recommendation.get('review_state')} and category is {recommendation.get('review_category')}.",
            }
        )
    setup_messages = _collect_setup_messages(payload)
    for index, message in enumerate(setup_messages[:8]):
        refs.append(
            {
                "evidence_ref_id": f"setup:evidence:{index}",
                "evidence_family": "setup_detection",
                "path": "$.payload.provenance_summary",
                "summary": message,
            }
        )
    return refs


def _supporting_evidence(refs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(ref)
        for ref in refs
        if ref.get("evidence_family") in {"readiness", "recommendation_review", "setup_detection"}
    ]


def _risk_evidence(
    payload: Mapping[str, Any],
    readiness: Mapping[str, Any],
    recommendation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    risks = []
    for reason in _list(payload.get("blocked_reasons")) + _list(readiness.get("blocked_reasons")):
        risks.append({"family": "blocked_state", "summary": str(reason)})
    for item in _list(payload.get("missing_data_summary")) + _list(readiness.get("evidence_families_missing")):
        risks.append({"family": "missing_data", "summary": str(item)})
    if recommendation.get("review_state") == "human_review_required":
        risks.append(
            {
                "family": "recommendation_review",
                "summary": "Recommendation Review requires human review.",
            }
        )
    return risks


def _collect_setup_messages(payload: Mapping[str, Any]) -> list[str]:
    messages: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"setup_message", "message"} and isinstance(child, str):
                    if child not in messages:
                        messages.append(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload.get("provenance_summary"))
    return messages


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


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
