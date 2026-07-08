from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from market_engine.advisory.advisory_artifact_validation import (
    ADVISORY_ARTIFACT_VALIDATOR_VERSION,
    validation_evidence_payload,
    validate_chatgpt_ready_advisory_artifact,
)


CHATGPT_READY_ADVISORY_ARTIFACT_SCHEMA_VERSION = (
    "market-engine-chatgpt-ready-advisory-artifact-v1"
)
CHATGPT_READY_ADVISORY_ARTIFACT_TYPE = (
    "market-engine-chatgpt-ready-advisory-artifact"
)
CHATGPT_READY_ADVISORY_ARTIFACT_FORMAT_VERSION = (
    "market-engine-chatgpt-ready-advisory-artifact-v1"
)
CHATGPT_READY_ADVISORY_MANIFEST_FORMAT_VERSION = (
    "market-engine-chatgpt-ready-advisory-artifact-manifest-v1"
)
CHATGPT_READY_ADVISORY_ARTIFACT_PATH_CATEGORY = (
    "artifacts/market_engine/chatgpt_ready_advisory"
)
LOCAL_ADVISORY_PERSISTENCE_MODE = "local_chatgpt_ready_advisory_only"

STRUCTURED_DECISION_OUTPUT_SCHEMA_VERSION = "structured-decision-output-v1"
STRUCTURED_DECISION_OUTPUT_ARTIFACT_TYPE = "market-engine-structured-decision-output"
CHATGPT_ADVISORY_CONTEXT_SCHEMA_VERSION = "chatgpt-advisory-context-v1"
CHATGPT_ADVISORY_CONTEXT_ARTIFACT_TYPE = "market-engine-chatgpt-advisory-context"
PORTFOLIO_INTELLIGENCE_SCHEMA_VERSION = (
    "chatgpt-portfolio-intelligence-context-v1"
)
PORTFOLIO_INTELLIGENCE_ARTIFACT_TYPE = (
    "market-engine-chatgpt-portfolio-intelligence-context"
)
EXPLAINABILITY_SCHEMA_VERSION = (
    "chatgpt-explainability-change-rationale-context-v1"
)
EXPLAINABILITY_ARTIFACT_TYPE = (
    "market-engine-chatgpt-explainability-change-rationale-context"
)

REQUIRED_INPUT_FILENAME = "structured_decision_output.json"
OPTIONAL_INPUT_FILENAMES = {
    "chatgpt_advisory_context": "chatgpt_advisory_context.json",
    "portfolio_intelligence_context": "chatgpt_portfolio_intelligence_context.json",
    "explainability_change_rationale_context": (
        "chatgpt_explainability_change_rationale_context.json"
    ),
    "governor_context": "governor_context.json",
    "dispatch_context": "dispatch_context.json",
}

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ChatGPTReadyAdvisoryArtifactError(ValueError):
    """Raised when a ChatGPT-ready advisory artifact cannot be assembled safely."""


@dataclass(frozen=True)
class ChatGPTReadyAdvisoryArtifactPersistenceResult:
    run_directory: Path
    instrument_directory: Path
    artifact_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def assemble_chatgpt_ready_advisory_artifact(
    *,
    structured_decision_output: Mapping[str, Any],
    generated_at: str,
    chatgpt_advisory_context: Mapping[str, Any] | None = None,
    portfolio_intelligence_context: Mapping[str, Any] | None = None,
    explainability_change_rationale_context: Mapping[str, Any] | None = None,
    governor_context: Mapping[str, Any] | None = None,
    dispatch_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a deterministic local ChatGPT-ready advisory artifact.

    The assembler validates identity and supported schema versions, preserves
    upstream fields, and fails closed by downgrading composition eligibility when
    supplied contexts conflict. It does not calculate market, portfolio,
    Governor, or Decision Engine semantics.
    """

    if not isinstance(generated_at, str) or not generated_at:
        raise ChatGPTReadyAdvisoryArtifactError("generated_at is required.")

    sdo = _required_mapping(structured_decision_output, "structured_decision_output")
    validation_errors = _validate_typed_artifact(
        sdo,
        family="structured_decision_output",
        expected_schema=STRUCTURED_DECISION_OUTPUT_SCHEMA_VERSION,
        expected_type=STRUCTURED_DECISION_OUTPUT_ARTIFACT_TYPE,
    )
    ticker = _text_or_none(sdo.get("ticker"))
    run_id = _text_or_none(sdo.get("run_id"))
    instrument = _mapping_or_none(sdo.get("instrument"))
    if not ticker:
        validation_errors.append("structured_decision_output.ticker_missing")
    if not run_id:
        validation_errors.append("structured_decision_output.run_id_missing")
    if not instrument:
        validation_errors.append("structured_decision_output.instrument_missing")
    elif _text_or_none(instrument.get("ticker")) != ticker:
        validation_errors.append("structured_decision_output.instrument_ticker_conflict")

    if validation_errors:
        raise ChatGPTReadyAdvisoryArtifactError("; ".join(sorted(validation_errors)))

    supplied_contexts = {
        "chatgpt_advisory_context": chatgpt_advisory_context,
        "portfolio_intelligence_context": portfolio_intelligence_context,
        "explainability_change_rationale_context": (
            explainability_change_rationale_context
        ),
        "governor_context": governor_context,
        "dispatch_context": dispatch_context,
    }
    normalized_contexts = {
        name: _json_ready(context, path=name) if context is not None else None
        for name, context in supplied_contexts.items()
    }

    validation_errors = []
    semantic_warnings: list[str] = []
    blockers: list[str] = []
    missing_context: list[dict[str, Any]] = []

    validation_errors.extend(
        _validate_optional_typed_context(
            normalized_contexts["chatgpt_advisory_context"],
            family="chatgpt_advisory_context",
            expected_schema=CHATGPT_ADVISORY_CONTEXT_SCHEMA_VERSION,
            expected_type=CHATGPT_ADVISORY_CONTEXT_ARTIFACT_TYPE,
            ticker=ticker,
            compatible_run_id=run_id,
        )
    )
    validation_errors.extend(
        _validate_optional_typed_context(
            normalized_contexts["portfolio_intelligence_context"],
            family="portfolio_intelligence_context",
            expected_schema=PORTFOLIO_INTELLIGENCE_SCHEMA_VERSION,
            expected_type=PORTFOLIO_INTELLIGENCE_ARTIFACT_TYPE,
            ticker=ticker,
            compatible_run_id=None,
        )
    )
    validation_errors.extend(
        _validate_optional_typed_context(
            normalized_contexts["explainability_change_rationale_context"],
            family="explainability_change_rationale_context",
            expected_schema=EXPLAINABILITY_SCHEMA_VERSION,
            expected_type=EXPLAINABILITY_ARTIFACT_TYPE,
            ticker=ticker,
            compatible_run_id=None,
        )
    )
    validation_errors.extend(
        _validate_optional_identity_context(
            normalized_contexts["governor_context"],
            family="governor_context",
            ticker=ticker,
            compatible_run_id=run_id,
        )
    )
    validation_errors.extend(
        _validate_optional_identity_context(
            normalized_contexts["dispatch_context"],
            family="dispatch_context",
            ticker=ticker,
            compatible_run_id=run_id,
        )
    )

    for error in validation_errors:
        blockers.append(error)

    if normalized_contexts["chatgpt_advisory_context"] is None:
        missing_context.append(
            _missing_context_entry(
                "chatgpt_advisory_context",
                "optional_but_recommended",
                "Advisory eligibility falls back to Structured Decision Output only.",
            )
        )
    if normalized_contexts["portfolio_intelligence_context"] is None:
        missing_context.append(
            _missing_context_entry(
                "portfolio_intelligence_context",
                "portfolio_specific_context_unavailable",
                "Portfolio absence is not converted to zero holdings or zero cash.",
            )
        )
    if normalized_contexts["explainability_change_rationale_context"] is None:
        missing_context.append(
            _missing_context_entry(
                "explainability_change_rationale_context",
                "change_rationale_unavailable",
                "Current-state fields may exist upstream, but change rationale is unavailable.",
            )
        )
    if normalized_contexts["dispatch_context"] is None:
        missing_context.append(
            _missing_context_entry(
                "dispatch_context",
                "presentation_context_absent",
                "Dispatch context is presentation-only and optional.",
            )
        )

    eligibility = _aggregate_advisory_eligibility(
        sdo=sdo,
        advisory_context=normalized_contexts["chatgpt_advisory_context"],
        portfolio_context=normalized_contexts["portfolio_intelligence_context"],
        explainability_context=(
            normalized_contexts["explainability_change_rationale_context"]
        ),
        validation_errors=validation_errors,
    )
    blockers.extend(eligibility["blocking_reasons"])

    source_artifact_references = _source_artifact_references(
        {
            "structured_decision_output": sdo,
            **normalized_contexts,
        }
    )
    composition_state = _composition_state(eligibility["state"], validation_errors)

    artifact = {
        "contract_identity": {
            "schema_version": CHATGPT_READY_ADVISORY_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": CHATGPT_READY_ADVISORY_ARTIFACT_TYPE,
            "contract_name": "chatgpt_ready_advisory_artifact",
            "contract_version": "v1",
        },
        "artifact_identity": {
            "artifact_type": CHATGPT_READY_ADVISORY_ARTIFACT_TYPE,
            "schema_version": CHATGPT_READY_ADVISORY_ARTIFACT_SCHEMA_VERSION,
            "artifact_format_version": CHATGPT_READY_ADVISORY_ARTIFACT_FORMAT_VERSION,
            "persistence_mode": LOCAL_ADVISORY_PERSISTENCE_MODE,
            "non_production_artifact": True,
        },
        "run_identity": {
            "run_id": run_id,
            "source_structured_decision_run_id": run_id,
            "context_run_ids": _context_run_ids(normalized_contexts),
        },
        "instrument_identity": {
            "ticker": ticker,
            "instrument": _json_ready(instrument, path="instrument"),
        },
        "generated_at": generated_at,
        "source_artifact_references": source_artifact_references,
        "composition_status": {
            "state": composition_state,
            "blocking_validation": bool(validation_errors),
            "semantic_override_performed": False,
            "source_precedence": [
                "structured_decision_output_over_dispatch_presentation",
                "portfolio_intelligence_context_over_narrative_text",
                "governor_context_over_governor_explanation_summary",
                "machine_readable_blockers_over_free_text",
            ],
        },
        "advisory_eligibility": eligibility,
        "structured_decision_context": {
            "include_mode": "embedded_canonical_context",
            "schema_version": sdo["schema_version"],
            "artifact_type": sdo["artifact_type"],
            "source_priority": "canonical",
            "payload": sdo,
        },
        "portfolio_intelligence_context": _context_payload_or_absent(
            normalized_contexts["portfolio_intelligence_context"],
            family="portfolio_intelligence_context",
            absent_state="unavailable",
            absent_reason="approved_portfolio_intelligence_context_absent",
        ),
        "explainability_change_rationale_context": _context_payload_or_absent(
            normalized_contexts["explainability_change_rationale_context"],
            family="explainability_change_rationale_context",
            absent_state="unavailable",
            absent_reason="approved_explainability_change_rationale_context_absent",
        ),
        "governor_context": _context_payload_or_absent(
            normalized_contexts["governor_context"],
            family="governor_context",
            absent_state="absent",
            absent_reason="approved_governor_context_absent",
        ),
        "dispatch_context": _context_payload_or_absent(
            normalized_contexts["dispatch_context"],
            family="dispatch_context",
            absent_state="absent",
            absent_reason="dispatch_presentation_context_absent",
        ),
        "provenance_context": {
            "source_artifact_refs": source_artifact_references,
            "raw_provider_payload_included": False,
            "context_families_present": _present_context_families(normalized_contexts),
            "context_families_missing": sorted(
                entry["family"] for entry in missing_context
            ),
        },
        "freshness_context": _freshness_context(sdo, normalized_contexts),
        "uncertainty_context": _uncertainty_context(
            sdo=sdo,
            advisory_context=normalized_contexts["chatgpt_advisory_context"],
            missing_context=missing_context,
            blockers=blockers,
        ),
        "blockers": _stable_unique(blockers),
        "missing_context": sorted(missing_context, key=lambda item: item["family"]),
        "validation_summary": {
            "validation_state": (
                "blocked" if validation_errors else "valid_with_limitations"
            ),
            "errors": sorted(validation_errors),
            "warnings": sorted(semantic_warnings),
            "required_sources_present": {
                "structured_decision_output": True,
            },
            "optional_sources_present": {
                name: context is not None
                for name, context in normalized_contexts.items()
            },
            "no_semantic_upgrade_performed": True,
        },
    }
    return _json_ready(artifact, path="artifact")


def load_chatgpt_ready_advisory_inputs(
    input_artifact_dir: Path | str,
) -> dict[str, dict[str, Any] | None]:
    """Load explicit ME-CI05 inputs from one non-ambiguous local directory."""

    input_dir = Path(input_artifact_dir)
    if not input_dir.is_dir():
        raise ChatGPTReadyAdvisoryArtifactError(
            f"Input artifact directory does not exist: {input_dir}"
        )
    required_path = input_dir / REQUIRED_INPUT_FILENAME
    if not required_path.is_file():
        raise ChatGPTReadyAdvisoryArtifactError(
            f"Required input artifact is missing: {required_path}"
        )
    loaded: dict[str, dict[str, Any] | None] = {
        "structured_decision_output": _read_json_object(required_path)
    }
    for family, filename in OPTIONAL_INPUT_FILENAMES.items():
        path = input_dir / filename
        loaded[family] = _read_json_object(path) if path.exists() else None
    return loaded


def compose_chatgpt_ready_advisory_artifact_from_directory(
    *,
    input_artifact_dir: Path | str,
    output_root: Path | str,
    generated_at: str,
    allow_overwrite: bool = False,
) -> ChatGPTReadyAdvisoryArtifactPersistenceResult:
    inputs = load_chatgpt_ready_advisory_inputs(input_artifact_dir)
    artifact = assemble_chatgpt_ready_advisory_artifact(
        structured_decision_output=inputs["structured_decision_output"],
        generated_at=generated_at,
        chatgpt_advisory_context=inputs["chatgpt_advisory_context"],
        portfolio_intelligence_context=inputs["portfolio_intelligence_context"],
        explainability_change_rationale_context=(
            inputs["explainability_change_rationale_context"]
        ),
        governor_context=inputs["governor_context"],
        dispatch_context=inputs["dispatch_context"],
    )
    return persist_chatgpt_ready_advisory_artifact(
        artifact,
        output_root=output_root,
        allow_overwrite=allow_overwrite,
        validated_at=generated_at,
    )


def persist_chatgpt_ready_advisory_artifact(
    artifact: Mapping[str, Any],
    *,
    output_root: Path | str = CHATGPT_READY_ADVISORY_ARTIFACT_PATH_CATEGORY,
    allow_overwrite: bool = False,
    validated_at: str | None = None,
) -> ChatGPTReadyAdvisoryArtifactPersistenceResult:
    payload = _validated_advisory_artifact(artifact)
    run_id = _safe_path_segment(
        payload["run_identity"]["run_id"],
        field_name="run_id",
    )
    ticker = _safe_path_segment(
        payload["instrument_identity"]["ticker"],
        field_name="ticker",
    )
    validation_result = validate_chatgpt_ready_advisory_artifact(payload)
    if not validation_result.valid:
        issue_codes = ", ".join(issue.code for issue in validation_result.issues)
        raise ChatGPTReadyAdvisoryArtifactError(
            "ChatGPT-ready advisory artifact failed contract validation: "
            + issue_codes
        )
    validation_timestamp = validated_at or payload["generated_at"]
    validation_evidence = validation_evidence_payload(
        validation_result,
        validated_at=validation_timestamp,
    )
    payload = {
        **payload,
        "validation_summary": {
            **payload["validation_summary"],
            "contract_validation": validation_evidence,
        },
    }
    root = _validated_output_root(output_root)
    root_resolved = root.resolve()
    run_directory = _resolved_child(root_resolved, run_id)
    instrument_directory = _resolved_child(run_directory, ticker)
    artifact_path = _resolved_child(instrument_directory, "chatgpt_ready_advisory.json")
    manifest_path = _resolved_child(instrument_directory, "manifest.json")

    if instrument_directory.exists() and not allow_overwrite:
        raise ChatGPTReadyAdvisoryArtifactError(
            f"ChatGPT-ready advisory artifact directory already exists: {instrument_directory}"
        )

    manifest = {
        "manifest_format_version": CHATGPT_READY_ADVISORY_MANIFEST_FORMAT_VERSION,
        "artifact_count": 1,
        "artifact_type": CHATGPT_READY_ADVISORY_ARTIFACT_TYPE,
        "schema_version": CHATGPT_READY_ADVISORY_ARTIFACT_SCHEMA_VERSION,
        "artifact_persistence_mode": LOCAL_ADVISORY_PERSISTENCE_MODE,
        "artifact_path_category": CHATGPT_READY_ADVISORY_ARTIFACT_PATH_CATEGORY,
        "non_production_artifact": True,
        "run_id": run_id,
        "ticker": ticker,
        "generated_at": payload["generated_at"],
        "composition_state": payload["composition_status"]["state"],
        "advisory_eligibility_state": payload["advisory_eligibility"]["state"],
        "validation_status": validation_result.status,
        "validator_version": ADVISORY_ARTIFACT_VALIDATOR_VERSION,
        "validated_schema_version": validation_result.validated_schema_version,
        "validation_timestamp": validation_timestamp,
        "validation_issue_count": len(validation_result.issues),
        "artifact_relative_path": _relative_posix(artifact_path, root_resolved),
        "manifest_relative_path": _relative_posix(manifest_path, root_resolved),
    }

    instrument_directory.mkdir(parents=True, exist_ok=allow_overwrite)
    _write_json(artifact_path, payload, allow_overwrite=allow_overwrite)
    _write_json(manifest_path, manifest, allow_overwrite=allow_overwrite)

    return ChatGPTReadyAdvisoryArtifactPersistenceResult(
        run_directory=run_directory,
        instrument_directory=instrument_directory,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ChatGPTReadyAdvisoryArtifactError(f"{name} must be a JSON object.")
    return _json_ready(value, path=name)


def _validate_typed_artifact(
    artifact: Mapping[str, Any],
    *,
    family: str,
    expected_schema: str,
    expected_type: str,
) -> list[str]:
    errors: list[str] = []
    if artifact.get("schema_version") != expected_schema:
        errors.append(f"{family}.unsupported_schema_version")
    if artifact.get("artifact_type") != expected_type:
        errors.append(f"{family}.unsupported_artifact_type")
    return errors


def _validate_optional_typed_context(
    context: Mapping[str, Any] | None,
    *,
    family: str,
    expected_schema: str,
    expected_type: str,
    ticker: str,
    compatible_run_id: str | None,
) -> list[str]:
    if context is None:
        return []
    errors = _validate_typed_artifact(
        context,
        family=family,
        expected_schema=expected_schema,
        expected_type=expected_type,
    )
    errors.extend(
        _validate_optional_identity_context(
            context,
            family=family,
            ticker=ticker,
            compatible_run_id=compatible_run_id,
        )
    )
    return errors


def _validate_optional_identity_context(
    context: Mapping[str, Any] | None,
    *,
    family: str,
    ticker: str,
    compatible_run_id: str | None,
) -> list[str]:
    if context is None:
        return []
    errors: list[str] = []
    observed_tickers = _context_tickers(context)
    conflicting_tickers = sorted(
        observed_ticker
        for observed_ticker in observed_tickers
        if observed_ticker != ticker
    )
    if conflicting_tickers:
        errors.append(f"{family}.ticker_conflict")
    observed_run_id = _text_or_none(context.get("run_id"))
    if compatible_run_id and observed_run_id and observed_run_id != compatible_run_id:
        errors.append(f"{family}.run_identity_conflict")
    return errors


def _context_tickers(context: Mapping[str, Any]) -> set[str]:
    tickers: set[str] = set()
    top_level_ticker = _text_or_none(context.get("ticker"))
    if top_level_ticker:
        tickers.add(top_level_ticker)
    instrument = _mapping_or_none(context.get("instrument"))
    if instrument:
        instrument_ticker = _text_or_none(instrument.get("ticker"))
        if instrument_ticker:
            tickers.add(instrument_ticker)
    subject = _mapping_or_none(context.get("subject"))
    if subject:
        subject_ticker = _text_or_none(subject.get("ticker"))
        if subject_ticker:
            tickers.add(subject_ticker)
    relationship = _mapping_or_none(context.get("recommendation_to_position_relationship"))
    if relationship:
        relationship_ticker = _text_or_none(relationship.get("ticker"))
        if relationship_ticker:
            tickers.add(relationship_ticker)
    return tickers


def _aggregate_advisory_eligibility(
    *,
    sdo: Mapping[str, Any],
    advisory_context: Mapping[str, Any] | None,
    portfolio_context: Mapping[str, Any] | None,
    explainability_context: Mapping[str, Any] | None,
    validation_errors: list[str],
) -> dict[str, Any]:
    source_state = _sdo_eligibility_state(sdo)
    advisory_state = _context_state(
        advisory_context,
        path=("advisory_eligibility", "state"),
        fallback=source_state,
    )
    portfolio_state = _context_state(
        portfolio_context,
        path=("availability", "state"),
        fallback="unavailable",
    )
    explainability_state = _context_state(
        explainability_context,
        path=("availability", "state"),
        fallback="unavailable",
    )
    states = [source_state, advisory_state]
    if validation_errors:
        state = "blocked"
    elif "blocked" in states:
        state = "blocked"
    elif "descriptive_only" in states:
        state = "descriptive_only"
    elif "eligible" in states:
        state = "eligible"
    else:
        state = "descriptive_only"

    blocking_reasons = []
    blocking_reasons.extend(validation_errors)
    blocking_reasons.extend(_sdo_blockers(sdo))
    if advisory_context is not None:
        blocking_reasons.extend(
            _string_list(
                _nested_get(advisory_context, ("advisory_eligibility", "blocking_reasons"))
            )
        )
    if portfolio_state in {"blocked"}:
        blocking_reasons.append("portfolio_intelligence_context.blocked")
    if explainability_state in {"blocked"}:
        blocking_reasons.append("explainability_change_rationale_context.blocked")

    return {
        "state": state,
        "source_state": source_state,
        "advisory_context_state": advisory_state,
        "portfolio_specific_context_state": portfolio_state,
        "change_rationale_context_state": explainability_state,
        "scope": _allowed_scope(advisory_context, state),
        "blocking_reasons": _stable_unique(blocking_reasons),
        "no_upgrade_from_upstream": True,
        "portfolio_specific_advisory_available": portfolio_state in {"available", "partial"},
        "change_rationale_available": explainability_state in {
            "available",
            "partial",
            "not_comparable",
        },
    }


def _sdo_eligibility_state(sdo: Mapping[str, Any]) -> str:
    validation = _mapping_or_none(sdo.get("validation")) or {}
    if validation.get("contract_status") not in {None, "valid"}:
        return "blocked"
    data_coverage = _mapping_or_none(sdo.get("data_coverage")) or {}
    coverage_status = data_coverage.get("coverage_status")
    if coverage_status == "blocked":
        return "blocked"
    if coverage_status == "descriptive_only":
        return "descriptive_only"
    decision = _mapping_or_none(sdo.get("decision")) or {}
    if decision.get("action") == "blocked":
        return "blocked"
    if decision.get("is_actionable") is True and coverage_status == "ready":
        return "eligible"
    return "descriptive_only"


def _sdo_blockers(sdo: Mapping[str, Any]) -> list[str]:
    blockers = []
    data_coverage = _mapping_or_none(sdo.get("data_coverage")) or {}
    blocked_reason = _text_or_none(data_coverage.get("blocked_reason"))
    if blocked_reason:
        blockers.append(blocked_reason)
    decision = _mapping_or_none(sdo.get("decision")) or {}
    blockers.extend(_string_list(decision.get("actionability_blockers")))
    explainability = _mapping_or_none(sdo.get("explainability")) or {}
    blockers.extend(_string_list(explainability.get("blocking_reasons")))
    validation = _mapping_or_none(sdo.get("validation")) or {}
    fail_closed_reason = _text_or_none(validation.get("fail_closed_reason"))
    if fail_closed_reason:
        blockers.append(fail_closed_reason)
    return blockers


def _allowed_scope(advisory_context: Mapping[str, Any] | None, state: str) -> list[str]:
    if advisory_context is not None:
        scope = _string_list(
            _nested_get(advisory_context, ("advisory_eligibility", "allowed_scope"))
        )
        if scope:
            return sorted(scope)
    if state == "eligible":
        return ["artifact_grounded_questions", "explain_decision_state"]
    if state == "descriptive_only":
        return ["describe_current_state", "explain_blockers"]
    return ["disclose_blockers"]


def _composition_state(state: str, validation_errors: list[str]) -> str:
    if validation_errors or state == "blocked":
        return "blocked_artifact_produced"
    if state == "eligible":
        return "eligible_artifact_produced"
    return "descriptive_only_artifact_produced"


def _source_artifact_references(contexts: Mapping[str, Mapping[str, Any] | None]) -> list[str]:
    refs: list[str] = []
    for family, context in contexts.items():
        if context is None:
            continue
        refs.append(f"{family}:{context.get('schema_version') or 'unversioned'}")
        refs.extend(_string_list(context.get("source_artifact_refs")))
        refs.extend(_string_list(context.get("artifact_refs")))
        evidence = _mapping_or_none(context.get("evidence")) or {}
        refs.extend(_string_list(evidence.get("artifact_refs")))
        provenance = _mapping_or_none(context.get("provenance")) or {}
        refs.extend(_string_list(provenance.get("artifact_refs")))
        provenance_context = _mapping_or_none(context.get("provenance_context")) or {}
        refs.extend(_string_list(provenance_context.get("artifact_refs")))
    return _stable_unique(refs)


def _context_run_ids(contexts: Mapping[str, Mapping[str, Any] | None]) -> dict[str, str | None]:
    return {
        name: _text_or_none(context.get("run_id")) if context is not None else None
        for name, context in sorted(contexts.items())
    }


def _context_payload_or_absent(
    context: Mapping[str, Any] | None,
    *,
    family: str,
    absent_state: str,
    absent_reason: str,
) -> dict[str, Any]:
    if context is not None:
        return {
            "include_mode": "embedded_preserved_context",
            "availability_state": _context_availability_state(context),
            "semantic_override_allowed": False,
            "payload": context,
        }
    return {
        "include_mode": "absent",
        "availability_state": absent_state,
        "missing_reason": absent_reason,
        "semantic_override_allowed": False,
        "payload": None,
        "family": family,
    }


def _context_availability_state(context: Mapping[str, Any]) -> str:
    return (
        _text_or_none(_nested_get(context, ("availability", "state")))
        or _text_or_none(_nested_get(context, ("advisory_eligibility", "state")))
        or "provided"
    )


def _present_context_families(contexts: Mapping[str, Mapping[str, Any] | None]) -> list[str]:
    return sorted(name for name, context in contexts.items() if context is not None)


def _freshness_context(
    sdo: Mapping[str, Any],
    contexts: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    family_freshness: list[dict[str, Any]] = []
    data_coverage = _mapping_or_none(sdo.get("data_coverage")) or {}
    family_freshness.append(
        {
            "family": "structured_decision_output",
            "status": data_coverage.get("freshness_status") or "unknown",
            "generated_at": sdo.get("generated_at"),
        }
    )
    for family, context in sorted(contexts.items()):
        if context is None:
            continue
        status = (
            _text_or_none(_nested_get(context, ("freshness_context", "global_freshness_status")))
            or _text_or_none(_nested_get(context, ("freshness", "portfolio_review_freshness")))
            or "unknown"
        )
        family_freshness.append(
            {
                "family": family,
                "status": status,
                "generated_at": context.get("generated_at"),
            }
        )
    statuses = {entry["status"] for entry in family_freshness}
    if "blocked" in statuses:
        global_status = "blocked"
    elif len(statuses) > 1:
        global_status = "mixed"
    elif statuses:
        global_status = sorted(statuses)[0]
    else:
        global_status = "unknown"
    return {
        "global_freshness_status": global_status,
        "family_freshness": family_freshness,
        "generated_at_is_not_upstream_freshness": True,
    }


def _uncertainty_context(
    *,
    sdo: Mapping[str, Any],
    advisory_context: Mapping[str, Any] | None,
    missing_context: list[dict[str, Any]],
    blockers: list[str],
) -> dict[str, Any]:
    upstream_uncertainty = (
        _mapping_or_none(advisory_context.get("uncertainty_context"))
        if advisory_context is not None
        else None
    )
    if upstream_uncertainty:
        confidence = upstream_uncertainty.get("confidence")
        uncertainty_level = upstream_uncertainty.get("uncertainty_level")
        missing_evidence = _string_list(upstream_uncertainty.get("missing_evidence"))
        limitations = _string_list(upstream_uncertainty.get("limitations"))
    else:
        confidence_slot = _mapping_or_none(
            (_mapping_or_none(sdo.get("scores")) or {}).get("confidence")
        ) or {}
        confidence = confidence_slot.get("value")
        uncertainty_level = "unknown"
        missing_evidence = _string_list(
            (_mapping_or_none(sdo.get("data_coverage")) or {}).get("missing_families")
        )
        limitations = ["advisory_context_absent"]
    return {
        "confidence": confidence,
        "uncertainty_level": uncertainty_level,
        "missing_evidence": _stable_unique(missing_evidence),
        "missing_context_families": sorted(entry["family"] for entry in missing_context),
        "unresolved_blockers": _stable_unique(blockers),
        "limitations": _stable_unique(limitations),
    }


def _missing_context_entry(family: str, state: str, note: str) -> dict[str, Any]:
    return {
        "family": family,
        "state": state,
        "note": note,
    }


def _context_state(
    context: Mapping[str, Any] | None,
    *,
    path: tuple[str, ...],
    fallback: str,
) -> str:
    if context is None:
        return fallback
    return _text_or_none(_nested_get(context, path)) or fallback


def _nested_get(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return sorted(str(item) for item in value if isinstance(item, str) and item)


def _stable_unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChatGPTReadyAdvisoryArtifactError(
            f"Input artifact JSON is malformed: {path}"
        ) from exc
    except OSError as exc:
        raise ChatGPTReadyAdvisoryArtifactError(
            f"Input artifact is not readable: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ChatGPTReadyAdvisoryArtifactError(
            f"Input artifact must be a JSON object: {path}"
        )
    return _json_ready(payload, path=path.as_posix())


def _validated_advisory_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    payload = _required_mapping(artifact, "artifact")
    identity = _mapping_or_none(payload.get("contract_identity")) or {}
    if identity.get("schema_version") != CHATGPT_READY_ADVISORY_ARTIFACT_SCHEMA_VERSION:
        raise ChatGPTReadyAdvisoryArtifactError(
            "ChatGPT-ready advisory artifact uses an unsupported schema version."
        )
    if identity.get("artifact_type") != CHATGPT_READY_ADVISORY_ARTIFACT_TYPE:
        raise ChatGPTReadyAdvisoryArtifactError(
            "ChatGPT-ready advisory artifact uses an unsupported artifact type."
        )
    for path, label in (
        (("run_identity", "run_id"), "run_id"),
        (("instrument_identity", "ticker"), "ticker"),
        (("generated_at",), "generated_at"),
    ):
        if not _text_or_none(_nested_get(payload, path)):
            raise ChatGPTReadyAdvisoryArtifactError(f"Artifact {label} is required.")
    return payload


def _json_ready(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        ready: dict[str, Any] = {}
        for key, nested_value in value.items():
            if isinstance(key, Enum):
                key = str(key.value)
            else:
                key = str(key)
            ready[key] = _json_ready(nested_value, path=f"{path}.{key}")
        return ready
    if isinstance(value, (list, tuple)):
        return [
            _json_ready(nested_value, path=f"{path}[{index}]")
            for index, nested_value in enumerate(value)
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ChatGPTReadyAdvisoryArtifactError(
        f"Artifact value is not JSON serializable at {path}."
    )


def _write_json(path: Path, payload: Mapping[str, Any], *, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise ChatGPTReadyAdvisoryArtifactError(
            f"ChatGPT-ready advisory artifact already exists: {path}"
        )
    try:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except TypeError as exc:
        raise ChatGPTReadyAdvisoryArtifactError(
            f"ChatGPT-ready advisory artifact could not be serialized: {path}"
        ) from exc
    except OSError as exc:
        raise ChatGPTReadyAdvisoryArtifactError(
            f"ChatGPT-ready advisory artifact could not be written: {path}"
        ) from exc


def _safe_path_segment(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ChatGPTReadyAdvisoryArtifactError(
            f"{field_name} must be a non-empty string."
        )
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ChatGPTReadyAdvisoryArtifactError(
            f"{field_name} is not a safe path segment."
        )
    if not _SAFE_PATH_SEGMENT_RE.fullmatch(value):
        raise ChatGPTReadyAdvisoryArtifactError(
            f"{field_name} contains unsafe characters."
        )
    return value


def _validated_output_root(output_root: Path | str) -> Path:
    root = Path(output_root)
    if not root.parts:
        raise ChatGPTReadyAdvisoryArtifactError(
            "ChatGPT-ready advisory artifact output root is required."
        )
    if any(part == ".." for part in root.parts):
        raise ChatGPTReadyAdvisoryArtifactError(
            "ChatGPT-ready advisory artifact output root may not contain parent traversal."
        )
    return root


def _resolved_child(parent: Path, child: str) -> Path:
    safe_child = _safe_path_segment(child, field_name="path_component")
    candidate = (parent / safe_child).resolve()
    try:
        candidate.relative_to(parent)
    except ValueError as exc:
        raise ChatGPTReadyAdvisoryArtifactError(
            f"Resolved artifact path escapes output root: {candidate}"
        ) from exc
    return candidate


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ChatGPTReadyAdvisoryArtifactError(
            f"Artifact path is outside output root: {path}"
        ) from exc
