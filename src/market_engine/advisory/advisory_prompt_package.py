from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from market_engine.advisory.advisory_artifact_validation import (
    VALIDATION_STATUS_VALID,
    validate_chatgpt_ready_advisory_artifact,
)


PROMPT_PACKAGE_SCHEMA_VERSION = "market-engine-advisory-prompt-package-v1"
PROMPT_PACKAGE_ARTIFACT_TYPE = "market-engine-advisory-prompt-package"
PROMPT_PACKAGE_CONTRACT_NAME = "controlled_advisory_prompt_package"
PROMPT_PACKAGE_CONTRACT_VERSION = "v1"

RESPONSE_CONTRACT_SCHEMA_VERSION = "chatgpt-advisory-prompt-response-grounding-v1"
RESPONSE_CONTRACT_ARTIFACT_TYPE = (
    "market-engine-chatgpt-advisory-prompt-response-grounding-contract"
)

QUESTION_CLASSES = frozenset(
    {
        "current_state_explanation",
        "recommendation_interpretation",
        "portfolio_context_question",
        "change_rationale_question",
        "risk_question",
        "freshness_question",
        "missing_evidence_question",
        "buy_zone_explanation",
        "position_management_explanation",
        "comparative_question",
        "sizing_question",
        "allocation_question",
        "execution_question",
        "unsupported_question",
    }
)

CONTEXT_FAMILIES = frozenset(
    {
        "structured_decision_output",
        "advisory_context",
        "portfolio_intelligence_context",
        "explainability_change_rationale_context",
        "governor_context",
        "dispatch_context",
        "provenance_context",
        "freshness_context",
        "uncertainty_context",
        "blockers",
        "missing_context",
    }
)

QUESTION_CONTEXT_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "current_state_explanation": ("structured_decision_output",),
    "recommendation_interpretation": ("structured_decision_output", "governor_context"),
    "portfolio_context_question": (
        "structured_decision_output",
        "portfolio_intelligence_context",
    ),
    "change_rationale_question": (
        "structured_decision_output",
        "explainability_change_rationale_context",
    ),
    "risk_question": ("structured_decision_output",),
    "freshness_question": ("freshness_context",),
    "missing_evidence_question": ("missing_context",),
    "buy_zone_explanation": ("structured_decision_output", "governor_context"),
    "position_management_explanation": (
        "structured_decision_output",
        "portfolio_intelligence_context",
        "governor_context",
    ),
    "comparative_question": (
        "structured_decision_output",
        "explainability_change_rationale_context",
    ),
    "sizing_question": ("portfolio_intelligence_context",),
    "allocation_question": ("portfolio_intelligence_context",),
    "execution_question": ("structured_decision_output",),
    "unsupported_question": (),
}

QUESTION_PERMITTED_USE_CASES: dict[str, str] = {
    "current_state_explanation": "bounded_current_state_interpretation",
    "recommendation_interpretation": "bounded_recommendation_state_interpretation",
    "portfolio_context_question": "portfolio_context_interpretation_when_available",
    "change_rationale_question": "bounded_change_rationale_interpretation",
    "risk_question": "bounded_risk_context_interpretation",
    "freshness_question": "freshness_description",
    "missing_evidence_question": "missing_evidence_description",
    "buy_zone_explanation": "bounded_governor_level_explanation",
    "position_management_explanation": "bounded_position_management_explanation",
    "comparative_question": "bounded_comparison_when_comparable_context_exists",
    "sizing_question": "authority_refusal_without_sizing_contract",
    "allocation_question": "authority_refusal_without_allocation_contract",
    "execution_question": "authority_refusal_without_execution_contract",
    "unsupported_question": "unable_to_determine",
}

FORBIDDEN_INFERENCES = (
    "invented_fact",
    "unsupported_causal_claim",
    "unsupported_materiality_claim",
    "unsupported_probability",
    "unsupported_price_claim",
    "unsupported_portfolio_claim",
    "unsupported_sizing_claim",
    "unsupported_allocation_claim",
    "unsupported_execution_claim",
    "authority_override",
    "recommendation_remapping",
    "semantic_override",
)


class AdvisoryPromptPackageError(ValueError):
    """Raised when a controlled advisory prompt package cannot be built."""


@dataclass(frozen=True)
class AdvisoryPromptPackageIssue:
    code: str
    path: str
    message: str
    severity: str = "error"
    expected: Any = None
    actual: Any = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
        }
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.actual is not None:
            payload["actual"] = self.actual
        return payload


@dataclass(frozen=True)
class AdvisoryPromptPackageValidationResult:
    valid: bool
    issues: tuple[AdvisoryPromptPackageIssue, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": "valid" if self.valid else "invalid",
            "issue_count": len(self.issues),
            "issues": [issue.to_payload() for issue in self.issues],
        }


def build_advisory_prompt_package(
    *,
    advisory_artifact: Mapping[str, Any],
    question: str,
    question_class: str,
    package_id: str,
) -> dict[str, Any]:
    source_validation = validate_chatgpt_ready_advisory_artifact(advisory_artifact)
    if not source_validation.valid:
        raise AdvisoryPromptPackageError(
            "Source advisory artifact is not CI06-valid: "
            + ", ".join(issue.code for issue in source_validation.issues)
        )
    contract_validation = _contract_validation(advisory_artifact)
    if contract_validation.get("validation_status") not in {
        VALIDATION_STATUS_VALID,
        None,
    }:
        raise AdvisoryPromptPackageError(
            "Source advisory artifact persisted validation evidence is invalid."
        )
    if not isinstance(question, str) or not question.strip():
        raise AdvisoryPromptPackageError("question is required.")
    if question_class not in QUESTION_CLASSES:
        raise AdvisoryPromptPackageError(f"Unsupported question class: {question_class}")
    if not isinstance(package_id, str) or not package_id:
        raise AdvisoryPromptPackageError("package_id is required.")

    run_identity = _object(advisory_artifact.get("run_identity"))
    instrument_identity = _object(advisory_artifact.get("instrument_identity"))
    selected_context = _selected_context(advisory_artifact, question_class)
    mandatory_disclosures = _mandatory_disclosures(
        advisory_artifact=advisory_artifact,
        selected_context=selected_context,
        question_class=question_class,
    )
    missing_families = tuple(
        family
        for family in QUESTION_CONTEXT_REQUIREMENTS[question_class]
        if _context_availability(selected_context, family) == "absent"
    )
    package = {
        "prompt_package_identity": {
            "schema_version": PROMPT_PACKAGE_SCHEMA_VERSION,
            "artifact_type": PROMPT_PACKAGE_ARTIFACT_TYPE,
            "contract_name": PROMPT_PACKAGE_CONTRACT_NAME,
            "contract_version": PROMPT_PACKAGE_CONTRACT_VERSION,
            "package_id": package_id,
            "non_production": True,
            "model_free": True,
        },
        "source_artifact_identity": {
            "schema_version": _nested_text(
                advisory_artifact, ("contract_identity", "schema_version")
            ),
            "artifact_type": _nested_text(
                advisory_artifact, ("contract_identity", "artifact_type")
            ),
            "run_id": run_identity.get("run_id"),
            "validation_status": VALIDATION_STATUS_VALID,
            "validation_issue_count": 0,
        },
        "instrument_identity": {
            "ticker": instrument_identity.get("ticker"),
            "instrument": instrument_identity.get("instrument"),
        },
        "question": question,
        "question_classification": {
            "question_class": question_class,
            "required_context_families": list(
                QUESTION_CONTEXT_REQUIREMENTS[question_class]
            ),
            "missing_required_context_families": list(missing_families),
        },
        "permitted_use_case": QUESTION_PERMITTED_USE_CASES[question_class],
        "selected_context": selected_context,
        "mandatory_disclosures": mandatory_disclosures,
        "forbidden_inferences": list(FORBIDDEN_INFERENCES),
        "required_response_contract": {
            "schema_version": RESPONSE_CONTRACT_SCHEMA_VERSION,
            "artifact_type": RESPONSE_CONTRACT_ARTIFACT_TYPE,
            "response_schema_version": "chatgpt-advisory-response-grounding-v1",
        },
        "grounding_requirements": {
            "material_claims_require_evidence": True,
            "evidence_references_required_shape": [
                "claim_id",
                "claim_type",
                "source_context_family",
                "artifact_ref",
                "run_id",
                "path",
                "support_type",
            ],
            "path_resolution": "restricted_deterministic_json_path",
        },
        "authority_boundary": _authority_boundary(question_class),
    }
    validation = validate_advisory_prompt_package(package)
    if not validation.valid:
        raise AdvisoryPromptPackageError(
            "Prompt package validation failed: "
            + ", ".join(issue.code for issue in validation.issues)
        )
    return package


def validate_advisory_prompt_package(
    package: Mapping[str, Any],
) -> AdvisoryPromptPackageValidationResult:
    issues: list[AdvisoryPromptPackageIssue] = []
    if not isinstance(package, Mapping):
        return AdvisoryPromptPackageValidationResult(
            False,
            (
                AdvisoryPromptPackageIssue(
                    "invalid_field_type",
                    "$",
                    "Prompt package must be a JSON object.",
                    expected="object",
                    actual=type(package).__name__,
                ),
            ),
        )
    for field_name in (
        "prompt_package_identity",
        "source_artifact_identity",
        "instrument_identity",
        "question",
        "question_classification",
        "permitted_use_case",
        "selected_context",
        "mandatory_disclosures",
        "forbidden_inferences",
        "required_response_contract",
        "grounding_requirements",
        "authority_boundary",
    ):
        if field_name not in package:
            issues.append(
                AdvisoryPromptPackageIssue(
                    "missing_required_field",
                    f"$.{field_name}",
                    f"Required prompt package field is missing: {field_name}.",
                )
            )
    identity = _object(package.get("prompt_package_identity"))
    _expect(identity.get("schema_version"), PROMPT_PACKAGE_SCHEMA_VERSION, "$.prompt_package_identity.schema_version", issues)
    _expect(identity.get("artifact_type"), PROMPT_PACKAGE_ARTIFACT_TYPE, "$.prompt_package_identity.artifact_type", issues)
    source = _object(package.get("source_artifact_identity"))
    if source.get("validation_status") != VALIDATION_STATUS_VALID:
        issues.append(
            AdvisoryPromptPackageIssue(
                "invalid_source_validation_status",
                "$.source_artifact_identity.validation_status",
                "Prompt package requires CI06-valid source artifact evidence.",
                expected=VALIDATION_STATUS_VALID,
                actual=source.get("validation_status"),
            )
        )
    if source.get("validation_issue_count") != 0:
        issues.append(
            AdvisoryPromptPackageIssue(
                "invalid_source_validation_issue_count",
                "$.source_artifact_identity.validation_issue_count",
                "Prompt package source artifact must have zero validation issues.",
                expected=0,
                actual=source.get("validation_issue_count"),
            )
        )
    classification = _object(package.get("question_classification"))
    question_class = classification.get("question_class")
    if question_class not in QUESTION_CLASSES:
        issues.append(
            AdvisoryPromptPackageIssue(
                "invalid_question_class",
                "$.question_classification.question_class",
                "Question class is not approved by ME-CI07.",
                expected=sorted(QUESTION_CLASSES),
                actual=question_class,
            )
        )
    if not isinstance(package.get("mandatory_disclosures"), list):
        issues.append(
            AdvisoryPromptPackageIssue(
                "invalid_field_type",
                "$.mandatory_disclosures",
                "mandatory_disclosures must be a list.",
                expected="array",
                actual=type(package.get("mandatory_disclosures")).__name__,
            )
        )
    if not isinstance(package.get("forbidden_inferences"), list):
        issues.append(
            AdvisoryPromptPackageIssue(
                "invalid_field_type",
                "$.forbidden_inferences",
                "forbidden_inferences must be a list.",
                expected="array",
                actual=type(package.get("forbidden_inferences")).__name__,
            )
        )
    response_contract = _object(package.get("required_response_contract"))
    _expect(
        response_contract.get("schema_version"),
        RESPONSE_CONTRACT_SCHEMA_VERSION,
        "$.required_response_contract.schema_version",
        issues,
    )
    authority = _object(package.get("authority_boundary"))
    for field_name in (
        "allocation_authority",
        "position_sizing_authority",
        "execution_authority",
        "broker_authority",
        "portfolio_write_authority",
        "watchlist_write_authority",
    ):
        if authority.get(field_name) is not False:
            issues.append(
                AdvisoryPromptPackageIssue(
                    "invalid_authority_boundary",
                    f"$.authority_boundary.{field_name}",
                    "ME-CI08 prompt packages must not grant downstream authority.",
                    expected=False,
                    actual=authority.get(field_name),
                )
            )
    return AdvisoryPromptPackageValidationResult(
        valid=not issues,
        issues=tuple(sorted(issues, key=lambda issue: (issue.path, issue.code))),
    )


def _selected_context(
    advisory_artifact: Mapping[str, Any],
    question_class: str,
) -> dict[str, Any]:
    always = {
        "contract_identity": advisory_artifact.get("contract_identity"),
        "artifact_identity": advisory_artifact.get("artifact_identity"),
        "validation_evidence": _contract_validation(advisory_artifact),
        "run_identity": advisory_artifact.get("run_identity"),
        "advisory_eligibility": advisory_artifact.get("advisory_eligibility"),
        "structured_decision_output": advisory_artifact.get("structured_decision_context"),
        "blockers": advisory_artifact.get("blockers"),
        "missing_context": advisory_artifact.get("missing_context"),
        "freshness_context": advisory_artifact.get("freshness_context"),
        "uncertainty_context": advisory_artifact.get("uncertainty_context"),
        "provenance_context": advisory_artifact.get("provenance_context"),
    }
    selected: dict[str, Any] = dict(always)
    for family in QUESTION_CONTEXT_REQUIREMENTS[question_class]:
        if family == "portfolio_intelligence_context":
            selected[family] = advisory_artifact.get("portfolio_intelligence_context")
        elif family == "explainability_change_rationale_context":
            selected[family] = advisory_artifact.get(
                "explainability_change_rationale_context"
            )
        elif family == "governor_context":
            selected[family] = advisory_artifact.get("governor_context")
        elif family == "dispatch_context":
            selected[family] = advisory_artifact.get("dispatch_context")
    if question_class in {"recommendation_interpretation", "buy_zone_explanation"}:
        selected.setdefault("governor_context", advisory_artifact.get("governor_context"))
    if question_class in {"current_state_explanation", "recommendation_interpretation"}:
        selected.setdefault("dispatch_context", advisory_artifact.get("dispatch_context"))
    return selected


def _mandatory_disclosures(
    *,
    advisory_artifact: Mapping[str, Any],
    selected_context: Mapping[str, Any],
    question_class: str,
) -> list[str]:
    disclosures: set[str] = set()
    eligibility_state = _nested_text(advisory_artifact, ("advisory_eligibility", "state"))
    if eligibility_state == "descriptive_only":
        disclosures.add("descriptive_only_disclosure")
    if question_class in {
        "portfolio_context_question",
        "position_management_explanation",
        "sizing_question",
        "allocation_question",
    } and _context_availability(selected_context, "portfolio_intelligence_context") == "absent":
        disclosures.add("missing_portfolio_disclosure")
    freshness = _object(selected_context.get("freshness_context"))
    global_freshness = freshness.get("global_freshness_status")
    if global_freshness == "stale":
        disclosures.add("staleness_disclosure")
    if global_freshness == "unknown":
        disclosures.add("unknown_freshness_disclosure")
    family_freshness = freshness.get("family_freshness")
    if isinstance(family_freshness, list):
        relevant_families = set(QUESTION_CONTEXT_REQUIREMENTS[question_class])
        statuses = set()
        for item in family_freshness:
            if not isinstance(item, Mapping):
                continue
            family = item.get("family")
            status = item.get("status")
            if family in relevant_families and status in {"stale", "unknown"}:
                statuses.add(status)
        if "stale" in statuses:
            disclosures.add("staleness_disclosure")
        if "unknown" in statuses:
            disclosures.add("unknown_freshness_disclosure")
    if question_class in {"sizing_question", "allocation_question", "execution_question"}:
        disclosures.add("authority_disclosure")
    return sorted(disclosures)


def _context_availability(selected_context: Mapping[str, Any], family: str) -> str:
    value = selected_context.get(family)
    if not isinstance(value, Mapping):
        return "absent"
    if value.get("include_mode") == "absent" or value.get("payload") is None:
        return "absent"
    return "available"


def _authority_boundary(question_class: str) -> dict[str, Any]:
    return {
        "allocation_authority": False,
        "position_sizing_authority": False,
        "execution_authority": False,
        "broker_authority": False,
        "portfolio_write_authority": False,
        "watchlist_write_authority": False,
        "question_class_requires_refusal": question_class
        in {"sizing_question", "allocation_question", "execution_question"},
    }


def _contract_validation(advisory_artifact: Mapping[str, Any]) -> dict[str, Any]:
    validation_summary = _object(advisory_artifact.get("validation_summary"))
    contract_validation = validation_summary.get("contract_validation")
    return dict(contract_validation) if isinstance(contract_validation, Mapping) else {}


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested_text(payload: Mapping[str, Any], path: tuple[str, ...]) -> str | None:
    current: Any = payload
    for segment in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current if isinstance(current, str) else None


def _expect(
    actual: Any,
    expected: Any,
    path: str,
    issues: list[AdvisoryPromptPackageIssue],
) -> None:
    if actual != expected:
        issues.append(
            AdvisoryPromptPackageIssue(
                "invalid_enum_value",
                path,
                "Prompt package field has an unsupported value.",
                expected=expected,
                actual=actual,
            )
        )
