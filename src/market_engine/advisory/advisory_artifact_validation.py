from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

ADVISORY_ARTIFACT_VALIDATOR_VERSION = (
    "market-engine-chatgpt-ready-advisory-artifact-validator-v1"
)
CHATGPT_READY_ADVISORY_ARTIFACT_SCHEMA_VERSION = (
    "market-engine-chatgpt-ready-advisory-artifact-v1"
)
CHATGPT_READY_ADVISORY_ARTIFACT_TYPE = (
    "market-engine-chatgpt-ready-advisory-artifact"
)
STRUCTURED_DECISION_OUTPUT_SCHEMA_VERSION = "structured-decision-output-v1"
STRUCTURED_DECISION_OUTPUT_ARTIFACT_TYPE = "market-engine-structured-decision-output"
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

VALIDATION_STATUS_VALID = "valid"
VALIDATION_STATUS_INVALID = "invalid"

_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_TOP_LEVEL_REQUIRED_FIELDS = (
    "contract_identity",
    "artifact_identity",
    "run_identity",
    "instrument_identity",
    "generated_at",
    "source_artifact_references",
    "composition_status",
    "advisory_eligibility",
    "structured_decision_context",
    "portfolio_intelligence_context",
    "explainability_change_rationale_context",
    "governor_context",
    "dispatch_context",
    "provenance_context",
    "freshness_context",
    "uncertainty_context",
    "blockers",
    "missing_context",
    "validation_summary",
)

_COMPOSITION_STATES = {
    "blocked_artifact_produced",
    "descriptive_only_artifact_produced",
    "eligible_artifact_produced",
}
_ADVISORY_STATES = {"eligible", "descriptive_only", "blocked"}
_CONTEXT_INCLUDE_MODES = {
    "embedded_canonical_context",
    "embedded_preserved_context",
    "referenced_context",
    "absent",
}
_FRESHNESS_STATUSES = {"fresh", "mixed", "stale", "unknown", "blocked"}
_PORTFOLIO_AVAILABILITY_STATES = {"available", "partial", "unavailable", "blocked"}
_EXPLAINABILITY_AVAILABILITY_STATES = {
    "available",
    "partial",
    "unavailable",
    "blocked",
    "not_comparable",
}
_ALLOWED_OPTIONAL_ABSENT_FAMILIES = {
    "portfolio_intelligence_context",
    "explainability_change_rationale_context",
    "governor_context",
    "dispatch_context",
}

_FORBIDDEN_AUTHORITY_FIELDS = {
    "allocation",
    "target_weight",
    "position_size",
    "position_sizing",
    "order_quantity",
    "broker_action",
    "execution_instruction",
    "fabricated_target_price",
    "fabricated_stop_loss",
    "fabricated_probability",
    "fabricated_conviction",
    "fabricated_urgency",
    "fabricated_tradeability",
    "tradeability",
    "urgency",
}


@dataclass(frozen=True)
class AdvisoryArtifactValidationIssue:
    code: str
    path: str
    message: str
    severity: str = "error"
    contract_family: str | None = None
    expected: Any = None
    actual: Any = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
        }
        if self.contract_family is not None:
            payload["contract_family"] = self.contract_family
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.actual is not None:
            payload["actual"] = self.actual
        return payload


@dataclass(frozen=True)
class AdvisoryArtifactValidationResult:
    status: str
    validator_version: str
    validated_schema_version: str | None
    issues: tuple[AdvisoryArtifactValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return self.status == VALIDATION_STATUS_VALID

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "validator_version": self.validator_version,
            "validated_schema_version": self.validated_schema_version,
            "issue_count": len(self.issues),
            "issues": [issue.to_payload() for issue in self.issues],
        }


def validate_chatgpt_ready_advisory_artifact(
    artifact: Mapping[str, Any],
) -> AdvisoryArtifactValidationResult:
    issues: list[AdvisoryArtifactValidationIssue] = []
    if not isinstance(artifact, Mapping):
        return _result(
            None,
            [
                _issue(
                    "invalid_field_type",
                    "$",
                    "Advisory artifact must be a JSON object.",
                    expected="object",
                    actual=type(artifact).__name__,
                )
            ],
        )

    payload = artifact
    _validate_required_top_level(payload, issues)
    _validate_contract_identity(payload, issues)
    _validate_artifact_identity(payload, issues)
    _validate_run_identity(payload, issues)
    _validate_instrument_identity(payload, issues)
    _validate_timestamp(payload.get("generated_at"), "$.generated_at", issues)
    _validate_string_list(
        payload.get("source_artifact_references"),
        "$.source_artifact_references",
        issues,
    )
    _validate_composition_status(payload, issues)
    _validate_advisory_eligibility(payload, issues)
    _validate_structured_decision_context(payload, issues)
    _validate_optional_contexts(payload, issues)
    _validate_provenance_context(payload, issues)
    _validate_freshness_context(payload, issues)
    _validate_uncertainty_context(payload, issues)
    _validate_string_list(payload.get("blockers"), "$.blockers", issues)
    _validate_missing_context(payload, issues)
    _validate_validation_summary(payload, issues)
    _validate_cross_context_consistency(payload, issues)
    _validate_contextual_forbidden_fields(payload, issues)

    schema_version = _nested_text(payload, ("contract_identity", "schema_version"))
    return _result(schema_version, issues)


def validation_evidence_payload(
    result: AdvisoryArtifactValidationResult,
    *,
    validated_at: str,
) -> dict[str, Any]:
    return {
        "validation_status": result.status,
        "validator_version": result.validator_version,
        "validated_schema_version": result.validated_schema_version,
        "validation_timestamp": validated_at,
        "issue_count": len(result.issues),
        "issues": [issue.to_payload() for issue in result.issues],
    }


def _validate_required_top_level(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    for field_name in _TOP_LEVEL_REQUIRED_FIELDS:
        if field_name not in payload:
            issues.append(
                _issue(
                    "missing_required_field",
                    f"$.{field_name}",
                    f"Required advisory artifact field is missing: {field_name}.",
                )
            )


def _validate_contract_identity(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    identity = _require_object(
        payload.get("contract_identity"),
        "$.contract_identity",
        issues,
        "chatgpt_ready_advisory_artifact",
    )
    if identity is None:
        return
    _expect_text(
        identity,
        "schema_version",
        "$.contract_identity.schema_version",
        issues,
        expected=CHATGPT_READY_ADVISORY_ARTIFACT_SCHEMA_VERSION,
        mismatch_code="unsupported_schema_version",
        contract_family="chatgpt_ready_advisory_artifact",
    )
    _expect_text(
        identity,
        "artifact_type",
        "$.contract_identity.artifact_type",
        issues,
        expected=CHATGPT_READY_ADVISORY_ARTIFACT_TYPE,
        mismatch_code="artifact_type_mismatch",
        contract_family="chatgpt_ready_advisory_artifact",
    )
    _expect_text(
        identity,
        "contract_name",
        "$.contract_identity.contract_name",
        issues,
        expected="chatgpt_ready_advisory_artifact",
        mismatch_code="unsupported_contract_version",
        contract_family="chatgpt_ready_advisory_artifact",
    )
    _expect_text(
        identity,
        "contract_version",
        "$.contract_identity.contract_version",
        issues,
        expected="v1",
        mismatch_code="unsupported_contract_version",
        contract_family="chatgpt_ready_advisory_artifact",
    )


def _validate_artifact_identity(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    identity = _require_object(
        payload.get("artifact_identity"),
        "$.artifact_identity",
        issues,
        "chatgpt_ready_advisory_artifact",
    )
    if identity is None:
        return
    _expect_text(
        identity,
        "schema_version",
        "$.artifact_identity.schema_version",
        issues,
        expected=CHATGPT_READY_ADVISORY_ARTIFACT_SCHEMA_VERSION,
        mismatch_code="unsupported_schema_version",
    )
    _expect_text(
        identity,
        "artifact_type",
        "$.artifact_identity.artifact_type",
        issues,
        expected=CHATGPT_READY_ADVISORY_ARTIFACT_TYPE,
        mismatch_code="artifact_type_mismatch",
    )
    if identity.get("non_production_artifact") is not True:
        issues.append(
            _issue(
                "invalid_enum_value",
                "$.artifact_identity.non_production_artifact",
                "ME-CI06 only validates local non-production advisory artifacts.",
                expected=True,
                actual=identity.get("non_production_artifact"),
            )
        )


def _validate_run_identity(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    run_identity = _require_object(payload.get("run_identity"), "$.run_identity", issues)
    if run_identity is None:
        return
    _require_non_empty_text(run_identity.get("run_id"), "$.run_identity.run_id", issues)
    _require_non_empty_text(
        run_identity.get("source_structured_decision_run_id"),
        "$.run_identity.source_structured_decision_run_id",
        issues,
    )
    context_run_ids = _require_object(
        run_identity.get("context_run_ids"),
        "$.run_identity.context_run_ids",
        issues,
    )
    if context_run_ids is not None:
        for family, run_id in sorted(context_run_ids.items()):
            if run_id is not None and not isinstance(run_id, str):
                issues.append(
                    _issue(
                        "invalid_field_type",
                        f"$.run_identity.context_run_ids.{family}",
                        "Context run id must be a string or null.",
                        expected="string|null",
                        actual=type(run_id).__name__,
                    )
                )


def _validate_instrument_identity(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    identity = _require_object(
        payload.get("instrument_identity"),
        "$.instrument_identity",
        issues,
    )
    if identity is None:
        return
    ticker = identity.get("ticker")
    _require_non_empty_text(ticker, "$.instrument_identity.ticker", issues)
    instrument = _require_object(
        identity.get("instrument"),
        "$.instrument_identity.instrument",
        issues,
    )
    if instrument is not None and isinstance(ticker, str):
        instrument_ticker = instrument.get("ticker")
        if instrument_ticker != ticker:
            issues.append(
                _issue(
                    "instrument_identity_mismatch",
                    "$.instrument_identity.instrument.ticker",
                    "Instrument ticker must match parent instrument identity ticker.",
                    expected=ticker,
                    actual=instrument_ticker,
                )
            )


def _validate_composition_status(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    status = _require_object(
        payload.get("composition_status"),
        "$.composition_status",
        issues,
    )
    if status is None:
        return
    _require_enum(
        status.get("state"),
        "$.composition_status.state",
        _COMPOSITION_STATES,
        issues,
    )
    if not isinstance(status.get("blocking_validation"), bool):
        issues.append(
            _issue(
                "invalid_field_type",
                "$.composition_status.blocking_validation",
                "blocking_validation must be boolean.",
                expected="boolean",
                actual=type(status.get("blocking_validation")).__name__,
            )
        )
    if status.get("semantic_override_performed") is not False:
        issues.append(
            _issue(
                "forbidden_field_present",
                "$.composition_status.semantic_override_performed",
                "Semantic override is forbidden for ME-CI05/CI06 advisory artifacts.",
                expected=False,
                actual=status.get("semantic_override_performed"),
            )
        )
    _validate_string_list(
        status.get("source_precedence"),
        "$.composition_status.source_precedence",
        issues,
    )


def _validate_advisory_eligibility(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    eligibility = _require_object(
        payload.get("advisory_eligibility"),
        "$.advisory_eligibility",
        issues,
    )
    if eligibility is None:
        return
    _require_enum(
        eligibility.get("state"),
        "$.advisory_eligibility.state",
        _ADVISORY_STATES,
        issues,
    )
    for field_name in (
        "source_state",
        "advisory_context_state",
        "portfolio_specific_context_state",
        "change_rationale_context_state",
    ):
        _require_non_empty_text(
            eligibility.get(field_name),
            f"$.advisory_eligibility.{field_name}",
            issues,
        )
    _validate_string_list(
        eligibility.get("scope"),
        "$.advisory_eligibility.scope",
        issues,
    )
    _validate_string_list(
        eligibility.get("blocking_reasons"),
        "$.advisory_eligibility.blocking_reasons",
        issues,
    )
    if eligibility.get("no_upgrade_from_upstream") is not True:
        issues.append(
            _issue(
                "validation_metadata_invalid",
                "$.advisory_eligibility.no_upgrade_from_upstream",
                "Advisory eligibility must explicitly preserve no-upgrade behavior.",
                expected=True,
                actual=eligibility.get("no_upgrade_from_upstream"),
            )
        )


def _validate_structured_decision_context(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    context = _require_object(
        payload.get("structured_decision_context"),
        "$.structured_decision_context",
        issues,
        "structured_decision_output",
    )
    if context is None:
        return
    _require_enum(
        context.get("include_mode"),
        "$.structured_decision_context.include_mode",
        {"embedded_canonical_context"},
        issues,
    )
    _expect_text(
        context,
        "schema_version",
        "$.structured_decision_context.schema_version",
        issues,
        expected=STRUCTURED_DECISION_OUTPUT_SCHEMA_VERSION,
        mismatch_code="unsupported_schema_version",
        contract_family="structured_decision_output",
    )
    _expect_text(
        context,
        "artifact_type",
        "$.structured_decision_context.artifact_type",
        issues,
        expected=STRUCTURED_DECISION_OUTPUT_ARTIFACT_TYPE,
        mismatch_code="artifact_type_mismatch",
        contract_family="structured_decision_output",
    )
    payload_context = _require_object(
        context.get("payload"),
        "$.structured_decision_context.payload",
        issues,
        "structured_decision_output",
    )
    if payload_context is not None:
        _validate_structured_decision_payload(payload_context, issues)


def _validate_structured_decision_payload(
    sdo: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    required_fields = (
        "schema_version",
        "artifact_type",
        "generated_at",
        "run_id",
        "ticker",
        "instrument",
        "data_coverage",
        "decision",
        "scores",
        "portfolio_context",
        "risk",
        "levels",
        "thesis",
        "evidence",
        "explainability",
        "consumer_guidance",
        "validation",
    )
    for field_name in required_fields:
        if field_name not in sdo:
            issues.append(
                _issue(
                    "missing_required_field",
                    f"$.structured_decision_context.payload.{field_name}",
                    f"Structured Decision Output field is required: {field_name}.",
                    contract_family="structured_decision_output",
                )
            )
    _expect_text(
        sdo,
        "schema_version",
        "$.structured_decision_context.payload.schema_version",
        issues,
        expected=STRUCTURED_DECISION_OUTPUT_SCHEMA_VERSION,
        mismatch_code="unsupported_schema_version",
        contract_family="structured_decision_output",
    )
    _expect_text(
        sdo,
        "artifact_type",
        "$.structured_decision_context.payload.artifact_type",
        issues,
        expected=STRUCTURED_DECISION_OUTPUT_ARTIFACT_TYPE,
        mismatch_code="artifact_type_mismatch",
        contract_family="structured_decision_output",
    )
    _validate_timestamp(
        sdo.get("generated_at"),
        "$.structured_decision_context.payload.generated_at",
        issues,
        allow_null=True,
    )
    _require_non_empty_text(
        sdo.get("run_id"),
        "$.structured_decision_context.payload.run_id",
        issues,
    )
    ticker = sdo.get("ticker")
    _require_non_empty_text(ticker, "$.structured_decision_context.payload.ticker", issues)
    instrument = _require_object(
        sdo.get("instrument"),
        "$.structured_decision_context.payload.instrument",
        issues,
        "structured_decision_output",
    )
    if instrument is not None and isinstance(ticker, str):
        if instrument.get("ticker") != ticker:
            issues.append(
                _issue(
                    "ticker_identity_mismatch",
                    "$.structured_decision_context.payload.instrument.ticker",
                    "Structured Decision Output instrument ticker must match top-level ticker.",
                    expected=ticker,
                    actual=instrument.get("ticker"),
                    contract_family="structured_decision_output",
                )
            )
    data_coverage = _require_object(
        sdo.get("data_coverage"),
        "$.structured_decision_context.payload.data_coverage",
        issues,
        "structured_decision_output",
    )
    if data_coverage is not None:
        _require_enum(
            data_coverage.get("coverage_status"),
            "$.structured_decision_context.payload.data_coverage.coverage_status",
            {"ready", "partial", "descriptive_only", "blocked"},
            issues,
        )
        _require_enum(
            data_coverage.get("freshness_status"),
            "$.structured_decision_context.payload.data_coverage.freshness_status",
            _FRESHNESS_STATUSES,
            issues,
        )
        _validate_string_list(
            data_coverage.get("missing_families"),
            "$.structured_decision_context.payload.data_coverage.missing_families",
            issues,
        )
        _validate_string_list(
            data_coverage.get("stale_families"),
            "$.structured_decision_context.payload.data_coverage.stale_families",
            issues,
        )
    decision = _require_object(
        sdo.get("decision"),
        "$.structured_decision_context.payload.decision",
        issues,
        "structured_decision_output",
    )
    if decision is not None:
        _require_enum(
            decision.get("action"),
            "$.structured_decision_context.payload.decision.action",
            {
                "no_action",
                "watch",
                "hold",
                "buy_candidate",
                "add_candidate",
                "trim_candidate",
                "exit_candidate",
                "blocked",
            },
            issues,
        )
        if not isinstance(decision.get("is_actionable"), bool):
            issues.append(
                _issue(
                    "invalid_field_type",
                    "$.structured_decision_context.payload.decision.is_actionable",
                    "Structured Decision Output is_actionable must be boolean.",
                    expected="boolean",
                    actual=type(decision.get("is_actionable")).__name__,
                )
            )
        _validate_string_list(
            decision.get("actionability_blockers"),
            "$.structured_decision_context.payload.decision.actionability_blockers",
            issues,
        )
    validation = _require_object(
        sdo.get("validation"),
        "$.structured_decision_context.payload.validation",
        issues,
        "structured_decision_output",
    )
    if validation is not None:
        if validation.get("contract_status") not in {"valid", "fail_closed"}:
            issues.append(
                _issue(
                    "validation_metadata_invalid",
                    "$.structured_decision_context.payload.validation.contract_status",
                    "Structured Decision Output validation contract_status is invalid.",
                    expected=["valid", "fail_closed"],
                    actual=validation.get("contract_status"),
                )
            )


def _validate_optional_contexts(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    _validate_optional_context_wrapper(
        payload.get("portfolio_intelligence_context"),
        "$.portfolio_intelligence_context",
        "portfolio_intelligence_context",
        issues,
        embedded_validator=_validate_portfolio_context_payload,
    )
    _validate_optional_context_wrapper(
        payload.get("explainability_change_rationale_context"),
        "$.explainability_change_rationale_context",
        "explainability_change_rationale_context",
        issues,
        embedded_validator=_validate_explainability_context_payload,
    )
    _validate_optional_context_wrapper(
        payload.get("governor_context"),
        "$.governor_context",
        "governor_context",
        issues,
        embedded_validator=_validate_governor_context_payload,
    )
    _validate_optional_context_wrapper(
        payload.get("dispatch_context"),
        "$.dispatch_context",
        "dispatch_context",
        issues,
        embedded_validator=_validate_dispatch_context_payload,
    )


def _validate_optional_context_wrapper(
    value: Any,
    path: str,
    family: str,
    issues: list[AdvisoryArtifactValidationIssue],
    *,
    embedded_validator: Any,
) -> None:
    wrapper = _require_object(value, path, issues, family)
    if wrapper is None:
        return
    mode = wrapper.get("include_mode")
    _require_enum(mode, f"{path}.include_mode", _CONTEXT_INCLUDE_MODES, issues)
    if "semantic_override_allowed" in wrapper and wrapper.get("semantic_override_allowed") is not False:
        issues.append(
            _issue(
                "forbidden_field_present",
                f"{path}.semantic_override_allowed",
                "Context semantic override must be false when present.",
                expected=False,
                actual=wrapper.get("semantic_override_allowed"),
            )
        )
    if mode in {"embedded_preserved_context", "embedded_canonical_context"}:
        embedded_payload = _require_object(wrapper.get("payload"), f"{path}.payload", issues, family)
        if embedded_payload is not None:
            embedded_validator(embedded_payload, f"{path}.payload", issues)
    elif mode == "referenced_context":
        if wrapper.get("payload") is not None:
            issues.append(
                _issue(
                    "embedded_reference_conflict",
                    f"{path}.payload",
                    "Referenced context must not include an embedded payload.",
                    expected=None,
                    actual="payload_present",
                )
            )
        reference = _require_object(wrapper.get("reference"), f"{path}.reference", issues, family)
        if reference is not None:
            _validate_reference_shape(reference, f"{path}.reference", issues)
    elif mode == "absent":
        if family not in _ALLOWED_OPTIONAL_ABSENT_FAMILIES:
            issues.append(
                _issue(
                    "missing_required_context",
                    path,
                    f"Context family is not allowed to be absent: {family}.",
                )
            )
        if wrapper.get("payload") is not None:
            issues.append(
                _issue(
                    "malformed_context",
                    f"{path}.payload",
                    "Absent context must preserve payload as null.",
                    expected=None,
                    actual="payload_present",
                )
            )
        if not isinstance(wrapper.get("missing_reason"), str) or not wrapper.get("missing_reason"):
            issues.append(
                _issue(
                    "missing_required_field",
                    f"{path}.missing_reason",
                    "Absent context must declare a missing reason.",
                )
            )


def _validate_reference_shape(
    reference: Mapping[str, Any],
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    for field_name in ("artifact_ref", "schema_version", "artifact_type"):
        _require_non_empty_text(reference.get(field_name), f"{path}.{field_name}", issues)
    if "run_id" in reference and reference.get("run_id") is not None:
        _require_non_empty_text(reference.get("run_id"), f"{path}.run_id", issues)
    if "ticker" in reference and reference.get("ticker") is not None:
        _require_non_empty_text(reference.get("ticker"), f"{path}.ticker", issues)


def _validate_portfolio_context_payload(
    context: Mapping[str, Any],
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    _expect_text(
        context,
        "schema_version",
        f"{path}.schema_version",
        issues,
        expected=PORTFOLIO_INTELLIGENCE_SCHEMA_VERSION,
        mismatch_code="unsupported_schema_version",
        contract_family="portfolio_intelligence_context",
    )
    _expect_text(
        context,
        "artifact_type",
        f"{path}.artifact_type",
        issues,
        expected=PORTFOLIO_INTELLIGENCE_ARTIFACT_TYPE,
        mismatch_code="artifact_type_mismatch",
        contract_family="portfolio_intelligence_context",
    )
    _validate_timestamp(context.get("generated_at"), f"{path}.generated_at", issues)
    _require_non_empty_text(context.get("run_id"), f"{path}.run_id", issues)
    for field_name in ("portfolio_identity", "portfolio_snapshot_identity", "availability"):
        _require_object(context.get(field_name), f"{path}.{field_name}", issues)
    availability = context.get("availability")
    if isinstance(availability, Mapping):
        _require_enum(
            availability.get("state"),
            f"{path}.availability.state",
            _PORTFOLIO_AVAILABILITY_STATES,
            issues,
        )
    if "source_artifact_refs" in context:
        _validate_string_list(context.get("source_artifact_refs"), f"{path}.source_artifact_refs", issues)
    if "holdings" in context and not isinstance(context.get("holdings"), list):
        issues.append(
            _issue(
                "invalid_field_type",
                f"{path}.holdings",
                "Portfolio holdings must be a list when supplied.",
                expected="array",
                actual=type(context.get("holdings")).__name__,
            )
        )
    cash_context = _require_object(context.get("cash_context"), f"{path}.cash_context", issues)
    if cash_context is not None and cash_context.get("state") == "not_provided":
        if cash_context.get("amount") is not None:
            issues.append(
                _issue(
                    "cross_context_conflict",
                    f"{path}.cash_context.amount",
                    "Unavailable cash must not be converted to a numeric amount.",
                    expected=None,
                    actual=cash_context.get("amount"),
                )
            )


def _validate_explainability_context_payload(
    context: Mapping[str, Any],
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    _expect_text(
        context,
        "schema_version",
        f"{path}.schema_version",
        issues,
        expected=EXPLAINABILITY_SCHEMA_VERSION,
        mismatch_code="unsupported_schema_version",
        contract_family="explainability_change_rationale_context",
    )
    _expect_text(
        context,
        "artifact_type",
        f"{path}.artifact_type",
        issues,
        expected=EXPLAINABILITY_ARTIFACT_TYPE,
        mismatch_code="artifact_type_mismatch",
        contract_family="explainability_change_rationale_context",
    )
    _validate_timestamp(context.get("generated_at"), f"{path}.generated_at", issues)
    _require_non_empty_text(context.get("run_id"), f"{path}.run_id", issues)
    _require_object(context.get("instrument"), f"{path}.instrument", issues)
    _require_object(context.get("current_run_identity"), f"{path}.current_run_identity", issues)
    _require_object(context.get("comparison_window"), f"{path}.comparison_window", issues)
    availability = _require_object(context.get("availability"), f"{path}.availability", issues)
    if availability is not None:
        _require_enum(
            availability.get("state"),
            f"{path}.availability.state",
            _EXPLAINABILITY_AVAILABILITY_STATES,
            issues,
        )
    validation = _require_object(context.get("validation"), f"{path}.validation", issues)
    if validation is not None and not isinstance(validation.get("contract_valid"), bool):
        issues.append(
            _issue(
                "validation_metadata_invalid",
                f"{path}.validation.contract_valid",
                "Explainability validation contract_valid must be boolean.",
                expected="boolean",
                actual=type(validation.get("contract_valid")).__name__,
            )
        )


def _validate_governor_context_payload(
    context: Mapping[str, Any],
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    if "schema_version" in context and not isinstance(context.get("schema_version"), str):
        issues.append(_issue("invalid_field_type", f"{path}.schema_version", "Governor schema_version must be a string."))
    if "artifact_type" in context and not isinstance(context.get("artifact_type"), str):
        issues.append(_issue("invalid_field_type", f"{path}.artifact_type", "Governor artifact_type must be a string."))
    _require_non_empty_text(context.get("run_id"), f"{path}.run_id", issues)
    _require_non_empty_text(context.get("ticker"), f"{path}.ticker", issues)
    if "blockers" in context:
        _validate_string_list(context.get("blockers"), f"{path}.blockers", issues)


def _validate_dispatch_context_payload(
    context: Mapping[str, Any],
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    _require_non_empty_text(
        context.get("report_contract_version") or context.get("schema_version"),
        f"{path}.report_contract_version",
        issues,
    )
    subject = _require_object(context.get("subject"), f"{path}.subject", issues)
    if subject is not None:
        _require_non_empty_text(subject.get("ticker"), f"{path}.subject.ticker", issues)


def _validate_provenance_context(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    context = _require_object(payload.get("provenance_context"), "$.provenance_context", issues)
    if context is None:
        return
    _validate_string_list(context.get("source_artifact_refs"), "$.provenance_context.source_artifact_refs", issues)
    _validate_string_list(context.get("context_families_present"), "$.provenance_context.context_families_present", issues)
    _validate_string_list(context.get("context_families_missing"), "$.provenance_context.context_families_missing", issues)
    if context.get("raw_provider_payload_included") is not False:
        issues.append(
            _issue(
                "invalid_provenance_shape",
                "$.provenance_context.raw_provider_payload_included",
                "Raw provider payloads are not approved for ME-CI05 advisory artifacts.",
                expected=False,
                actual=context.get("raw_provider_payload_included"),
            )
        )


def _validate_freshness_context(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    context = _require_object(payload.get("freshness_context"), "$.freshness_context", issues)
    if context is None:
        return
    _require_enum(
        context.get("global_freshness_status"),
        "$.freshness_context.global_freshness_status",
        _FRESHNESS_STATUSES,
        issues,
    )
    entries = context.get("family_freshness")
    if not isinstance(entries, list):
        issues.append(
            _issue(
                "invalid_freshness_shape",
                "$.freshness_context.family_freshness",
                "family_freshness must be a list.",
                expected="array",
                actual=type(entries).__name__,
            )
        )
        return
    for index, entry in enumerate(entries):
        entry_path = f"$.freshness_context.family_freshness[{index}]"
        if not isinstance(entry, Mapping):
            issues.append(_issue("invalid_freshness_shape", entry_path, "Freshness entry must be an object."))
            continue
        _require_non_empty_text(entry.get("family"), f"{entry_path}.family", issues)
        _require_enum(entry.get("status"), f"{entry_path}.status", _FRESHNESS_STATUSES, issues)
    if context.get("generated_at_is_not_upstream_freshness") is not True:
        issues.append(
            _issue(
                "validation_metadata_invalid",
                "$.freshness_context.generated_at_is_not_upstream_freshness",
                "Freshness context must state that artifact generation is not upstream freshness.",
                expected=True,
                actual=context.get("generated_at_is_not_upstream_freshness"),
            )
        )


def _validate_uncertainty_context(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    context = _require_object(payload.get("uncertainty_context"), "$.uncertainty_context", issues)
    if context is None:
        return
    _validate_string_list(context.get("missing_evidence"), "$.uncertainty_context.missing_evidence", issues)
    _validate_string_list(context.get("missing_context_families"), "$.uncertainty_context.missing_context_families", issues)
    _validate_string_list(context.get("unresolved_blockers"), "$.uncertainty_context.unresolved_blockers", issues)
    _validate_string_list(context.get("limitations"), "$.uncertainty_context.limitations", issues)


def _validate_missing_context(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    missing_context = payload.get("missing_context")
    if not isinstance(missing_context, list):
        issues.append(
            _issue(
                "invalid_field_type",
                "$.missing_context",
                "missing_context must be a list.",
                expected="array",
                actual=type(missing_context).__name__,
            )
        )
        return
    for index, entry in enumerate(missing_context):
        entry_path = f"$.missing_context[{index}]"
        if not isinstance(entry, Mapping):
            issues.append(_issue("malformed_context", entry_path, "Missing context entry must be an object."))
            continue
        _require_non_empty_text(entry.get("family"), f"{entry_path}.family", issues)
        _require_non_empty_text(entry.get("state"), f"{entry_path}.state", issues)
        _require_non_empty_text(entry.get("note"), f"{entry_path}.note", issues)


def _validate_validation_summary(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    summary = _require_object(payload.get("validation_summary"), "$.validation_summary", issues)
    if summary is None:
        return
    if summary.get("validation_state") not in {"valid_with_limitations", "blocked"}:
        issues.append(
            _issue(
                "validation_metadata_invalid",
                "$.validation_summary.validation_state",
                "validation_state must be valid_with_limitations or blocked.",
                expected=["valid_with_limitations", "blocked"],
                actual=summary.get("validation_state"),
            )
        )
    _validate_string_list(summary.get("errors"), "$.validation_summary.errors", issues)
    _validate_string_list(summary.get("warnings"), "$.validation_summary.warnings", issues)
    _require_object(summary.get("required_sources_present"), "$.validation_summary.required_sources_present", issues)
    _require_object(summary.get("optional_sources_present"), "$.validation_summary.optional_sources_present", issues)
    if summary.get("no_semantic_upgrade_performed") is not True:
        issues.append(
            _issue(
                "validation_metadata_invalid",
                "$.validation_summary.no_semantic_upgrade_performed",
                "validation_summary must preserve no semantic upgrade evidence.",
                expected=True,
                actual=summary.get("no_semantic_upgrade_performed"),
            )
        )


def _validate_cross_context_consistency(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    parent_ticker = _nested_text(payload, ("instrument_identity", "ticker"))
    parent_run_id = _nested_text(payload, ("run_identity", "run_id"))
    source_structured_run_id = _nested_text(
        payload,
        ("run_identity", "source_structured_decision_run_id"),
    )
    sdo = _nested_mapping(payload, ("structured_decision_context", "payload"))
    if sdo is not None:
        _expect_match(
            sdo.get("ticker"),
            parent_ticker,
            "$.structured_decision_context.payload.ticker",
            issues,
            "ticker_identity_mismatch",
        )
        _expect_match(
            sdo.get("run_id"),
            parent_run_id,
            "$.structured_decision_context.payload.run_id",
            issues,
            "run_identity_mismatch",
        )
        _expect_match(
            sdo.get("run_id"),
            source_structured_run_id,
            "$.run_identity.source_structured_decision_run_id",
            issues,
            "run_identity_mismatch",
        )
    context_run_ids = _nested_mapping(payload, ("run_identity", "context_run_ids")) or {}
    advisory_run_id = context_run_ids.get("chatgpt_advisory_context")
    if advisory_run_id is not None and advisory_run_id != parent_run_id:
        issues.append(
            _issue(
                "run_identity_mismatch",
                "$.run_identity.context_run_ids.chatgpt_advisory_context",
                "ME-CI02 advisory context run id must match parent advisory artifact run id.",
                expected=parent_run_id,
                actual=advisory_run_id,
            )
        )
    for family in (
        "portfolio_intelligence_context",
        "explainability_change_rationale_context",
        "governor_context",
        "dispatch_context",
    ):
        _validate_context_ticker_compatibility(payload, family, parent_ticker, issues)
    explainability = _embedded_payload(payload, "explainability_change_rationale_context")
    if explainability is not None:
        current_run_id = _nested_text(explainability, ("current_run_identity", "run_id"))
        if current_run_id != parent_run_id:
            issues.append(
                _issue(
                    "context_identity_mismatch",
                    "$.explainability_change_rationale_context.payload.current_run_identity.run_id",
                    "Explainability current_run_identity must match parent run id.",
                    expected=parent_run_id,
                    actual=current_run_id,
                )
            )


def _validate_context_ticker_compatibility(
    payload: Mapping[str, Any],
    family: str,
    parent_ticker: str | None,
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    embedded = _embedded_payload(payload, family)
    if embedded is None:
        reference = _nested_mapping(payload, (family, "reference"))
        if reference is not None and parent_ticker and reference.get("ticker") not in {None, parent_ticker}:
            issues.append(
                _issue(
                    "context_identity_mismatch",
                    f"$.{family}.reference.ticker",
                    "Referenced context ticker conflicts with parent artifact ticker.",
                    expected=parent_ticker,
                    actual=reference.get("ticker"),
                )
            )
        return
    tickers = _context_tickers(embedded)
    for ticker in sorted(tickers):
        if parent_ticker is not None and ticker != parent_ticker:
            issues.append(
                _issue(
                    "context_identity_mismatch",
                    f"$.{family}.payload",
                    "Embedded context ticker conflicts with parent artifact ticker.",
                    expected=parent_ticker,
                    actual=ticker,
                )
            )


def _validate_contextual_forbidden_fields(
    payload: Mapping[str, Any],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    for path in (
        ("advisory_eligibility",),
        ("composition_status",),
        ("validation_summary",),
        ("dispatch_context", "payload"),
        ("governor_context", "payload"),
    ):
        value = _nested_mapping(payload, path)
        if value is not None:
            _scan_forbidden_fields(value, "$." + ".".join(path), issues)


def _scan_forbidden_fields(
    value: Mapping[str, Any],
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    for key, nested_value in sorted(value.items()):
        child_path = f"{path}.{key}"
        if key in _FORBIDDEN_AUTHORITY_FIELDS:
            issues.append(
                _issue(
                    "forbidden_field_present",
                    child_path,
                    "Field is not authorized at this advisory artifact validation boundary.",
                    actual=key,
                )
            )
        if isinstance(nested_value, Mapping):
            _scan_forbidden_fields(nested_value, child_path, issues)


def _embedded_payload(payload: Mapping[str, Any], family: str) -> Mapping[str, Any] | None:
    context = _nested_mapping(payload, (family,))
    if context is None:
        return None
    if context.get("include_mode") not in {"embedded_preserved_context", "embedded_canonical_context"}:
        return None
    return _nested_mapping(context, ("payload",))


def _context_tickers(context: Mapping[str, Any]) -> set[str]:
    tickers = set()
    for path in (
        ("ticker",),
        ("instrument", "ticker"),
        ("subject", "ticker"),
        ("recommendation_to_position_relationship", "ticker"),
    ):
        ticker = _nested_text(context, path)
        if ticker:
            tickers.add(ticker)
    holdings = context.get("holdings")
    if isinstance(holdings, list):
        for holding in holdings:
            if isinstance(holding, Mapping) and isinstance(holding.get("ticker"), str):
                tickers.add(holding["ticker"])
    return tickers


def _require_object(
    value: Any,
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
    contract_family: str | None = None,
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        issues.append(
            _issue(
                "invalid_field_type",
                path,
                "Field must be a JSON object.",
                expected="object",
                actual=type(value).__name__,
                contract_family=contract_family,
            )
        )
        return None
    return value


def _require_non_empty_text(
    value: Any,
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    if not isinstance(value, str) or not value:
        issues.append(
            _issue(
                "invalid_field_type",
                path,
                "Field must be a non-empty string.",
                expected="non-empty string",
                actual=value,
            )
        )


def _expect_text(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
    *,
    expected: str,
    mismatch_code: str,
    contract_family: str | None = None,
) -> None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        issues.append(
            _issue(
                "missing_required_field" if value is None else "invalid_field_type",
                path,
                "Required contract identity field must be a non-empty string.",
                expected=expected,
                actual=value,
                contract_family=contract_family,
            )
        )
        return
    if value != expected:
        issues.append(
            _issue(
                mismatch_code,
                path,
                "Contract identity value is unsupported.",
                expected=expected,
                actual=value,
                contract_family=contract_family,
            )
        )


def _require_enum(
    value: Any,
    path: str,
    allowed: set[str],
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    if not isinstance(value, str) or value not in allowed:
        issues.append(
            _issue(
                "invalid_enum_value",
                path,
                "Field value is not an allowed enum value.",
                expected=sorted(allowed),
                actual=value,
            )
        )


def _validate_timestamp(
    value: Any,
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
    *,
    allow_null: bool = False,
) -> None:
    if value is None and allow_null:
        return
    if not isinstance(value, str) or not _ISO_UTC_RE.fullmatch(value):
        issues.append(
            _issue(
                "timestamp_invalid",
                path,
                "Timestamp must be an ISO-8601 UTC string with seconds and Z suffix.",
                expected="YYYY-MM-DDTHH:MM:SSZ",
                actual=value,
            )
        )


def _validate_string_list(
    value: Any,
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        issues.append(
            _issue(
                "invalid_field_type",
                path,
                "Field must be a list of strings.",
                expected="array<string>",
                actual=type(value).__name__,
            )
        )


def _expect_match(
    actual: Any,
    expected: Any,
    path: str,
    issues: list[AdvisoryArtifactValidationIssue],
    code: str,
) -> None:
    if expected is None or actual is None or actual != expected:
        issues.append(
            _issue(
                code,
                path,
                "Identity value does not match parent advisory artifact.",
                expected=expected,
                actual=actual,
            )
        )


def _nested_mapping(value: Mapping[str, Any], path: tuple[str, ...]) -> Mapping[str, Any] | None:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _nested_text(value: Mapping[str, Any], path: tuple[str, ...]) -> str | None:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, str) and current else None


def _issue(
    code: str,
    path: str,
    message: str,
    *,
    severity: str = "error",
    contract_family: str | None = None,
    expected: Any = None,
    actual: Any = None,
) -> AdvisoryArtifactValidationIssue:
    return AdvisoryArtifactValidationIssue(
        code=code,
        path=path,
        message=message,
        severity=severity,
        contract_family=contract_family,
        expected=expected,
        actual=actual,
    )


def _result(
    schema_version: str | None,
    issues: list[AdvisoryArtifactValidationIssue],
) -> AdvisoryArtifactValidationResult:
    ordered = tuple(sorted(issues, key=lambda issue: (issue.path, issue.code, issue.message)))
    return AdvisoryArtifactValidationResult(
        status=VALIDATION_STATUS_INVALID if ordered else VALIDATION_STATUS_VALID,
        validator_version=ADVISORY_ARTIFACT_VALIDATOR_VERSION,
        validated_schema_version=schema_version,
        issues=ordered,
    )
