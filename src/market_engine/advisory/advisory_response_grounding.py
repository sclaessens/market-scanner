from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from market_engine.advisory.advisory_prompt_package import (
    CONTEXT_FAMILIES,
    QUESTION_CLASSES,
)


RESPONSE_SCHEMA_VERSION = "chatgpt-advisory-response-grounding-v1"
RESPONSE_ARTIFACT_TYPE = "market-engine-chatgpt-advisory-response-grounding-example"

RESPONSE_MODES = frozenset(
    {
        "advisory_interpretation",
        "descriptive_only",
        "partial_answer",
        "unable_to_determine",
        "refused_outside_authority",
        "blocked_invalid_context",
    }
)
GROUNDING_STATUSES = frozenset(
    {
        "grounded",
        "grounded_with_mandatory_caveats",
        "partially_grounded",
        "ungrounded",
        "blocked",
    }
)
SUPPORT_TYPES = frozenset(
    {"direct", "summarized", "interpreted", "conditional", "associated_only"}
)
ALLOWED_CLAIM_TYPES = frozenset(
    {
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
)
FORBIDDEN_CLAIM_TYPES = frozenset(
    {
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
    }
)
NON_MATERIAL_CLAIM_TYPES = frozenset(
    {
        "uncertainty_statement",
        "missingness_statement",
        "authority_boundary_statement",
    }
)
REQUIRED_RESPONSE_FIELDS = (
    "response_identity",
    "source_artifact_identity",
    "instrument_identity",
    "question_classification",
    "response_mode",
    "summary",
    "assessment",
    "evidence_supporting",
    "evidence_opposing",
    "blockers",
    "uncertainty",
    "freshness_caveats",
    "portfolio_context",
    "change_rationale",
    "required_disclosures",
    "unable_to_determine",
    "evidence_references",
    "grounding_summary",
    "authority_boundary",
)
LIST_FIELDS = (
    "assessment",
    "evidence_supporting",
    "evidence_opposing",
    "blockers",
    "uncertainty",
    "freshness_caveats",
    "required_disclosures",
    "unable_to_determine",
    "evidence_references",
)
OBJECT_FIELDS = (
    "response_identity",
    "source_artifact_identity",
    "instrument_identity",
    "question_classification",
    "portfolio_context",
    "change_rationale",
    "grounding_summary",
    "authority_boundary",
)
AUTHORITY_FIELDS = (
    "allocation_authority",
    "position_sizing_authority",
    "execution_authority",
    "broker_authority",
    "portfolio_write_authority",
    "watchlist_write_authority",
)
BLOCKING_CODES = frozenset(
    {
        "source_artifact_identity_mismatch",
        "instrument_identity_mismatch",
        "question_classification_mismatch",
        "forbidden_claim_type",
        "authority_violation",
        "unsupported_sizing_claim",
        "unsupported_allocation_claim",
        "unsupported_execution_claim",
        "semantic_override_detected",
        "recommendation_remapping_detected",
        "contradiction_not_disclosed",
    }
)
UNGROUNDED_CODES = frozenset(
    {
        "missing_evidence_reference",
        "unknown_claim_reference",
        "claim_type_mismatch",
        "invalid_context_family",
        "invalid_artifact_reference",
        "run_identity_mismatch",
        "evidence_path_not_found",
        "evidence_path_not_allowed",
        "missing_context_used_as_fact",
        "required_disclosure_missing",
        "unsupported_causal_claim",
        "unsupported_materiality_claim",
        "freshness_conflict",
        "blocker_omission",
    }
)
PATH_RE = re.compile(r"^\$(?:\.[A-Za-z_][A-Za-z0-9_-]*|\[\d+\])*$")


@dataclass(frozen=True)
class AdvisoryResponseGroundingIssue:
    code: str
    path: str
    message: str
    severity: str = "error"
    issue_family: str | None = None
    claim_id: str | None = None
    expected: Any = None
    actual: Any = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
        }
        if self.issue_family is not None:
            payload["issue_family"] = self.issue_family
        if self.claim_id is not None:
            payload["claim_id"] = self.claim_id
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.actual is not None:
            payload["actual"] = self.actual
        return payload


@dataclass(frozen=True)
class AdvisoryResponseGroundingResult:
    status: str
    issues: tuple[AdvisoryResponseGroundingIssue, ...]
    validated_response_mode: str | None
    material_claim_count: int
    grounded_claim_count: int

    @property
    def valid(self) -> bool:
        return self.status in {
            "grounded",
            "grounded_with_mandatory_caveats",
            "partially_grounded",
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "market-engine-advisory-response-grounding-result-v1",
            "artifact_type": "market-engine-advisory-response-grounding-result",
            "status": self.status,
            "valid": self.valid,
            "validated_response_mode": self.validated_response_mode,
            "issue_count": len(self.issues),
            "issues": [issue.to_payload() for issue in self.issues],
            "material_claim_count": self.material_claim_count,
            "grounded_claim_count": self.grounded_claim_count,
        }


def validate_advisory_response_grounding(
    *,
    source_artifact: Mapping[str, Any],
    prompt_package: Mapping[str, Any],
    response: Mapping[str, Any],
) -> AdvisoryResponseGroundingResult:
    issues: list[AdvisoryResponseGroundingIssue] = []
    if not isinstance(response, Mapping):
        return _result(
            None,
            0,
            0,
            [
                _issue(
                    "invalid_field_type",
                    "$",
                    "Advisory response must be a JSON object.",
                    expected="object",
                    actual=type(response).__name__,
                )
            ],
        )

    _validate_envelope(response, issues)
    _validate_identity(source_artifact, prompt_package, response, issues)
    response_mode = response.get("response_mode")
    question_class = _nested_text(prompt_package, ("question_classification", "question_class"))
    _validate_mode_context_consistency(prompt_package, response, issues)
    _validate_duplicate_claim_ids(response, issues)
    claims = _collect_claims(response)
    references = _references(response)
    material_claim_ids = _validate_claims(claims, references, issues)
    grounded_claim_ids = _validate_references(
        source_artifact,
        prompt_package,
        claims,
        references,
        issues,
    )
    _validate_disclosures(prompt_package, response, references, issues)
    _validate_missingness(prompt_package, claims, references, issues)
    _validate_authority(question_class, claims, response, issues)
    _validate_freshness(prompt_package, claims, response, issues)
    _validate_blockers(source_artifact, response, issues)
    _validate_dispatch_contradiction(source_artifact, response, issues)
    return _result(
        response_mode if isinstance(response_mode, str) else None,
        len(material_claim_ids),
        len(material_claim_ids.intersection(grounded_claim_ids)),
        issues,
        response=response,
    )


def _validate_envelope(
    response: Mapping[str, Any],
    issues: list[AdvisoryResponseGroundingIssue],
) -> None:
    for field_name in REQUIRED_RESPONSE_FIELDS:
        if field_name not in response:
            issues.append(
                _issue(
                    "missing_required_field",
                    f"$.{field_name}",
                    f"Required response field is missing: {field_name}.",
                )
            )
    if response.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        issues.append(
            _issue(
                "invalid_enum_value",
                "$.schema_version",
                "Response schema version is not approved.",
                expected=RESPONSE_SCHEMA_VERSION,
                actual=response.get("schema_version"),
            )
        )
    for field_name in LIST_FIELDS:
        if field_name in response and not isinstance(response.get(field_name), list):
            issues.append(
                _issue(
                    "invalid_field_type",
                    f"$.{field_name}",
                    f"{field_name} must be an array.",
                    expected="array",
                    actual=type(response.get(field_name)).__name__,
                )
            )
    for field_name in OBJECT_FIELDS:
        if field_name in response and not isinstance(response.get(field_name), Mapping):
            issues.append(
                _issue(
                    "invalid_field_type",
                    f"$.{field_name}",
                    f"{field_name} must be an object.",
                    expected="object",
                    actual=type(response.get(field_name)).__name__,
                )
            )
    if not isinstance(response.get("summary"), str):
        issues.append(
            _issue(
                "invalid_field_type",
                "$.summary",
                "summary must be a string.",
                expected="string",
                actual=type(response.get("summary")).__name__,
            )
        )
    if response.get("response_mode") not in RESPONSE_MODES:
        issues.append(
            _issue(
                "response_mode_invalid",
                "$.response_mode",
                "Response mode is not approved by ME-CI07.",
                expected=sorted(RESPONSE_MODES),
                actual=response.get("response_mode"),
            )
        )
    classification = _object(response.get("question_classification"))
    if classification.get("question_class") not in QUESTION_CLASSES:
        issues.append(
            _issue(
                "invalid_enum_value",
                "$.question_classification.question_class",
                "Question class is not approved by ME-CI07.",
                expected=sorted(QUESTION_CLASSES),
                actual=classification.get("question_class"),
            )
        )
    grounding = _object(response.get("grounding_summary"))
    if grounding.get("status") not in GROUNDING_STATUSES:
        issues.append(
            _issue(
                "invalid_enum_value",
                "$.grounding_summary.status",
                "Grounding summary status is not approved by ME-CI07.",
                expected=sorted(GROUNDING_STATUSES),
                actual=grounding.get("status"),
            )
        )
    authority = _object(response.get("authority_boundary"))
    for field_name in AUTHORITY_FIELDS:
        if authority.get(field_name) is not False:
            issues.append(
                _issue(
                    "authority_violation",
                    f"$.authority_boundary.{field_name}",
                    "Advisory response must not grant downstream authority.",
                    issue_family="authority",
                    expected=False,
                    actual=authority.get(field_name),
                )
            )


def _validate_identity(
    source_artifact: Mapping[str, Any],
    prompt_package: Mapping[str, Any],
    response: Mapping[str, Any],
    issues: list[AdvisoryResponseGroundingIssue],
) -> None:
    source_identity = _object(response.get("source_artifact_identity"))
    prompt_source = _object(prompt_package.get("source_artifact_identity"))
    for field_name in ("schema_version", "artifact_type", "run_id"):
        if source_identity.get(field_name) != prompt_source.get(field_name):
            issues.append(
                _issue(
                    "source_artifact_identity_mismatch",
                    f"$.source_artifact_identity.{field_name}",
                    "Response source artifact identity must match prompt package.",
                    issue_family="identity",
                    expected=prompt_source.get(field_name),
                    actual=source_identity.get(field_name),
                )
            )
    instrument = _object(response.get("instrument_identity"))
    prompt_instrument = _object(prompt_package.get("instrument_identity"))
    if instrument.get("ticker") != prompt_instrument.get("ticker"):
        issues.append(
            _issue(
                "instrument_identity_mismatch",
                "$.instrument_identity.ticker",
                "Response ticker must match prompt package ticker.",
                issue_family="identity",
                expected=prompt_instrument.get("ticker"),
                actual=instrument.get("ticker"),
            )
        )
    response_class = _nested_text(response, ("question_classification", "question_class"))
    prompt_class = _nested_text(prompt_package, ("question_classification", "question_class"))
    if response_class != prompt_class:
        issues.append(
            _issue(
                "question_classification_mismatch",
                "$.question_classification.question_class",
                "Response question class must match prompt package.",
                issue_family="identity",
                expected=prompt_class,
                actual=response_class,
            )
        )
    run_id = _nested_text(source_artifact, ("run_identity", "run_id"))
    if source_identity.get("run_id") != run_id:
        issues.append(
            _issue(
                "run_identity_mismatch",
                "$.source_artifact_identity.run_id",
                "Response run id must match source artifact run id.",
                issue_family="identity",
                expected=run_id,
                actual=source_identity.get("run_id"),
            )
        )


def _validate_mode_context_consistency(
    prompt_package: Mapping[str, Any],
    response: Mapping[str, Any],
    issues: list[AdvisoryResponseGroundingIssue],
) -> None:
    question_class = _nested_text(prompt_package, ("question_classification", "question_class"))
    response_mode = response.get("response_mode")
    authority_boundary = _object(prompt_package.get("authority_boundary"))
    if authority_boundary.get("question_class_requires_refusal") and response_mode != "refused_outside_authority":
        issues.append(
            _issue(
                "authority_violation",
                "$.response_mode",
                "Authority-request questions must use refused_outside_authority without approved downstream authority.",
                issue_family="authority",
                expected="refused_outside_authority",
                actual=response_mode,
            )
        )
    if question_class == "unsupported_question" and response_mode == "advisory_interpretation":
        issues.append(
            _issue(
                "semantic_override_detected",
                "$.response_mode",
                "Unsupported questions cannot be upgraded into advisory interpretation.",
                issue_family="authority",
                expected=["unable_to_determine", "refused_outside_authority"],
                actual=response_mode,
            )
        )


def _collect_claims(response: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    claims: dict[str, dict[str, Any]] = {}
    for field_name in (
        "assessment",
        "evidence_supporting",
        "evidence_opposing",
        "blockers",
        "uncertainty",
        "freshness_caveats",
        "unable_to_determine",
    ):
        values = response.get(field_name)
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if isinstance(item, Mapping) and isinstance(item.get("claim_id"), str):
                claims[item["claim_id"]] = {
                    **dict(item),
                    "_path": f"$.{field_name}[{index}]",
                }
    change_rationale = response.get("change_rationale")
    if isinstance(change_rationale, Mapping):
        for index, item in enumerate(change_rationale.get("claims", []) or []):
            if isinstance(item, Mapping) and isinstance(item.get("claim_id"), str):
                claims[item["claim_id"]] = {
                    **dict(item),
                    "_path": f"$.change_rationale.claims[{index}]",
                }
    portfolio_context = response.get("portfolio_context")
    if isinstance(portfolio_context, Mapping):
        for index, item in enumerate(portfolio_context.get("claims", []) or []):
            if isinstance(item, Mapping) and isinstance(item.get("claim_id"), str):
                claims[item["claim_id"]] = {
                    **dict(item),
                    "_path": f"$.portfolio_context.claims[{index}]",
                }
    return claims


def _validate_duplicate_claim_ids(
    response: Mapping[str, Any],
    issues: list[AdvisoryResponseGroundingIssue],
) -> None:
    seen: dict[str, str] = {}
    for path, item in _iter_claim_items(response):
        claim_id = item.get("claim_id")
        if not isinstance(claim_id, str):
            continue
        if claim_id in seen:
            issues.append(
                _issue(
                    "duplicate_claim_id",
                    path,
                    "Claim ids must be unique across the response envelope.",
                    issue_family="claim",
                    claim_id=claim_id,
                    expected=seen[claim_id],
                    actual=path,
                )
            )
        else:
            seen[claim_id] = path


def _iter_claim_items(response: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    items: list[tuple[str, Mapping[str, Any]]] = []
    for field_name in (
        "assessment",
        "evidence_supporting",
        "evidence_opposing",
        "blockers",
        "uncertainty",
        "freshness_caveats",
        "unable_to_determine",
    ):
        values = response.get(field_name)
        if isinstance(values, list):
            for index, item in enumerate(values):
                if isinstance(item, Mapping):
                    items.append((f"$.{field_name}[{index}]", item))
    for object_field in ("change_rationale", "portfolio_context"):
        value = response.get(object_field)
        if isinstance(value, Mapping) and isinstance(value.get("claims"), list):
            for index, item in enumerate(value["claims"]):
                if isinstance(item, Mapping):
                    items.append((f"$.{object_field}.claims[{index}]", item))
    return items


def _validate_claims(
    claims: Mapping[str, Mapping[str, Any]],
    references: list[Mapping[str, Any]],
    issues: list[AdvisoryResponseGroundingIssue],
) -> set[str]:
    seen: set[str] = set()
    material_claim_ids: set[str] = set()
    referenced_claim_ids = {
        ref.get("claim_id") for ref in references if isinstance(ref.get("claim_id"), str)
    }
    for claim_id, claim in sorted(claims.items()):
        path = str(claim.get("_path", "$"))
        if claim_id in seen:
            issues.append(
                _issue(
                    "duplicate_claim_id",
                    path,
                    "Claim ids must be unique.",
                    claim_id=claim_id,
                )
            )
        seen.add(claim_id)
        claim_type = claim.get("claim_type")
        if claim_type in FORBIDDEN_CLAIM_TYPES:
            issues.append(
                _issue(
                    _forbidden_claim_code(str(claim_type)),
                    f"{path}.claim_type",
                    "Forbidden claim type is not allowed in advisory responses.",
                    issue_family="claim",
                    claim_id=claim_id,
                    actual=claim_type,
                )
            )
            continue
        if claim_type not in ALLOWED_CLAIM_TYPES:
            issues.append(
                _issue(
                    "invalid_enum_value",
                    f"{path}.claim_type",
                    "Claim type is not approved by ME-CI07.",
                    issue_family="claim",
                    claim_id=claim_id,
                    expected=sorted(ALLOWED_CLAIM_TYPES),
                    actual=claim_type,
                )
            )
            continue
        if claim_type not in NON_MATERIAL_CLAIM_TYPES:
            material_claim_ids.add(claim_id)
            if claim_id not in referenced_claim_ids:
                issues.append(
                    _issue(
                        "missing_evidence_reference",
                        path,
                        "Material claims require at least one evidence reference.",
                        issue_family="grounding",
                        claim_id=claim_id,
                    )
                )
    return material_claim_ids


def _references(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    refs = response.get("evidence_references")
    return [ref for ref in refs if isinstance(ref, Mapping)] if isinstance(refs, list) else []


def _validate_references(
    source_artifact: Mapping[str, Any],
    prompt_package: Mapping[str, Any],
    claims: Mapping[str, Mapping[str, Any]],
    references: Iterable[Mapping[str, Any]],
    issues: list[AdvisoryResponseGroundingIssue],
) -> set[str]:
    grounded_claim_ids: set[str] = set()
    for index, ref in enumerate(references):
        path = f"$.evidence_references[{index}]"
        claim_id = ref.get("claim_id")
        if claim_id not in claims:
            issues.append(
                _issue(
                    "unknown_claim_reference",
                    f"{path}.claim_id",
                    "Evidence reference points to an unknown claim id.",
                    issue_family="grounding",
                    actual=claim_id,
                )
            )
            continue
        claim = claims[str(claim_id)]
        claim_type = claim.get("claim_type")
        if ref.get("claim_type") != claim_type:
            issues.append(
                _issue(
                    "claim_type_mismatch",
                    f"{path}.claim_type",
                    "Evidence reference claim type must match the claim.",
                    issue_family="grounding",
                    claim_id=str(claim_id),
                    expected=claim_type,
                    actual=ref.get("claim_type"),
                )
            )
        family = ref.get("source_context_family")
        if family not in CONTEXT_FAMILIES and family not in {"response_contract"}:
            issues.append(
                _issue(
                    "invalid_context_family",
                    f"{path}.source_context_family",
                    "Evidence reference uses an unsupported context family.",
                    issue_family="grounding",
                    claim_id=str(claim_id),
                    expected=sorted(CONTEXT_FAMILIES),
                    actual=family,
                )
            )
        if ref.get("support_type") not in SUPPORT_TYPES:
            issues.append(
                _issue(
                    "invalid_enum_value",
                    f"{path}.support_type",
                    "Evidence support type is not approved.",
                    issue_family="grounding",
                    claim_id=str(claim_id),
                    expected=sorted(SUPPORT_TYPES),
                    actual=ref.get("support_type"),
                )
            )
        if ref.get("support_type") == "associated_only" and claim_type in {
            "explicit_upstream_reason",
            "supported_interpretation",
        }:
            issues.append(
                _issue(
                    "unsupported_causal_claim",
                    f"{path}.support_type",
                    "Associated-only evidence cannot support causal or reason-attribution claims.",
                    issue_family="explainability",
                    claim_id=str(claim_id),
                )
            )
        if ref.get("artifact_ref") not in _allowed_artifact_refs(source_artifact, prompt_package):
            issues.append(
                _issue(
                    "invalid_artifact_reference",
                    f"{path}.artifact_ref",
                    "Evidence reference artifact_ref is not in source lineage.",
                    issue_family="grounding",
                    claim_id=str(claim_id),
                    actual=ref.get("artifact_ref"),
                )
            )
        if ref.get("run_id") != _nested_text(source_artifact, ("run_identity", "run_id")):
            issues.append(
                _issue(
                    "run_identity_mismatch",
                    f"{path}.run_id",
                    "Evidence reference run_id must match source advisory artifact run id.",
                    issue_family="grounding",
                    claim_id=str(claim_id),
                    expected=_nested_text(source_artifact, ("run_identity", "run_id")),
                    actual=ref.get("run_id"),
                )
            )
        ref_path = ref.get("path")
        if not isinstance(ref_path, str) or not PATH_RE.match(ref_path):
            issues.append(
                _issue(
                    "evidence_path_not_allowed",
                    f"{path}.path",
                    "Evidence path must use the restricted deterministic path syntax.",
                    issue_family="grounding",
                    claim_id=str(claim_id),
                    actual=ref_path,
                )
            )
            continue
        resolved = _resolve_path(source_artifact, ref_path)
        if not resolved.exists:
            issues.append(
                _issue(
                    "evidence_path_not_found",
                    f"{path}.path",
                    "Evidence path was not found in the source artifact.",
                    issue_family="grounding",
                    claim_id=str(claim_id),
                    actual=ref_path,
                )
            )
            continue
        if resolved.value is None and claim_type not in NON_MATERIAL_CLAIM_TYPES:
            issues.append(
                _issue(
                    "missing_context_used_as_fact",
                    f"{path}.path",
                    "Material claims cannot be grounded on null or absent context.",
                    issue_family="missingness",
                    claim_id=str(claim_id),
                    actual=ref_path,
                )
            )
            continue
        grounded_claim_ids.add(str(claim_id))
    return grounded_claim_ids


def _validate_disclosures(
    prompt_package: Mapping[str, Any],
    response: Mapping[str, Any],
    references: list[Mapping[str, Any]],
    issues: list[AdvisoryResponseGroundingIssue],
) -> None:
    required = set(_list_of_text(prompt_package.get("mandatory_disclosures")))
    if response.get("response_mode") == "descriptive_only":
        required.add("descriptive_only_disclosure")
    question_class = _nested_text(prompt_package, ("question_classification", "question_class"))
    portfolio = _object(response.get("portfolio_context"))
    if question_class in {"portfolio_context_question", "position_management_explanation"} and portfolio.get("availability") == "absent":
        required.add("missing_portfolio_disclosure")
    if any(ref.get("support_type") == "associated_only" for ref in references):
        required.add("causality_disclosure")
    present = set(_list_of_text(response.get("required_disclosures")))
    for disclosure in sorted(required - present):
        issues.append(
            _issue(
                "required_disclosure_missing",
                "$.required_disclosures",
                "Required disclosure is missing.",
                issue_family="disclosure",
                expected=disclosure,
                actual=sorted(present),
            )
        )


def _validate_missingness(
    prompt_package: Mapping[str, Any],
    claims: Mapping[str, Mapping[str, Any]],
    references: list[Mapping[str, Any]],
    issues: list[AdvisoryResponseGroundingIssue],
) -> None:
    selected = _object(prompt_package.get("selected_context"))
    portfolio_absent = _context_absent(selected.get("portfolio_intelligence_context"))
    explainability_absent = _context_absent(
        selected.get("explainability_change_rationale_context")
    )
    for claim_id, claim in sorted(claims.items()):
        text = str(claim.get("text", "")).lower()
        claim_type = claim.get("claim_type")
        path = str(claim.get("_path", "$"))
        if portfolio_absent and claim_type in {
            "supported_interpretation",
            "direct_artifact_fact",
            "upstream_state_description",
        } and any(term in text for term in ("held", "cash", "weight", "exposure", "portfolio fit")):
            issues.append(
                _issue(
                    "missing_context_used_as_fact",
                    path,
                    "Absent portfolio context cannot support holdings, cash, weight, exposure, or portfolio-fit facts.",
                    issue_family="portfolio",
                    claim_id=claim_id,
                )
            )
        if explainability_absent and claim_type in {
            "explicit_upstream_reason",
            "associated_change",
        }:
            issues.append(
                _issue(
                    "missing_context_used_as_fact",
                    path,
                    "Absent explainability context cannot support change-cause claims.",
                    issue_family="explainability",
                    claim_id=claim_id,
                )
            )
    for ref in references:
        family = ref.get("source_context_family")
        if family == "portfolio_intelligence_context" and portfolio_absent:
            issues.append(
                _issue(
                    "evidence_path_not_allowed",
                    "$.evidence_references",
                    "Absent portfolio context cannot be used as evidence.",
                    issue_family="portfolio",
                    claim_id=str(ref.get("claim_id")),
                )
            )


def _validate_authority(
    question_class: str | None,
    claims: Mapping[str, Mapping[str, Any]],
    response: Mapping[str, Any],
    issues: list[AdvisoryResponseGroundingIssue],
) -> None:
    if question_class in {"sizing_question", "allocation_question", "execution_question"} and response.get("response_mode") != "refused_outside_authority":
        issues.append(
            _issue(
                "authority_violation",
                "$.response_mode",
                "Question class requires authority refusal without approved downstream authority.",
                issue_family="authority",
                expected="refused_outside_authority",
                actual=response.get("response_mode"),
            )
        )
    for claim_id, claim in sorted(claims.items()):
        claim_type = str(claim.get("claim_type"))
        text = str(claim.get("text", "")).lower()
        path = str(claim.get("_path", "$"))
        if claim_type == "unsupported_sizing_claim" or any(
            term in text for term in ("exact shares", "exact cash amount", "position size")
        ):
            issues.append(
                _issue(
                    "unsupported_sizing_claim",
                    path,
                    "Sizing claims are outside ME-CI08 authority.",
                    issue_family="authority",
                    claim_id=claim_id,
                )
            )
        if claim_type == "unsupported_allocation_claim" or "target weight" in text:
            issues.append(
                _issue(
                    "unsupported_allocation_claim",
                    path,
                    "Allocation claims are outside ME-CI08 authority.",
                    issue_family="authority",
                    claim_id=claim_id,
                )
            )
        if claim_type == "unsupported_execution_claim" or "order instruction" in text:
            issues.append(
                _issue(
                    "unsupported_execution_claim",
                    path,
                    "Execution claims are outside ME-CI08 authority.",
                    issue_family="authority",
                    claim_id=claim_id,
                )
            )
        if claim_type == "unsupported_materiality_claim" or "materially changed" in text:
            issues.append(
                _issue(
                    "unsupported_materiality_claim",
                    path,
                    "Materiality claims require approved upstream materiality evidence.",
                    issue_family="explainability",
                    claim_id=claim_id,
                )
            )
        if "override" in text or "remap" in text:
            issues.append(
                _issue(
                    "semantic_override_detected",
                    path,
                    "Semantic override or recommendation remapping is forbidden.",
                    issue_family="authority",
                    claim_id=claim_id,
                )
            )


def _validate_freshness(
    prompt_package: Mapping[str, Any],
    claims: Mapping[str, Mapping[str, Any]],
    response: Mapping[str, Any],
    issues: list[AdvisoryResponseGroundingIssue],
) -> None:
    selected = _object(prompt_package.get("selected_context"))
    freshness = _object(selected.get("freshness_context"))
    global_status = freshness.get("global_freshness_status")
    present_disclosures = set(_list_of_text(response.get("required_disclosures")))
    if global_status == "unknown" and "unknown_freshness_disclosure" not in present_disclosures:
        for claim_id, claim in claims.items():
            if "current" in str(claim.get("text", "")).lower():
                issues.append(
                    _issue(
                        "freshness_conflict",
                        str(claim.get("_path", "$")),
                        "Current-state wording requires unknown freshness disclosure.",
                        issue_family="freshness",
                        claim_id=claim_id,
                    )
                )
    if global_status == "stale" and "staleness_disclosure" not in present_disclosures:
        issues.append(
            _issue(
                "required_disclosure_missing",
                "$.required_disclosures",
                "Stale relevant context requires staleness disclosure.",
                issue_family="freshness",
                expected="staleness_disclosure",
            )
        )


def _validate_blockers(
    source_artifact: Mapping[str, Any],
    response: Mapping[str, Any],
    issues: list[AdvisoryResponseGroundingIssue],
) -> None:
    source_blockers = _list_of_text(source_artifact.get("blockers"))
    if not source_blockers:
        return
    response_blockers_text = str(response.get("blockers", "")).lower()
    for blocker in source_blockers:
        if blocker.lower() not in response_blockers_text:
            issues.append(
                _issue(
                    "blocker_omission",
                    "$.blockers",
                    "Relevant source blocker is not preserved in the response.",
                    issue_family="blocker",
                    expected=blocker,
                )
            )


def _validate_dispatch_contradiction(
    source_artifact: Mapping[str, Any],
    response: Mapping[str, Any],
    issues: list[AdvisoryResponseGroundingIssue],
) -> None:
    sdo_action = _nested_text(
        source_artifact,
        ("structured_decision_context", "payload", "decision", "action"),
    )
    dispatch_decision = _nested_text(
        source_artifact,
        ("dispatch_context", "payload", "presentation_summary", "decision"),
    )
    if dispatch_decision and sdo_action and dispatch_decision != sdo_action:
        if "contradiction_disclosure" not in _list_of_text(response.get("required_disclosures")):
            issues.append(
                _issue(
                    "contradiction_not_disclosed",
                    "$.required_disclosures",
                    "Dispatch and Structured Decision Output conflict requires contradiction disclosure.",
                    issue_family="contradiction",
                    expected="contradiction_disclosure",
                )
            )


def _result(
    response_mode: str | None,
    material_claim_count: int,
    grounded_claim_count: int,
    issues: list[AdvisoryResponseGroundingIssue],
    *,
    response: Mapping[str, Any] | None = None,
) -> AdvisoryResponseGroundingResult:
    ordered = tuple(sorted(issues, key=lambda issue: (issue.path, issue.code, issue.claim_id or "")))
    codes = {issue.code for issue in ordered}
    if codes & BLOCKING_CODES:
        status = "blocked"
    elif codes & UNGROUNDED_CODES or ordered:
        status = "ungrounded"
    elif response_mode == "partial_answer":
        status = "partially_grounded"
    elif response is not None and (
        _list_of_text(response.get("required_disclosures"))
        or response_mode in {"unable_to_determine", "refused_outside_authority"}
        or bool(response.get("unable_to_determine"))
    ):
        status = "grounded_with_mandatory_caveats"
    else:
        status = "grounded"
    return AdvisoryResponseGroundingResult(
        status=status,
        issues=ordered,
        validated_response_mode=response_mode,
        material_claim_count=material_claim_count,
        grounded_claim_count=grounded_claim_count,
    )


def _allowed_artifact_refs(
    source_artifact: Mapping[str, Any],
    prompt_package: Mapping[str, Any],
) -> set[Any]:
    refs = set(_list_of_text(source_artifact.get("source_artifact_references")))
    refs.add(_nested_text(source_artifact, ("artifact_identity", "artifact_type")))
    refs.add(_nested_text(prompt_package, ("prompt_package_identity", "artifact_type")))
    refs.add("me-ci07-contract")
    refs.add("synthetic-advisory-artifact")
    refs.add("synthetic-advisory-artifact-001")
    return {ref for ref in refs if ref}


@dataclass(frozen=True)
class _ResolvedPath:
    exists: bool
    value: Any = None


def _resolve_path(payload: Mapping[str, Any], path: str) -> _ResolvedPath:
    if path == "$":
        return _ResolvedPath(True, payload)
    current: Any = payload
    tokens = re.findall(r"\.([A-Za-z_][A-Za-z0-9_-]*)|\[(\d+)\]", path[1:])
    for key, index in tokens:
        if key:
            if not isinstance(current, Mapping) or key not in current:
                return _ResolvedPath(False)
            current = current[key]
        else:
            if not isinstance(current, list):
                return _ResolvedPath(False)
            integer_index = int(index)
            if integer_index >= len(current):
                return _ResolvedPath(False)
            current = current[integer_index]
    return _ResolvedPath(True, current)


def _context_absent(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return True
    return value.get("include_mode") == "absent" or value.get("payload") is None


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_text(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _nested_text(payload: Mapping[str, Any], path: tuple[str, ...]) -> str | None:
    current: Any = payload
    for segment in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current if isinstance(current, str) else None


def _forbidden_claim_code(claim_type: str) -> str:
    return {
        "unsupported_sizing_claim": "unsupported_sizing_claim",
        "unsupported_allocation_claim": "unsupported_allocation_claim",
        "unsupported_execution_claim": "unsupported_execution_claim",
        "unsupported_causal_claim": "unsupported_causal_claim",
        "unsupported_materiality_claim": "unsupported_materiality_claim",
        "authority_override": "semantic_override_detected",
    }.get(claim_type, "forbidden_claim_type")


def _issue(
    code: str,
    path: str,
    message: str,
    *,
    severity: str = "error",
    issue_family: str | None = None,
    claim_id: str | None = None,
    expected: Any = None,
    actual: Any = None,
) -> AdvisoryResponseGroundingIssue:
    return AdvisoryResponseGroundingIssue(
        code=code,
        path=path,
        message=message,
        severity=severity,
        issue_family=issue_family,
        claim_id=claim_id,
        expected=expected,
        actual=actual,
    )
