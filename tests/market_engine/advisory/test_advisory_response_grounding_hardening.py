from __future__ import annotations

from copy import deepcopy

import pytest

from market_engine.advisory.advisory_response_grounding import (
    validate_advisory_response_grounding,
)
from tests.market_engine.advisory.test_advisory_artifact import _assemble
from tests.market_engine.advisory.test_advisory_response_grounding import (
    _prompt,
    _response,
)


def test_one_claim_can_have_multiple_valid_evidence_references() -> None:
    source, prompt, response = _baseline()
    response["evidence_references"].append(
        {
            **response["evidence_references"][0],
            "path": "$.structured_decision_context.payload.data_coverage.coverage_status",
            "support_type": "summarized",
        }
    )

    result = _validate(source, prompt, response)

    assert result.status == "grounded"


def test_multiple_claims_can_reuse_same_valid_source_path() -> None:
    source, prompt, response = _baseline()
    response["evidence_supporting"] = [
        {
            "claim_id": "claim-002",
            "claim_type": "evidence_summary",
            "text": "The same source path can support a separate evidence summary.",
        }
    ]
    response["evidence_references"].append(
        {
            **response["evidence_references"][0],
            "claim_id": "claim-002",
            "claim_type": "evidence_summary",
            "support_type": "summarized",
        }
    )

    result = _validate(source, prompt, response)

    assert result.status == "grounded"


def test_exact_duplicate_evidence_reference_is_rejected() -> None:
    source, prompt, response = _baseline()
    response["evidence_references"].append(dict(response["evidence_references"][0]))

    assert "duplicate_evidence_reference" in _codes(_validate(source, prompt, response))


def test_orphan_evidence_reference_is_rejected() -> None:
    source, prompt, response = _baseline()
    response["evidence_references"][0]["claim_id"] = "claim-orphan"

    assert "unknown_claim_reference" in _codes(_validate(source, prompt, response))


def test_material_claim_with_accidentally_different_claim_id_is_not_matched() -> None:
    source, prompt, response = _baseline()
    response["assessment"][0]["claim_id"] = "claim-999"

    result = _validate(source, prompt, response)

    assert "missing_evidence_reference" in _codes(result)
    assert "unknown_claim_reference" in _codes(result)


@pytest.mark.parametrize(
    ("claim_type", "support_type", "expected_valid"),
    [
        ("direct_artifact_fact", "direct", True),
        ("direct_artifact_fact", "interpreted", False),
        ("conditional_interpretation", "conditional", True),
        ("conditional_interpretation", "interpreted", False),
        ("associated_change", "associated_only", True),
        ("explicit_upstream_reason", "associated_only", False),
    ],
)
def test_support_type_compatibility_matrix(
    claim_type: str,
    support_type: str,
    expected_valid: bool,
) -> None:
    source, prompt, response = _baseline(question_class="change_rationale_question")
    response["assessment"][0]["claim_type"] = claim_type
    response["evidence_references"][0]["claim_type"] = claim_type
    response["evidence_references"][0]["support_type"] = support_type
    response["evidence_references"][0]["source_context_family"] = (
        "explainability_change_rationale_context"
    )
    response["evidence_references"][0]["path"] = (
        "$.explainability_change_rationale_context.payload.current_state_rationale.current_state"
    )
    if support_type == "associated_only":
        response["required_disclosures"] = ["causality_disclosure"]
    if expected_valid:
        response["required_disclosures"] = sorted(
            set(response["required_disclosures"] + ["unknown_freshness_disclosure"])
        )
        response["grounding_summary"]["status"] = "grounded_with_mandatory_caveats"

    result = _validate(source, prompt, response)

    if expected_valid:
        assert result.valid is True
    else:
        assert {
            "support_type_incompatible",
            "unsupported_causal_claim",
        }.intersection(_codes(result))


@pytest.mark.parametrize(
    "path",
    [
        "$.structured_decision_context.payload.decision.action",
        "$.provenance_context.source_artifact_refs[0]",
    ],
)
def test_path_resolver_accepts_valid_nested_and_array_paths(path: str) -> None:
    source, prompt, response = _baseline()
    response["assessment"][0]["claim_type"] = "direct_artifact_fact"
    response["evidence_references"][0]["claim_type"] = "direct_artifact_fact"
    response["evidence_references"][0]["support_type"] = "direct"
    response["evidence_references"][0]["path"] = path
    if path.startswith("$.provenance_context"):
        response["evidence_references"][0]["source_context_family"] = "provenance_context"

    assert _validate(source, prompt, response).valid is True


@pytest.mark.parametrize(
    "path",
    [
        "$.source_artifact_references[99]",
        "$.source_artifact_references[-1]",
        "$.structured_decision_context.*",
        "$..decision",
        "",
        "$.structured_decision_context.",
        "$.structured_decision_context.payload.decision.action[0]",
    ],
)
def test_path_resolver_rejects_invalid_or_unsupported_paths(path: str) -> None:
    source, prompt, response = _baseline()
    response["evidence_references"][0]["path"] = path

    result = _validate(source, prompt, response)

    assert {
        "evidence_path_not_allowed",
        "evidence_path_not_found",
    }.intersection(_codes(result))


def test_broad_parent_path_cannot_ground_material_claim() -> None:
    source, prompt, response = _baseline()
    response["evidence_references"][0]["path"] = "$.structured_decision_context"

    assert "evidence_path_not_allowed" in _codes(_validate(source, prompt, response))


def test_evidence_path_must_match_declared_context_family() -> None:
    source, prompt, response = _baseline()
    response["evidence_references"][0]["source_context_family"] = (
        "portfolio_intelligence_context"
    )

    assert "evidence_path_not_allowed" in _codes(_validate(source, prompt, response))


def test_referenced_context_is_not_treated_as_embedded_proof() -> None:
    source = _assemble()
    source["portfolio_intelligence_context"] = {
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
    prompt = _prompt(source, "portfolio_context_question")
    response = _response(prompt, response_mode="partial_answer")
    response["assessment"][0]["text"] = "Portfolio fit is positive."
    response["portfolio_context"] = {"availability": "referenced", "claims": []}
    response["required_disclosures"] = ["missing_portfolio_disclosure"]
    response["unable_to_determine"] = [
        {
            "claim_id": "claim-003",
            "claim_type": "missingness_statement",
            "text": "Portfolio content is referenced only and cannot be proven locally.",
        }
    ]
    response["evidence_references"][0]["source_context_family"] = (
        "portfolio_intelligence_context"
    )
    response["evidence_references"][0]["path"] = "$.portfolio_intelligence_context"
    response["grounding_summary"]["status"] = "partially_grounded"

    result = _validate(source, prompt, response)

    assert "evidence_path_not_allowed" in _codes(result)


def test_absent_context_cannot_ground_known_fact() -> None:
    source = _assemble(portfolio_intelligence_context=None)
    prompt = _prompt(source, "portfolio_context_question")
    response = _response(prompt, response_mode="partial_answer")
    response["assessment"][0]["text"] = "The portfolio is held with known exposure."
    response["portfolio_context"] = {"availability": "absent", "claims": []}
    response["required_disclosures"] = ["missing_portfolio_disclosure"]
    response["unable_to_determine"] = [
        {
            "claim_id": "claim-003",
            "claim_type": "missingness_statement",
            "text": "Portfolio impact cannot be determined.",
        }
    ]

    result = _validate(source, prompt, response)

    assert "missing_context_used_as_fact" in _codes(result)


@pytest.mark.parametrize(
    "text",
    [
        "Known holding quantity proves full portfolio fit.",
        "Cash balance is deployable cash for a position size.",
        "Current weight should become the target weight.",
        "The concentration warning means sell now.",
    ],
)
def test_portfolio_boundary_subtle_misuse_is_detected(text: str) -> None:
    source, prompt, response = _baseline(question_class="portfolio_context_question")
    response["assessment"][0]["text"] = text
    response["evidence_references"][0]["source_context_family"] = (
        "portfolio_intelligence_context"
    )
    response["evidence_references"][0]["path"] = (
        "$.portfolio_intelligence_context.payload.holdings[0].position_state"
    )

    result = _validate(source, prompt, response)

    assert {
        "unsupported_sizing_claim",
        "unsupported_allocation_claim",
        "semantic_override_detected",
    }.intersection(_codes(result))


@pytest.mark.parametrize(
    "summary",
    [
        "Nothing changed.",
        "The main reason is proven.",
        "This is the root cause.",
    ],
)
def test_explainability_summary_without_comparable_context_fails(summary: str) -> None:
    source = _assemble(explainability_change_rationale_context=None)
    prompt = _prompt(source, "change_rationale_question")
    response = _response(prompt)
    response["summary"] = summary

    assert "unsupported_causal_claim" in _codes(_validate(source, prompt, response))


def test_irrelevant_stale_family_does_not_force_disclosure() -> None:
    source, prompt, response = _baseline(question_class="current_state_explanation")
    source["freshness_context"]["family_freshness"].append(
        {"family": "portfolio_intelligence_context", "status": "stale"}
    )
    prompt = _prompt(source, "current_state_explanation")

    result = _validate(source, prompt, response)

    assert result.status == "grounded"


def test_relevant_stale_family_requires_disclosure() -> None:
    source, prompt, response = _baseline(question_class="portfolio_context_question")
    source["freshness_context"]["family_freshness"].append(
        {"family": "portfolio_intelligence_context", "status": "stale"}
    )
    prompt = _prompt(source, "portfolio_context_question")
    response["evidence_references"][0]["source_context_family"] = (
        "portfolio_intelligence_context"
    )
    response["evidence_references"][0]["path"] = (
        "$.portfolio_intelligence_context.payload.holdings[0].position_state"
    )

    assert "required_disclosure_missing" in _codes(_validate(source, prompt, response))


def test_blocker_neutralization_in_summary_is_detected() -> None:
    source, prompt, response = _baseline()
    source["blockers"] = ["required_source_missing"]
    prompt = _prompt(source, "current_state_explanation")
    response["blockers"] = [
        {
            "claim_id": "claim-003",
            "claim_type": "missingness_statement",
            "text": "required_source_missing",
        }
    ]
    response["summary"] = "There are no relevant blockers."

    assert "blocker_omission" in _codes(_validate(source, prompt, response))


def test_dispatch_contradiction_disclosed_but_cherry_picked_is_blocked() -> None:
    dispatch = {
        "report_contract_version": "market-engine-dispatch-station-governor-report-v1",
        "subject": {"ticker": "NVDA"},
        "presentation_summary": {"decision": "different"},
    }
    source = _assemble(dispatch_context=dispatch)
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt, response_mode="partial_answer")
    response["summary"] = "Dispatch says different."
    response["required_disclosures"] = ["contradiction_disclosure"]
    response["unable_to_determine"] = [
        {
            "claim_id": "claim-003",
            "claim_type": "missingness_statement",
            "text": "Dispatch conflict cannot be used as canonical state.",
        }
    ]
    response["grounding_summary"]["status"] = "partially_grounded"

    assert "contradiction_not_disclosed" in _codes(_validate(source, prompt, response))


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (lambda r: r["source_artifact_identity"].update({"schema_version": "wrong"}), "source_artifact_identity_mismatch"),
        (lambda r: r["source_artifact_identity"].update({"artifact_type": "wrong"}), "source_artifact_identity_mismatch"),
        (lambda r: r["source_artifact_identity"].update({"run_id": "wrong"}), "source_artifact_identity_mismatch"),
        (lambda r: r["instrument_identity"].update({"ticker": "AMD"}), "instrument_identity_mismatch"),
        (lambda r: r["question_classification"].update({"question_class": "risk_question"}), "question_classification_mismatch"),
        (lambda r: r["evidence_references"][0].update({"artifact_ref": "wrong"}), "invalid_artifact_reference"),
        (lambda r: r["evidence_references"][0].update({"run_id": "wrong"}), "run_identity_mismatch"),
    ],
)
def test_lineage_mismatches_are_reported_separately(mutator, expected_code: str) -> None:
    source, prompt, response = _baseline()
    mutator(response)

    assert expected_code in _codes(_validate(source, prompt, response))


def test_declared_grounding_status_mismatch_is_detected() -> None:
    source, prompt, response = _baseline()
    response["grounding_summary"]["status"] = "ungrounded"

    result = _validate(source, prompt, response)

    assert result.status == "ungrounded"
    assert "declared_grounding_status_mismatch" in _codes(result)


def test_partial_answer_requires_unable_to_determine() -> None:
    source = _assemble(portfolio_intelligence_context=None)
    prompt = _prompt(source, "portfolio_context_question")
    response = _response(prompt, response_mode="partial_answer")
    response["portfolio_context"] = {"availability": "absent", "claims": []}
    response["required_disclosures"] = ["missing_portfolio_disclosure"]
    response["grounding_summary"]["status"] = "partially_grounded"

    assert "partial_answer_incomplete" in _codes(_validate(source, prompt, response))


def test_deterministic_issue_order_for_multi_error_fixture() -> None:
    source, prompt, response = _baseline()
    response["source_artifact_identity"]["run_id"] = "wrong"
    response["instrument_identity"]["ticker"] = "AMD"
    response["evidence_references"][0]["path"] = "$.missing.path"

    first = _validate(source, prompt, deepcopy(response)).to_payload()
    second = _validate(source, prompt, deepcopy(response)).to_payload()

    assert first == second
    assert [
        (issue["code"], issue["path"], issue.get("claim_id"))
        for issue in first["issues"]
    ] == [
        (issue["code"], issue["path"], issue.get("claim_id"))
        for issue in second["issues"]
    ]


def _baseline(
    *,
    question_class: str = "current_state_explanation",
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source = _assemble()
    prompt = _prompt(source, question_class)
    response = _response(prompt)
    return source, prompt, response


def _validate(source, prompt, response):
    return validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )


def _codes(result) -> set[str]:
    return {issue.code for issue in result.issues}
