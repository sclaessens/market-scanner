from __future__ import annotations

from copy import deepcopy

from market_engine.advisory.advisory_prompt_package import build_advisory_prompt_package
from market_engine.advisory.advisory_response_grounding import (
    validate_advisory_response_grounding,
)
from tests.market_engine.advisory.test_advisory_artifact import _assemble


def test_grounded_current_state_response_validates() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "grounded"
    assert result.valid is True
    assert result.issues == ()


def test_descriptive_only_with_required_caveat_validates() -> None:
    source = _assemble()
    prompt = _prompt(source, "missing_evidence_question")
    response = _response(prompt, response_mode="descriptive_only")
    response["required_disclosures"] = ["descriptive_only_disclosure"]
    response["grounding_summary"]["status"] = "grounded_with_mandatory_caveats"

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "grounded_with_mandatory_caveats"


def test_partial_answer_with_unable_to_determine_validates() -> None:
    source = _assemble(portfolio_intelligence_context=None)
    prompt = _prompt(source, "portfolio_context_question")
    response = _response(prompt, response_mode="partial_answer")
    response["portfolio_context"] = {
        "availability": "absent",
        "disclosure_required": True,
        "claims": [],
    }
    response["required_disclosures"] = ["missing_portfolio_disclosure"]
    response["unable_to_determine"] = [
        {
            "claim_id": "claim-003",
            "claim_type": "missingness_statement",
            "text": "Portfolio impact cannot be determined because context is absent.",
        }
    ]
    response["grounding_summary"]["status"] = "partially_grounded"

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "partially_grounded"


def test_valid_refusal_outside_authority_validates_with_caveat() -> None:
    source = _assemble()
    prompt = _prompt(source, "sizing_question")
    response = _response(prompt, response_mode="refused_outside_authority")
    response["assessment"] = []
    response["evidence_supporting"] = []
    response["evidence_references"] = []
    response["blockers"] = [
        {
            "claim_id": "claim-001",
            "claim_type": "authority_boundary_statement",
            "text": "No approved sizing authority is present.",
        }
    ]
    response["required_disclosures"] = ["authority_disclosure"]
    response["grounding_summary"]["status"] = "grounded_with_mandatory_caveats"

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "grounded_with_mandatory_caveats"


def test_valid_unable_to_determine_validates() -> None:
    source = _assemble()
    prompt = _prompt(source, "unsupported_question")
    response = _response(prompt, response_mode="unable_to_determine")
    response["assessment"] = []
    response["evidence_supporting"] = []
    response["evidence_references"] = []
    response["unable_to_determine"] = [
        {
            "claim_id": "claim-001",
            "claim_type": "missingness_statement",
            "text": "The requested claim cannot be determined from supplied context.",
        }
    ]
    response["grounding_summary"]["status"] = "grounded_with_mandatory_caveats"

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "grounded_with_mandatory_caveats"


def test_valid_contradiction_disclosure_partial_response_is_allowed() -> None:
    dispatch = {
        "report_contract_version": "market-engine-dispatch-station-governor-report-v1",
        "subject": {"ticker": "NVDA"},
        "presentation_summary": {"decision": "different"},
    }
    source = _assemble(dispatch_context=dispatch)
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt, response_mode="partial_answer")
    response["required_disclosures"] = ["contradiction_disclosure"]
    response["unable_to_determine"] = [
        {
            "claim_id": "claim-003",
            "claim_type": "missingness_statement",
            "text": "The conflicting Dispatch presentation cannot be used as canonical state.",
        }
    ]
    response["grounding_summary"]["status"] = "partially_grounded"

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "partially_grounded"


def test_source_artifact_identity_mismatch_blocks() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["source_artifact_identity"]["run_id"] = "other-run"

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "blocked"
    assert "source_artifact_identity_mismatch" in _codes(result)


def test_ticker_mismatch_blocks() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["instrument_identity"]["ticker"] = "AMD"

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "blocked"
    assert "instrument_identity_mismatch" in _codes(result)


def test_invalid_question_class_is_detected() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["question_classification"]["question_class"] = "unknown"

    assert "question_classification_mismatch" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_invalid_response_mode_is_detected() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["response_mode"] = "invented_mode"

    assert "response_mode_invalid" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_material_claim_without_evidence_ref_is_ungrounded() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["evidence_references"] = []

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "ungrounded"
    assert "missing_evidence_reference" in _codes(result)


def test_duplicate_claim_id_is_detected() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["evidence_supporting"] = [
        {
            "claim_id": "claim-001",
            "claim_type": "evidence_summary",
            "text": "Duplicate claim id.",
        }
    ]

    assert "duplicate_claim_id" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_evidence_ref_to_unknown_claim_is_ungrounded() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["evidence_references"][0]["claim_id"] = "unknown"

    assert "unknown_claim_reference" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_evidence_path_not_found_is_ungrounded() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["evidence_references"][0]["path"] = "$.not.present"

    assert "evidence_path_not_found" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_claim_uses_wrong_context_family() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["evidence_references"][0]["source_context_family"] = "unknown_family"

    assert "invalid_context_family" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_missing_mandatory_disclosure_is_ungrounded() -> None:
    source = _assemble(portfolio_intelligence_context=None)
    prompt = _prompt(source, "portfolio_context_question")
    response = _response(prompt, response_mode="partial_answer")
    response["portfolio_context"] = {"availability": "absent", "claims": []}

    assert "required_disclosure_missing" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_sizing_claim_without_authority_blocks() -> None:
    source = _assemble()
    prompt = _prompt(source, "sizing_question")
    response = _response(prompt, response_mode="advisory_interpretation")
    response["assessment"][0]["claim_type"] = "unsupported_sizing_claim"

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "blocked"
    assert "authority_violation" in _codes(result)
    assert "unsupported_sizing_claim" in _codes(result)


def test_allocation_claim_without_authority_blocks() -> None:
    source = _assemble()
    prompt = _prompt(source, "allocation_question")
    response = _response(prompt, response_mode="advisory_interpretation")
    response["assessment"][0]["claim_type"] = "unsupported_allocation_claim"

    assert "unsupported_allocation_claim" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_execution_claim_without_authority_blocks() -> None:
    source = _assemble()
    prompt = _prompt(source, "execution_question")
    response = _response(prompt, response_mode="advisory_interpretation")
    response["assessment"][0]["claim_type"] = "unsupported_execution_claim"

    assert "unsupported_execution_claim" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_associated_only_cannot_support_reason_claim() -> None:
    source = _assemble()
    prompt = _prompt(source, "change_rationale_question")
    response = _response(prompt)
    response["assessment"][0]["claim_type"] = "explicit_upstream_reason"
    response["evidence_references"][0]["claim_type"] = "explicit_upstream_reason"
    response["evidence_references"][0]["support_type"] = "associated_only"

    assert "unsupported_causal_claim" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_materiality_claim_without_upstream_model_blocks() -> None:
    source = _assemble()
    prompt = _prompt(source, "change_rationale_question")
    response = _response(prompt)
    response["assessment"][0]["text"] = "The change materially changed the setup."

    assert "unsupported_materiality_claim" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_current_wording_against_unknown_freshness_requires_disclosure() -> None:
    source = _assemble()
    source["freshness_context"]["global_freshness_status"] = "unknown"
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)

    assert "freshness_conflict" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_portfolio_claim_with_absent_portfolio_context_is_ungrounded() -> None:
    source = _assemble(portfolio_intelligence_context=None)
    prompt = _prompt(source, "portfolio_context_question")
    response = _response(prompt, response_mode="partial_answer")
    response["assessment"][0]["text"] = "The portfolio is held with known exposure."
    response["required_disclosures"] = ["missing_portfolio_disclosure"]

    assert "missing_context_used_as_fact" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_omitted_mandatory_blocker_is_detected() -> None:
    source = _assemble()
    source["blockers"] = ["required_source_missing"]
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)

    assert "blocker_omission" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_dispatch_structured_decision_contradiction_without_disclosure_blocks() -> None:
    dispatch = {
        "report_contract_version": "market-engine-dispatch-station-governor-report-v1",
        "subject": {"ticker": "NVDA"},
        "presentation_summary": {"decision": "different"},
    }
    source = _assemble(dispatch_context=dispatch)
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)

    assert "contradiction_not_disclosed" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_forbidden_recommendation_remapping_blocks() -> None:
    source = _assemble()
    prompt = _prompt(source, "recommendation_interpretation")
    response = _response(prompt)
    response["assessment"][0]["text"] = "This remap overrides the recommendation."

    assert "semantic_override_detected" in _codes(
        validate_advisory_response_grounding(
            source_artifact=source,
            prompt_package=prompt,
            response=response,
        )
    )


def test_forbidden_claim_type_blocks() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["assessment"][0]["claim_type"] = "invented_fact"

    result = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=response,
    )

    assert result.status == "blocked"
    assert "forbidden_claim_type" in _codes(result)


def test_issue_ordering_is_deterministic() -> None:
    source = _assemble()
    prompt = _prompt(source, "current_state_explanation")
    response = _response(prompt)
    response["source_artifact_identity"]["run_id"] = "other"
    response["instrument_identity"]["ticker"] = "AMD"

    first = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=deepcopy(response),
    ).to_payload()
    second = validate_advisory_response_grounding(
        source_artifact=source,
        prompt_package=prompt,
        response=deepcopy(response),
    ).to_payload()

    assert first == second


def _prompt(source: dict[str, object], question_class: str) -> dict[str, object]:
    return build_advisory_prompt_package(
        advisory_artifact=source,
        question="Synthetic question",
        question_class=question_class,
        package_id="ci08-run-001",
    )


def _response(
    prompt: dict[str, object],
    *,
    response_mode: str = "advisory_interpretation",
) -> dict[str, object]:
    run_id = prompt["source_artifact_identity"]["run_id"]
    ticker = prompt["instrument_identity"]["ticker"]
    question_class = prompt["question_classification"]["question_class"]
    return {
        "schema_version": "chatgpt-advisory-response-grounding-v1",
        "artifact_type": "market-engine-chatgpt-advisory-response-grounding-example",
        "response_identity": {
            "response_id": "synthetic-response-001",
            "response_mode": response_mode,
            "generated_at": "2026-07-08T12:00:00Z",
            "non_production_example": True,
        },
        "source_artifact_identity": dict(prompt["source_artifact_identity"]),
        "instrument_identity": {"ticker": ticker, "asset_type": "equity"},
        "question_classification": {
            "question_class": question_class,
            "requested_scope": "synthetic",
            "required_context_families": prompt["question_classification"][
                "required_context_families"
            ],
            "unavailable_context_families": prompt["question_classification"][
                "missing_required_context_families"
            ],
        },
        "response_mode": response_mode,
        "summary": "The current synthetic context supports a bounded explanation.",
        "assessment": [
            {
                "claim_id": "claim-001",
                "claim_type": "supported_interpretation",
                "text": "The current source state is available for bounded interpretation.",
            }
        ],
        "evidence_supporting": [],
        "evidence_opposing": [],
        "blockers": [],
        "uncertainty": [],
        "freshness_caveats": [],
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
        "required_disclosures": [],
        "unable_to_determine": [],
        "evidence_references": [
            {
                "ref_id": "ref-001",
                "claim_id": "claim-001",
                "claim_type": "supported_interpretation",
                "source_context_family": "structured_decision_output",
                "artifact_ref": "artifact:sdo:NVDA:run-001",
                "run_id": run_id,
                "path": "$.structured_decision_context.payload.decision.action",
                "support_type": "interpreted",
            }
        ],
        "grounding_summary": {
            "status": "grounded",
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


def _codes(result) -> set[str]:
    return {issue.code for issue in result.issues}
