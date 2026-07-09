from __future__ import annotations

from copy import deepcopy

import pytest

from market_engine.advisory.advisory_prompt_package import (
    AdvisoryPromptPackageError,
    build_advisory_prompt_package,
    validate_advisory_prompt_package,
)
from tests.market_engine.advisory.test_advisory_artifact import _assemble


def test_valid_source_artifact_builds_deterministic_prompt_package() -> None:
    artifact = _assemble()

    package_a = build_advisory_prompt_package(
        advisory_artifact=artifact,
        question="Explain the current state.",
        question_class="current_state_explanation",
        package_id="ci08-run-001",
    )
    package_b = build_advisory_prompt_package(
        advisory_artifact=deepcopy(artifact),
        question="Explain the current state.",
        question_class="current_state_explanation",
        package_id="ci08-run-001",
    )

    assert package_a == package_b
    assert validate_advisory_prompt_package(package_a).valid is True
    assert package_a["source_artifact_identity"]["run_id"] == "run-001"
    assert package_a["instrument_identity"]["ticker"] == "NVDA"
    assert package_a["prompt_package_identity"]["model_free"] is True


def test_invalid_ci06_artifact_is_rejected_before_prompt_package() -> None:
    artifact = _assemble()
    artifact["composition_status"]["state"] = "invalid"

    with pytest.raises(AdvisoryPromptPackageError, match="CI06-valid"):
        build_advisory_prompt_package(
            advisory_artifact=artifact,
            question="Explain the current state.",
            question_class="current_state_explanation",
            package_id="ci08-run-001",
        )


def test_question_class_must_be_explicit_and_approved() -> None:
    with pytest.raises(AdvisoryPromptPackageError, match="Unsupported question class"):
        build_advisory_prompt_package(
            advisory_artifact=_assemble(),
            question="Guess what I mean.",
            question_class="keyword_guessed_question",
            package_id="ci08-run-001",
        )


def test_portfolio_question_preserves_absent_context_and_disclosure() -> None:
    artifact = _assemble(portfolio_intelligence_context=None)

    package = build_advisory_prompt_package(
        advisory_artifact=artifact,
        question="What is the portfolio impact?",
        question_class="portfolio_context_question",
        package_id="ci08-run-002",
    )

    assert package["selected_context"]["portfolio_intelligence_context"]["include_mode"] == "absent"
    assert package["question_classification"]["missing_required_context_families"] == [
        "portfolio_intelligence_context"
    ]
    assert "missing_portfolio_disclosure" in package["mandatory_disclosures"]


def test_sizing_question_requires_refusal_authority_boundary() -> None:
    package = build_advisory_prompt_package(
        advisory_artifact=_assemble(),
        question="How much should I buy?",
        question_class="sizing_question",
        package_id="ci08-run-003",
    )

    assert package["authority_boundary"]["question_class_requires_refusal"] is True
    assert "authority_disclosure" in package["mandatory_disclosures"]
    assert package["authority_boundary"]["position_sizing_authority"] is False
