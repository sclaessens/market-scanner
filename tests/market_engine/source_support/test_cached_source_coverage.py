from __future__ import annotations

from dataclasses import replace

import pytest

from market_engine.source_support.cached_source_coverage import (
    CACHED_SOURCE_COVERAGE_CONTRACT_VERSION,
    BlockerCode,
    CachedSourceCoverageError,
    CachedSourceCoverageInput,
    CoverageStatus,
    ReadinessStatus,
    RequirementMode,
    SourceFamily,
    SourceFamilyEvidence,
    SourceFamilyRequirement,
    TargetCapability,
    classify_cached_source_coverage,
    classify_cached_source_coverage_batch,
    to_plain_dict,
)


def _evidence(
    source_family: SourceFamily,
    **overrides: CoverageStatus | str | None,
) -> SourceFamilyEvidence:
    values = {
        "source_family": source_family,
        "evidence_reference": f"artifact://{source_family.value}",
    }
    values.update(overrides)
    return SourceFamilyEvidence(**values)


def _input(
    *,
    ticker: str = "AAA",
    universe_supported: bool = True,
    target_capability: TargetCapability = TargetCapability.ANALYSIS,
    requirements: tuple[SourceFamilyRequirement, ...] = (),
    source_evidence: tuple[SourceFamilyEvidence, ...] = (),
) -> CachedSourceCoverageInput:
    return CachedSourceCoverageInput(
        ticker=ticker,
        universe_supported=universe_supported,
        target_capability=target_capability,
        requirements=requirements,
        source_evidence=source_evidence,
    )


def _recommendation_evidence() -> tuple[SourceFamilyEvidence, ...]:
    return tuple(
        _evidence(source_family)
        for source_family in (
            SourceFamily.FUNDAMENTAL_FACTS,
            SourceFamily.PRICE_HISTORY,
            SourceFamily.SETUP_DETECTION_INPUT,
            SourceFamily.RECOMMENDATION_REVIEW_INPUT,
        )
    )


def _handoff_evidence() -> tuple[SourceFamilyEvidence, ...]:
    return _recommendation_evidence() + (
        _evidence(SourceFamily.PORTFOLIO_CONTEXT),
        _evidence(SourceFamily.DECISION_ENGINE_HANDOFF_INPUT),
    )


def _blocker_codes(classification: object) -> set[BlockerCode]:
    return {blocker.code for blocker in classification.blockers}


def test_unsupported_universe_blocks_without_using_ticker_identity() -> None:
    classification = classify_cached_source_coverage(
        _input(
            universe_supported=False,
            source_evidence=(_evidence(SourceFamily.FUNDAMENTAL_FACTS),),
        )
    )

    assert classification.coverage_status is CoverageStatus.UNSUPPORTED
    assert classification.readiness_status is ReadinessStatus.BLOCKED
    assert BlockerCode.UNSUPPORTED_UNIVERSE in _blocker_codes(classification)
    assert classification.actionable is False
    assert classification.de_ready is False


def test_missing_required_snapshot_is_unavailable_and_blocked() -> None:
    classification = classify_cached_source_coverage(_input())

    assert classification.coverage_status is CoverageStatus.MISSING_SNAPSHOT
    assert classification.readiness_status is ReadinessStatus.UNAVAILABLE
    assert BlockerCode.MISSING_CACHED_SOURCE_SNAPSHOT in _blocker_codes(
        classification
    )
    assert BlockerCode.MISSING_FUNDAMENTAL_EVIDENCE in _blocker_codes(
        classification
    )


def test_invalid_manifest_fails_closed() -> None:
    classification = classify_cached_source_coverage(
        _input(
            source_evidence=(
                _evidence(
                    SourceFamily.FUNDAMENTAL_FACTS,
                    manifest_status=CoverageStatus.INVALID_MANIFEST,
                ),
            )
        )
    )

    assert classification.coverage_status is CoverageStatus.INVALID_MANIFEST
    assert classification.readiness_status is ReadinessStatus.BLOCKED
    assert BlockerCode.INVALID_MANIFEST in _blocker_codes(classification)


def test_unsupported_required_source_family_fails_closed() -> None:
    classification = classify_cached_source_coverage(
        _input(
            source_evidence=(
                _evidence(
                    SourceFamily.FUNDAMENTAL_FACTS,
                    support_status=CoverageStatus.UNSUPPORTED,
                ),
            )
        )
    )

    assert classification.coverage_status is CoverageStatus.UNSUPPORTED
    assert classification.readiness_status is ReadinessStatus.BLOCKED
    assert BlockerCode.UNSUPPORTED_SOURCE_FAMILY in _blocker_codes(
        classification
    )


def test_missing_provenance_fails_closed() -> None:
    classification = classify_cached_source_coverage(
        _input(
            source_evidence=(
                _evidence(
                    SourceFamily.FUNDAMENTAL_FACTS,
                    provenance_status=CoverageStatus.UNPROVENANCED,
                ),
            )
        )
    )

    assert classification.coverage_status is CoverageStatus.UNPROVENANCED
    assert classification.readiness_status is ReadinessStatus.BLOCKED
    assert BlockerCode.MISSING_PROVENANCE in _blocker_codes(classification)


def test_stale_required_source_blocks_actionable_and_de_ready() -> None:
    evidence = tuple(
        replace(
            item,
            freshness_status=CoverageStatus.STALE,
        )
        if item.source_family is SourceFamily.PRICE_HISTORY
        else item
        for item in _handoff_evidence()
    )
    classification = classify_cached_source_coverage(
        _input(
            target_capability=TargetCapability.DECISION_ENGINE_HANDOFF,
            source_evidence=evidence,
        )
    )

    assert classification.coverage_status is CoverageStatus.STALE
    assert classification.readiness_status is ReadinessStatus.BLOCKED
    assert BlockerCode.STALE_SOURCE in _blocker_codes(classification)
    assert classification.actionable is False
    assert classification.de_ready is False


def test_non_consumable_source_fails_closed() -> None:
    classification = classify_cached_source_coverage(
        _input(
            source_evidence=(
                _evidence(
                    SourceFamily.FUNDAMENTAL_FACTS,
                    consumability_status=CoverageStatus.NOT_CONSUMABLE,
                ),
            )
        )
    )

    assert classification.coverage_status is CoverageStatus.NOT_CONSUMABLE
    assert classification.readiness_status is ReadinessStatus.BLOCKED
    assert BlockerCode.SOURCE_NOT_CONSUMABLE in _blocker_codes(classification)


def test_company_profile_only_is_descriptive_and_non_actionable() -> None:
    classification = classify_cached_source_coverage(
        _input(
            target_capability=TargetCapability.RECOMMENDATION_REVIEW,
            source_evidence=(_evidence(SourceFamily.COMPANY_PROFILE),),
        )
    )

    assert classification.coverage_status is CoverageStatus.DESCRIPTIVE_ONLY
    assert classification.readiness_status is ReadinessStatus.DESCRIPTIVE_ONLY
    assert (
        BlockerCode.COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE
        in _blocker_codes(classification)
    )
    assert classification.recommendation_review_allowed is False
    assert classification.actionable is False
    assert classification.de_ready is False


def test_partial_coverage_preserves_explicit_family_blockers() -> None:
    classification = classify_cached_source_coverage(
        _input(
            target_capability=TargetCapability.RECOMMENDATION_REVIEW,
            source_evidence=(
                _evidence(SourceFamily.FUNDAMENTAL_FACTS),
                _evidence(SourceFamily.PRICE_HISTORY),
            ),
        )
    )

    assert classification.coverage_status is CoverageStatus.PARTIAL
    assert classification.readiness_status is ReadinessStatus.PARTIAL
    assert BlockerCode.MISSING_SETUP_OR_PRICE_CONTEXT in _blocker_codes(
        classification
    )
    assert BlockerCode.RECOMMENDATION_REVIEW_BLOCKED in _blocker_codes(
        classification
    )


def test_complete_analysis_coverage_is_not_automatically_actionable() -> None:
    classification = classify_cached_source_coverage(
        _input(
            source_evidence=(_evidence(SourceFamily.FUNDAMENTAL_FACTS),),
        )
    )

    assert classification.coverage_status is CoverageStatus.ACCEPTED
    assert classification.readiness_status is ReadinessStatus.ANALYSIS_READY
    assert classification.recommendation_review_allowed is False
    assert classification.actionable is False
    assert classification.de_ready is False


def test_missing_setup_input_blocks_recommendation_review() -> None:
    classification = classify_cached_source_coverage(
        _input(
            target_capability=TargetCapability.RECOMMENDATION_REVIEW,
            source_evidence=tuple(
                item
                for item in _recommendation_evidence()
                if item.source_family is not SourceFamily.SETUP_DETECTION_INPUT
            ),
        )
    )

    assert classification.readiness_status is ReadinessStatus.PARTIAL
    assert BlockerCode.MISSING_SETUP_OR_PRICE_CONTEXT in _blocker_codes(
        classification
    )
    assert classification.recommendation_review_allowed is False


def test_missing_portfolio_context_blocks_handoff() -> None:
    classification = classify_cached_source_coverage(
        _input(
            target_capability=TargetCapability.DECISION_ENGINE_HANDOFF,
            source_evidence=tuple(
                item
                for item in _handoff_evidence()
                if item.source_family is not SourceFamily.PORTFOLIO_CONTEXT
            ),
        )
    )

    assert classification.coverage_status is CoverageStatus.PARTIAL
    assert classification.readiness_status is ReadinessStatus.PARTIAL
    assert (
        BlockerCode.BLOCKED_MISSING_PORTFOLIO_CONTEXT
        in _blocker_codes(classification)
    )
    assert classification.de_ready is False


def test_complete_recommendation_input_is_ready_but_not_actionable() -> None:
    classification = classify_cached_source_coverage(
        _input(
            target_capability=TargetCapability.RECOMMENDATION_REVIEW,
            source_evidence=_recommendation_evidence(),
        )
    )

    assert classification.coverage_status is CoverageStatus.ACCEPTED
    assert (
        classification.readiness_status
        is ReadinessStatus.RECOMMENDATION_REVIEW_READY
    )
    assert classification.recommendation_review_allowed is True
    assert classification.actionable is False
    assert classification.de_ready is False


def test_actionable_remains_reserved_and_unreachable() -> None:
    classification = classify_cached_source_coverage(
        _input(
            target_capability=TargetCapability.ACTIONABLE_REVIEW,
            source_evidence=_recommendation_evidence(),
        )
    )

    assert (
        classification.readiness_status
        is ReadinessStatus.RECOMMENDATION_REVIEW_READY
    )
    assert BlockerCode.ACTIONABLE_CONTRACT_NOT_APPROVED in _blocker_codes(
        classification
    )
    assert classification.actionable is False
    assert classification.de_ready is False


def test_complete_handoff_remains_structurally_blocked_by_reserved_authority() -> None:
    classification = classify_cached_source_coverage(
        _input(
            target_capability=TargetCapability.DECISION_ENGINE_HANDOFF,
            source_evidence=_handoff_evidence(),
        )
    )

    assert classification.coverage_status is CoverageStatus.ACCEPTED
    assert (
        classification.readiness_status
        is ReadinessStatus.RECOMMENDATION_REVIEW_READY
    )
    assert classification.actionable is False
    assert classification.de_ready is False
    assert classification.recommendation_review_allowed is True
    assert classification.decision_engine_handoff_allowed is False
    assert BlockerCode.ACTIONABLE_CONTRACT_NOT_APPROVED in _blocker_codes(
        classification
    )
    assert BlockerCode.DECISION_ENGINE_HANDOFF_BLOCKED in _blocker_codes(
        classification
    )


@pytest.mark.parametrize("ticker", ("NVDA", "AMD", "ASML"))
def test_me_run28_company_profile_regression_class(
    ticker: str,
) -> None:
    classification = classify_cached_source_coverage(
        _input(
            ticker=ticker,
            target_capability=TargetCapability.RECOMMENDATION_REVIEW,
            source_evidence=(_evidence(SourceFamily.COMPANY_PROFILE),),
        )
    )

    assert classification.readiness_status is ReadinessStatus.DESCRIPTIVE_ONLY
    assert classification.actionable is False
    assert classification.de_ready is False


@pytest.mark.parametrize("ticker", ("AAPL", "GOOGL", "AMZN", "MU"))
def test_me_run28_missing_snapshot_regression_class(
    ticker: str,
) -> None:
    classification = classify_cached_source_coverage(
        _input(ticker=ticker)
    )

    assert classification.coverage_status is CoverageStatus.MISSING_SNAPSHOT
    assert classification.readiness_status is ReadinessStatus.UNAVAILABLE


@pytest.mark.parametrize(
    "ticker",
    ("AVGO", "CLS", "VRT", "COST", "META", "MSFT", "TSM", "CRDO", "IREN"),
)
def test_me_run28_partial_downstream_regression_class(
    ticker: str,
) -> None:
    classification = classify_cached_source_coverage(
        _input(
            ticker=ticker,
            target_capability=TargetCapability.RECOMMENDATION_REVIEW,
            source_evidence=(_evidence(SourceFamily.FUNDAMENTAL_FACTS),),
        )
    )

    assert classification.coverage_status is CoverageStatus.PARTIAL
    assert classification.readiness_status is ReadinessStatus.PARTIAL
    assert classification.actionable is False
    assert classification.de_ready is False


def test_identical_coverage_is_ticker_independent() -> None:
    classifications = tuple(
        classify_cached_source_coverage(
            _input(
                ticker=ticker,
                target_capability=TargetCapability.RECOMMENDATION_REVIEW,
                source_evidence=_recommendation_evidence(),
            )
        )
        for ticker in ("AAA", "NVDA", "FUTURE_XYZ")
    )

    normalized = tuple(
        replace(classification, ticker="")
        for classification in classifications
    )
    assert normalized[0] == normalized[1] == normalized[2]


def test_not_required_family_does_not_block() -> None:
    classification = classify_cached_source_coverage(
        _input(
            target_capability=TargetCapability.DESCRIPTIVE_ANALYSIS,
            requirements=(
                SourceFamilyRequirement(
                    source_family=SourceFamily.PRICE_HISTORY,
                    mode=RequirementMode.NOT_REQUIRED,
                ),
            ),
            source_evidence=(
                _evidence(SourceFamily.COMPANY_PROFILE),
                _evidence(
                    SourceFamily.PRICE_HISTORY,
                    freshness_status=CoverageStatus.STALE,
                ),
            ),
        )
    )

    price_result = next(
        result
        for result in classification.source_family_results
        if result.source_family is SourceFamily.PRICE_HISTORY
    )
    assert price_result.coverage_status is CoverageStatus.NOT_REQUIRED
    assert price_result.requirement_satisfied is True
    assert classification.readiness_status is ReadinessStatus.DESCRIPTIVE_ONLY


def test_batch_classification_preserves_order_and_counts() -> None:
    batch = classify_cached_source_coverage_batch(
        (
            _input(
                ticker="READY",
                source_evidence=(_evidence(SourceFamily.FUNDAMENTAL_FACTS),),
            ),
            _input(ticker="MISSING"),
        )
    )

    assert [item.ticker for item in batch.classifications] == [
        "READY",
        "MISSING",
    ]
    assert {(item.status, item.count) for item in batch.coverage_counts} == {
        ("accepted", 1),
        ("missing_snapshot", 1),
    }
    assert {(item.status, item.count) for item in batch.readiness_counts} == {
        ("analysis_ready", 1),
        ("unavailable", 1),
    }
    assert batch.actionable_count == 0
    assert batch.de_ready_count == 0


def test_plain_dict_preserves_contract_and_audit_fields() -> None:
    classification = classify_cached_source_coverage(
        _input(
            source_evidence=(_evidence(SourceFamily.FUNDAMENTAL_FACTS),),
        )
    )

    payload = to_plain_dict(classification)

    assert payload["contract_version"] == CACHED_SOURCE_COVERAGE_CONTRACT_VERSION
    assert payload["ticker"] == "AAA"
    assert payload["coverage_status"] is CoverageStatus.ACCEPTED
    assert payload["source_family_results"][0]["evidence_reference"].startswith(
        "artifact://"
    )


@pytest.mark.parametrize(
    "coverage_input, message",
    (
        (_input(ticker=" PADDED "), "ticker"),
        (
            _input(
                requirements=(
                    SourceFamilyRequirement(SourceFamily.FUNDAMENTAL_FACTS),
                    SourceFamilyRequirement(SourceFamily.FUNDAMENTAL_FACTS),
                )
            ),
            "requirements",
        ),
        (
            _input(
                source_evidence=(
                    _evidence(SourceFamily.FUNDAMENTAL_FACTS),
                    _evidence(SourceFamily.FUNDAMENTAL_FACTS),
                )
            ),
            "evidence",
        ),
    ),
)
def test_invalid_inputs_fail_closed(
    coverage_input: CachedSourceCoverageInput,
    message: str,
) -> None:
    with pytest.raises(CachedSourceCoverageError, match=message):
        classify_cached_source_coverage(coverage_input)


def test_empty_or_duplicate_batch_fails_closed() -> None:
    with pytest.raises(CachedSourceCoverageError, match="must not be empty"):
        classify_cached_source_coverage_batch(())

    duplicate = _input(ticker="DUPLICATE")
    with pytest.raises(CachedSourceCoverageError, match="must be unique"):
        classify_cached_source_coverage_batch((duplicate, duplicate))
