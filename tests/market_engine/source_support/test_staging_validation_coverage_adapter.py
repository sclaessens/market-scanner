from __future__ import annotations

from dataclasses import replace

import pytest

from market_engine.source_refresh.cached_source_snapshot_staging_validator import (
    CACHED_SOURCE_SNAPSHOT_STAGING_VALIDATION_FORMAT_VERSION,
)
from market_engine.source_support.cached_source_coverage import (
    BlockerCode,
    CoverageStatus,
    ReadinessStatus,
    SourceFamily,
    TargetCapability,
    classify_cached_source_coverage,
    classify_cached_source_coverage_batch,
)
from market_engine.source_support.staging_validation_coverage_adapter import (
    StagingValidationCoverageAdapterError,
    adapt_staging_validation_batch_to_cached_source_coverage_inputs,
    adapt_staging_validation_to_cached_source_coverage_input,
)


def _entry(
    *,
    ticker: str = "AAA",
    market: str | None = "NASDAQ",
    source_family: str | None = "company_profile",
    staging_validation_status: str = "accepted",
    accepted: bool = True,
    manifest_path: str | None = "AAA/company_profile/manifest.json",
    payload_path: str | None = "AAA/company_profile/company_profile.json",
    directory_path: str | None = None,
    source_name: str | None = "local-company-profile-adapter",
    source_retrieved_at_utc: str | None = "2026-07-02T12:00:00Z",
    validation_status: str | None = "passed",
    validation_errors: tuple[str, ...] = (),
    validation_warnings: tuple[str, ...] = (),
    staleness_status: str | None = "fresh",
    usable: bool | None = True,
    issues: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "market": market,
        "snapshot_id": f"{ticker}-snapshot",
        "source_family": source_family,
        "source_name": source_name,
        "source_retrieved_at_utc": source_retrieved_at_utc,
        "source_publication_date": None,
        "manifest_path": manifest_path,
        "directory_path": directory_path,
        "payload_path": payload_path,
        "manifest_format_version": (
            "market-engine-cached-source-acquisition-manifest-v1"
        ),
        "staging_validation_status": staging_validation_status,
        "accepted_for_cached_source_staging": accepted,
        "validation_status": validation_status,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "staleness_status": staleness_status,
        "usable_for_cached_source_dry_run": usable,
        "blocked_reason": None,
        "issues": issues,
    }


def _blocker_codes(classification: object) -> set[BlockerCode]:
    return {blocker.code for blocker in classification.blockers}


def test_accepted_company_profile_maps_to_descriptive_non_actionable_input() -> None:
    coverage_input = adapt_staging_validation_to_cached_source_coverage_input(
        _entry(),
        universe_supported=True,
        target_capability=TargetCapability.RECOMMENDATION_REVIEW,
    )
    classification = classify_cached_source_coverage(coverage_input)

    assert coverage_input.ticker == "AAA"
    assert coverage_input.market == "NASDAQ"
    assert (
        coverage_input.source_evidence[0].source_family
        is SourceFamily.COMPANY_PROFILE
    )
    assert classification.readiness_status is ReadinessStatus.DESCRIPTIVE_ONLY
    assert classification.actionable is False
    assert classification.de_ready is False


def test_rejected_staging_validation_maps_to_explicit_consumability_blocker() -> None:
    coverage_input = adapt_staging_validation_to_cached_source_coverage_input(
        _entry(
            staging_validation_status="rejected",
            accepted=False,
            validation_status="failed",
            validation_errors=("payload_schema_invalid",),
            usable=False,
            issues=("validation_status_failed",),
        ),
        universe_supported=True,
        target_capability=TargetCapability.DESCRIPTIVE_ANALYSIS,
    )
    classification = classify_cached_source_coverage(coverage_input)

    assert (
        coverage_input.source_evidence[0].consumability_status
        is CoverageStatus.NOT_CONSUMABLE
    )
    assert BlockerCode.SOURCE_NOT_CONSUMABLE in _blocker_codes(classification)
    assert classification.readiness_status is ReadinessStatus.BLOCKED


def test_missing_manifest_fails_closed_with_hint() -> None:
    coverage_input = adapt_staging_validation_to_cached_source_coverage_input(
        _entry(
            source_family=None,
            staging_validation_status="missing_manifest",
            accepted=False,
            manifest_path=None,
            payload_path=None,
            directory_path="AAA/company_profile",
            source_name=None,
            source_retrieved_at_utc=None,
            validation_status=None,
            staleness_status=None,
            usable=None,
            issues=("manifest_missing",),
        ),
        universe_supported=True,
        target_capability=TargetCapability.ANALYSIS,
        source_family_hint=SourceFamily.FUNDAMENTAL_FACTS,
    )
    classification = classify_cached_source_coverage(coverage_input)

    assert (
        coverage_input.source_evidence[0].manifest_status
        is CoverageStatus.INVALID_MANIFEST
    )
    assert BlockerCode.INVALID_MANIFEST in _blocker_codes(classification)


def test_missing_provenance_fails_closed() -> None:
    coverage_input = adapt_staging_validation_to_cached_source_coverage_input(
        _entry(
            source_retrieved_at_utc=None,
            staging_validation_status="rejected",
            accepted=False,
            usable=False,
            issues=("source_retrieved_at_utc_missing",),
        ),
        universe_supported=True,
        target_capability=TargetCapability.DESCRIPTIVE_ANALYSIS,
    )
    classification = classify_cached_source_coverage(coverage_input)

    assert (
        coverage_input.source_evidence[0].provenance_status
        is CoverageStatus.UNPROVENANCED
    )
    assert BlockerCode.MISSING_PROVENANCE in _blocker_codes(classification)


def test_stale_evidence_maps_to_stale_blocked_state() -> None:
    coverage_input = adapt_staging_validation_to_cached_source_coverage_input(
        _entry(
            staging_validation_status="rejected",
            accepted=False,
            staleness_status="stale",
            issues=("snapshot_stale",),
        ),
        universe_supported=True,
        target_capability=TargetCapability.DESCRIPTIVE_ANALYSIS,
    )
    classification = classify_cached_source_coverage(coverage_input)

    assert coverage_input.source_evidence[0].freshness_status is CoverageStatus.STALE
    assert classification.coverage_status is CoverageStatus.STALE
    assert BlockerCode.STALE_SOURCE in _blocker_codes(classification)


def test_unsupported_source_family_remains_explicitly_unsupported() -> None:
    coverage_input = adapt_staging_validation_to_cached_source_coverage_input(
        _entry(source_family="unapproved_fundamental_feed"),
        universe_supported=True,
        target_capability=TargetCapability.ANALYSIS,
        source_family_hint=SourceFamily.FUNDAMENTAL_FACTS,
    )
    classification = classify_cached_source_coverage(coverage_input)

    assert (
        coverage_input.source_evidence[0].support_status
        is CoverageStatus.UNSUPPORTED
    )
    assert classification.coverage_status is CoverageStatus.UNSUPPORTED
    assert BlockerCode.UNSUPPORTED_SOURCE_FAMILY in _blocker_codes(classification)


def test_sec_companyfacts_staging_evidence_remains_partial_and_non_actionable() -> None:
    coverage_input = adapt_staging_validation_to_cached_source_coverage_input(
        _entry(source_family="sec_companyfacts"),
        universe_supported=True,
        target_capability=TargetCapability.RECOMMENDATION_REVIEW,
    )
    classification = classify_cached_source_coverage(coverage_input)

    evidence = coverage_input.source_evidence[0]
    assert evidence.source_family is SourceFamily.FUNDAMENTAL_FACTS
    assert evidence.completeness_status is CoverageStatus.PARTIAL
    assert classification.readiness_status is ReadinessStatus.UNAVAILABLE
    assert classification.actionable is False
    assert classification.de_ready is False


def test_batch_adapter_preserves_order_and_classifier_is_deterministic() -> None:
    report = {
        "report_format_version": (
            CACHED_SOURCE_SNAPSHOT_STAGING_VALIDATION_FORMAT_VERSION
        ),
        "entries": (
            _entry(ticker="ZZZ"),
            _entry(ticker="AAA"),
        ),
    }
    inputs = adapt_staging_validation_batch_to_cached_source_coverage_inputs(
        report,
        universe_supported=True,
        target_capability=TargetCapability.DESCRIPTIVE_ANALYSIS,
    )
    batch = classify_cached_source_coverage_batch(inputs)

    assert [item.ticker for item in inputs] == ["ZZZ", "AAA"]
    assert [item.ticker for item in batch.classifications] == ["ZZZ", "AAA"]
    assert batch.actionable_count == 0
    assert batch.de_ready_count == 0


def test_same_ticker_on_different_markets_remains_distinct_data() -> None:
    inputs = adapt_staging_validation_batch_to_cached_source_coverage_inputs(
        (
            _entry(ticker="DUAL", market="PRIMARY"),
            _entry(ticker="DUAL", market="SECONDARY"),
        ),
        universe_supported=True,
        target_capability=TargetCapability.DESCRIPTIVE_ANALYSIS,
    )
    batch = classify_cached_source_coverage_batch(inputs)

    assert [
        (item.ticker, item.market)
        for item in batch.classifications
    ] == [
        ("DUAL", "PRIMARY"),
        ("DUAL", "SECONDARY"),
    ]


def test_ticker_values_are_data_only() -> None:
    inputs = tuple(
        adapt_staging_validation_to_cached_source_coverage_input(
            _entry(ticker=ticker),
            universe_supported=True,
            target_capability=TargetCapability.DESCRIPTIVE_ANALYSIS,
        )
        for ticker in ("AAA", "FUTURE_XYZ")
    )
    classifications = tuple(
        classify_cached_source_coverage(coverage_input)
        for coverage_input in inputs
    )

    assert replace(classifications[0], ticker="") == replace(
        classifications[1],
        ticker="",
    )


def test_reserved_states_remain_unreachable_after_adaptation() -> None:
    coverage_input = adapt_staging_validation_to_cached_source_coverage_input(
        _entry(),
        universe_supported=True,
        target_capability=TargetCapability.DECISION_ENGINE_HANDOFF,
    )
    classification = classify_cached_source_coverage(coverage_input)

    assert classification.readiness_status not in {
        ReadinessStatus.ACTIONABLE,
        ReadinessStatus.DE_READY,
    }
    assert classification.actionable is False
    assert classification.de_ready is False


@pytest.mark.parametrize(
    "entry, message",
    (
        (_entry(ticker=""), "ticker"),
        (_entry(staging_validation_status="invented"), "status"),
        (_entry(validation_errors=("ok", 3)), "validation_errors"),
    ),
)
def test_malformed_evidence_fails_closed(
    entry: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(StagingValidationCoverageAdapterError, match=message):
        adapt_staging_validation_to_cached_source_coverage_input(
            entry,
            universe_supported=True,
            target_capability=TargetCapability.ANALYSIS,
        )


def test_unsupported_report_format_fails_closed() -> None:
    with pytest.raises(
        StagingValidationCoverageAdapterError,
        match="format",
    ):
        adapt_staging_validation_batch_to_cached_source_coverage_inputs(
            {"report_format_version": "future", "entries": (_entry(),)},
            universe_supported=True,
            target_capability=TargetCapability.ANALYSIS,
        )
