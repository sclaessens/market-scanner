from __future__ import annotations

import json
from dataclasses import asdict, replace

from market_engine.analysis_review.sec_companyfacts_analysis_review import (
    SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION,
    SecCompanyFactsAnalysisReviewCategory,
    SecCompanyFactsAnalysisReviewState,
    build_sec_companyfacts_analysis_review,
    persist_sec_companyfacts_analysis_review,
)
from market_engine.derived_observations.sec_companyfacts_cash_generation import (
    build_sec_companyfacts_derived_cash_generation_observations,
)
from market_engine.fundamental_observations.sec_companyfacts_observations import (
    build_sec_companyfacts_fundamental_observations,
)
from market_engine.source_context.sec_companyfacts_context import (
    build_sec_companyfacts_source_context_from_snapshot_path,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    persist_sec_companyfacts_raw_snapshot,
)
from market_engine.setup_detection.sec_companyfacts_setup_detection import (
    SecCompanyFactsSetupCategory,
    SecCompanyFactsSetupDetectionItem,
    SecCompanyFactsSetupState,
    build_sec_companyfacts_setup_detection,
)


def test_builds_analysis_review_from_complete_positive_observations(tmp_path):
    fundamental_set, derived_set = _upstream_observation_sets(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )

    analysis_review = build_sec_companyfacts_analysis_review(fundamental_set, derived_set)
    review_items_by_category = _review_items_by_category(analysis_review)

    assert analysis_review.ticker == "NVDA"
    assert analysis_review.cik == "0001045810"
    assert analysis_review.provider_name == "SEC_COMPANYFACTS"
    assert (
        analysis_review.analysis_review_format_version
        == SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION
    )
    assert (
        analysis_review.fundamental_observation_format_version
        == fundamental_set.observation_format_version
    )
    assert (
        analysis_review.derived_observation_format_version
        == derived_set.derived_observation_format_version
    )
    assert analysis_review.source_context_format_version == fundamental_set.source_context_format_version
    assert analysis_review.source_context_state == "AVAILABLE"
    assert analysis_review.source_refresh_snapshot_id == "nvda_companyfacts"
    assert analysis_review.source_refresh_fetched_at == "2026-06-15T12:00:00Z"

    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.SOURCE_AVAILABILITY_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.SOURCE_HEALTHY
    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.OBSERVATIONS_COMPLETE
    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.CASH_GENERATION_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.CASH_GENERATION_POSITIVE
    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.FREE_CASH_FLOW_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.CASH_GENERATION_POSITIVE
    assert SecCompanyFactsAnalysisReviewCategory.DATA_LIMITATION_REVIEW not in review_items_by_category
    assert SecCompanyFactsAnalysisReviewCategory.HUMAN_REVIEW_REQUIREMENT not in review_items_by_category


def test_builds_negative_cash_generation_review(tmp_path):
    fundamental_set, derived_set = _upstream_observation_sets(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(5, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(30, "2025-12-31")],
            }
        ),
    )

    analysis_review = build_sec_companyfacts_analysis_review(fundamental_set, derived_set)
    review_items_by_category = _review_items_by_category(analysis_review)

    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.CASH_GENERATION_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEGATIVE
    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.FREE_CASH_FLOW_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEGATIVE


def test_builds_neutral_cash_generation_review(tmp_path):
    fundamental_set, derived_set = _upstream_observation_sets(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(10, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(10, "2025-12-31")],
            }
        ),
    )

    analysis_review = build_sec_companyfacts_analysis_review(fundamental_set, derived_set)
    review_items_by_category = _review_items_by_category(analysis_review)

    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.CASH_GENERATION_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEUTRAL
    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.FREE_CASH_FLOW_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEUTRAL


def test_missing_capex_emits_limitation_and_human_review(tmp_path):
    fundamental_set, derived_set = _upstream_observation_sets(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
            }
        ),
    )

    analysis_review = build_sec_companyfacts_analysis_review(fundamental_set, derived_set)
    review_items_by_category = _review_items_by_category(analysis_review)

    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.SOURCE_AVAILABILITY_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.SOURCE_LIMITED
    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.OBSERVATIONS_LIMITED
    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.CASH_GENERATION_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.DATA_LIMITED
    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.FREE_CASH_FLOW_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.DATA_LIMITED
    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.DATA_LIMITATION_REVIEW
    ].state == SecCompanyFactsAnalysisReviewState.DATA_LIMITED
    assert review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.HUMAN_REVIEW_REQUIREMENT
    ].state == SecCompanyFactsAnalysisReviewState.REQUIRES_HUMAN_REVIEW


def test_analysis_review_preserves_upstream_references(tmp_path):
    fundamental_set, derived_set = _upstream_observation_sets(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )

    analysis_review = build_sec_companyfacts_analysis_review(fundamental_set, derived_set)
    review_items_by_category = _review_items_by_category(analysis_review)

    source_availability = review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.SOURCE_AVAILABILITY_REVIEW
    ]
    free_cash_flow = review_items_by_category[
        SecCompanyFactsAnalysisReviewCategory.FREE_CASH_FLOW_REVIEW
    ]

    assert set(source_availability.source_observation_references) == {
        "SOURCE_CONTEXT_AVAILABILITY"
    }
    assert source_availability.source_observation_references[
        "SOURCE_CONTEXT_AVAILABILITY"
    ]["state"] == "PRESENT"

    assert set(free_cash_flow.derived_observation_references) == {
        "FREE_CASH_FLOW_DERIVATION"
    }
    assert free_cash_flow.derived_observation_references[
        "FREE_CASH_FLOW_DERIVATION"
    ]["derived_values"] == {"free_cash_flow": 25}


def test_setup_detection_input_creates_setup_aware_analysis_review(tmp_path):
    fundamental_set, derived_set, setup_detection = _upstream_with_setup_detection(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )

    analysis_review = build_sec_companyfacts_analysis_review(
        fundamental_set,
        derived_set,
        setup_detection,
    )
    setup_reviews = _review_items_by_setup_category(analysis_review)

    assert analysis_review.analysis_review_format_version == (
        SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION
    )
    assert analysis_review.setup_detection_format_version == (
        setup_detection.setup_detection_format_version
    )
    assert analysis_review.setup_detection_run_id == "setup-detection-run"
    assert setup_reviews["cash_generation_setup"].state == (
        SecCompanyFactsAnalysisReviewState.SETUP_DETECTED
    )
    assert setup_reviews["cash_generation_setup"].category == (
        SecCompanyFactsAnalysisReviewCategory.SETUP_DETECTION_REVIEW
    )
    assert setup_reviews["cash_generation_setup"].setup_states == ("setup_detected",)
    assert "FREE_CASH_FLOW_DERIVATION" in (
        setup_reviews["cash_generation_setup"].derived_observation_references
    )


def test_partial_setup_input_creates_partial_setup_review(tmp_path):
    fundamental_set, derived_set, setup_detection = _upstream_with_setup_detection(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
            }
        ),
    )

    analysis_review = build_sec_companyfacts_analysis_review(
        fundamental_set,
        derived_set,
        setup_detection,
    )
    setup_reviews = _review_items_by_setup_category(analysis_review)

    assert setup_reviews["fundamental_availability_setup"].state == (
        SecCompanyFactsAnalysisReviewState.SETUP_PARTIALLY_DETECTED
    )
    assert _has_setup_human_review_item(analysis_review)


def test_missing_setup_evidence_creates_blocked_review(tmp_path):
    fundamental_set, derived_set, setup_detection = _upstream_with_setup_detection(
        tmp_path,
        _payload({"Revenues": [_fact(100, "2025-12-31")]}),
    )

    analysis_review = build_sec_companyfacts_analysis_review(
        fundamental_set,
        derived_set,
        setup_detection,
    )
    setup_reviews = _review_items_by_setup_category(analysis_review)

    assert setup_reviews["cash_generation_setup"].state == (
        SecCompanyFactsAnalysisReviewState.SETUP_BLOCKED_BY_MISSING_DATA
    )
    assert setup_reviews["cash_generation_setup"].missing_observations
    assert _has_setup_human_review_item(analysis_review)


def test_conflicted_setup_input_creates_conflicted_review_and_human_review(tmp_path):
    fundamental_set, derived_set, setup_detection = _upstream_with_setup_detection(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(5, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(30, "2025-12-31")],
            }
        ),
    )

    analysis_review = build_sec_companyfacts_analysis_review(
        fundamental_set,
        derived_set,
        setup_detection,
    )
    setup_reviews = _review_items_by_setup_category(analysis_review)

    assert setup_reviews["cash_generation_setup"].state == (
        SecCompanyFactsAnalysisReviewState.SETUP_CONFLICTED
    )
    assert _has_setup_human_review_item(analysis_review)


def test_not_assessed_setup_input_remains_not_assessed(tmp_path):
    fundamental_set, derived_set, setup_detection = _upstream_with_setup_detection(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )
    not_assessed_item = _setup_item(
        category=SecCompanyFactsSetupCategory.NOT_ASSESSED_SETUP,
        state=SecCompanyFactsSetupState.SETUP_NOT_ASSESSED,
        message="Setup Detection could not assess the pattern from approved inputs.",
        required_observations=("APPROVED_SETUP_INPUT",),
        missing_observations=("APPROVED_SETUP_INPUT",),
    )
    setup_detection = replace(
        setup_detection,
        setup_items=setup_detection.setup_items + (not_assessed_item,),
    )

    analysis_review = build_sec_companyfacts_analysis_review(
        fundamental_set,
        derived_set,
        setup_detection,
    )
    setup_reviews = _review_items_by_setup_category(analysis_review)

    assert setup_reviews["not_assessed_setup"].state == (
        SecCompanyFactsAnalysisReviewState.SETUP_NOT_ASSESSED
    )
    assert _has_setup_human_review_item(analysis_review)


def test_unsupported_setup_detection_contract_fails_closed(tmp_path):
    fundamental_set, derived_set, setup_detection = _upstream_with_setup_detection(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )
    setup_detection = replace(setup_detection, setup_detection_format_version="bad-v1")

    try:
        build_sec_companyfacts_analysis_review(
            fundamental_set,
            derived_set,
            setup_detection,
        )
    except ValueError as error:
        assert "unsupported SEC CompanyFacts Setup Detection contract: bad-v1" in str(error)
    else:
        raise AssertionError("expected ValueError for unsupported Setup Detection contract")


def test_setup_detection_alignment_mismatch_fails_safely(tmp_path):
    fundamental_set, derived_set, setup_detection = _upstream_with_setup_detection(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )
    setup_detection = replace(setup_detection, ticker="AMD")

    try:
        build_sec_companyfacts_analysis_review(
            fundamental_set,
            derived_set,
            setup_detection,
        )
    except ValueError as error:
        assert "analysis review and setup detection inputs do not align on: ticker" in str(error)
    else:
        raise AssertionError("expected ValueError for mismatched Setup Detection input")


def test_numeric_zero_from_setup_detection_is_preserved(tmp_path):
    fundamental_set, derived_set, setup_detection = _upstream_with_setup_detection(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(0, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(10, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(10, "2025-12-31")],
            }
        ),
    )

    analysis_review = build_sec_companyfacts_analysis_review(
        fundamental_set,
        derived_set,
        setup_detection,
    )
    setup_reviews = _review_items_by_setup_category(analysis_review)

    assert setup_reviews["cash_generation_setup"].setup_evidence["derived_values"][
        "free_cash_flow"
    ] == 0
    assert setup_reviews["profitability_evidence_setup"].setup_evidence[
        "net_income"
    ] == 0


def test_persistence_preserves_setup_detection_references(tmp_path):
    fundamental_set, derived_set, setup_detection = _upstream_with_setup_detection(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )
    analysis_review = build_sec_companyfacts_analysis_review(
        fundamental_set,
        derived_set,
        setup_detection,
    )

    analysis_review_path = persist_sec_companyfacts_analysis_review(
        analysis_review,
        run_id="analysis-review-run",
        root_dir=tmp_path / "analysis_reviews",
    )
    payload = json.loads(analysis_review_path.read_text(encoding="utf-8"))

    assert payload["setup_detection_format_version"] == (
        setup_detection.setup_detection_format_version
    )
    assert payload["setup_detection_run_id"] == "setup-detection-run"
    setup_review_payloads = [
        item
        for item in payload["review_items"]
        if item["input_observation_families"] == ["ME-SD"]
    ]
    assert setup_review_payloads
    assert setup_review_payloads[0]["setup_detection_references"]


def test_alignment_mismatch_fails_safely(tmp_path):
    fundamental_set, derived_set = _upstream_observation_sets(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )
    mismatched_derived_set = replace(derived_set, ticker="AMD")

    try:
        build_sec_companyfacts_analysis_review(fundamental_set, mismatched_derived_set)
    except ValueError as error:
        assert "fundamental and derived observation sets do not align on: ticker" in str(error)
    else:
        raise AssertionError("expected ValueError for mismatched upstream observation sets")


def test_persistence_writes_json_without_overwrite(tmp_path):
    fundamental_set, derived_set = _upstream_observation_sets(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )
    analysis_review = build_sec_companyfacts_analysis_review(fundamental_set, derived_set)

    analysis_review_path = persist_sec_companyfacts_analysis_review(
        analysis_review,
        run_id="analysis-review-run",
        root_dir=tmp_path / "analysis_reviews",
    )

    assert (
        analysis_review_path
        == tmp_path / "analysis_reviews" / "analysis-review-run" / "NVDA" / "analysis_review.json"
    )

    payload = json.loads(analysis_review_path.read_text(encoding="utf-8"))
    assert payload["ticker"] == "NVDA"
    assert payload["analysis_review_format_version"] == SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION
    assert payload["source_context_state"] == "AVAILABLE"
    assert payload["review_items"][0]["category"] == "SOURCE_AVAILABILITY_REVIEW"

    try:
        persist_sec_companyfacts_analysis_review(
            analysis_review,
            run_id="analysis-review-run",
            root_dir=tmp_path / "analysis_reviews",
        )
    except FileExistsError as error:
        assert "refusing to overwrite existing SEC CompanyFacts Analysis Review" in str(error)
    else:
        raise AssertionError("expected FileExistsError for existing analysis review output")


def test_analysis_review_does_not_emit_recommendation_or_decision_authority(tmp_path):
    fundamental_set, derived_set = _upstream_observation_sets(
        tmp_path,
        _payload(
            {
                "Revenues": [_fact(100, "2025-12-31")],
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
            }
        ),
    )

    analysis_review = build_sec_companyfacts_analysis_review(fundamental_set, derived_set)
    payload = asdict(analysis_review)

    forbidden_authority_fields = {
        "BUY",
        "SELL",
        "HOLD",
        "recommendation",
        "rating",
        "score",
        "rank",
        "ranking",
        "conviction",
        "urgency",
        "tradeability",
        "allocation",
        "position_size",
        "position_sizing",
        "execution",
        "target_price",
        "portfolio_action",
        "decision",
        "telegram",
        "delivery",
        "report_instruction",
    }

    assert forbidden_authority_fields.isdisjoint(payload)
    for review_item in payload["review_items"]:
        assert forbidden_authority_fields.isdisjoint(review_item)
        assert all(
            forbidden_term not in review_item["message"].lower()
            for forbidden_term in forbidden_authority_fields
        )


def test_analysis_review_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


def _upstream_observation_sets(tmp_path, raw_payload: dict[str, object]):
    snapshot_path = persist_sec_companyfacts_raw_snapshot(
        raw_payload=raw_payload,
        ticker="NVDA",
        cik="0001045810",
        run_id="source-refresh-run",
        snapshot_id="nvda_companyfacts",
        fetched_at="2026-06-15T12:00:00Z",
        root_dir=tmp_path / "source_snapshots",
    )
    source_context = build_sec_companyfacts_source_context_from_snapshot_path(snapshot_path)
    fundamental_set = build_sec_companyfacts_fundamental_observations(source_context)
    derived_set = build_sec_companyfacts_derived_cash_generation_observations(
        fundamental_set
    )
    return fundamental_set, derived_set


def _upstream_with_setup_detection(tmp_path, raw_payload: dict[str, object]):
    fundamental_set, derived_set = _upstream_observation_sets(tmp_path, raw_payload)
    setup_detection = build_sec_companyfacts_setup_detection(
        fundamental_set,
        derived_set,
        setup_detection_run_id="setup-detection-run",
    )
    return fundamental_set, derived_set, setup_detection


def _review_items_by_category(analysis_review):
    return {
        review_item.category: review_item
        for review_item in analysis_review.review_items
    }


def _review_items_by_setup_category(analysis_review):
    review_items = {}
    for review_item in analysis_review.review_items:
        for setup_category in review_item.setup_categories:
            review_items.setdefault(setup_category, review_item)
    return review_items


def _has_setup_human_review_item(analysis_review):
    return any(
        review_item.category
        == SecCompanyFactsAnalysisReviewCategory.SETUP_HUMAN_REVIEW_REQUIREMENT
        and review_item.state
        == SecCompanyFactsAnalysisReviewState.SETUP_REQUIRES_HUMAN_REVIEW
        for review_item in analysis_review.review_items
    )


def _setup_item(
    *,
    category: SecCompanyFactsSetupCategory,
    state: SecCompanyFactsSetupState,
    message: str,
    required_observations: tuple[str, ...],
    missing_observations: tuple[str, ...],
):
    return SecCompanyFactsSetupDetectionItem(
        category=category,
        state=state,
        message=message,
        input_observation_families=("ME-SD",),
        required_observations=required_observations,
        missing_observations=missing_observations,
        source_observation_references={},
        derived_observation_references={},
        setup_evidence={},
        setup_limitations=("manual synthetic test item",),
    )


def _payload(facts: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": values}}
                for tag, values in facts.items()
            }
        }
    }


def _fact(value: int | None, end: str) -> dict[str, object]:
    return {
        "val": value,
        "fy": int(end[:4]),
        "fp": "FY",
        "form": "10-K",
        "filed": f"{int(end[:4]) + 1}-02-15",
        "start": f"{end[:4]}-01-01",
        "end": end,
        "accn": f"0000000000-{end[:4]}-000001",
        "frame": f"CY{end[:4]}",
    }
