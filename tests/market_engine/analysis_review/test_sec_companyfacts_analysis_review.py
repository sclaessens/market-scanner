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


def _review_items_by_category(analysis_review):
    return {
        review_item.category: review_item
        for review_item in analysis_review.review_items
    }


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