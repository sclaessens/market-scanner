from __future__ import annotations

from dataclasses import asdict

from market_engine.fundamentals.analysis_pass import (
    FundamentalObservationCategory,
    FundamentalObservationState,
    build_fundamental_analysis_pass,
)
from market_engine.fundamentals.source_context import build_sec_fundamental_source_context
from market_engine.source_intake.models import SourceIntakeError, TickerSourceResult
from market_engine.source_intake.provider_boundary import ProviderSourceResponse
from market_engine.source_intake.readiness import SourceReadinessStatus
from market_engine.source_intake.sec_companyfacts_fields import SEC_COMPANYFACTS_PROVIDER_NAME


def test_available_source_context_produces_positive_source_readiness_observation():
    analysis = build_fundamental_analysis_pass(_available_context())

    observation = _observation(analysis, FundamentalObservationCategory.SOURCE_READINESS)
    assert observation.state == FundamentalObservationState.POSITIVE
    assert observation.message == "All required source fields are available for the selected period."


def test_partial_source_context_produces_missing_data_source_readiness_observation():
    analysis = build_fundamental_analysis_pass(
        _context(
            revenue=100,
            net_income=20,
            operating_cash_flow=30,
            capital_expenditures=None,
        )
    )

    observation = _observation(analysis, FundamentalObservationCategory.SOURCE_READINESS)
    assert observation.state == FundamentalObservationState.MISSING_DATA


def test_missing_source_context_produces_missing_data_source_readiness_observation():
    analysis = build_fundamental_analysis_pass(_context())

    observation = _observation(analysis, FundamentalObservationCategory.SOURCE_READINESS)
    assert observation.state == FundamentalObservationState.MISSING_DATA


def test_provider_error_context_produces_not_assessed_source_readiness_observation():
    analysis = build_fundamental_analysis_pass(_terminal_context(SourceReadinessStatus.PROVIDER_ERROR))

    observation = _observation(analysis, FundamentalObservationCategory.SOURCE_READINESS)
    assert observation.state == FundamentalObservationState.NOT_ASSESSED


def test_revenue_presence_and_missing_observations():
    present = build_fundamental_analysis_pass(_context(revenue=100))
    missing = build_fundamental_analysis_pass(_context(revenue=None))

    assert _observation(present, FundamentalObservationCategory.REVENUE_PRESENCE).state == (
        FundamentalObservationState.POSITIVE
    )
    assert _observation(missing, FundamentalObservationCategory.REVENUE_PRESENCE).state == (
        FundamentalObservationState.MISSING_DATA
    )


def test_net_income_positive_negative_zero_and_missing_observations():
    positive = build_fundamental_analysis_pass(_context(net_income=20))
    negative = build_fundamental_analysis_pass(_context(net_income=-5))
    zero = build_fundamental_analysis_pass(_context(net_income=0))
    missing = build_fundamental_analysis_pass(_context(net_income=None))

    assert _observation(positive, FundamentalObservationCategory.PROFITABILITY_PRESENCE).state == (
        FundamentalObservationState.POSITIVE
    )
    assert _observation(negative, FundamentalObservationCategory.PROFITABILITY_PRESENCE).state == (
        FundamentalObservationState.NEGATIVE
    )
    assert _observation(zero, FundamentalObservationCategory.PROFITABILITY_PRESENCE).state == (
        FundamentalObservationState.NEUTRAL
    )
    missing_observation = _observation(missing, FundamentalObservationCategory.PROFITABILITY_PRESENCE)
    assert missing_observation.state == FundamentalObservationState.MISSING_DATA
    assert missing_observation.source_values["net_income"] is None


def test_operating_cash_flow_positive_negative_zero_and_missing_observations():
    positive = build_fundamental_analysis_pass(_context(operating_cash_flow=30))
    negative = build_fundamental_analysis_pass(_context(operating_cash_flow=-10))
    zero = build_fundamental_analysis_pass(_context(operating_cash_flow=0))
    missing = build_fundamental_analysis_pass(_context(operating_cash_flow=None))

    assert _observation(positive, FundamentalObservationCategory.OPERATING_CASH_FLOW_PRESENCE).state == (
        FundamentalObservationState.POSITIVE
    )
    assert _observation(negative, FundamentalObservationCategory.OPERATING_CASH_FLOW_PRESENCE).state == (
        FundamentalObservationState.NEGATIVE
    )
    assert _observation(zero, FundamentalObservationCategory.OPERATING_CASH_FLOW_PRESENCE).state == (
        FundamentalObservationState.NEUTRAL
    )
    missing_observation = _observation(missing, FundamentalObservationCategory.OPERATING_CASH_FLOW_PRESENCE)
    assert missing_observation.state == FundamentalObservationState.MISSING_DATA
    assert missing_observation.source_values["operating_cash_flow"] is None


def test_capex_presence_and_missing_observations():
    present = build_fundamental_analysis_pass(_context(capital_expenditures=5))
    missing = build_fundamental_analysis_pass(_context(capital_expenditures=None))

    assert _observation(present, FundamentalObservationCategory.CAPEX_PRESENCE).state == (
        FundamentalObservationState.POSITIVE
    )
    assert _observation(missing, FundamentalObservationCategory.CAPEX_PRESENCE).state == (
        FundamentalObservationState.MISSING_DATA
    )


def test_cash_generation_source_completeness_observations():
    complete = build_fundamental_analysis_pass(
        _context(operating_cash_flow=30, capital_expenditures=5)
    )
    incomplete = build_fundamental_analysis_pass(
        _context(operating_cash_flow=30, capital_expenditures=None)
    )

    assert _observation(
        complete,
        FundamentalObservationCategory.CASH_GENERATION_SOURCE_COMPLETENESS,
    ).state == FundamentalObservationState.POSITIVE
    assert _observation(
        incomplete,
        FundamentalObservationCategory.CASH_GENERATION_SOURCE_COMPLETENESS,
    ).state == FundamentalObservationState.MISSING_DATA


def test_observations_preserve_canonical_field_references_and_provider_grounding():
    analysis = build_fundamental_analysis_pass(_available_context())
    observation = _observation(analysis, FundamentalObservationCategory.REVENUE_PRESENCE)

    assert analysis.ticker == "NVDA"
    assert analysis.provider == SEC_COMPANYFACTS_PROVIDER_NAME
    assert observation.canonical_fields == ("revenue",)
    assert observation.source_values == {"revenue": 100}
    assert observation.source_references["revenue"]["sec_tag_selected"] == "Revenues"
    assert observation.source_references["revenue"]["provider_name"] == SEC_COMPANYFACTS_PROVIDER_NAME


def test_analysis_pass_does_not_emit_forbidden_authority_or_derived_fields():
    analysis = build_fundamental_analysis_pass(_available_context())
    payload = asdict(analysis)

    forbidden_fields = {
        "BUY",
        "SELL",
        "HOLD",
        "recommendation",
        "allocation",
        "ranking",
        "score",
        "conviction",
        "urgency",
        "tradeability",
        "position_sizing",
        "execution",
        "rating",
        "signal",
        "free_cash_flow",
        "growth",
        "margin",
    }
    assert forbidden_fields.isdisjoint(payload)
    for observation in analysis.observations:
        assert forbidden_fields.isdisjoint(asdict(observation))


def test_analysis_pass_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


def _available_context():
    return _context(
        revenue=100,
        net_income=20,
        operating_cash_flow=30,
        capital_expenditures=5,
    )


def _context(
    *,
    revenue: int | None = None,
    net_income: int | None = None,
    operating_cash_flow: int | None = None,
    capital_expenditures: int | None = None,
):
    facts: dict[str, list[dict[str, object]]] = {}
    if revenue is not None:
        facts["Revenues"] = [_fact(revenue, "2025-12-31")]
    if net_income is not None:
        facts["NetIncomeLoss"] = [_fact(net_income, "2025-12-31")]
    if operating_cash_flow is not None:
        facts["NetCashProvidedByUsedInOperatingActivities"] = [_fact(operating_cash_flow, "2025-12-31")]
    if capital_expenditures is not None:
        facts["PaymentsToAcquirePropertyPlantAndEquipment"] = [_fact(capital_expenditures, "2025-12-31")]
    return build_sec_fundamental_source_context(
        ticker="NVDA",
        response=ProviderSourceResponse(
            ticker="NVDA",
            raw_evidence={"facts": {"us-gaap": {tag: {"units": {"USD": values}} for tag, values in facts.items()}}},
            raw_evidence_summary="mocked SEC CompanyFacts payload",
        ),
    )


def _terminal_context(status: SourceReadinessStatus):
    return build_sec_fundamental_source_context(
        ticker="NVDA",
        source_result=TickerSourceResult(
            ticker="NVDA",
            provider_name=SEC_COMPANYFACTS_PROVIDER_NAME,
            readiness_status=status,
            missing_fields=(
                "revenue",
                "net_income",
                "operating_cash_flow",
                "capital_expenditures",
            ),
            error=SourceIntakeError(error_type="ControlledProviderError", message="controlled failure"),
        ),
    )


def _observation(analysis, category: FundamentalObservationCategory):
    return next(observation for observation in analysis.observations if observation.category == category)


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
