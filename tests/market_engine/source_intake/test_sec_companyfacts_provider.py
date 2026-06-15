from __future__ import annotations

from dataclasses import asdict

import pytest

from market_engine.source_intake.coverage_review import build_source_coverage_review
from market_engine.source_intake.readiness import SourceReadinessStatus
from market_engine.source_intake.runner import run_source_intake
from market_engine.source_intake.sec_companyfacts_provider import (
    SEC_COMPANYFACTS_REQUIRED_FIELDS,
    SecCompanyFactsProvider,
)


def test_sec_provider_maps_full_mocked_companyfacts_to_available_fields():
    provider = SecCompanyFactsProvider(fetch_json=lambda url: _companyfacts_payload())

    summary = run_source_intake(
        tickers=["NVDA"],
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.AVAILABLE
    assert result.available_fields == SEC_COMPANYFACTS_REQUIRED_FIELDS
    assert result.normalized_data == {
        "revenue": 100,
        "net_income": 20,
        "operating_cash_flow": 30,
        "capital_expenditures": 5,
    }
    assert result.raw_evidence_present is True


def test_sec_provider_partial_mocked_response_preserves_missing_fields():
    provider = SecCompanyFactsProvider(fetch_json=lambda url: _companyfacts_payload(include_capex=False))

    summary = run_source_intake(
        tickers=["NVDA"],
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.PARTIAL
    assert result.missing_fields == ("capital_expenditures",)
    assert result.normalized_data["capital_expenditures"] is None


def test_sec_provider_missing_companyfacts_response_returns_missing():
    provider = SecCompanyFactsProvider(fetch_json=lambda url: None)

    summary = run_source_intake(
        tickers=["NVDA"],
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.MISSING
    assert result.missing_fields == SEC_COMPANYFACTS_REQUIRED_FIELDS


def test_sec_provider_no_cik_ticker_returns_unsupported():
    provider = SecCompanyFactsProvider(fetch_json=lambda url: _companyfacts_payload())

    summary = run_source_intake(
        tickers=["NOPE"],
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.UNSUPPORTED
    assert result.error is not None


def test_sec_provider_invalid_ticker_returns_invalid_ticker():
    provider = SecCompanyFactsProvider(fetch_json=lambda url: _companyfacts_payload())

    summary = run_source_intake(
        tickers=["BAD TICKER"],
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.INVALID_TICKER
    assert result.error is not None


def test_sec_provider_network_error_is_captured_as_provider_error():
    def fail_fetch(url: str) -> dict[str, object]:
        raise TimeoutError("network unavailable")

    provider = SecCompanyFactsProvider(fetch_json=fail_fetch)

    summary = run_source_intake(
        tickers=["NVDA"],
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.PROVIDER_ERROR
    assert result.error is not None
    assert result.error.error_type == "ProviderUnavailableError"


def test_sec_provider_missing_numeric_fields_are_not_converted_to_zero():
    provider = SecCompanyFactsProvider(
        fetch_json=lambda url: _companyfacts_payload(operating_cash_flow=None, capex=0)
    )

    summary = run_source_intake(
        tickers=["NVDA"],
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.normalized_data["operating_cash_flow"] is None
    assert result.normalized_data["capital_expenditures"] == 0
    assert result.missing_fields == ("operating_cash_flow",)


def test_sec_provider_import_does_not_call_provider():
    calls: list[str] = []
    provider = SecCompanyFactsProvider(fetch_json=lambda url: calls.append(url) or _companyfacts_payload())

    assert calls == []
    provider.fetch_source("NVDA")
    assert len(calls) == 1


def test_real_provider_tests_do_not_require_network():
    def fail_if_called(url: str) -> dict[str, object]:
        raise AssertionError("test should control provider calls")

    provider = SecCompanyFactsProvider(fetch_json=fail_if_called)

    with pytest.raises(AssertionError):
        provider.fetch_source("NVDA")


def test_source_coverage_review_summarizes_readiness_counts():
    provider = SecCompanyFactsProvider(
        fetch_json=lambda url: _companyfacts_payload(include_capex=False)
    )

    summary = run_source_intake(
        tickers=["NVDA", "NOPE"],
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )
    review = build_source_coverage_review(summary)

    assert review.readiness_counts == {
        "PARTIAL": 1,
        "UNSUPPORTED": 1,
    }
    assert review.unsupported_count == 1
    assert review.failed_or_unsupported_tickers == ("NOPE",)


def test_source_coverage_review_includes_missing_field_frequency():
    provider = SecCompanyFactsProvider(
        fetch_json=lambda url: _companyfacts_payload(include_capex=False)
    )

    summary = run_source_intake(
        tickers=["NVDA"],
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )
    review = build_source_coverage_review(summary)

    assert review.missing_field_frequency == {"capital_expenditures": 1}
    assert review.top_missing_fields == (("capital_expenditures", 1),)


def test_source_coverage_review_does_not_include_forbidden_authority_fields():
    provider = SecCompanyFactsProvider(fetch_json=lambda url: _companyfacts_payload())
    summary = run_source_intake(
        tickers=["NVDA"],
        provider=provider,
        required_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
    )
    review_payload = asdict(build_source_coverage_review(summary))

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
    }

    assert forbidden_fields.isdisjoint(review_payload)


def test_sec_provider_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


def _companyfacts_payload(
    *,
    revenue: int | None = 100,
    net_income: int | None = 20,
    operating_cash_flow: int | None = 30,
    capex: int | None = 5,
    include_capex: bool = True,
) -> dict[str, object]:
    facts: dict[str, object] = {
        "Revenues": {"units": {"USD": [_fact(revenue, "2025-12-31")]}},
        "NetIncomeLoss": {"units": {"USD": [_fact(net_income, "2025-12-31")]}},
        "NetCashProvidedByUsedInOperatingActivities": {
            "units": {"USD": [_fact(operating_cash_flow, "2025-12-31")]}
        },
    }
    if include_capex:
        facts["PaymentsToAcquirePropertyPlantAndEquipment"] = {
            "units": {"USD": [_fact(capex, "2025-12-31")]}
        }
    return {"facts": {"us-gaap": facts}}


def _fact(value: int | None, end: str) -> dict[str, object]:
    return {"val": value, "end": end}
