from __future__ import annotations

from dataclasses import asdict

from market_engine.fundamentals.source_context import build_sec_fundamental_source_context
from market_engine.source_intake.models import SourceIntakeError, TickerSourceResult
from market_engine.source_intake.provider_boundary import ProviderSourceResponse
from market_engine.source_intake.readiness import SourceReadinessStatus
from market_engine.source_intake.sec_companyfacts_fields import SEC_COMPANYFACTS_PROVIDER_NAME


def test_all_four_mapped_fields_create_available_source_context():
    context = build_sec_fundamental_source_context(
        ticker="NVDA",
        response=_response("NVDA", _complete_payload()),
    )

    assert context.source_status == SourceReadinessStatus.AVAILABLE
    assert context.missing_canonical_fields == ()
    assert context.canonical_fields == {
        "revenue": 100,
        "net_income": 20,
        "operating_cash_flow": 30,
        "capital_expenditures": 5,
    }


def test_one_required_field_missing_creates_partial_source_context():
    context = build_sec_fundamental_source_context(
        ticker="NVDA",
        response=_response(
            "NVDA",
            _payload(
                {
                    "Revenues": [_fact(100, "2025-12-31")],
                    "NetIncomeLoss": [_fact(20, "2025-12-31")],
                    "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                }
            ),
        ),
    )

    assert context.source_status == SourceReadinessStatus.PARTIAL
    assert context.missing_canonical_fields == ("capital_expenditures",)
    assert context.canonical_fields["capital_expenditures"] is None


def test_no_approved_fields_creates_missing_source_context():
    context = build_sec_fundamental_source_context(
        ticker="NVDA",
        response=_response("NVDA", _payload({"GrossProfit": [_fact(40, "2025-12-31")]})),
    )

    assert context.source_status == SourceReadinessStatus.MISSING
    assert context.missing_canonical_fields == (
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capital_expenditures",
    )


def test_unsupported_invalid_and_provider_failure_results_are_preserved():
    unsupported = build_sec_fundamental_source_context(
        ticker="NOPE",
        source_result=_terminal_result("NOPE", SourceReadinessStatus.UNSUPPORTED, "UnsupportedTickerError"),
    )
    invalid = build_sec_fundamental_source_context(
        ticker="BAD TICKER",
        source_result=_terminal_result("BAD TICKER", SourceReadinessStatus.INVALID_TICKER, "InvalidTickerError"),
    )
    failed = build_sec_fundamental_source_context(
        ticker="NVDA",
        source_result=_terminal_result(
            "NVDA",
            SourceReadinessStatus.PROVIDER_ERROR,
            "SecCompanyFactsNetworkError",
        ),
    )

    assert unsupported.source_status == SourceReadinessStatus.UNSUPPORTED
    assert invalid.source_status == SourceReadinessStatus.INVALID_TICKER
    assert failed.source_status == SourceReadinessStatus.PROVIDER_ERROR
    assert failed.provider_error_category == "SecCompanyFactsNetworkError"
    assert failed.provider_error_message == "controlled failure"


def test_missing_numeric_values_are_not_converted_to_zero():
    context = build_sec_fundamental_source_context(
        ticker="NVDA",
        response=_response("NVDA", _payload({"Revenues": [_fact(0, "2025-12-31")]})),
    )

    assert context.canonical_fields["revenue"] == 0
    assert context.canonical_fields["net_income"] is None
    assert context.canonical_fields["operating_cash_flow"] is None
    assert context.canonical_fields["capital_expenditures"] is None
    assert context.source_status == SourceReadinessStatus.PARTIAL


def test_context_contains_provenance_and_period_metadata_for_available_fields():
    context = build_sec_fundamental_source_context(
        ticker="NVDA",
        response=_response("NVDA", _complete_payload()),
    )

    revenue = context.provenance["revenue"]
    assert revenue.sec_tag_selected == "Revenues"
    assert revenue.provider_name == SEC_COMPANYFACTS_PROVIDER_NAME
    assert revenue.unit == "USD"
    assert revenue.raw_value == 100
    assert context.period_metadata["revenue"] == {
        "fiscal_year": 2025,
        "fiscal_period": "FY",
        "filing_form": "10-K",
        "filing_date": "2026-02-15",
        "period_start_date": "2025-01-01",
        "period_end_date": "2025-12-31",
        "accession_number": "0000000000-2025-000001",
        "frame": "CY2025",
    }


def test_context_does_not_emit_forbidden_authority_or_analysis_fields():
    context = build_sec_fundamental_source_context(
        ticker="NVDA",
        response=_response("NVDA", _complete_payload()),
    )
    payload = asdict(context)

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
        "free_cash_flow",
        "growth",
        "margin",
    }
    assert forbidden_fields.isdisjoint(payload)


def test_source_context_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


def _response(ticker: str, payload: dict[str, object]) -> ProviderSourceResponse:
    return ProviderSourceResponse(
        ticker=ticker,
        fields={},
        raw_evidence=payload,
        raw_evidence_summary="mocked SEC CompanyFacts payload",
    )


def _terminal_result(
    ticker: str,
    status: SourceReadinessStatus,
    error_type: str,
) -> TickerSourceResult:
    return TickerSourceResult(
        ticker=ticker,
        provider_name=SEC_COMPANYFACTS_PROVIDER_NAME,
        readiness_status=status,
        missing_fields=(
            "revenue",
            "net_income",
            "operating_cash_flow",
            "capital_expenditures",
        ),
        error=SourceIntakeError(error_type=error_type, message="controlled failure"),
    )


def _complete_payload() -> dict[str, object]:
    return _payload(
        {
            "Revenues": [_fact(100, "2025-12-31")],
            "NetIncomeLoss": [_fact(20, "2025-12-31")],
            "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
            "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
        }
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
