from __future__ import annotations

from dataclasses import asdict

import pytest

from market_engine.source_intake.fake_provider import FakeProviderScenario, FakeSourceProvider
from market_engine.source_intake.readiness import SourceReadinessStatus
from market_engine.source_intake.runner import run_source_intake


REQUIRED_FIELDS = ("revenue", "operating_cash_flow", "capital_expenditures")


def test_full_data_returns_available():
    summary = run_source_intake(
        tickers=["AAPL"],
        provider=FakeSourceProvider(
            scenarios={
                "AAPL": FakeProviderScenario(
                    fields={
                        "revenue": 100,
                        "operating_cash_flow": 25,
                        "capital_expenditures": 5,
                    },
                    raw_evidence={"ticker": "AAPL"},
                    raw_evidence_summary="complete",
                )
            }
        ),
        required_fields=REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.AVAILABLE
    assert result.available_fields == REQUIRED_FIELDS
    assert result.missing_fields == ()
    assert result.raw_evidence_present is True
    assert result.intake_success is True


def test_partial_data_returns_partial_and_preserves_missing_fields():
    summary = run_source_intake(
        tickers=["PARTIAL"],
        provider=FakeSourceProvider(
            scenarios={
                "PARTIAL": FakeProviderScenario(
                    fields={
                        "revenue": 100,
                        "operating_cash_flow": None,
                    }
                )
            }
        ),
        required_fields=REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.PARTIAL
    assert result.available_fields == ("revenue",)
    assert result.missing_fields == ("operating_cash_flow", "capital_expenditures")
    assert result.normalized_data["operating_cash_flow"] is None
    assert "capital_expenditures" not in result.normalized_data


def test_missing_source_returns_missing():
    summary = run_source_intake(
        tickers=["MISSING"],
        provider=FakeSourceProvider(scenarios={"MISSING": FakeProviderScenario(missing_source=True)}),
        required_fields=REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.MISSING
    assert result.missing_fields == REQUIRED_FIELDS
    assert result.intake_success is False


def test_unsupported_ticker_returns_unsupported():
    summary = run_source_intake(
        tickers=["UNSUPPORTED"],
        provider=FakeSourceProvider(scenarios={"UNSUPPORTED": FakeProviderScenario(unsupported=True)}),
        required_fields=REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.UNSUPPORTED
    assert result.error is not None
    assert result.error.error_type == "UnsupportedTickerError"


def test_invalid_ticker_returns_invalid_ticker():
    summary = run_source_intake(
        tickers=["BAD TICKER"],
        provider=FakeSourceProvider(scenarios={"BAD TICKER": FakeProviderScenario(invalid=True)}),
        required_fields=REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.INVALID_TICKER
    assert result.error is not None
    assert result.error.error_type == "InvalidTickerError"


def test_provider_exception_returns_provider_error():
    summary = run_source_intake(
        tickers=["ERROR"],
        provider=FakeSourceProvider(scenarios={"ERROR": FakeProviderScenario(provider_error=True)}),
        required_fields=REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.readiness_status == SourceReadinessStatus.PROVIDER_ERROR
    assert result.error is not None
    assert result.error.error_type == "ProviderUnavailableError"


def test_batch_continues_after_provider_error():
    summary = run_source_intake(
        tickers=["ERROR", "AAPL"],
        provider=FakeSourceProvider(
            scenarios={
                "ERROR": FakeProviderScenario(provider_error=True),
                "AAPL": FakeProviderScenario(
                    fields={
                        "revenue": 100,
                        "operating_cash_flow": 25,
                        "capital_expenditures": 5,
                    }
                ),
            }
        ),
        required_fields=REQUIRED_FIELDS,
    )

    assert [result.readiness_status for result in summary.results] == [
        SourceReadinessStatus.PROVIDER_ERROR,
        SourceReadinessStatus.AVAILABLE,
    ]
    assert summary.total_tickers == 2


def test_summary_counts_by_readiness_status_are_correct():
    summary = run_source_intake(
        tickers=["FULL", "PARTIAL", "MISSING", "UNSUPPORTED", "INVALID", "ERROR"],
        provider=FakeSourceProvider(
            scenarios={
                "FULL": FakeProviderScenario(
                    fields={
                        "revenue": 100,
                        "operating_cash_flow": 25,
                        "capital_expenditures": 5,
                    }
                ),
                "PARTIAL": FakeProviderScenario(fields={"revenue": 100}),
                "MISSING": FakeProviderScenario(missing_source=True),
                "UNSUPPORTED": FakeProviderScenario(unsupported=True),
                "INVALID": FakeProviderScenario(invalid=True),
                "ERROR": FakeProviderScenario(provider_error=True),
            }
        ),
        required_fields=REQUIRED_FIELDS,
    )

    assert summary.status_counts == {
        SourceReadinessStatus.AVAILABLE: 1,
        SourceReadinessStatus.PARTIAL: 1,
        SourceReadinessStatus.MISSING: 1,
        SourceReadinessStatus.UNSUPPORTED: 1,
        SourceReadinessStatus.INVALID_TICKER: 1,
        SourceReadinessStatus.PROVIDER_ERROR: 1,
    }
    assert summary.intake_success_count == 2
    assert summary.intake_failure_count == 4


def test_missing_field_frequency_is_recorded_correctly():
    summary = run_source_intake(
        tickers=["ONE", "TWO"],
        provider=FakeSourceProvider(
            scenarios={
                "ONE": FakeProviderScenario(fields={"revenue": 100}),
                "TWO": FakeProviderScenario(fields={"revenue": 200, "operating_cash_flow": None}),
            }
        ),
        required_fields=REQUIRED_FIELDS,
    )

    assert summary.missing_field_frequency == {
        "operating_cash_flow": 2,
        "capital_expenditures": 2,
    }


def test_missing_numeric_data_is_not_converted_to_zero():
    summary = run_source_intake(
        tickers=["PARTIAL"],
        provider=FakeSourceProvider(
            scenarios={
                "PARTIAL": FakeProviderScenario(
                    fields={
                        "revenue": 100,
                        "operating_cash_flow": None,
                        "capital_expenditures": 0,
                    }
                )
            }
        ),
        required_fields=REQUIRED_FIELDS,
    )

    result = summary.results[0]
    assert result.normalized_data["operating_cash_flow"] is None
    assert result.normalized_data["capital_expenditures"] == 0
    assert result.available_fields == ("revenue", "capital_expenditures")
    assert result.missing_fields == ("operating_cash_flow",)


def test_results_do_not_contain_forbidden_authority_fields():
    summary = run_source_intake(
        tickers=["AAPL"],
        provider=FakeSourceProvider(
            scenarios={
                "AAPL": FakeProviderScenario(
                    fields={
                        "revenue": 100,
                        "operating_cash_flow": 25,
                        "capital_expenditures": 5,
                    }
                )
            }
        ),
        required_fields=REQUIRED_FIELDS,
    )

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
    result_payload = asdict(summary.results[0])
    summary_payload = asdict(summary)

    assert forbidden_fields.isdisjoint(result_payload)
    assert forbidden_fields.isdisjoint(summary_payload)
    assert forbidden_fields.isdisjoint(result_payload["normalized_data"])


def test_tests_use_fake_provider_only():
    provider = FakeSourceProvider(scenarios={})
    assert provider.name == "fake-source-provider"


def test_tests_do_not_import_legacy_runtime_modules():
    assert "market_scanner" not in globals()
    assert "scripts" not in globals()


def test_empty_ticker_list_returns_clean_empty_summary():
    summary = run_source_intake(
        tickers=[],
        provider=FakeSourceProvider(scenarios={}),
        required_fields=REQUIRED_FIELDS,
    )

    assert summary.total_tickers == 0
    assert summary.results == ()
    assert summary.status_counts == {}
    assert summary.missing_field_frequency == {}
