import json
from pathlib import Path

from market_scanner.fundamentals import sec_companyfacts_live_smoke
from market_scanner.fundamentals.sec_companyfacts_live_smoke import (
    APPROVED_LIVE_SMOKE_CIK,
    APPROVED_LIVE_SMOKE_TICKER,
    SEC_COMPANYFACTS_ENDPOINT,
    SecCompanyFactsHttpResponse,
    run_controlled_live_sec_companyfacts_smoke,
    sec_user_agent_from_env,
)


FORBIDDEN_SCRIPT_IMPORTS = (
    "scripts.",
    "archive.legacy_runtime",
)

FORBIDDEN_OUTPUT_PATHS = (
    Path("data"),
    Path("reports"),
    Path("reports/daily/telegram_message.txt"),
)


def _minimal_companyfacts_payload() -> str:
    return """
{
  "cik": 1045810,
  "entityName": "NVIDIA Corporation",
  "facts": {
    "us-gaap": {
      "Revenues": {
        "units": {
          "USD": [
            {
              "fy": 2025,
              "fp": "FY",
              "end": "2025-01-26",
              "val": 1200,
              "accn": "0001045810-25-000023",
              "filed": "2025-02-26"
            },
            {
              "fy": 2024,
              "fp": "FY",
              "end": "2024-01-28",
              "val": 1000,
              "accn": "0001045810-24-000029",
              "filed": "2024-02-21"
            }
          ]
        }
      },
      "NetIncomeLoss": {
        "units": {
          "USD": [
            {
              "fy": 2025,
              "fp": "FY",
              "end": "2025-01-26",
              "val": 250,
              "accn": "0001045810-25-000023",
              "filed": "2025-02-26"
            },
            {
              "fy": 2024,
              "fp": "FY",
              "end": "2024-01-28",
              "val": 200,
              "accn": "0001045810-24-000029",
              "filed": "2024-02-21"
            }
          ]
        }
      },
      "OperatingIncomeLoss": {
        "units": {
          "USD": [
            {
              "fy": 2025,
              "fp": "FY",
              "end": "2025-01-26",
              "val": 300,
              "accn": "0001045810-25-000023",
              "filed": "2025-02-26"
            },
            {
              "fy": 2024,
              "fp": "FY",
              "end": "2024-01-28",
              "val": 240,
              "accn": "0001045810-24-000029",
              "filed": "2024-02-21"
            }
          ]
        }
      },
      "NetCashProvidedByUsedInOperatingActivities": {
        "units": {
          "USD": [
            {
              "fy": 2025,
              "fp": "FY",
              "end": "2025-01-26",
              "val": 900,
              "accn": "0001045810-25-000023",
              "filed": "2025-02-26"
            },
            {
              "fy": 2024,
              "fp": "FY",
              "end": "2024-01-28",
              "val": 700,
              "accn": "0001045810-24-000029",
              "filed": "2024-02-21"
            }
          ]
        }
      },
      "PaymentsToAcquirePropertyPlantAndEquipment": {
        "units": {
          "USD": [
            {
              "fy": 2025,
              "fp": "FY",
              "end": "2025-01-26",
              "val": 100,
              "accn": "0001045810-25-000023",
              "filed": "2025-02-26"
            },
            {
              "fy": 2024,
              "fp": "FY",
              "end": "2024-01-28",
              "val": 100,
              "accn": "0001045810-24-000029",
              "filed": "2024-02-21"
            }
          ]
        }
      }
    }
  }
}
"""

def _payload_with_equivalent_latest_filed_duplicate() -> str:
    payload = json.loads(_minimal_companyfacts_payload())
    revenues = payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    duplicate = dict(revenues[0])
    duplicate["frame"] = "CY2024"
    revenues.append(duplicate)
    return json.dumps(payload)


def _payload_with_distinct_latest_filed_duplicate() -> str:
    payload = json.loads(_minimal_companyfacts_payload())
    revenues = payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    duplicate = dict(revenues[0])
    duplicate["val"] = 999999
    revenues.append(duplicate)
    return json.dumps(payload)


def _payload_with_later_non_annual_form_candidate() -> str:
    payload = json.loads(_minimal_companyfacts_payload())
    revenues = payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    revenues[0]["form"] = "10-K"
    later_non_annual_form = dict(revenues[0])
    later_non_annual_form["form"] = "10-Q"
    later_non_annual_form["filed"] = "2025-03-15"
    later_non_annual_form["val"] = 999999
    revenues.append(later_non_annual_form)
    return json.dumps(payload)

def _payload_with_comparative_annual_facts_in_latest_filing() -> str:
    payload = json.loads(_minimal_companyfacts_payload())
    revenues = payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"]

    base_current = dict(revenues[0])
    base_current["form"] = "10-K"
    base_current["start"] = "2024-01-29"
    base_current["end"] = "2025-01-26"
    base_current["frame"] = "CY2024"
    base_current["val"] = 1200

    comparative_prior = dict(base_current)
    comparative_prior["start"] = "2023-01-30"
    comparative_prior["end"] = "2024-01-28"
    comparative_prior["frame"] = "CY2023"
    comparative_prior["val"] = 1000

    comparative_older = dict(base_current)
    comparative_older["start"] = "2022-01-31"
    comparative_older["end"] = "2023-01-29"
    comparative_older["frame"] = "CY2022"
    comparative_older["val"] = 800

    revenues[:] = [comparative_older, comparative_prior, base_current]

    return json.dumps(payload)

def test_live_smoke_is_disabled_by_default():
    calls: list[str] = []

    def fake_fetcher(endpoint, user_agent):
        calls.append(endpoint)
        return SecCompanyFactsHttpResponse(
            status_code=200,
            body=_minimal_companyfacts_payload(),
        )

    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        network_fetcher=fake_fetcher,
        retrieval_timestamp="2026-06-06T00:00:00Z",
    )

    assert result.status == "smoke_failed"
    assert result.failure_category == "explicit_invocation_missing"
    assert result.request_executed is False
    assert result.request_count == 0
    assert calls == []


def test_wrong_ticker_fails_closed_without_network_call():
    calls: list[str] = []

    result = run_controlled_live_sec_companyfacts_smoke(
        ticker="AAPL",
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=lambda endpoint, user_agent: calls.append(endpoint),
    )

    assert result.failure_category == "ticker_cik_mismatch"
    assert result.request_executed is False
    assert result.request_count == 0
    assert calls == []


def test_wrong_cik_fails_closed_without_network_call():
    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik="0000320193",
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=lambda endpoint, user_agent: SecCompanyFactsHttpResponse(
            status_code=200,
            body="{}",
        ),
    )

    assert result.failure_category == "ticker_cik_mismatch"
    assert result.request_executed is False
    assert result.request_count == 0


def test_missing_user_agent_fails_closed_without_network_call():
    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="",
        execute_live=True,
        network_fetcher=lambda endpoint, user_agent: SecCompanyFactsHttpResponse(
            status_code=200,
            body="{}",
        ),
    )

    assert result.failure_category == "user_agent_missing"
    assert result.request_executed is False
    assert result.request_count == 0


def test_user_agent_is_read_from_injected_environment_only_in_tests():
    assert sec_user_agent_from_env(
        {"SEC_USER_AGENT": "MarketScannerControlledSmoke/1.0"}
    ) == "MarketScannerControlledSmoke/1.0"
    assert sec_user_agent_from_env({}) == ""


def test_network_function_is_injectable_and_single_request_only():
    calls: list[tuple[str, str]] = []

    def fake_fetcher(endpoint, user_agent):
        calls.append((endpoint, user_agent))
        return SecCompanyFactsHttpResponse(
            status_code=200,
            body=_minimal_companyfacts_payload(),
        )

    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=fake_fetcher,
        retrieval_timestamp="2026-06-06T00:00:00Z",
    )

    assert calls == [
        (
            SEC_COMPANYFACTS_ENDPOINT,
            "MarketScannerControlledSmoke/1.0",
        )
    ]
    assert result.request_executed is True
    assert result.request_count == 1
    assert result.http_status_category == "2xx"
    assert result.status == "passed"
    assert result.canonical_fields_found == (
        "revenue",
        "net_income",
        "operating_income",
        "operating_cash_flow",
        "capital_expenditures",
        "free_cash_flow",
    )
    assert result.free_cash_flow_status == "derived"
    assert result.growth_evidence_status == "available"
    assert result.readiness_state == "available"

def test_equivalent_latest_filed_duplicates_are_collapsed():
    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=lambda endpoint, user_agent: SecCompanyFactsHttpResponse(
            status_code=200,
            body=_payload_with_equivalent_latest_filed_duplicate(),
        ),
        retrieval_timestamp="2026-06-06T00:00:00Z",
    )

    assert result.status == "passed"
    assert result.failure_category == ""
    assert "revenue" in result.canonical_fields_found
    assert result.request_count == 1


def test_distinct_latest_filed_duplicates_still_fail_closed_as_ambiguous():
    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=lambda endpoint, user_agent: SecCompanyFactsHttpResponse(
            status_code=200,
            body=_payload_with_distinct_latest_filed_duplicate(),
        ),
        retrieval_timestamp="2026-06-06T00:00:00Z",
    )

    assert result.status == "smoke_failed"
    assert result.failure_category == "ambiguous_facts"
    assert result.request_count == 1
    assert result.boundary_result is None
    assert "ambiguous_facts:Revenues:2025" in result.issues


def test_annual_report_form_is_preferred_over_later_non_annual_form_candidate():
    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=lambda endpoint, user_agent: SecCompanyFactsHttpResponse(
            status_code=200,
            body=_payload_with_later_non_annual_form_candidate(),
        ),
        retrieval_timestamp="2026-06-06T00:00:00Z",
    )

    assert result.status == "passed"
    assert result.failure_category == ""
    assert "revenue" in result.canonical_fields_found
    assert result.request_count == 1

def test_http_failure_fails_closed_without_retry():
    calls: list[str] = []

    def fake_fetcher(endpoint, user_agent):
        calls.append(endpoint)
        return SecCompanyFactsHttpResponse(status_code=429, body="rate limited")

    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=fake_fetcher,
    )

    assert result.status == "smoke_failed"
    assert result.failure_category == "http_error"
    assert result.request_count == 1
    assert calls == [SEC_COMPANYFACTS_ENDPOINT]


def test_invalid_json_fails_closed_without_raw_payload_persistence():
    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=lambda endpoint, user_agent: SecCompanyFactsHttpResponse(
            status_code=200,
            body="{not json",
        ),
    )

    assert result.failure_category == "invalid_json"
    assert result.request_count == 1
    assert result.boundary_result is None


def test_ticker_cik_mismatch_in_response_fails_closed():
    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=lambda endpoint, user_agent: SecCompanyFactsHttpResponse(
            status_code=200,
            body='{"cik": 320193, "entityName": "Apple Inc.", "facts": {}}',
        ),
    )

    assert result.failure_category == "ticker_cik_mismatch"
    assert result.request_count == 1
    assert result.boundary_result is None


def test_redacted_result_excludes_raw_payload_and_user_agent():
    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=lambda endpoint, user_agent: SecCompanyFactsHttpResponse(
            status_code=200,
            body=_minimal_companyfacts_payload(),
        ),
    )

    rendered = " ".join(str(value) for value in result.__dict__.values())
    assert "0001045810-25-000023" not in rendered
    assert "MarketScannerControlledSmoke" not in rendered
    assert '"facts"' not in rendered


def test_module_import_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    __import__("importlib").reload(sec_companyfacts_live_smoke)

    assert list(tmp_path.iterdir()) == []
    for path in FORBIDDEN_OUTPUT_PATHS:
        assert not path.exists()


def test_live_smoke_module_does_not_import_script_era_files():
    source = Path(sec_companyfacts_live_smoke.__file__).read_text()

    for forbidden in FORBIDDEN_SCRIPT_IMPORTS:
        assert forbidden not in source


def test_live_smoke_module_has_no_yfinance_or_requests_dependency():
    source = Path(sec_companyfacts_live_smoke.__file__).read_text()

    assert "yfinance" not in source
    assert "yf." not in source
    assert "requests" not in source

def test_latest_period_end_is_selected_within_latest_annual_filing():
    result = run_controlled_live_sec_companyfacts_smoke(
        ticker=APPROVED_LIVE_SMOKE_TICKER,
        cik=APPROVED_LIVE_SMOKE_CIK,
        user_agent="MarketScannerControlledSmoke/1.0",
        execute_live=True,
        network_fetcher=lambda endpoint, user_agent: SecCompanyFactsHttpResponse(
            status_code=200,
            body=_payload_with_comparative_annual_facts_in_latest_filing(),
        ),
        retrieval_timestamp="2026-06-06T00:00:00Z",
    )

    assert result.status == "passed"
    assert result.failure_category == ""
    assert result.fiscal_context_summary == "FY 2025; period_end=2025-01-26"
    assert "revenue" in result.canonical_fields_found
    assert result.request_count == 1