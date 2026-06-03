from importlib import reload
from pathlib import Path

from market_scanner.fundamentals import fundamentals_real_source_smoke
from market_scanner.fundamentals.fundamentals_provider_contracts import (
    ProviderCategory,
    ProviderRawFieldEvidence,
    ProviderSourceResponse,
    ProviderSourceStatus,
)
from market_scanner.fundamentals.fundamentals_real_source_smoke import (
    RealSourceSmokeStatus,
    review_injected_source_response,
    run_controlled_real_source_smoke_test,
)


FORBIDDEN_AUTHORITY_TERMS = {
    "BUY",
    "SELL",
    "HOLD",
    "allocation",
    "company quality",
    "conviction",
    "investment quality",
    "recommendation",
    "tradeability",
    "urgency",
    "valuation attractiveness",
}

FORBIDDEN_RESULT_FIELDS = {
    "allocation",
    "conviction",
    "final_action",
    "investment_quality",
    "recommendation",
    "score",
    "target_price",
    "threshold_price",
    "tradeability",
    "urgency",
}

ZERO_LIKE_MISSING_VALUES = (0, 0.0, "0", False, "")


class FakeRealSourceClient:
    def __init__(self, response: ProviderSourceResponse):
        self.response = response
        self.requested_tickers: list[str] = []

    def fetch_fundamentals(self, ticker: str) -> ProviderSourceResponse:
        self.requested_tickers.append(ticker)
        return self.response


class FailingRealSourceClient:
    def __init__(self):
        self.requested_tickers: list[str] = []

    def fetch_fundamentals(self, ticker: str) -> ProviderSourceResponse:
        self.requested_tickers.append(ticker)
        raise RuntimeError("fake provider unavailable")


def _field(
    name: str,
    value: object,
    *,
    currency: str = "EUR",
    unit: str = "EUR",
) -> ProviderRawFieldEvidence:
    return ProviderRawFieldEvidence(
        original_field_name=name,
        original_field_value=value,
        original_currency=currency,
        original_unit=unit,
    )


def _response(**overrides) -> ProviderSourceResponse:
    response = ProviderSourceResponse(
        provider_name="Official Filing Smoke Fixture",
        provider_category=ProviderCategory.REGULATORY_FILING.value,
        provider_record_id="ASML-FY-2025-SMOKE",
        original_source_reference="official-filing-smoke://ASML/FY/2025",
        ticker="ASML",
        symbol="ASML",
        entity_identifier="ASML-HOLDING-NV-SMOKE",
        source_timestamp="2026-02-11T08:00:00Z",
        retrieval_timestamp="2026-06-03T00:00:00Z",
        reported_period="FY",
        fiscal_year="2025",
        fiscal_quarter="",
        raw_fields={
            "Revenues": _field("Revenues", "28000000000"),
            "OperatingIncomeLoss": _field("OperatingIncomeLoss", "9200000000"),
            "NetIncomeLoss": _field("NetIncomeLoss", "7600000000"),
            "EarningsPerShareDiluted": _field(
                "EarningsPerShareDiluted",
                "19.25",
                unit="EUR per share",
            ),
            "Assets": _field("Assets", "42000000000"),
            "Liabilities": _field("Liabilities", "16000000000"),
            "StockholdersEquity": _field("StockholdersEquity", "26000000000"),
            "NetCashProvidedByUsedInOperatingActivities": _field(
                "NetCashProvidedByUsedInOperatingActivities",
                "8900000000",
            ),
            "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "1100000000",
            ),
        },
        provider_status=ProviderSourceStatus.AVAILABLE.value,
        provider_error_status="",
        missing_field_evidence=("GrossProfit", "FreeCashFlow"),
        provenance_metadata="controlled official-source smoke fixture metadata",
        raw_payload_hash="sha256:controlled-smoke-fixture",
        capture_version="v2-controlled-smoke-v1",
    )
    values = {
        field_name: getattr(response, field_name)
        for field_name in response.__dataclass_fields__
    }
    values.update(overrides)
    return ProviderSourceResponse(**values)


def _metric_map(result):
    assert result.ingestion_result is not None
    return {
        record.metric_name: record
        for record in result.ingestion_result.normalized_records
    }


def _assert_no_authority_text(record):
    rendered = " ".join(str(value) for value in record.__dict__.values())
    for forbidden in FORBIDDEN_AUTHORITY_TERMS:
        assert forbidden not in rendered


def test_real_source_smoke_module_import_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(fundamentals_real_source_smoke)

    assert list(tmp_path.iterdir()) == []
    assert not Path("data/raw").exists()
    assert not Path("data/normalized").exists()
    assert not Path("data/generated").exists()
    assert not Path("data/processed").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()


def test_explicit_invocation_is_required_before_fake_client_is_called():
    client = FakeRealSourceClient(_response())

    assert client.requested_tickers == []

    result = run_controlled_real_source_smoke_test(client, ticker="ASML")

    assert client.requested_tickers == ["ASML"]
    assert result.ticker == "ASML"


def test_fake_real_source_response_passes_through_provider_boundary():
    client = FakeRealSourceClient(_response())

    result = run_controlled_real_source_smoke_test(client, ticker="ASML")

    assert result.provider_name == "Official Filing Smoke Fixture"
    assert result.smoke_status == RealSourceSmokeStatus.REVIEW_REQUIRED.value
    assert result.ingestion_result is not None
    assert result.ingestion_result.raw_evidence.ticker == "ASML"
    assert result.ingestion_result.normalized_records
    assert result.ingestion_result.readiness_record.readiness_state == "partial"
    assert result.missing_field_summary == ("GrossProfit", "FreeCashFlow")
    assert "missing_fundamentals:2" in result.warnings


def test_smoke_result_raw_evidence_preserves_provenance():
    result = run_controlled_real_source_smoke_test(
        FakeRealSourceClient(_response()),
        ticker="ASML",
    )
    assert result.ingestion_result is not None
    raw = result.ingestion_result.raw_evidence

    assert raw.provider_name == "Official Filing Smoke Fixture"
    assert raw.provider_category == "regulatory_filing"
    assert raw.original_source_reference == "official-filing-smoke://ASML/FY/2025"
    assert raw.ticker == "ASML"
    assert raw.source_timestamp == "2026-02-11T08:00:00Z"
    assert raw.retrieval_timestamp == "2026-06-03T00:00:00Z"
    assert raw.fiscal_year == "2025"
    assert raw.reported_period == "FY"
    assert {field.original_currency for field in raw.raw_fields} == {"EUR"}
    assert "controlled official-source" in raw.provenance_metadata
    assert result.provenance_summary == (
        "Official Filing Smoke Fixture|official-filing-smoke://ASML/FY/2025|"
        "2026-02-11T08:00:00Z|2026-06-03T00:00:00Z|FY|2025"
    )


def test_normalized_fundamentals_remain_program_ready_input_only():
    result = run_controlled_real_source_smoke_test(
        FakeRealSourceClient(_response()),
        ticker="ASML",
    )
    metrics = _metric_map(result)

    assert metrics["revenue"].metric_value == "28000000000"
    assert metrics["revenue"].currency == "EUR"
    assert metrics["revenue"].source_reference == "official-filing-smoke://ASML/FY/2025"
    assert metrics["eps_diluted"].metric_unit == "EUR per share"

    for record in result.ingestion_result.normalized_records:
        assert set(record.__dataclass_fields__).isdisjoint(FORBIDDEN_RESULT_FIELDS)
        _assert_no_authority_text(record)


def test_smoke_missing_values_remain_explicit_and_not_zero():
    result = run_controlled_real_source_smoke_test(
        FakeRealSourceClient(_response()),
        ticker="ASML",
    )
    metrics = _metric_map(result)

    assert metrics["gross_profit"].metric_value is None
    assert metrics["gross_profit"].metric_value not in ZERO_LIKE_MISSING_VALUES
    assert metrics["free_cash_flow"].metric_value is None
    assert metrics["free_cash_flow"].metric_value not in ZERO_LIKE_MISSING_VALUES


def test_smoke_readiness_remains_neutral():
    result = run_controlled_real_source_smoke_test(
        FakeRealSourceClient(_response()),
        ticker="ASML",
    )
    assert result.ingestion_result is not None
    readiness = result.ingestion_result.readiness_record

    assert readiness.readiness_state == "partial"
    assert readiness.source_data_status == "partial"
    assert readiness.missing_fundamentals_count == 2
    assert set(readiness.__dataclass_fields__).isdisjoint(FORBIDDEN_RESULT_FIELDS)
    _assert_no_authority_text(readiness)


def test_smoke_execution_has_no_file_io_or_downstream_side_effects(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    run_controlled_real_source_smoke_test(
        FakeRealSourceClient(_response()),
        ticker="ASML",
    )

    assert list(tmp_path.iterdir()) == []
    assert not Path("data/raw").exists()
    assert not Path("data/normalized").exists()
    assert not Path("data/generated").exists()
    assert not Path("data/processed").exists()
    assert not Path("data/portfolio").exists()
    assert not Path("data/watchlist").exists()
    assert not Path("data/logs").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()


def test_fake_provider_failure_returns_neutral_smoke_failure():
    client = FailingRealSourceClient()

    result = run_controlled_real_source_smoke_test(client, ticker="ASML")

    assert client.requested_tickers == ["ASML"]
    assert result.ticker == "ASML"
    assert result.provider_name == ""
    assert result.smoke_status == RealSourceSmokeStatus.PROVIDER_ERROR.value
    assert result.ingestion_result is None
    assert result.warnings == ("provider_source_error:RuntimeError",)
    _assert_no_authority_text(result)


def test_injected_response_review_does_not_call_a_client():
    response = _response()

    result = review_injected_source_response(response)

    assert result.ticker == "ASML"
    assert result.provider_name == "Official Filing Smoke Fixture"
    assert result.ingestion_result is not None


def test_smoke_module_source_has_no_legacy_runtime_or_live_network_imports():
    source = Path(fundamentals_real_source_smoke.__file__).read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "scripts",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "yfinance",
        "telegram",
    ):
        assert forbidden not in source
