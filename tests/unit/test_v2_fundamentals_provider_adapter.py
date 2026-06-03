from importlib import reload
from pathlib import Path

from market_scanner.fundamentals import fundamentals_provider_adapter
from market_scanner.fundamentals.fundamentals_provider_adapter import (
    ingest_provider_fundamentals,
    ingest_provider_fundamentals_from_client,
)
from market_scanner.fundamentals.fundamentals_provider_contracts import (
    ProviderCategory,
    ProviderRawFieldEvidence,
    ProviderSourceResponse,
    ProviderSourceStatus,
)


FORBIDDEN_ADAPTER_FIELDS = {
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


class FakeProviderClient:
    def __init__(self, response: ProviderSourceResponse):
        self.response = response
        self.requested_tickers: list[str] = []

    def fetch_fundamentals(self, ticker: str) -> ProviderSourceResponse:
        self.requested_tickers.append(ticker)
        return self.response


def _field(
    name: str,
    value: object,
    *,
    currency: str = "USD",
    unit: str = "USD",
) -> ProviderRawFieldEvidence:
    return ProviderRawFieldEvidence(
        original_field_name=name,
        original_field_value=value,
        original_currency=currency,
        original_unit=unit,
    )


def _response(**overrides) -> ProviderSourceResponse:
    response = ProviderSourceResponse(
        provider_name="SEC Companyfacts",
        provider_category=ProviderCategory.REGULATORY_FILING.value,
        provider_record_id="CIK0000320193-FY-2025",
        original_source_reference="sec-companyfacts:CIK0000320193/FY/2025",
        ticker="AAPL",
        symbol="AAPL",
        entity_identifier="CIK0000320193",
        source_timestamp="2026-02-15",
        retrieval_timestamp="2026-06-03T00:00:00Z",
        reported_period="FY",
        fiscal_year="2025",
        fiscal_quarter="",
        raw_fields={
            "Revenues": _field("Revenues", "1000"),
            "NetIncomeLoss": _field("NetIncomeLoss", "155"),
            "EarningsPerShareDiluted": _field(
                "EarningsPerShareDiluted",
                "3.14",
                unit="USD per share",
            ),
            "Assets": _field("Assets", "5000"),
            "Liabilities": _field("Liabilities", "1700"),
            "StockholdersEquity": _field("StockholdersEquity", "3300"),
            "NetCashProvidedByUsedInOperatingActivities": _field(
                "NetCashProvidedByUsedInOperatingActivities",
                "222",
            ),
            "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "33",
            ),
        },
        provider_status=ProviderSourceStatus.AVAILABLE.value,
        provider_error_status="",
        missing_field_evidence=("free_cash_flow",),
        provenance_metadata="official SEC companyfacts response metadata",
        raw_payload_hash="sha256:provider-adapter-fixture",
        capture_version="v2-provider-adapter-v1",
    )
    values = {
        field_name: getattr(response, field_name)
        for field_name in response.__dataclass_fields__
    }
    values.update(overrides)
    return ProviderSourceResponse(**values)


def _metric_map(result):
    return {record.metric_name: record for record in result.normalized_records}


def test_injected_provider_response_becomes_raw_evidence():
    result = ingest_provider_fundamentals(_response())

    assert result.raw_evidence.provider_name == "SEC Companyfacts"
    assert result.raw_evidence.provider_category == "regulatory_filing"
    assert result.raw_evidence.provider_record_id == "CIK0000320193-FY-2025"
    assert result.raw_evidence.original_source_reference == (
        "sec-companyfacts:CIK0000320193/FY/2025"
    )
    assert result.raw_evidence.source_timestamp == "2026-02-15"
    assert result.raw_evidence.retrieval_timestamp == "2026-06-03T00:00:00Z"
    assert result.raw_evidence.raw_payload_hash.startswith("sha256:")
    assert {field.original_field_name for field in result.raw_evidence.raw_fields} >= {
        "Revenues",
        "NetIncomeLoss",
    }


def test_raw_evidence_maps_to_normalized_program_ready_records():
    result = ingest_provider_fundamentals(_response())
    metrics = _metric_map(result)

    assert metrics["revenue"].metric_value == "1000"
    assert metrics["revenue"].metric_unit == "USD"
    assert metrics["revenue"].currency == "USD"
    assert metrics["revenue"].original_field_name == "Revenues"
    assert metrics["net_income"].metric_value == "155"
    assert metrics["eps_diluted"].metric_unit == "USD per share"
    assert metrics["free_cash_flow"].metric_value is None
    assert result.readiness_record.readiness_state == "partial"


def test_missing_values_remain_explicit_and_are_not_zero():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "Revenues": _field("Revenues", None),
                "NetIncomeLoss": _field("NetIncomeLoss", "155"),
            }
        )
    )
    revenue = _metric_map(result)["revenue"]

    assert revenue.metric_value is None
    assert revenue.metric_value not in ZERO_LIKE_MISSING_VALUES
    assert result.readiness_record.missing_fundamentals_count > 0
    assert result.readiness_record.readiness_state == "partial"


def test_missing_source_fields_are_not_backfilled_or_derived_as_zero():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "222",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "33",
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    metrics = _metric_map(result)

    assert metrics["free_cash_flow"].metric_value is None
    assert metrics["free_cash_flow"].metric_value not in ZERO_LIKE_MISSING_VALUES
    assert metrics["free_cash_flow"].normalization_status == "missing_source_field"
    assert metrics["operating_cash_flow"].metric_value == "222"
    assert metrics["capital_expenditures"].metric_value == "33"


def test_provider_errors_become_neutral_invalid_readiness():
    result = ingest_provider_fundamentals(
        _response(
            provider_status=ProviderSourceStatus.PROVIDER_ERROR.value,
            provider_error_status="provider_unavailable",
            raw_fields={},
            missing_field_evidence=("provider_unavailable",),
        )
    )

    assert result.readiness_record.readiness_state == "invalid"
    assert result.readiness_record.source_data_status == "invalid"
    assert result.readiness_record.provider_error_status == "provider_unavailable"
    assert "quality" not in result.readiness_record.__dataclass_fields__


def test_stale_provider_status_becomes_neutral_stale_readiness():
    result = ingest_provider_fundamentals(
        _response(provider_status=ProviderSourceStatus.STALE_DATA.value)
    )

    assert result.readiness_record.readiness_state == "stale"
    assert result.readiness_record.source_data_status == "stale"
    assert result.readiness_record.stale_data_count == 1


def test_normalized_values_retain_source_references():
    result = ingest_provider_fundamentals(_response())

    for record in result.normalized_records:
        assert record.source_provider == "SEC Companyfacts"
        assert record.source_reference == "sec-companyfacts:CIK0000320193/FY/2025"
        assert record.source_record_identity == "CIK0000320193-FY-2025"
        assert record.normalized_at == "2026-06-03T00:00:00Z"
        assert record.fiscal_period == "FY"
        assert record.fiscal_year == "2025"


def test_adapter_uses_dependency_injection_for_provider_access():
    client = FakeProviderClient(_response())

    result = ingest_provider_fundamentals_from_client(client, ticker="AAPL")

    assert client.requested_tickers == ["AAPL"]
    assert result.raw_evidence.ticker == "AAPL"
    assert result.normalized_records


def test_adapter_records_do_not_add_forbidden_authority_fields():
    result = ingest_provider_fundamentals(_response())
    field_names = (
        set(result.raw_evidence.__dataclass_fields__)
        | set(result.readiness_record.__dataclass_fields__)
        | set(result.normalized_records[0].__dataclass_fields__)
    )

    assert field_names.isdisjoint(FORBIDDEN_ADAPTER_FIELDS)


def test_adapter_import_and_execution_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(fundamentals_provider_adapter)
    ingest_provider_fundamentals(_response())

    assert list(tmp_path.iterdir()) == []
    assert not Path("data/raw").exists()
    assert not Path("data/normalized").exists()
    assert not Path("data/generated").exists()
    assert not Path("data/processed").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()


def test_adapter_source_has_no_legacy_import_or_live_network_client():
    source = Path(fundamentals_provider_adapter.__file__).read_text(encoding="utf-8")

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
