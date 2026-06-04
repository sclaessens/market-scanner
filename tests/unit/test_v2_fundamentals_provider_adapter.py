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
    reported_period: str = "",
    fiscal_year: str = "",
    fiscal_quarter: str = "",
) -> ProviderRawFieldEvidence:
    return ProviderRawFieldEvidence(
        original_field_name=name,
        original_field_value=value,
        original_currency=currency,
        original_unit=unit,
        reported_period=reported_period,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
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
    assert metrics["free_cash_flow"].metric_value == "189"
    assert metrics["free_cash_flow"].normalization_status == "source_derived"
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


def test_direct_free_cash_flow_source_field_remains_source_reported():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                **_response().raw_fields,
                "FreeCashFlow": _field("FreeCashFlow", "190"),
            },
            missing_field_evidence=(),
        )
    )
    metrics = _metric_map(result)

    free_cash_flow = metrics["free_cash_flow"]
    assert free_cash_flow.metric_value == "190"
    assert free_cash_flow.normalization_status == "source_reported"
    assert free_cash_flow.original_field_name == "FreeCashFlow"
    assert free_cash_flow.derivation_formula == ""
    assert free_cash_flow.source_field_names == ("FreeCashFlow",)
    assert "free_cash_flow:source_reported" in result.readiness_record.readiness_warnings


def test_missing_direct_free_cash_flow_is_governed_source_derived():
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
    free_cash_flow = metrics["free_cash_flow"]

    assert free_cash_flow.metric_value == "189"
    assert free_cash_flow.normalization_status == "source_derived"
    assert free_cash_flow.validation_status == "valid"
    assert free_cash_flow.derivation_formula == (
        "free_cash_flow = operating_cash_flow - capital_expenditures"
    )
    assert free_cash_flow.source_field_names == (
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
    )
    assert free_cash_flow.original_field_name == (
        "NetCashProvidedByUsedInOperatingActivities|"
        "PaymentsToAcquirePropertyPlantAndEquipment"
    )
    assert "free_cash_flow:source_derived" in free_cash_flow.validation_warnings
    assert "free_cash_flow:source_derived" in result.readiness_record.readiness_warnings
    assert result.readiness_record.readiness_state == "partial"
    assert metrics["operating_cash_flow"].metric_value == "222"
    assert metrics["capital_expenditures"].metric_value == "33"


def test_derived_free_cash_flow_can_clear_readiness_when_no_other_inputs_are_missing():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "Revenues": _field("Revenues", "1000"),
                "GrossProfit": _field("GrossProfit", "600"),
                "OperatingIncomeLoss": _field("OperatingIncomeLoss", "240"),
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
            missing_field_evidence=("FreeCashFlow",),
        )
    )

    assert _metric_map(result)["free_cash_flow"].normalization_status == "source_derived"
    assert result.readiness_record.missing_fundamentals_count == 0
    assert result.readiness_record.readiness_state == "available"
    assert result.readiness_record.source_data_status == "available"


def test_missing_operating_cash_flow_keeps_free_cash_flow_missing():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "33",
                ),
            },
            missing_field_evidence=("free_cash_flow", "operating_cash_flow"),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.metric_value not in ZERO_LIKE_MISSING_VALUES
    assert free_cash_flow.normalization_status == "missing"
    assert "free_cash_flow:missing_required_input:operating_cash_flow" in (
        free_cash_flow.validation_warnings
    )
    assert result.readiness_record.readiness_state == "partial"


def test_missing_capital_expenditures_keeps_free_cash_flow_missing():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "222",
                ),
            },
            missing_field_evidence=("free_cash_flow", "capital_expenditures"),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.metric_value not in ZERO_LIKE_MISSING_VALUES
    assert free_cash_flow.normalization_status == "missing"
    assert "free_cash_flow:missing_required_input:capital_expenditures" in (
        free_cash_flow.validation_warnings
    )


def test_invalid_operating_cash_flow_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    False,
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "33",
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.normalization_status == "invalid"
    assert "free_cash_flow:invalid:invalid" in free_cash_flow.validation_warnings


def test_invalid_capital_expenditures_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "222",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    False,
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.normalization_status == "invalid"
    assert "free_cash_flow:invalid:invalid" in free_cash_flow.validation_warnings


def test_not_parseable_operating_cash_flow_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "not available",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "33",
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.normalization_status == "not_parseable"
    assert "free_cash_flow:not_parseable:not_parseable" in (
        free_cash_flow.validation_warnings
    )


def test_not_parseable_capital_expenditures_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "222",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "not available",
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.normalization_status == "not_parseable"
    assert "free_cash_flow:not_parseable:not_parseable" in (
        free_cash_flow.validation_warnings
    )


def test_free_cash_flow_derivation_currency_mismatch_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "222",
                    currency="USD",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "33",
                    currency="EUR",
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.normalization_status == "not_derivable"
    assert "free_cash_flow:currency_mismatch" in free_cash_flow.validation_warnings


def test_free_cash_flow_derivation_unit_mismatch_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "222",
                    unit="USD",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "33",
                    unit="USD millions",
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.normalization_status == "not_derivable"
    assert "free_cash_flow:unit_mismatch" in free_cash_flow.validation_warnings


def test_free_cash_flow_derivation_period_mismatch_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "222",
                    reported_period="FY",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "33",
                    reported_period="Q4",
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.normalization_status == "not_derivable"
    assert "free_cash_flow:period_mismatch" in free_cash_flow.validation_warnings


def test_free_cash_flow_derivation_fiscal_context_mismatch_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "222",
                    fiscal_year="2025",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "33",
                    fiscal_year="2024",
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.normalization_status == "not_derivable"
    assert "free_cash_flow:fiscal_context_mismatch" in (
        free_cash_flow.validation_warnings
    )


def test_free_cash_flow_derivation_missing_provenance_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "222",
                    currency="",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "33",
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.normalization_status == "not_derivable"
    assert "free_cash_flow:missing_provenance" in free_cash_flow.validation_warnings


def test_ambiguous_negative_capex_sign_convention_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "222",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "-33",
                ),
            },
            missing_field_evidence=("free_cash_flow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value is None
    assert free_cash_flow.normalization_status == "not_derivable"
    assert "free_cash_flow:sign_convention_ambiguous" in (
        free_cash_flow.validation_warnings
    )


def test_nvda_shaped_input_derives_free_cash_flow_from_valid_cash_flow_inputs():
    result = ingest_provider_fundamentals(
        _response(
            provider_record_id="CIK0001045810-FY-2025",
            original_source_reference="sec-edgar-form-10-k:0001045810-25-000023",
            ticker="NVDA",
            symbol="NVDA",
            entity_identifier="CIK0001045810",
            raw_fields={
                "Revenues": _field("Revenues", "1000"),
                "NetCashProvidedByUsedInOperatingActivities": _field(
                    "NetCashProvidedByUsedInOperatingActivities",
                    "900",
                ),
                "PaymentsToAcquirePropertyPlantAndEquipment": _field(
                    "PaymentsToAcquirePropertyPlantAndEquipment",
                    "100",
                ),
            },
            missing_field_evidence=("FreeCashFlow",),
        )
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value == "800"
    assert free_cash_flow.normalization_status == "source_derived"
    assert result.readiness_record.readiness_warnings == (
        "free_cash_flow:source_derived",
    )


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
