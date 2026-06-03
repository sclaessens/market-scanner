import json
from importlib import reload
from pathlib import Path

from market_scanner.fundamentals import fundamentals_provider_adapter
from market_scanner.fundamentals.fundamentals_provider_adapter import (
    ingest_provider_fundamentals,
)
from market_scanner.fundamentals.fundamentals_provider_contracts import (
    ProviderRawFieldEvidence,
    ProviderSourceResponse,
    ProviderSourceStatus,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "fundamentals"
    / "provider_dry_run_fixture.json"
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

ZERO_LIKE_MISSING_VALUES = (0, 0.0, "0", False, "")


def _fixture_payload() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _fixture_response() -> ProviderSourceResponse:
    payload = _fixture_payload()
    raw_fields = {
        field["original_field_name"]: ProviderRawFieldEvidence(
            original_field_name=field["original_field_name"],
            original_field_value=field["original_field_value"],
            original_currency=field["original_currency"],
            original_unit=field["original_unit"],
        )
        for field in payload["raw_fields"]
    }

    return ProviderSourceResponse(
        provider_name=payload["provider_name"],
        provider_category=payload["provider_category"],
        provider_record_id=payload["provider_record_id"],
        original_source_reference=payload["original_source_reference"],
        ticker=payload["ticker"],
        symbol=payload["symbol"],
        entity_identifier=payload["entity_identifier"],
        source_timestamp=payload["source_timestamp"],
        retrieval_timestamp=payload["retrieval_timestamp"],
        reported_period=payload["reported_period"],
        fiscal_year=payload["fiscal_year"],
        fiscal_quarter=payload["fiscal_quarter"],
        raw_fields=raw_fields,
        provider_status=payload["provider_status"],
        provider_error_status=payload["provider_error_status"],
        missing_field_evidence=tuple(payload["missing_field_evidence"]),
        provenance_metadata=payload["provenance_metadata"],
        raw_payload_hash=payload["raw_payload_hash"],
        capture_version=payload["capture_version"],
    )


def _metric_map(result):
    return {record.metric_name: record for record in result.normalized_records}


def _assert_no_authority_text(record):
    rendered = " ".join(str(value) for value in record.__dict__.values())
    for forbidden in FORBIDDEN_AUTHORITY_TERMS:
        assert forbidden not in rendered


def test_provider_dry_run_fixture_loads_without_live_calls_or_credentials():
    payload = _fixture_payload()

    assert FIXTURE_PATH.is_file()
    assert payload["ticker"] == "ASML"
    assert payload["provider_category"] == "regulatory_filing"
    assert "api_key" not in payload
    assert "credential" not in payload
    assert "token" not in payload


def test_provider_dry_run_fixture_captures_raw_evidence_correctly():
    result = ingest_provider_fundamentals(_fixture_response())
    raw = result.raw_evidence
    raw_fields = {field.original_field_name: field for field in raw.raw_fields}

    assert raw.provider_name == "Official Filing Dry-Run Fixture"
    assert raw.provider_category == "regulatory_filing"
    assert raw.provider_record_id == "ASML-FY-2025-DRY-RUN"
    assert raw.original_source_reference == "official-filing-fixture://ASML/FY/2025"
    assert raw.ticker == "ASML"
    assert raw.source_timestamp == "2026-02-11T08:00:00Z"
    assert raw.retrieval_timestamp == "2026-06-03T00:00:00Z"
    assert raw.reported_period == "FY"
    assert raw.fiscal_year == "2025"
    assert raw.fiscal_quarter == ""
    assert raw.provider_status == ProviderSourceStatus.AVAILABLE.value
    assert raw.missing_field_evidence == ("GrossProfit", "FreeCashFlow")
    assert "official regulatory source evidence" in raw.provenance_metadata
    assert raw_fields["Revenues"].original_field_value == "28000000000"
    assert raw_fields["Revenues"].original_currency == "EUR"
    assert raw_fields["Revenues"].original_unit == "EUR"
    assert raw_fields["ProviderUnavailableExample"].original_field_value is None


def test_provider_dry_run_fixture_normalizes_supported_fields_correctly():
    result = ingest_provider_fundamentals(_fixture_response())
    metrics = _metric_map(result)

    assert metrics["revenue"].metric_value == "28000000000"
    assert metrics["revenue"].metric_unit == "EUR"
    assert metrics["revenue"].currency == "EUR"
    assert metrics["revenue"].original_field_name == "Revenues"
    assert metrics["revenue"].normalization_status == "mapped_from_raw_evidence"
    assert metrics["revenue"].validation_status == "valid"
    assert metrics["net_income"].metric_value == "7600000000"
    assert metrics["operating_income"].metric_value == "9200000000"
    assert metrics["eps_diluted"].metric_unit == "EUR per share"
    assert metrics["total_assets"].metric_value == "42000000000"
    assert metrics["shareholders_equity"].metric_value == "26000000000"
    assert metrics["operating_cash_flow"].metric_value == "8900000000"
    assert metrics["capital_expenditures"].metric_value == "1100000000"


def test_provider_dry_run_missing_values_remain_explicit():
    result = ingest_provider_fundamentals(_fixture_response())
    metrics = _metric_map(result)

    assert metrics["gross_profit"].metric_value is None
    assert metrics["gross_profit"].metric_value not in ZERO_LIKE_MISSING_VALUES
    assert metrics["gross_profit"].normalization_status == "missing_source_field"
    assert metrics["gross_profit"].validation_status == "review_required"
    assert metrics["free_cash_flow"].metric_value is None
    assert metrics["free_cash_flow"].metric_value not in ZERO_LIKE_MISSING_VALUES
    assert metrics["free_cash_flow"].normalization_status == "missing_source_field"
    assert result.readiness_record.missing_fundamentals_count == 2


def test_provider_dry_run_missing_values_are_not_converted_to_zero():
    result = ingest_provider_fundamentals(_fixture_response())
    metrics = _metric_map(result)

    for metric_name in ("gross_profit", "free_cash_flow"):
        assert metrics[metric_name].metric_value is None
        assert metrics[metric_name].metric_value != 0
        assert metrics[metric_name].metric_value != 0.0
        assert metrics[metric_name].metric_value != "0"
        assert metrics[metric_name].metric_value is not False
        assert metrics[metric_name].metric_value != ""


def test_provider_dry_run_readiness_remains_neutral():
    result = ingest_provider_fundamentals(_fixture_response())
    readiness = result.readiness_record

    assert readiness.readiness_state == "partial"
    assert readiness.source_data_status == "partial"
    assert readiness.provider_status == "available"
    assert readiness.provider_error_status == ""
    _assert_no_authority_text(readiness)


def test_provider_dry_run_provenance_is_traceable_from_normalized_values():
    result = ingest_provider_fundamentals(_fixture_response())

    for record in result.normalized_records:
        assert record.ticker == "ASML"
        assert record.source_provider == "Official Filing Dry-Run Fixture"
        assert record.source_reference == "official-filing-fixture://ASML/FY/2025"
        assert record.source_record_identity == "ASML-FY-2025-DRY-RUN"
        assert record.normalized_at == "2026-06-03T00:00:00Z"
        assert record.fiscal_period == "FY"
        assert record.fiscal_year == "2025"


def test_provider_dry_run_review_has_no_downstream_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(fundamentals_provider_adapter)
    result = ingest_provider_fundamentals(_fixture_response())
    source = Path(fundamentals_provider_adapter.__file__).read_text(
        encoding="utf-8"
    )

    assert result.normalized_records
    for forbidden_import in (
        "scripts",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "yfinance",
        "telegram",
    ):
        assert forbidden_import not in source

    assert list(tmp_path.iterdir()) == []
    assert not Path("data/raw").exists()
    assert not Path("data/normalized").exists()
    assert not Path("data/generated").exists()
    assert not Path("data/processed").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
