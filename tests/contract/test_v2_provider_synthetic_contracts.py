from importlib import reload
from pathlib import Path

from market_scanner.fundamentals import fundamentals_normalization_adapter
from market_scanner.fundamentals.fundamentals_normalization_adapter import (
    SyntheticRawFundamentalRecord,
    normalize_synthetic_fundamentals,
)
from market_scanner.fundamentals.fundamentals_normalization_contracts import (
    forbidden_normalized_fundamentals_fields,
)


FORBIDDEN_INVESTMENT_AUTHORITY_TERMS = {
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


def _provider_raw_record(**overrides):
    record = SyntheticRawFundamentalRecord(
        source_provider="SYNTHETIC_PRIMARY_FILINGS",
        source_record_id="official-filing-ASML-FY-2025",
        ticker="ASML",
        fiscal_period="FY",
        fiscal_year="2025",
        captured_at="2026-06-03T00:00:00Z",
        source_reference="official-filing://ASML/2025/FY",
        raw_payload_hash="sha256:synthetic-provider-evidence",
        metrics={
            "revenue": "28000000000",
            "net_income": "7600000000",
            "eps_diluted": "19.25",
            "total_assets": "42000000000",
            "total_liabilities": "16000000000",
            "shareholders_equity": "26000000000",
            "operating_cash_flow": "8900000000",
            "capital_expenditures": "1100000000",
            "free_cash_flow": "7800000000",
        },
        metric_units={
            "revenue": "EUR",
            "net_income": "EUR",
            "eps_diluted": "EUR per share",
            "total_assets": "EUR",
            "total_liabilities": "EUR",
            "shareholders_equity": "EUR",
            "operating_cash_flow": "EUR",
            "capital_expenditures": "EUR",
            "free_cash_flow": "EUR",
        },
        currency="EUR",
    )
    values = {
        field_name: getattr(record, field_name)
        for field_name in record.__dataclass_fields__
    }
    values.update(overrides)
    return SyntheticRawFundamentalRecord(**values)


def _metric_map(result):
    return {record.metric_name: record for record in result.normalized_records}


def _issue_values(result, field_name):
    return tuple(
        issue.observed_value for issue in result.issues if issue.field_name == field_name
    )


def _assert_no_investment_authority_text(record):
    text = " ".join(str(value) for value in record.__dict__.values())
    for forbidden in FORBIDDEN_INVESTMENT_AUTHORITY_TERMS:
        assert forbidden not in text


def test_synthetic_raw_provider_evidence_can_represent_design_metadata():
    provider_metadata = {
        "provider_category": "primary_regulatory_filing",
        "original_source_reference": "official-filing://ASML/2025/FY",
        "entity_identifier": "ASML-HOLDING-NV-SYNTHETIC",
        "source_timestamp": "2026-02-11T08:00:00Z",
        "retrieval_timestamp": "2026-06-03T00:00:00Z",
        "reported_period": "FY",
        "fiscal_quarter": None,
        "provider_status": "available",
        "provider_error_status": None,
        "missing_field_evidence": (),
        "provenance_metadata": "official filing evidence, synthetic only",
        "capture_version": "v2-provider-contract-synthetic-1",
    }
    raw_record = _provider_raw_record()

    assert raw_record.source_provider == "SYNTHETIC_PRIMARY_FILINGS"
    assert raw_record.source_record_id == "official-filing-ASML-FY-2025"
    assert raw_record.ticker == "ASML"
    assert raw_record.fiscal_period == provider_metadata["reported_period"]
    assert raw_record.fiscal_year == "2025"
    assert raw_record.captured_at == provider_metadata["retrieval_timestamp"]
    assert raw_record.source_reference == provider_metadata["original_source_reference"]
    assert raw_record.raw_payload_hash.startswith("sha256:")
    assert {"revenue", "net_income", "eps_diluted"}.issubset(raw_record.metrics)
    assert raw_record.currency == "EUR"
    assert raw_record.metric_units["eps_diluted"] == "EUR per share"
    assert provider_metadata["provider_category"] == "primary_regulatory_filing"


def test_synthetic_raw_evidence_maps_to_normalized_program_ready_fundamentals():
    result = normalize_synthetic_fundamentals((_provider_raw_record(),))
    metrics = _metric_map(result)

    assert set(metrics) == {
        "revenue",
        "net_income",
        "eps_diluted",
        "total_assets",
        "total_liabilities",
        "shareholders_equity",
        "operating_cash_flow",
        "capital_expenditures",
        "free_cash_flow",
    }
    assert metrics["revenue"].metric_value == "28000000000"
    assert metrics["revenue"].metric_unit == "EUR"
    assert metrics["eps_diluted"].metric_unit == "EUR per share"
    assert metrics["free_cash_flow"].source_record_identity == (
        "official-filing-ASML-FY-2025"
    )
    assert result.readiness_records[0].readiness_state == "available"
    assert result.issues == ()


def test_missing_source_values_remain_missing_and_not_zero_like():
    result = normalize_synthetic_fundamentals(
        (
            _provider_raw_record(
                metrics={
                    "revenue": None,
                    "net_income": "7600000000",
                    "eps_diluted": "19.25",
                }
            ),
        )
    )
    revenue = _metric_map(result)["revenue"]

    assert revenue.metric_value is None
    assert revenue.metric_value not in ZERO_LIKE_MISSING_VALUES
    assert result.readiness_records[0].readiness_state == "partial"
    assert result.readiness_records[0].missing_fundamentals_count == 1
    assert _issue_values(result, "metric_value") == (None,)


def test_missing_unknown_parse_and_provider_error_values_are_never_zero():
    result = normalize_synthetic_fundamentals(
        (
            _provider_raw_record(
                metrics={
                    "missing_source_field": None,
                    "unknown_provider_value": None,
                    "not_reported_value": None,
                    "parse_error_value": None,
                    "provider_error_value": None,
                },
                metric_units={
                    "missing_source_field": "EUR",
                    "unknown_provider_value": "EUR",
                    "not_reported_value": "EUR",
                    "parse_error_value": "EUR",
                    "provider_error_value": "EUR",
                },
            ),
        )
    )

    for record in result.normalized_records:
        assert record.metric_value is None
        assert record.metric_value not in ZERO_LIKE_MISSING_VALUES

    assert result.readiness_records[0].readiness_state == "missing"
    assert result.readiness_records[0].missing_fundamentals_count == 5


def test_provider_errors_produce_neutral_readiness_not_investment_conclusions():
    result = normalize_synthetic_fundamentals(
        (_provider_raw_record(source_record_id="", raw_payload_hash=""),)
    )
    readiness = result.readiness_records[0]

    assert readiness.readiness_state == "invalid"
    assert readiness.source_data_status == "invalid"
    assert set(_issue_values(result, "source_record_id")) == {""}
    assert set(_issue_values(result, "raw_payload_hash")) == {""}
    _assert_no_investment_authority_text(readiness)


def test_stale_source_evidence_produces_neutral_stale_readiness():
    result = normalize_synthetic_fundamentals(
        (_provider_raw_record(stale_metric_names=("revenue",)),)
    )
    readiness = result.readiness_records[0]

    assert readiness.readiness_state == "stale"
    assert readiness.source_data_status == "stale"
    assert readiness.stale_data_count == 1
    _assert_no_investment_authority_text(readiness)


def test_every_normalized_value_preserves_source_traceability():
    result = normalize_synthetic_fundamentals((_provider_raw_record(),))

    for record in result.normalized_records:
        assert record.source_record_identity == "official-filing-ASML-FY-2025"
        assert record.source_reference == "official-filing://ASML/2025/FY"
        assert record.source_provider == "SYNTHETIC_PRIMARY_FILINGS"
        assert record.normalized_at == "2026-06-03T00:00:00Z"
        assert record.fiscal_period == "FY"
        assert record.fiscal_year == "2025"


def test_derived_fields_are_only_preserved_when_explicitly_supplied():
    without_supplied_derivative = normalize_synthetic_fundamentals(
        (
            _provider_raw_record(
                metrics={
                    "operating_cash_flow": "8900000000",
                    "capital_expenditures": "1100000000",
                },
                metric_units={
                    "operating_cash_flow": "EUR",
                    "capital_expenditures": "EUR",
                },
            ),
        )
    )
    with_supplied_derivative = normalize_synthetic_fundamentals(
        (
            _provider_raw_record(
                metrics={
                    "operating_cash_flow": "8900000000",
                    "capital_expenditures": "1100000000",
                    "free_cash_flow": "7800000000",
                },
                metric_units={
                    "operating_cash_flow": "EUR",
                    "capital_expenditures": "EUR",
                    "free_cash_flow": "EUR",
                },
            ),
        )
    )

    assert "free_cash_flow" not in _metric_map(without_supplied_derivative)
    assert _metric_map(with_supplied_derivative)["free_cash_flow"].metric_value == (
        "7800000000"
    )


def test_source_data_readiness_remains_neutral_contract_metadata():
    result = normalize_synthetic_fundamentals(
        (
            _provider_raw_record(metrics={"revenue": None}),
            _provider_raw_record(metrics={}, source_record_id="source-missing"),
            _provider_raw_record(stale_metric_names=("revenue",)),
            _provider_raw_record(raw_payload_hash=""),
        )
    )

    assert tuple(record.readiness_state for record in result.readiness_records) == (
        "missing",
        "source_missing",
        "stale",
        "invalid",
    )
    for readiness in result.readiness_records:
        _assert_no_investment_authority_text(readiness)


def test_synthetic_provider_contracts_do_not_define_forbidden_authority_fields():
    adapter_fields = (
        set(SyntheticRawFundamentalRecord.__dataclass_fields__)
        | set(
            fundamentals_normalization_adapter.SyntheticNormalizedFundamentalRecord.__dataclass_fields__
        )
        | set(
            fundamentals_normalization_adapter.SyntheticSourceDataReadinessRecord.__dataclass_fields__
        )
    )

    assert adapter_fields.isdisjoint(forbidden_normalized_fundamentals_fields())


def test_provider_synthetic_path_uses_no_legacy_runtime_network_or_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    reload(fundamentals_normalization_adapter)
    result = normalize_synthetic_fundamentals((_provider_raw_record(),))
    source = Path(fundamentals_normalization_adapter.__file__).read_text(
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
        "EDGAR",
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
