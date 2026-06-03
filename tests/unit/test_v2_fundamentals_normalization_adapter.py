from importlib import reload
from pathlib import Path

from market_scanner.fundamentals import fundamentals_normalization_adapter
from market_scanner.fundamentals.fundamentals_normalization_adapter import (
    SyntheticRawFundamentalRecord,
    normalize_synthetic_fundamentals,
)
from market_scanner.fundamentals.fundamentals_normalization_contracts import (
    FundamentalsNormalizationIssueCode,
)


FORBIDDEN_ADAPTER_FIELD_NAMES = {
    "final_action",
    "decision_state",
    "investment_quality",
    "investment_quality_score",
    "quality_score",
    "target_price",
    "threshold_price",
    "allocation",
    "allocation_amount",
    "execution_instruction",
    "urgency",
    "conviction",
    "tradeability",
    "rank",
    "ranking",
    "score",
    "recommendation",
}


def _asml_raw_record(**overrides):
    record = SyntheticRawFundamentalRecord(
        source_provider="SYNTHETIC_PROVIDER",
        source_record_id="raw-asml-fy-2025",
        ticker="ASML",
        fiscal_period="FY",
        fiscal_year="2025",
        captured_at="2026-06-03T00:00:00Z",
        source_reference="synthetic/raw/asml/fy-2025",
        raw_payload_hash="sha256:synthetic-asml",
        metrics={
            "revenue": "28000000000",
            "gross_margin": "51.3%",
            "free_cash_flow": "7800000000",
        },
        metric_units={
            "revenue": "EUR",
            "gross_margin": "percent",
            "free_cash_flow": "EUR",
        },
        currency="EUR",
    )
    return _replace_record(record, **overrides)


def _replace_record(record, **overrides):
    values = {
        field_name: getattr(record, field_name)
        for field_name in record.__dataclass_fields__
    }
    values.update(overrides)
    return type(record)(**values)


def _has_issue(result, *, field_name, issue_code, observed_value):
    return any(
        issue.field_name == field_name
        and issue.issue_code == issue_code
        and issue.observed_value == observed_value
        for issue in result.issues
    )


def test_synthetic_raw_records_are_accepted_in_memory():
    result = normalize_synthetic_fundamentals((_asml_raw_record(),))

    assert len(result.normalized_records) == 3
    assert len(result.readiness_records) == 1
    assert result.issues == ()


def test_normalized_records_preserve_identity_and_traceability():
    result = normalize_synthetic_fundamentals((_asml_raw_record(),))

    for record in result.normalized_records:
        assert record.ticker == "ASML"
        assert record.fiscal_period == "FY"
        assert record.fiscal_year == "2025"
        assert record.source_provider == "SYNTHETIC_PROVIDER"
        assert record.source_reference == "synthetic/raw/asml/fy-2025"
        assert record.source_record_identity == "raw-asml-fy-2025"
        assert record.normalized_at == "2026-06-03T00:00:00Z"


def test_metric_values_and_units_are_preserved_exactly():
    result = normalize_synthetic_fundamentals((_asml_raw_record(),))

    metrics = {record.metric_name: record for record in result.normalized_records}

    assert metrics["revenue"].metric_value == "28000000000"
    assert metrics["revenue"].metric_unit == "EUR"
    assert metrics["gross_margin"].metric_value == "51.3%"
    assert metrics["gross_margin"].metric_unit == "percent"
    assert metrics["free_cash_flow"].metric_value == "7800000000"
    assert metrics["free_cash_flow"].metric_unit == "EUR"


def test_missing_metric_values_remain_explicit_and_not_zero():
    raw_record = _asml_raw_record(
        metrics={
            "revenue": "28000000000",
            "gross_margin": "",
            "free_cash_flow": "7800000000",
        }
    )

    result = normalize_synthetic_fundamentals((raw_record,))
    metrics = {record.metric_name: record for record in result.normalized_records}

    assert metrics["gross_margin"].metric_value == ""
    assert metrics["gross_margin"].metric_value != 0
    assert result.readiness_records[0].missing_fundamentals_count == 1
    assert result.readiness_records[0].readiness_state == "partial"
    assert _has_issue(
        result,
        field_name="metric_value",
        issue_code=FundamentalsNormalizationIssueCode.MISSING_REQUIRED_VALUE,
        observed_value="",
    )


def test_source_data_readiness_records_are_explicit_and_not_quality():
    result = normalize_synthetic_fundamentals((_asml_raw_record(),))
    readiness = result.readiness_records[0]

    assert readiness.ticker == "ASML"
    assert readiness.fiscal_period == "FY"
    assert readiness.fiscal_year == "2025"
    assert readiness.readiness_state == "available"
    assert readiness.source_data_status == "available"
    assert readiness.source_reference == "synthetic/raw/asml/fy-2025"
    assert "quality" not in readiness.__dataclass_fields__


def test_missing_partial_invalid_source_missing_and_stale_states_remain_explicit():
    missing = _asml_raw_record(metrics={"revenue": ""})
    partial = _asml_raw_record(metrics={"revenue": "", "gross_margin": "51.3%"})
    invalid = _asml_raw_record(source_record_id="")
    source_missing = _asml_raw_record(metrics={})
    stale = _asml_raw_record(stale_metric_names=("revenue",))

    result = normalize_synthetic_fundamentals(
        (missing, partial, invalid, source_missing, stale)
    )

    assert tuple(record.readiness_state for record in result.readiness_records) == (
        "missing",
        "partial",
        "invalid",
        "source_missing",
        "stale",
    )


def test_invalid_or_incomplete_raw_records_produce_explicit_issues():
    result = normalize_synthetic_fundamentals(
        (_asml_raw_record(source_record_id=""),)
    )

    assert _has_issue(
        result,
        field_name="source_record_id",
        issue_code=FundamentalsNormalizationIssueCode.MISSING_REQUIRED_VALUE,
        observed_value="",
    )
    assert result.readiness_records[0].readiness_state == "invalid"


def test_adapter_records_do_not_create_decision_or_quality_authority_fields():
    dataclass_fields = (
        set(fundamentals_normalization_adapter.SyntheticRawFundamentalRecord.__dataclass_fields__)
        | set(
            fundamentals_normalization_adapter.SyntheticNormalizedFundamentalRecord.__dataclass_fields__
        )
        | set(
            fundamentals_normalization_adapter.SyntheticSourceDataReadinessRecord.__dataclass_fields__
        )
        | set(
            fundamentals_normalization_adapter.SyntheticFundamentalsNormalizationResult.__dataclass_fields__
        )
    )

    assert dataclass_fields.isdisjoint(FORBIDDEN_ADAPTER_FIELD_NAMES)


def test_adapter_source_does_not_import_legacy_or_network_modules():
    source = Path(fundamentals_normalization_adapter.__file__).read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "scripts",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "yfinance",
        "EDGAR",
    ):
        assert forbidden not in source


def test_adapter_import_and_execution_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(fundamentals_normalization_adapter)
    normalize_synthetic_fundamentals((_asml_raw_record(),))

    assert list(tmp_path.iterdir()) == []
    assert not Path("data/raw").exists()
    assert not Path("data/normalized").exists()
    assert not Path("data/generated").exists()
    assert not Path("data/processed").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()


def test_adapter_does_not_create_target_threshold_score_or_recommendation_data():
    result = normalize_synthetic_fundamentals((_asml_raw_record(),))

    for record in (*result.normalized_records, *result.readiness_records):
        fields = set(record.__dataclass_fields__)
        assert "target_price" not in fields
        assert "threshold_price" not in fields
        assert "score" not in fields
        assert "recommendation" not in fields


def test_full_synthetic_raw_to_normalized_flow():
    raw_record = _asml_raw_record()

    result = normalize_synthetic_fundamentals((raw_record,))

    assert tuple(record.metric_name for record in result.normalized_records) == (
        "revenue",
        "gross_margin",
        "free_cash_flow",
    )
    assert result.readiness_records[0].readiness_state == "available"
    assert result.readiness_records[0].missing_fundamentals_count == 0
    assert result.readiness_records[0].partial_data_count == 0
    assert result.readiness_records[0].stale_data_count == 0
    assert result.issues == ()
