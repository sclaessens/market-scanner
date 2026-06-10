from importlib import reload
from pathlib import Path

from market_scanner.fundamentals import fundamental_contracts
from market_scanner.fundamentals.fundamental_contracts import (
    FundamentalContractIssue,
    FundamentalContractIssueCode,
    fundamental_history_date_fields,
    fundamental_history_numeric_fields,
    supported_fundamental_history_periods,
    validate_fundamental_history_contract_records,
)


def _history_record(**overrides):
    record = {
        "ticker": "ASML",
        "fiscal_year": "2025",
        "fiscal_period": "FY",
        "period_end_date": "2025-12-31",
        "report_date": "2026-02-15",
        "currency": "EUR",
        "revenue": "1000",
        "gross_profit": "600",
        "operating_income": "300",
        "net_income": "200",
        "diluted_eps": "5",
        "total_debt": "150",
        "total_equity": "500",
        "free_cash_flow": "250",
        "source_name": "synthetic filing",
        "source_reference": "synthetic FY2025 filing",
        "source_freshness_date": "2026-05-08",
        "extraction_date": "2026-05-08",
        "notes": "",
    }
    record.update(overrides)
    return record


def test_history_validation_contract_exposes_bl81_numeric_date_and_period_fields():
    assert fundamental_history_numeric_fields() == (
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "diluted_eps",
        "total_debt",
        "total_equity",
        "free_cash_flow",
    )

    assert fundamental_history_date_fields() == (
        "period_end_date",
        "report_date",
        "source_freshness_date",
        "extraction_date",
    )

    assert supported_fundamental_history_periods() == (
        "FY",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "TTM",
    )


def test_complete_history_records_are_valid_contract_records():
    assert validate_fundamental_history_contract_records((_history_record(),)) == ()


def test_supported_fiscal_periods_are_case_insensitive():
    for fiscal_period in ("fy", "q1", "q2", "q3", "q4", "ttm"):
        assert (
            validate_fundamental_history_contract_records(
                (_history_record(fiscal_period=fiscal_period),)
            )
            == ()
        )


def test_unsupported_fiscal_period_is_reported_explicitly():
    assert validate_fundamental_history_contract_records(
        (_history_record(fiscal_period="H1"),)
    ) == (
        FundamentalContractIssue(
            field_name="fiscal_period",
            issue_code=FundamentalContractIssueCode.INVALID_FISCAL_PERIOD,
            observed_value="H1",
        ),
    )


def test_fiscal_year_must_parse_as_integer_in_allowed_range():
    assert validate_fundamental_history_contract_records(
        (_history_record(fiscal_year="not-a-year"),)
    ) == (
        FundamentalContractIssue(
            field_name="fiscal_year",
            issue_code=FundamentalContractIssueCode.INVALID_FISCAL_YEAR,
            observed_value="not-a-year",
        ),
    )

    assert validate_fundamental_history_contract_records(
        (_history_record(fiscal_year="1899"),)
    ) == (
        FundamentalContractIssue(
            field_name="fiscal_year",
            issue_code=FundamentalContractIssueCode.INVALID_FISCAL_YEAR,
            observed_value="1899",
        ),
    )

    assert validate_fundamental_history_contract_records(
        (_history_record(fiscal_year="2201"),)
    ) == (
        FundamentalContractIssue(
            field_name="fiscal_year",
            issue_code=FundamentalContractIssueCode.INVALID_FISCAL_YEAR,
            observed_value="2201",
        ),
    )


def test_date_fields_must_be_valid_iso_dates_when_present():
    assert validate_fundamental_history_contract_records(
        (_history_record(report_date="not-a-date"),)
    ) == (
        FundamentalContractIssue(
            field_name="report_date",
            issue_code=FundamentalContractIssueCode.INVALID_DATE_VALUE,
            observed_value="not-a-date",
        ),
    )


def test_multiple_invalid_date_fields_are_reported_by_column():
    assert validate_fundamental_history_contract_records(
        (
            _history_record(
                period_end_date="2025-99-99",
                source_freshness_date="fresh",
            ),
        )
    ) == (
        FundamentalContractIssue(
            field_name="period_end_date",
            issue_code=FundamentalContractIssueCode.INVALID_DATE_VALUE,
            observed_value="2025-99-99",
        ),
        FundamentalContractIssue(
            field_name="source_freshness_date",
            issue_code=FundamentalContractIssueCode.INVALID_DATE_VALUE,
            observed_value="fresh",
        ),
    )


def test_duplicate_history_keys_are_reported_after_normalization():
    first = _history_record(ticker="asml", fiscal_year="2025", fiscal_period="fy")
    second = _history_record(ticker="ASML", fiscal_year="2025", fiscal_period="FY")

    assert validate_fundamental_history_contract_records((first, second)) == (
        FundamentalContractIssue(
            field_name="ticker,fiscal_year,fiscal_period",
            issue_code=FundamentalContractIssueCode.DUPLICATE_HISTORY_KEY,
            observed_value="ASML|2025|FY",
        ),
    )


def test_required_values_are_still_reported_separately_from_invalid_values():
    assert validate_fundamental_history_contract_records(
        (_history_record(ticker="", fiscal_period=""),)
    ) == (
        FundamentalContractIssue(
            field_name="ticker",
            issue_code=FundamentalContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
        FundamentalContractIssue(
            field_name="fiscal_period",
            issue_code=FundamentalContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_numeric_validation_still_reports_non_numeric_values_without_scoring():
    assert validate_fundamental_history_contract_records(
        (_history_record(total_equity="not-a-number"),)
    ) == (
        FundamentalContractIssue(
            field_name="total_equity",
            issue_code=FundamentalContractIssueCode.INVALID_NUMERIC_VALUE,
            observed_value="not-a-number",
        ),
    )


def test_history_validation_contract_reports_forbidden_authority_fields():
    assert validate_fundamental_history_contract_records(
        (_history_record(final_action="BUY"),)
    ) == (
        FundamentalContractIssue(
            field_name="final_action",
            issue_code=FundamentalContractIssueCode.FORBIDDEN_FIELD,
            observed_value="BUY",
        ),
    )


def test_history_validation_contract_source_has_no_legacy_network_or_provider_imports():
    source = Path(fundamental_contracts.__file__).read_text(encoding="utf-8")

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


def test_history_validation_contract_helpers_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(fundamental_contracts)
    validate_fundamental_history_contract_records((_history_record(),))

    assert list(tmp_path.iterdir()) == []
    assert not Path("data/raw").exists()
    assert not Path("data/processed").exists()
    assert not Path("data/local").exists()
    assert not Path("reports").exists()