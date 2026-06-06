from importlib import reload
from pathlib import Path

from market_scanner.fundamentals import sec_companyfacts_smoke_boundary
from market_scanner.fundamentals.sec_companyfacts_smoke_boundary import (
    SEC_COMPANYFACTS_SOURCE_FAMILY,
    SecCompanyFactsFact,
    SecCompanyFactsSmokeInput,
    build_sec_companyfacts_smoke_result,
)


FORBIDDEN_SCRIPT_IMPORTS = (
    "scripts.fundamentals.sec_companyfacts_bulk_intake",
    "scripts.fundamentals.sec_companyfacts_transform",
    "scripts.data_sources.common",
    "scripts.data_sources.prefill_fundamentals",
    "scripts.fundamentals.build_quality",
)

FORBIDDEN_RUNTIME_TERMS = (
    "requests",
    "urlopen",
    "yfinance",
    "yf.",
    "os.environ",
    "getenv",
    "API_KEY",
    "TOKEN",
    "SECRET",
)

ZERO_LIKE_MISSING_VALUES = (0, 0.0, "0", False, "")


def _fact(
    concept: str,
    value: object,
    *,
    fiscal_year: str = "2025",
    fiscal_period: str = "FY",
    period_end_date: str = "2025-01-26",
    accession: str = "redacted-nvda-2025-10k",
    unit: str = "USD",
    currency: str = "USD",
    ticker: str = "NVDA",
    cik: str = "0001045810",
) -> SecCompanyFactsFact:
    return SecCompanyFactsFact(
        concept=concept,
        value=value,
        unit=unit,
        currency=currency,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_end_date=period_end_date,
        accession=accession,
        source_reference=(
            f"redacted-sec-companyfacts://{cik}/{concept}/{fiscal_year}"
        ),
        source_timestamp="2026-02-26T16:48:33Z",
        ticker=ticker,
        cik=cik,
    )


def _facts(**overrides):
    values = {
        "Revenues": (_fact("Revenues", "1200"),),
        "NetIncomeLoss": (_fact("NetIncomeLoss", "250"),),
        "OperatingIncomeLoss": (_fact("OperatingIncomeLoss", "300"),),
        "NetCashProvidedByUsedInOperatingActivities": (
            _fact("NetCashProvidedByUsedInOperatingActivities", "900"),
        ),
        "PaymentsToAcquirePropertyPlantAndEquipment": (
            _fact("PaymentsToAcquirePropertyPlantAndEquipment", "100"),
        ),
    }
    values.update(overrides)
    return values


def _prior_facts(**overrides):
    values = {
        "Revenues": (
            _fact(
                "Revenues",
                "1000",
                fiscal_year="2024",
                period_end_date="2024-01-28",
                accession="redacted-nvda-2024-10k",
            ),
        ),
        "NetIncomeLoss": (
            _fact(
                "NetIncomeLoss",
                "200",
                fiscal_year="2024",
                period_end_date="2024-01-28",
                accession="redacted-nvda-2024-10k",
            ),
        ),
        "OperatingIncomeLoss": (
            _fact(
                "OperatingIncomeLoss",
                "240",
                fiscal_year="2024",
                period_end_date="2024-01-28",
                accession="redacted-nvda-2024-10k",
            ),
        ),
        "NetCashProvidedByUsedInOperatingActivities": (
            _fact(
                "NetCashProvidedByUsedInOperatingActivities",
                "700",
                fiscal_year="2024",
                period_end_date="2024-01-28",
                accession="redacted-nvda-2024-10k",
            ),
        ),
        "PaymentsToAcquirePropertyPlantAndEquipment": (
            _fact(
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "100",
                fiscal_year="2024",
                period_end_date="2024-01-28",
                accession="redacted-nvda-2024-10k",
            ),
        ),
    }
    values.update(overrides)
    return values


def _input(**overrides) -> SecCompanyFactsSmokeInput:
    smoke_input = SecCompanyFactsSmokeInput(
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA Corporation",
        fiscal_year="2025",
        fiscal_period="FY",
        period_end_date="2025-01-26",
        retrieval_timestamp="2026-06-06T00:00:00Z",
        facts=_facts(),
        ticker_candidates=("NVDA",),
        cik_candidates=("0001045810",),
        prior_fiscal_year="2024",
        prior_facts=_prior_facts(),
    )
    values = {
        field_name: getattr(smoke_input, field_name)
        for field_name in smoke_input.__dataclass_fields__
    }
    values.update(overrides)
    return SecCompanyFactsSmokeInput(**values)


def _redacted_nvda_fact(
    concept: str,
    value: object,
    *,
    fiscal_year: str = "2025",
    fiscal_period: str = "FY",
    period_end_date: str = "2025-01-26",
    accession: str = "redacted-nvda-2025-10k-accession",
    frame: str = "CY2025",
    unit: str = "USD",
    currency: str = "USD",
    ticker: str = "NVDA",
    cik: str = "0001045810",
) -> SecCompanyFactsFact:
    return SecCompanyFactsFact(
        concept=concept,
        value=value,
        unit=unit,
        currency=currency,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_end_date=period_end_date,
        accession=accession,
        source_reference=(
            "redacted-sec-companyfacts://"
            f"CIK{cik}/{concept}/{fiscal_year}/{fiscal_period}/{frame}"
        ),
        source_timestamp="2025-02-26T16:48:33Z",
        ticker=ticker,
        cik=cik,
    )


def _redacted_nvda_source_shaped_facts(**overrides):
    values = {
        "Revenues": (
            _redacted_nvda_fact("Revenues", "1200"),
            _redacted_nvda_fact(
                "Revenues",
                "320",
                fiscal_period="Q1",
                frame="CY2025Q1",
            ),
        ),
        "NetIncomeLoss": (
            _redacted_nvda_fact("NetIncomeLoss", "250"),
        ),
        "OperatingIncomeLoss": (
            _redacted_nvda_fact("OperatingIncomeLoss", "300"),
        ),
        "NetCashProvidedByUsedInOperatingActivities": (
            _redacted_nvda_fact(
                "NetCashProvidedByUsedInOperatingActivities",
                "900",
            ),
        ),
        "PaymentsToAcquirePropertyPlantAndEquipment": (
            _redacted_nvda_fact(
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "100",
            ),
        ),
    }
    values.update(overrides)
    return values


def _redacted_nvda_prior_source_shaped_facts(**overrides):
    values = {
        "Revenues": (
            _redacted_nvda_fact(
                "Revenues",
                "1000",
                fiscal_year="2024",
                period_end_date="2024-01-28",
                accession="redacted-nvda-2024-10k-accession",
                frame="CY2024",
            ),
        ),
        "NetIncomeLoss": (
            _redacted_nvda_fact(
                "NetIncomeLoss",
                "200",
                fiscal_year="2024",
                period_end_date="2024-01-28",
                accession="redacted-nvda-2024-10k-accession",
                frame="CY2024",
            ),
        ),
        "OperatingIncomeLoss": (
            _redacted_nvda_fact(
                "OperatingIncomeLoss",
                "240",
                fiscal_year="2024",
                period_end_date="2024-01-28",
                accession="redacted-nvda-2024-10k-accession",
                frame="CY2024",
            ),
        ),
        "NetCashProvidedByUsedInOperatingActivities": (
            _redacted_nvda_fact(
                "NetCashProvidedByUsedInOperatingActivities",
                "700",
                fiscal_year="2024",
                period_end_date="2024-01-28",
                accession="redacted-nvda-2024-10k-accession",
                frame="CY2024",
            ),
        ),
        "PaymentsToAcquirePropertyPlantAndEquipment": (
            _redacted_nvda_fact(
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "100",
                fiscal_year="2024",
                period_end_date="2024-01-28",
                accession="redacted-nvda-2024-10k-accession",
                frame="CY2024",
            ),
        ),
    }
    values.update(overrides)
    return values


def _redacted_nvda_source_shaped_input(**overrides) -> SecCompanyFactsSmokeInput:
    smoke_input = SecCompanyFactsSmokeInput(
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA Corporation",
        fiscal_year="2025",
        fiscal_period="FY",
        period_end_date="2025-01-26",
        retrieval_timestamp="2026-06-06T00:00:00Z",
        facts=_redacted_nvda_source_shaped_facts(),
        ticker_candidates=("NVDA",),
        cik_candidates=("0001045810",),
        prior_fiscal_year="2024",
        prior_facts=_redacted_nvda_prior_source_shaped_facts(),
    )
    values = {
        field_name: getattr(smoke_input, field_name)
        for field_name in smoke_input.__dataclass_fields__
    }
    values.update(overrides)
    return SecCompanyFactsSmokeInput(**values)


def _metric_map(result):
    assert result.ingestion_result is not None
    return {
        record.metric_name: record
        for record in result.ingestion_result.normalized_records
    }


def _growth_map(result):
    return {record.metric_name: record for record in result.growth_evidence}


def test_redacted_nvda_source_shaped_evidence_is_accepted():
    result = build_sec_companyfacts_smoke_result(
        _redacted_nvda_source_shaped_input()
    )
    metrics = _metric_map(result)

    assert result.smoke_status == "passed"
    assert result.source_family == SEC_COMPANYFACTS_SOURCE_FAMILY
    assert result.provider_name == "SEC CompanyFacts"
    assert result.ticker == "NVDA"
    assert result.cik == "0001045810"
    assert result.company_name == "NVIDIA Corporation"
    assert result.ingestion_result is not None
    assert result.ingestion_result.raw_evidence.entity_identifier == "0001045810"
    assert result.ingestion_result.readiness_record.readiness_state == "available"
    assert result.ingestion_result.readiness_record.source_data_status == "available"
    assert result.ingestion_result.raw_evidence.reported_period == "FY"
    assert result.ingestion_result.raw_evidence.fiscal_year == "2025"
    assert "period_end_date=2025-01-26" in (
        result.ingestion_result.raw_evidence.provenance_metadata
    )
    assert "FreeCashFlow" in result.ingestion_result.raw_evidence.missing_field_evidence

    assert metrics["revenue"].metric_value == "1200"
    assert metrics["net_income"].metric_value == "250"
    assert metrics["operating_income"].metric_value == "300"
    assert metrics["operating_cash_flow"].metric_value == "900"
    assert metrics["capital_expenditures"].metric_value == "100"


def test_redacted_nvda_source_shaped_fact_selection_is_deterministic():
    result = build_sec_companyfacts_smoke_result(
        _redacted_nvda_source_shaped_input()
    )
    selected = {
        selection.metric_name: selection.selected_fact
        for selection in result.fact_selection
    }

    assert selected["revenue"] is not None
    assert selected["revenue"].fiscal_period == "FY"
    assert selected["revenue"].source_reference.endswith("/2025/FY/CY2025")
    assert selected["revenue"].accession == "redacted-nvda-2025-10k-accession"
    assert selected["net_income"] is not None
    assert selected["net_income"].concept == "NetIncomeLoss"
    assert selected["free_cash_flow"] is None


def test_redacted_nvda_free_cash_flow_is_source_derived():
    result = build_sec_companyfacts_smoke_result(
        _redacted_nvda_source_shaped_input()
    )
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value == "800"
    assert free_cash_flow.normalization_status == "source_derived"
    assert free_cash_flow.derivation_formula == (
        "free_cash_flow = operating_cash_flow - capital_expenditures"
    )
    assert free_cash_flow.source_field_names == (
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
    )
    assert free_cash_flow.metric_value not in ZERO_LIKE_MISSING_VALUES


def test_redacted_nvda_prior_year_growth_evidence_is_available():
    result = build_sec_companyfacts_smoke_result(
        _redacted_nvda_source_shaped_input()
    )
    growth = _growth_map(result)

    assert growth["revenue"].growth_status == "growth_available"
    assert growth["free_cash_flow"].growth_status == "growth_available"
    assert growth["net_income"].growth_status == "growth_available"
    assert growth["operating_income"].growth_status == "growth_available"
    assert growth["revenue"].current_period_reference == "2025|FY"
    assert growth["revenue"].prior_period_reference == "2024|FY"
    assert growth["free_cash_flow"].current_source_field_names == (
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
    )
    assert growth["free_cash_flow"].prior_source_field_names == (
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
    )


def test_redacted_nvda_cik_mismatch_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _redacted_nvda_source_shaped_input(
            facts=_redacted_nvda_source_shaped_facts(
                Revenues=(
                    _redacted_nvda_fact(
                        "Revenues",
                        "1200",
                        cik="0000000000",
                    ),
                )
            )
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "ambiguous_cik" in result.issues


def test_redacted_nvda_missing_fiscal_context_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _redacted_nvda_source_shaped_input(fiscal_period="")
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "missing_required_fiscal_context" in result.issues


def test_redacted_nvda_ambiguous_annual_facts_fail_closed():
    result = build_sec_companyfacts_smoke_result(
        _redacted_nvda_source_shaped_input(
            facts=_redacted_nvda_source_shaped_facts(
                Revenues=(
                    _redacted_nvda_fact("Revenues", "1200"),
                    _redacted_nvda_fact(
                        "Revenues",
                        "1201",
                        accession="redacted-nvda-amended-2025-10k-accession",
                    ),
                )
            )
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "revenue:ambiguous_fact_candidates" in result.issues


def test_redacted_nvda_period_mismatch_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _redacted_nvda_source_shaped_input(
            facts=_redacted_nvda_source_shaped_facts(
                NetIncomeLoss=(
                    _redacted_nvda_fact(
                        "NetIncomeLoss",
                        "250",
                        fiscal_period="Q4",
                        frame="CY2025Q4",
                    ),
                )
            )
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "net_income:missing_fact" in result.issues


def test_sec_companyfacts_smoke_accepts_one_ticker_nvda_shape():
    result = build_sec_companyfacts_smoke_result(_input())

    assert result.smoke_status == "passed"
    assert result.ticker == "NVDA"
    assert result.cik == "0001045810"
    assert result.source_family == SEC_COMPANYFACTS_SOURCE_FAMILY
    assert result.provider_name == "SEC CompanyFacts"
    assert result.company_name == "NVIDIA Corporation"
    assert result.issues == ()
    assert result.ingestion_result is not None
    assert result.ingestion_result.raw_evidence.entity_identifier == "0001045810"
    assert "company=NVIDIA Corporation" in (
        result.ingestion_result.raw_evidence.provenance_metadata
    )


def test_fact_selection_is_deterministic_and_preserves_source_provenance():
    result = build_sec_companyfacts_smoke_result(_input())
    selected = {
        selection.metric_name: selection.selected_fact
        for selection in result.fact_selection
    }

    assert selected["revenue"] is not None
    assert selected["revenue"].concept == "Revenues"
    assert selected["net_income"] is not None
    assert selected["net_income"].accession == "redacted-nvda-2025-10k"
    assert selected["free_cash_flow"] is None

    revenue = _metric_map(result)["revenue"]
    assert revenue.source_reference.startswith("sec-companyfacts-smoke:")
    assert revenue.source_record_identity.startswith(
        "0001045810-redacted-nvda-2025-10k"
    )
    assert revenue.source_field_names == ("Revenues",)


def test_free_cash_flow_is_source_derived_when_direct_fact_is_absent():
    result = build_sec_companyfacts_smoke_result(_input())
    free_cash_flow = _metric_map(result)["free_cash_flow"]

    assert free_cash_flow.metric_value == "800"
    assert free_cash_flow.normalization_status == "source_derived"
    assert free_cash_flow.derivation_formula == (
        "free_cash_flow = operating_cash_flow - capital_expenditures"
    )
    assert free_cash_flow.source_field_names == (
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
    )
    assert free_cash_flow.metric_value not in ZERO_LIKE_MISSING_VALUES


def test_prior_year_growth_evidence_is_available_for_comparable_facts():
    result = build_sec_companyfacts_smoke_result(_input())
    growth = _growth_map(result)

    assert growth["revenue"].growth_status == "growth_available"
    assert growth["free_cash_flow"].growth_status == "growth_available"
    assert growth["net_income"].growth_status == "growth_available"
    assert growth["operating_income"].growth_status == "growth_available"
    assert growth["free_cash_flow"].current_source_field_names == (
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
    )
    assert growth["free_cash_flow"].prior_source_field_names == (
        "NetCashProvidedByUsedInOperatingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
    )


def test_missingness_remains_explicit_when_fact_cannot_be_derived():
    result = build_sec_companyfacts_smoke_result(
        _input(
            facts=_facts(
                NetCashProvidedByUsedInOperatingActivities=(),
            )
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "operating_cash_flow:missing_fact" in result.issues


def test_wrong_ticker_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(ticker="AAPL", ticker_candidates=("AAPL",)),
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "ticker_mismatch" in result.issues


def test_missing_cik_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(cik="", cik_candidates=()),
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "missing_cik" in result.issues


def test_multi_ticker_input_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(ticker_candidates=("NVDA", "AAPL")),
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "multi_ticker_input" in result.issues


def test_ambiguous_cik_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(cik_candidates=("0001045810", "0000000000")),
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "ambiguous_cik" in result.issues


def test_ambiguous_facts_fail_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(
            facts=_facts(
                Revenues=(
                    _fact("Revenues", "1200"),
                    _fact("Revenues", "1201"),
                ),
            )
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "revenue:ambiguous_fact_candidates" in result.issues


def test_unit_mismatch_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(
            facts=_facts(
                NetIncomeLoss=(_fact("NetIncomeLoss", "250", unit="shares"),),
            )
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "net_income:unit_mismatch" in result.issues


def test_currency_mismatch_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(
            facts=_facts(
                NetIncomeLoss=(_fact("NetIncomeLoss", "250", currency="EUR"),),
            )
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "net_income:currency_mismatch" in result.issues


def test_period_mismatch_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(
            facts=_facts(
                NetIncomeLoss=(_fact("NetIncomeLoss", "250", fiscal_period="Q1"),),
            )
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "net_income:missing_fact" in result.issues


def test_missing_provenance_fails_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(
            facts=_facts(
                NetIncomeLoss=(
                    _fact("NetIncomeLoss", "250", accession=""),
                ),
            )
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "net_income:missing_provenance" in result.issues


def test_non_numeric_fact_values_fail_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(
            facts=_facts(
                Revenues=(_fact("Revenues", "not-a-number"),),
            )
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert "revenue:non_numeric_fact_value" in result.issues


def test_wrong_source_family_and_live_modes_fail_closed():
    result = build_sec_companyfacts_smoke_result(
        _input(
            source_family="yfinance",
            attempted_live_mode=True,
            production_persistence_requested=True,
        )
    )

    assert result.smoke_status == "review_required"
    assert result.ingestion_result is None
    assert result.issues == (
        "attempted_live_or_network_mode",
        "attempted_production_persistence",
        "wrong_source_family",
    )


def test_import_and_smoke_boundary_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(sec_companyfacts_smoke_boundary)
    result = build_sec_companyfacts_smoke_result(_input())

    assert result.smoke_status == "passed"
    assert list(tmp_path.iterdir()) == []
    assert not Path("data").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()


def test_smoke_boundary_has_no_network_credentials_or_script_imports():
    source = Path(sec_companyfacts_smoke_boundary.__file__).read_text(
        encoding="utf-8"
    )

    for forbidden in (*FORBIDDEN_SCRIPT_IMPORTS, *FORBIDDEN_RUNTIME_TERMS):
        assert forbidden not in source


def test_smoke_boundary_records_do_not_expose_investment_authority():
    result = build_sec_companyfacts_smoke_result(_input())
    rendered = " ".join(str(value) for value in result.__dict__.values())

    for forbidden in (
        "allocation",
        "conviction",
        "urgency",
        "target_price",
        "tradeability",
        "recommendation",
    ):
        assert forbidden not in rendered
