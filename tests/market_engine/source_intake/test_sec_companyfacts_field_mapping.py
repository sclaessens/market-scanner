from __future__ import annotations

from dataclasses import asdict

from market_engine.source_intake.sec_companyfacts_fields import (
    SEC_COMPANYFACTS_REQUIRED_FIELDS,
    map_sec_companyfacts_fields,
)


def test_revenues_maps_to_revenue():
    mapped = map_sec_companyfacts_fields(_payload({"Revenues": [_fact(100, "2025-12-31")]}))

    assert mapped["revenue"] is not None
    assert mapped["revenue"].raw_value == 100
    assert mapped["revenue"].sec_tag_selected == "Revenues"


def test_contract_revenue_fallback_priority_is_deterministic():
    mapped = map_sec_companyfacts_fields(
        _payload(
            {
                "RevenueFromContractWithCustomerExcludingAssessedTax": [_fact(90, "2025-12-31")],
                "SalesRevenueNet": [_fact(80, "2025-12-31")],
            }
        )
    )

    assert mapped["revenue"] is not None
    assert mapped["revenue"].raw_value == 90
    assert mapped["revenue"].sec_tag_selected == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert mapped["revenue"].fallback_alias_used == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_sales_revenue_net_maps_only_when_higher_priority_revenue_tags_are_missing():
    mapped = map_sec_companyfacts_fields(_payload({"SalesRevenueNet": [_fact(80, "2025-12-31")]}))

    assert mapped["revenue"] is not None
    assert mapped["revenue"].raw_value == 80
    assert mapped["revenue"].sec_tag_selected == "SalesRevenueNet"


def test_revenue_aliases_are_not_summed_or_combined():
    mapped = map_sec_companyfacts_fields(
        _payload(
            {
                "SalesRevenueGoodsNet": [_fact(40, "2025-12-31")],
                "SalesRevenueServicesNet": [_fact(60, "2025-12-31")],
            }
        )
    )

    assert mapped["revenue"] is not None
    assert mapped["revenue"].raw_value == 40
    assert mapped["revenue"].sec_tag_selected == "SalesRevenueGoodsNet"


def test_net_income_primary_and_fallback_rules():
    primary = map_sec_companyfacts_fields(
        _payload(
            {
                "NetIncomeLoss": [_fact(20, "2025-12-31")],
                "ProfitLoss": [_fact(18, "2025-12-31")],
            }
        )
    )
    fallback = map_sec_companyfacts_fields(_payload({"ProfitLoss": [_fact(18, "2025-12-31")]}))

    assert primary["net_income"] is not None
    assert primary["net_income"].raw_value == 20
    assert primary["net_income"].sec_tag_selected == "NetIncomeLoss"
    assert fallback["net_income"] is not None
    assert fallback["net_income"].raw_value == 18
    assert fallback["net_income"].sec_tag_selected == "ProfitLoss"
    assert fallback["net_income"].fallback_alias_used == "ProfitLoss"


def test_unapproved_net_income_tags_are_not_selected():
    mapped = map_sec_companyfacts_fields(
        _payload(
            {
                "NetIncomeLossAvailableToCommonStockholdersBasic": [_fact(12, "2025-12-31")],
                "ComprehensiveIncomeNetOfTax": [_fact(15, "2025-12-31")],
            }
        )
    )

    assert mapped["net_income"] is None


def test_operating_cash_flow_primary_and_fallback_rules():
    primary = map_sec_companyfacts_fields(
        _payload(
            {
                "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": [
                    _fact(28, "2025-12-31")
                ],
            }
        )
    )
    fallback = map_sec_companyfacts_fields(
        _payload(
            {
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": [
                    _fact(28, "2025-12-31")
                ]
            }
        )
    )

    assert primary["operating_cash_flow"] is not None
    assert primary["operating_cash_flow"].raw_value == 30
    assert primary["operating_cash_flow"].sec_tag_selected == "NetCashProvidedByUsedInOperatingActivities"
    assert fallback["operating_cash_flow"] is not None
    assert fallback["operating_cash_flow"].raw_value == 28
    assert (
        fallback["operating_cash_flow"].sec_tag_selected
        == "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
    )
    assert (
        fallback["operating_cash_flow"].fallback_alias_used
        == "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
    )


def test_investing_cash_flow_is_not_accepted_as_operating_cash_flow():
    mapped = map_sec_companyfacts_fields(
        _payload({"NetCashProvidedByUsedInInvestingActivities": [_fact(-10, "2025-12-31")]})
    )

    assert mapped["operating_cash_flow"] is None


def test_capital_expenditures_primary_and_fallback_rules():
    primary = map_sec_companyfacts_fields(
        _payload(
            {
                "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
                "PaymentsToAcquireProductiveAssets": [_fact(6, "2025-12-31")],
            }
        )
    )
    fallback = map_sec_companyfacts_fields(
        _payload({"PaymentsToAcquireProductiveAssets": [_fact(6, "2025-12-31")]})
    )

    assert primary["capital_expenditures"] is not None
    assert primary["capital_expenditures"].raw_value == 5
    assert (
        primary["capital_expenditures"].sec_tag_selected
        == "PaymentsToAcquirePropertyPlantAndEquipment"
    )
    assert fallback["capital_expenditures"] is not None
    assert fallback["capital_expenditures"].raw_value == 6
    assert fallback["capital_expenditures"].sec_tag_selected == "PaymentsToAcquireProductiveAssets"


def test_future_review_capex_aliases_are_not_selected():
    mapped = map_sec_companyfacts_fields(
        _payload(
            {
                "PaymentsToAcquirePropertyPlantAndEquipmentAndOtherProductiveAssets": [
                    _fact(7, "2025-12-31")
                ],
                "PaymentsToAcquireProductiveAssetsAndBusinesses": [_fact(8, "2025-12-31")],
            }
        )
    )

    assert mapped["capital_expenditures"] is None


def test_missing_fields_remain_missing_and_do_not_become_zero():
    mapped = map_sec_companyfacts_fields(_payload({"Revenues": [_fact(0, "2025-12-31")]}))

    assert mapped["revenue"] is not None
    assert mapped["revenue"].raw_value == 0
    assert mapped["net_income"] is None
    assert mapped["operating_cash_flow"] is None
    assert mapped["capital_expenditures"] is None


def test_selected_tag_unit_filing_and_period_metadata_are_preserved():
    mapped = map_sec_companyfacts_fields(_payload({"Revenues": [_fact(100, "2025-12-31")]}))

    revenue = mapped["revenue"]
    assert revenue is not None
    assert revenue.sec_tag_selected == "Revenues"
    assert revenue.unit == "USD"
    assert revenue.fiscal_year == 2025
    assert revenue.fiscal_period == "FY"
    assert revenue.filing_form == "10-K"
    assert revenue.filing_date == "2026-02-15"
    assert revenue.period_start_date == "2025-01-01"
    assert revenue.period_end_date == "2025-12-31"
    assert revenue.accession_number == "0000000000-2025-000001"
    assert revenue.frame == "CY2025"


def test_foreign_issuer_us_gaap_20f_eur_facts_are_preserved():
    mapped = map_sec_companyfacts_fields(
        {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {"EUR": [_fact(32667300000, "2025-12-31", form="20-F")]}
                    },
                    "NetIncomeLoss": {
                        "units": {"EUR": [_fact(9609400000, "2025-12-31", form="20-F")]}
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {"EUR": [_fact(12658500000, "2025-12-31", form="20-F")]}
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {"EUR": [_fact(1573600000, "2025-12-31", form="20-F")]}
                    },
                }
            }
        }
    )

    assert mapped["revenue"] is not None
    assert mapped["revenue"].sec_tag_selected == (
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    )
    assert mapped["revenue"].taxonomy_namespace == "us-gaap"
    assert mapped["revenue"].unit == "EUR"
    assert mapped["revenue"].filing_form == "20-F"
    assert mapped["capital_expenditures"] is not None
    assert mapped["capital_expenditures"].unit == "EUR"


def test_ifrs_20f_usd_facts_are_preserved_without_currency_conversion():
    mapped = map_sec_companyfacts_fields(
        {
            "facts": {
                "ifrs-full": {
                    "Revenue": {
                        "units": {"USD": [_fact(88268000000, "2024-12-31", form="20-F")]}
                    },
                    "ProfitLoss": {
                        "units": {"USD": [_fact(35301100000, "2024-12-31", form="20-F")]}
                    },
                    "CashFlowsFromUsedInOperatingActivities": {
                        "units": {"USD": [_fact(55693100000, "2024-12-31", form="20-F")]}
                    },
                    "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": {
                        "units": {"USD": [_fact(29155400000, "2024-12-31", form="20-F")]}
                    },
                }
            }
        }
    )

    assert mapped["revenue"] is not None
    assert mapped["revenue"].sec_tag_selected == "Revenue"
    assert mapped["revenue"].taxonomy_namespace == "ifrs-full"
    assert mapped["revenue"].unit == "USD"
    assert mapped["revenue"].raw_value == 88268000000
    assert mapped["net_income"] is not None
    assert mapped["net_income"].fallback_alias_used == "ProfitLoss"
    assert mapped["operating_cash_flow"] is not None
    assert mapped["operating_cash_flow"].sec_tag_selected == (
        "CashFlowsFromUsedInOperatingActivities"
    )
    assert mapped["capital_expenditures"] is not None
    assert mapped["capital_expenditures"].sec_tag_selected == (
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"
    )


def test_mapping_does_not_include_forbidden_authority_fields():
    mapped = map_sec_companyfacts_fields(_complete_payload())
    payload = {field: asdict(value) if value is not None else None for field, value in mapped.items()}

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
    assert forbidden_fields.isdisjoint(payload)


def test_approved_required_fields_are_stable():
    assert SEC_COMPANYFACTS_REQUIRED_FIELDS == (
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capital_expenditures",
    )


def _complete_payload() -> dict[str, object]:
    return _payload(
        {
            "Revenues": [_fact(100, "2025-12-31")],
            "NetIncomeLoss": [_fact(20, "2025-12-31")],
            "NetCashProvidedByUsedInOperatingActivities": [_fact(30, "2025-12-31")],
            "PaymentsToAcquirePropertyPlantAndEquipment": [_fact(5, "2025-12-31")],
        }
    )


def _payload(facts: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": values}}
                for tag, values in facts.items()
            }
        }
    }


def _fact(value: int | None, end: str, *, form: str = "10-K") -> dict[str, object]:
    return {
        "val": value,
        "fy": int(end[:4]),
        "fp": "FY",
        "form": form,
        "filed": f"{int(end[:4]) + 1}-02-15",
        "start": f"{end[:4]}-01-01",
        "end": end,
        "accn": f"0000000000-{end[:4]}-000001",
        "frame": f"CY{end[:4]}",
    }
