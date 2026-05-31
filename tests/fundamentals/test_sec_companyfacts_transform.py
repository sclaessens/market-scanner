from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.fundamentals import sec_companyfacts_transform as transform
from scripts.fundamentals.build_history_intake import REQUIRED_COLUMNS, validate_fundamentals_history


def _fact(value: int | float, *, fy: int = 2024, fp: str = "FY", end: str = "2024-12-31", unit_fields: dict | None = None) -> dict:
    base = {
        "val": value,
        "fy": fy,
        "fp": fp,
        "end": end,
        "filed": "2025-02-15",
        "form": "10-K",
        "frame": f"CY{fy}",
        "accn": f"0000000000-{fy}-000001",
    }
    if unit_fields:
        base.update(unit_fields)
    return base


def _companyfacts_payload(facts: dict[str, dict[str, list[dict]]]) -> dict:
    payload: dict = {"cik": 320193, "entityName": "Synthetic Company", "facts": {"us-gaap": {}}}
    for tag, units in facts.items():
        payload["facts"]["us-gaap"][tag] = {"units": units}
    return payload


def _base_payload() -> dict:
    return _companyfacts_payload(
        {
            "Revenues": {"USD": [_fact(1000)]},
            "GrossProfit": {"USD": [_fact(420)]},
            "OperatingIncomeLoss": {"USD": [_fact(210)]},
            "NetIncomeLoss": {"USD": [_fact(155)]},
            "StockholdersEquity": {"USD": [_fact(900)]},
            "EarningsPerShareDiluted": {"USD/shares": [_fact(3.14)]},
            "DebtCurrent": {"USD": [_fact(111)]},
            "NetCashProvidedByUsedInOperatingActivities": {"USD": [_fact(222)]},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"USD": [_fact(33)]},
        }
    )


def _transform(payload: dict) -> pd.DataFrame:
    return transform.transform_companyfacts_payload(
        payload,
        ticker="test",
        cik="320193",
        source_freshness_date="2026-05-31",
        extraction_date="2026-05-31",
    )


def test_transform_minimal_valid_companyfacts_fixture_for_one_ticker(tmp_path: Path) -> None:
    df = _transform(_base_payload())
    output_path = tmp_path / "fundamentals_history.csv"
    df.to_csv(output_path, index=False)

    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "TEST"
    assert row["fiscal_year"] == "2024"
    assert row["fiscal_period"] == "FY"
    assert row["period_end_date"] == "2024-12-31"
    assert row["source_name"] == transform.SOURCE_NAME
    assert validate_fundamentals_history(output_path)["status"] == "VALID"


def test_revenue_candidate_tag_selection_uses_primary_before_alternate() -> None:
    payload = _companyfacts_payload(
        {
            "Revenues": {"USD": [_fact(1000)]},
            "SalesRevenueNet": {"USD": [_fact(950)]},
        }
    )

    row = _transform(payload).iloc[0]

    notes = json.loads(row["notes"])
    assert row["revenue"] == "1000"
    assert notes["evidence"]["revenue"]["source_tag"] == "us-gaap:Revenues"
    assert "revenue: selected us-gaap:Revenues by candidate order" in "|".join(notes["review_notes"])


def test_alternate_revenue_tag_selection_when_primary_missing() -> None:
    payload = _companyfacts_payload({"SalesRevenueNet": {"USD": [_fact(950)]}})

    row = _transform(payload).iloc[0]

    assert row["revenue"] == "950"
    assert json.loads(row["notes"])["evidence"]["revenue"]["source_tag"] == "us-gaap:SalesRevenueNet"


def test_gross_profit_missing_is_preserved_without_dropping_row() -> None:
    payload = _companyfacts_payload(
        {
            "Revenues": {"USD": [_fact(1000)]},
            "OperatingIncomeLoss": {"USD": [_fact(210)]},
            "NetIncomeLoss": {"USD": [_fact(155)]},
        }
    )

    row = _transform(payload).iloc[0]

    assert row["gross_profit"] == ""
    assert row["revenue"] == "1000"
    assert "gross_profit: missing optional" in "|".join(json.loads(row["notes"])["review_notes"])


def test_operating_income_and_net_income_direct_mapping() -> None:
    row = _transform(_base_payload()).iloc[0]

    assert row["operating_income"] == "210"
    assert row["net_income"] == "155"
    evidence = json.loads(row["notes"])["evidence"]
    assert evidence["operating_income"]["source_tag"] == "us-gaap:OperatingIncomeLoss"
    assert evidence["net_income"]["source_tag"] == "us-gaap:NetIncomeLoss"


def test_total_equity_instant_fact_mapping_to_period_end() -> None:
    row = _transform(_base_payload()).iloc[0]

    assert row["total_equity"] == "900"
    assert row["period_end_date"] == "2024-12-31"
    evidence = json.loads(row["notes"])["evidence"]["total_equity"]
    assert evidence["source_tag"] == "us-gaap:StockholdersEquity"
    assert evidence["period_end_date"] == "2024-12-31"


def test_diluted_eps_maps_only_with_per_share_unit_and_review_note() -> None:
    row = _transform(_base_payload()).iloc[0]

    assert row["diluted_eps"] == "3.14"
    notes = json.loads(row["notes"])
    assert notes["evidence"]["diluted_eps"]["unit"] == "USD/shares"
    assert "diluted_eps: review-required per-share field" in "|".join(notes["review_notes"])


def test_total_debt_remains_blank_and_is_not_derived() -> None:
    row = _transform(_base_payload()).iloc[0]

    assert row["total_debt"] == ""
    assert "missing current or noncurrent simple debt component" in "|".join(json.loads(row["notes"])["review_notes"])


def test_free_cash_flow_not_derived_when_capex_is_missing() -> None:
    payload = _companyfacts_payload(
        {
            "Revenues": {"USD": [_fact(1000)]},
            "NetCashProvidedByUsedInOperatingActivities": {"USD": [_fact(222)]},
        }
    )

    row = _transform(payload).iloc[0]

    assert row["free_cash_flow"] == ""
    assert "free_cash_flow: missing capital expenditure component" in "|".join(json.loads(row["notes"])["review_notes"])


def test_missing_values_are_not_treated_as_zero() -> None:
    payload = _companyfacts_payload({"Revenues": {"USD": [_fact(0)]}})

    row = _transform(payload).iloc[0]

    assert row["revenue"] == "0"
    assert row["gross_profit"] == ""
    assert row["total_debt"] == ""
    assert row["free_cash_flow"] == ""


def test_total_debt_derived_from_clean_current_and_noncurrent_debt_fixture() -> None:
    payload = _companyfacts_payload(
        {
            "DebtCurrent": {"USD": [_fact(100)]},
            "LongTermDebtNoncurrent": {"USD": [_fact(900)]},
        }
    )

    row = _transform(payload).iloc[0]
    notes = json.loads(row["notes"])

    assert row["total_debt"] == "1000"
    assert notes["evidence"]["total_debt"]["formula_version"] == "SEC-6C_TOTAL_DEBT_SIMPLE_V1"
    assert notes["evidence"]["total_debt"]["components"]["current_debt"]["source_tag"] == "us-gaap:DebtCurrent"
    assert notes["evidence"]["total_debt"]["components"]["noncurrent_debt"]["source_tag"] == "us-gaap:LongTermDebtNoncurrent"


def test_total_debt_derived_from_clean_lease_inclusive_current_and_noncurrent_debt_fixture() -> None:
    payload = _companyfacts_payload(
        {
            "LongTermDebtAndFinanceLeaseObligationsCurrent": {"USD": [_fact(125)]},
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent": {"USD": [_fact(875)]},
        }
    )

    row = _transform(payload).iloc[0]
    notes = json.loads(row["notes"])

    assert row["total_debt"] == "1000"
    assert notes["evidence"]["total_debt"]["formula_version"] == "SEC-6C_TOTAL_DEBT_LEASE_INCLUSIVE_V1"
    assert (
        notes["evidence"]["total_debt"]["components"]["lease_inclusive_current_debt"]["source_tag"]
        == "us-gaap:LongTermDebtAndFinanceLeaseObligationsCurrent"
    )


def test_total_debt_blocked_when_lease_inclusive_and_lease_exclusive_families_overlap() -> None:
    payload = _companyfacts_payload(
        {
            "DebtCurrent": {"USD": [_fact(100)]},
            "LongTermDebtNoncurrent": {"USD": [_fact(900)]},
            "LongTermDebtAndFinanceLeaseObligationsCurrent": {"USD": [_fact(125)]},
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent": {"USD": [_fact(875)]},
        }
    )

    row = _transform(payload).iloc[0]

    assert row["total_debt"] == ""
    assert "simple and lease-inclusive debt component families overlap" in "|".join(json.loads(row["notes"])["review_notes"])


def test_total_debt_not_derived_when_components_are_insufficient() -> None:
    payload = _companyfacts_payload({"LongTermDebtNoncurrent": {"USD": [_fact(900)]}})

    row = _transform(payload).iloc[0]

    assert row["total_debt"] == ""
    assert "missing current or noncurrent simple debt component" in "|".join(json.loads(row["notes"])["review_notes"])


def test_short_term_borrowings_and_finance_leases_are_not_silently_mixed_into_debt() -> None:
    payload = _companyfacts_payload(
        {
            "DebtCurrent": {"USD": [_fact(100)]},
            "LongTermDebtNoncurrent": {"USD": [_fact(900)]},
            "ShortTermBorrowings": {"USD": [_fact(50)]},
            "FinanceLeaseLiabilityCurrent": {"USD": [_fact(25)]},
            "FinanceLeaseLiabilityNoncurrent": {"USD": [_fact(75)]},
        }
    )

    row = _transform(payload).iloc[0]
    review_notes = "|".join(json.loads(row["notes"])["review_notes"])

    assert row["total_debt"] == ""
    assert "review-required components present and not mixed automatically" in review_notes


def test_free_cash_flow_derived_from_operating_cash_flow_and_positive_capex_outflow() -> None:
    row = _transform(_base_payload()).iloc[0]
    notes = json.loads(row["notes"])

    assert row["free_cash_flow"] == "189"
    assert notes["evidence"]["free_cash_flow"]["formula_version"] == "SEC-6C_FREE_CASH_FLOW_V1"
    assert notes["evidence"]["free_cash_flow"]["components"]["operating_cash_flow"]["source_tag"] == (
        "us-gaap:NetCashProvidedByUsedInOperatingActivities"
    )
    assert notes["evidence"]["free_cash_flow"]["components"]["capital_expenditures"]["source_tag"] == (
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
    )


def test_free_cash_flow_negative_capex_is_treated_as_already_signed_cash_flow() -> None:
    payload = _companyfacts_payload(
        {
            "NetCashProvidedByUsedInOperatingActivities": {"USD": [_fact(222)]},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"USD": [_fact(-33)]},
        }
    )

    row = _transform(payload).iloc[0]

    assert row["free_cash_flow"] == "189"
    assert "already signed negative capex" in "|".join(json.loads(row["notes"])["review_notes"])


def test_free_cash_flow_not_derived_when_operating_cash_flow_is_missing() -> None:
    payload = _companyfacts_payload({"PaymentsToAcquirePropertyPlantAndEquipment": {"USD": [_fact(33)]}})

    row = _transform(payload).iloc[0]

    assert row["free_cash_flow"] == ""
    assert "missing operating cash flow component" in "|".join(json.loads(row["notes"])["review_notes"])


def test_deterministic_output_ordering() -> None:
    payload = _companyfacts_payload(
        {
            "Revenues": {
                "USD": [
                    _fact(2000, fy=2025, end="2025-12-31"),
                    _fact(1000, fy=2024, end="2024-12-31"),
                    _fact(250, fy=2025, fp="Q1", end="2025-03-31"),
                ]
            }
        }
    )

    df = _transform(payload)

    assert list(zip(df["fiscal_year"], df["fiscal_period"], df["period_end_date"])) == [
        ("2024", "FY", "2024-12-31"),
        ("2025", "FY", "2025-12-31"),
        ("2025", "Q1", "2025-03-31"),
    ]


def test_source_evidence_is_recorded() -> None:
    row = _transform(_base_payload()).iloc[0]

    notes = json.loads(row["notes"])
    assert row["source_reference"].startswith("sec-companyfacts:CIK0000320193")
    assert notes["evidence"]["revenue"]["source_tag"] == "us-gaap:Revenues"
    assert notes["evidence"]["revenue"]["unit"] == "USD"
    assert notes["evidence"]["revenue"]["fiscal_year"] == 2024
    assert row["source_freshness_date"] == "2026-05-31"
    assert row["extraction_date"] == "2026-05-31"


def test_derived_source_evidence_and_notes_are_recorded() -> None:
    row = _transform(
        _companyfacts_payload(
            {
                "DebtCurrent": {"USD": [_fact(100)]},
                "LongTermDebtNoncurrent": {"USD": [_fact(900)]},
                "NetCashProvidedByUsedInOperatingActivities": {"USD": [_fact(222)]},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"USD": [_fact(33)]},
            }
        )
    ).iloc[0]

    notes = json.loads(row["notes"])
    assert notes["evidence"]["total_debt"]["formula"] == "current_debt + noncurrent_debt"
    assert notes["evidence"]["free_cash_flow"]["formula"] == "operating_cash_flow - positive_capex_outflow"
    assert "total_debt: derived from clean simple current and noncurrent debt components" in "|".join(notes["review_notes"])
    assert "free_cash_flow: derived by subtracting positive capex outflow" in "|".join(notes["review_notes"])


def test_unit_conflict_fails_clearly() -> None:
    payload = _companyfacts_payload(
        {
            "Revenues": {"USD": [_fact(1000)], "CAD": [_fact(1000)]},
        }
    )

    with pytest.raises(ValueError, match="unit conflict"):
        _transform(payload)


def test_conflicting_same_tag_period_unit_facts_fail_clearly() -> None:
    payload = _companyfacts_payload(
        {
            "Revenues": {"USD": [_fact(1000), _fact(1001)]},
        }
    )

    with pytest.raises(ValueError, match="conflicting SEC facts"):
        _transform(payload)


def test_no_live_sec_network_call_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("SEC-6A import must not use network access")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    importlib.reload(transform)


def test_no_pipeline_or_downstream_integration_is_exposed() -> None:
    assert not hasattr(transform, "build_fundamental_metrics")
    assert not hasattr(transform, "build_fundamental_layer")
    assert not hasattr(transform, "build_fundamental_analysis")
    assert not hasattr(transform, "decision_engine")
    assert not hasattr(transform, "telegram")
    assert not hasattr(transform, "portfolio")


def test_generated_output_writes_only_to_provided_temp_path(tmp_path: Path) -> None:
    source_path = tmp_path / "CIK0000320193.json"
    output_path = tmp_path / "generated" / "fundamentals_history.csv"
    source_path.write_text(json.dumps(_base_payload()), encoding="utf-8")

    df = transform.transform_companyfacts_file(
        source_path,
        ticker="TEST",
        cik="320193",
        source_freshness_date="2026-05-31",
        extraction_date="2026-05-31",
        output_path=output_path,
    )

    written = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    assert output_path.exists()
    assert output_path.is_relative_to(tmp_path)
    assert written.to_dict(orient="records") == df.to_dict(orient="records")


def test_validate_only_cli_does_not_write_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source_path = tmp_path / "CIK0000320193.json"
    output_path = tmp_path / "generated" / "fundamentals_history.csv"
    source_path.write_text(json.dumps(_base_payload()), encoding="utf-8")

    exit_code = transform.main(
        [
            "--companyfacts-json",
            str(source_path),
            "--ticker",
            "TEST",
            "--cik",
            "320193",
            "--output",
            str(output_path),
            "--source-freshness-date",
            "2026-05-31",
            "--extraction-date",
            "2026-05-31",
            "--validate-only",
        ]
    )

    captured = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert captured["row_count"] == 1
    assert captured["output_path"] == ""
    assert not output_path.exists()
