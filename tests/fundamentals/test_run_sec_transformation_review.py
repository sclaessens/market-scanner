from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.fundamentals import run_sec_transformation_review as review


def _fact(value: int | float, *, fy: int = 2024, fp: str = "FY", end: str = "2024-12-31") -> dict:
    return {
        "val": value,
        "fy": fy,
        "fp": fp,
        "end": end,
        "filed": "2025-02-15",
        "form": "10-K",
        "frame": f"CY{fy}",
        "accn": f"0000000000-{fy}-000001",
    }


def _companyfacts_payload() -> dict:
    return {
        "cik": 320193,
        "entityName": "Synthetic Full",
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_fact(1000)]}},
                "GrossProfit": {"units": {"USD": [_fact(420)]}},
                "OperatingIncomeLoss": {"units": {"USD": [_fact(210)]}},
                "NetIncomeLoss": {"units": {"USD": [_fact(155)]}},
                "StockholdersEquity": {"units": {"USD": [_fact(900)]}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": [_fact(3.14)]}},
                "DebtCurrent": {"units": {"USD": [_fact(100)]}},
                "LongTermDebtNoncurrent": {"units": {"USD": [_fact(900)]}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [_fact(222)]}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [_fact(33)]}},
            }
        },
    }


def _companyfacts_missing_derived_payload() -> dict:
    payload = _companyfacts_payload()
    payload["facts"]["us-gaap"].pop("DebtCurrent")
    payload["facts"]["us-gaap"].pop("LongTermDebtNoncurrent")
    payload["facts"]["us-gaap"].pop("PaymentsToAcquirePropertyPlantAndEquipment")
    return payload


def _companyfacts_messy_payload() -> dict:
    payload = _companyfacts_payload()
    payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"] = [
        _fact(900, fy=2023, end="2023-12-31"),
        _fact(901, fy=2023, end="2023-12-31"),
        _fact(1000),
        _fact(999, fy="", end="2024-12-31"),
    ]
    return payload


def _write_fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    project_tickers = tmp_path / "project_tickers.csv"
    ticker_source = tmp_path / "company_tickers.json"
    companyfacts_dir = tmp_path / "companyfacts"
    companyfacts_dir.mkdir()

    pd.DataFrame({"ticker": ["FULL", "NOMAP", "NOFACTS", "PARTIAL", "MESSY"]}).to_csv(project_tickers, index=False)
    ticker_source.write_text(
        json.dumps(
            {
                "0": {"ticker": "FULL", "cik_str": 320193, "title": "Synthetic Full"},
                "1": {"ticker": "NOFACTS", "cik_str": 789019, "title": "Synthetic Missing Facts"},
                "2": {"ticker": "PARTIAL", "cik_str": 111111, "title": "Synthetic Partial"},
                "3": {"ticker": "MESSY", "cik_str": 222222, "title": "Synthetic Messy Facts"},
            }
        ),
        encoding="utf-8",
    )
    (companyfacts_dir / "CIK0000320193.json").write_text(json.dumps(_companyfacts_payload()), encoding="utf-8")
    (companyfacts_dir / "CIK0000111111.json").write_text(json.dumps(_companyfacts_missing_derived_payload()), encoding="utf-8")
    (companyfacts_dir / "CIK0000222222.json").write_text(json.dumps(_companyfacts_messy_payload()), encoding="utf-8")
    return project_tickers, ticker_source, companyfacts_dir


def _run_review(tmp_path: Path, *, output_path: Path | None = None) -> pd.DataFrame:
    project_tickers, ticker_source, companyfacts_dir = _write_fixture_inputs(tmp_path)
    return review.build_sec_transformation_review(
        project_tickers_path=project_tickers,
        ticker_cik_source_path=ticker_source,
        companyfacts_dir=companyfacts_dir,
        output_path=output_path,
        source_freshness_date="2026-05-31",
        extraction_date="2026-05-31",
    )


def test_controlled_review_runner_works_with_explicit_local_fixture_inputs(tmp_path: Path) -> None:
    review_df = _run_review(tmp_path)

    assert list(review_df["ticker"]) == ["FULL", "NOMAP", "NOFACTS", "PARTIAL", "MESSY", "MESSY"]
    assert set(review.REVIEW_COLUMNS).issubset(review_df.columns)


def test_no_network_or_sec_call_is_performed_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("SEC-6D import must not use network access")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    importlib.reload(review)


def test_output_is_written_only_to_provided_temp_path(tmp_path: Path) -> None:
    output_path = tmp_path / "generated" / "review.csv"

    review_df = _run_review(tmp_path, output_path=output_path)

    written = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    assert output_path.exists()
    assert output_path.is_relative_to(tmp_path)
    assert written.to_dict(orient="records") == review_df.to_dict(orient="records")


def test_one_requested_ticker_with_full_fixture_data_produces_transformed_review_output(tmp_path: Path) -> None:
    review_df = _run_review(tmp_path)
    row = review_df.loc[review_df["ticker"] == "FULL"].iloc[0]

    assert row["transformation_status"] == "TRANSFORMED"
    assert row["mapping_status"] == "CIK_MATCHED"
    assert row["revenue"] == "1000"
    assert row["source_reference"].startswith("sec-companyfacts:CIK0000320193")


def test_missing_cik_mapping_preserves_ticker_row_with_review_status(tmp_path: Path) -> None:
    row = _run_review(tmp_path).loc[lambda df: df["ticker"] == "NOMAP"].iloc[0]

    assert row["mapping_status"] == "CIK_MISSING"
    assert row["transformation_status"] == "CIK_REVIEW_REQUIRED"
    assert row["review_required"] == "true"
    assert row["missing_fields"] == "cik_padded"


def test_missing_companyfacts_file_preserves_ticker_row_with_review_status(tmp_path: Path) -> None:
    row = _run_review(tmp_path).loc[lambda df: df["ticker"] == "NOFACTS"].iloc[0]

    assert row["mapping_status"] == "CIK_MATCHED"
    assert row["transformation_status"] == "COMPANYFACTS_MISSING"
    assert row["review_required"] == "true"
    assert row["missing_fields"] == "companyfacts_json"


def test_direct_fields_are_transformed_when_fixture_data_is_present(tmp_path: Path) -> None:
    row = _run_review(tmp_path).loc[lambda df: df["ticker"] == "FULL"].iloc[0]

    assert row["gross_profit"] == "420"
    assert row["operating_income"] == "210"
    assert row["net_income"] == "155"
    assert row["diluted_eps"] == "3.14"
    assert row["total_equity"] == "900"


def test_derived_fields_are_populated_only_under_approved_fixture_conditions(tmp_path: Path) -> None:
    row = _run_review(tmp_path).loc[lambda df: df["ticker"] == "FULL"].iloc[0]

    assert row["total_debt"] == "1000"
    assert row["free_cash_flow"] == "189"
    assert row["derived_fields_status"] == "DERIVED_FIELDS_PRESENT"


def test_missing_derived_components_remain_blank_with_review_notes(tmp_path: Path) -> None:
    row = _run_review(tmp_path).loc[lambda df: df["ticker"] == "PARTIAL"].iloc[0]
    notes = json.loads(row["notes"])

    assert row["total_debt"] == ""
    assert row["free_cash_flow"] == ""
    assert row["derived_fields_status"] == "DERIVED_FIELDS_MISSING_OR_REVIEW_REQUIRED"
    assert "total_debt: missing source-supported debt components" in "|".join(notes["review_notes"])
    assert "free_cash_flow: missing capital expenditure component" in "|".join(notes["review_notes"])


def test_messy_companyfacts_preserve_clean_periods_without_whole_ticker_failure(tmp_path: Path) -> None:
    messy_rows = _run_review(tmp_path).loc[lambda df: df["ticker"] == "MESSY"]

    assert set(messy_rows["transformation_status"]) == {"TRANSFORMED"}
    clean_row = messy_rows.loc[messy_rows["fiscal_year"] == "2024"].iloc[0]
    old_row = messy_rows.loc[messy_rows["fiscal_year"] == "2023"].iloc[0]
    clean_notes = json.loads(clean_row["notes"])
    old_notes = json.loads(old_row["notes"])

    assert clean_row["revenue"] == "1000"
    assert old_row["revenue"] == ""
    assert "missing fiscal year" in "|".join(clean_notes["review_notes"])
    assert "conflicting SEC facts for revenue" in "|".join(old_notes["review_notes"])


def test_review_only_columns_do_not_contain_allocation_trade_or_action_semantics(tmp_path: Path) -> None:
    review_df = _run_review(tmp_path)
    forbidden = [
        "allocation",
        "ranking",
        "score",
        "tradeability",
        "urgency",
        "conviction",
        "buy",
        "sell",
        "final_action",
        "eligible",
        "eligibility",
    ]
    rendered = " ".join(review.REVIEW_COLUMNS + review_df[review.REVIEW_COLUMNS].astype(str).to_numpy().ravel().tolist()).lower()

    assert all(term not in rendered for term in forbidden)


def test_no_pipeline_or_downstream_integration_is_exposed() -> None:
    assert not hasattr(review, "build_fundamental_metrics")
    assert not hasattr(review, "build_fundamental_layer")
    assert not hasattr(review, "build_fundamental_analysis")
    assert not hasattr(review, "decision_engine")
    assert not hasattr(review, "telegram")
    assert not hasattr(review, "portfolio")


def test_no_generated_operational_data_is_committed_by_default(tmp_path: Path) -> None:
    _run_review(tmp_path)

    assert not (Path("data/processed") / "sec_transformation_review.csv").exists()


def test_validate_only_cli_does_not_write_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project_tickers, ticker_source, companyfacts_dir = _write_fixture_inputs(tmp_path)
    output_path = tmp_path / "generated" / "review.csv"

    exit_code = review.main(
        [
            "--project-tickers",
            str(project_tickers),
            "--ticker-cik-source",
            str(ticker_source),
            "--companyfacts-dir",
            str(companyfacts_dir),
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
    assert captured["row_count"] == 6
    assert captured["output_path"] == ""
    assert not output_path.exists()
