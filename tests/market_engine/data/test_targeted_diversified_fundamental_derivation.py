from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from market_engine.data.targeted_diversified_fundamental_derivation import (
    COMPARISON_SCHEMA_VERSION,
    DEFAULT_MANIFEST,
    DEFAULT_RANKING,
    DEFAULT_UNIVERSE,
    SEC_COMPANYFACTS_URL,
    SEC_TICKER_INDEX_URL,
    TargetedDerivationError,
    _latest_aligned_observations,
    _freshness_status,
    _ticker_index,
    build_candidate_funnel,
    build_fact_package,
    build_metric_comparison,
    derive_cohort_metrics,
    run_targeted_derivation,
    select_pilot_cohort,
)


def _inputs() -> tuple[dict, dict, dict]:
    return (
        json.loads(DEFAULT_RANKING.read_text(encoding="utf-8")),
        json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8")),
        json.loads(DEFAULT_UNIVERSE.read_text(encoding="utf-8")),
    )


def _funnel(ranking: dict | None = None, manifest: dict | None = None, universe: dict | None = None) -> dict:
    actual_ranking, actual_manifest, actual_universe = _inputs()
    return build_candidate_funnel(
        ranking or actual_ranking,
        manifest or actual_manifest,
        universe or actual_universe,
        ranking_path=DEFAULT_RANKING,
        ranking_manifest_path=DEFAULT_MANIFEST,
        universe_path=DEFAULT_UNIVERSE,
    )


def _companyfacts(ticker: str, *, revenue: float = 100, gross: float | None = 45, operating: float | None = 20) -> dict:
    def concept(value: float) -> dict:
        return {
            "units": {
                "USD": [{
                    "start": "2026-01-01", "end": "2026-03-31", "val": value,
                    "accn": "0000000000-26-000001", "fy": 2026, "fp": "Q1",
                    "form": "10-Q", "filed": "2026-05-01",
                }]
            }
        }

    facts = {"RevenueFromContractWithCustomerExcludingAssessedTax": concept(revenue)}
    if gross is not None:
        facts["GrossProfit"] = concept(gross)
    if operating is not None:
        facts["OperatingIncomeLoss"] = concept(operating)
    return {"cik": "1", "entityName": f"{ticker} Incorporated", "facts": {"us-gaap": facts}}


def test_valid_authoritative_run30_input_yields_checksum_bound_top_25() -> None:
    funnel = _funnel()
    assert funnel["candidate_count"] == 25
    assert funnel["candidates"][0]["ticker"] == "ASB"
    assert funnel["candidates"][-1]["ticker"] == "GATX"
    assert len(funnel["source_bindings"]["ranking_sha256"]) == 64
    assert funnel["source_bindings"]["cutoff_date"] == "2026-07-10"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r, m, u: r.update(schema_version="wrong"), "ranking schema"),
        (lambda r, m, u: r["ranking_policy"].update(ranking_scope="wrong"), "technical screening"),
        (lambda r, m, u: m.update(run_id="other"), "run identities"),
        (lambda r, m, u: m["input"].update(universe_version="other"), "universe versions"),
        (lambda r, m, u: r["candidates"][1].update(rank=0), "ranking order"),
        (lambda r, m, u: r["candidates"][1].update(instrument_id=r["candidates"][0]["instrument_id"]), "duplicate"),
        (lambda r, m, u: r["candidates"][0].update(instrument_id="equity:unknown"), "tracked authority"),
        (lambda r, m, u: r["candidates"][0]["traceability"].update(price_history_path="elsewhere.csv"), "tracked authority"),
    ],
)
def test_funnel_rejects_invalid_machine_authority(mutation, message: str) -> None:
    ranking, manifest, universe = _inputs()
    mutation(ranking, manifest, universe)
    with pytest.raises(TargetedDerivationError, match=message):
        _funnel(ranking, manifest, universe)


def test_changed_ranking_eligibility_is_rejected_by_tracked_authority() -> None:
    ranking, manifest, universe = _inputs()
    ranking["candidates"][0]["ranking_eligible"] = False
    with pytest.raises(TargetedDerivationError, match="tracked authority"):
        _funnel(ranking, manifest, universe)


def test_cohort_is_bounded_rank_first_and_has_explicit_reason_codes() -> None:
    cohort = select_pilot_cohort(_funnel(), cohort_size=10)
    assert cohort["selected_tickers"] == ["ASB", "ASH", "ATR", "AXP", "BIO", "BKH", "BMRN", "BMY", "CHRW", "CI"]
    assert {row["selection_reason"] for row in cohort["candidates"]} == {
        "SELECTED_RANK_PRIORITY", "NOT_SELECTED_COHORT_LIMIT", "BLOCKED_APPLICABILITY_UNPROVEN"
    }
    assert cohort["selection_policy"]["ticker_specific_runtime_branches"] is False


@pytest.mark.parametrize("size", [7, 13])
def test_cohort_limit_fails_closed(size: int) -> None:
    with pytest.raises(TargetedDerivationError, match="cohort_size"):
        run_targeted_derivation(run_id="test", generated_at="2026-08-13T00:00:00Z", cohort_size=size)


def test_no_ifrs_is_forced_when_authoritative_funnel_has_none() -> None:
    funnel = _funnel()
    cohort = select_pilot_cohort(funnel, cohort_size=8)
    assert funnel["accounting_framework_inventory"]["ifrs_candidates"] == 0
    assert cohort["cohort_size"] == 8
    assert "not forced" in cohort["selection_policy"]["framework_diversity"]


def test_sec_identity_index_is_generic_and_normalized() -> None:
    index = _ticker_index({"0": {"ticker": "aaa", "cik_str": 123, "title": "AAA Corp"}})
    assert index == {"AAA": {"cik": "0000000123", "title": "AAA Corp"}}


@pytest.mark.parametrize(
    ("source_date", "acquired_at", "expected"),
    [
        ("2026-07-01", "2026-08-13T00:00:00Z", "current"),
        ("2024-01-01", "2026-08-13T00:00:00Z", "stale"),
        (None, "2026-08-13T00:00:00Z", "missing"),
        ("2027-01-01", "2026-08-13T00:00:00Z", "invalid"),
    ],
)
def test_evidence_freshness_is_explicit(source_date, acquired_at, expected: str) -> None:
    assert _freshness_status(source_date, acquired_at) == expected


def test_generic_fact_extraction_binds_identity_tag_period_unit_scale_and_source_checksum() -> None:
    candidate = {"ticker": "AAA", "instrument_id": "equity:aaa", "rank": 1}
    package, status = build_fact_package(
        candidate,
        payload=_companyfacts("AAA"),
        source_url="https://data.sec.gov/example",
        source_checksum="a" * 64,
        generated_at="2026-08-13T00:00:00Z",
        run_id="me-data11-test",
    )
    assert status["fact_extraction_status"] == "candidate_ready"
    assert package is not None
    assert {row["canonical_concept"] for row in package["facts"]} == {"revenue", "gross_profit", "operating_income"}
    assert all(row["ticker"] == "AAA" and row["instrument_id"] == "equity:aaa" for row in package["facts"])
    assert all(row["period_type"] == "duration" and row["unit"] == "USD" and row["scale"] == 0 for row in package["facts"])
    assert all(row["raw_source_concept"].startswith("us-gaap:") for row in package["facts"])
    assert all(row["source_document_checksum"] == "a" * 64 for row in package["facts"])


def test_period_alignment_never_combines_different_accessions() -> None:
    base = _companyfacts("AAA")
    revenue = base["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"][0]
    operating = copy.deepcopy(revenue)
    operating["accn"] = "different"
    aligned = _latest_aligned_observations({"revenue": [("Revenue", revenue)], "operating_income": [("Op", operating)]})
    assert set(aligned) == {"revenue"}


def test_latest_period_is_not_replaced_by_older_more_complete_period() -> None:
    latest = _companyfacts("AAA")["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"]["USD"][0]
    older_revenue = copy.deepcopy(latest)
    older_revenue.update(end="2025-12-31", start="2025-10-01", filed="2026-02-01", accn="older")
    older_gross = copy.deepcopy(older_revenue)
    aligned = _latest_aligned_observations({
        "revenue": [("Revenue", older_revenue), ("Revenue", latest)],
        "gross_profit": [("GrossProfit", older_gross)],
    })
    assert aligned["revenue"][1]["end"] == "2026-03-31"
    assert "gross_profit" not in aligned


@pytest.mark.parametrize("ticker", ["AAA", "BBB", "CCC"])
def test_same_data10_formula_engine_processes_multiple_issuers(ticker: str) -> None:
    candidate = {"ticker": ticker, "instrument_id": f"equity:{ticker.lower()}", "rank": 1}
    package, _ = build_fact_package(
        candidate, payload=_companyfacts(ticker), source_url="https://data.sec.gov/example",
        source_checksum="b" * 64, generated_at="2026-08-13T00:00:00Z", run_id="me-data11-test",
    )
    assert package is not None
    formula_catalog = json.loads(Path("config/market_engine/data10_fundamental_metric_formula_catalog.json").read_text())
    cohort = {"candidates": [{**candidate, "selected": True}]}
    results = derive_cohort_metrics(cohort, fact_packages={ticker: package}, formula_catalog=formula_catalog, generated_at="2026-08-13T00:00:00Z")
    row = results["instruments"][0]
    assert row["status"] == "pending_approval"
    assert row["approval_state"] == "pending_no_authority"
    assert {item["formula_id"] for item in row["derivations"] if item["status"] == "derived"} == {"gross_margin", "operating_margin"}


@pytest.mark.parametrize(
    ("gross", "operating", "derived_count"),
    [(None, 20, 1), (45, None, 1), (None, None, 0)],
)
def test_missing_numerators_remain_blocked_not_zero(gross, operating, derived_count: int) -> None:
    candidate = {"ticker": "AAA", "instrument_id": "equity:aaa", "rank": 1}
    package, _ = build_fact_package(
        candidate, payload=_companyfacts("AAA", gross=gross, operating=operating),
        source_url="https://data.sec.gov/example", source_checksum="c" * 64,
        generated_at="2026-08-13T00:00:00Z", run_id="me-data11-test",
    )
    assert package is not None
    catalog = json.loads(Path("config/market_engine/data10_fundamental_metric_formula_catalog.json").read_text())
    results = derive_cohort_metrics(
        {"candidates": [{**candidate, "selected": True}]}, fact_packages={"AAA": package},
        formula_catalog=catalog, generated_at="2026-08-13T00:00:00Z",
    )
    rows = results["instruments"][0]["derivations"]
    assert sum(row["status"] == "derived" for row in rows) == derived_count
    assert all(row.get("calculation_result") != 0 for row in rows)


def test_one_ticker_failure_does_not_discard_another_result() -> None:
    funnel = _funnel()
    cohort = select_pilot_cohort(funnel, cohort_size=8)
    first, second = cohort["selected_tickers"][:2]
    candidate = {row["ticker"]: row for row in cohort["candidates"]}
    package, _ = build_fact_package(
        candidate[first], payload=_companyfacts(first), source_url="https://data.sec.gov/example",
        source_checksum="d" * 64, generated_at="2026-08-13T00:00:00Z", run_id="me-data11-test",
    )
    catalog = json.loads(Path("config/market_engine/data10_fundamental_metric_formula_catalog.json").read_text())
    results = derive_cohort_metrics(cohort, fact_packages={first: package}, formula_catalog=catalog, generated_at="2026-08-13T00:00:00Z")
    by_ticker = {row["ticker"]: row for row in results["instruments"]}
    assert by_ticker[first]["status"] == "pending_approval"
    assert by_ticker[second]["status"] == "blocked"


def test_comparison_reconciles_all_25_without_advice_or_float_values() -> None:
    funnel = _funnel()
    cohort = select_pilot_cohort(funnel, cohort_size=8)
    ticker = cohort["selected_tickers"][0]
    inventory = {"instruments": [{
        "ticker": ticker, "accounting_framework": "us_gaap", "company_identity": "Issuer",
        "reporting_period": "2026-Q1", "source_publication_date": "2026-05-01", "source_url": "https://sec.example",
    }]}
    results = {"instruments": [{
        "ticker": ticker, "instrument_id": f"equity:{ticker.lower()}", "status": "pending_approval",
        "approval_state": "pending_no_authority", "reason_codes": ["CHECKSUM_BOUND_DERIVATION_APPROVAL_REQUIRED"],
        "derivations": [{
            "status": "derived", "canonical_metric": "gross_margin", "calculation_result": 0.45,
            "formula_id": "gross_margin", "formula_version": "2.0.0", "reporting_period": "2026-Q1",
            "calculation_checksum": "e" * 64,
        }],
    }]}
    comparison = build_metric_comparison(funnel, cohort, inventory, results)
    assert comparison["schema_version"] == COMPARISON_SCHEMA_VERSION
    assert len(comparison["candidates"]) == 25
    assert [row["rank"] for row in comparison["candidates"]] == list(range(1, 26))
    first = comparison["candidates"][0]
    assert first["metrics"]["gross_margin"]["value"] == "0.450000000000"
    assert first["metrics"]["eps_growth_yoy"]["value"] is None
    serialized = json.dumps(comparison).lower()
    assert all(term not in serialized for term in ("buy now", "investment_score", "price_target", "fair_value"))


def test_full_bounded_run_writes_required_compact_artifacts_and_no_downstream_authority(tmp_path: Path) -> None:
    funnel = _funnel()
    tickers = [row["ticker"] for row in funnel["candidates"] if row["asset_type"] == "equity"][:8]
    index = {str(i): {"ticker": ticker, "cik_str": i + 1, "title": ticker} for i, ticker in enumerate(tickers)}

    def fetch(url: str) -> dict:
        if url == SEC_TICKER_INDEX_URL:
            return index
        cik = url.split("CIK", 1)[1].split(".", 1)[0]
        ticker = tickers[int(cik) - 1]
        return _companyfacts(ticker)

    _, output = run_targeted_derivation(
        run_id="me-data11-test-run", generated_at="2026-08-13T00:00:00Z",
        output_root=tmp_path / "evidence", source_root=tmp_path / "sources", cohort_size=8,
        fetch_json=fetch,
    )
    required = {
        "manifest.json", "candidate_funnel.json", "cohort_selection.json", "source_inventory.json",
        "derivation_summary.json", "fundamental_comparison_matrix.json",
        "downstream_readiness_delta.json", "checksum_index.json", "report.md",
    }
    assert {path.name for path in output.iterdir()} == required | {"approval_candidates"}
    assert sorted(path.name for path in (output / "approval_candidates").iterdir()) == tickers
    derivation = json.loads((output / "derivation_summary.json").read_text())
    downstream = json.loads((output / "downstream_readiness_delta.json").read_text())
    assert derivation["pilot_summary"]["minimum_safe_processing_target_met"] is True
    assert derivation["pilot_summary"]["approved_import_count"] == 0
    assert downstream["downstream_executed"] is False
    assert downstream["after_authoritative"] == downstream["before"]


@pytest.mark.parametrize(
    "forbidden",
    [
        "if ticker ==", "if issuer ==", "US_GAAP_TICKERS", "IFRS_TICKERS", "total liabilities",
        "price_to_earnings", "peg_ratio", "price_to_sales", "free_cashflow_yield", "fair_value",
        "price_target", "investment_score", "position_size", "broker_order", "notification_send",
    ],
)
def test_production_route_contains_no_ticker_or_investment_specific_branch(forbidden: str) -> None:
    source = Path("src/market_engine/data/targeted_diversified_fundamental_derivation.py").read_text(encoding="utf-8")
    assert forbidden not in source
