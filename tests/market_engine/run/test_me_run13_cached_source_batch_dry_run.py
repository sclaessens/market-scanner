from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from market_engine.run.cached_source_batch_execution import (
    CACHED_SOURCE_BATCH_INPUT_MODE,
    MARKET_ENGINE_CACHED_SOURCE_BATCH_DRY_RUN_FORMAT_VERSION,
    CachedSourceBatchDryRunError,
    build_cached_source_batch_dry_run,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    SEC_COMPANYFACTS_SNAPSHOT_FORMAT_VERSION,
    SEC_COMPANYFACTS_SOURCE_NAME,
    persist_sec_companyfacts_raw_snapshot,
)


FIXTURES = {
    "NVDA": {
        "cik": "0001045810",
        "values": {
            "revenue": 100,
            "net_income": 20,
            "operating_cash_flow": 30,
            "capital_expenditures": 5,
        },
        "quantity": 0,
        "market_value": 0.0,
        "position_state": "not_held",
    },
    "MSFT": {
        "cik": "0000789019",
        "values": {
            "revenue": 200,
            "net_income": 50,
            "operating_cash_flow": 75,
            "capital_expenditures": 12,
        },
        "quantity": 2,
        "market_value": 800.0,
        "position_state": "held",
    },
    "AMD": {
        "cik": "0000002488",
        "values": {
            "revenue": 80,
            "net_income": 0,
            "operating_cash_flow": 15,
            "capital_expenditures": 0,
        },
        "quantity": 4,
        "market_value": 320.0,
        "position_state": "held",
    },
}


def test_cached_source_batch_runs_multiple_tickers_with_deterministic_order(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    _persist_fixture(source_root, "NVDA")
    _persist_fixture(source_root, "MSFT")
    _persist_fixture(source_root, "AMD")

    payload = build_cached_source_batch_dry_run(
        source_snapshot_root=source_root,
        batch_id="run13-batch",
        generated_at="2026-06-18T09:00:00Z",
        requested_tickers=("MSFT", "AMD", "NVDA"),
        portfolio_contexts_by_ticker=_portfolio_contexts("MSFT", "AMD", "NVDA"),
    )

    assert payload["contract_version"] == (
        MARKET_ENGINE_CACHED_SOURCE_BATCH_DRY_RUN_FORMAT_VERSION
    )
    assert payload["input_mode"] == CACHED_SOURCE_BATCH_INPUT_MODE
    assert payload["source_mode"] == "cached_source_local_only"
    assert payload["requested_tickers"] == ("MSFT", "AMD", "NVDA")
    assert [result["ticker"] for result in payload["per_ticker_results"]] == [
        "MSFT",
        "AMD",
        "NVDA",
    ]
    assert payload["batch_execution_state"] == "completed"
    assert payload["batch_counts"]["requested_count"] == 3
    assert payload["batch_counts"]["discovered_cached_source_count"] == 3
    assert payload["batch_counts"]["executed_count"] == 3
    assert payload["batch_counts"]["completed_count"] == 3
    assert payload["batch_counts"]["blocked_count"] == 0
    assert payload["live_provider_call_made"] is False
    assert all(
        result["end_to_end_dry_run_reference"]["dry_run_format_version"]
        == "market-engine-end-to-end-dry-run-v1"
        for result in payload["per_ticker_results"]
    )
    assert all(
        result["source_snapshot_format_version"]
        == SEC_COMPANYFACTS_SNAPSHOT_FORMAT_VERSION
        for result in payload["per_ticker_results"]
    )
    assert all(result["artifact_reference"] is None for result in payload["per_ticker_results"])
    assert not (tmp_path / "artifacts").exists()
    assert _result(payload, "AMD")["numeric_zero_evidence_present"] is True


def test_cached_source_batch_discovery_orders_tickers_deterministically(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    _persist_fixture(source_root, "NVDA")
    _persist_fixture(source_root, "AMD")
    _persist_fixture(source_root, "MSFT")

    payload = build_cached_source_batch_dry_run(
        source_snapshot_root=source_root,
        batch_id="run13-discovery",
        generated_at="2026-06-18T09:00:00Z",
        discover_cached_tickers=True,
        portfolio_contexts_by_ticker=_portfolio_contexts("NVDA", "AMD", "MSFT"),
    )

    assert payload["requested_tickers"] == ("AMD", "MSFT", "NVDA")
    assert [result["ticker"] for result in payload["per_ticker_results"]] == [
        "AMD",
        "MSFT",
        "NVDA",
    ]


def test_cached_source_batch_isolates_missing_invalid_and_ambiguous_tickers(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    _persist_fixture(source_root, "NVDA")
    _persist_fixture(source_root, "MSFT", run_id="run13-a")
    _persist_fixture(source_root, "MSFT", run_id="run13-b")
    _write_invalid_snapshot(source_root, "ASML")

    payload = build_cached_source_batch_dry_run(
        source_snapshot_root=source_root,
        batch_id="run13-failures",
        generated_at="2026-06-18T09:00:00Z",
        requested_tickers=("NVDA", "MSFT", "ASML", "COST"),
        portfolio_contexts_by_ticker=_portfolio_contexts("NVDA", "MSFT"),
    )

    assert payload["batch_execution_state"] == "completed_with_ticker_failures"
    assert payload["batch_counts"]["completed_count"] == 1
    assert payload["batch_counts"]["blocked_count"] == 3
    assert payload["batch_counts"]["missing_cached_source_count"] == 1
    assert payload["batch_counts"]["ambiguous_cached_source_count"] == 1
    assert _result(payload, "NVDA")["execution_state"] == "completed"
    assert _result(payload, "MSFT")["execution_state"] == (
        "blocked_ambiguous_cached_source"
    )
    assert _result(payload, "ASML")["execution_state"] == "blocked_invalid_cached_source"
    assert _result(payload, "COST")["execution_state"] == "blocked_missing_cached_source"
    assert _result(payload, "ASML")["end_to_end_dry_run_reference"] is None


def test_cached_source_batch_rejects_missing_root_and_duplicate_tickers(
    tmp_path: Path,
) -> None:
    with pytest.raises(CachedSourceBatchDryRunError, match="root does not exist"):
        build_cached_source_batch_dry_run(
            source_snapshot_root=tmp_path / "missing",
            batch_id="run13-missing-root",
            generated_at="2026-06-18T09:00:00Z",
            requested_tickers=("NVDA",),
        )

    source_root = tmp_path / "source_snapshots"
    source_root.mkdir()
    with pytest.raises(CachedSourceBatchDryRunError, match="unique"):
        build_cached_source_batch_dry_run(
            source_snapshot_root=source_root,
            batch_id="run13-duplicate",
            generated_at="2026-06-18T09:00:00Z",
            requested_tickers=("NVDA", "NVDA"),
        )


def test_cached_source_batch_artifacts_are_opt_in_and_refuse_overwrite(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    artifact_root = tmp_path / "artifacts"
    _persist_fixture(source_root, "NVDA")
    _persist_fixture(source_root, "AMD")

    no_artifact_payload = build_cached_source_batch_dry_run(
        source_snapshot_root=source_root,
        batch_id="run13-no-artifact",
        generated_at="2026-06-18T09:00:00Z",
        requested_tickers=("NVDA", "AMD"),
        portfolio_contexts_by_ticker=_portfolio_contexts("NVDA", "AMD"),
        artifact_output_root=artifact_root,
    )

    assert no_artifact_payload["artifact_manifest_reference"] is None
    assert not artifact_root.exists()

    payload = build_cached_source_batch_dry_run(
        source_snapshot_root=source_root,
        batch_id="run13-artifact",
        generated_at="2026-06-18T09:00:00Z",
        requested_tickers=("NVDA", "AMD"),
        portfolio_contexts_by_ticker=_portfolio_contexts("NVDA", "AMD"),
        write_local_artifacts=True,
        artifact_output_root=artifact_root,
        artifact_created_at="2026-06-18T09:01:00Z",
    )

    batch_manifest = artifact_root / "run13-artifact" / "batch_manifest.json"
    nvda_artifact = artifact_root / "run13-artifact" / "NVDA" / "dry_run.json"
    amd_manifest = artifact_root / "run13-artifact" / "AMD" / "manifest.json"

    assert payload["artifact_manifest_reference"] == (
        "run13-artifact/batch_manifest.json"
    )
    assert batch_manifest.exists()
    assert nvda_artifact.exists()
    assert amd_manifest.exists()
    assert json.loads(nvda_artifact.read_text(encoding="utf-8"))[
        "artifact_format_version"
    ] == "market-engine-local-dry-run-artifact-v1"

    with pytest.raises(CachedSourceBatchDryRunError, match="already exists"):
        build_cached_source_batch_dry_run(
            source_snapshot_root=source_root,
            batch_id="run13-artifact",
            generated_at="2026-06-18T09:00:00Z",
            requested_tickers=("NVDA",),
            portfolio_contexts_by_ticker=_portfolio_contexts("NVDA"),
            write_local_artifacts=True,
            artifact_output_root=artifact_root,
            artifact_created_at="2026-06-18T09:01:00Z",
        )


def test_cached_source_batch_runtime_has_no_side_effect_dependencies() -> None:
    module_source = Path(
        "src/market_engine/run/cached_source_batch_execution.py"
    ).read_text(encoding="utf-8")
    forbidden_terms = (
        "from " "scripts",
        "import " "scripts",
        "from " "market_" "scanner",
        "import " "market_" "scanner",
        "tele" "gram",
        "smtp" "lib",
        "y" "finance",
        "url" "lib",
        "requ" "ests",
        "sock" "et",
        "sub" "process",
    )

    assert not any(term in module_source for term in forbidden_terms)


def _persist_fixture(
    source_root: Path,
    ticker: str,
    *,
    run_id: str = "run13-fixtures",
) -> Path:
    fixture = FIXTURES[ticker]
    return persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root,
        run_id=run_id,
        ticker=ticker,
        cik=str(fixture["cik"]),
        raw_payload=_companyfacts_payload(fixture["values"]),
        fetched_at="2026-06-18T08:00:00Z",
    )


def _write_invalid_snapshot(source_root: Path, ticker: str) -> Path:
    path = source_root / "run13-invalid" / "raw" / f"{ticker}_companyfacts.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    return path


def _portfolio_contexts(*tickers: str) -> dict[str, dict[str, Any]]:
    return {ticker: _portfolio_context(ticker) for ticker in tickers}


def _portfolio_context(ticker: str) -> dict[str, Any]:
    fixture = FIXTURES[ticker]
    return {
        "portfolio_context_format_version": "market-engine-portfolio-context-v1",
        "portfolio_context_run_id": f"{ticker}-portfolio-context",
        "portfolio_snapshot_timestamp": "2026-06-18T08:55:00Z",
        "portfolio_base_currency": "USD",
        "ticker": ticker,
        "position_state": fixture["position_state"],
        "current_quantity": fixture["quantity"],
        "current_market_value": fixture["market_value"],
        "portfolio_total_value": 100000.0,
        "current_ticker_exposure_pct": 0,
        "exposure_buckets": {"technology": 0},
        "concentration_thresholds": {"single_ticker_review_pct": 10},
        "policy_constraints": {},
        "missing_portfolio_context_fields": [],
        "stale_portfolio_context_fields": [],
        "context_provenance": {
            "portfolio_context_source": "local_non_production_fixture"
        },
    }


def _companyfacts_payload(values: Mapping[str, int]) -> dict[str, object]:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_fact(values["revenue"])]}},
                "NetIncomeLoss": {"units": {"USD": [_fact(values["net_income"])]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_fact(values["operating_cash_flow"])]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_fact(values["capital_expenditures"])]}
                },
            }
        }
    }


def _fact(value: int) -> dict[str, object]:
    return {
        "val": value,
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2026-02-15",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "accn": "0000000000-2025-000001",
        "frame": "CY2025",
    }


def _result(payload: Mapping[str, Any], ticker: str) -> Mapping[str, Any]:
    for result in payload["per_ticker_results"]:
        if result["ticker"] == ticker:
            return result
    raise AssertionError(f"missing ticker result: {ticker}")
