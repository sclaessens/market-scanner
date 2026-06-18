from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from market_engine.run.cached_source_execution import CACHED_SOURCE_SNAPSHOT_INPUT_MODE
from market_engine.run.end_to_end_dry_run_command import (
    run_market_engine_end_to_end_dry_run_command,
)
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    persist_sec_companyfacts_raw_snapshot,
)


TICKER_BUNDLE = (
    {
        "ticker": "NVDA",
        "cik": "0001045810",
        "values": {
            "revenue": 100,
            "net_income": 20,
            "operating_cash_flow": 30,
            "capital_expenditures": 5,
        },
        "position_state": "not_held",
        "quantity": 0,
        "market_value": 0.0,
        "expected_run_state": "dry_run_completed",
    },
    {
        "ticker": "MSFT",
        "cik": "0000789019",
        "values": {
            "revenue": 200,
            "net_income": 50,
            "operating_cash_flow": 75,
            "capital_expenditures": 12,
        },
        "position_state": "held",
        "quantity": 0,
        "market_value": 0,
        "expected_run_state": "dry_run_completed",
    },
    {
        "ticker": "AMD",
        "cik": "0000002488",
        "values": {
            "revenue": 80,
            "net_income": 0,
            "operating_cash_flow": 15,
            "capital_expenditures": 0,
        },
        "position_state": "unknown",
        "quantity": 4,
        "market_value": 320.0,
        "expected_run_state": "dry_run_completed",
    },
)


def test_cached_source_ticker_bundle_executes_ticker_by_ticker(
    tmp_path: Path,
    capsys,
) -> None:
    source_root = tmp_path / "source_snapshots"
    artifact_root = tmp_path / "artifacts"
    payloads: dict[str, dict[str, Any]] = {}

    for fixture in TICKER_BUNDLE:
        snapshot_path = _persist_snapshot(source_root, fixture)
        portfolio_context_path = _portfolio_context_path(tmp_path, fixture)
        dry_run_id = f"run11-{fixture['ticker'].lower()}"

        exit_code = run_market_engine_end_to_end_dry_run_command(
            [
                "--input-mode",
                CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
                "--source-snapshot-json",
                str(snapshot_path),
                "--source-snapshot-root",
                str(source_root),
                "--portfolio-context-json",
                str(portfolio_context_path),
                "--dry-run-id",
                dry_run_id,
                "--generated-at",
                "2026-06-17T16:00:00Z",
                "--artifact-output-root",
                str(artifact_root),
                "--compact",
            ]
        )
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        payloads[str(fixture["ticker"])] = payload

        assert exit_code == 0
        assert captured.err == ""
        assert payload["dry_run_format_version"] == (
            "market-engine-end-to-end-dry-run-v1"
        )
        assert payload["input_mode"] == CACHED_SOURCE_SNAPSHOT_INPUT_MODE
        assert payload["ticker"] == fixture["ticker"]
        assert payload["cik"] == fixture["cik"]
        assert payload["run_state"] == fixture["expected_run_state"]
        assert payload["provenance_summary"]["source_context"][
            "cached_source_reference"
        ]["input_mode"] == CACHED_SOURCE_SNAPSHOT_INPUT_MODE
        assert payload["provenance_summary"]["source_context"][
            "source_refresh_snapshot_id"
        ] == f"{fixture['ticker']}_companyfacts"
        _assert_no_authority_fields(payload)

        assert not (artifact_root / dry_run_id).exists()

    assert payloads["AMD"]["numeric_zero_evidence_summary"][
        "source_context.canonical_fields.capital_expenditures"
    ] == 0
    assert payloads["MSFT"]["numeric_zero_evidence_summary"][
        "portfolio_review.portfolio_context_provenance.current_quantity"
    ] == 0
    assert payloads["MSFT"]["numeric_zero_evidence_summary"][
        "portfolio_review.portfolio_context_provenance.current_market_value"
    ] == 0


def test_cached_source_bundle_artifact_writing_remains_explicit_for_selected_ticker(
    tmp_path: Path,
    capsys,
) -> None:
    source_root = tmp_path / "source_snapshots"
    artifact_root = tmp_path / "artifacts"
    fixture = TICKER_BUNDLE[1]
    snapshot_path = _persist_snapshot(source_root, fixture)
    portfolio_context_path = _portfolio_context_path(tmp_path, fixture)

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
            "--source-snapshot-json",
            str(snapshot_path),
            "--source-snapshot-root",
            str(source_root),
            "--portfolio-context-json",
            str(portfolio_context_path),
            "--dry-run-id",
            "run11-msft-artifact",
            "--generated-at",
            "2026-06-17T16:00:00Z",
            "--write-local-artifact",
            "--artifact-output-root",
            str(artifact_root),
            "--artifact-created-at",
            "2026-06-17T16:01:00Z",
            "--compact",
        ]
    )
    captured = capsys.readouterr()

    manifest_path = artifact_root / "run11-msft-artifact" / "manifest.json"
    artifact_path = (
        artifact_root
        / "run11-msft-artifact"
        / "artifacts"
        / "market_engine_dry_run_run11-msft-artifact_2026-06-17.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out)["ticker"] == "MSFT"
    assert manifest["source_dry_run_id"] == "run11-msft-artifact"
    assert artifact["payload"]["ticker"] == "MSFT"
    assert artifact["payload"]["input_mode"] == CACHED_SOURCE_SNAPSHOT_INPUT_MODE
    assert artifact["payload"]["provenance_summary"]["source_context"][
        "cached_source_reference"
    ]["source_snapshot_reference"].endswith("MSFT_companyfacts.json")


def test_malformed_snapshot_in_bundle_fails_closed_without_live_fallback(
    tmp_path: Path,
    capsys,
) -> None:
    source_root = tmp_path / "source_snapshots"
    malformed_path = source_root / "run11-bad" / "raw" / "ASML_companyfacts.json"
    malformed_path.parent.mkdir(parents=True)
    malformed_path.write_text("{not-json", encoding="utf-8")

    exit_code = run_market_engine_end_to_end_dry_run_command(
        [
            "--input-mode",
            CACHED_SOURCE_SNAPSHOT_INPUT_MODE,
            "--source-snapshot-json",
            str(malformed_path),
            "--source-snapshot-root",
            str(source_root),
            "--dry-run-id",
            "run11-asml-bad",
            "--generated-at",
            "2026-06-17T16:00:00Z",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "invalid JSON" in captured.err
    assert "provider" not in captured.err.lower()


def test_run11_cached_source_modules_do_not_import_side_effect_dependencies() -> None:
    paths = (
        Path("src/market_engine/run/cached_source_execution.py"),
        Path("src/market_engine/run/end_to_end_dry_run_command.py"),
    )
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

    for path in paths:
        module_source = path.read_text(encoding="utf-8")
        assert not any(term in module_source for term in forbidden_terms)


def _persist_snapshot(source_root: Path, fixture: dict[str, Any]) -> Path:
    return persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root,
        run_id="run11-fixture-bundle",
        ticker=str(fixture["ticker"]),
        cik=str(fixture["cik"]),
        raw_payload=_companyfacts_payload(fixture["values"]),
        fetched_at="2026-06-17T16:00:00Z",
    )


def _portfolio_context_path(tmp_path: Path, fixture: dict[str, Any]) -> Path:
    path = tmp_path / f"{fixture['ticker']}_portfolio_context.json"
    path.write_text(
        json.dumps(
            {
                "portfolio_context_format_version": (
                    "market-engine-portfolio-context-v1"
                ),
                "portfolio_context_run_id": f"{fixture['ticker']}-portfolio-context",
                "portfolio_snapshot_timestamp": "2026-06-17T15:55:00Z",
                "portfolio_base_currency": "USD",
                "ticker": fixture["ticker"],
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
        ),
        encoding="utf-8",
    )
    return path


def _companyfacts_payload(values: dict[str, int | None]) -> dict[str, object]:
    us_gaap: dict[str, object] = {
        "Revenues": {"units": {"USD": [_fact(values["revenue"], "2025-12-31")]}},
        "NetIncomeLoss": {
            "units": {"USD": [_fact(values["net_income"], "2025-12-31")]}
        },
        "NetCashProvidedByUsedInOperatingActivities": {
            "units": {"USD": [_fact(values["operating_cash_flow"], "2025-12-31")]}
        },
    }
    if values["capital_expenditures"] is not None:
        us_gaap["PaymentsToAcquirePropertyPlantAndEquipment"] = {
            "units": {
                "USD": [_fact(values["capital_expenditures"], "2025-12-31")]
            }
        }
    return {"facts": {"us-gaap": us_gaap}}


def _fact(value: int | None, end: str) -> dict[str, object]:
    return {
        "val": value,
        "fy": int(end[:4]),
        "fp": "FY",
        "form": "10-K",
        "filed": f"{int(end[:4]) + 1}-02-15",
        "start": f"{end[:4]}-01-01",
        "end": end,
        "accn": f"0000000000-{end[:4]}-000001",
        "frame": f"CY{end[:4]}",
    }


def _assert_no_authority_fields(value: Any) -> None:
    forbidden_field_names = {
        "buy_instruction",
        "sell_instruction",
        "hold_instruction",
        "allocation_advice",
        "target_weight",
        "target_weights",
        "target_price",
        "position_size",
        "position_sizing",
        "order_generation",
        "execution_instruction",
        "broker_ready_payload",
        "trade_ticket",
        "urgency",
        "conviction",
        "ranking",
        "best_pick",
        "watchlist_mutation",
        "portfolio_mutation",
    }
    if isinstance(value, dict):
        assert not (set(value) & forbidden_field_names)
        for nested_value in value.values():
            _assert_no_authority_fields(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _assert_no_authority_fields(nested_value)
