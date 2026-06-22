from __future__ import annotations

import argparse
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from market_engine.run import cached_source_batch_dry_run_command as command
from market_engine.source_refresh.sec_companyfacts_snapshots import (
    persist_sec_companyfacts_raw_snapshot,
)


RUN17_SELECTED_TICKERS = (
    "NVDA",
    "AMD",
    "ASML",
    "META",
    "MSFT",
    "VRT",
    "CLS",
    "CRDO",
    "IREN",
    "COST",
    "HO",
    "AVGO",
    "TSM",
)
RUN17_AVAILABLE_SNAPSHOT_TICKERS = tuple(
    ticker for ticker in RUN17_SELECTED_TICKERS if ticker != "HO"
)


def test_command_result_uses_explicit_tickers_and_visibility_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_build_cached_source_batch_dry_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _batch_payload(
            requested_tickers=("NVDA", "MSFT"),
            results=(
                _ticker_result("NVDA", "completed"),
                _ticker_result("MSFT", "completed"),
            ),
        )

    monkeypatch.setattr(
        command,
        "build_cached_source_batch_dry_run",
        fake_build_cached_source_batch_dry_run,
    )
    args = _parse(
        "--source-snapshot-root",
        "data/market_engine/source_snapshots",
        "--tickers",
        "nvda, msft",
        "--batch-id",
        "run15-test",
        "--generated-at",
        "2026-06-18T10:00:00Z",
    )

    result = command.build_command_result(args)

    assert result["visibility_contract_version"] == (
        command.MARKET_ENGINE_REAL_CACHED_SOURCE_BATCH_DRY_RUN_VISIBILITY_FORMAT_VERSION
    )
    assert result["run_context"]["batch_id"] == "run15-test"
    assert result["run_context"]["artifact_writing_enabled"] is False
    assert result["run_context"]["portfolio_context"] == {"enabled": False}
    assert captured["requested_tickers"] == ("NVDA", "MSFT")
    assert captured["discover_cached_tickers"] is False
    assert captured["portfolio_contexts_by_ticker"] == {}
    assert captured["write_local_artifacts"] is False
    assert captured["artifact_created_at"] is None


def test_command_result_uses_discovery_mode_and_artifact_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_build_cached_source_batch_dry_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _batch_payload(
            requested_tickers=("AMD",),
            results=(
                _ticker_result("AMD", "completed", artifact_reference="run15/AMD/dry_run.json"),
            ),
            artifact_manifest_reference="run15/batch_manifest.json",
        )

    monkeypatch.setattr(
        command,
        "build_cached_source_batch_dry_run",
        fake_build_cached_source_batch_dry_run,
    )
    args = _parse(
        "--discover-cached-tickers",
        "--ticker-limit",
        "1",
        "--write-local-artifacts",
        "--artifact-output-root",
        "artifacts/market_engine",
        "--generated-at",
        "2026-06-18T10:00:00Z",
    )

    result = command.build_command_result(args)

    assert captured["requested_tickers"] is None
    assert captured["discover_cached_tickers"] is True
    assert captured["ticker_limit"] == 1
    assert captured["write_local_artifacts"] is True
    assert captured["artifact_created_at"] == "2026-06-18T10:00:00Z"
    assert result["run_context"]["artifact_writing_enabled"] is True


def test_render_human_visible_output_contains_required_sections() -> None:
    result = {
        "visibility_contract_version": command.MARKET_ENGINE_REAL_CACHED_SOURCE_BATCH_DRY_RUN_VISIBILITY_FORMAT_VERSION,
        "command": "market-engine-cached-source-batch-dry-run --tickers NVDA,MSFT",
        "run_context": {
            "batch_id": "run15-test",
            "generated_at": "2026-06-18T10:00:00Z",
            "source_snapshot_root": "data/market_engine/source_snapshots",
            "artifact_writing_enabled": False,
            "artifact_output_root": "artifacts/market_engine",
            "ticker_limit": None,
            "operator_ticker_input_reference": "explicit_requested_tickers",
            "overwrite_protection": "enabled",
        },
        "batch_payload": _batch_payload(
            requested_tickers=("NVDA", "MSFT"),
            results=(
                _ticker_result("NVDA", "completed"),
                _ticker_result(
                    "MSFT",
                    "blocked_missing_cached_source",
                    blocked_reasons=("No matching cached source snapshot was found.",),
                ),
            ),
        ),
        "next_review_actions": ("Capture this terminal output.",),
    }
    stdout = StringIO()

    command.render_human_visible_output(result, stdout=stdout)

    output = stdout.getvalue()
    for section in (
        "RUN CONTEXT",
        "INPUT DISCOVERY",
        "PORTFOLIO CONTEXT",
        "SELECTED TICKERS",
        "EXECUTION PROGRESS",
        "BATCH SUMMARY",
        "BLOCKED / FAILED TICKERS",
        "ARTIFACTS",
        "FORBIDDEN SIDE-EFFECT CONFIRMATION",
        "NEXT REVIEW ACTIONS",
    ):
        assert section in output
    assert "MSFT | blocked_missing_cached_source" in output
    assert "Generated artifacts are not committed by default." in output


def test_ticker_file_parses_comments_commas_and_uppercases(tmp_path: Path) -> None:
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("# comment\nnvda, msft\n\namd\n", encoding="utf-8")
    args = _parse("--ticker-file", str(ticker_file))

    tickers, metadata = command._requested_tickers_from_args(args)

    assert tickers == ("NVDA", "MSFT", "AMD")
    assert metadata is None


def test_canonical_ticker_universe_selects_active_cached_source_only_rows(
    tmp_path: Path,
) -> None:
    canonical_csv = tmp_path / "ticker_universe.csv"
    canonical_csv.write_text(
        "\n".join(
            [
                "ticker,name,market,asset_type,active,priority,source_policy,"
                "portfolio_relevant,telegram_preview_eligible,"
                "telegram_delivery_eligible,notes",
                "NVDA,NVIDIA,USA,equity,true,2,cached_source_only,true,true,false,",
                "SMCI,Super Micro Computer,USA,equity,true,1,manual_review_only,true,true,false,",
                "AMD,AMD,USA,equity,true,1,cached_source_only,true,true,false,",
                "COST,Costco,USA,equity,false,3,cached_source_only,true,true,false,",
                "XYZ,Blocked,USA,equity,true,4,blocked,true,true,false,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = _parse("--canonical-ticker-universe", str(canonical_csv))

    tickers, metadata = command._requested_tickers_from_args(args)

    assert tickers == ("AMD", "NVDA")
    assert metadata is not None
    assert metadata["selection_policy"] == (
        "active_true_and_source_policy_cached_source_only"
    )
    assert metadata["loaded_row_count"] == 5
    assert metadata["selected_tickers"] == ("AMD", "NVDA")
    assert metadata["excluded_manual_review_only_tickers"] == ("SMCI",)
    assert metadata["excluded_inactive_tickers"] == ("COST",)
    assert metadata["excluded_blocked_tickers"] == ("XYZ",)


def test_command_result_passes_canonical_ticker_universe_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_csv = tmp_path / "ticker_universe.csv"
    canonical_csv.write_text(
        "\n".join(
            [
                "ticker,name,market,asset_type,active,priority,source_policy,"
                "portfolio_relevant,telegram_preview_eligible,"
                "telegram_delivery_eligible,notes",
                "SMCI,Super Micro Computer,USA,equity,true,1,manual_review_only,true,true,false,",
                "NVDA,NVIDIA,USA,equity,true,2,cached_source_only,true,true,false,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_build_cached_source_batch_dry_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _batch_payload(
            requested_tickers=("NVDA",),
            results=(_ticker_result("NVDA", "completed"),),
        )

    monkeypatch.setattr(
        command,
        "build_cached_source_batch_dry_run",
        fake_build_cached_source_batch_dry_run,
    )
    args = _parse(
        "--canonical-ticker-universe",
        str(canonical_csv),
        "--batch-id",
        "run16-test",
        "--generated-at",
        "2026-06-19T10:00:00Z",
    )

    result = command.build_command_result(args)

    assert captured["requested_tickers"] == ("NVDA",)
    assert captured["discover_cached_tickers"] is False
    canonical = result["run_context"]["canonical_ticker_universe"]
    assert canonical["selected_tickers"] == ("NVDA",)
    assert canonical["excluded_manual_review_only_tickers"] == ("SMCI",)


def test_command_result_passes_local_portfolio_contexts_by_ticker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    portfolio_context_path = _write_run18_portfolio_context(tmp_path)
    captured: dict[str, Any] = {}

    def fake_build_cached_source_batch_dry_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _batch_payload(
            requested_tickers=("NVDA", "AMD"),
            results=(
                _ticker_result("NVDA", "completed"),
                _ticker_result("AMD", "completed"),
            ),
        )

    monkeypatch.setattr(
        command,
        "build_cached_source_batch_dry_run",
        fake_build_cached_source_batch_dry_run,
    )
    args = _parse(
        "--tickers",
        "nvda,amd",
        "--portfolio-context",
        str(portfolio_context_path),
        "--batch-id",
        "run18-test",
        "--generated-at",
        "2026-06-22T12:00:00Z",
    )

    result = command.build_command_result(args)

    contexts = captured["portfolio_contexts_by_ticker"]
    assert tuple(sorted(contexts)) == ("AMD", "NVDA")
    assert contexts["NVDA"]["portfolio_context_format_version"] == (
        "market-engine-portfolio-context-v1"
    )
    assert contexts["NVDA"]["portfolio_context_run_id"] == (
        "run18-test-nvda-portfolio-context"
    )
    assert contexts["NVDA"]["position_state"] == "not_held"
    assert contexts["NVDA"]["current_quantity"] == 0
    assert contexts["AMD"]["current_quantity"] == 4
    assert contexts["AMD"]["position_state"] == "held"
    metadata = result["run_context"]["portfolio_context"]
    assert metadata["enabled"] is True
    assert metadata["portfolio_write_authority"] is False
    assert metadata["context_ticker_count"] == 2
    assert "--portfolio-context" in result["command"]
    assert any("portfolio context" in action for action in result["next_review_actions"])


def test_canonical_universe_batch_discovers_me_sr02_style_snapshots(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source_snapshots"
    canonical_csv = _write_run17_canonical_universe(tmp_path)
    for ticker in RUN17_AVAILABLE_SNAPSHOT_TICKERS:
        _persist_run17_snapshot(source_root, ticker)
    artifact_root = tmp_path / "artifacts"
    args = _parse(
        "--source-snapshot-root",
        str(source_root),
        "--canonical-ticker-universe",
        str(canonical_csv),
        "--batch-id",
        "run17-test",
        "--generated-at",
        "2026-06-22T09:00:00Z",
        "--write-local-artifacts",
        "--artifact-output-root",
        str(artifact_root),
    )

    result = command.build_command_result(args)

    batch = result["batch_payload"]
    canonical = result["run_context"]["canonical_ticker_universe"]
    assert canonical["selected_tickers"] == RUN17_SELECTED_TICKERS
    assert canonical["excluded_manual_review_only_tickers"] == ("SMCI",)
    assert batch["requested_tickers"] == RUN17_SELECTED_TICKERS
    assert batch["ticker_universe_metadata"]["discovery_policy"] == (
        "scan_local_sec_companyfacts_raw_snapshot_layouts"
    )
    assert batch["batch_counts"]["discovered_cached_source_count"] == 12
    assert batch["batch_counts"]["executed_count"] == 12
    assert batch["batch_counts"]["blocked_count"] == 13
    assert batch["batch_counts"]["missing_cached_source_count"] == 1
    assert _batch_ticker_result(batch, "HO")["execution_state"] == (
        "blocked_missing_cached_source"
    )
    assert _batch_ticker_result(batch, "NVDA")["execution_state"] == (
        "blocked_downstream_contract_failure"
    )
    assert _batch_ticker_result(batch, "AMD")["numeric_zero_evidence_present"] is True
    assert all(
        _batch_ticker_result(batch, ticker)["end_to_end_dry_run_reference"] is not None
        for ticker in RUN17_AVAILABLE_SNAPSHOT_TICKERS
    )
    assert all(
        "sec_companyfacts/me-sr02-test/raw" in str(
            _batch_ticker_result(batch, ticker)["source_snapshot_reference"]
        )
        for ticker in RUN17_AVAILABLE_SNAPSHOT_TICKERS
    )
    assert batch["live_provider_call_made"] is False
    assert "No provider" in batch["forbidden_side_effect_confirmation"]
    assert (
        artifact_root / "run17-test" / "batch_manifest.json"
    ).exists()
    assert (
        artifact_root / "run17-test" / "NVDA" / "dry_run.json"
    ).exists()
    assert _batch_ticker_result(batch, "HO")["artifact_reference"] is None


def test_run_command_returns_error_for_invalid_ticker_file() -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = command.run_command(
        ["--ticker-file", "does-not-exist.txt"],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 2
    assert "Unable to read ticker file" in stderr.getvalue()
    assert stdout.getvalue() == ""


def _parse(*argv: str) -> argparse.Namespace:
    return command._argument_parser().parse_args(list(argv))


def _write_run18_portfolio_context(tmp_path: Path) -> Path:
    path = tmp_path / "portfolio_context.json"
    path.write_text(
        json.dumps(
            {
                "portfolio_context_batch_format_version": "market-engine-local-portfolio-context-batch-v1",
                "non_production_local_context": True,
                "portfolio_write_authority": False,
                "portfolio_snapshot_timestamp": "2026-06-22T11:00:00Z",
                "portfolio_base_currency": "EUR",
                "portfolio_total_value": 32459,
                "default_position_state": "not_held",
                "positions_by_ticker": {
                    "AMD": {
                        "position_state": "held",
                        "current_quantity": 4,
                        "current_market_value": 520,
                        "current_ticker_exposure_pct": 1.6,
                    }
                },
                "exposure_buckets": {"single_position_max_pct": 10},
                "concentration_thresholds": {"review_above_pct": 5},
                "policy_constraints": {"non_actionable_review_only": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_run17_canonical_universe(tmp_path: Path) -> Path:
    path = tmp_path / "ticker_universe.csv"
    rows = [
        "ticker,name,market,asset_type,active,priority,source_policy,"
        "portfolio_relevant,telegram_preview_eligible,telegram_delivery_eligible,notes",
    ]
    for priority, ticker in enumerate(RUN17_SELECTED_TICKERS, start=1):
        rows.append(
            f"{ticker},{ticker} Inc,USA,equity,true,{priority},cached_source_only,"
            "true,true,false,"
        )
    rows.append(
        "SMCI,Super Micro Computer,USA,equity,true,14,manual_review_only,"
        "true,true,false,manual review only"
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _persist_run17_snapshot(source_root: Path, ticker: str) -> Path:
    return persist_sec_companyfacts_raw_snapshot(
        root_dir=source_root / "sec_companyfacts",
        run_id="me-sr02-test",
        ticker=ticker,
        cik=_run17_cik(ticker),
        raw_payload=_companyfacts_payload(ticker),
        fetched_at="2026-06-19T00:00:00Z",
    )


def _run17_cik(ticker: str) -> str:
    cik_by_ticker = {
        ticker: str(index).zfill(10)
        for index, ticker in enumerate(RUN17_AVAILABLE_SNAPSHOT_TICKERS, start=1)
    }
    return cik_by_ticker[ticker]


def _companyfacts_payload(ticker: str) -> dict[str, object]:
    zero = ticker == "AMD"
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_fact(100 if not zero else 0)]}},
                "NetIncomeLoss": {"units": {"USD": [_fact(20 if not zero else 0)]}},
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": [_fact(30 if not zero else 0)]}
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": [_fact(5 if not zero else 0)]}
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


def _batch_ticker_result(batch: dict[str, Any], ticker: str) -> dict[str, Any]:
    for result in batch["per_ticker_results"]:
        if result["ticker"] == ticker:
            return result
    raise AssertionError(f"missing ticker result: {ticker}")


def _batch_payload(
    *,
    requested_tickers: tuple[str, ...],
    results: tuple[dict[str, Any], ...],
    artifact_manifest_reference: str | None = None,
) -> dict[str, Any]:
    return {
        "contract_version": "market-engine-cached-source-batch-dry-run-v1",
        "batch_id": "run15-test",
        "generated_at": "2026-06-18T10:00:00Z",
        "input_mode": "cached_source_batch",
        "source_mode": "cached_source_local_only",
        "source_snapshot_root": "data/market_engine/source_snapshots",
        "operator_ticker_input_reference": "explicit_requested_tickers",
        "requested_tickers": requested_tickers,
        "ticker_universe_metadata": {
            "requested_count": len(requested_tickers),
            "discovered_cached_source_tickers": list(requested_tickers),
        },
        "batch_execution_state": "completed_with_ticker_failures"
        if any(result["execution_state"].startswith("blocked") for result in results)
        else "completed",
        "batch_counts": {
            "requested_count": len(requested_tickers),
            "discovered_cached_source_count": len(requested_tickers),
            "eligible_count": sum(
                1 for result in results if result["execution_state"] == "completed"
            ),
            "executed_count": sum(
                1 for result in results if result.get("end_to_end_dry_run_reference")
            ),
            "completed_count": sum(
                1 for result in results if result["execution_state"] == "completed"
            ),
            "completed_with_limitations_count": 0,
            "blocked_count": sum(
                1 for result in results if result["execution_state"].startswith("blocked")
            ),
            "failed_count": 0,
            "skipped_count": 0,
        },
        "per_ticker_results": results,
        "artifact_manifest_reference": artifact_manifest_reference,
        "forbidden_side_effect_confirmation": "No disallowed side effects are performed.",
        "authority_boundary_confirmation": "The command reports local dry-run state only.",
        "live_provider_call_made": False,
    }


def _ticker_result(
    ticker: str,
    execution_state: str,
    *,
    blocked_reasons: tuple[str, ...] = (),
    artifact_reference: str | None = None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "source_snapshot_reference": f"run15/raw/{ticker}_companyfacts.json"
        if execution_state == "completed"
        else None,
        "execution_state": execution_state,
        "blocked_reasons": blocked_reasons,
        "artifact_reference": artifact_reference,
        "end_to_end_dry_run_reference": {"dry_run_id": f"run15-{ticker.lower()}"}
        if execution_state == "completed"
        else None,
    }
