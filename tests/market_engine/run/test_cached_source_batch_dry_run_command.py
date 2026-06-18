from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from market_engine.run import cached_source_batch_dry_run_command as command


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
    assert captured["requested_tickers"] == ("NVDA", "MSFT")
    assert captured["discover_cached_tickers"] is False
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

    assert command._requested_tickers_from_args(args) == ("NVDA", "MSFT", "AMD")


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
