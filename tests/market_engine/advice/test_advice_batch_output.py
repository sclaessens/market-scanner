from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path
from typing import Any

from market_engine.advice.advice_batch_command import run_command
from market_engine.advice.deterministic_advice import ADVICE_LABELS


def test_batch_command_writes_required_output_files(tmp_path: Path) -> None:
    index_path = _write_status_index(tmp_path, [_watchlist_row(tmp_path, "BBB"), _watchlist_row(tmp_path, "AAA")])

    exit_code = run_command(
        [
            "--ticker-status-index",
            index_path.as_posix(),
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--run-id",
            "batch-run",
            "--generated-at",
            "2026-07-11T00:00:00Z",
            "--target-size",
            "500",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 0
    output_dir = tmp_path / "out" / "batch-run"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "advice_index.json",
        "advice_index.md",
        "advice_summary.json",
        "avoid_for_now.md",
        "buy_candidates.md",
        "coverage_report.md",
        "manifest.json",
        "missing_data_report.md",
        "unable_to_advise.md",
        "wait_for_price.md",
        "watchlist.md",
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "market-engine-advice-batch-manifest-v1"
    assert manifest["baseline_guardrail"]["openai_api_required"] is False


def test_label_markdown_files_are_generated_with_empty_buy_message(tmp_path: Path) -> None:
    index_path = _write_status_index(tmp_path, [_watchlist_row(tmp_path, "AAA")])
    output_dir = _run_batch(tmp_path, index_path)

    assert "No buy candidates in this run." in (output_dir / "buy_candidates.md").read_text()
    assert "# Wait For Price" in (output_dir / "wait_for_price.md").read_text()
    assert "# Avoid For Now" in (output_dir / "avoid_for_now.md").read_text()
    assert "# Unable To Advise" in (output_dir / "unable_to_advise.md").read_text()


def test_watchlist_report_contains_watchlist_tickers_sorted(tmp_path: Path) -> None:
    index_path = _write_status_index(
        tmp_path,
        [_watchlist_row(tmp_path, "BBB"), _watchlist_row(tmp_path, "AAA")],
    )
    output_dir = _run_batch(tmp_path, index_path)

    watchlist = (output_dir / "watchlist.md").read_text()
    assert "| Ticker | Confidence | Setup | Trend | Price position | Risk | Reason | Missing for buy candidate | Next action |" in watchlist
    assert "| AAA | low | unknown | unknown | unknown | unknown |" in watchlist
    assert "| BBB | low | unknown | unknown | unknown | unknown |" in watchlist
    assert watchlist.index("| AAA |") < watchlist.index("| BBB |")


def test_missing_data_report_counts_missing_inputs(tmp_path: Path) -> None:
    index_path = _write_status_index(
        tmp_path,
        [
            _watchlist_row(tmp_path, "AAA"),
            _watchlist_row(tmp_path, "BBB"),
            _unable_row("CCC"),
        ],
    )
    output_dir = _run_batch(tmp_path, index_path)

    report = (output_dir / "missing_data_report.md").read_text()
    assert "| portfolio_context | 2 |" in report
    assert "| setup_price_market_context | 2 |" in report
    assert "| fundamental_context | 1 |" in report
    assert "| Ticker | Advice | Setup | Trend | Price position | Risk | Missing for buy candidate | Blockers |" in report


def test_coverage_report_and_summary_include_target_size_and_percentage(
    tmp_path: Path,
) -> None:
    index_path = _write_status_index(tmp_path, [_watchlist_row(tmp_path, "AAA")])
    output_dir = _run_batch(tmp_path, index_path, target_size=500)

    coverage = (output_dir / "coverage_report.md").read_text()
    summary = json.loads((output_dir / "advice_summary.json").read_text())
    assert "| Target tickers | 500 |" in coverage
    assert "| Coverage percentage | 0.20% |" in coverage
    assert summary["target_size"] == 500
    assert summary["tickers_in_status_index"] == 1
    assert summary["tickers_with_advice"] == 1
    assert summary["coverage_percentage"] == 0.2


def test_summary_marks_buy_candidate_batch_ready_for_evaluation(tmp_path: Path) -> None:
    index_path = _write_status_index(tmp_path, [_buy_candidate_row(tmp_path, "AAA")])
    output_dir = _run_batch(tmp_path, index_path)

    summary = json.loads((output_dir / "advice_summary.json").read_text())
    assert summary["advice_counts"]["buy_candidate"] == 1
    assert summary["evaluation_readiness"]["ready_for_outcome_tracking"] is True
    assert summary["recommended_next_sprint"] == (
        "ME-EVAL01 - Advice outcome tracking and feedback loop"
    )


def test_summary_marks_all_watchlist_batch_not_ready_for_evaluation(
    tmp_path: Path,
) -> None:
    index_path = _write_status_index(tmp_path, [_watchlist_row(tmp_path, "AAA")])
    output_dir = _run_batch(tmp_path, index_path)

    summary = json.loads((output_dir / "advice_summary.json").read_text())
    assert summary["advice_counts"]["watchlist"] == 1
    assert summary["evaluation_readiness"]["ready_for_outcome_tracking"] is False
    assert "Only watchlist labels were produced" in summary["evaluation_readiness"]["reason"]
    assert summary["recommended_next_sprint"] == (
        "ME-DATA02 - Import local price/setup snapshots for advice diversity"
    )


def test_batch_output_is_deterministically_sorted(tmp_path: Path) -> None:
    index_path = _write_status_index(
        tmp_path,
        [_watchlist_row(tmp_path, "CCC"), _watchlist_row(tmp_path, "AAA"), _watchlist_row(tmp_path, "BBB")],
    )
    output_dir = _run_batch(tmp_path, index_path)

    advice = json.loads((output_dir / "advice_index.json").read_text())
    assert [row["ticker"] for row in advice["tickers"]] == ["AAA", "BBB", "CCC"]


def test_advice_batch_does_not_require_openai_env(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_" + "API_KEY", raising=False)
    monkeypatch.delenv("MARKET_ENGINE_" + "ADVISORY_MODEL", raising=False)
    index_path = _write_status_index(tmp_path, [_watchlist_row(tmp_path, "AAA")])

    exit_code = run_command(
        [
            "--ticker-status-index",
            index_path.as_posix(),
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--run-id",
            "batch-run",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 0


def test_advice_batch_does_not_touch_network(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ME-ADV02 must not make provider/network calls")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    index_path = _write_status_index(tmp_path, [_watchlist_row(tmp_path, "AAA")])

    exit_code = run_command(
        [
            "--ticker-status-index",
            index_path.as_posix(),
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--run-id",
            "batch-run",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert exit_code == 0


def test_batch_guardrail_manifest_has_no_side_effect_flags(tmp_path: Path) -> None:
    index_path = _write_status_index(tmp_path, [_watchlist_row(tmp_path, "AAA")])
    output_dir = _run_batch(tmp_path, index_path)

    guardrail = json.loads((output_dir / "manifest.json").read_text())[
        "baseline_guardrail"
    ]
    assert guardrail == {
        "advice_labels_produced": True,
        "broker_order_execution_performed": False,
        "openai_api_required": False,
        "portfolio_watchlist_mutation_performed": False,
        "provider_invocation_allowed": False,
        "source_acquisition_performed": False,
    }


def test_allowed_advice_labels_remain_exact_set() -> None:
    assert ADVICE_LABELS == (
        "buy_candidate",
        "wait_for_price",
        "watchlist",
        "avoid_for_now",
        "hold_existing",
        "take_loss_review",
        "unable_to_advise",
    )


def _run_batch(
    tmp_path: Path,
    index_path: Path,
    *,
    target_size: int = 10,
) -> Path:
    exit_code = run_command(
        [
            "--ticker-status-index",
            index_path.as_posix(),
            "--output-root",
            (tmp_path / "out").as_posix(),
            "--run-id",
            "batch-run",
            "--generated-at",
            "2026-07-11T00:00:00Z",
            "--target-size",
            str(target_size),
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert exit_code == 0
    return tmp_path / "out" / "batch-run"


def _write_status_index(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "ticker_status_index.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "market-engine-ticker-status-index-v1",
                "artifact_type": "market-engine-ticker-status-index",
                "run_id": "status-run",
                "generated_at": "2026-07-11T00:00:00Z",
                "summary": {"tickers_total": len(rows)},
                "tickers": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _watchlist_row(tmp_path: Path, ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "status": "blocked",
        "readiness_level": "partial_analysis",
        "context_stale": False,
        "actionable_review_allowed": False,
        "decision_engine_ready": False,
        "blocked_stage": "portfolio_review",
        "blocked_reasons": ["Stage preserves an upstream blocked state."],
        "readiness_blocked_reasons": ["missing_setup_or_price_context"],
        "missing_data_summary": ["portfolio_context"],
        "evidence_families_missing": ["setup_price_market"],
        "artifact_path": _write_dry_run(tmp_path, ticker).as_posix(),
        "artifact_sha256": "sha",
    }


def _buy_candidate_row(tmp_path: Path, ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "status": "ready",
        "readiness_level": "decision_ready",
        "context_stale": False,
        "actionable_review_allowed": True,
        "decision_engine_ready": True,
        "blocked_stage": None,
        "blocked_reasons": [],
        "readiness_blocked_reasons": [],
        "missing_data_summary": [],
        "evidence_families_missing": [],
        "artifact_path": _write_dry_run(
            tmp_path,
            ticker,
            available_context_families=["setup_price_context"],
        ).as_posix(),
        "artifact_sha256": "sha",
    }


def _unable_row(ticker: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "status": "invalid_artifact",
        "readiness_level": "not_ready",
        "context_stale": False,
        "actionable_review_allowed": False,
        "decision_engine_ready": False,
        "blocked_stage": "source_context",
        "blocked_reasons": ["No valid artifact."],
        "readiness_blocked_reasons": [],
        "missing_data_summary": ["fundamental_context"],
        "evidence_families_missing": [],
        "artifact_path": None,
        "artifact_sha256": "",
    }


def _write_dry_run(
    tmp_path: Path,
    ticker: str,
    *,
    available_context_families: list[str] | None = None,
) -> Path:
    path = tmp_path / ticker / "dry_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "payload": {
                    "ticker": ticker,
                    "stage_results": [
                        {
                            "stage_name": "fundamental_observations",
                            "status": "completed",
                        }
                    ],
                    "provenance_summary": {
                        "fundamental_observations": {
                            "fundamental_observations_run_id": f"{ticker.lower()}-fundamental"
                        }
                    },
                    "available_context_families": available_context_families or [],
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path
