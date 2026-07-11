from __future__ import annotations

import io
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from market_engine.batch_status.artifact_discovery import discover_dry_run_artifacts
from market_engine.batch_status.status_index import (
    build_ticker_status_index,
    write_batch_status_outputs,
)
from market_engine.batch_status.status_index_command import run_command


def test_discovers_multiple_valid_dry_run_artifacts(tmp_path: Path) -> None:
    _write_dry_run(tmp_path / "NVDA" / "dry_run.json", ticker="NVDA")
    _write_dry_run(tmp_path / "AMD" / "dry_run.json", ticker="AMD")

    discovery = discover_dry_run_artifacts(tmp_path)

    assert [candidate.ticker for candidate in discovery.valid_candidates] == ["AMD", "NVDA"]
    assert discovery.summary_dict()["valid_dry_run_artifacts"] == 2


def test_invalid_json_is_recorded_without_crashing(tmp_path: Path) -> None:
    invalid_path = tmp_path / "broken" / "dry_run.json"
    invalid_path.parent.mkdir(parents=True)
    invalid_path.write_text("{not-json", encoding="utf-8")

    discovery = discover_dry_run_artifacts(tmp_path)

    assert len(discovery.invalid_candidates) == 1
    assert discovery.invalid_candidates[0].invalid_reasons == ("invalid_json",)
    assert discovery.failures[0]["failure_type"] == "invalid_json"


def test_duplicate_ticker_uses_deterministic_canonical_selection(tmp_path: Path) -> None:
    older = tmp_path / "older" / "dry_run.json"
    newer = tmp_path / "newer" / "dry_run.json"
    _write_dry_run(
        older,
        ticker="NVDA",
        artifact_created_at="2026-01-01T00:00:00Z",
        dry_run_id="older",
    )
    _write_dry_run(
        newer,
        ticker="NVDA",
        artifact_created_at="2026-01-02T00:00:00Z",
        dry_run_id="newer",
    )

    index = build_ticker_status_index(
        discover_dry_run_artifacts(tmp_path),
        run_id="run",
        generated_at="2026-07-11T00:00:00Z",
    )

    row = index["tickers"][0]
    assert row["ticker"] == "NVDA"
    assert row["dry_run_id"] == "newer"
    assert row["candidate_artifact_count"] == 2


def test_extracts_status_readiness_blockers_missing_and_provenance(tmp_path: Path) -> None:
    _write_dry_run(
        tmp_path / "NVDA" / "dry_run.json",
        ticker="NVDA",
        blocked_stage="portfolio_review",
        blocked_reasons=["Stage preserves an upstream blocked state."],
        missing_data_summary=["portfolio_context"],
        readiness={
            "readiness_level": "partial_analysis",
            "actionable_review_allowed": False,
            "decision_engine_ready": False,
            "context_stale": True,
            "blocked_reasons": ["missing_setup_or_price_context"],
            "evidence_families_missing": ["setup_price_market"],
        },
    )

    index = build_ticker_status_index(
        discover_dry_run_artifacts(tmp_path),
        run_id="run",
        generated_at="2026-07-11T00:00:00Z",
    )

    row = index["tickers"][0]
    assert row["status"] == "blocked"
    assert row["readiness_level"] == "partial_analysis"
    assert row["context_stale"] is True
    assert row["missing_data_summary"] == ["portfolio_context"]
    assert row["readiness_blocked_reasons"] == ["missing_setup_or_price_context"]
    assert row["evidence_families_missing"] == ["setup_price_market"]
    assert row["provenance"]["recommendation_review_state"] == "blocked"


def test_writes_all_output_files_and_markdown_rows(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "out"
    _write_dry_run(artifact_root / "AMD" / "dry_run.json", ticker="AMD")
    discovery = discover_dry_run_artifacts(artifact_root)
    index = build_ticker_status_index(
        discovery,
        run_id="sample",
        generated_at="2026-07-11T00:00:00Z",
    )

    output_dir = write_batch_status_outputs(
        index,
        discovery,
        output_root=output_root,
        run_id="sample",
    )

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "discovery_summary.json",
        "failures.json",
        "manifest.json",
        "ticker_status_index.json",
        "ticker_status_index.md",
    ]
    markdown = (output_dir / "ticker_status_index.md").read_text(encoding="utf-8")
    assert "# Market Engine Ticker Status Index" in markdown
    assert "| AMD | blocked | partial_analysis | no | no | no | portfolio_review | portfolio_context |" in markdown


def test_command_writes_outputs_when_invalid_files_exist(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "out"
    _write_dry_run(artifact_root / "NVDA" / "dry_run.json", ticker="NVDA")
    invalid_path = artifact_root / "broken" / "dry_run.json"
    invalid_path.parent.mkdir(parents=True)
    invalid_path.write_text("{", encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = run_command(
        [
            "--artifact-root",
            artifact_root.as_posix(),
            "--output-root",
            output_root.as_posix(),
            "--run-id",
            "sample",
            "--generated-at",
            "2026-07-11T00:00:00Z",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert (output_root / "sample" / "ticker_status_index.json").exists()
    failures = json.loads((output_root / "sample" / "failures.json").read_text())
    assert failures["failures"][0]["failure_type"] == "invalid_json"


def test_batch_status_index_does_not_require_openai_env(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_" + "API_KEY", raising=False)
    monkeypatch.delenv("MARKET_ENGINE_" + "ADVISORY_MODEL", raising=False)
    _write_dry_run(tmp_path / "NVDA" / "dry_run.json", ticker="NVDA")

    index = build_ticker_status_index(
        discover_dry_run_artifacts(tmp_path),
        run_id="sample",
        generated_at="2026-07-11T00:00:00Z",
    )

    assert index["summary"]["tickers_total"] == 1


def test_batch_status_index_does_not_touch_openai_network(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("batch status index must not make network/provider calls")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    _write_dry_run(tmp_path / "AMD" / "dry_run.json", ticker="AMD")

    index = build_ticker_status_index(
        discover_dry_run_artifacts(tmp_path),
        run_id="sample",
        generated_at="2026-07-11T00:00:00Z",
    )

    assert index["tickers"][0]["ticker"] == "AMD"


def _write_dry_run(
    path: Path,
    *,
    ticker: str,
    artifact_created_at: str = "2026-07-11T00:00:00Z",
    dry_run_id: str | None = None,
    blocked_stage: str | None = "portfolio_review",
    blocked_reasons: list[str] | None = None,
    missing_data_summary: list[str] | None = None,
    readiness: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_format_version": "market-engine-local-dry-run-artifact-v1",
        "artifact_type": "market_engine_end_to_end_dry_run",
        "artifact_created_at": artifact_created_at,
        "payload": {
            "dry_run_id": dry_run_id or f"dry-run-{ticker.lower()}",
            "ticker": ticker,
            "input_mode": "cached_source_snapshot",
            "blocked_stage": blocked_stage,
            "blocked_reasons": blocked_reasons
            if blocked_reasons is not None
            else ["Stage preserves an upstream blocked state."],
            "missing_data_summary": missing_data_summary
            if missing_data_summary is not None
            else ["portfolio_context"],
            "analysis_context_readiness": readiness
            if readiness is not None
            else {
                "readiness_level": "partial_analysis",
                "actionable_review_allowed": False,
                "decision_engine_ready": False,
                "context_stale": False,
                "blocked_reasons": ["missing_setup_or_price_context"],
                "evidence_families_missing": ["setup_price_market"],
            },
            "provenance_summary": {
                "source_context": {"source_refresh_snapshot_id": "snapshot-1"},
                "recommendation_review": {
                    "state": "blocked",
                    "category": "missing_setup_or_price_context",
                },
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.utime(path, (1_700_000_000, 1_700_000_000))
