from __future__ import annotations

import json
from pathlib import Path

from scripts.market_engine.me_run29_expanded_generic_coverage_classification import (
    NEXT_SPRINT,
    run_expanded_generic_coverage_classification,
)


FIXTURE_PATH = Path(
    "tests/fixtures/market_engine/run/me_run29_staging_validation_evidence.json"
)
CLASSIFIED_AT = "2026-07-03T00:00:00Z"


def _run(artifact_root: Path) -> dict[str, object]:
    return run_expanded_generic_coverage_classification(
        input_evidence_path=FIXTURE_PATH,
        run_id="me-run29-test",
        classification_timestamp=CLASSIFIED_AT,
        artifact_root=artifact_root,
    )


def test_run29_writes_deterministic_json_and_markdown(tmp_path: Path) -> None:
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")

    assert first == second
    json_path = tmp_path / "first" / "coverage_classification_summary.json"
    markdown_path = tmp_path / "first" / "coverage_classification_report.md"
    assert json.loads(json_path.read_text(encoding="utf-8")) == first
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# ME-RUN29 Expanded Generic Coverage Classification" in markdown
    assert "## Dominant Blockers" in markdown
    assert "deterministic committed staging-validation fixtures" in markdown
    assert NEXT_SPRINT in markdown


def test_run29_order_and_summary_are_deterministic(tmp_path: Path) -> None:
    summary = _run(tmp_path / "artifacts")

    assert [item["ticker"] for item in summary["results"]] == [
        "ALPHA",
        "BETA",
        "DELTA",
        "EPSILON",
        "ETA",
        "GAMMA",
        "ZETA",
    ]
    assert summary["tickers_total"] == 7
    assert summary["source_families_total"] == 2
    assert summary["staging_entries_total"] == 7
    assert summary["readiness_counts"] == {
        "blocked": 5,
        "descriptive_only": 1,
        "unavailable": 1,
    }


def test_run29_preserves_fail_closed_source_gate_results(tmp_path: Path) -> None:
    summary = _run(tmp_path / "artifacts")
    results = {item["ticker"]: item for item in summary["results"]}

    assert results["ALPHA"]["readiness_status"] == "descriptive_only"
    assert results["ALPHA"]["actionable"] is False
    assert results["BETA"]["source_family_coverage_status"] == "partial"
    assert results["BETA"]["actionable"] is False
    assert results["DELTA"]["coverage_status"] == "unprovenanced"
    assert results["GAMMA"]["coverage_status"] == "stale"
    assert results["EPSILON"]["coverage_status"] == "not_consumable"
    assert results["ETA"]["coverage_status"] == "invalid_manifest"
    assert results["ZETA"]["source_family_coverage_status"] == "unsupported"

    blocker_counts = summary["blocker_counts"]
    assert blocker_counts["missing_provenance"] == 1
    assert blocker_counts["stale_source"] == 1
    assert blocker_counts["source_not_consumable"] == 1
    assert blocker_counts["invalid_manifest"] == 1
    assert blocker_counts["unsupported_source_family"] == 1


def test_run29_reserved_states_and_side_effects_remain_zero(
    tmp_path: Path,
) -> None:
    summary = _run(tmp_path / "artifacts")

    assert summary["reserved_state_counts"] == {
        "actionable": 0,
        "actionable_review": 0,
        "decision_ready": 0,
        "de_ready": 0,
    }
    assert summary["actionable_count"] == 0
    assert summary["decision_ready_count"] == 0
    assert summary["de_ready_count"] == 0
    assert summary["recommendation_eligible_count"] == 0
    assert not any(summary["forbidden_side_effects_confirmed"].values())
    assert summary["next_sprint"] == NEXT_SPRINT


def test_run29_script_has_no_ticker_specific_or_external_dependencies() -> None:
    script_path = Path(
        "scripts/market_engine/me_run29_expanded_generic_coverage_classification.py"
    )
    script_source = script_path.read_text(encoding="utf-8").lower()

    fixture_tickers = {
        str(entry["ticker"]).lower()
        for entry in json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["entries"]
    }
    assert not any(ticker in script_source for ticker in fixture_tickers)
    assert not any(
        dependency in script_source
        for dependency in (
            "import requests",
            "import urllib",
            "import yfinance",
            "import telegram",
            "import smtplib",
            "import socket",
            "import subprocess",
        )
    )
