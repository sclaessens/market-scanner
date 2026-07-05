from __future__ import annotations

import json
from pathlib import Path

from scripts.market_engine.me_gv03_governor_non_actionable_dry_run import (
    NEXT_SPRINT,
    run_governor_non_actionable_dry_run,
)


FIXTURE_PATH = Path(
    "tests/fixtures/market_engine/governor/"
    "me_gv03_governor_evidence_cases.json"
)
EVALUATED_AT = "2026-07-05T12:00:00Z"


def _run(root: Path) -> dict[str, object]:
    return run_governor_non_actionable_dry_run(
        input_path=FIXTURE_PATH,
        run_id="me-gv03-test",
        evaluation_timestamp=EVALUATED_AT,
        artifact_root=root,
    )


def test_runner_writes_deterministic_json_and_markdown(tmp_path: Path) -> None:
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")

    assert first == second
    json_path = tmp_path / "first" / "governor_evaluation.json"
    markdown_path = tmp_path / "first" / "governor_evaluation_report.md"
    assert json.loads(json_path.read_text(encoding="utf-8")) == first
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# ME-GV03 Governor Non-Actionable Dry-Run" in markdown
    assert "## Factor-State Matrix" in markdown
    assert "Non-null score fields: 0" in markdown
    assert NEXT_SPRINT in markdown


def test_runner_order_timestamp_and_state_counts_are_deterministic(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path / "artifacts")

    assert result["evaluation_timestamp"] == EVALUATED_AT
    assert [item["ticker"] for item in result["evaluations"]] == [
        "GX001",
        "GX002",
        "GX003",
        "GX004",
        "GX005",
        "GX006",
    ]
    assert result["summary"]["counts_by_evaluation_state"] == {
        "blocked": 1,
        "descriptive_only": 1,
        "evaluation_completed_non_actionable": 1,
        "partial_evaluation": 3,
    }


def test_runner_score_weight_and_reserved_counts_are_zero(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path / "artifacts")
    summary = result["summary"]

    assert summary["non_null_score_count"] == 0
    assert summary["non_null_weight_count"] == 0
    assert summary["non_null_rank_count"] == 0
    assert result["reserved_state_counts"] == {
        "actionable": 0,
        "actionable_review": 0,
        "recommendation_state_ready": 0,
        "decision_ready": 0,
        "de_ready": 0,
    }
    assert not any(result["forbidden_side_effects_confirmed"].values())


def test_runner_preserves_conflicts_and_blocked_boundaries(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path / "artifacts")
    evaluations = {item["ticker"]: item for item in result["evaluations"]}
    conflict = evaluations["GX005"]
    growth = next(
        item
        for item in conflict["factor_evaluations"]
        if item["factor"] == "growth"
    )

    assert growth["state"] == "partial"
    assert growth["conflicting_evidence_references"] == [
        "fixture://conflict/growth-series-a",
        "fixture://conflict/growth-series-b",
    ]
    for item in result["evaluations"]:
        assert item["recommendation_state"]["state"] == "blocked"
        assert item["recommendation_state"]["actionable"] is False
        assert item["recommendation_state"]["decision_engine_ready"] is False
        assert item["buy_zone_explanation"]["state"] == "blocked"
        assert item["buy_zone_explanation"]["execution_authorized"] is False
        assert (
            item["position_management_explanation"]["state"]
            == "no_position_context"
        )


def test_runner_has_no_clock_ticker_or_external_dependency() -> None:
    script_path = Path(
        "scripts/market_engine/me_gv03_governor_non_actionable_dry_run.py"
    )
    source = script_path.read_text(encoding="utf-8").lower()
    fixture_tickers = {
        item["ticker"].lower()
        for item in json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]
    }

    assert not any(ticker in source for ticker in fixture_tickers)
    assert "datetime" not in source
    assert "time.time" not in source
    assert not any(
        dependency in source
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
