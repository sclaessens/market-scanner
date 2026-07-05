from __future__ import annotations

import json
from pathlib import Path

from scripts.market_engine.me_gv04_governor_factor_scoring import (
    NEXT_SPRINT,
    run_governor_factor_scoring,
)


FIXTURE_PATH = Path(
    "tests/fixtures/market_engine/governor/"
    "me_gv04_governor_scoring_cases.json"
)
EVALUATED_AT = "2026-07-05T14:00:00Z"


def _run(root: Path) -> dict[str, object]:
    return run_governor_factor_scoring(
        input_path=FIXTURE_PATH,
        run_id="me-gv04-test",
        evaluation_timestamp=EVALUATED_AT,
        artifact_root=root,
    )


def test_runner_writes_byte_deterministic_json_and_markdown(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")

    assert first == second
    first_json = (
        tmp_path / "first" / "governor_factor_scoring.json"
    ).read_bytes()
    second_json = (
        tmp_path / "second" / "governor_factor_scoring.json"
    ).read_bytes()
    first_markdown = (
        tmp_path / "first" / "governor_factor_scoring_report.md"
    ).read_bytes()
    second_markdown = (
        tmp_path / "second" / "governor_factor_scoring_report.md"
    ).read_bytes()
    assert first_json == second_json
    assert first_markdown == second_markdown
    assert json.loads(first_json) == first


def test_runner_summary_and_order_are_exact(tmp_path: Path) -> None:
    result = _run(tmp_path / "artifacts")
    summary = result["summary"]

    assert result["evaluation_timestamp"] == EVALUATED_AT
    assert [item["ticker"] for item in result["evaluations"]] == [
        "GS001",
        "GS002",
    ]
    assert summary["evaluations_total"] == 2
    assert summary["scored_factor_count"] == 4
    assert summary["unscored_factor_count"] == 14
    assert summary["score_null_count"] == 14
    assert summary["conflict_blocked_score_count"] == 1
    assert summary["counts_by_score_contract"] == {
        "market-engine-governor-factor-scoring-v1": 4
    }


def test_runner_preserves_all_aggregation_and_authority_boundaries(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path / "artifacts")
    summary = result["summary"]

    for field in (
        "actionable_count",
        "actionable_review_count",
        "recommendation_state_ready_count",
        "decision_ready_count",
        "de_ready_count",
        "non_null_weight_count",
        "non_null_weighted_score_count",
        "non_null_overall_score_count",
        "non_null_rank_count",
    ):
        assert summary[field] == 0
    assert not any(result["forbidden_side_effects_confirmed"].values())
    for evaluation in result["evaluations"]:
        assert evaluation["overall_evaluation"]["score"] is None
        assert evaluation["overall_evaluation"]["weighted_score"] is None
        assert evaluation["overall_evaluation"]["rank"] is None
        assert (
            evaluation["recommendation_state"]["state"]
            == "blocked"
        )


def test_markdown_contains_scoring_and_boundary_evidence(
    tmp_path: Path,
) -> None:
    _run(tmp_path / "artifacts")
    report = (
        tmp_path / "artifacts" / "governor_factor_scoring_report.md"
    ).read_text(encoding="utf-8")

    assert "# ME-GV04 Governor Factor Scoring Dry-Run" in report
    assert "## Score Contract and Scale" in report
    assert "## Score Component Breakdown" in report
    assert "Conflict-blocked scores: 1" in report
    assert "Non-null overall scores: 0" in report
    assert "`blocked_not_authorized`" in report
    assert NEXT_SPRINT in report


def test_runner_has_no_clock_ticker_or_external_dependency() -> None:
    script_path = Path(
        "scripts/market_engine/me_gv04_governor_factor_scoring.py"
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
