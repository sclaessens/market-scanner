from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.market_engine.me_gv05_governor_recommendation_mapping import (
    NEXT_SPRINT,
    run_governor_recommendation_mapping,
)


BASE_FIXTURE_PATH = Path(
    "tests/fixtures/market_engine/governor/"
    "me_gv05_governor_recommendation_case.json"
)
EVALUATED_AT = "2026-07-05T16:00:00Z"


def _base_case() -> dict[str, object]:
    return json.loads(BASE_FIXTURE_PATH.read_text(encoding="utf-8"))


def _set_values(
    case: dict[str, object],
    factor: str,
    values: tuple[float, float, float],
) -> None:
    components = case["factor_evidence"][factor]["score_inputs"]["components"]
    for component, value in zip(components, values, strict=True):
        component["input_value"] = value


def _named_case(
    base: dict[str, object],
    *,
    number: int,
    label: str,
) -> dict[str, object]:
    case = deepcopy(base)
    case["ticker"] = f"GR{number:03d}"
    case["evaluation_id"] = f"gv05-{label}"
    case["company_name"] = f"{label.replace('-', ' ').title()} Example"
    return case


def _batch_cases() -> list[dict[str, object]]:
    base = _base_case()
    preferred = _named_case(base, number=1, label="preferred")

    consider = _named_case(base, number=2, label="consider")
    _set_values(consider, "fundamentals", (0.16, 0.16, 0.1125))
    _set_values(consider, "growth", (0.14, 0.16, 0.16))
    _set_values(consider, "risk", (0.41, 1.75, 1.3))
    _set_values(consider, "data_confidence", (0.8, 0.8, 0.8))

    watch = _named_case(consider, number=3, label="watch")
    _set_values(watch, "growth", (0.1, 0.1, 0.1))

    avoid = _named_case(base, number=4, label="avoid")
    _set_values(avoid, "growth", (-0.1, -0.2, -0.2))

    low_confidence = _named_case(
        base,
        number=5,
        label="low-data-confidence",
    )
    _set_values(
        low_confidence,
        "data_confidence",
        (0.7, 0.7, 0.7),
    )

    hard_conflict = _named_case(base, number=6, label="hard-conflict")
    growth = hard_conflict["factor_evidence"]["growth"]
    growth["conflicting_evidence_references"] = [
        growth["evidence_references"][0],
        growth["evidence_references"][1],
    ]

    missing_valuation = _named_case(
        base,
        number=7,
        label="missing-valuation",
    )
    missing_valuation["factor_evidence"].pop("valuation")

    soft_conflict = _named_case(base, number=8, label="soft-conflict")
    fundamentals = soft_conflict["factor_evidence"]["fundamentals"]
    fundamentals["soft_conflicting_evidence_references"] = [
        fundamentals["evidence_references"][0],
        fundamentals["evidence_references"][1],
    ]

    missing_score = _named_case(base, number=9, label="missing-growth-score")
    missing_score["factor_evidence"]["growth"].pop("score_inputs")
    return [
        preferred,
        consider,
        watch,
        avoid,
        low_confidence,
        hard_conflict,
        missing_valuation,
        soft_conflict,
        missing_score,
    ]


def _write_batch(path: Path) -> None:
    path.write_text(
        json.dumps({"cases": _batch_cases()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(input_path: Path, root: Path) -> dict[str, object]:
    return run_governor_recommendation_mapping(
        input_path=input_path,
        run_id="me-gv05-test",
        evaluation_timestamp=EVALUATED_AT,
        artifact_root=root,
    )


def test_runner_writes_byte_deterministic_json_and_markdown(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.json"
    _write_batch(input_path)
    first = _run(input_path, tmp_path / "first")
    second = _run(input_path, tmp_path / "second")

    assert first == second
    first_json = (
        tmp_path / "first" / "governor_recommendation_mapping.json"
    ).read_bytes()
    second_json = (
        tmp_path / "second" / "governor_recommendation_mapping.json"
    ).read_bytes()
    first_markdown = (
        tmp_path / "first" / "governor_recommendation_mapping_report.md"
    ).read_bytes()
    second_markdown = (
        tmp_path / "second" / "governor_recommendation_mapping_report.md"
    ).read_bytes()
    assert first_json == second_json
    assert first_markdown == second_markdown
    assert json.loads(first_json) == first


def test_runner_summary_counts_and_order_are_exact(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    _write_batch(input_path)
    result = _run(input_path, tmp_path / "artifacts")
    summary = result["summary"]

    assert [item["ticker"] for item in result["evaluations"]] == [
        f"GR{number:03d}" for number in range(1, 10)
    ]
    assert result["evaluation_timestamp"] == EVALUATED_AT
    assert summary["evaluations_total"] == 9
    assert summary["recommendation_eligible_count"] == 5
    assert summary["recommendation_ineligible_count"] == 4
    assert summary["counts_by_recommendation_state"] == {
        "avoid": 1,
        "blocked": 1,
        "consider": 1,
        "insufficient_evidence": 3,
        "preferred": 1,
        "watch": 2,
    }
    assert summary["counts_by_eligibility_state"] == {
        "eligible": 5,
        "ineligible": 4,
    }
    assert summary["hard_conflict_block_count"] == 1
    assert summary["data_confidence_block_count"] == 1
    assert summary["critical_factor_coverage_block_count"] == 1


def test_runner_preserves_aggregation_and_authority_boundaries(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.json"
    _write_batch(input_path)
    result = _run(input_path, tmp_path / "artifacts")

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
        assert result["summary"][field] == 0
    assert not any(result["forbidden_side_effects_confirmed"].values())
    for evaluation in result["evaluations"]:
        assert evaluation["recommendation_state"]["actionable"] is False
        assert (
            evaluation["recommendation_state"]["decision_engine_ready"]
            is False
        )
        assert evaluation["overall_evaluation"]["score"] is None
        assert evaluation["overall_evaluation"]["rank"] is None
        assert (
            evaluation["buy_zone_explanation"]["state"]
            == "blocked_not_authorized"
        )


def test_markdown_contains_mapping_and_boundary_evidence(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.json"
    _write_batch(input_path)
    _run(input_path, tmp_path / "artifacts")
    report = (
        tmp_path
        / "artifacts"
        / "governor_recommendation_mapping_report.md"
    ).read_text(encoding="utf-8")

    assert "# ME-GV05 Governor Recommendation-State Mapping" in report
    assert "## Recommendation Eligibility Summary" in report
    assert "## Mapping Rule Summary" in report
    assert "## Data Confidence, Conflict, and Risk Boundaries" in report
    assert "Non-null overall scores: 0" in report
    assert "`blocked_not_authorized`" in report
    assert NEXT_SPRINT in report


def test_runner_has_no_clock_ticker_or_external_dependency() -> None:
    script_path = Path(
        "scripts/market_engine/me_gv05_governor_recommendation_mapping.py"
    )
    source = script_path.read_text(encoding="utf-8").lower()
    fixture_ticker = _base_case()["ticker"].lower()

    assert fixture_ticker not in source
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
