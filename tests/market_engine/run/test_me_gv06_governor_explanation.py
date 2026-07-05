from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.market_engine.me_gv06_governor_explanation import (
    NEXT_SPRINT,
    run_governor_explanation,
)


BASE_FIXTURE_PATH = Path(
    "tests/fixtures/market_engine/governor/"
    "me_gv05_governor_recommendation_case.json"
)
EVALUATED_AT = "2026-07-05T18:00:00Z"


def _base_case() -> dict[str, object]:
    return json.loads(BASE_FIXTURE_PATH.read_text(encoding="utf-8"))


def _named_case(
    base: dict[str, object],
    *,
    number: int,
    label: str,
) -> dict[str, object]:
    case = deepcopy(base)
    case["ticker"] = f"GE{number:03d}"
    case["evaluation_id"] = f"gv06-{label}"
    case["company_name"] = f"{label.replace('-', ' ').title()} Example"
    case["price_setup_context"]["ticker"] = case["ticker"]
    return case


def _set_values(
    case: dict[str, object],
    factor: str,
    values: tuple[float, float, float],
) -> None:
    components = case["factor_evidence"][factor]["score_inputs"]["components"]
    for component, value in zip(components, values, strict=True):
        component["input_value"] = value


def _with_position(
    case: dict[str, object],
    *,
    position_state: str,
) -> None:
    case["approved_portfolio_context"] = True
    case["factor_evidence"]["portfolio_fit"] = {
        "evidence_references": ["fixture://explanation/portfolio-fit"],
        "level": "complete",
    }
    case["position_context"] = {
        "contract_version": "market-engine-portfolio-context-v1",
        "fresh": True,
        "position_state": position_state,
        "provenance_valid": True,
        "reference": "fixture://explanation/position-context",
        "ticker": case["ticker"],
    }


def _batch_cases() -> list[dict[str, object]]:
    base = _base_case()
    acceptable = _named_case(base, number=1, label="acceptable")

    missing_price = _named_case(base, number=2, label="missing-price")
    missing_price.pop("price_setup_context")

    stale_price = _named_case(base, number=3, label="stale-price")
    stale_price["price_setup_context"]["fresh"] = False

    pullback = _named_case(base, number=4, label="pullback")
    pullback["price_setup_context"]["condition_state"] = "pullback_preferred"

    breakout = _named_case(base, number=5, label="breakout")
    breakout["price_setup_context"][
        "condition_state"
    ] = "breakout_confirmation_required"

    extended = _named_case(base, number=6, label="extended")
    extended["price_setup_context"]["condition_state"] = "extended"
    extended["price_setup_context"][
        "extension_reference"
    ] = "fixture://explanation/extended"

    hard_conflict = _named_case(base, number=7, label="hard-conflict")
    hard_conflict["price_setup_context"]["hard_conflict_references"] = [
        "fixture://explanation/conflict/a",
        "fixture://explanation/conflict/b",
    ]

    soft_conflict = _named_case(base, number=8, label="soft-conflict")
    soft_conflict["price_setup_context"]["soft_conflict_references"] = [
        "fixture://explanation/conflict/soft"
    ]

    not_held = _named_case(base, number=9, label="not-held")
    _with_position(not_held, position_state="not_held")

    held = _named_case(base, number=10, label="held")
    _with_position(held, position_state="held")

    add_review = _named_case(base, number=11, label="add-review")
    _with_position(add_review, position_state="held")
    add_review["price_setup_context"]["additional_setup_confirmation"] = True
    add_review["price_setup_context"][
        "additional_setup_confirmation_reference"
    ] = "fixture://explanation/additional-confirmation"

    reduce_review = _named_case(base, number=12, label="reduce-review")
    _with_position(reduce_review, position_state="held")
    reduce_review["price_setup_context"]["setup_state"] = "deteriorating"

    exit_review = _named_case(base, number=13, label="exit-review")
    _with_position(exit_review, position_state="held")
    _set_values(exit_review, "growth", (-0.1, -0.2, -0.2))
    exit_review["price_setup_context"]["setup_state"] = "invalidated"
    exit_review["price_setup_context"]["invalidation_context"][
        "state"
    ] = "invalidated"

    low_confidence = _named_case(base, number=14, label="low-confidence")
    _set_values(low_confidence, "data_confidence", (0.7, 0.7, 0.7))
    return [
        acceptable,
        missing_price,
        stale_price,
        pullback,
        breakout,
        extended,
        hard_conflict,
        soft_conflict,
        not_held,
        held,
        add_review,
        reduce_review,
        exit_review,
        low_confidence,
    ]


def _write_batch(path: Path) -> None:
    path.write_text(
        json.dumps({"cases": _batch_cases()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(input_path: Path, root: Path) -> dict[str, object]:
    return run_governor_explanation(
        input_path=input_path,
        run_id="me-gv06-test",
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
        tmp_path / "first" / "governor_explanation.json"
    ).read_bytes()
    second_json = (
        tmp_path / "second" / "governor_explanation.json"
    ).read_bytes()
    first_markdown = (
        tmp_path / "first" / "governor_explanation_report.md"
    ).read_bytes()
    second_markdown = (
        tmp_path / "second" / "governor_explanation_report.md"
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
        f"GE{number:03d}" for number in range(1, 15)
    ]
    assert result["evaluation_timestamp"] == EVALUATED_AT
    assert summary["evaluations_total"] == 14
    assert summary["buy_zone_eligible_count"] == 10
    assert summary["buy_zone_ineligible_count"] == 4
    assert summary["counts_by_buy_zone_state"] == {
        "acceptable_zone_context": 4,
        "blocked": 3,
        "extended_avoid_chasing": 1,
        "insufficient_evidence": 1,
        "no_favorable_zone_identified": 3,
        "wait_for_breakout_confirmation": 1,
        "wait_for_pullback": 1,
    }
    assert summary["position_management_eligible_count"] == 5
    assert summary["position_management_ineligible_count"] == 9
    assert summary["counts_by_position_management_state"] == {
        "add_review_context": 1,
        "exit_review_context": 1,
        "hold_context": 1,
        "no_position_context": 10,
        "reduce_review_context": 1,
    }
    assert summary["stale_price_block_count"] == 1
    assert summary["hard_conflict_block_count"] == 1
    assert summary["missing_price_evidence_count"] == 1
    assert summary["missing_position_context_count"] == 9


def test_runner_preserves_all_authority_and_aggregation_boundaries(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.json"
    _write_batch(input_path)
    result = _run(input_path, tmp_path / "artifacts")

    for field in (
        "execution_authorized_count",
        "portfolio_mutation_authorized_count",
        "order_generation_authorized_count",
        "actionable_count",
        "recommendation_state_ready_count",
        "actionable_review_count",
        "decision_ready_count",
        "de_ready_count",
        "non_null_weight_count",
        "non_null_weighted_score_count",
        "non_null_overall_score_count",
        "non_null_rank_count",
    ):
        assert result["summary"][field] == 0
    assert not any(result["forbidden_side_effects_confirmed"].values())


def test_markdown_contains_explanations_and_guardrails(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.json"
    _write_batch(input_path)
    _run(input_path, tmp_path / "artifacts")
    report = (
        tmp_path / "artifacts" / "governor_explanation_report.md"
    ).read_text(encoding="utf-8")

    assert "# ME-GV06 Governor Buy-Zone and Position-Management" in report
    assert "## Buy-Zone Eligibility Summary" in report
    assert "## Position-Management Eligibility Summary" in report
    assert "## Approved Price Conditions and Invalidation" in report
    assert "Execution-authorized: 0" in report
    assert "Portfolio-mutation-authorized: 0" in report
    assert NEXT_SPRINT in report


def test_runner_has_no_clock_ticker_or_external_dependency() -> None:
    script_path = Path(
        "scripts/market_engine/me_gv06_governor_explanation.py"
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
