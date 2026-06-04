import importlib
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from market_scanner.decision.decision_boundary import (
    ALLOWED_REVIEW_STATES,
    BLOCKED_BEHAVIOR_CODES,
    BLOCKED_FINAL_STATE_CODES,
    DECISION_CANONICAL_OWNER,
    LEGACY_DECISION_AUTHORITIES,
    build_decision_review_plan,
    build_decision_review_stage,
    build_review_policy,
)


FORBIDDEN_OUTPUT_TERMS = {
    "BUY",
    "SELL",
    "HOLD",
    "allocation",
    "conviction",
    "urgency",
    "scoring",
    "target-price",
    "tradeability",
    "recommendation",
}

BLOCKED_POLICY_FIELDS = {
    "blocked_final_state_codes",
    "blocked_behavior_codes",
}


def _flatten_values(value):
    if is_dataclass(value):
        yield from _flatten_values(asdict(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in BLOCKED_POLICY_FIELDS:
                continue
            yield str(key)
            yield from _flatten_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_values(item)
    else:
        yield str(value)


def test_decision_review_plan_is_deterministic():
    assert build_decision_review_plan() == build_decision_review_plan()
    assert build_decision_review_stage() == build_decision_review_stage()
    assert build_review_policy() == build_review_policy()


def test_decision_review_plan_exposes_canonical_owner_and_stage_order():
    plan = build_decision_review_plan()

    assert plan.canonical_owner == DECISION_CANONICAL_OWNER
    assert tuple(stage.name for stage in plan.stages) == (
        "review_state_boundary",
        "policy_block_review",
    )
    assert plan.legacy_decision_authorities == LEGACY_DECISION_AUTHORITIES
    assert plan.migration_status == "canonical_decision_review_boundary_established"


def test_decision_review_plan_forbids_side_effects_and_final_outputs_by_default():
    for stage in build_decision_review_plan().stages:
        assert stage.provider_calls_allowed is False
        assert stage.data_writes_allowed is False
        assert stage.reports_allowed is False
        assert stage.telegram_delivery_allowed is False
        assert stage.portfolio_watchlist_mutation_allowed is False
        assert stage.final_outcomes_allowed is False
        assert stage.capital_action_outputs_allowed is False
        assert stage.priority_label_outputs_allowed is False
        assert stage.numeric_rank_outputs_allowed is False
        assert stage.price_projection_outputs_allowed is False
        assert stage.execution_quality_outputs_allowed is False


def test_decision_review_plan_exposes_review_states_only():
    plan = build_decision_review_plan()

    assert plan.review_policy.allowed_review_states == ALLOWED_REVIEW_STATES
    assert plan.review_policy.allowed_review_states == (
        "review_required",
        "limited_analysis",
        "insufficient_evidence",
        "evidence_available",
        "blocked_by_policy",
    )
    for stage in plan.stages:
        assert stage.allowed_review_states == ALLOWED_REVIEW_STATES
        assert set(stage.allowed_review_states).isdisjoint(BLOCKED_FINAL_STATE_CODES)


def test_decision_review_plan_explicitly_blocks_final_decision_semantics():
    plan = build_decision_review_plan()

    assert plan.review_policy.blocked_final_state_codes == BLOCKED_FINAL_STATE_CODES
    assert plan.review_policy.blocked_behavior_codes == BLOCKED_BEHAVIOR_CODES
    assert set(BLOCKED_FINAL_STATE_CODES) == {
        "buy",
        "sell",
        "hold",
        "allocate",
        "increase_position",
        "reduce_position",
        "target_price",
        "tradeable",
        "not_tradeable",
    }
    assert set(BLOCKED_BEHAVIOR_CODES) == {
        "allocation",
        "conviction",
        "urgency",
        "scoring",
        "target-price",
        "tradeability",
        "recommendation",
    }
    for stage in plan.stages:
        assert stage.blocked_final_state_codes == BLOCKED_FINAL_STATE_CODES
        assert stage.blocked_behavior_codes == BLOCKED_BEHAVIOR_CODES


def test_decision_review_boundary_import_and_plan_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.decision.decision_boundary")
    build_decision_review_plan()

    assert list(tmp_path.iterdir()) == []
    assert not Path("data").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
    assert not Path(".github/workflows").exists()


def test_decision_review_boundary_does_not_import_legacy_scripts_or_engine():
    legacy_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts"
        or module_name.startswith("scripts.")
        or module_name.startswith("market_scanner.decisions")
    }

    importlib.import_module("market_scanner.decision.decision_boundary")
    build_decision_review_plan()

    legacy_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts"
        or module_name.startswith("scripts.")
        or module_name.startswith("market_scanner.decisions")
    }

    assert legacy_modules_after == legacy_modules_before


def test_decision_review_plan_contains_no_investment_behavior_outside_blocked_policy():
    output_text = " ".join(_flatten_values(build_decision_review_plan()))

    for term in FORBIDDEN_OUTPUT_TERMS:
        assert term not in output_text


def test_legacy_runners_and_decision_files_are_not_expanded_to_import_canonical_decision():
    legacy_sources = (
        Path("scripts/run_scan.py").read_text(encoding="utf-8"),
        Path("scripts/run_full_pipeline.py").read_text(encoding="utf-8"),
        Path("scripts/core/decision_engine.py").read_text(encoding="utf-8"),
        Path("src/market_scanner/decisions/decision_engine.py").read_text(
            encoding="utf-8"
        ),
        Path("src/market_scanner/decisions/decision_records.py").read_text(
            encoding="utf-8"
        ),
    )

    for source in legacy_sources:
        assert "market_scanner.decision.decision_boundary" not in source
        assert "market_scanner.decision.decision_contracts" not in source
        assert "from market_scanner.decision import" not in source
        assert "market_scanner.app" not in source
