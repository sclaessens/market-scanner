import importlib
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from market_scanner.messaging.message_boundary import (
    ALLOWED_MESSAGE_TYPES,
    BLOCKED_BEHAVIOR_CODES,
    BLOCKED_DELIVERY_CODES,
    BLOCKED_FINAL_STATE_CODES,
    LEGACY_MESSAGE_AUTHORITIES,
    MESSAGING_CANONICAL_OWNER,
    build_message_composition_plan,
    build_message_composition_policy,
    build_review_message_plan,
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
    "blocked_delivery_codes",
    "blocked_final_state_codes",
    "blocked_behavior_codes",
}

ARCHIVED_LEGACY_RUNTIME_DIR = Path("archive") / "legacy_runtime" / "scripts"


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


def test_message_composition_plan_is_deterministic():
    assert build_message_composition_plan() == build_message_composition_plan()
    assert build_review_message_plan() == build_review_message_plan()
    assert build_message_composition_policy() == build_message_composition_policy()


def test_message_composition_plan_exposes_canonical_owner_and_stage_order():
    plan = build_message_composition_plan()

    assert plan.canonical_owner == MESSAGING_CANONICAL_OWNER
    assert tuple(stage.name for stage in plan.stages) == (
        "review_message_composition",
        "delivery_separation_review",
    )
    assert plan.legacy_message_authorities == LEGACY_MESSAGE_AUTHORITIES
    assert plan.migration_status == "canonical_message_composition_boundary_established"


def test_message_composition_plan_forbids_side_effects_by_default():
    for stage in build_message_composition_plan().stages:
        assert stage.provider_calls_allowed is False
        assert stage.data_writes_allowed is False
        assert stage.report_files_allowed is False
        assert stage.telegram_delivery_allowed is False
        assert stage.portfolio_watchlist_mutation_allowed is False
        assert stage.final_outcomes_allowed is False
        assert stage.capital_action_outputs_allowed is False
        assert stage.priority_label_outputs_allowed is False
        assert stage.numeric_rank_outputs_allowed is False
        assert stage.price_projection_outputs_allowed is False
        assert stage.execution_quality_outputs_allowed is False
        assert stage.delivery_outputs_allowed is False


def test_message_composition_plan_exposes_composition_message_types_only():
    plan = build_message_composition_plan()

    assert plan.composition_policy.allowed_message_types == ALLOWED_MESSAGE_TYPES
    assert plan.composition_policy.allowed_message_types == (
        "review_summary",
        "limited_analysis_summary",
        "evidence_gap_summary",
        "dry_run_summary",
        "operator_review_message",
    )
    assert plan.composition_policy.composition_only is True
    assert plan.composition_policy.requires_upstream_review_data is True
    assert plan.composition_policy.final_outcomes_allowed is False

    blocked = set(BLOCKED_DELIVERY_CODES) | set(BLOCKED_FINAL_STATE_CODES) | set(
        BLOCKED_BEHAVIOR_CODES
    )
    assert set(ALLOWED_MESSAGE_TYPES).isdisjoint(blocked)


def test_message_composition_plan_explicitly_blocks_delivery_behavior():
    plan = build_message_composition_plan()

    assert plan.composition_policy.blocked_delivery_codes == BLOCKED_DELIVERY_CODES
    assert set(BLOCKED_DELIVERY_CODES) == {
        "telegram_send",
        "telegram_delivery",
        "email_send",
        "write_report_file",
        "write_daily_message_file",
        "production_notification",
    }
    for stage in plan.stages:
        assert stage.blocked_delivery_codes == BLOCKED_DELIVERY_CODES


def test_message_composition_plan_explicitly_blocks_investment_semantics():
    plan = build_message_composition_plan()

    assert plan.composition_policy.blocked_final_state_codes == (
        BLOCKED_FINAL_STATE_CODES
    )
    assert plan.composition_policy.blocked_behavior_codes == BLOCKED_BEHAVIOR_CODES
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
        "recommendation",
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


def test_message_composition_boundary_import_and_plan_create_no_files(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.messaging.message_boundary")
    build_message_composition_plan()

    assert list(tmp_path.iterdir()) == []
    assert not Path("data").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
    assert not Path(".github/workflows").exists()


def test_message_composition_boundary_does_not_import_legacy_delivery_or_reporting():
    legacy_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts"
        or module_name.startswith("scripts.")
        or module_name.startswith("market_scanner.reporting")
    }

    importlib.import_module("market_scanner.messaging.message_boundary")
    build_message_composition_plan()

    legacy_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts"
        or module_name.startswith("scripts.")
        or module_name.startswith("market_scanner.reporting")
    }

    assert legacy_modules_after == legacy_modules_before


def test_message_composition_plan_contains_no_investment_behavior_outside_blocked_policy():
    output_text = " ".join(_flatten_values(build_message_composition_plan()))

    for term in FORBIDDEN_OUTPUT_TERMS:
        assert term not in output_text


def test_legacy_message_report_telegram_files_are_not_expanded_to_import_messaging():
    legacy_sources = (
        (ARCHIVED_LEGACY_RUNTIME_DIR / "run_scan.py").read_text(encoding="utf-8"),
        (ARCHIVED_LEGACY_RUNTIME_DIR / "run_full_pipeline.py").read_text(
            encoding="utf-8"
        ),
        Path("src/market_scanner/reporting/reporting_input_adapter.py").read_text(
            encoding="utf-8"
        ),
        Path("src/market_scanner/reporting/telegram_renderer.py").read_text(
            encoding="utf-8"
        ),
    )

    assert not (Path("scripts") / "run_scan.py").exists()
    assert not (Path("scripts") / "run_full_pipeline.py").exists()

    for source in legacy_sources:
        assert "market_scanner.messaging.message_boundary" not in source
        assert "market_scanner.messaging.message_contracts" not in source
        assert "from market_scanner.messaging import" not in source
        assert "market_scanner.app" not in source
