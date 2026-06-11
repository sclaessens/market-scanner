import importlib
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from market_scanner.reporting.report_boundary import (
    ALLOWED_ARTIFACT_TYPES,
    BLOCKED_BEHAVIOR_CODES,
    BLOCKED_DELIVERY_CODES,
    BLOCKED_FINAL_STATE_CODES,
    BLOCKED_WRITE_CODES,
    LEGACY_REPORT_AUTHORITIES,
    REPORTING_CANONICAL_OWNER,
    build_report_artifact_plan,
    build_report_artifact_policy,
    build_review_report_plan,
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
    "blocked_write_codes",
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


def test_report_artifact_plan_is_deterministic():
    assert build_report_artifact_plan() == build_report_artifact_plan()
    assert build_review_report_plan() == build_review_report_plan()
    assert build_report_artifact_policy() == build_report_artifact_policy()


def test_report_artifact_plan_exposes_canonical_owner_and_stage_order():
    plan = build_report_artifact_plan()

    assert plan.canonical_owner == REPORTING_CANONICAL_OWNER
    assert tuple(stage.name for stage in plan.stages) == (
        "review_report_artifact_planning",
        "report_write_policy_block",
    )
    assert plan.legacy_report_authorities == LEGACY_REPORT_AUTHORITIES
    assert plan.migration_status == "canonical_report_artifact_boundary_established"


def test_report_artifact_plan_forbids_side_effects_by_default():
    for stage in build_report_artifact_plan().stages:
        assert stage.provider_calls_allowed is False
        assert stage.production_data_writes_allowed is False
        assert stage.report_file_writes_allowed is False
        assert stage.daily_message_file_writes_allowed is False
        assert stage.telegram_delivery_allowed is False
        assert stage.portfolio_watchlist_mutation_allowed is False
        assert stage.final_outcomes_allowed is False
        assert stage.capital_action_outputs_allowed is False
        assert stage.priority_label_outputs_allowed is False
        assert stage.numeric_rank_outputs_allowed is False
        assert stage.price_projection_outputs_allowed is False
        assert stage.execution_quality_outputs_allowed is False
        assert stage.delivery_outputs_allowed is False
        assert stage.production_pipeline_allowed is False


def test_report_artifact_plan_exposes_artifact_planning_types_only():
    plan = build_report_artifact_plan()

    assert plan.artifact_policy.allowed_artifact_types == ALLOWED_ARTIFACT_TYPES
    assert plan.artifact_policy.allowed_artifact_types == (
        "review_report_artifact",
        "limited_analysis_report_artifact",
        "evidence_gap_report_artifact",
        "dry_run_report_artifact",
        "operator_review_artifact",
    )
    assert plan.artifact_policy.artifact_planning_only is True
    assert plan.artifact_policy.requires_upstream_message_or_review_data is True
    assert plan.artifact_policy.final_outcomes_allowed is False

    blocked = (
        set(BLOCKED_WRITE_CODES)
        | set(BLOCKED_DELIVERY_CODES)
        | set(BLOCKED_FINAL_STATE_CODES)
        | set(BLOCKED_BEHAVIOR_CODES)
    )
    assert set(ALLOWED_ARTIFACT_TYPES).isdisjoint(blocked)


def test_report_artifact_plan_explicitly_blocks_report_writes():
    plan = build_report_artifact_plan()

    assert plan.artifact_policy.blocked_write_codes == BLOCKED_WRITE_CODES
    assert set(BLOCKED_WRITE_CODES) == {
        "write_report_file",
        "write_daily_report",
        "write_daily_message_file",
        "write_telegram_message_file",
        "write_reports_daily_telegram_message_txt",
        "production_report_artifact",
    }
    for stage in plan.stages:
        assert stage.blocked_write_codes == BLOCKED_WRITE_CODES


def test_report_artifact_plan_explicitly_blocks_delivery_behavior():
    plan = build_report_artifact_plan()

    assert plan.artifact_policy.blocked_delivery_codes == BLOCKED_DELIVERY_CODES
    assert set(BLOCKED_DELIVERY_CODES) == {
        "telegram_send",
        "telegram_delivery",
        "email_send",
        "production_notification",
    }
    for stage in plan.stages:
        assert stage.blocked_delivery_codes == BLOCKED_DELIVERY_CODES


def test_report_artifact_plan_explicitly_blocks_investment_semantics():
    plan = build_report_artifact_plan()

    assert plan.artifact_policy.blocked_final_state_codes == (
        BLOCKED_FINAL_STATE_CODES
    )
    assert plan.artifact_policy.blocked_behavior_codes == BLOCKED_BEHAVIOR_CODES
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


def test_report_artifact_boundary_import_and_plan_create_no_files(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.reporting.report_boundary")
    build_report_artifact_plan()

    assert list(tmp_path.iterdir()) == []
    assert not Path("data").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
    assert not Path(".github/workflows").exists()


def test_report_artifact_boundary_does_not_import_legacy_report_or_telegram():
    legacy_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts"
        or module_name.startswith("scripts.")
        or module_name in {"market_scanner.reporting.telegram_renderer"}
        or module_name in {"market_scanner.reporting.reporting_input_adapter"}
    }

    importlib.import_module("market_scanner.reporting.report_boundary")
    build_report_artifact_plan()

    legacy_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts"
        or module_name.startswith("scripts.")
        or module_name in {"market_scanner.reporting.telegram_renderer"}
        or module_name in {"market_scanner.reporting.reporting_input_adapter"}
    }

    assert legacy_modules_after == legacy_modules_before


def test_report_artifact_plan_contains_no_investment_behavior_outside_blocked_policy():
    output_text = " ".join(_flatten_values(build_report_artifact_plan()))

    for term in FORBIDDEN_OUTPUT_TERMS:
        assert term not in output_text


def test_legacy_report_message_telegram_files_are_not_expanded_to_import_reporting():
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
        assert "market_scanner.reporting.report_boundary" not in source
        assert "market_scanner.reporting.report_contracts" not in source
        assert "market_scanner.app" not in source
