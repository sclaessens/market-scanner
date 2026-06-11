from __future__ import annotations

from market_scanner.reporting.report_boundary import (
    BLOCKED_BEHAVIOR_CODES,
    BLOCKED_DELIVERY_CODES,
    BLOCKED_FINAL_STATE_CODES,
    BLOCKED_WRITE_CODES,
    build_report_artifact_plan,
    build_report_artifact_policy,
)


def _flatten_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten_values(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten_values(item)
    elif hasattr(value, "__dict__"):
        yield from _flatten_values(vars(value))
    else:
        yield str(value)


def test_report_artifact_policy_blocks_runtime_writes_and_delivery():
    policy = build_report_artifact_policy()

    assert policy.artifact_planning_only is True
    assert policy.requires_upstream_message_or_review_data is True
    assert policy.final_outcomes_allowed is False

    assert "write_report_file" in policy.blocked_write_codes
    assert "write_telegram_message_file" in policy.blocked_write_codes
    assert "telegram_delivery" in policy.blocked_delivery_codes
    assert "production_notification" in policy.blocked_delivery_codes


def test_report_artifact_plan_is_planning_only_and_side_effect_free():
    plan = build_report_artifact_plan()

    assert plan.canonical_owner == "src/market_scanner/reporting/"
    assert plan.migration_status == "canonical_report_artifact_boundary_established"
    assert plan.stages

    for stage in plan.stages:
        assert stage.provider_calls_allowed is False
        assert stage.production_data_writes_allowed is False
        assert stage.report_file_writes_allowed is False
        assert stage.daily_message_file_writes_allowed is False
        assert stage.telegram_delivery_allowed is False
        assert stage.portfolio_watchlist_mutation_allowed is False
        assert stage.final_outcomes_allowed is False
        assert stage.production_pipeline_allowed is False


def test_report_artifact_plan_contains_required_blocked_codes():
    output_text = " ".join(_flatten_values(build_report_artifact_plan()))

    for code in BLOCKED_WRITE_CODES:
        assert code in output_text

    for code in BLOCKED_DELIVERY_CODES:
        assert code in output_text

    for code in BLOCKED_FINAL_STATE_CODES:
        assert code in output_text

    for code in BLOCKED_BEHAVIOR_CODES:
        assert code in output_text


def test_report_artifact_plan_no_longer_names_active_script_era_reporting_files():
    output_text = " ".join(_flatten_values(build_report_artifact_plan()))

    forbidden_active_paths = (
        "scripts/reporting/build_reporting_layer.py",
        "scripts/reporting/build_telegram_summary.py",
        "scripts/reporting/send_telegram.py",
        "scripts/telegram/process_telegram_commands.py",
    )

    for path in forbidden_active_paths:
        assert path not in output_text
