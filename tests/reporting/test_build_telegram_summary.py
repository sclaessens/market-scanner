from __future__ import annotations

from market_scanner.delivery.delivery_boundary import build_delivery_plan
from market_scanner.messaging.message_boundary import build_message_composition_plan
from market_scanner.reporting.report_boundary import build_report_artifact_plan


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


def test_message_and_report_boundaries_remain_separated_from_delivery_execution():
    message_plan = build_message_composition_plan()
    report_plan = build_report_artifact_plan()
    delivery_plan = build_delivery_plan()

    assert message_plan.composition_policy.composition_only is True
    assert report_plan.artifact_policy.artifact_planning_only is True
    assert delivery_plan.delivery_policy.delivery_planning_only is True

    assert delivery_plan.delivery_policy.credentials_allowed is False
    assert delivery_plan.delivery_policy.network_calls_allowed is False
    assert delivery_plan.delivery_policy.final_outcomes_allowed is False


def test_telegram_summary_contract_is_represented_as_planning_not_script_wrapper():
    output_text = " ".join(
        (
            " ".join(_flatten_values(build_message_composition_plan())),
            " ".join(_flatten_values(build_report_artifact_plan())),
            " ".join(_flatten_values(build_delivery_plan())),
        )
    )

    assert "review_summary" in output_text
    assert "operator_review_message" in output_text
    assert "telegram_planned" in output_text
    assert "telegram_delivery" in output_text
    assert "telegram_send" in output_text

    forbidden_active_paths = (
        "scripts/reporting/build_reporting_layer.py",
        "scripts/reporting/build_telegram_summary.py",
        "scripts/reporting/send_telegram.py",
        "scripts/telegram/process_telegram_commands.py",
    )

    for path in forbidden_active_paths:
        assert path not in output_text


def test_no_delivery_execution_or_investment_recommendation_language_is_allowed():
    output_text = " ".join(
        (
            " ".join(_flatten_values(build_message_composition_plan())),
            " ".join(_flatten_values(build_report_artifact_plan())),
            " ".join(_flatten_values(build_delivery_plan())),
        )
    ).lower()

    forbidden_outputs = (
        "buy now",
        "urgent buy",
        "ranked buy",
        "best buy",
        "top pick",
        "recommended buy",
        "priority action",
        "actionable trade",
    )

    for term in forbidden_outputs:
        assert term not in output_text

