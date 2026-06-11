"""Side-effect-free canonical delivery boundary."""

from __future__ import annotations

from market_scanner.delivery.delivery_contracts import (
    DeliveryPlan,
    DeliveryPolicy,
    DeliveryStage,
)


DELIVERY_CANONICAL_OWNER = "src/market_scanner/delivery/"
ARCHIVED_LEGACY_RUNTIME_ROOT = "archive/legacy_runtime/scripts"

LEGACY_DELIVERY_AUTHORITIES = (
    f"{ARCHIVED_LEGACY_RUNTIME_ROOT}/run_scan.py",
    f"{ARCHIVED_LEGACY_RUNTIME_ROOT}/run_full_pipeline.py",
)

ALLOWED_DELIVERY_CHANNELS = (
    "telegram_planned",
    "operator_review_delivery",
    "dry_run_delivery",
    "manual_delivery_review",
)

BLOCKED_EXECUTION_CODES = (
    "telegram_send",
    "telegram_api_call",
    "telegram_bot_post",
    "network_post",
    "network_get",
    "credential_read",
    "production_notification",
    "email_send",
    "write_delivery_artifact",
    "write_reports_daily_telegram_message_txt",
)

BLOCKED_FINAL_STATE_CODES = (
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
)

BLOCKED_BEHAVIOR_CODES = (
    "allocation",
    "conviction",
    "urgency",
    "scoring",
    "target-price",
    "tradeability",
    "recommendation",
)


def build_delivery_policy() -> DeliveryPolicy:
    """Return the planning-only policy for canonical delivery."""

    return DeliveryPolicy(
        allowed_delivery_channels=ALLOWED_DELIVERY_CHANNELS,
        blocked_execution_codes=BLOCKED_EXECUTION_CODES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        delivery_planning_only=True,
        requires_upstream_message_or_report_artifact=True,
        credentials_allowed=False,
        network_calls_allowed=False,
        final_outcomes_allowed=False,
    )


def build_telegram_delivery_plan() -> DeliveryStage:
    """Return the canonical Telegram delivery planning stage without sending."""

    return DeliveryStage(
        name="telegram_delivery_planning",
        upstream_message_report_artifact_category="message_and_report_artifact_data",
        allowed_delivery_channels=ALLOWED_DELIVERY_CHANNELS,
        blocked_execution_codes=BLOCKED_EXECUTION_CODES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        provider_calls_allowed=False,
        network_calls_allowed=False,
        credentials_allowed=False,
        production_data_writes_allowed=False,
        report_file_writes_allowed=False,
        daily_message_file_writes_allowed=False,
        telegram_sending_allowed=False,
        portfolio_watchlist_mutation_allowed=False,
        final_outcomes_allowed=False,
        capital_action_outputs_allowed=False,
        priority_label_outputs_allowed=False,
        numeric_rank_outputs_allowed=False,
        price_projection_outputs_allowed=False,
        execution_quality_outputs_allowed=False,
        report_generation_allowed=False,
        message_composition_allowed=False,
        production_pipeline_allowed=False,
        legacy_migration_status="legacy_delivery_telegram_execution_pending_migration",
    )


def _build_delivery_execution_block_plan() -> DeliveryStage:
    return DeliveryStage(
        name="delivery_execution_policy_block",
        upstream_message_report_artifact_category="delivery_policy_guardrails",
        allowed_delivery_channels=ALLOWED_DELIVERY_CHANNELS,
        blocked_execution_codes=BLOCKED_EXECUTION_CODES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        provider_calls_allowed=False,
        network_calls_allowed=False,
        credentials_allowed=False,
        production_data_writes_allowed=False,
        report_file_writes_allowed=False,
        daily_message_file_writes_allowed=False,
        telegram_sending_allowed=False,
        portfolio_watchlist_mutation_allowed=False,
        final_outcomes_allowed=False,
        capital_action_outputs_allowed=False,
        priority_label_outputs_allowed=False,
        numeric_rank_outputs_allowed=False,
        price_projection_outputs_allowed=False,
        execution_quality_outputs_allowed=False,
        report_generation_allowed=False,
        message_composition_allowed=False,
        production_pipeline_allowed=False,
        legacy_migration_status="network_credentials_and_delivery_execution_blocked",
    )


def build_delivery_plan() -> DeliveryPlan:
    """Return a deterministic delivery plan without delivery execution."""

    return DeliveryPlan(
        canonical_owner=DELIVERY_CANONICAL_OWNER,
        stages=(
            build_telegram_delivery_plan(),
            _build_delivery_execution_block_plan(),
        ),
        delivery_policy=build_delivery_policy(),
        legacy_delivery_authorities=LEGACY_DELIVERY_AUTHORITIES,
        migration_status="canonical_delivery_boundary_established",
    )
