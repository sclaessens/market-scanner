"""Side-effect-free canonical message composition boundary."""

from __future__ import annotations

from market_scanner.messaging.message_contracts import (
    MessageCompositionPlan,
    MessageCompositionPolicy,
    MessageCompositionStage,
)


MESSAGING_CANONICAL_OWNER = "src/market_scanner/messaging/"

LEGACY_MESSAGE_AUTHORITIES = (
    "scripts/reporting/build_reporting_layer.py",
    "scripts/reporting/build_telegram_summary.py",
    "scripts/reporting/send_telegram.py",
    "src/market_scanner/reporting/reporting_input_adapter.py",
    "src/market_scanner/reporting/telegram_renderer.py",
)

ALLOWED_MESSAGE_TYPES = (
    "review_summary",
    "limited_analysis_summary",
    "evidence_gap_summary",
    "dry_run_summary",
    "operator_review_message",
)

BLOCKED_DELIVERY_CODES = (
    "telegram_send",
    "telegram_delivery",
    "email_send",
    "write_report_file",
    "write_daily_message_file",
    "production_notification",
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


def build_message_composition_policy() -> MessageCompositionPolicy:
    """Return the composition-only policy for canonical message planning."""

    return MessageCompositionPolicy(
        allowed_message_types=ALLOWED_MESSAGE_TYPES,
        blocked_delivery_codes=BLOCKED_DELIVERY_CODES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        composition_only=True,
        requires_upstream_review_data=True,
        final_outcomes_allowed=False,
    )


def build_review_message_plan() -> MessageCompositionStage:
    """Return the canonical review-message composition planning stage."""

    return MessageCompositionStage(
        name="review_message_composition",
        upstream_review_data_category="decision_review_states_and_limitations",
        allowed_message_types=ALLOWED_MESSAGE_TYPES,
        blocked_delivery_codes=BLOCKED_DELIVERY_CODES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        provider_calls_allowed=False,
        data_writes_allowed=False,
        report_files_allowed=False,
        telegram_delivery_allowed=False,
        portfolio_watchlist_mutation_allowed=False,
        final_outcomes_allowed=False,
        capital_action_outputs_allowed=False,
        priority_label_outputs_allowed=False,
        numeric_rank_outputs_allowed=False,
        price_projection_outputs_allowed=False,
        execution_quality_outputs_allowed=False,
        delivery_outputs_allowed=False,
        legacy_migration_status="legacy_message_report_delivery_pending_migration",
    )


def _build_delivery_separation_plan() -> MessageCompositionStage:
    return MessageCompositionStage(
        name="delivery_separation_review",
        upstream_review_data_category="message_composition_policy_guardrails",
        allowed_message_types=ALLOWED_MESSAGE_TYPES,
        blocked_delivery_codes=BLOCKED_DELIVERY_CODES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        provider_calls_allowed=False,
        data_writes_allowed=False,
        report_files_allowed=False,
        telegram_delivery_allowed=False,
        portfolio_watchlist_mutation_allowed=False,
        final_outcomes_allowed=False,
        capital_action_outputs_allowed=False,
        priority_label_outputs_allowed=False,
        numeric_rank_outputs_allowed=False,
        price_projection_outputs_allowed=False,
        execution_quality_outputs_allowed=False,
        delivery_outputs_allowed=False,
        legacy_migration_status="delivery_and_report_outputs_blocked",
    )


def build_message_composition_plan() -> MessageCompositionPlan:
    """Return a deterministic composition plan without report or delivery behavior."""

    return MessageCompositionPlan(
        canonical_owner=MESSAGING_CANONICAL_OWNER,
        stages=(
            build_review_message_plan(),
            _build_delivery_separation_plan(),
        ),
        composition_policy=build_message_composition_policy(),
        legacy_message_authorities=LEGACY_MESSAGE_AUTHORITIES,
        migration_status="canonical_message_composition_boundary_established",
    )
