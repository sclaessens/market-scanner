"""Side-effect-free canonical report artifact boundary."""

from __future__ import annotations

from market_scanner.reporting.report_contracts import (
    ReportArtifactPlan,
    ReportArtifactPolicy,
    ReportArtifactStage,
)


REPORTING_CANONICAL_OWNER = "src/market_scanner/reporting/"

LEGACY_REPORT_AUTHORITIES = (
    "scripts/reporting/build_reporting_layer.py",
    "scripts/reporting/build_telegram_summary.py",
    "scripts/reporting/send_telegram.py",
    "src/market_scanner/reporting/reporting_engine.py",
    "src/market_scanner/reporting/reporting_input_adapter.py",
    "src/market_scanner/reporting/telegram_renderer.py",
)

ALLOWED_ARTIFACT_TYPES = (
    "review_report_artifact",
    "limited_analysis_report_artifact",
    "evidence_gap_report_artifact",
    "dry_run_report_artifact",
    "operator_review_artifact",
)

BLOCKED_WRITE_CODES = (
    "write_report_file",
    "write_daily_report",
    "write_daily_message_file",
    "write_telegram_message_file",
    "write_reports_daily_telegram_message_txt",
    "production_report_artifact",
)

BLOCKED_DELIVERY_CODES = (
    "telegram_send",
    "telegram_delivery",
    "email_send",
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


def build_report_artifact_policy() -> ReportArtifactPolicy:
    """Return the planning-only policy for canonical report artifacts."""

    return ReportArtifactPolicy(
        allowed_artifact_types=ALLOWED_ARTIFACT_TYPES,
        blocked_write_codes=BLOCKED_WRITE_CODES,
        blocked_delivery_codes=BLOCKED_DELIVERY_CODES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        artifact_planning_only=True,
        requires_upstream_message_or_review_data=True,
        final_outcomes_allowed=False,
    )


def build_review_report_plan() -> ReportArtifactStage:
    """Return the canonical review-report artifact planning stage."""

    return ReportArtifactStage(
        name="review_report_artifact_planning",
        upstream_message_review_data_category="message_composition_and_review_data",
        allowed_artifact_types=ALLOWED_ARTIFACT_TYPES,
        blocked_write_codes=BLOCKED_WRITE_CODES,
        blocked_delivery_codes=BLOCKED_DELIVERY_CODES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        provider_calls_allowed=False,
        production_data_writes_allowed=False,
        report_file_writes_allowed=False,
        daily_message_file_writes_allowed=False,
        telegram_delivery_allowed=False,
        portfolio_watchlist_mutation_allowed=False,
        final_outcomes_allowed=False,
        capital_action_outputs_allowed=False,
        priority_label_outputs_allowed=False,
        numeric_rank_outputs_allowed=False,
        price_projection_outputs_allowed=False,
        execution_quality_outputs_allowed=False,
        delivery_outputs_allowed=False,
        production_pipeline_allowed=False,
        legacy_migration_status="legacy_report_artifact_generation_pending_migration",
    )


def _build_report_write_block_plan() -> ReportArtifactStage:
    return ReportArtifactStage(
        name="report_write_policy_block",
        upstream_message_review_data_category="report_artifact_policy_guardrails",
        allowed_artifact_types=ALLOWED_ARTIFACT_TYPES,
        blocked_write_codes=BLOCKED_WRITE_CODES,
        blocked_delivery_codes=BLOCKED_DELIVERY_CODES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        provider_calls_allowed=False,
        production_data_writes_allowed=False,
        report_file_writes_allowed=False,
        daily_message_file_writes_allowed=False,
        telegram_delivery_allowed=False,
        portfolio_watchlist_mutation_allowed=False,
        final_outcomes_allowed=False,
        capital_action_outputs_allowed=False,
        priority_label_outputs_allowed=False,
        numeric_rank_outputs_allowed=False,
        price_projection_outputs_allowed=False,
        execution_quality_outputs_allowed=False,
        delivery_outputs_allowed=False,
        production_pipeline_allowed=False,
        legacy_migration_status="report_file_writes_and_delivery_blocked",
    )


def build_report_artifact_plan() -> ReportArtifactPlan:
    """Return a deterministic report artifact plan without writing files."""

    return ReportArtifactPlan(
        canonical_owner=REPORTING_CANONICAL_OWNER,
        stages=(
            build_review_report_plan(),
            _build_report_write_block_plan(),
        ),
        artifact_policy=build_report_artifact_policy(),
        legacy_report_authorities=LEGACY_REPORT_AUTHORITIES,
        migration_status="canonical_report_artifact_boundary_established",
    )
