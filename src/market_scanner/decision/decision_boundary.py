"""Side-effect-free canonical decision/review boundary."""

from __future__ import annotations

from market_scanner.decision.decision_contracts import (
    DecisionReviewPlan,
    DecisionReviewStage,
    ReviewPolicy,
)


DECISION_CANONICAL_OWNER = "src/market_scanner/decision/"

LEGACY_DECISION_AUTHORITIES = (
    "scripts/core/decision_engine.py",
    "src/market_scanner/decisions/decision_engine.py",
    "src/market_scanner/decisions/decision_records.py",
)

ALLOWED_REVIEW_STATES = (
    "review_required",
    "limited_analysis",
    "insufficient_evidence",
    "evidence_available",
    "blocked_by_policy",
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


def build_review_policy() -> ReviewPolicy:
    """Return the review-only policy for canonical decision planning."""

    return ReviewPolicy(
        allowed_review_states=ALLOWED_REVIEW_STATES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        preserves_evidence_limitations=True,
        requires_upstream_analysis_evidence=True,
        final_outcomes_allowed=False,
    )


def build_decision_review_stage() -> DecisionReviewStage:
    """Return the canonical decision/review planning stage."""

    return DecisionReviewStage(
        name="review_state_boundary",
        upstream_analysis_evidence_category="analysis_evidence_and_limitations",
        allowed_review_states=ALLOWED_REVIEW_STATES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        provider_calls_allowed=False,
        data_writes_allowed=False,
        reports_allowed=False,
        telegram_delivery_allowed=False,
        portfolio_watchlist_mutation_allowed=False,
        final_outcomes_allowed=False,
        capital_action_outputs_allowed=False,
        priority_label_outputs_allowed=False,
        numeric_rank_outputs_allowed=False,
        price_projection_outputs_allowed=False,
        execution_quality_outputs_allowed=False,
        legacy_migration_status="legacy_decision_engine_pending_migration",
    )


def _build_policy_block_stage() -> DecisionReviewStage:
    return DecisionReviewStage(
        name="policy_block_review",
        upstream_analysis_evidence_category="review_policy_guardrails",
        allowed_review_states=ALLOWED_REVIEW_STATES,
        blocked_final_state_codes=BLOCKED_FINAL_STATE_CODES,
        blocked_behavior_codes=BLOCKED_BEHAVIOR_CODES,
        provider_calls_allowed=False,
        data_writes_allowed=False,
        reports_allowed=False,
        telegram_delivery_allowed=False,
        portfolio_watchlist_mutation_allowed=False,
        final_outcomes_allowed=False,
        capital_action_outputs_allowed=False,
        priority_label_outputs_allowed=False,
        numeric_rank_outputs_allowed=False,
        price_projection_outputs_allowed=False,
        execution_quality_outputs_allowed=False,
        legacy_migration_status="legacy_investment_semantics_blocked",
    )


def build_decision_review_plan() -> DecisionReviewPlan:
    """Return a deterministic review plan without running Decision Engine logic."""

    return DecisionReviewPlan(
        canonical_owner=DECISION_CANONICAL_OWNER,
        stages=(
            build_decision_review_stage(),
            _build_policy_block_stage(),
        ),
        review_policy=build_review_policy(),
        legacy_decision_authorities=LEGACY_DECISION_AUTHORITIES,
        migration_status="canonical_decision_review_boundary_established",
    )

