"""Contract records for the canonical v2 decision/review boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewPolicy:
    """Review-only policy for the canonical decision boundary."""

    allowed_review_states: tuple[str, ...]
    blocked_final_state_codes: tuple[str, ...]
    blocked_behavior_codes: tuple[str, ...]
    preserves_evidence_limitations: bool
    requires_upstream_analysis_evidence: bool
    final_outcomes_allowed: bool


@dataclass(frozen=True)
class DecisionReviewStage:
    """One side-effect-free decision/review planning stage."""

    name: str
    upstream_analysis_evidence_category: str
    allowed_review_states: tuple[str, ...]
    blocked_final_state_codes: tuple[str, ...]
    blocked_behavior_codes: tuple[str, ...]
    provider_calls_allowed: bool
    data_writes_allowed: bool
    reports_allowed: bool
    telegram_delivery_allowed: bool
    portfolio_watchlist_mutation_allowed: bool
    final_outcomes_allowed: bool
    capital_action_outputs_allowed: bool
    priority_label_outputs_allowed: bool
    numeric_rank_outputs_allowed: bool
    price_projection_outputs_allowed: bool
    execution_quality_outputs_allowed: bool
    legacy_migration_status: str


@dataclass(frozen=True)
class DecisionReviewPlan:
    """Deterministic decision/review boundary plan without engine execution."""

    canonical_owner: str
    stages: tuple[DecisionReviewStage, ...]
    review_policy: ReviewPolicy
    legacy_decision_authorities: tuple[str, ...]
    migration_status: str

