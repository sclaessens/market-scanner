"""Contract records for the canonical v2 analysis boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisStage:
    """One side-effect-free analysis planning stage."""

    name: str
    input_evidence_category: str
    required_upstream_ownership: tuple[str, ...]
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
class AnalysisInputPolicy:
    """Approved evidence intake policy for canonical analysis planning."""

    governed_evidence_required: bool
    accepts_source_derived_free_cash_flow: bool
    accepts_growth_available_evidence: bool
    preserves_review_limitations: bool
    missing_values_must_remain_explicit: bool
    final_outcomes_allowed: bool


@dataclass(frozen=True)
class AnalysisPlan:
    """Deterministic analysis boundary plan without analysis execution."""

    canonical_owner: str
    stages: tuple[AnalysisStage, ...]
    input_policy: AnalysisInputPolicy
    legacy_analysis_authorities: tuple[str, ...]
    migration_status: str

