"""Side-effect-free canonical analysis boundary."""

from __future__ import annotations

from market_scanner.analysis.analysis_contracts import (
    AnalysisInputPolicy,
    AnalysisPlan,
    AnalysisStage,
)

ANALYSIS_CANONICAL_OWNER = "src/market_scanner/analysis/"

LEGACY_ANALYSIS_AUTHORITIES = (
    "scripts/fundamentals/build_analysis.py",
    "scripts/fundamentals/build_quality.py",
    "scripts/core/build_fundamental_analysis.py",
)

MIGRATED_FUNDAMENTALS_CONTRACT_AUTHORITIES = (
    "src/market_scanner/fundamentals/fundamental_contracts.py",
    "src/market_scanner/fundamentals/fundamentals_metrics_contracts.py",
)

ANALYSIS_REQUIRED_UPSTREAM_OWNERS = (
    "src/market_scanner/scanner/",
    "src/market_scanner/fundamentals/",
)


def build_analysis_input_policy() -> AnalysisInputPolicy:
    """Return the approved evidence intake policy for analysis planning."""

    return AnalysisInputPolicy(
        governed_evidence_required=True,
        accepts_source_derived_free_cash_flow=True,
        accepts_growth_available_evidence=True,
        preserves_review_limitations=True,
        missing_values_must_remain_explicit=True,
        final_outcomes_allowed=False,
    )


def build_fundamental_analysis_plan() -> AnalysisStage:
    """Return the canonical fundamental analysis planning stage."""

    return AnalysisStage(
        name="fundamental_evidence_review",
        input_evidence_category="scanner_candidates_and_governed_fundamentals",
        required_upstream_ownership=ANALYSIS_REQUIRED_UPSTREAM_OWNERS,
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
        legacy_migration_status="legacy_fundamental_analysis_pending_migration",
    )


def _build_profile_review_plan() -> AnalysisStage:
    return AnalysisStage(
        name="profile_evidence_review",
        input_evidence_category="cash_flow_growth_quality_evidence",
        required_upstream_ownership=ANALYSIS_REQUIRED_UPSTREAM_OWNERS,
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
        legacy_migration_status="legacy_profile_analysis_pending_migration",
    )


def _build_limitation_review_plan() -> AnalysisStage:
    return AnalysisStage(
        name="limitation_review",
        input_evidence_category="review_limitation_evidence",
        required_upstream_ownership=ANALYSIS_REQUIRED_UPSTREAM_OWNERS,
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
        legacy_migration_status="legacy_limitation_review_pending_migration",
    )


def build_analysis_plan() -> AnalysisPlan:
    """Return a deterministic analysis plan without running analysis logic."""

    return AnalysisPlan(
        canonical_owner=ANALYSIS_CANONICAL_OWNER,
        stages=(
            build_fundamental_analysis_plan(),
            _build_profile_review_plan(),
            _build_limitation_review_plan(),
        ),
        input_policy=build_analysis_input_policy(),
        legacy_analysis_authorities=LEGACY_ANALYSIS_AUTHORITIES,
        migration_status="canonical_analysis_boundary_established",
    )