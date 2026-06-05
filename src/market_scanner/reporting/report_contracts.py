"""Contract records for the canonical v2 report artifact boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportArtifactPolicy:
    """Planning-only policy for canonical report artifact ownership."""

    allowed_artifact_types: tuple[str, ...]
    blocked_write_codes: tuple[str, ...]
    blocked_delivery_codes: tuple[str, ...]
    blocked_final_state_codes: tuple[str, ...]
    blocked_behavior_codes: tuple[str, ...]
    artifact_planning_only: bool
    requires_upstream_message_or_review_data: bool
    final_outcomes_allowed: bool


@dataclass(frozen=True)
class ReportArtifactStage:
    """One side-effect-free report artifact planning stage."""

    name: str
    upstream_message_review_data_category: str
    allowed_artifact_types: tuple[str, ...]
    blocked_write_codes: tuple[str, ...]
    blocked_delivery_codes: tuple[str, ...]
    blocked_final_state_codes: tuple[str, ...]
    blocked_behavior_codes: tuple[str, ...]
    provider_calls_allowed: bool
    production_data_writes_allowed: bool
    report_file_writes_allowed: bool
    daily_message_file_writes_allowed: bool
    telegram_delivery_allowed: bool
    portfolio_watchlist_mutation_allowed: bool
    final_outcomes_allowed: bool
    capital_action_outputs_allowed: bool
    priority_label_outputs_allowed: bool
    numeric_rank_outputs_allowed: bool
    price_projection_outputs_allowed: bool
    execution_quality_outputs_allowed: bool
    delivery_outputs_allowed: bool
    production_pipeline_allowed: bool
    legacy_migration_status: str


@dataclass(frozen=True)
class ReportArtifactPlan:
    """Deterministic report artifact plan without file writes."""

    canonical_owner: str
    stages: tuple[ReportArtifactStage, ...]
    artifact_policy: ReportArtifactPolicy
    legacy_report_authorities: tuple[str, ...]
    migration_status: str
