"""Contract records for the canonical v2 delivery boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeliveryPolicy:
    """Planning-only policy for canonical delivery ownership."""

    allowed_delivery_channels: tuple[str, ...]
    blocked_execution_codes: tuple[str, ...]
    blocked_final_state_codes: tuple[str, ...]
    blocked_behavior_codes: tuple[str, ...]
    delivery_planning_only: bool
    requires_upstream_message_or_report_artifact: bool
    credentials_allowed: bool
    network_calls_allowed: bool
    final_outcomes_allowed: bool


@dataclass(frozen=True)
class DeliveryStage:
    """One side-effect-free delivery planning stage."""

    name: str
    upstream_message_report_artifact_category: str
    allowed_delivery_channels: tuple[str, ...]
    blocked_execution_codes: tuple[str, ...]
    blocked_final_state_codes: tuple[str, ...]
    blocked_behavior_codes: tuple[str, ...]
    provider_calls_allowed: bool
    network_calls_allowed: bool
    credentials_allowed: bool
    production_data_writes_allowed: bool
    report_file_writes_allowed: bool
    daily_message_file_writes_allowed: bool
    telegram_sending_allowed: bool
    portfolio_watchlist_mutation_allowed: bool
    final_outcomes_allowed: bool
    capital_action_outputs_allowed: bool
    priority_label_outputs_allowed: bool
    numeric_rank_outputs_allowed: bool
    price_projection_outputs_allowed: bool
    execution_quality_outputs_allowed: bool
    report_generation_allowed: bool
    message_composition_allowed: bool
    production_pipeline_allowed: bool
    legacy_migration_status: str


@dataclass(frozen=True)
class DeliveryPlan:
    """Deterministic delivery plan without delivery execution."""

    canonical_owner: str
    stages: tuple[DeliveryStage, ...]
    delivery_policy: DeliveryPolicy
    legacy_delivery_authorities: tuple[str, ...]
    migration_status: str
