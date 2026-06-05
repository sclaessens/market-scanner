"""Contract records for the canonical v2 message composition boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MessageCompositionPolicy:
    """Composition-only policy for canonical message planning."""

    allowed_message_types: tuple[str, ...]
    blocked_delivery_codes: tuple[str, ...]
    blocked_final_state_codes: tuple[str, ...]
    blocked_behavior_codes: tuple[str, ...]
    composition_only: bool
    requires_upstream_review_data: bool
    final_outcomes_allowed: bool


@dataclass(frozen=True)
class MessageCompositionStage:
    """One side-effect-free message composition planning stage."""

    name: str
    upstream_review_data_category: str
    allowed_message_types: tuple[str, ...]
    blocked_delivery_codes: tuple[str, ...]
    blocked_final_state_codes: tuple[str, ...]
    blocked_behavior_codes: tuple[str, ...]
    provider_calls_allowed: bool
    data_writes_allowed: bool
    report_files_allowed: bool
    telegram_delivery_allowed: bool
    portfolio_watchlist_mutation_allowed: bool
    final_outcomes_allowed: bool
    capital_action_outputs_allowed: bool
    priority_label_outputs_allowed: bool
    numeric_rank_outputs_allowed: bool
    price_projection_outputs_allowed: bool
    execution_quality_outputs_allowed: bool
    delivery_outputs_allowed: bool
    legacy_migration_status: str


@dataclass(frozen=True)
class MessageCompositionPlan:
    """Deterministic message composition plan without delivery execution."""

    canonical_owner: str
    stages: tuple[MessageCompositionStage, ...]
    composition_policy: MessageCompositionPolicy
    legacy_message_authorities: tuple[str, ...]
    migration_status: str
