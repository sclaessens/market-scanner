"""Canonical v2 application boundary.

This module owns the approved application entrypoint shape without running the
legacy production pipeline. It is intentionally plan-first until scanner,
analysis, reporting, and delivery logic are migrated into canonical v2 owners.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeStage:
    """One planned stage in the canonical v2 runtime flow."""

    name: str
    canonical_owner: str
    status: str
    side_effects_allowed: bool


@dataclass(frozen=True)
class SideEffectGuarantees:
    """Dry-run side-effect guarantees for the canonical app boundary."""

    provider_calls_made: bool
    production_data_writes: bool
    reports_generated: bool
    telegram_artifacts_created: bool
    portfolio_or_watchlist_updates: bool
    legacy_runners_invoked: bool


@dataclass(frozen=True)
class CanonicalRuntimePlan:
    """Deterministic metadata plan for the canonical v2 application flow."""

    entrypoint: str
    stages: tuple[RuntimeStage, ...]
    legacy_runtime_authorities: tuple[str, ...]
    migration_status: str


@dataclass(frozen=True)
class CanonicalAppResult:
    """Dry-run result returned by the canonical app boundary."""

    mode: str
    runtime_plan: CanonicalRuntimePlan
    side_effect_guarantees: SideEffectGuarantees


CANONICAL_ENTRYPOINT = "src/market_scanner/app.py"

LEGACY_RUNTIME_AUTHORITIES = (
    "scripts/run_scan.py",
    "scripts/run_full_pipeline.py",
)

CANONICAL_RUNTIME_STAGES = (
    RuntimeStage(
        name="application_entrypoint",
        canonical_owner="src/market_scanner/app.py",
        status="canonical_boundary_established",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="scanner_universe_selection",
        canonical_owner="src/market_scanner/scanner/",
        status="planned_for_migration",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="provider_source_access",
        canonical_owner="src/market_scanner/fundamentals/",
        status="canonical_boundary_available",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="fundamentals_normalization_evidence",
        canonical_owner="src/market_scanner/fundamentals/",
        status="canonical_boundary_available",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="analysis",
        canonical_owner="src/market_scanner/analysis/",
        status="planned_for_migration",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="decision_review_boundary",
        canonical_owner="src/market_scanner/decision/",
        status="planned_for_migration",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="message_composition",
        canonical_owner="src/market_scanner/messaging/",
        status="planned_for_migration",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="report_generation_where_approved",
        canonical_owner="src/market_scanner/reporting/",
        status="approval_required",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="delivery_telegram_where_approved",
        canonical_owner="src/market_scanner/delivery/",
        status="approval_required",
        side_effects_allowed=False,
    ),
)


def build_canonical_runtime_plan() -> CanonicalRuntimePlan:
    """Return the approved v2 runtime sequence without executing any stage."""

    return CanonicalRuntimePlan(
        entrypoint=CANONICAL_ENTRYPOINT,
        stages=CANONICAL_RUNTIME_STAGES,
        legacy_runtime_authorities=LEGACY_RUNTIME_AUTHORITIES,
        migration_status="canonical_entrypoint_established_legacy_runners_pending",
    )


def run_canonical_app(*, dry_run: bool = True) -> CanonicalAppResult:
    """Return a deterministic dry-run result for the canonical app boundary."""

    if not dry_run:
        raise ValueError("Only dry-run canonical app planning is approved.")

    return CanonicalAppResult(
        mode="dry_run",
        runtime_plan=build_canonical_runtime_plan(),
        side_effect_guarantees=SideEffectGuarantees(
            provider_calls_made=False,
            production_data_writes=False,
            reports_generated=False,
            telegram_artifacts_created=False,
            portfolio_or_watchlist_updates=False,
            legacy_runners_invoked=False,
        ),
    )
