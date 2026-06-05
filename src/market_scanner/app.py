"""Canonical v2 application boundary.

This module owns the approved application entrypoint shape without running the
legacy production pipeline. It is intentionally plan-first until scanner,
analysis, message composition, reporting, and delivery logic are migrated into
canonical v2 owners.
"""

from __future__ import annotations

from dataclasses import dataclass

from market_scanner.analysis.analysis_boundary import (
    ANALYSIS_CANONICAL_OWNER,
    build_analysis_plan,
)
from market_scanner.analysis.analysis_contracts import AnalysisPlan
from market_scanner.decision.decision_boundary import (
    DECISION_CANONICAL_OWNER,
    build_decision_review_plan,
)
from market_scanner.decision.decision_contracts import DecisionReviewPlan
from market_scanner.delivery.delivery_boundary import (
    DELIVERY_CANONICAL_OWNER,
    build_delivery_plan,
)
from market_scanner.delivery.delivery_contracts import DeliveryPlan
from market_scanner.messaging.message_boundary import (
    MESSAGING_CANONICAL_OWNER,
    build_message_composition_plan,
)
from market_scanner.messaging.message_contracts import MessageCompositionPlan
from market_scanner.reporting.report_boundary import (
    REPORTING_CANONICAL_OWNER,
    build_report_artifact_plan,
)
from market_scanner.reporting.report_contracts import ReportArtifactPlan

from market_scanner.scanner.scanner_boundary import (
    SCANNER_CANONICAL_OWNER,
    build_scanner_plan,
)
from market_scanner.scanner.scanner_contracts import ScannerPlan


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
    scanner_plan: ScannerPlan
    analysis_plan: AnalysisPlan
    decision_review_plan: DecisionReviewPlan
    message_composition_plan: MessageCompositionPlan
    report_artifact_plan: ReportArtifactPlan
    delivery_plan: DeliveryPlan
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
        canonical_owner=SCANNER_CANONICAL_OWNER,
        status="canonical_boundary_established",
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
        canonical_owner=ANALYSIS_CANONICAL_OWNER,
        status="canonical_boundary_established",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="decision_review_boundary",
        canonical_owner=DECISION_CANONICAL_OWNER,
        status="canonical_boundary_established",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="message_composition",
        canonical_owner=MESSAGING_CANONICAL_OWNER,
        status="canonical_boundary_established",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="report_generation_where_approved",
        canonical_owner=REPORTING_CANONICAL_OWNER,
        status="canonical_boundary_established",
        side_effects_allowed=False,
    ),
    RuntimeStage(
        name="delivery_telegram_where_approved",
        canonical_owner=DELIVERY_CANONICAL_OWNER,
        status="canonical_boundary_established",
        side_effects_allowed=False,
    ),
)


def build_canonical_runtime_plan() -> CanonicalRuntimePlan:
    """Return the approved v2 runtime sequence without executing any stage."""

    return CanonicalRuntimePlan(
        entrypoint=CANONICAL_ENTRYPOINT,
        stages=CANONICAL_RUNTIME_STAGES,
        scanner_plan=build_scanner_plan(),
        analysis_plan=build_analysis_plan(),
        decision_review_plan=build_decision_review_plan(),
        message_composition_plan=build_message_composition_plan(),
        report_artifact_plan=build_report_artifact_plan(),
        delivery_plan=build_delivery_plan(),
        legacy_runtime_authorities=LEGACY_RUNTIME_AUTHORITIES,
        migration_status=(
            "canonical_entrypoint_scanner_analysis_decision_review_message_report_and_delivery_boundary_established"
        ),
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
