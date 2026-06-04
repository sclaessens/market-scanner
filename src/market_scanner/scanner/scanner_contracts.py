"""Contract records for the canonical v2 scanner boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScannerStage:
    """One side-effect-free scanner or universe-selection planning stage."""

    name: str
    input_source_category: str
    candidate_universe_source: str
    provider_calls_allowed: bool
    data_writes_allowed: bool
    portfolio_watchlist_mutation_allowed: bool
    reports_allowed: bool
    telegram_delivery_allowed: bool
    legacy_migration_status: str


@dataclass(frozen=True)
class ScannerPlan:
    """Deterministic scanner boundary plan without scanner execution."""

    canonical_owner: str
    stages: tuple[ScannerStage, ...]
    legacy_scanner_authorities: tuple[str, ...]
    migration_status: str
