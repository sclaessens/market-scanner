"""Side-effect-free canonical scanner and universe-selection boundary."""

from __future__ import annotations

from market_scanner.scanner.scanner_contracts import ScannerPlan, ScannerStage


SCANNER_CANONICAL_OWNER = "src/market_scanner/scanner/"

LEGACY_SCANNER_AUTHORITIES = (
    "scripts/run_scan.py",
    "scripts/core/data_fetcher.py",
    "scripts/core/scanner.py",
)


def build_universe_selection_plan() -> ScannerStage:
    """Return the canonical universe-selection planning stage."""

    return ScannerStage(
        name="universe_selection",
        input_source_category="configured_universe_reference",
        candidate_universe_source="canonical_scanner_boundary_plan",
        provider_calls_allowed=False,
        data_writes_allowed=False,
        portfolio_watchlist_mutation_allowed=False,
        reports_allowed=False,
        telegram_delivery_allowed=False,
        legacy_migration_status="legacy_universe_loading_pending_migration",
    )


def _build_candidate_construction_plan() -> ScannerStage:
    return ScannerStage(
        name="candidate_construction",
        input_source_category="scanner_candidate_inputs",
        candidate_universe_source="canonical_scanner_boundary_plan",
        provider_calls_allowed=False,
        data_writes_allowed=False,
        portfolio_watchlist_mutation_allowed=False,
        reports_allowed=False,
        telegram_delivery_allowed=False,
        legacy_migration_status="legacy_scan_execution_pending_migration",
    )


def build_scanner_plan() -> ScannerPlan:
    """Return a deterministic scanner plan without running a real scan."""

    return ScannerPlan(
        canonical_owner=SCANNER_CANONICAL_OWNER,
        stages=(
            build_universe_selection_plan(),
            _build_candidate_construction_plan(),
        ),
        legacy_scanner_authorities=LEGACY_SCANNER_AUTHORITIES,
        migration_status="canonical_scanner_boundary_established",
    )
