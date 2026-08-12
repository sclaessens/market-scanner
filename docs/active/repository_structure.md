# Repository Structure

Status: ACTIVE POINTER

This document identifies the current authoritative areas relevant to Market
Engine work. It is not an exhaustive file inventory.

## Root authority

- `AGENTS.md`: repository instructions and authority boundaries.
- `pyproject.toml`: Python project and test configuration.
- `.github/workflows/`: automation and scheduled execution.

## Active documentation

- `docs/active/architecture_current_state.md`: concise current architecture.
- `docs/active/governance_v2.md`: concise active governance.
- `docs/active/repository_structure.md`: this path map.
- `docs/market_engine/roadmap/ACTIVE_BASELINE_DIRECTION.md`: current sprint
  order.
- `docs/market_engine/roadmap/market_engine_roadmap.md`: detailed roadmap and
  historical sequence.
- `docs/market_engine/backlog/market_engine_backlog.md`: detailed backlog.
- `docs/market_engine/audits/`: sprint and run evidence.

## Runtime areas

- `src/market_engine/`: current Market Engine job-family runtime.
- `src/market_scanner/portfolio/`: portfolio source and contract metadata.
- `scripts/portfolio/`: legacy/reference portfolio utilities unless a sprint
  explicitly promotes or replaces a narrow surface.
- `tests/market_engine/`: Market Engine tests.
- `tests/portfolio/`: portfolio source and compatibility tests.

## Data boundaries

- `data/market_engine/`: generated Market Engine data and run outputs subject
  to each contract.
- `data/portfolio/private/`: optional ignored repository-local boundary for
  private ME-PR03 ledgers and projections; live personal data must not be
  committed. Other `data/portfolio/` files are legacy/reference state.
- `artifacts/market_engine/`: run artifacts and compact evidence where
  explicitly approved.
- `market-data` branch: canonical price dataset; separate from planned
  advisory price artifacts.

## Historical material

- `docs/archive/` and `archive/`: reference and historical material.
- Historical content does not override `AGENTS.md`, `docs/active/`, or the
  active baseline pointer.
