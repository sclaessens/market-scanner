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
- `src/market_engine/source_refresh/advisory_price_evidence.py`: ME-SR25
  advisory-only price acquisition, artifact validation, and consumer boundary.
- `src/market_engine/source_refresh/advisory_ohlc_history.py`: ME-SR26 bounded
  history acquisition, semantic validation, freshness, and integrity replay.
- `src/market_engine/run/current_technical_screening.py`: current screening,
  SR25 reconciliation, and conditional RUN33 handoff contracts.
- `src/market_engine/data/targeted_diversified_fundamental_derivation.py`:
  ME-DATA11 top-25 funnel binding, bounded SEC acquisition, generic DATA10
  derivation orchestration, and comparison evidence.
- `src/market_engine/data/data11_governance.py`: ME-DATA11 RUN30 and downstream
  authority validation, period policy, approval replay, and bounded execution
  gate.
- `config/market_engine/data11_*_authority.json`: tracked checksum authorities
  for RUN30 input and DATA06/RUN31 prestate.
- `config/market_engine/data11_governance_artifacts_v1.schema.json`: schema for
  the new authority, approval, source-evidence, and downstream variants.
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
- `config/market_engine/advisory_price/` and
  `config/market_engine/advisory_price_freshness_policy.json`: advisory price
  schemas and freshness policy.
- `config/market_engine/advisory_ohlc_history_policy.json` and the ME-SR26
  history/RUN33 schemas: non-canonical history and handoff policy authority.

## Historical material

- `docs/archive/` and `archive/`: reference and historical material.
- Historical content does not override `AGENTS.md`, `docs/active/`, or the
  active baseline pointer.
