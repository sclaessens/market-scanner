# ME-RUN24 — Non-production portfolio-context fixture for expanded scans backlog entry

## Status

Completed by ME-RUN24.

## Goal

Add an explicit, controlled, non-production portfolio-context fixture path for expanded cached-source scans so ME-RUN23-style `supported_cached` tickers can proceed into Portfolio Review without broker data, live portfolio data, or portfolio/watchlist mutations.

## Rationale

ME-RUN23 proved source support and cached-source scan selection over the expanded/proposed Professional Swing Universe. Its next blocker was missing `portfolio_context` at Portfolio Review.

This is not a source-support problem and does not require universe canonicalization.

## Scope

Implemented:

- shared loader for the existing `market-engine-local-portfolio-context-batch-v1` fixture contract;
- explicit `--non-production-portfolio-context-fixture <path>` argument for expanded supported-universe scans;
- provenance output for absent versus non-production fixture context;
- fail-closed fixture validation;
- tests for default behavior, fixture pass-through, provenance, deterministic output, and import boundaries.

Out of scope:

- universe canonicalization;
- live provider calls;
- broker APIs;
- real portfolio access;
- portfolio or watchlist mutation;
- Telegram or delivery side effects;
- Decision Engine action-authority changes;
- trading/action semantics.

## Safety boundary

ME-RUN24 is non-production fixture support only. It does not add target prices, ranking, scoring, urgency, conviction, tradeability, allocation, order, or execution semantics.

## Documentation

Implemented documentation:

```text
docs/market_engine/run_reports/me_run24_non_production_portfolio_context_fixture_expanded_scans.md
docs/market_engine/audits/me_run24_non_production_portfolio_context_fixture_expanded_scans_audit.md
docs/market_engine/backlog/me_run24_non_production_portfolio_context_fixture_expanded_scans_backlog_entry.md
docs/market_engine/roadmap/me_run24_non_production_portfolio_context_fixture_expanded_scans_roadmap_entry.md
```
