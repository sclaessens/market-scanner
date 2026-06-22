# ME-RUN18 Roadmap Entry - Local Portfolio Context for Canonical Cached-Source Dry-Runs

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN18

## Placement

ME-RUN18 follows ME-RUN17.

ME-RUN17 fixed ME-SR02 cached-source snapshot discovery and executed canonical-universe cached-source dry-runs, but downstream review remained blocked because required local portfolio context was not provided.

## Completed outcome

ME-RUN18 added local non-production portfolio-context support to the cached-source batch dry-run command.

Implemented behavior:

* `--portfolio-context [PATH]` command argument;
* default path when the flag is supplied without a value: `data/market_engine/portfolio_contexts/local_portfolio_context.json`;
* local wrapper contract: `market-engine-local-portfolio-context-batch-v1`;
* required `non_production_local_context=true`;
* required absence of portfolio-write authority through `portfolio_write_authority=false` or omission;
* per-ticker expansion into `market-engine-portfolio-context-v1` payloads;
* pass-through to existing `portfolio_contexts_by_ticker` runtime input;
* human-visible `PORTFOLIO CONTEXT` command output;
* deterministic test coverage.

## Preserved boundaries

ME-RUN18 remains local, cached-source-only, provider-free, broker-free, portfolio-write-free, watchlist-write-free, delivery-free, scheduler-free, UI-free, and non-actionable.

It does not introduce Decision Engine action semantics, BUY / SELL / HOLD action semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Next RUN candidate

### ME-RUN19 - Execute canonical cached-source dry-run with local portfolio context

Goal: run the canonical-universe cached-source batch with ME-SR02 snapshots and the approved local portfolio context file, then inspect whether Portfolio Review, Decision Engine handoff review, and Delivery Reporting review progress beyond the ME-RUN17 missing-portfolio-context blocker.

Scope: local execution and artifact inspection only. No provider refresh, live market data, broker calls, portfolio writes, watchlist writes, Telegram/email delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, target weights, position sizing, ranking, scoring, urgency, conviction, tradeability, or execution advice.
