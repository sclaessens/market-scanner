# ME-RUN14 - First real cached-source batch dry-run execution and visibility contract backlog entry

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN14

## Goal

Define the first real cached-source batch dry-run execution and terminal/artifact visibility contract.

## Rationale

ME-RUN13 implemented the safe cached-source batch dry-run runtime behavior, but the project still needs a precise operator contract for the first real cached-source batch run.

The next step must not jump straight into broad production-like execution. It must first define exactly how the first real local batch dry-run is launched, observed, copied from terminal output, inspected through artifacts, and reviewed as evidence.

## Scope

Documentation-only sprint.

ME-RUN14 defines:

* first real cached-source batch dry-run objective;
* approved local cached-source input boundaries;
* approved ticker-set modes;
* minimum first-run ticker-set expectations;
* pre-run checks;
* terminal visibility sections;
* per-ticker execution progress visibility;
* batch summary visibility;
* artifact visibility;
* evidence bundle requirements;
* blocked/failure triage requirements;
* batch-level fail-closed conditions;
* forbidden side effects;
* ME-RUN15 implementation requirements.

## Implemented documentation

```text
docs/market_engine/run/me_run14_first_real_cached_source_batch_dry_run_visibility_contract.md
docs/market_engine/audits/me_run14_first_real_cached_source_batch_dry_run_visibility_contract_audit.md
docs/market_engine/backlog/me_run14_first_real_cached_source_batch_dry_run_visibility_contract_backlog_entry.md
docs/market_engine/roadmap/me_run14_first_real_cached_source_batch_dry_run_visibility_contract_roadmap_entry.md
```

## Defined contract

ME-RUN14 preserves:

```text
market-engine-cached-source-batch-dry-run-v1
market-engine-end-to-end-dry-run-v1
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

ME-RUN14 defines the operator execution visibility contract:

```text
market-engine-real-cached-source-batch-dry-run-visibility-v1
```

This visibility contract is a documentation and acceptance contract only. It is not a new runtime output family.

## Outcome

ME-RUN14 defines how the first real cached-source batch dry-run must be executed and reviewed.

The first real run must remain local, cached-source-only, deterministic, provider-free, broker-free, delivery-free, portfolio/write-free, watchlist/write-free, scheduler-free, UI-free, and non-actionable.

The run must expose terminal sections for run context, discovery, selected tickers, progress, summary, blocked/failed tickers, artifacts, forbidden side-effect confirmation, and next review actions.

Artifact writing remains opt-in only.

## Non-goals

ME-RUN14 does not introduce runtime code, tests, fixtures, provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, external API calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, generated artifact commits, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Recommended next sprint

```text
ME-RUN15 - Implement first real cached-source batch dry-run command visibility
```

ME-RUN15 should implement only the narrow operator-facing command and visibility behavior defined by ME-RUN14.
