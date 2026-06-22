# ME-RUN18 Backlog Entry - Local Portfolio Context for Canonical Cached-Source Dry-Runs

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN18

## Problem

ME-RUN17 fixed cached-source discovery for ME-SR02 snapshots and executed 12 canonical-universe dry-run payloads, but downstream review remained blocked because local portfolio context was not provided to Portfolio Review.

## Goal

Provide approved local non-production portfolio context to canonical-universe cached-source dry-runs without production portfolio writes or financial-action authority.

## Acceptance criteria

* Command accepts an explicit local portfolio context file.
* Local context uses a batch wrapper with clear non-production semantics.
* Command rejects context files that omit `non_production_local_context=true`.
* Command rejects context files that grant portfolio write authority.
* Context expands deterministically to per-ticker `market-engine-portfolio-context-v1` payloads.
* Context is passed to the existing cached-source batch runtime boundary.
* Human-visible command output shows portfolio-context metadata.
* Tests cover default-off behavior and enabled local portfolio context behavior.
* No provider, broker, portfolio write, watchlist write, Telegram, production report, scheduler, UI, or Decision Engine action authority is introduced.

## Outcome

ME-RUN18 implemented `--portfolio-context` support in:

```text
src/market_engine/run/cached_source_batch_dry_run_command.py
```

and added deterministic tests in:

```text
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

Documentation and audit were added under:

```text
docs/market_engine/run/me_run18_local_portfolio_context_for_canonical_cached_source_dry_runs.md
docs/market_engine/audits/me_run18_local_portfolio_context_for_canonical_cached_source_dry_runs_audit.md
```

## Next candidate

After ME-RUN18 is validated locally, the next candidate should execute the canonical-universe cached-source batch with ME-SR02 snapshots and the approved local portfolio context file, then inspect whether downstream stages progress beyond the ME-RUN17 missing-portfolio-context blocker.
