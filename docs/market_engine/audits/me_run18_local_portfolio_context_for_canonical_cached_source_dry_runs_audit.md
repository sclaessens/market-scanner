# ME-RUN18 Audit - Local Portfolio Context for Canonical Cached-Source Dry-Runs

Owner roles: Governance Auditor / Technical Architect / QA Lead

Status: IMPLEMENTATION AUDIT CREATED BY ME-RUN18

## Audit target

ME-RUN18 audits the cached-source batch dry-run command changes that allow an operator-supplied local portfolio context file to be passed into canonical-universe cached-source dry-runs.

Changed files:

```text
src/market_engine/run/cached_source_batch_dry_run_command.py
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
docs/market_engine/run/me_run18_local_portfolio_context_for_canonical_cached_source_dry_runs.md
```

## Scope verification

The implementation remains inside the ME-RUN command layer and uses the existing runtime parameter:

```text
portfolio_contexts_by_ticker
```

No new provider, broker, source refresh, portfolio persistence, watchlist persistence, Telegram, scheduler, UI, or Decision Engine action authority was introduced.

## Contract verification

The command accepts a local wrapper contract:

```text
market-engine-local-portfolio-context-batch-v1
```

It expands the wrapper to per-ticker payloads using the already-approved downstream context contract:

```text
market-engine-portfolio-context-v1
```

The command requires:

```text
non_production_local_context=true
portfolio_write_authority=false
```

A file that does not meet those constraints fails closed before runtime execution.

## Side-effect audit

The implementation performs read-only JSON loading. It does not:

* fetch provider data;
* refresh source snapshots;
* read broker data;
* write portfolio state;
* write watchlist state;
* send Telegram/email messages;
* create production reports;
* schedule anything;
* expose UI behavior;
* generate orders;
* create target prices;
* create position sizing;
* create ranking/scoring/conviction/urgency/tradeability fields.

Artifact writing remains controlled only by the existing explicit `--write-local-artifacts` flag.

## Operator visibility audit

Human-visible command output now includes a `PORTFOLIO CONTEXT` section with:

* enabled status;
* context path;
* local batch contract;
* downstream portfolio context contract;
* snapshot timestamp;
* context ticker count;
* default position state.

The generated command string includes `--portfolio-context` when used.

## Test audit

Added coverage confirms:

* default behavior keeps `portfolio_contexts_by_ticker={}`;
* `--portfolio-context` is parsed;
* the local wrapper is expanded into per-ticker contexts;
* held-position overrides are passed through;
* absent tickers use `default_position_state`;
* metadata exposes portfolio-write authority as false;
* human-visible output contains the new `PORTFOLIO CONTEXT` section;
* next-review actions mention portfolio context verification.

## Validation limitation

The execution environment could not access the user's local checkout at `/Users/sclaessens/Documents/market-scanner` and could not clone GitHub from the container. The changed Python files were syntax-checked before commit. Full validation remains required locally:

```text
python -m pytest tests/market_engine/run/test_cached_source_batch_dry_run_command.py -q | tee /dev/tty | pbcopy
python -m pytest tests/market_engine -q | tee /dev/tty | pbcopy
```

## Audit conclusion

ME-RUN18 is a bounded RUN command-interface implementation that supplies local non-production portfolio context to canonical cached-source dry-runs without expanding authority, side effects, production execution, or financial-action semantics.
