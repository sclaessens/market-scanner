# ME-RUN15 - Cached-source batch dry-run command visibility implementation audit

Status: IMPLEMENTATION AUDIT

## Scope reviewed

ME-RUN15 adds an operator-facing command for the first real cached-source batch dry-run visibility flow.

Changed files:

* `src/market_engine/run/cached_source_batch_dry_run_command.py`
* `tests/market_engine/run/test_cached_source_batch_dry_run_command.py`
* `docs/market_engine/run/me_run15_cached_source_batch_dry_run_command_visibility.md`
* `pyproject.toml`

## Contract coverage

The implementation preserves:

```text
market-engine-cached-source-batch-dry-run-v1
```

The implementation exposes:

```text
market-engine-real-cached-source-batch-dry-run-visibility-v1
```

## Visibility coverage

The command renders the required human-readable sections:

```text
RUN CONTEXT
INPUT DISCOVERY
SELECTED TICKERS
EXECUTION PROGRESS
BATCH SUMMARY
BLOCKED / FAILED TICKERS
ARTIFACTS
FORBIDDEN SIDE-EFFECT CONFIRMATION
NEXT REVIEW ACTIONS
```

The command can optionally append JSON with `--emit-json`, but human-readable output is always rendered first.

## Input boundary

The command supports exactly one ticker input mode per invocation:

* `--tickers`
* `--ticker-file`
* `--discover-cached-tickers`

The command passes ticker selection to the existing ME-RUN13 cached-source batch runtime.

## Artifact boundary

Artifact writing remains opt-in only through `--write-local-artifacts`.

By default, artifacts are not written.

Generated artifacts are not committed by default.

## Forbidden expansion check

ME-RUN15 does not add source refresh, provider access, production delivery, portfolio mutation, watchlist mutation, scheduling, UI behavior, ranking, scoring, allocation, or action authority.

## Validation target

Relevant validation command:

```bash
python -m pytest tests/market_engine/run -q
```

The branch requires a local validation run after these GitHub connector writes because the connector environment cannot execute the repository test suite directly.
