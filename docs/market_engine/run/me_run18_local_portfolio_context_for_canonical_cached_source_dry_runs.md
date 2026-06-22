# ME-RUN18 - Local Portfolio Context for Canonical Cached-Source Dry-Runs

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: IMPLEMENTED BY ME-RUN18

## Goal

ME-RUN18 provides approved local portfolio context to canonical-universe cached-source dry-runs so downstream Portfolio Review, Decision Engine handoff review, and Delivery Reporting review stages can progress beyond the ME-RUN17 missing-portfolio-context blocker without production portfolio writes.

## Scope

ME-RUN18 is limited to local cached-source RUN behavior:

* read-only operator-supplied local portfolio context JSON;
* command-level `--portfolio-context` support;
* validation of the local non-production portfolio-context batch wrapper;
* conversion of batch-level local context into per-ticker `market-engine-portfolio-context-v1` payloads;
* forwarding those payloads to the existing `portfolio_contexts_by_ticker` runtime boundary;
* human-visible command output for portfolio-context metadata;
* deterministic tests for the command path.

## Implemented command behavior

The cached-source batch command now accepts:

```text
--portfolio-context [PATH]
```

When the flag is supplied without a path, the command uses:

```text
data/market_engine/portfolio_contexts/local_portfolio_context.json
```

The portfolio context file must use:

```text
market-engine-local-portfolio-context-batch-v1
```

The file must explicitly set:

```text
non_production_local_context=true
portfolio_write_authority=false
```

The command reads the file only. It does not write, mutate, refresh, import from broker, import from portfolio CSV automatically, or create production portfolio state.

## Local portfolio-context wrapper shape

The approved local wrapper is intentionally narrow. Example:

```json
{
  "portfolio_context_batch_format_version": "market-engine-local-portfolio-context-batch-v1",
  "non_production_local_context": true,
  "portfolio_write_authority": false,
  "portfolio_snapshot_timestamp": "2026-06-22T11:00:00Z",
  "portfolio_base_currency": "EUR",
  "portfolio_total_value": 32459,
  "default_position_state": "not_held",
  "positions_by_ticker": {
    "AMD": {
      "position_state": "held",
      "current_quantity": 4,
      "current_market_value": 520,
      "current_ticker_exposure_pct": 1.6
    }
  },
  "exposure_buckets": {
    "single_position_max_pct": 10
  },
  "concentration_thresholds": {
    "review_above_pct": 5
  },
  "policy_constraints": {
    "non_actionable_review_only": true
  }
}
```

For every requested ticker, the command creates a `market-engine-portfolio-context-v1` payload. Tickers absent from `positions_by_ticker` use the batch `default_position_state` and zero local quantity/value defaults.

## Operator command example

```text
python -m market_engine.run.cached_source_batch_dry_run_command \
  --source-snapshot-root data/market_engine/source_snapshots \
  --canonical-ticker-universe \
  --portfolio-context data/market_engine/portfolio_contexts/local_portfolio_context.json \
  --write-local-artifacts \
  --artifact-output-root artifacts/market_engine \
  --emit-json | tee /dev/tty | pbcopy
```

## Expected effect

ME-RUN17 proved that ME-SR02 snapshots can be discovered and 12 local dry-run payloads can be executed, while the chain remains downstream-blocked because portfolio context is missing.

ME-RUN18 removes that specific local command-interface gap by passing approved local portfolio context into the existing runtime boundary. Tickers can still block for other downstream contract, missing source, invalid source, stale data, unsupported source, or unexpected local failure reasons.

## Preserved boundaries

ME-RUN18 does not introduce:

* provider refresh;
* live SEC/EDGAR calls;
* yfinance calls;
* live market data calls;
* broker calls;
* portfolio writes;
* watchlist writes;
* Telegram/email delivery;
* production reports;
* scheduler behavior;
* UI behavior;
* Decision Engine action semantics;
* BUY / SELL / HOLD action semantics;
* allocation advice;
* target prices;
* target weights;
* position sizing;
* order generation;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability;
* execution advice.

## Validation

Implemented test coverage in:

```text
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

The added test confirms that:

* the command accepts `--portfolio-context`;
* the local batch wrapper is validated;
* contexts are expanded to per-ticker `market-engine-portfolio-context-v1` payloads;
* provided held positions override the default not-held state;
* portfolio write authority remains false;
* command output metadata includes portfolio context visibility;
* review actions remind the operator to verify portfolio-context metadata.

## Notes

The execution environment used for this sprint did not have the user's local checkout mounted and could not clone from GitHub. Code and test files were syntax-checked locally before committing through the GitHub connector, but the full repository test suite must still be run in the project checkout before merge.
