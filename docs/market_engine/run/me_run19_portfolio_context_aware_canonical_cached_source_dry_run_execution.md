# ME-RUN19 - Portfolio-Context-Aware Canonical Cached-Source Dry-Run Execution

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN19

## Purpose

ME-RUN19 executes the existing ME-RUN18 portfolio-context-aware cached-source batch command against the canonical ticker universe and ME-SR02 cached SEC CompanyFacts snapshots.

The sprint is run-first. It validates whether the existing runtime can progress beyond the ME-RUN17 missing-portfolio-context blocker without adding runtime behavior, provider calls, production writes, delivery channels, or action authority.

## Input Sources

Canonical ticker universe:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

Cached SEC CompanyFacts snapshots:

```text
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/
```

Local non-production portfolio context:

```text
data/market_engine/portfolio_contexts/local_portfolio_context.json
```

The portfolio context uses:

```text
market-engine-local-portfolio-context-batch-v1
```

It explicitly sets:

```text
non_production_local_context=true
portfolio_write_authority=false
```

## Command Used

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.cached_source_batch_dry_run_command \
  --canonical-ticker-universe \
  --source-snapshot-root data/market_engine/source_snapshots \
  --portfolio-context data/market_engine/portfolio_contexts/local_portfolio_context.json \
  --batch-id me-run19-20260622T103000Z \
  --generated-at 2026-06-22T10:30:00Z \
  --write-local-artifacts \
  --artifact-output-root artifacts/market_engine \
  --emit-json
```

## Selected And Excluded Tickers

Canonical rows loaded: 14.

Selected active `cached_source_only` tickers:

```text
NVDA, AMD, ASML, META, MSFT, VRT, CLS, CRDO, IREN, COST, HO, AVGO, TSM
```

Excluded ticker:

```text
SMCI
```

Reason: `source_policy=manual_review_only`.

## Run Result

Batch contract:

```text
market-engine-cached-source-batch-dry-run-v1
```

Per-ticker output contract:

```text
market-engine-end-to-end-dry-run-v1
```

Batch state:

```text
completed_with_ticker_failures
```

Counts:

```text
requested_count=13
discovered_cached_source_count=12
executed_count=12
completed_count=10
completed_with_limitations_count=0
blocked_count=3
failed_count=0
skipped_count=0
missing_cached_source_count=1
stale_source_count=0
ambiguous_cached_source_count=0
unsupported_cached_source_count=0
```

Completed tickers:

```text
NVDA, AMD, META, MSFT, VRT, CLS, CRDO, IREN, COST, AVGO
```

Blocked tickers:

```text
ASML, HO, TSM
```

## Blocked Ticker Reasons

### ASML

State:

```text
blocked_downstream_contract_failure
```

Reason:

```text
Stage preserves an upstream blocked state.
```

Observed blocked stage:

```text
recommendation_review
```

Observed missing source fields:

```text
revenue, net_income, operating_cash_flow, capital_expenditures
```

### HO

State:

```text
blocked_missing_cached_source
```

Reason:

```text
No matching cached source snapshot was found.
```

HO was selected by the canonical universe but has no ME-SR02 raw SEC CompanyFacts snapshot in the local source snapshot directory.

### TSM

State:

```text
blocked_downstream_contract_failure
```

Reason:

```text
Stage preserves an upstream blocked state.
```

Observed blocked stage:

```text
recommendation_review
```

Observed missing source fields:

```text
revenue, net_income, operating_cash_flow, capital_expenditures
```

## Downstream Output Availability

For 10 completed tickers, the dry-run reached:

* Source Context;
* Fundamental Observations;
* Derived Observations;
* Setup Detection;
* Analysis Review;
* Recommendation Review;
* Portfolio Review;
* Decision Engine handoff;
* Delivery / Reporting;
* dry-run summary.

For ASML and TSM, Source Context completed with limitations, Analysis Review completed, and Recommendation Review blocked. Portfolio Review, Decision Engine handoff, and Delivery / Reporting were not started for those tickers.

For HO, no per-ticker dry-run payload was created because no cached source snapshot was available.

## Artifact Paths

Generated artifact root:

```text
artifacts/market_engine/me-run19-20260622T103000Z/
```

Batch manifest:

```text
artifacts/market_engine/me-run19-20260622T103000Z/batch_manifest.json
```

Per-ticker artifacts were written for the 12 tickers with cached snapshots:

```text
artifacts/market_engine/me-run19-20260622T103000Z/<TICKER>/dry_run.json
artifacts/market_engine/me-run19-20260622T103000Z/<TICKER>/manifest.json
```

Generated artifacts are local run evidence and are intentionally not committed.

## Boundary Confirmation

ME-RUN19 did not introduce:

* provider calls;
* SEC or EDGAR live calls;
* yfinance calls;
* live market data calls;
* broker calls;
* portfolio writes;
* watchlist writes;
* Telegram or email delivery;
* production reports;
* scheduler behavior;
* UI behavior;
* Decision Engine action semantics;
* BUY / SELL / HOLD semantics;
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

## Conclusion

ME-RUN19 succeeded as a run-first sprint without code changes.

The existing ME-RUN18 command support is sufficient to run the canonical-universe cached-source batch with local portfolio context. The portfolio-context blocker from ME-RUN17 is resolved for tickers with complete cached source evidence.

The remaining blockers are source-coverage issues:

* HO has no cached source snapshot.
* ASML and TSM preserve upstream missing-field evidence and block at Recommendation Review.

Recommended next sprint:

```text
ME-SR03 - Resolve canonical-universe cached-source coverage blockers
```
