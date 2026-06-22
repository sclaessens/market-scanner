# ME-RUN17 - Canonical-universe cached-source batch dry-run with ME-SR02 snapshots

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED WITH DOWNSTREAM BLOCKED OUTCOME BY ME-RUN17

## Purpose

ME-RUN17 executes the canonical-universe cached-source batch dry-run after ME-SR02 added bounded SEC CompanyFacts cached source snapshots.

This sprint stays inside the cached-source local RUN boundary. It does not refresh providers, call live SEC/EDGAR endpoints, call yfinance, send Telegram messages, mutate portfolio or watchlist state, generate production reports, or introduce action authority.

## Initial blocked outcome

The first ME-RUN17 execution selected the canonical universe correctly but discovered zero cached-source tickers.

Observed initial result:

```text
discovered_cached_source_count=0
requested_count=13
blocked_count=13
missing_cached_source_count=13
```

Diagnosis:

```text
RUN discovery scanned only the older run/raw/*.json layout.
ME-SR02 stores snapshots under sec_companyfacts/<snapshot_id>/raw/*.json.
```

## Fix implemented

`src/market_engine/run/cached_source_batch_execution.py` now discovers both supported local snapshot layouts:

```text
<source_snapshot_root>/<run_id>/raw/<TICKER>_companyfacts.json
<source_snapshot_root>/sec_companyfacts/<snapshot_id>/raw/<TICKER>_companyfacts.json
```

The fix preserves existing supported layouts and does not introduce provider calls or source refresh behavior.

## Canonical universe input

Canonical ticker universe:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

Canonical rows loaded:

```text
14
```

Selected active `cached_source_only` tickers:

```text
NVDA
AMD
ASML
META
MSFT
VRT
CLS
CRDO
IREN
COST
HO
AVGO
TSM
```

Excluded ticker:

```text
SMCI
```

Reason:

```text
source_policy=manual_review_only
```

## ME-SR02 source snapshot input

Source snapshot root:

```text
data/market_engine/source_snapshots
```

ME-SR02 raw snapshot layout:

```text
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/
```

Discovered cached-source tickers:

```text
AMD
ASML
AVGO
CLS
COST
CRDO
IREN
META
MSFT
NVDA
TSM
VRT
```

HO has no raw CompanyFacts JSON snapshot in the ME-SR02 snapshot set and remains blocked as missing cached source.

## Final command

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.cached_source_batch_dry_run_command \
  --canonical-ticker-universe \
  --source-snapshot-root data/market_engine/source_snapshots \
  --batch-id me-run17-20260622T090414Z \
  --generated-at 2026-06-22T09:04:14Z \
  --write-local-artifacts \
  --artifact-output-root artifacts/market_engine \
  --emit-json
```

## Final result

Batch contract:

```text
market-engine-cached-source-batch-dry-run-v1
```

Batch state:

```text
completed_with_ticker_failures
```

Final counts:

```text
requested_count=13
discovered_cached_source_count=12
executed_count=12
completed_count=0
completed_with_limitations_count=0
blocked_count=13
missing_cached_source_count=1
failed_count=0
skipped_count=0
```

Per-ticker final state:

```text
NVDA blocked_downstream_contract_failure
AMD blocked_downstream_contract_failure
ASML blocked_downstream_contract_failure
META blocked_downstream_contract_failure
MSFT blocked_downstream_contract_failure
VRT blocked_downstream_contract_failure
CLS blocked_downstream_contract_failure
CRDO blocked_downstream_contract_failure
IREN blocked_downstream_contract_failure
COST blocked_downstream_contract_failure
HO blocked_missing_cached_source
AVGO blocked_downstream_contract_failure
TSM blocked_downstream_contract_failure
```

The 12 discovered snapshots executed into local end-to-end dry-run payloads and wrote per-ticker artifacts. Those dry-runs remain blocked downstream because current review-chain stages preserve upstream blocked states and missing data. HO remains blocked before dry-run execution because its raw snapshot is absent.

## Artifact output

Generated local artifacts:

```text
artifacts/market_engine/me-run17-20260622T090414Z/batch_manifest.json
artifacts/market_engine/me-run17-20260622T090414Z/<TICKER>/dry_run.json
artifacts/market_engine/me-run17-20260622T090414Z/<TICKER>/manifest.json
```

Per-ticker artifacts were written for the 12 discovered snapshot tickers only.

Generated artifacts are local non-production execution evidence and are not committed by default.

## Safety boundary confirmation

ME-RUN17 did not introduce or perform:

* provider calls;
* live SEC/EDGAR calls;
* yfinance calls;
* live market data calls;
* broker calls;
* Telegram delivery;
* email delivery;
* production report generation;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* automatic cache refresh;
* automatic cleanup;
* Decision Engine decisions;
* BUY / SELL / HOLD semantics;
* allocation advice;
* target prices;
* position sizing;
* order generation;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability;
* execution advice.

## Next sprint recommendation

Recommended next sprint:

```text
ME-RUN18 - Provide portfolio context for canonical-universe cached-source dry-runs
```

Rationale: ME-RUN17 now discovers ME-SR02 snapshots and executes 12 ticker dry-runs, but the downstream chain blocks at review stages that require additional context. A narrow RUN sprint should define or supply local portfolio context safely before another canonical-universe execution review.
