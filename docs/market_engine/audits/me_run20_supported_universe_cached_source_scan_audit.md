# ME-RUN20 Audit - Supported-universe cached-source scan

Sprint: ME-RUN20 - Execute clean supported-universe cached-source scan

Branch: `me-run20-execute-clean-supported-universe-cached-source-scan`

Status: Completed

## Goal

Execute a clean, local, cached-source Market Engine scan for the supported Professional Swing Universe subset using existing local execution and artifact-writing behavior.

## Execution Summary

ME-SR05 classified 12 Professional Swing Universe rows as `supported_cached`.

ME-RUN20 executed the existing cached-source batch dry-run command against exactly those 12 supported tickers:

```text
NVDA, AMD, ASML, META, MSFT, VRT, CLS, CRDO, IREN, COST, AVGO, TSM
```

The current runtime command does not have a dedicated `--supported-universe` flag. The run used the existing explicit `--tickers` path with the ME-SR05-supported subset. No runtime code change was required.

## Exact Command

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.cached_source_batch_dry_run_command \
  --source-snapshot-root data/market_engine/source_snapshots \
  --tickers NVDA,AMD,ASML,META,MSFT,VRT,CLS,CRDO,IREN,COST,AVGO,TSM \
  --portfolio-context data/market_engine/portfolio_contexts/local_portfolio_context.json \
  --batch-id me-run20-supported-universe-20260623T120000Z \
  --generated-at 2026-06-23T12:00:00Z \
  --write-local-artifacts \
  --artifact-output-root artifacts/market_engine \
  --emit-json
```

## Input Sources

* Professional Swing Universe CSV: `data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv`
* Source snapshot root: `data/market_engine/source_snapshots`
* Portfolio context: `data/market_engine/portfolio_contexts/local_portfolio_context.json`

## Output Artifacts

Local artifact root:

```text
artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/
```

Batch manifest:

```text
artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/batch_manifest.json
```

Per-ticker artifacts:

```text
artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/<TICKER>/dry_run.json
artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/<TICKER>/manifest.json
```

Artifacts were generated locally for:

```text
AMD, ASML, AVGO, CLS, COST, CRDO, IREN, META, MSFT, NVDA, TSM, VRT
```

Generated artifacts are local execution evidence and were not committed by default.

## Batch Counts

```text
requested: 12
discovered_cached_source: 12
eligible: 12
executed: 12
completed: 12
completed_with_limitations: 0
blocked: 0
failed: 0
skipped: 0
missing_cached_source: 0
ambiguous_cached_source: 0
unsupported_cached_source: 0
stale_source: 0
```

Supported universe completion result: clean completion.

## Artifact Verification

The generated batch manifest was inspected and reported:

```text
contract_version: market-engine-cached-source-batch-dry-run-v1
batch_execution_state: completed
```

Sample ticker artifacts were inspected:

```text
NVDA: market-engine-local-dry-run-artifact-v1 / market-engine-end-to-end-dry-run-v1 / dry_run_completed
TSM: market-engine-local-dry-run-artifact-v1 / market-engine-end-to-end-dry-run-v1 / dry_run_completed
```

## Tests Run

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run tests/market_engine/source_support -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
git diff --check
```

Results are recorded in the final sprint report.

## Boundaries Preserved

ME-RUN20 did not introduce:

* provider calls;
* SEC or EDGAR live calls;
* yfinance calls;
* source refresh;
* synthetic source data;
* production data writes;
* Telegram/email delivery;
* production reports;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* Decision Engine behavior;
* BUY / SELL / HOLD semantics;
* allocation advice;
* target prices;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability;
* position sizing;
* order generation;
* execution advice.

## Blockers

No ME-RUN20 execution blocker was observed.

Known limitation: the current command does not expose a dedicated `--supported-universe` flag. ME-RUN20 used explicit ticker input derived from ME-SR05 classification. This did not block the run.

## Next Sprint

ME-OUT01 - Define readable operator report from dry-run artifacts.
