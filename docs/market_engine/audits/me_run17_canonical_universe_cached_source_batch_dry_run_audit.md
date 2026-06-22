# ME-RUN17 - Canonical-universe cached-source batch dry-run audit

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED WITH DOWNSTREAM BLOCKED OUTCOME BY ME-RUN17

## Sprint goal

Execute and fix the canonical-universe cached-source batch dry-run using ME-SR02 SEC CompanyFacts snapshots.

## Files changed

```text
src/market_engine/run/cached_source_batch_execution.py
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
docs/market_engine/run/me_run17_canonical_universe_cached_source_batch_dry_run_with_me_sr02_snapshots.md
docs/market_engine/audits/me_run17_canonical_universe_cached_source_batch_dry_run_audit.md
docs/market_engine/backlog/me_run17_canonical_universe_cached_source_batch_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run17_canonical_universe_cached_source_batch_dry_run_roadmap_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

Generated local artifacts, not committed:

```text
artifacts/market_engine/me-run17-20260622T090414Z/
```

## Initial failure

Initial RUN17 output discovered zero cached-source tickers and blocked all selected canonical tickers as `blocked_missing_cached_source`.

Root cause:

```text
ME-RUN discovery did not scan the ME-SR02 source-refresh layout:
data/market_engine/source_snapshots/sec_companyfacts/<snapshot_id>/raw/*.json
```

## Fix

The batch discovery function now scans both:

```text
*/raw/*.json
sec_companyfacts/*/raw/*.json
```

This preserves existing fixture layouts and adds support for the approved ME-SR02 snapshot structure.

## Tests added or updated

Updated:

```text
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

Coverage added:

* canonical universe selection with SMCI excluded as `manual_review_only`;
* ME-SR02-style snapshot discovery under `source_snapshots/sec_companyfacts/<snapshot_id>/raw/`;
* 12 available snapshots executing into dry-run payloads;
* HO remaining safely blocked as missing cached source;
* opt-in artifact writing under a temporary root;
* no live provider or forbidden side effects;
* numeric-zero evidence preservation;
* cached-source provenance preservation.

## Final execution command

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

## Final execution result

```text
batch_execution_state=completed_with_ticker_failures
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

HO result:

```text
blocked_missing_cached_source
```

Reason:

```text
No matching cached source snapshot was found.
```

The other 12 selected tickers discovered ME-SR02 raw snapshots and executed into local end-to-end dry-run payloads, then preserved downstream blocked states.

## Artifact behavior

Artifact writing was explicitly enabled.

The command wrote:

* one batch manifest;
* 12 per-ticker dry-run artifacts;
* 12 per-ticker manifests.

Generated artifacts were left uncommitted.

## Validation

Commands run:

```text
.venv/bin/python -m pytest tests/market_engine/run/test_cached_source_batch_dry_run_command.py tests/market_engine/run/test_me_run13_cached_source_batch_dry_run.py -q
```

Initial targeted result after fix:

```text
14 passed
```

Full Market Engine validation is recorded in the final sprint report.

## Boundaries preserved

ME-RUN17 did not introduce provider calls, live network calls, SEC/EDGAR calls, yfinance calls, Telegram behavior, email delivery, broker calls, production writes, source refresh jobs, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cleanup, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next recommended sprint

```text
ME-RUN18 - Provide portfolio context for canonical-universe cached-source dry-runs
```

Rationale: snapshot discovery is fixed and 12 local dry-run payloads are generated, but downstream review stages remain blocked without additional approved local portfolio context.
