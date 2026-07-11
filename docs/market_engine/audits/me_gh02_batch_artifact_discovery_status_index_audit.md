# ME-GH02 - Batch Artifact Discovery and Ticker Status Index Audit

## Objective

Implement deterministic batch artifact discovery and ticker status indexing for the GitHub-first no-API baseline.

## Baseline guardrail

- OpenAI API required: no.
- Provider invocation: no.
- API key required: no.
- Source acquisition: no.
- Ranking: no.
- Recommendation semantics: no.
- Broker/order/portfolio/watchlist mutation: no.

## Implementation

- Modules added:
  - `src/market_engine/batch_status/__init__.py`
  - `src/market_engine/batch_status/artifact_discovery.py`
  - `src/market_engine/batch_status/status_index.py`
  - `src/market_engine/batch_status/status_index_command.py`
- CLI command: `python -m market_engine.batch_status.status_index_command`.
- Supported artifact type: `market_engine_end_to_end_dry_run`.
- Supported artifact format: `market-engine-local-dry-run-artifact-v1`.
- Output root: `artifacts/market_engine/batch_status`.
- Output files:
  - `manifest.json`
  - `ticker_status_index.json`
  - `ticker_status_index.md`
  - `discovery_summary.json`
  - `failures.json`

## Discovery rule

A supported artifact is a `dry_run.json` file with:

```text
artifact_format_version = market-engine-local-dry-run-artifact-v1
artifact_type = market_engine_end_to_end_dry_run
payload.ticker present
```

Invalid JSON and unsupported dry-run files are recorded without crashing. The canonical artifact per ticker is selected deterministically:

1. valid artifacts before invalid artifacts;
2. newest `artifact_created_at`;
3. newest file modified time;
4. lexicographically smallest path.

## Status rule

Status values:

- `invalid_artifact`
- `blocked`
- `stale`
- `review_ready`
- `descriptive_only`

Precedence:

1. `invalid_artifact`: no valid canonical artifact.
2. `blocked`: `blocked_stage` is present or `blocked_reasons` is non-empty.
3. `stale`: `analysis_context_readiness.context_stale` is true.
4. `review_ready`: `actionable_review_allowed` or `decision_engine_ready` is true.
5. `descriptive_only`: fallback for valid artifacts without actionability.

This status index does not create BUY, SELL, HOLD, allocation, sizing, conviction, recommendation, or investment ranking semantics.

## Sample run

- Artifact root: `artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z`.
- Run ID: `me-gh02-sample-status-index-20260711T120000Z`.
- Output directory: `artifacts/market_engine/batch_status/me-gh02-sample-status-index-20260711T120000Z`.
- Tickers total: 12.
- Valid artifacts: 12.
- Invalid artifacts: 0.
- Status counts: `blocked=12`.
- Readiness counts: `partial_analysis=12`.
- Stale count: 0.
- Actionable review allowed count: 0.
- Decision engine ready count: 0.

## Test evidence

- py_compile: passed.
- Targeted tests: `tests/market_engine/batch_status -q` passed, 8 tests.
- Advisory no-api regression: `tests/market_engine/advisory/test_grounded_advisory_no_api_baseline.py -q` passed, 1 test.
- Diff check: passed.

## Governance review

- No OpenAI API call: confirmed.
- No provider invocation: confirmed.
- No source acquisition: confirmed.
- No ranking: confirmed.
- No recommendation semantics: confirmed.
- No side effects: confirmed.

## Outcome

`completed_batch_status_index`

## Next sprint

`ME-GH03 - Deterministic ranking and review queue`
