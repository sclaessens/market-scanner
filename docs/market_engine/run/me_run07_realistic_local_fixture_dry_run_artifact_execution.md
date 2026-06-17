# ME-RUN07 - Realistic local fixture dry-run artifact execution

## Status

COMPLETED BY ME-RUN07

## Sprint

ME-RUN07 - Execute realistic local fixture dry-run and persist review artifact

## Job family

ME-RUN - Run / orchestration jobs

## Purpose

ME-RUN07 demonstrates the Market Engine end-to-end dry-run command path with a realistic non-production local snapshot fixture and explicit local artifact persistence.

The sprint remains a local review workflow only. It does not add provider access, live market data, broker access, delivery channels, production artifacts, portfolio writes, watchlist writes, allocation authority, scoring, ranking, conviction, urgency, tradeability, or BUY / SELL / HOLD semantics.

## Implemented fixture

```text
tests/fixtures/market_engine/run/me_run07_realistic_local_snapshot_fixture.json
```

The fixture uses the existing local input wrapper:

```text
market-engine-local-dry-run-input-fixture-v1
```

The fixture is explicitly marked:

```text
non_production_fixture: true
input_mode: local_snapshot_fixture
```

The fixture contains realistic local snapshot-style payloads for ticker `MSFT` / CIK `0000789019` across the approved Market Engine dry-run stage chain:

```text
source_context
fundamental_observations
derived_observations
setup_detection
analysis_review
recommendation_review
portfolio_review
decision_engine_handoff
delivery_reporting
```

## Existing contracts used

ME-RUN07 does not introduce new runtime contracts.

The fixture uses the existing stage contract identities already accepted by `market-engine-end-to-end-dry-run-v1`:

```text
sec-companyfacts-source-context-v1
sec-companyfacts-fundamental-observations-v1
sec-companyfacts-derived-cash-generation-observations-v1
sec-companyfacts-setup-detection-v1
sec-companyfacts-analysis-review-v1
sec-companyfacts-recommendation-review-v1
sec-companyfacts-portfolio-review-v1
market-engine-decision-engine-handoff-v1
market-engine-delivery-report-v1
```

The persisted review artifact uses the existing RUN05 contract:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

## Exact local dry-run command

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command \
  --input-mode local_snapshot_fixture \
  --stage-payloads-json tests/fixtures/market_engine/run/me_run07_realistic_local_snapshot_fixture.json \
  --dry-run-id me-run07-realistic-local-fixture-artifact \
  --generated-at 2026-06-17T14:15:00Z \
  --write-local-artifact \
  --artifact-output-root artifacts/market_engine/dry_runs \
  --artifact-created-at 2026-06-17T14:30:00Z
```

Clipboard-friendly local command:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode local_snapshot_fixture --stage-payloads-json tests/fixtures/market_engine/run/me_run07_realistic_local_snapshot_fixture.json --dry-run-id me-run07-realistic-local-fixture-artifact --generated-at 2026-06-17T14:15:00Z --write-local-artifact --artifact-output-root artifacts/market_engine/dry_runs --artifact-created-at 2026-06-17T14:30:00Z | tee /dev/tty | pbcopy
```

## Expected stdout

The command emits the complete `market-engine-end-to-end-dry-run-v1` payload to stdout.

Expected identity and state fields:

```text
dry_run_format_version: market-engine-end-to-end-dry-run-v1
dry_run_id: me-run07-realistic-local-fixture-artifact
input_mode: local_snapshot_fixture
ticker: MSFT
cik: 0000789019
run_state: dry_run_blocked
blocked_stage: delivery_reporting
```

The dry-run intentionally blocks at `delivery_reporting` with:

```text
ME-RUN07 fixture intentionally blocks before any delivery channel.
```

This proves blocked-state preservation without performing Telegram, email, report delivery, broker, portfolio, watchlist, scheduler, UI, or production-write side effects.

## Expected local artifact paths

The command writes local artifacts only because `--write-local-artifact` is explicitly present.

Expected manifest:

```text
artifacts/market_engine/dry_runs/me-run07-realistic-local-fixture-artifact/manifest.json
```

Expected artifact:

```text
artifacts/market_engine/dry_runs/me-run07-realistic-local-fixture-artifact/artifacts/market_engine_dry_run_me-run07-realistic-local-fixture-artifact_2026-06-17.json
```

Generated local artifacts remain non-production review evidence and must not be committed.

## Local validation evidence

The command was validated from a local checkout on the RUN07 branch with `PYTHONPATH=src` so the `src/market_engine` package is importable for direct `python -m` execution.

Observed artifact files:

```text
artifacts/market_engine/dry_runs/me-run07-realistic-local-fixture-artifact/artifacts/market_engine_dry_run_me-run07-realistic-local-fixture-artifact_2026-06-17.json
artifacts/market_engine/dry_runs/me-run07-realistic-local-fixture-artifact/manifest.json
```

Observed manifest fields:

```text
manifest_format_version: market-engine-local-dry-run-artifact-manifest-v1
artifact_count: 1
artifact_persistence_mode: local_dry_run_only
non_production_artifact: true
source_input_mode: local_snapshot_fixture
source_run_state: dry_run_blocked
```

## Evidence intentionally preserved

ME-RUN07 fixture evidence includes:

* accepted local fixture wrapper metadata;
* realistic local snapshot stage payloads;
* numeric-zero evidence in portfolio context: `current_quantity`, `current_market_value`, and `cash_available_for_review`;
* missing-data markers from Fundamental Observations and Setup Detection;
* stale-data markers from Source Context, Analysis Review, and Delivery / Reporting;
* delivery blocked reason;
* stage provenance references and run IDs;
* dry-run side-effect confirmation;
* dry-run authority-boundary confirmation.

## Exact pytest command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest \
  tests/market_engine/run/test_local_dry_run_inputs.py \
  tests/market_engine/run/test_end_to_end_dry_run_command.py \
  tests/market_engine/run/test_me_run07_realistic_local_snapshot_fixture_dry_run.py \
  -q
```

Clipboard-friendly test command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/run/test_local_dry_run_inputs.py tests/market_engine/run/test_end_to_end_dry_run_command.py tests/market_engine/run/test_me_run07_realistic_local_snapshot_fixture_dry_run.py -q | tee /dev/tty | pbcopy
```

Observed local pytest result:

```text
18 passed in 0.04s
```

## Side-effect boundaries

ME-RUN07 does not introduce or invoke:

* provider calls;
* SEC/EDGAR calls;
* yfinance or live market data calls;
* broker calls;
* Telegram delivery;
* email delivery;
* production report generation;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* new financial analysis logic;
* Decision Engine action semantics;
* BUY / SELL / HOLD semantics;
* allocation advice;
* target weights;
* target prices;
* position sizing;
* order generation;
* execution advice;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability.

## Execution-environment note

The implementation is connector-backed in this ChatGPT session. Local validation was performed from the user's local checkout after fetching the RUN07 branch. Direct `python -m market_engine...` execution requires `PYTHONPATH=src` in this repository layout unless the package is otherwise installed into the active virtual environment.
