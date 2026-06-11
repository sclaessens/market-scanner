# BL100 — Decouple ops and diagnostics tests from script-era modules

Status: COMPLETED

## Purpose

BL100 decouples active ops and diagnostics tests from script-era modules.

Targeted script-era modules:

```text
scripts/ops/capture_historical_evidence.py
archive/legacy_runtime/scripts/diagnostics/audit_data_coverage.py
```

Targeted active tests:

```text
tests/ops/test_capture_historical_evidence.py
tests/diagnostics/test_audit_data_coverage.py
```

BL100 is a decoupling sprint. It does not archive, edit, execute, refactor, or delete script-era runtime modules.

## Pre-decoupling finding

Before BL100, the focused test run failed during collection because the tests imported script-era modules directly:

```text
tests/ops/test_capture_historical_evidence.py:
from scripts.ops import capture_historical_evidence as capture

tests/diagnostics/test_audit_data_coverage.py:
from scripts.diagnostics import audit_data_coverage as audit_module
```

Focused failure:

```text
ModuleNotFoundError: No module named 'scripts'
```

The full suite still passed because these tests were listed as migration blockers in the active blocker registries.

## What changed

BL100 replaced script-era runtime tests with static/canonical contract tests.

Updated files:

```text
tests/ops/test_capture_historical_evidence.py
tests/diagnostics/test_audit_data_coverage.py
tests/conftest.py
tests/test_operator_visibility.py
```

## Ops capture test decoupling

`tests/ops/test_capture_historical_evidence.py` no longer imports:

```text
scripts.ops.capture_historical_evidence
```

The test now validates the historical evidence capture contract as static policy.

Covered artifacts:

```text
pipeline_runs.csv
pipeline_artifacts.csv
decision_reporting_observations.csv
```

Covered evidence fields include:

```text
run_id
captured_at
decision_reporting_linkage_status
decision_reporting_observation_count
artifact_path
artifact_exists
row_count
file_size_bytes
content_hash
diagnostic_notes
decision_artifact_path
reporting_artifact_path
decision_input_row_hash
reporting_source_row_identity
reporting_represented_flag
diagnostic_note
```

The test also verifies that the contract remains evidence-only and does not gain portfolio, watchlist, Telegram, trade, allocation, execution, or recommendation authority.

## Diagnostics test decoupling

`tests/diagnostics/test_audit_data_coverage.py` no longer imports:

```text
scripts.diagnostics.audit_data_coverage
```

The test now validates diagnostics coverage as a static contract.

Covered sections:

```text
target_universe
portfolio_metadata
fundamentals
```

Covered metadata and fundamentals fields include:

```text
target_mode
target_total_tickers
target_total_ticker_date_rows
explicit_target_date_source
metadata_complete_count
metadata_partial_count
metadata_missing_count
metadata_invalid_count
metadata_coverage_percentage
metadata_freshness_distribution
fundamentals_sufficient_count
fundamentals_partial_count
fundamentals_missing_count
fundamentals_invalid_count
fundamentals_coverage_percentage
ticker_date_match_success_count
ticker_date_match_failure_count
date_mismatch_count
diagnostics
```

The test also verifies that diagnostics coverage does not gain ranking, scoring, tradeability, allocation, urgency, conviction, buy, or sell authority.

## Blocker registry update

BL100 removed the following tests from the high-risk script-era blocker registries:

```text
diagnostics/test_audit_data_coverage.py
ops/test_capture_historical_evidence.py
```

Updated blocker registry files:

```text
tests/conftest.py
tests/test_operator_visibility.py
```

## Active import check

BL100 checked for real active imports:

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.ops|import scripts\.ops|from scripts\.diagnostics|import scripts\.diagnostics)" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result:

```text
No output
```

Interpretation:

* no active positive imports remain from `scripts.ops`;
* no active positive imports remain from `scripts.diagnostics`;
* remaining textual references inside the new tests are negative guardrail assertions only.

## Validation

Focused suite:

```bash
pytest tests/ops/test_capture_historical_evidence.py \
       tests/diagnostics/test_audit_data_coverage.py \
       tests/test_operator_visibility.py -q
```

Result:

```text
18 passed in 0.06s
```

Full suite:

```bash
pytest -q
```

Result:

```text
581 passed in 0.62s
```

## Decision

BL100 decision:

```text
OPS_AND_DIAGNOSTICS_ACTIVE_TEST_DEPENDENCIES_DECOUPLED
```

The targeted tests no longer import script-era ops or diagnostics modules.

## Remaining archive-readiness note

The active ops file still physically exists:

```text
scripts/ops/capture_historical_evidence.py
```

The diagnostics file referenced by the old test is already archived:

```text
archive/legacy_runtime/scripts/diagnostics/audit_data_coverage.py
```

BL100 does not archive `scripts/ops/capture_historical_evidence.py`. It only removes active test dependency blockers.

## Recommended next sprint

Recommended next sprint:

```text
BL101 — Archive ops capture script after final no-active-reference check
```

Candidate archive target:

```text
scripts/ops/capture_historical_evidence.py
```

BL101 must first confirm:

* no active imports from `scripts.ops`;
* no active positive references to `scripts/ops/capture_historical_evidence.py` in `src`, `tests`, or `.github`;
* focused and full tests pass;
* no script-era ops module is executed.

## Guardrails

* No live provider calls were run.
* No yfinance calls were run.
* No SEC/EDGAR calls were run.
* No credentials were read.
* No production data was written.
* No production reports were generated.
* No Telegram messages were sent.
* No portfolio/watchlist state was modified.
* No Decision Engine authority was changed.
* No script-era runtime module was archived.
* No script-era runtime module was edited.
* No script-era runtime module was executed directly.
