# ME09 Bounded SEC CompanyFacts Coverage Artifact Review Audit

Owner role: Governance Auditor / Data Steward / Technical Architect

Status: COMPLETED BY ME09

## Purpose

ME09 runs the first bounded multi-ticker SEC CompanyFacts source coverage review and creates isolated non-production Market Engine smoke artifacts.

ME09 remains a source/data-layer sprint. It does not authorize analysis, fundamental scoring, scanner ranking, reporting, Telegram delivery, portfolio mutation, watchlist mutation, or Decision Engine behavior.

## Files Created

* `src/market_engine/source_intake/smoke_artifacts.py`
* `tests/market_engine/source_intake/test_smoke_artifacts.py`
* `docs/market_engine/audits/me09_bounded_sec_companyfacts_coverage_artifact_review_audit.md`

## Files Updated

* `src/market_engine/source_intake/manual_smoke.py`
* `src/market_engine/source_intake/sec_companyfacts_provider.py`
* `docs/market_engine/architecture/source_intake_smoke.md`
* `docs/market_engine/backlog/market_engine_backlog.md`

## ME08 Decision Being Extended

ME08 source-data owner decision:

```text
APPROVED_FOR_BOUNDED_SEC_COVERAGE_REVIEW
```

ME09 extends that decision by running a bounded 10-ticker SEC CompanyFacts coverage review and writing isolated smoke artifacts.

## Bounded Ticker List Used

```text
NVDA AMD META COST AAPL MSFT GOOGL AMZN TSLA AVGO
```

## Max Ticker Limit

```text
10
```

ME09 did not run more than 10 real-provider tickers.

## Required Fields

* `revenue`
* `net_income`
* `operating_cash_flow`
* `capital_expenditures`

ME09 does not derive free cash flow or any other financial metric.

## Manual Smoke Commands Used

Fake provider smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.source_intake.manual_smoke
```

Bounded SEC smoke without artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.source_intake.manual_smoke --provider sec-companyfacts --tickers NVDA AMD META COST AAPL MSFT GOOGL AMZN TSLA AVGO --max-tickers 10
```

Bounded SEC smoke with artifact:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.source_intake.manual_smoke --provider sec-companyfacts --tickers NVDA AMD META COST AAPL MSFT GOOGL AMZN TSLA AVGO --max-tickers 10 --write-smoke-artifact
```

## Network Access

Escalated network access was required for the live SEC provider calls.

The sandboxed SEC run returned controlled network errors, while the escalated run produced live coverage evidence.

## Bounded SEC Smoke Result Without Artifact

```text
bounded_real_provider_smoke=true
provider_warning=manual SEC CompanyFacts smoke; source coverage evidence only
provider=SEC_COMPANYFACTS
tickers=10
readiness=AVAILABLE=10
missing_fields=none
provider_errors=0
provider_error_categories=none
unsupported=0
invalid_tickers=0
failed_or_unsupported_tickers=none
note=Source coverage evidence only. Not analysis.
```

## Bounded SEC Smoke Result With Artifact

```text
bounded_real_provider_smoke=true
provider_warning=manual SEC CompanyFacts smoke; source coverage evidence only
provider=SEC_COMPANYFACTS
tickers=10
readiness=AVAILABLE=10
missing_fields=none
provider_errors=0
provider_error_categories=none
unsupported=0
invalid_tickers=0
failed_or_unsupported_tickers=none
note=Source coverage evidence only. Not analysis.
smoke_artifact_path=data/market_engine/smokes/source_intake/sec_companyfacts/20260615T103333Z
```

## Artifact Path

```text
data/market_engine/smokes/source_intake/sec_companyfacts/20260615T103333Z/
```

## Artifact Files

* `coverage_summary.csv`
* `ticker_results.csv`
* `missing_fields.csv`
* `provider_errors.csv`
* `smoke_metadata.json`

Artifact summary:

* `coverage_summary.csv`: 1 data row, `AVAILABLE=10`
* `ticker_results.csv`: 10 ticker rows
* `missing_fields.csv`: header only, no missing fields
* `provider_errors.csv`: header only, no provider errors
* `smoke_metadata.json`: provider, run id, timestamp, ticker list, max ticker limit, required fields, readiness counts, and smoke disclaimer

## Artifact Commit Decision

The generated smoke artifacts were intentionally not committed.

They are non-production smoke evidence and are not source truth. The implementation and audit record the artifact path and summary without adding generated data to version control.

## Readiness Counts

* `AVAILABLE`: 10

## Missing-Field Frequency

No missing fields were reported.

## Provider-Error Categories

No provider errors were reported in the escalated artifact run.

## Ticker Status Summary

Reached `AVAILABLE`:

* `NVDA`
* `AMD`
* `META`
* `COST`
* `AAPL`
* `MSFT`
* `GOOGL`
* `AMZN`
* `TSLA`
* `AVGO`

Reached `PARTIAL`: none

Reached `MISSING`: none

Reached `UNSUPPORTED`: none

Reached `INVALID_TICKER`: none

Reached `PROVIDER_ERROR`: none

## Source-Data Owner Decision After ME09

Decision:

```text
APPROVED_FOR_BOUNDED_SEC_FIELD_MAPPING_CONTRACT
```

Rationale:

* The bounded 10-ticker SEC CompanyFacts sample returned full required-field availability.
* Missing-field and provider-error frequencies were empty for the sample.
* Smoke artifacts were isolated under the Market Engine smoke path.
* The artifact format preserves per-ticker readiness, available fields, missing fields, provider errors, run metadata, and a non-analysis disclaimer.

This decision does not approve all-ticker production runs, analysis, scoring, recommendation behavior, reporting, Telegram delivery, portfolio/watchlist mutation, or Decision Engine behavior.

## Data Isolation Confirmation

ME09 confirms:

* No old data or report paths were written.
* No files were written under `data/processed/`.
* No files were written under `data/generated/`.
* No files were written under `data/logs/`.
* No files were written under `data/normalized/`.
* No files were written under `reports/`.
* No files were written under `data/portfolio/`.
* No files were written under `data/watchlist/`.
* Smoke artifacts were isolated under `data/market_engine/smokes/source_intake/sec_companyfacts/`.

## Implementation Changes Made

ME09 added:

* smoke artifact writer for SEC CompanyFacts source-intake summaries;
* timestamp-based run ids;
* artifact overwrite protection;
* artifact path validation;
* SEC smoke CIK mappings for the 10-ticker bounded sample;
* manual smoke support for `--write-smoke-artifact` without requiring a path argument;
* tests for artifact path, overwrite protection, file columns, metadata, missing numeric preservation, and forbidden-field absence.

## Test Coverage Summary

Targeted test command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/source_intake -q
```

Result:

```text
46 passed
```

Automated tests use fake or mocked provider data only. They do not call live SEC endpoints.

## Boundary Confirmations

ME09 confirms:

* Live provider calls were not used in automated tests.
* Old runtime files were not modified.
* `src/market_scanner/` was not modified.
* `scripts/` was not modified.
* New Market Engine code does not import from `market_scanner`.
* New Market Engine code does not import from `scripts`.
* New tests do not import from `market_scanner`.
* New tests do not import from `scripts`.
* No existing production data, CSV, or report files were modified.
* No generated smoke artifacts were committed.
* No reports were generated.
* No Telegram messages were sent.
* No portfolio data was mutated.
* No watchlist data was mutated.
* No Decision Engine behavior was called or changed.
* No BUY / SELL / HOLD, recommendation, allocation, ranking, score, conviction, urgency, tradeability, position sizing, or execution behavior is emitted by runtime models, smoke artifacts, or coverage review output.

## Known Limitations

* SEC live calls require network permission outside the restricted sandbox.
* The 10-ticker sample is bounded evidence, not all-ticker coverage.
* The ticker-to-CIK map remains in-code and smoke-oriented.
* SEC fact aliases remain limited to the current required fields.
* Artifact files are not committed and must not be treated as source truth.
* ME09 does not build source context, financial interpretation, scoring, or analysis.

## Recommended Next Sprint

Proceed to:

`ME10 - Define approved SEC CompanyFacts field mapping and source coverage contract`

ME10 should convert the successful bounded smoke evidence into a source-field contract, ticker-to-CIK ownership decision, artifact retention policy, and readiness criteria for first source context.
