# ME06 Bounded Real Provider Source Intake Smoke Audit

Owner role: Governance Auditor

Status: COMPLETED BY ME06

## Purpose

This audit records ME06: add a bounded real-provider source-intake smoke path and local source coverage review.

ME06 remains a source/data-layer sprint. It does not authorize analysis, scanner ranking, fundamental analysis, reporting, Telegram delivery, portfolio mutation, watchlist mutation, or Decision Engine behavior.

## Files Created

* `src/market_engine/source_intake/sec_companyfacts_provider.py`
* `src/market_engine/source_intake/coverage_review.py`
* `tests/market_engine/source_intake/test_sec_companyfacts_provider.py`
* `docs/market_engine/audits/me06_bounded_real_provider_source_intake_smoke_audit.md`

## Files Updated

* `src/market_engine/source_intake/manual_smoke.py`
* `docs/market_engine/architecture/source_intake_smoke.md`
* `docs/market_engine/backlog/market_engine_backlog.md`

## Provider Selected

Provider: SEC CompanyFacts

Reason:

* SEC CompanyFacts is relevant to fundamental source coverage.
* It supports a controlled source-readiness smoke without entering analysis.
* It fits the ME03 and ME04 source-boundary direction.

## Implementation Summary

ME06 extends the ME05 source-intake package with:

* a SEC CompanyFacts provider adapter;
* mocked/synthetic provider tests;
* bounded manual real-provider smoke behavior behind explicit flags;
* source coverage review summarization;
* strict ticker limit protection for real-provider manual smoke;
* stdout-only default behavior;
* optional artifact writing only under `data/market_engine/smokes/source_intake/` with no silent overwrite.

## Provider Boundary Summary

The SEC adapter implements the existing Market Engine `SourceProvider` protocol.

Provider access occurs only when `fetch_source()` is explicitly called. No provider access occurs at import time.

The manual smoke defaults to fake provider mode. Real provider mode requires:

* `--provider sec-companyfacts`;
* explicit `--tickers`, `--ticker-file`, or `--use-sec-sample`;
* a bounded `--max-tickers` limit.

## SEC CompanyFacts Adapter Behavior

The adapter:

* exposes provider name `SEC_COMPANYFACTS`;
* supports one ticker per explicit provider call;
* uses a small bounded in-code CIK mapping for smoke examples;
* maps SEC CompanyFacts payloads into the Market Engine source-intake field model;
* preserves missing fields as `None`;
* treats absent required fields as missing;
* raises controlled unsupported ticker, invalid ticker, and provider error paths;
* avoids network calls at import time;
* avoids writes, reports, Telegram, portfolio/watchlist mutation, and Decision Engine calls.

## Required Fields

ME06 uses these required fields:

* `revenue`
* `net_income`
* `operating_cash_flow`
* `capital_expenditures`

## Alias And Mapping Limitations

SEC CompanyFacts concepts are mapped conservatively.

Current aliases:

* `revenue`: `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet`
* `net_income`: `NetIncomeLoss`
* `operating_cash_flow`: `NetCashProvidedByUsedInOperatingActivities`
* `capital_expenditures`: `PaymentsToAcquirePropertyPlantAndEquipment`, `PaymentsToAcquireProductiveAssets`

The adapter selects the latest available USD fact by period end date.

Known limitations:

* The ticker-to-CIK map is intentionally small and smoke-only.
* SEC taxonomy differences may require more aliases later.
* ME06 does not derive free cash flow.
* ME06 does not validate fiscal periods beyond choosing the latest USD fact.
* Smoke evidence is not source truth by default.

## Source-Readiness Behavior

Source readiness is produced by the existing ME05 runner:

* full required-field coverage returns `AVAILABLE`;
* partial coverage returns `PARTIAL`;
* no CompanyFacts payload or no required fields returns `MISSING`;
* unsupported/no-CIK tickers return `UNSUPPORTED`;
* invalid ticker format returns `INVALID_TICKER`;
* network/provider failures return `PROVIDER_ERROR`.

Missing numeric fields remain missing. They are not converted to `0`.

## Coverage Review Behavior

`coverage_review.py` turns a batch summary into a local source coverage review.

The review includes:

* provider name;
* ticker count;
* readiness counts;
* missing-field frequency;
* provider-error count;
* unsupported count;
* invalid ticker count;
* top missing fields;
* failed or unsupported tickers;
* a note that the review is source coverage evidence only and not analysis.

The review does not emit scores, rankings, recommendations, BUY / SELL / HOLD, allocation, conviction, urgency, tradeability, position sizing, or execution advice.

## Test Coverage Summary

Targeted tests cover:

* mocked SEC CompanyFacts full response mapping;
* mocked partial response missingness;
* missing CompanyFacts response;
* unsupported/no-CIK ticker;
* invalid ticker;
* provider/network error capture;
* missing numeric fields staying missing;
* no provider call at import time;
* real provider tests not requiring network;
* coverage review readiness counts;
* coverage review missing-field frequency;
* coverage review absence of forbidden authority fields;
* no imports from legacy runtime modules.

Targeted test command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/source_intake -q
```

Result:

```text
27 passed
```

## Manual Smoke Commands Used

Fake provider smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.source_intake.manual_smoke
```

Result:

```text
provider=fake-source-provider
tickers=7
readiness=AVAILABLE=2, INVALID_TICKER=1, MISSING=1, PARTIAL=1, PROVIDER_ERROR=1, UNSUPPORTED=1
missing_fields=capital_expenditures=5, operating_cash_flow=5, revenue=4
provider_errors=1
unsupported=1
invalid_tickers=1
failed_or_unsupported_tickers=UNSUPPORTED, INVALID, ERROR
note=Source coverage evidence only. Not analysis.
```

Bounded real-provider smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.source_intake.manual_smoke --provider sec-companyfacts --tickers NVDA AMD META COST --max-tickers 4
```

Result:

```text
bounded_real_provider_smoke=true
provider_warning=manual SEC CompanyFacts smoke; source coverage evidence only
provider=SEC_COMPANYFACTS
tickers=4
readiness=PROVIDER_ERROR=4
missing_fields=capital_expenditures=4, net_income=4, operating_cash_flow=4, revenue=4
provider_errors=4
unsupported=0
invalid_tickers=0
failed_or_unsupported_tickers=NVDA, AMD, META, COST
note=Source coverage evidence only. Not analysis.
```

Interpretation:

The bounded real-provider path executed safely and isolated the provider failures per ticker. The environment/provider response did not yield successful SEC CompanyFacts payloads during this run, so ME06 records provider availability as unresolved for follow-up rather than forcing network access.

## Boundary Confirmations

ME06 confirms:

* Live provider calls were not used in automated tests.
* Automated tests use mocked/synthetic provider responses only.
* Old runtime files were not modified.
* `src/market_scanner/` was not modified.
* `scripts/` was not modified.
* New Market Engine source-intake code does not import from `market_scanner`.
* New Market Engine source-intake code does not import from `scripts`.
* New tests do not import from `market_scanner`.
* New tests do not import from `scripts`.
* No existing production data, CSV, or report files were modified.
* No smoke artifact was committed.
* No reports were generated.
* No Telegram messages were sent.
* No portfolio data was mutated.
* No watchlist data was mutated.
* No Decision Engine behavior was called or changed.
* No BUY / SELL / HOLD, recommendation, allocation, ranking, score, conviction, urgency, tradeability, position sizing, or execution behavior is emitted by runtime models or coverage review output.

## Known Limitations

* SEC CompanyFacts live access returned controlled provider errors for the bounded manual smoke in this environment.
* The in-code ticker-to-CIK mapping is intentionally small and smoke-only.
* SEC concept aliases are conservative and may miss provider-specific fact variants.
* ME06 does not persist smoke evidence by default.
* ME06 does not derive free cash flow.
* ME06 does not build fundamental source context or analysis.
* ME06 does not decide generated-output archive policy.

## Recommended Next Sprint

Proceed to:

`ME07 - Review real-provider coverage and define source-data owner decisions`

ME07 should review why the bounded SEC smoke returned provider errors in this environment, decide whether to adjust provider access/user-agent/network configuration, and make data-owner decisions before building first fundamental source context.
