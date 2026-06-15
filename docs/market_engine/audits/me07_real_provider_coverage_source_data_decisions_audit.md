# ME07 Real Provider Coverage And Source-Data Decisions Audit

Owner role: Governance Auditor / Data Steward / Technical Architect

Status: COMPLETED BY ME07

## Purpose

ME07 reviews the bounded SEC CompanyFacts provider smoke introduced in ME06, improves controlled diagnostics, and records source-data owner decisions before any analysis layer is built.

ME07 remains a source/provider diagnostics and source-data ownership sprint. It does not authorize analysis, scanner ranking, fundamental scoring, reporting, Telegram delivery, portfolio mutation, watchlist mutation, or Decision Engine behavior.

## Files Created

* `docs/market_engine/audits/me07_real_provider_coverage_source_data_decisions_audit.md`

## Files Updated

* `src/market_engine/source_intake/sec_companyfacts_provider.py`
* `src/market_engine/source_intake/coverage_review.py`
* `tests/market_engine/source_intake/test_sec_companyfacts_provider.py`
* `docs/market_engine/architecture/source_intake_smoke.md`
* `docs/market_engine/backlog/market_engine_backlog.md`

## ME06 Finding Investigated

ME06 ran the bounded SEC CompanyFacts smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.source_intake.manual_smoke --provider sec-companyfacts --tickers NVDA AMD META COST --max-tickers 4
```

ME06 result:

```text
readiness=PROVIDER_ERROR=4
provider_errors=4
failed_or_unsupported_tickers=NVDA, AMD, META, COST
```

ME06 proved failure isolation but did not yet distinguish the provider-error category.

## Diagnosis Summary

A bounded diagnostic request against:

```text
https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json
```

returned:

```text
url_error gaierror [Errno 8] nodename nor servname provided, or not known
```

Best current hypothesis:

The ME06 `PROVIDER_ERROR` result was caused by environment/network DNS resolution failure before SEC returned an HTTP response. The evidence does not point to bad CIK formatting, missing smoke CIK mapping, wrong endpoint shape, missing facts, or SEC fact-label mapping as the immediate cause.

## Provider Selected

Provider: SEC CompanyFacts

SEC CompanyFacts remains the first provider candidate for bounded source-intake smoke because it is relevant to fundamental source coverage and can be tested through a narrow source boundary.

## Implementation Changes Made

ME07 added more precise controlled provider diagnostics:

* `SecCompanyFactsNetworkError`
* `SecCompanyFactsHttpError`
* `SecCompanyFactsJsonParseError`

ME07 also updated coverage review output to include provider error categories.

Missing CIK mappings remain `UNSUPPORTED`.

Invalid ticker text remains `INVALID_TICKER`.

Missing CompanyFacts payloads or missing required facts remain `MISSING` or `PARTIAL`, not generic provider errors.

## Required Fields

ME07 keeps the ME06 required field set:

* `revenue`
* `net_income`
* `operating_cash_flow`
* `capital_expenditures`

Missing numeric fields remain missing and are not converted to `0`.

## Exact Bounded Manual Smoke Command Used

Fake provider smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.source_intake.manual_smoke
```

Bounded SEC CompanyFacts smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.source_intake.manual_smoke --provider sec-companyfacts --tickers NVDA AMD META COST --max-tickers 4
```

## Bounded Manual Smoke Result

Fake provider smoke result:

```text
provider=fake-source-provider
tickers=7
readiness=AVAILABLE=2, INVALID_TICKER=1, MISSING=1, PARTIAL=1, PROVIDER_ERROR=1, UNSUPPORTED=1
missing_fields=capital_expenditures=5, operating_cash_flow=5, revenue=4
provider_errors=1
provider_error_categories=ProviderUnavailableError=1
unsupported=1
invalid_tickers=1
failed_or_unsupported_tickers=UNSUPPORTED, INVALID, ERROR
note=Source coverage evidence only. Not analysis.
```

Bounded SEC CompanyFacts smoke result:

```text
bounded_real_provider_smoke=true
provider_warning=manual SEC CompanyFacts smoke; source coverage evidence only
provider=SEC_COMPANYFACTS
tickers=4
readiness=PROVIDER_ERROR=4
missing_fields=capital_expenditures=4, net_income=4, operating_cash_flow=4, revenue=4
provider_errors=4
provider_error_categories=SecCompanyFactsNetworkError=4
unsupported=0
invalid_tickers=0
failed_or_unsupported_tickers=NVDA, AMD, META, COST
note=Source coverage evidence only. Not analysis.
```

## Controlled Error Categories Observed

Observed in live bounded smoke:

* `SecCompanyFactsNetworkError`: 4

Covered by automated mocked tests:

* `SecCompanyFactsNetworkError`
* `SecCompanyFactsHttpError`
* `SecCompanyFactsJsonParseError`
* `UNSUPPORTED` for missing CIK mapping
* `INVALID_TICKER` for invalid ticker text
* `MISSING` for missing CompanyFacts payload or missing required facts
* `PARTIAL` for partial required-field coverage

## Source-Data Owner Decision

Decision: `APPROVED_FOR_BOUNDED_SMOKE_ONLY`

SEC CompanyFacts remains a suitable first provider candidate for bounded smoke diagnostics, but it is not yet approved as the general US fundamental source-intake provider for analysis.

Reasons:

* The adapter can classify mocked successful, partial, missing, unsupported, invalid, HTTP, JSON, and network cases.
* The live bounded smoke is blocked by network/DNS resolution in this environment.
* The current ticker-to-CIK mapping is intentionally small and smoke-only.
* Source evidence retention and artifact policy are not finalized.
* SEC alias coverage is conservative and requires review before source context or analysis work.

Second provider decision: not required yet.

A second provider may be evaluated later if SEC access remains blocked after network/request diagnostics are repaired.

## Whether SEC Remains First Provider Candidate

SEC CompanyFacts remains the first bounded provider candidate for source-intake smoke.

It should not yet be used for all-ticker coverage, source context, or analysis until ME08 resolves network access and source-data ownership decisions.

## Test Coverage Summary

Targeted tests cover:

* full mocked CompanyFacts mapping;
* partial mocked CompanyFacts mapping;
* missing CompanyFacts response;
* missing required facts as `MISSING`;
* unsupported/no-CIK ticker;
* invalid ticker;
* network error as `SecCompanyFactsNetworkError`;
* HTTP error as `SecCompanyFactsHttpError`;
* JSON parse error as `SecCompanyFactsJsonParseError`;
* deterministic ticker normalization;
* deterministic CIK formatting;
* no provider call at import time;
* no live network required in automated tests;
* coverage review readiness counts;
* coverage review missing-field frequency;
* coverage review provider-error categories;
* coverage review absence of forbidden authority fields;
* no imports from legacy runtime modules.

Targeted test command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/source_intake -q
```

Result:

```text
34 passed
```

## Boundary Confirmations

ME07 confirms:

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

* Live SEC access still returns controlled network failures in this environment.
* The smoke ticker-to-CIK mapping is intentionally small.
* SEC alias coverage remains conservative.
* Source evidence retention policy is not finalized.
* The adapter does not derive free cash flow.
* No analysis or fundamental source context is built in ME07.

## Recommended Next Sprint

Proceed to:

`ME08 - Repair SEC CompanyFacts network access and rerun bounded coverage review`

ME08 should resolve the environment/request access issue, rerun bounded SEC coverage, and decide whether SEC can move from bounded-smoke-only to approved source-intake coverage.
