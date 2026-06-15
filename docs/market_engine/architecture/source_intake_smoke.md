# Market Engine Source Intake Smoke

Owner role: Technical Architect / Development Lead / Data Steward / QA / Test Lead

Status: ME05 IMPLEMENTED

## Purpose

ME05 adds the first controlled Market Engine implementation slice: source intake through an explicit provider boundary.

The slice answers whether Market Engine can process a ticker list, capture source availability and missingness per ticker, isolate provider failures, and return a batch summary without producing analysis, recommendations, reports, Telegram messages, portfolio mutations, watchlist mutations, or Decision Engine behavior.

## What ME05 Built

ME05 created a new package under `src/market_engine/source_intake/`.

The package includes:

* typed source-intake result and batch summary models;
* source readiness statuses;
* an explicit provider boundary;
* fake provider scenarios for automated tests;
* a batch runner that continues after per-ticker failures;
* missing-field frequency tracking;
* a bounded manual smoke entrypoint using the fake provider by default.

The implementation does not import from `market_scanner` or `scripts`.

## Fit In The Market Engine Flow

ME05 covers these ME02 flow stages:

```text
ticker universe / watchlist selection
-> source intake request
-> provider/source access
-> source coverage validation
-> raw source result preservation
-> normalized data view
-> missing-data and quality-state handling
```

The ME05 implementation stops at source readiness, raw evidence presence, normalized source view, missing fields, controlled errors, and batch summary.

It does not enter scanner context, fundamental context, analysis, decision preparation, reporting, Telegram, portfolio, or watchlist mutation.

## Readiness Statuses

`AVAILABLE`

All required fields are present and not missing.

`PARTIAL`

At least one required field is present and at least one required field is missing.

`MISSING`

No provider response exists or none of the required fields are available.

`PROVIDER_ERROR`

The provider boundary returned a controlled provider failure.

`UNSUPPORTED`

The provider boundary indicates that the ticker is not supported.

`INVALID_TICKER`

The provider boundary indicates that the ticker is malformed or invalid for intake.

## Missing Data Rule

Missing data remains missing. The runner treats absent required fields and `None` values as missing.

The runner does not convert missing numeric values to `0`. A real `0` remains a real value and is not treated as missing.

## Manual Smoke Behavior

The manual smoke entrypoint is:

```bash
PYTHONPATH=src python -m market_engine.source_intake.manual_smoke
```

Default behavior:

* uses a fake provider only;
* uses a small built-in fake ticker set;
* prints a concise local summary to stdout;
* does not write files unless `--write-artifact` is explicitly provided;
* does not call live providers.

Optional local ticker file:

```bash
PYTHONPATH=src python -m market_engine.source_intake.manual_smoke --ticker-file path/to/tickers.txt
```

Optional non-production local artifact:

```bash
PYTHONPATH=src python -m market_engine.source_intake.manual_smoke --write-artifact tmp/market_engine_source_intake_smoke.json
```

ME05 does not implement real provider access. Real provider source intake belongs to ME06 and must be bounded explicitly.

## What ME05 Does Not Do

ME05 does not:

* call live providers;
* call yfinance, SEC, or EDGAR;
* run scanner logic;
* run fundamental analysis;
* run the Decision Engine;
* generate production reports;
* send Telegram messages;
* mutate portfolio data;
* mutate watchlist data;
* emit BUY / SELL / HOLD;
* emit recommendation, allocation, ranking, conviction, urgency, tradeability, position sizing, or execution advice.

## ME06 Follow-Up

ME06 should add bounded real-provider source intake smoke and coverage review.

ME06 should focus on:

* selecting the first real provider boundary;
* requiring explicit operator invocation;
* enforcing a ticker limit;
* recording source coverage evidence;
* preserving missing data;
* distinguishing provider errors, unsupported tickers, invalid tickers, missing sources, partial sources, and available sources;
* keeping live-provider checks out of normal automated tests;
* avoiding analysis and recommendation behavior.

## ME06 Bounded Real-Provider Smoke

ME06 adds SEC CompanyFacts as the first real-provider source-intake smoke candidate.

The SEC adapter lives under the existing Market Engine source-intake boundary:

```text
src/market_engine/source_intake/sec_companyfacts_provider.py
```

It implements the ME05 provider protocol and is only invoked when explicitly called. There are no provider calls at import time.

## SEC CompanyFacts Scope

The ME06 adapter is intentionally small and bounded.

Required fields:

* `revenue`
* `net_income`
* `operating_cash_flow`
* `capital_expenditures`

Smoke ticker-to-CIK mapping is limited to a small in-code sample:

* `NVDA`
* `AMD`
* `META`
* `COST`

The mapping exists only for bounded smoke behavior. It is not a full ticker master, data pipeline, or source truth.

## SEC Fact Mapping

Current aliases:

* `revenue`: `Revenues`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet`
* `net_income`: `NetIncomeLoss`
* `operating_cash_flow`: `NetCashProvidedByUsedInOperatingActivities`
* `capital_expenditures`: `PaymentsToAcquirePropertyPlantAndEquipment`, `PaymentsToAcquireProductiveAssets`

The adapter selects the latest available USD fact by period end date.

Missing values remain missing. ME06 does not derive free cash flow and does not convert missing numeric values to `0`.

## Manual Real-Provider Invocation

Default manual smoke remains fake-provider only:

```bash
PYTHONPATH=src python -m market_engine.source_intake.manual_smoke
```

SEC CompanyFacts smoke requires explicit flags:

```bash
PYTHONPATH=src python -m market_engine.source_intake.manual_smoke \
  --provider sec-companyfacts \
  --tickers NVDA AMD META COST \
  --max-tickers 4
```

Real-provider smoke refuses to run without one of:

* `--tickers`
* `--ticker-file`
* `--use-sec-sample`

Real-provider smoke also refuses ticker counts above `--max-tickers`.

## Coverage Review

ME06 adds `coverage_review.py`, which turns a batch source-intake summary into a local coverage review.

The coverage review includes:

* provider name;
* ticker count;
* readiness counts;
* missing-field frequency;
* provider-error count;
* unsupported count;
* invalid ticker count;
* top missing fields;
* failed or unsupported tickers.

Coverage review is source coverage evidence only. It is not analysis.

It does not emit scores, rankings, recommendations, BUY / SELL / HOLD, allocation, conviction, urgency, tradeability, position sizing, or execution advice.

## Optional Smoke Artifact

Manual smoke writes no files by default.

If `--write-smoke-artifact` is passed, the path must be under:

```text
data/market_engine/smokes/source_intake/
```

Existing files are not overwritten silently. Smoke artifacts are local evidence only and are not source truth by default.

## ME07 Follow-Up

ME07 should review real-provider coverage behavior and define source-data owner decisions before any first fundamental source context is built.

ME07 should decide:

* whether SEC access/user-agent/network setup needs adjustment;
* whether ticker-to-CIK mapping should remain in-code for smoke or move to an approved input;
* how smoke evidence should be retained or excluded;
* which SEC aliases are sufficient for first source context;
* whether generated smoke artifacts belong outside version control.

## ME07 Real-Provider Diagnostic Outcome

ME07 reviewed the ME06 bounded SEC CompanyFacts smoke failure.

The bounded diagnostic evidence showed DNS/name resolution failure before an SEC HTTP response was returned:

```text
url_error gaierror [Errno 8] nodename nor servname provided, or not known
```

The current best hypothesis is that the ME06 `PROVIDER_ERROR` result was caused by environment/network resolution, not by:

* missing CIK mapping for the sampled tickers;
* invalid CIK formatting;
* wrong CompanyFacts endpoint shape;
* missing facts being mislabeled as provider errors;
* SEC fact-label mapping.

ME07 improved diagnostics so coverage review now shows controlled provider error categories.

## ME07 SEC CompanyFacts Coverage Status

Bounded SEC CompanyFacts smoke still returns provider errors in this environment, but now identifies the controlled category:

```text
readiness=PROVIDER_ERROR=4
provider_error_categories=SecCompanyFactsNetworkError=4
```

This means source coverage is blocked by provider/network access in the local execution environment.

It does not prove SEC CompanyFacts is unsuitable as a source.

## ME07 Source-Data Owner Decision

Decision: `APPROVED_FOR_BOUNDED_SMOKE_ONLY`

SEC CompanyFacts remains the first provider candidate for bounded source-intake smoke, but it is not yet approved for all-ticker US fundamental source intake or analysis.

Before analysis can be built, Market Engine still needs:

* successful bounded SEC access or a documented source-access resolution;
* source-data owner decision for ticker-to-CIK mapping ownership;
* source evidence retention and artifact policy;
* alias coverage review for required SEC fields;
* confirmation that missing facts become `MISSING` or `PARTIAL`, not provider errors.

## What Source Intake Can Do After ME07

After ME07, source intake can:

* run fake-provider smoke;
* run mocked SEC provider tests;
* run bounded manual SEC smoke with explicit flags;
* distinguish network, HTTP, and JSON parse provider errors;
* distinguish unsupported and invalid tickers;
* classify missing or partial required facts without turning them into provider errors;
* summarize provider error categories in coverage review.

Source intake still cannot:

* perform broad or unbounded provider calls;
* become analysis;
* become recommendation behavior;
* produce scores or rankings;
* mutate data, reports, portfolio, or watchlist files;
* call reporting, Telegram, or the Decision Engine.

## What Remains Blocked Before Analysis

First fundamental source context remains blocked until:

* bounded SEC access succeeds or a second provider is selected;
* ticker-to-CIK mapping ownership is settled;
* source evidence retention policy is settled;
* required-field alias coverage is reviewed;
* data-owner readiness criteria are documented.
