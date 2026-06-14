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
