# V2 Controlled Real-Source Smoke Test

Status: ACTIVE
Reset stage: RESET-10L-BL6

## Purpose

This document defines the controlled v2 real-source smoke-test harness.

The harness is a manual-only boundary for checking whether an explicitly supplied or injected source response can pass through:

```text
source response
-> raw source evidence
-> normalized fundamentals
-> neutral source-data readiness
-> in-memory smoke result
```

## Design

The smoke-test module is:

```text
src/market_scanner/fundamentals/fundamentals_real_source_smoke.py
```

It uses dependency injection. Importing the module does not call providers, SEC, EDGAR, brokers, network services, reporting, Telegram, the production pipeline, or the Decision Engine.

Any source access must be triggered by an explicit function call with an injected client:

```text
run_controlled_real_source_smoke_test(...)
```

The result is returned in memory. The harness does not write raw data, normalized data, generated data, reports, logs, or Telegram artifacts.

## Tests

Tests use fake clients only. They do not make live calls and do not require credentials.

The test coverage verifies:

- explicit invocation is required before a fake client is called;
- fake source responses pass through the v2 provider boundary;
- raw evidence preserves provenance;
- normalized fundamentals remain program-ready input;
- missing values remain explicit and are not converted to zero;
- source-data readiness remains neutral;
- provider/source failure returns a neutral smoke failure status;
- no files are written.

## Non-Goals

RESET-10L-BL6 does not:

- add a live provider client;
- make SEC, EDGAR, broker, provider, or network calls;
- add credentials or API keys;
- commit live provider output;
- create or modify data files;
- generate reports;
- send Telegram messages;
- run the production pipeline;
- add Decision Engine behavior;
- add BUY, SELL, HOLD, allocation, conviction, urgency, tradeability, scoring, target-price, or recommendation logic.

## Difference From Production Integration

This is not production provider integration.

Production use still requires a separately approved manual execution review, provider access governance, credential handling, operational runbook, storage/write policy, and explicit confirmation that Decision Engine authority is not expanded.

## Next Step

The next candidate step is `RESET-10L-BL7 — Manual Real-Source Smoke Execution Review`.

That future step should define how a developer may manually run the controlled smoke harness with explicit local parameters without committing credentials, live output, data files, reports, Telegram artifacts, or production-pipeline behavior.
