# ME05 Source Intake Smoke Audit

Owner role: Governance Auditor

Status: COMPLETED BY ME05

## Purpose

This audit records ME05: build the first controlled Market Engine source-intake smoke harness.

ME05 is a source/data-layer implementation sprint. It does not authorize analysis, recommendations, reporting, Telegram delivery, portfolio mutation, watchlist mutation, or Decision Engine behavior.

## Files Created

* `src/market_engine/__init__.py`
* `src/market_engine/source_intake/__init__.py`
* `src/market_engine/source_intake/models.py`
* `src/market_engine/source_intake/provider_boundary.py`
* `src/market_engine/source_intake/readiness.py`
* `src/market_engine/source_intake/runner.py`
* `src/market_engine/source_intake/fake_provider.py`
* `src/market_engine/source_intake/manual_smoke.py`
* `tests/market_engine/source_intake/test_source_intake_runner.py`
* `docs/market_engine/architecture/source_intake_smoke.md`
* `docs/market_engine/audits/me05_source_intake_smoke_audit.md`

## Files Updated

* `docs/market_engine/backlog/market_engine_backlog.md`

## Implementation Summary

ME05 created a clean `market_engine` package under `src/market_engine/`.

The implementation includes:

* source readiness statuses;
* per-ticker source intake results;
* batch source intake summaries;
* explicit provider boundary protocol;
* controlled provider error types;
* fake provider scenarios;
* batch runner with per-ticker failure isolation;
* missing-field frequency tracking;
* bounded manual smoke entrypoint using fake provider data by default.

## Provider Boundary Summary

Provider access is isolated behind `SourceProvider`.

The boundary exposes:

* provider name;
* one-ticker source fetch behavior;
* controlled unsupported ticker, invalid ticker, and provider error paths.

ME05 does not implement live provider access.

## Readiness Status Definitions

* `AVAILABLE` - all required fields are present and not missing.
* `PARTIAL` - at least one required field is available and at least one required field is missing.
* `MISSING` - no source response exists or no required fields are available.
* `PROVIDER_ERROR` - controlled provider failure.
* `UNSUPPORTED` - provider does not support the ticker.
* `INVALID_TICKER` - ticker is malformed or invalid for provider intake.

## Test Coverage Summary

Targeted tests cover:

* full data returns `AVAILABLE`;
* partial data returns `PARTIAL`;
* missing fields are preserved;
* missing source returns `MISSING`;
* unsupported ticker returns `UNSUPPORTED`;
* invalid ticker returns `INVALID_TICKER`;
* provider error returns `PROVIDER_ERROR`;
* batch processing continues after provider error;
* summary counts by readiness status;
* missing-field frequency;
* missing numeric data is not converted to zero;
* result payloads do not contain forbidden authority fields;
* tests use fake provider behavior;
* tests do not import legacy runtime modules;
* empty ticker lists return clean empty summaries.

Targeted test command:

```bash
.venv/bin/pytest tests/market_engine/source_intake -q
```

Result:

```text
14 passed
```

Plain `pytest` was not available on the shell PATH, so the repository virtualenv pytest executable was used for the same targeted test path.

## Manual Smoke Behavior

Manual smoke entrypoint:

```bash
PYTHONPATH=src python -m market_engine.source_intake.manual_smoke
```

The manual smoke:

* uses fake provider data by default;
* can optionally read a local ticker file when explicitly passed;
* prints a concise local summary;
* writes no file unless `--write-artifact` is explicitly passed;
* does not call live providers.

## Boundary Confirmations

ME05 confirms:

* Old runtime files were not modified.
* `src/market_scanner/` was not modified.
* `scripts/` was not modified.
* New `src/market_engine/` code does not import from `market_scanner`.
* New `src/market_engine/` code does not import from `scripts`.
* New tests do not import from `market_scanner`.
* New tests do not import from `scripts`.
* Automated tests used fake provider behavior only.
* No live provider calls were used in tests.
* No yfinance, SEC, or EDGAR calls were used in tests.
* No existing data, CSV, or report files were modified.
* No production data writes were introduced by default.
* No reports were generated.
* No Telegram messages were sent.
* No portfolio data was mutated.
* No watchlist data was mutated.
* No Decision Engine behavior was called or changed.
* No BUY / SELL / HOLD, recommendation, allocation, ranking, conviction, urgency, tradeability, position sizing, or execution behavior is emitted by runtime models.

## Known Limitations

* The ME04 files named in the ME05 prompt, `docs/market_engine/architecture/technical_coding_testing_architecture.md` and `docs/market_engine/reference_extraction/me04_technical_coding_testing_extraction.md`, were not present on `main` when this branch started.
* ME05 therefore used the available Market Engine docs, backlog, coding/testing baselines, and ME04-PREP-D inventory, plus the explicit ME05 prompt boundaries.
* Live provider access is not implemented.
* Manual smoke uses fake provider scenarios by default.
* Source freshness, provider provenance depth, retry behavior, and artifact retention policy are deferred to ME06 or later architecture decisions.

## Recommended Next Sprint

Proceed to:

`ME06 - Add bounded real provider source intake smoke and coverage review`
