# ME-SR01 — SEC CompanyFacts Raw Snapshot Persistence Audit

Owner role: Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Sprint ID: ME-SR01

Job family: Source Refresh

Branch name: `me-sr01-sec-companyfacts-raw-snapshot-cache`

Status: COMPLETED BY ME-SR01

## Purpose

This audit records ME-SR01 implementation of raw SEC CompanyFacts source snapshot persistence and cached raw snapshot loading.

ME-SR01 stays inside the Source Refresh job family. It does not create source context, observations, analysis, recommendations, portfolio review, delivery, Telegram output, or Decision Engine behavior.

## Files Changed

Created:

- `src/market_engine/source_refresh/__init__.py`
- `src/market_engine/source_refresh/sec_companyfacts_snapshots.py`
- `tests/market_engine/source_refresh/test_sec_companyfacts_snapshots.py`
- `docs/market_engine/source_refresh/me_sr01_sec_companyfacts_raw_snapshot_cache.md`
- `docs/market_engine/audits/me_sr01_sec_companyfacts_raw_snapshot_persistence_audit.md`

Updated:

- `src/market_engine/source_intake/sec_companyfacts_provider.py`
- `docs/market_engine/backlog/market_engine_backlog.md`

## Python Code Changed

Yes.

Runtime changes were limited to:

- a new Market Engine Source Refresh package for SEC CompanyFacts raw snapshot persistence and cached loading;
- a narrow SEC CompanyFacts provider cached-snapshot option that loads an explicit cached snapshot path without calling the provider fetcher.

## Tests Changed

Yes.

Added Source Refresh tests for raw snapshot persistence, cached loading, explicit failure behavior, and cached provider behavior.

## Data Files Changed

No committed data files were changed.

Tests write only to temporary test directories.

## Generated Files Changed

No generated files were committed.

## Provider Calls Introduced

No live provider calls were introduced in automated tests.

Cached-loading tests prove the provider fetcher is not called when an explicit cached snapshot path is supplied.

## Runtime Behavior Changed

Only Source Refresh and explicit cached SEC provider behavior changed.

Existing live SEC fetch behavior remains available through the existing provider path.

## Cached Loading Behavior

Cached loading supports:

- explicit snapshot file loading;
- latest matching snapshot loading by ticker and/or CIK;
- metadata validation;
- raw payload preservation;
- controlled failures for missing files, invalid JSON, missing metadata, unsupported formats, and entity mismatches.

Cached loading returns raw payload evidence only.

## Recommendation Behavior Changed

No.

No recommendation behavior was added or changed.

## Portfolio Behavior Changed

No.

No portfolio behavior was added or changed.

## Delivery Or Telegram Behavior Changed

No.

No delivery or Telegram behavior was added or changed.

## Decision Engine Behavior Changed

No.

No Decision Engine behavior was added or changed.

## Source Refresh Scope Confirmation

ME-SR01 stayed inside Source Refresh scope:

- raw SEC CompanyFacts payload persistence;
- cached raw payload loading;
- source refresh metadata and manifests;
- controlled source refresh failures.

ME-SR01 did not create source context, fundamental observations, derived observations, analysis review, recommendation review, portfolio review, delivery output, Telegram output, or Decision Engine behavior.

## Tests Run

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/source_refresh tests/market_engine/source_intake -q
```

Result:

```text
71 passed
```

## Follow-Up Work

Recommended follow-ups:

- `ME-SR02` may add a bounded Source Refresh job runner that fetches a ticker set and writes run-level source snapshots.
- `ME-SC01` may build SEC CompanyFacts source context from persisted cached snapshots.
- `ME-DATA01` may define source snapshot retention policy.
