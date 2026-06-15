# ME-SR01 — SEC CompanyFacts Raw Snapshot Persistence And Cached Loading

Owner role: Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Sprint ID: ME-SR01

Job family: Source Refresh

Status: COMPLETED BY ME-SR01

## Purpose

ME-SR01 implements raw SEC CompanyFacts source snapshot persistence and cached raw snapshot loading for the Market Engine Source Refresh job family.

The purpose is to let Market Engine preserve raw SEC CompanyFacts payload evidence and later load cached raw source payloads without making a provider or network call.

## Background

The ME01–ME13 foundation phase established SEC CompanyFacts source intake, field mapping, source context, non-decision observations, and the job-based persistence contract.

ME-GOV01 then established job-scoped sprint naming. This sprint is therefore `ME-SR01`, not a generic continuation of the foundation numbering.

## Scope

In scope:

- raw SEC CompanyFacts snapshot envelopes;
- snapshot metadata;
- raw payload preservation;
- ticker manifest writing;
- provider error manifest writing;
- explicit cached snapshot loading by file path;
- deterministic latest cached snapshot loading by ticker or CIK;
- cached loading through the existing SEC provider boundary when an explicit cached snapshot path is supplied;
- Source Refresh tests using temporary local payloads only.

## Non-Scope

Out of scope:

- source context construction;
- fundamental observations;
- analysis review;
- recommendation review;
- portfolio review;
- delivery jobs;
- Telegram output;
- Decision Engine behavior;
- free cash flow;
- growth;
- margins;
- ratios;
- valuation metrics;
- scores;
- rankings;
- recommendations;
- BUY / SELL / HOLD;
- portfolio or watchlist mutation;
- production report generation.

## Source-Refresh Boundary

ME-SR01 handles raw source payload evidence only.

The Source Refresh boundary stops at:

- persisted raw payload envelopes;
- cache metadata;
- manifest rows;
- provider error rows;
- controlled cached-load failures.

It does not interpret the payload into source context, observations, analysis, recommendations, delivery output, portfolio state, or Decision Engine behavior.

## Raw Snapshot Persistence Path

Canonical path:

```text
data/market_engine/source_snapshots/sec_companyfacts/<run_id>/
```

Implemented structure:

```text
raw/
  <TICKER>_companyfacts.json
snapshot_metadata.json
ticker_manifest.csv
provider_errors.csv
```

Raw snapshot files are JSON envelopes with:

- `metadata`;
- `raw_payload`.

Required metadata:

- `ticker`;
- `cik`;
- `source_name`;
- `fetched_at`;
- `snapshot_id`;
- `payload_format_version`.

Source name:

```text
sec_companyfacts
```

Payload format version:

```text
sec-companyfacts-raw-v1
```

## Cached Loading Behavior

ME-SR01 supports:

- loading an explicitly requested snapshot file;
- loading the latest matching snapshot under a root directory by ticker and/or CIK;
- validating expected ticker and CIK metadata when supplied;
- using an explicit cached snapshot path through the SEC CompanyFacts provider boundary without calling the configured fetcher.

Cached loading returns raw payload evidence. It does not create source context or observations.

## Failure Behavior

Cached loading fails explicitly for:

- missing snapshot file;
- invalid JSON;
- missing required metadata;
- missing raw payload;
- unsupported snapshot format;
- ticker mismatch;
- CIK mismatch.

Failures are not converted into fake available data.

## Testing Approach

Automated tests use temporary local payloads only.

Tests prove:

- raw snapshots can be persisted;
- cached snapshots can be loaded;
- raw payloads are preserved;
- cached loading avoids provider calls;
- missing files fail explicitly;
- invalid JSON fails explicitly;
- missing metadata fails explicitly;
- unsupported formats fail explicitly;
- ticker mismatch fails explicitly;
- cached loading does not create source context, observations, analysis, recommendation, portfolio, delivery, Telegram, or Decision Engine output.

Automated tests do not call live SEC endpoints.

## Affected Files

Runtime:

- `src/market_engine/source_refresh/__init__.py`
- `src/market_engine/source_refresh/sec_companyfacts_snapshots.py`
- `src/market_engine/source_intake/sec_companyfacts_provider.py`

Tests:

- `tests/market_engine/source_refresh/test_sec_companyfacts_snapshots.py`

Documentation:

- `docs/market_engine/source_refresh/me_sr01_sec_companyfacts_raw_snapshot_cache.md`
- `docs/market_engine/audits/me_sr01_sec_companyfacts_raw_snapshot_persistence_audit.md`
- `docs/market_engine/backlog/market_engine_backlog.md`

## Acceptance Criteria

- ME-SR01 is implemented as a Source Refresh sprint.
- Raw SEC CompanyFacts source snapshots can be persisted.
- Cached SEC CompanyFacts snapshots can be loaded without provider/network calls.
- Raw source payload evidence is preserved.
- Failure states are explicit and safe.
- Missing or invalid cache data is not converted into fake available data.
- Automated tests use mocked, synthetic, or temporary local data only.
- No source context, observations, analysis, recommendations, portfolio review, delivery, Telegram, or Decision Engine behavior is introduced.

## Follow-Up Candidates

Candidate follow-ups:

- `ME-SR02` may add a bounded Source Refresh job runner that fetches a ticker set and writes run-level snapshots.
- `ME-SC01` may build cached SEC CompanyFacts source context from persisted raw snapshots.
- `ME-DATA01` may define retention policy for source snapshots.
