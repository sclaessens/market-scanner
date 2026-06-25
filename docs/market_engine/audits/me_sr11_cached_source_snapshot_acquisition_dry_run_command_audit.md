# ME-SR11 - Cached-source snapshot acquisition dry-run command audit

## Purpose

ME-SR11 implements a local, deterministic dry-run planner for cached-source snapshot acquisition intent. The command reports what a future acquisition or operator import would need to produce, where snapshots would be staged, which ME-SR08 manifest contract fields are required, and which requests are blocked before any acquisition can occur.

ME-SR11 does not acquire, fetch, stage, import, normalize, or refresh source data.

## Implemented Command

Command shape:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.source_refresh.cached_source_snapshot_acquisition_dry_run_command --ticker NVDA --source-family sec_companyfacts --output-root <path>
```

Options:

```text
--ticker <ticker-or-comma-list>
--source-family <source-family-or-comma-list>
--output-root <path>
--dry-run-at <utc timestamp>
--batch-id <id>
--output-json <path>
--human
```

Default output is JSON. `--output-json` writes only the dry-run report JSON. `--human` prints a compact operator-readable summary.

## Dry-Run Report Format

Report format:

```text
market-engine-cached-source-snapshot-acquisition-dry-run-v1
```

The report includes:

* `report_format_version`;
* `dry_run_at`;
* `output_root`;
* `batch_id`;
* `requested_tickers`;
* `requested_source_families`;
* `supported_source_families`;
* deterministic counts;
* per-entry planned or blocked records;
* `staging_validator_handoff`;
* forbidden side-effect confirmation.

Per-entry records include ticker, source family, `acquisition_mode=dry_run_only`, planned/blocked status, false side-effect flags, expected ME-SR08 manifest version, proposed staging path, proposed manifest path, required manifest fields, required payload metadata fields, ME-SR10 validation handoff, and stable issue codes.

## Planned and Blocked Semantics

Planned entries use:

```text
acquisition_dry_run_status=planned
would_acquire=false
would_acquire_external_data=false
would_write_payload=false
would_write_manifest=false
issues=[]
```

Blocked entries use:

```text
acquisition_dry_run_status=blocked
would_acquire=false
would_acquire_external_data=false
would_write_payload=false
would_write_manifest=false
issues=[...]
```

No ticker request and no source-family request produce no-op reports with deterministic missing-input counts.

## Supported Source-Family Scope

ME-SR11 supports only this dry-run source family:

```text
sec_companyfacts
```

This means the dry-run report understands the family as a planning target. It does not add a provider client, HTTP client, SEC/EDGAR adapter, yfinance dependency, acquisition client, normalization path, or credential path.

Unknown source families are blocked with `source_family_unsupported`.

## Issue Codes

ME-SR11 emits deterministic issue codes including:

* `ticker_invalid`;
* `source_family_unsupported`;
* `output_root_missing`.

Counts also expose missing request classes:

* `missing_ticker_count`;
* `missing_source_family_count`;
* `invalid_ticker_count`;
* `unsupported_source_family_count`.

## Relationship to ME-SR09 and ME-SR10

ME-SR09 inventory answers what local cached-source snapshots and manifests are present under a local root.

ME-SR10 staging validation answers whether manually staged payloads and acquisition manifests can be accepted as local cached-source input.

ME-SR11 acquisition dry-run answers what a future acquisition or operator-import step would need to produce, without producing it.

ME-SR11 preserves ME-SR09 and ME-SR10 report format versions and field semantics.

## Tests Added

ME-SR11 added deterministic `tmp_path` tests for:

* one valid ticker and supported source family producing one planned entry;
* multiple tickers and source families producing deterministic sorted entries;
* repeatable/comma-separated ticker and source-family parsing;
* invalid ticker rejection;
* missing ticker no-op report;
* missing source-family no-op report;
* unsupported source-family rejection;
* missing output root blocked entry;
* `--output-json` writing only the dry-run report;
* `--human` compact output;
* no provider/network module dependency on the command path.

## Non-Goals

ME-SR11 does not:

* acquire snapshots;
* fetch live data;
* call providers, SEC, EDGAR, yfinance, brokers, Telegram, scraping, or external APIs;
* write payload files;
* write acquisition manifests;
* normalize provider payloads;
* add credentials, provider adapters, HTTP clients, or acquisition clients;
* mutate portfolio or watchlist state;
* change Decision Engine or Recommendation Review behavior;
* add ranking, scoring, conviction, urgency, allocation, position sizing, order, execution, tradeability, BUY, SELL, or HOLD semantics;
* mark unavailable data as acquired or supported.

## Backlog and Roadmap Alignment

ME-SR11 follows ME-SR10 because staging validation gates must exist before acquisition/import planning can be made visible. The next recommended sprint is ME-SR12: implement an operator-supplied cached-source snapshot import command. That sprint should copy or register operator-supplied local payloads into a controlled staging layout with generated or verified manifest metadata, still without provider calls.

Non-US source-family and source-mapping governance remains future work and must not be bypassed by import tooling.

## Validation

Expected validation commands:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_refresh -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
git diff --check
```
