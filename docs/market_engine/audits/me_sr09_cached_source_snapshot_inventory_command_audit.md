# ME-SR09 — Cached-source snapshot inventory command audit

## Purpose

ME-SR09 implements a local, deterministic inventory surface for cached-source snapshot acquisition manifests. The command inspects local files only and reports which snapshot entries are usable, unusable, missing a manifest, malformed, unknown-format, stale, or missing a referenced payload.

The sprint does not acquire snapshots, call providers, fetch live data, mutate portfolio/watchlist state, send notifications, write production data, or change Decision Engine behavior.

## Implemented Command

Command shape:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.source_refresh.cached_source_snapshot_inventory_command --input-root <path>
```

Options:

```text
--input-root <path>
--ticker <ticker-or-comma-list>
--output-json <path>
--inspected-at <utc timestamp>
--human
```

Default output is JSON. `--output-json` writes the same JSON report to a local path. `--human` prints a compact operator-readable summary instead of JSON.

## Inventory Report Format

Report format:

```text
market-engine-cached-source-snapshot-inventory-v1
```

The report includes:

* `report_format_version`;
* `inspected_at`;
* `input_root`;
* `ticker_filter`;
* deterministic counts;
* per-entry inventory records;
* forbidden side-effect confirmation.

Counts include:

* `total_inspected_entries`;
* `usable_entries`;
* `unusable_entries`;
* `missing_manifest_count`;
* `malformed_manifest_count`;
* `unknown_format_count`;
* `missing_referenced_file_count`;
* `stale_count`.

Per-entry records include ticker, snapshot id, source family, source name, manifest path, snapshot path, manifest format version, inventory status, validation status, staleness status, usability, and deterministic issue codes.

## Fail-Closed Behavior

The inventory command is fail-closed:

* malformed JSON is reported as `malformed_manifest`;
* leaf snapshot directories without `manifest.json` are reported as `missing_manifest`;
* unsupported manifest format versions are reported as `unknown_format`;
* missing referenced payload files are reported as unusable;
* hash and size mismatches are reported as unusable;
* stale manifests are counted and surfaced;
* `test_fixture` and `synthetic_fixture` entries cannot be treated as real usable coverage;
* unexpected leaf directories are inventoried deterministically instead of crashing the run.

The command never calls source providers, broker APIs, notification channels, portfolio/watchlist writers, or Decision Engine code.

## Files Changed

```text
src/market_engine/source_refresh/cached_source_snapshot_inventory.py
src/market_engine/source_refresh/cached_source_snapshot_inventory_command.py
src/market_engine/source_refresh/__init__.py
tests/market_engine/source_refresh/test_cached_source_snapshot_inventory.py
docs/market_engine/audits/me_sr09_cached_source_snapshot_inventory_command_audit.md
docs/market_engine/backlog/me_sr09_cached_source_snapshot_inventory_command_backlog_entry.md
docs/market_engine/roadmap/me_sr09_cached_source_snapshot_inventory_command_roadmap_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## Tests Added

ME-SR09 added deterministic `tmp_path` tests for:

* empty input root;
* valid manifest and payload;
* missing manifest;
* malformed manifest JSON;
* missing referenced payload file;
* unknown manifest format;
* stale manifest;
* synthetic/test fixture fail-closed behavior;
* ticker filtering;
* JSON report writing;
* human-readable output;
* no provider module dependency on the command path.

## Non-Goals

ME-SR09 does not:

* acquire, stage, import, refresh, or generate snapshots;
* implement provider access;
* call SEC, EDGAR, yfinance, brokers, Telegram, scraping, or external APIs;
* modify cached-source dry-run semantics;
* mutate portfolio or watchlist state;
* change Decision Engine behavior;
* add ranking, scoring, conviction, urgency, allocation, position sizing, order, execution, or tradeability semantics;
* mark unavailable data as acquired or supported.

## Backlog and Roadmap Alignment

ME-SR09 follows ME-SR08 because the inventory command depends on the acquisition manifest contract. The existing sequence is preserved: ME-SR10 remains the next logical sprint as the manual cached-source snapshot staging validator. ME-SR10 should validate staged payloads and manifests against the ME-SR08 contract before any bounded acquisition or import workflow is implemented.

## Validation

Expected validation commands:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_refresh/test_cached_source_snapshot_inventory.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q -k "inventory or cached_source or snapshot"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
```
