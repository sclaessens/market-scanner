# ME-SR12 - Operator-supplied cached-source snapshot import command audit

## Purpose

ME-SR12 implements a local operator-facing import command for cached-source snapshots. The command validates one operator-supplied snapshot directory or `manifest.json` file using the existing ME-SR10 staging validator, then copies the validated snapshot into the canonical local cached-source snapshot workspace.

ME-SR12 is a local import capability only. It does not acquire source data from providers.

## Implemented Command

Command shape:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.source_refresh.cached_source_snapshot_import_command --source-path <path>
```

Options:

```text
--source-path <snapshot-directory-or-manifest-json>
--destination-root <path>
--validated-at <utc timestamp>
```

Default destination root:

```text
data/market_engine/cached_source_snapshots
```

The command follows the existing source-refresh module command pattern used by ME-SR09, ME-SR10, and ME-SR11.

## Validation Behavior

The command fails closed when:

* source path is missing;
* source path does not exist;
* source path is not readable;
* source file is not `manifest.json`;
* no manifest exists below the supplied directory;
* more than one manifest exists below the supplied directory;
* manifest JSON is malformed;
* manifest is not a JSON object;
* ME-SR10 staging validation rejects the snapshot;
* destination already exists;
* temporary import destination already exists;
* copy fails.

Expected validation failures are rendered as operator-readable terminal output. They do not produce Python tracebacks.

## Destination Behavior

Imported snapshots are copied to:

```text
<destination-root>/<batch_id>/<ticker>/<snapshot_id>/
```

The `batch_id`, `ticker`, and `snapshot_id` values come from the ME-SR08 acquisition manifest. The source directory is not mutated. The import preserves the snapshot directory contents, including the manifest and referenced payload files.

Destination overwrite is not supported in ME-SR12. Existing destinations fail with `destination_already_exists`.

Copy is staged through a sibling `.importing` directory before the final rename. If copy fails, the temporary import directory is cleaned up and no success result is emitted.

## Terminal Output

Successful imports print:

* snapshot id;
* batch id;
* source family;
* source path;
* destination path;
* manifest path;
* imported entities;
* validation status;
* warnings;
* forbidden side-effect confirmation.

Failures print:

* failure heading;
* reason;
* source path when known;
* expected manifest when known;
* issue codes when available.

No Telegram or notification output is sent.

## Safety Constraints

ME-SR12 does not:

* call providers;
* call SEC or EDGAR;
* use yfinance;
* fetch live market data;
* send Telegram messages;
* mutate portfolio or watchlist state;
* modify Decision Engine or Recommendation Review behavior;
* add ranking, scoring, conviction, urgency, allocation, position sizing, order, execution, tradeability, BUY, SELL, or HOLD semantics;
* silently accept malformed snapshots;
* overwrite existing imported snapshots.

## Files Changed

```text
src/market_engine/source_refresh/cached_source_snapshot_importer.py
src/market_engine/source_refresh/cached_source_snapshot_import_command.py
tests/market_engine/source_refresh/test_cached_source_snapshot_importer.py
docs/market_engine/audits/me_sr12_operator_supplied_cached_source_snapshot_import_command_audit.md
docs/market_engine/backlog/me_sr12_operator_supplied_cached_source_snapshot_import_command_backlog_entry.md
docs/market_engine/roadmap/me_sr12_operator_supplied_cached_source_snapshot_import_command_roadmap_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## Tests Added

ME-SR12 added deterministic `tmp_path` tests for:

* successful import of a valid operator snapshot;
* direct `manifest.json` source path import;
* missing source path failure;
* nonexistent source path failure;
* missing manifest failure;
* malformed manifest failure;
* staging validation failure without copying;
* existing destination failure without overwrite;
* file preservation after import;
* stable terminal success summary;
* stable terminal failure summary;
* no provider/network module dependency on the command path.

## Out of Scope

ME-SR12 does not implement provider acquisition, live refresh, payload normalization, batch import of multiple manifests, overwrite/force behavior, import metadata sidecars, non-US source-family governance, or cached-source dry-run consumption of newly imported snapshots.

## Recommended Next Sprint

ME-RUN25 should run an expanded cached-source coverage audit after operator-supplied snapshots have been imported and validated locally. Non-US source-family and source-mapping governance remains separate future work.

## Validation

Expected validation commands:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_refresh -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
git diff --check
```
