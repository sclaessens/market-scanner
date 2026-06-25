# ME-SR10 — Manual cached-source snapshot staging validator audit

## Purpose

ME-SR10 implements a local, deterministic staging validator for manually staged cached-source snapshots. The validator inspects acquisition manifests and referenced local payload files before staged snapshots can be accepted as usable local cached-source inputs for future dry-runs.

The validator answers a stricter question than ME-SR09 inventory:

* ME-SR09 inventory answers what is present under a local root.
* ME-SR10 staging validation answers what may be accepted as manually staged cached-source input.

ME-SR10 does not acquire, fetch, stage, import, normalize, or refresh source data.

## Implemented Command

Command shape:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.source_refresh.cached_source_snapshot_staging_validator_command --staging-root <path>
```

Options:

```text
--staging-root <path>
--ticker <ticker-or-comma-list>
--output-json <path>
--validated-at <utc timestamp>
--human
```

Default output is JSON. `--output-json` writes the same report to a local file. `--human` prints a compact operator-readable summary.

## Validation Report Format

Report format:

```text
market-engine-cached-source-snapshot-staging-validation-v1
```

The report includes:

* `report_format_version`;
* `validated_at`;
* `staging_root`;
* `ticker_filter`;
* deterministic counts;
* `acceptance_summary`;
* per-entry validation records;
* forbidden side-effect confirmation.

Per-entry records include ticker, snapshot id, source family, source name, manifest path, payload path, manifest format version, staging validation status, accepted flag, manifest validation status, staleness status, and stable issue codes.

## Accepted and Rejected Semantics

Accepted entries use:

```text
staging_validation_status=accepted
accepted_for_cached_source_staging=true
issues=[]
```

Rejected entries use one of:

```text
staging_validation_status=rejected
staging_validation_status=missing_manifest
staging_validation_status=malformed_manifest
staging_validation_status=unknown_format
accepted_for_cached_source_staging=false
```

An entry can be accepted only when the ME-SR08 acquisition manifest format is recognized, all required fields are present with valid primitive types, the local payload exists under the staging root or manifest directory, hash and size match, staleness and manifest validation states do not block use, fixture/test material is not used as real coverage, and `usable_for_cached_source_dry_run=true` has no conflicting issues.

## Issue-Code Families

ME-SR10 emits deterministic issue codes including:

* manifest shape: `manifest_missing`, `manifest_json_malformed`, `manifest_must_be_json_object`, `manifest_format_unknown`;
* required metadata: `<field>_missing`, `<field>_invalid`;
* local references: `local_snapshot_path_missing`, `referenced_snapshot_path_outside_staging_root`, `referenced_snapshot_file_missing`, `referenced_snapshot_not_file`, `local_manifest_path_mismatch`;
* payload integrity: `referenced_snapshot_hash_mismatch`, `referenced_snapshot_size_mismatch`, `referenced_snapshot_unreadable`;
* governance blockers: `snapshot_stale`, `validation_status_failed`, `validation_status_not_validated`, `test_fixture_not_real_coverage`, `synthetic_fixture_not_real_coverage`, `usable_for_cached_source_dry_run_false`, `usable_flag_conflicts_with_staging_issues`.

## Fail-Closed Behavior

The validator rejects staged entries when:

* the manifest is missing;
* the manifest is malformed JSON;
* the manifest is not a JSON object;
* the manifest format is unknown;
* required fields are missing or invalid;
* the referenced local payload is missing, unreadable, outside the allowed local staging boundary, or not a file;
* payload hash or size mismatches;
* `staleness_status=stale`;
* `validation_status=failed` or `validation_status=not_validated`;
* `acquisition_mode=test_fixture`;
* `source_material_type=synthetic_fixture`;
* `usable_for_cached_source_dry_run=false`;
* `usable_for_cached_source_dry_run=true` conflicts with blocking issues.

Malformed or unreadable entries do not crash the report. They become rejected entries with explicit issue codes.

## Files Changed

```text
src/market_engine/source_refresh/cached_source_snapshot_staging_validator.py
src/market_engine/source_refresh/cached_source_snapshot_staging_validator_command.py
src/market_engine/source_refresh/__init__.py
tests/market_engine/source_refresh/test_cached_source_snapshot_staging_validator.py
docs/market_engine/audits/me_sr10_manual_cached_source_snapshot_staging_validator_audit.md
docs/market_engine/backlog/me_sr10_manual_cached_source_snapshot_staging_validator_backlog_entry.md
docs/market_engine/roadmap/me_sr10_manual_cached_source_snapshot_staging_validator_roadmap_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## Tests Added

ME-SR10 added deterministic `tmp_path` tests for:

* empty staging root;
* valid staged manifest and payload accepted;
* missing manifest rejected;
* malformed manifest rejected without crashing;
* unknown manifest format rejected;
* missing referenced payload rejected;
* payload hash mismatch rejected;
* payload size mismatch rejected;
* stale snapshot rejected;
* `validation_status=failed` rejected;
* `validation_status=not_validated` rejected;
* `usable_for_cached_source_dry_run=false` rejected;
* true usable flag with blocking issues rejected and flagged as conflict;
* synthetic/test fixture material rejected;
* primitive type validation;
* ticker filtering;
* JSON report writing;
* human-readable output;
* no provider/network module dependency on the command path.

## Non-Goals

ME-SR10 does not:

* acquire, stage, import, refresh, or generate snapshots;
* implement provider access;
* call SEC, EDGAR, yfinance, brokers, Telegram, scraping, or external APIs;
* normalize provider payloads;
* modify cached-source dry-run semantics;
* mutate portfolio or watchlist state;
* change Decision Engine or Recommendation Review behavior;
* add ranking, scoring, conviction, urgency, allocation, position sizing, order, execution, or tradeability semantics;
* mark unavailable data as acquired or supported.

## Backlog and Roadmap Alignment

ME-SR10 follows ME-SR09 because inventory must first show what is present, then staging validation can decide what may be accepted. The next logical sprint remains ME-SR11, renamed in the roadmap as the cached-source snapshot acquisition dry-run command. ME-SR11 should exercise acquisition/import workflow boundaries without live provider access, production writes, or bypassing ME-SR08 and ME-SR10 validation gates.

## Validation

Expected validation commands:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_refresh -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
git diff --check
```
