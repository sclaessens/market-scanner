# ME-SR08 — Cached-source snapshot acquisition manifest contract backlog entry

## Status

COMPLETED BY ME-SR08.

## Summary

ME-SR08 defines the cached-source snapshot acquisition manifest contract.

The contract specifies:

- snapshot-level acquisition manifest scope;
- optional batch-level acquisition manifest scope;
- required manifest fields;
- field semantics and allowed values;
- source-family handling;
- governance and source-use constraints;
- local path and dry-run artifact relationships;
- validation behavior;
- failure modes;
- syntactically valid JSON examples.

## Explicit Non-Scope

ME-SR08 is docs-only.

ME-SR08 did not acquire snapshots, stage snapshots, implement provider access, fetch live data, modify dry-run behavior, change runtime code, change Decision Engine behavior, add broker/Telegram/portfolio/watchlist behavior, or create action authority.

## Follow-Up Sprint Candidates

- ME-SR09 — Implement missing expanded-universe snapshot coverage inventory command.
- ME-SR10 — Implement manual cached-source snapshot staging validator.
- ME-SR11 — Implement approved bounded acquisition or import workflow.
- ME-SR12 — Define non-US ticker source-family and source-mapping governance contract.
- ME-RUN25 — Rerun expanded cached-source coverage audit after staged snapshots exist.

ME-SR10 and ME-SR11 must implement ME-SR08 manifest validation/writing without bypassing governance constraints.

## Documentation

Implemented documentation:

```text
docs/market_engine/audits/me_sr08_cached_source_snapshot_acquisition_manifest_contract.md
docs/market_engine/backlog/me_sr08_cached_source_snapshot_acquisition_manifest_contract_backlog_entry.md
docs/market_engine/roadmap/me_sr08_cached_source_snapshot_acquisition_manifest_contract_roadmap_entry.md
```
