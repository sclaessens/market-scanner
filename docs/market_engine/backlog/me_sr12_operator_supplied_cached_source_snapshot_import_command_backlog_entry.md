# ME-SR12 - Operator-supplied cached-source snapshot import command backlog entry

COMPLETED BY ME-SR12.

## Summary

ME-SR12 implements a local operator-supplied cached-source snapshot import command.

It provides:

* module command `market_engine.source_refresh.cached_source_snapshot_import_command`;
* source path validation for snapshot directories and direct `manifest.json` paths;
* ME-SR10 staging validation before import;
* deterministic destination layout under `data/market_engine/cached_source_snapshots`;
* no-overwrite destination behavior;
* local copy that preserves the operator-supplied snapshot directory contents;
* stable terminal success and failure output;
* deterministic fixture-based tests.

ME-SR12 did not call providers, fetch data, use SEC/EDGAR/yfinance, send Telegram output, mutate portfolio/watchlist state, write outside the configured destination root, modify Decision Engine or Recommendation Review behavior, or add action authority.

## Implemented documentation

```text
docs/market_engine/audits/me_sr12_operator_supplied_cached_source_snapshot_import_command_audit.md
docs/market_engine/backlog/me_sr12_operator_supplied_cached_source_snapshot_import_command_backlog_entry.md
docs/market_engine/roadmap/me_sr12_operator_supplied_cached_source_snapshot_import_command_roadmap_entry.md
```

## Follow-up candidates

* ME-RUN25 - Rerun expanded cached-source coverage audit after validated local imports exist.
* Future source-family governance - Define non-US ticker source-family and source-mapping governance.
* Future import enhancement - Add explicitly tested overwrite or batch-import behavior only if operator workflow requires it.
