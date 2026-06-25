# ME-SR11 - Cached-source snapshot acquisition dry-run command roadmap entry

COMPLETED BY ME-SR11.

## Roadmap position

```text
ME-UNI09 -> ME-SR06 -> ME-RUN23 -> ME-RUN24 -> ME-SR07 -> ME-SR08 -> ME-SR09 -> ME-SR10 -> ME-SR11
```

ME-SR11 implements a local, deterministic cached-source snapshot acquisition dry-run command. It plans acquisition/import intent only and reports planned or blocked ticker/source-family entries without acquiring data, writing payloads, writing acquisition manifests, calling providers, or changing runtime dry-run behavior.

## Delivered scope

ME-SR11 provides:

* module command `market_engine.source_refresh.cached_source_snapshot_acquisition_dry_run_command`;
* JSON report format `market-engine-cached-source-snapshot-acquisition-dry-run-v1`;
* supported dry-run source family `sec_companyfacts`;
* deterministic `--dry-run-at` and optional `--batch-id`;
* `--ticker`, `--source-family`, `--output-root`, `--output-json`, and `--human` options;
* ME-SR08 manifest requirements in the report;
* ME-SR10 staging validator handoff;
* fail-closed planned/blocked semantics.

## Next logical sprint

```text
ME-SR12 - Implement operator-supplied cached-source snapshot import command
```

ME-SR12 should implement a local operator-supplied import surface that copies or registers local payloads into a controlled staging layout with generated or verified manifest metadata. It must remain local-only, avoid provider calls, preserve ME-SR08 manifest requirements, and hand off to ME-SR10 staging validation before any cached-source dry-run can consume imported snapshots.

Non-US ticker source-family and source-mapping governance remains future work and must not be bypassed by import tooling.
