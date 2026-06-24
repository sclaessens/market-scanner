# ME-SR08 — Cached-source snapshot acquisition manifest contract roadmap entry

## Status

COMPLETED BY ME-SR08.

## Roadmap Position

ME-SR08 follows:

```text
ME-UNI09 -> ME-SR06 -> ME-RUN23 -> ME-RUN24 -> ME-SR07 -> ME-SR08
```

ME-SR07 planned missing expanded-universe snapshot acquisition. ME-SR08 formalizes the acquisition manifest contract required before later inventory, validation, staging, acquisition, or import work can safely begin.

## Outcome

ME-SR08 defines:

- `market-engine-cached-source-snapshot-acquisition-manifest-v1`;
- `market-engine-cached-source-snapshot-acquisition-batch-manifest-v1`;
- required snapshot and batch manifest fields;
- field semantics and allowed values;
- source-family and non-US metadata requirements;
- governance/source-use constraints;
- local path and dry-run reference patterns;
- validation and failure behavior.

## Next Logical Sprint

Recommended next sprint:

```text
ME-SR09 — Implement missing expanded-universe snapshot coverage inventory command
```

ME-SR09 should remain inventory-only. It may reference ME-SR08 manifest requirements but must not acquire snapshots or implement provider access.

## Boundaries

ME-SR08 is docs-only. It does not acquire snapshots, implement provider access, perform live fetches, stage data, modify runtime analysis behavior, mutate portfolio/watchlist state, or add action semantics.
