# ME-SR12 - Operator-supplied cached-source snapshot import command roadmap entry

COMPLETED BY ME-SR12.

## Roadmap position

```text
ME-UNI09 -> ME-SR06 -> ME-RUN23 -> ME-RUN24 -> ME-SR07 -> ME-SR08 -> ME-SR09 -> ME-SR10 -> ME-SR11 -> ME-SR12
```

ME-SR12 implements a local operator-supplied cached-source snapshot import command. It validates one operator-supplied snapshot directory or manifest using ME-SR10 staging validation, then copies the accepted snapshot into the configured cached-source snapshot workspace.

## Delivered scope

ME-SR12 provides:

* module command `market_engine.source_refresh.cached_source_snapshot_import_command`;
* default destination root `data/market_engine/cached_source_snapshots`;
* destination layout `<destination-root>/<batch_id>/<ticker>/<snapshot_id>/`;
* fail-closed source and manifest validation;
* no-overwrite destination behavior;
* stable terminal output for success and expected failures;
* deterministic tests.

## Next logical sprint

```text
ME-RUN25 - Rerun expanded cached-source coverage audit after validated local imports exist
```

ME-RUN25 should inspect whether imported and staged snapshots can improve expanded cached-source coverage through existing local-only paths. It must not bypass ME-SR08 manifest requirements, ME-SR10 staging validation, or source-family governance boundaries.

Non-US ticker source-family and source-mapping governance remains future work.
