# ME-SR13 - Real-world operator-supplied cached-source sample import backlog entry

BLOCKED BY ME-SR13.

## Summary

ME-SR13 attempted to start the real-world operator-supplied cached-source sample import for `NVDA`, `AMD`, and `ASML`.

Expected input root:

```text
/Users/sclaessens/Documents/market-scanner/operator_input/market_engine/me-sr13-real-world-sample/
```

The expected `operator_input` root was absent from the workspace, so the sprint did not run import, staging validation, or local cached-source dry-runs. No fixture was substituted and no fake ticker source files were created.

## Conclusion

```text
BLOCKED
```

The blocker is operational input availability, not runtime implementation.

## Implemented documentation

```text
docs/market_engine/audits/me_sr13_real_world_operator_supplied_cached_source_sample_import.md
docs/market_engine/backlog/me_sr13_real_world_operator_supplied_cached_source_sample_import_backlog_entry.md
docs/market_engine/roadmap/me_sr13_real_world_operator_supplied_cached_source_sample_import_roadmap_entry.md
```

## Follow-up candidate

* ME-SR13A - Prepare real-world operator-supplied cached-source input package for NVDA, AMD, ASML.

ME-SR13A should verify local operator input availability and ME-SR08-compatible manifest/payload layout before any rerun of real sample import, staging validation, cached-source dry-run, or downstream real cached-source analysis.

Post-ME-SR13 correction: ME-RM03 supersedes ME-SR13A as the primary next sprint. ME-SA01 is now the active next sprint, and ME-SR13A remains only a fallback/manual diagnostic candidate.
