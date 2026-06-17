# ME-RUN10 - Cached-source local execution implementation audit

## Status

COMPLETED BY ME-RUN10

## Audited sprint

ME-RUN10 - Implement cached-source end-to-end local execution path

## Job family

ME-RUN - Run / orchestration jobs

## Audit scope

This audit reviews the ME-RUN10 implementation against the ME-RUN09 cached-source end-to-end local execution contract.

## Files changed

Runtime:

```text
src/market_engine/run/cached_source_execution.py
src/market_engine/run/end_to_end_dry_run.py
src/market_engine/run/end_to_end_dry_run_command.py
src/market_engine/run/local_dry_run_artifacts.py
src/market_engine/run/__init__.py
```

Tests:

```text
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
```

Documentation:

```text
docs/market_engine/run/me_run10_cached_source_local_execution_implementation.md
docs/market_engine/audits/me_run10_cached_source_local_execution_implementation_audit.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/backlog/me_run10_cached_source_local_execution_backlog_entry.md
docs/market_engine/roadmap/market_engine_roadmap.md
docs/market_engine/roadmap/me_run10_cached_source_local_execution_roadmap_entry.md
```

## Implementation result

PASS.

ME-RUN10 implements `cached_source_snapshot` as a local dry-run input mode.

The implementation consumes already-existing cached SEC CompanyFacts source snapshots and explicitly supplied local portfolio context. It builds existing Market Engine downstream contract payloads through the current approved builders, then emits the existing `market-engine-end-to-end-dry-run-v1` dry-run payload.

## Contracts preserved

Input mode:

```text
cached_source_snapshot
```

Wrapper input contract:

```text
market-engine-cached-source-local-execution-input-v1
```

Final output contract:

```text
market-engine-end-to-end-dry-run-v1
```

Optional artifact contracts:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

## Provenance behavior

The generated dry-run payload preserves cached-source provenance including:

* cached source input mode;
* source snapshot path;
* source snapshot reference relative to the configured root;
* source snapshot root;
* wrapper input format version where applicable;
* source refresh snapshot id;
* source refresh fetched timestamp;
* downstream run identifiers.

## Fail-closed behavior

ME-RUN10 fails closed for:

* missing cached source snapshot;
* malformed cached source snapshot JSON;
* cached source path outside the configured source root;
* unsupported cached-source wrapper contract;
* missing wrapper non-production marker;
* malformed portfolio context input;
* missing required portfolio context identity fields;
* downstream contract construction errors;
* local artifact persistence errors.

## Numeric-zero behavior

Numeric zero values from local portfolio context remain visible in the final dry-run `numeric_zero_evidence_summary` and in local artifact payloads when artifact writing is explicitly requested.

## Artifact behavior

Artifact writing remains opt-in only through:

```text
--write-local-artifact
```

No generated artifacts were committed.

Tests write artifacts only under temporary directories.

## Tests added

Added:

```text
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
```

Coverage includes successful cached-source local execution, missing cached source failure, malformed cached source failure, live-provider mode rejection, explicit-only artifact writing, provenance preservation, wrapper input support, source path containment, numeric-zero preservation, and import guardrails.

## Validation performed

Validation commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/run/test_me_run10_cached_source_local_execution.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/run -q
```

Initial focused validation after implementation:

```text
8 passed
57 passed
```

Final validation is recorded in the sprint completion report.

## Boundaries preserved

ME-RUN10 did not introduce:

* provider calls;
* SEC/EDGAR live calls;
* yfinance calls;
* live market data calls;
* broker calls;
* Telegram delivery;
* email delivery;
* production report generation;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* all-ticker production runs;
* automatic cache refresh;
* automatic cache cleanup;
* Decision Engine decisions;
* BUY / SELL / HOLD semantics;
* allocation advice;
* target prices;
* target weights;
* position sizing;
* order generation;
* execution advice;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability.

## Conclusion

ME-RUN10 satisfies the ME-RUN09 implementation requirements. Market Engine can now execute the local dry-run chain from an already-existing cached SEC CompanyFacts snapshot and optional local portfolio context, preserving provenance, missing/stale markers, numeric-zero evidence, opt-in artifacts, and non-actionable boundaries.

Recommended next sprint:

```text
ME-RUN11 - Run cached-source local execution against a broader deterministic ticker bundle
```
