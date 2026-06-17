# ME-RUN05 - Local dry-run artifact persistence implementation

## Status

COMPLETED BY ME-RUN05

## Sprint

ME-RUN05 - Implement local dry-run artifact persistence

## Job family

ME-RUN - Run / orchestration jobs

## Purpose

ME-RUN05 implements the local non-production persistence boundary defined by ME-RUN04.

The implementation lets a local Market Engine dry-run persist an already-built `market-engine-end-to-end-dry-run-v1` payload as deterministic, inspectable JSON artifacts. It is intended for operator and developer review only.

The persistence layer does not build, enrich, repair, normalize, reinterpret, or rerun upstream Market Engine stages.

## Implemented runtime

```text
src/market_engine/run/local_dry_run_artifacts.py
src/market_engine/run/end_to_end_dry_run_command.py
src/market_engine/run/__init__.py
```

`src/market_engine/run/local_dry_run_artifacts.py` provides:

* `persist_market_engine_local_dry_run_artifact(...)`;
* `LocalDryRunArtifactError`;
* `LocalDryRunArtifactPersistenceResult`;
* local artifact and manifest format constants.

The dry-run command now supports explicit local persistence through:

```text
--write-local-artifact
--artifact-output-root
--artifact-created-at
```

Artifact writing remains disabled by default.

## Approved input

The persistence function accepts only an already-built mapping with:

```text
market-engine-end-to-end-dry-run-v1
```

Required payload identity fields:

* `dry_run_format_version`;
* `dry_run_id`;
* `input_mode`;
* `run_state`.

Unsupported payload format versions fail closed.

Missing identity fields fail closed.

Non-mapping payloads fail closed.

## Output behavior

The default approved path category remains:

```text
artifacts/market_engine/dry_runs/
```

For a dry-run id such as `dry-run-001`, the writer creates:

```text
artifacts/market_engine/dry_runs/dry-run-001/manifest.json
artifacts/market_engine/dry_runs/dry-run-001/artifacts/market_engine_dry_run_dry-run-001_<date>.json
```

The artifact file contains:

* artifact format version: `market-engine-local-dry-run-artifact-v1`;
* artifact type;
* artifact creation timestamp supplied by the caller;
* local dry-run persistence mode;
* approved path category;
* explicit non-production marker;
* source dry-run format version;
* source dry-run id;
* source dry-run generated timestamp;
* source input mode;
* source run state;
* the complete source dry-run payload.

The manifest file contains:

* manifest format version: `market-engine-local-dry-run-artifact-manifest-v1`;
* artifact count;
* local persistence metadata;
* source dry-run identity;
* relative artifact path.

## Determinism

The writer uses:

* caller-supplied artifact timestamp;
* dry-run id as a safe run directory name;
* generated date or artifact creation date in the artifact filename;
* stable JSON key ordering;
* UTF-8 human-readable JSON.

Tests inject timestamps and temporary output roots.

## Path safety

The persistence layer validates:

* output root does not contain parent traversal;
* dry-run id is a safe path segment;
* generated artifact date is a safe path segment;
* resolved child paths remain under the configured output root;
* existing run directories are not overwritten by default.

Unsafe path segments, parent-directory traversal, absolute-path escape attempts through dry-run id, and existing artifact directories fail closed.

## Serialization behavior

The writer accepts JSON-compatible mappings, lists, tuples, strings, numbers, booleans, and null values.

Unsupported Python objects fail with `LocalDryRunArtifactError`.

Numeric zero values are preserved as valid values.

Missing data, stale data, blocked states, blocked reasons, provenance, delivery report references, forbidden-side-effect confirmation, and authority-boundary confirmation remain inside the persisted payload.

## Command behavior

Default command behavior remains stdout-only:

```bash
python -m market_engine.run.end_to_end_dry_run_command
```

Local artifact writing requires explicit invocation:

```bash
python -m market_engine.run.end_to_end_dry_run_command \
  --dry-run-id local-run-001 \
  --generated-at 2026-06-17T14:00:00Z \
  --write-local-artifact \
  --artifact-output-root artifacts/market_engine/dry_runs \
  --artifact-created-at 2026-06-17T14:01:00Z
```

The command still prints the dry-run payload to stdout. Artifact persistence errors return exit code `2`.

## Implemented tests

```text
tests/market_engine/run/test_local_dry_run_artifacts.py
tests/market_engine/run/test_end_to_end_dry_run_command.py
```

Tests cover:

* valid local artifact persistence;
* readable JSON artifact and manifest content;
* contract/version metadata preservation;
* output staying under the configured dry-run root;
* overwrite refusal;
* parent-directory traversal rejection;
* absolute path escape rejection through dry-run id;
* serialization failure;
* unsupported payload contract failure;
* command default no-write behavior;
* command explicit artifact write behavior;
* import guardrails against legacy runtime and side-effect dependencies.

## Forbidden behavior preserved

ME-RUN05 does not introduce provider calls, SEC/EDGAR calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine decisions, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, BUY / SELL / HOLD semantics, allocation advice, target weights, target prices, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability.

## Outcome

ME-RUN05 implements optional local dry-run artifact persistence as non-production JSON review evidence. It makes the local runtime more inspectable while preserving all Market Engine side-effect and authority boundaries.
