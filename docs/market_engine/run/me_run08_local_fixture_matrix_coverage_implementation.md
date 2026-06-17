# ME-RUN08 — Local fixture matrix coverage implementation

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: IMPLEMENTED BY ME-RUN08

## Goal

Expand local non-production dry-run fixture coverage from the single ME-RUN07 realistic fixture into a deterministic matrix that exercises multiple dry-run states through the existing `local_snapshot_fixture` input mode.

## Scope

ME-RUN08 is limited to local fixtures, local tests, command documentation, and audit documentation.

ME-RUN08 does not add provider calls, live data access, broker calls, message delivery, portfolio writes, watchlist writes, production artifacts, new financial logic, action semantics, allocation behavior, order generation, ranking, scoring, urgency, conviction, or tradeability authority.

## Implemented coverage

Implemented test module:

```text
tests/market_engine/run/test_me_run08_local_fixture_matrix_coverage.py
```

The matrix creates local non-production fixture wrappers using:

```text
market-engine-local-dry-run-input-fixture-v1
```

Every case runs through:

```text
--input-mode local_snapshot_fixture
```

The matrix covers:

| Case | Expected dry-run state | Purpose |
| --- | --- | --- |
| completed | `dry_run_completed` | clean completed local fixture path |
| completed-with-limitations | `dry_run_completed_with_limitations` | combined missing and stale marker propagation |
| blocked | `dry_run_blocked` | explicit upstream blocked stage handling |
| stale-data | `dry_run_completed_with_limitations` | stale marker propagation without blocking |
| missing-data | `dry_run_completed_with_limitations` | missing marker propagation without blocking |
| numeric-zero | `dry_run_completed` | numeric zero preservation across observation and portfolio context paths |
| unsupported-input | `dry_run_unsupported_input` | unsupported stage contract version handling |
| provenance-heavy | `dry_run_completed` | provenance collection across multiple stages |

## Artifact behavior

ME-RUN08 adds a targeted assertion that local artifact writing remains disabled unless `--write-local-artifact` is explicitly passed.

The explicit artifact case verifies:

* manifest format version;
* artifact format version;
* non-production artifact marker;
* `local_snapshot_fixture` source input mode;
* source run state;
* numeric-zero preservation inside the persisted artifact payload.

## Local validation command

Run locally from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/run/test_me_run08_local_fixture_matrix_coverage.py -q | tee /dev/tty | pbcopy
```

Recommended broader RUN validation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/run -q | tee /dev/tty | pbcopy
```

Recommended full Market Engine validation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine -q | tee /dev/tty | pbcopy
```

## Boundary conclusion

ME-RUN08 expands deterministic local dry-run review confidence without widening runtime authority. The sprint keeps fixture input local, non-production, deterministic, and explicitly opt-in for artifact writing.
