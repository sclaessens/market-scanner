# BL105 — Fail-close selected decoupled core layer modules before archive

Status: COMPLETED

## Purpose

BL105 fail-closes manual execution surfaces for selected decoupled core layer script-era modules.

Targeted modules:

```text
scripts/core/build_context_layer.py
scripts/core/build_validation_layer.py
scripts/core/build_timing_state_layer.py
scripts/core/build_stability_layer.py
```

BL105 is a de-run / fail-close sprint. It does not archive, delete, refactor, or execute script-era runtime modules.

## Background

BL103 decoupled active tests from these script-era core layer modules.

BL104 confirmed that the modules no longer have active positive imports from the selected tests, but still contain manual execution and write-risk surfaces, including:

```text
if __name__ == "__main__"
main()
pd.read_csv(...)
to_csv(...)
mkdir(...)
fixed data/processed paths
fixed data/logs paths
```

BL105 neutralizes only the direct manual execution surfaces.

## What changed

The following modules were updated:

```text
scripts/core/build_context_layer.py
scripts/core/build_validation_layer.py
scripts/core/build_timing_state_layer.py
scripts/core/build_stability_layer.py
```

Manual entrypoints now fail closed with explicit `SystemExit` messages.

## Fail-closed entrypoints

### `scripts/core/build_context_layer.py`

Previous behavior:

```text
if __name__ == "__main__":
    build_context_layer()
```

New behavior:

```text
if __name__ == "__main__":
    raise SystemExit(
        "FAIL_CLOSED: scripts/core/build_context_layer.py is a legacy script-era module. "
        "Use the canonical market_scanner runtime instead."
    )
```

### `scripts/core/build_validation_layer.py`

Previous behavior:

```text
if __name__ == "__main__":
    build_validation_layer()
```

New behavior:

```text
if __name__ == "__main__":
    raise SystemExit(
        "FAIL_CLOSED: scripts/core/build_validation_layer.py is a legacy script-era module. "
        "Use the canonical market_scanner runtime instead."
    )
```

### `scripts/core/build_timing_state_layer.py`

Previous behavior:

```text
if __name__ == "__main__":
    build_timing_state_layer()
```

New behavior:

```text
if __name__ == "__main__":
    raise SystemExit(
        "FAIL_CLOSED: scripts/core/build_timing_state_layer.py is a legacy script-era module. "
        "Use the canonical market_scanner runtime instead."
    )
```

### `scripts/core/build_stability_layer.py`

Previous behavior:

```text
def main() -> None:
    df = build_stability_layer()
    print(f"Stability state written to: {OUTPUT_PATH}")
    print(f"Stability layer log written to: {LOG_PATH}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
```

New behavior:

```text
def main() -> None:
    raise SystemExit(
        "FAIL_CLOSED: scripts/core/build_stability_layer.py is a legacy script-era module. "
        "Use the canonical market_scanner runtime instead."
    )


if __name__ == "__main__":
    main()
```

## Side-effect status after BL105

BL105 does not remove internal historical read/write behavior from the modules.

The following markers still exist inside the functions:

```text
Path(...)
pd.read_csv(...)
mkdir(...)
to_csv(...)
```

Interpretation:

* manual CLI-style execution is now blocked;
* historical function bodies are preserved;
* archive-readiness is improved but not automatically complete;
* a later sprint must decide whether these modules may now be archived as a controlled cluster.

## Validation

Focused suite:

```bash
pytest tests/core/test_build_context_layer.py \
       tests/core/test_build_validation_layer.py \
       tests/core/test_entry_quality.py \
       tests/core/test_build_timing_state_layer.py \
       tests/core/test_build_stability_layer.py \
       tests/test_operator_visibility.py -q
```

Result:

```text
35 passed in 0.07s
```

Full suite:

```bash
pytest -q
```

Result:

```text
610 passed in 0.64s
```

## Decision

BL105 decision:

```text
SELECTED_DECOUPLED_CORE_LAYER_MANUAL_ENTRYPOINTS_FAIL_CLOSED
```

The selected decoupled core layer modules can no longer be executed directly through their script-era manual entrypoints.

## Remaining archive-readiness status

The four selected modules are better isolated, but still require a final archive-readiness review before moving them to `archive/legacy_runtime`.

Remaining concerns:

* fixed production-like data paths remain inside function bodies;
* CSV read/write behavior remains inside function bodies;
* the modules still represent historical runtime implementations;
* broader `scripts/core` still contains active positive imports outside BL105 scope.

Remaining positive `scripts.core` imports outside BL105 scope:

```text
tests/core/test_build_entry_quality_backfill.py
tests/core/test_build_context_backfill.py
tests/core/test_decision_engine.py
tests/core/test_build_portfolio_intelligence.py
tests/portfolio/test_portfolio_source_contract.py
```

## Recommended next sprint

Recommended next sprint:

```text
BL106 — Final archive-readiness review for fail-closed core layer modules
```

Candidate modules:

```text
scripts/core/build_context_layer.py
scripts/core/build_validation_layer.py
scripts/core/build_timing_state_layer.py
scripts/core/build_stability_layer.py
```

BL106 should determine whether these modules can be archived as a controlled cluster.

BL106 should not touch:

```text
scripts/core/decision_engine.py
scripts/core/data_fetcher.py
scripts/core/scanner.py
scripts/core/build_portfolio_intelligence.py
scripts/portfolio/*
scripts/watchlist/*
scripts/validate_scans.py
```

## Guardrails

* No live provider calls were run.
* No yfinance calls were run.
* No SEC/EDGAR calls were run.
* No credentials were read.
* No production data was written.
* No production reports were generated.
* No Telegram messages were sent.
* No portfolio/watchlist state was modified.
* No Decision Engine authority was changed.
* No script-era runtime module was archived.
* No script-era runtime module was executed directly.
* Only manual entrypoint fail-close behavior was changed.
