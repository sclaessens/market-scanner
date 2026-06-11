# BL97 — Decouple active data-source tests from script-era data_sources modules

Status: COMPLETED

## Purpose

BL97 decouples active data-source tests from script-era `scripts/data_sources` modules.

Targeted script-era modules:

```text id="t7x2oh"
scripts/data_sources/common.py
scripts/data_sources/prefill_fundamentals.py
scripts/data_sources/prefill_portfolio_metadata.py
```

BL97 is a decoupling sprint. It does not archive, delete, execute, or refactor the targeted script-era runtime modules.

## Scope

Updated active tests so they no longer import script-era data-source modules directly.

Changed files:

```text id="b343h9"
tests/conftest.py
tests/data_sources/test_prefill_common.py
tests/data_sources/test_prefill_fundamentals.py
tests/data_sources/test_prefill_portfolio_metadata.py
tests/test_operator_visibility.py
```

Out of scope:

* archiving `scripts/data_sources/*.py`;
* editing `scripts/data_sources/*.py`;
* executing script-era data-source modules;
* provider calls;
* production data writes;
* production report generation;
* Telegram delivery;
* portfolio/watchlist mutation;
* Decision Engine changes.

## Pre-decoupling findings

Before BL97, the focused data-source test run failed during collection because active tests imported `scripts.data_sources` directly:

```text id="vnrqqs"
tests/data_sources/test_prefill_common.py imported scripts.data_sources.common
tests/data_sources/test_prefill_fundamentals.py imported scripts.data_sources.prefill_fundamentals
tests/data_sources/test_prefill_portfolio_metadata.py imported scripts.data_sources.prefill_portfolio_metadata
```

Focused collection failed with:

```text id="xkgify"
ModuleNotFoundError: No module named 'scripts'
```

The full suite still passed because these tests were excluded from normal active collection through existing blocker handling.

## What changed

### Data-source tests decoupled

BL97 replaced direct script-era import tests with canonical/static contract tests.

Removed active imports from:

```text id="qqnj3e"
scripts.data_sources.common
scripts.data_sources.prefill_fundamentals
scripts.data_sources.prefill_portfolio_metadata
```

The tests now verify:

* active code no longer imports `scripts.data_sources`;
* canonical runtime metadata does not claim script-era data-source authority;
* data-source prefill policy remains source-data oriented;
* fundamentals prefill contract remains explicit and non-recommendation authority;
* portfolio metadata prefill contract remains bounded and non-action authority;
* dry-run/write visibility remains part of the contract.

### Operator visibility blockers updated

Because the data-source tests are now decoupled, BL97 removed them from high-risk script-era blocker registries in:

```text id="9bkxqf"
tests/conftest.py
tests/test_operator_visibility.py
```

Removed blocker entries:

```text id="o7jtg0"
data_sources/test_prefill_common.py
data_sources/test_prefill_fundamentals.py
data_sources/test_prefill_portfolio_metadata.py
```

## Active import and path check

BL97 checked for remaining active data-source script-era imports and path references:

```bash id="cxetrc"
grep -RIn \
  "from scripts.data_sources\|import scripts.data_sources\|scripts/data_sources" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result:

```text id="eyujsd"
No output
```

Interpretation:

* no active positive imports remain from `scripts.data_sources`;
* no active positive path references remain to `scripts/data_sources` in `src`, `tests`, or `.github`.

## Validation

Focused suite:

```bash id="w4jup2"
pytest tests/data_sources/test_prefill_common.py \
       tests/data_sources/test_prefill_fundamentals.py \
       tests/data_sources/test_prefill_portfolio_metadata.py \
       tests/test_operator_visibility.py -q
```

Result:

```text id="btlav0"
15 passed in 0.05s
```

Full suite:

```bash id="e3735b"
pytest -q
```

Result:

```text id="5dxwkj"
569 passed in 0.59s
```

## Decision

BL97 decision:

```text id="1ax7mo"
DATA_SOURCE_ACTIVE_TEST_DEPENDENCIES_DECOUPLED
```

The targeted script-era data-source modules are not archived yet, but active test dependency blockers have been removed.

## Remaining archive-readiness note

The targeted files still physically exist under `scripts/data_sources/`:

```text id="04uehx"
scripts/data_sources/common.py
scripts/data_sources/prefill_fundamentals.py
scripts/data_sources/prefill_portfolio_metadata.py
```

They should remain in place until a dedicated archive sprint confirms:

* no active imports;
* no active path references in `src`, `tests`, or `.github`;
* no active workflow references;
* focused and full tests pass;
* files are moved to `archive/legacy_runtime/scripts/data_sources/`.

## Recommended next sprint

Recommended next sprint:

```text id="tcrdxf"
BL98 — Archive data_sources script-era modules after final no-active-reference check
```

Candidate archive targets:

```text id="8apd4o"
scripts/data_sources/common.py
scripts/data_sources/prefill_fundamentals.py
scripts/data_sources/prefill_portfolio_metadata.py
```

BL98 must perform a final no-active-reference check before moving files to:

```text id="1q2fgw"
archive/legacy_runtime/scripts/data_sources/
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
* No script-era data-source module was executed.
* No script-era data-source module was edited.
* No script-era runtime module was archived.
