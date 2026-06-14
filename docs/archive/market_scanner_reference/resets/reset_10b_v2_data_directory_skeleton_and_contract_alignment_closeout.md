# RESET-10B Closeout: V2 Data Directory Skeleton and Contract Alignment

## Purpose

RESET-10B aligned the repository skeleton and data contract metadata with the v2 data lifecycle introduced in `docs/active/data_lifecycle.md`.

Mandatory lifecycle separation remains:

```text
raw source data != normalized input != generated output != report
```

This step did not migrate data, implement provider ingestion, implement raw-to-normalized transformation, or create generated business outputs.

## Directories Created or Preserved

The v2 lifecycle skeleton now has visible placeholders for:

- `data/raw/.gitkeep`
- `data/normalized/.gitkeep`
- `data/generated/.gitkeep`
- `data/local/.gitkeep`

`data/fixtures/v2/` already existed and remains tracked with the approved synthetic RESET-4 fixtures:

- `data/fixtures/v2/universe_candidates.csv`
- `data/fixtures/v2/portfolio_transactions.csv`
- `data/fixtures/v2/source_data_readiness.csv`

Existing ignored local files under `data/raw/` and existing ignored local SEC cache directories under `data/local/` were not modified, moved, deleted, or staged.

## `.gitignore` Changes

`.gitignore` was updated to align with the v2 lifecycle skeleton:

- `data/local/*` is ignored except `data/local/.gitkeep`;
- `data/raw/*` is ignored except `data/raw/.gitkeep`;
- `data/generated/*` is ignored except `data/generated/.gitkeep`;
- `data/normalized/*` is ignored except `data/normalized/.gitkeep` until normalized production-like input tracking is explicitly approved.

`data/fixtures/v2/` remains trackable and is not ignored.

The relevant `.gitignore` comments were normalized to English while preserving existing ignore intent.

## Data Contract Changes

`docs/active/data_contracts.md` now includes lifecycle-stage alignment for:

- `RAW_SOURCE`
- `NORMALIZED_INPUT`
- `FIXTURE_INPUT`
- `GENERATED_OUTPUT`
- `REPORTING_OUTPUT`
- `LOCAL_ONLY`

The documentation clarifies that:

- raw source data is evidence, not program-ready input;
- normalized input is contract-compliant program input;
- fixtures are synthetic and tracked for tests;
- generated output is not source-of-truth unless explicitly approved;
- reports are communication only;
- local data is private or machine-specific and not canonical.

## Python Metadata Changes

`src/market_scanner/shared/data_contracts.py` now defines `DataLifecycleStage`.

Existing approved fixture contracts are marked as `DataLifecycleStage.FIXTURE_INPUT`.

No transformation logic, file writes, provider calls, runtime scanning, or generated-output behavior was added.

## Tests Added or Updated

`tests/contract/test_v2_data_contracts.py` was updated to verify:

- lifecycle stages exist;
- approved fixture contracts are marked `FIXTURE_INPUT`;
- fixture contracts do not approve production, generated, raw, local, legacy, watchlist, portfolio, or report paths as source-of-truth inputs.

## Scope Confirmation

No legacy data was moved, deleted, or modified.

No CSV contents were changed.

No files under these legacy/generated paths were changed:

- `data/processed/`
- `data/portfolio/`
- `data/watchlist/`
- `data/logs/`
- `reports/`

No production pipeline, `scripts/run_scan.py`, Telegram script, SEC diagnostics, provider calls, network calls, or live data calls were run.

The daily scan schedule was not re-enabled.

## Validation Commands and Results

Commands run:

```bash
.venv/bin/python -m pytest
git diff --check
git status --short
git diff --stat
git diff --summary
find data -maxdepth 2 -type d | sort
find data/raw data/normalized data/generated data/local -maxdepth 1 -type f | sort
git diff --name-only
git diff -- data/processed data/portfolio data/watchlist data/logs reports || true
```

Results:

- `.venv/bin/python -m pytest` passed: `460 passed`.
- `git diff --check` passed.
- `find data -maxdepth 2 -type d | sort` showed the v2 lifecycle directories present.
- `find data/raw data/normalized data/generated data/local -maxdepth 1 -type f | sort` showed the four `.gitkeep` placeholders plus pre-existing ignored raw local files.
- `git diff --name-only` showed only approved RESET-10B paths before staging placeholders and this closeout.
- `git diff -- data/processed data/portfolio data/watchlist data/logs reports || true` produced no diff.

## Recommended Next Action

RESET-9C - Legacy Runtime Inventory and Retirement Decision.
