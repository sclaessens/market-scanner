# ME-DATA05 Incremental Market Data Refresh and Forward Evaluation Audit

Status: `incremental_refresh_operational`

## Executive Summary

ME-DATA05 adds a safe incremental price-history refresh command for the canonical Market Engine dataset. The command reads the canonical universe, inspects existing local `data/processed` CSV histories, requests only stale recent windows with a configurable overlap, merges provider data into the canonical CSV schema, validates before atomic replacement, refreshes coverage, and runs the existing ME-EVAL02 unresolved-outcome refresh.

Two same-cutoff operational runs were executed against the local 952-instrument canonical dataset. Both runs completed with no file rewrites, no failed downloads, no merge failures, no validation failures, no duplicate-date regressions, and automatic coverage and ME-EVAL02 artifacts. The provider returned overlap-only data for `BLD` and `JHG`, so those tickers remain `stale_after_update`. Four newer listings remain `insufficient_history`. No synthetic forward data was used and ME-EVAL02 correctly left all 12 selected outcomes unresolved.

## Baseline

Baseline source: ME-DATA04 completed canonical local market dataset.

Baseline coverage before ME-DATA05 run 1:

| Metric | Value |
|---|---:|
| Total canonical instruments | 952 |
| Valid histories | 946 |
| Insufficient histories | 6 |
| Missing histories | 0 |
| Invalid histories | 0 |
| Unsupported mappings | 0 |
| ME-EVAL02 selected outcomes | 12 |
| ME-EVAL02 resolved outcomes | 0 |
| ME-EVAL02 remaining outcomes | 12 |
| ME-EVAL02 blocker reason | `insufficient_forward_data: 12` |

## Existing ME-DATA04 Limitations

ME-DATA04 proved canonical universe construction, symbol mapping, local price-history storage, validation, coverage reporting, and ME-EVAL02 local-history consumption. Its operational limitation was refresh repeatability: a regular operator run still needed a safe path that avoids full re-downloads for already valid histories and that records per-ticker update status.

## Incremental Update Design

New implementation:

```text
src/market_engine/data/price_history_loader.py
src/market_engine/data/incremental_market_data_refresh.py
```

The refresh command derives the local CSV path from the canonical instrument `source_symbol`, validates the current file, reads the local end date, and chooses one of the explicit per-ticker statuses. Existing valid histories that already reach the cutoff are not downloaded or rewritten. Existing stale histories request `local_end_date - overlap_calendar_days` through `cutoff_date + 1 day`, allowing recent provider corrections without a full historical download.

Missing histories and corrupt or incompatible histories use the full historical start date as an explicit initialization or rebuild path. Current but short histories are reported as `insufficient_history` without noisy repeated downloads.

## Cutoff Policy

Default cutoff is the previous fully closed weekday. Explicit `--cutoff-date YYYY-MM-DD` is supported for reproducible runs. The sample and idempotency runs used:

```text
cutoff_date: 2026-07-10
cutoff_reason: explicit operator override
```

Weekend handling is covered by unit tests. Broader market holiday support remains conservative and should be revisited if non-US instruments become operationally material.

## Overlap Policy

The command exposes:

```text
--overlap-calendar-days
```

The operational runs used:

```text
overlap_calendar_days: 7
```

Provider rows inside the overlap replace local rows for the same date during merge. Duplicate dates are removed before validation.

## Atomic Write Policy

The command writes merged data to a temporary file in the destination directory, validates the temporary file, and only then uses atomic replacement. Download, merge, validation, and write failures return per-ticker failure status and preserve the existing file. Tests cover download failure and validation failure preserving the original file byte-for-byte.

## Full Rebuild Policy

Full rebuild is not the normal path. It is reserved for missing snapshots and invalid, corrupt, or incompatible local CSVs. The rebuild path is covered by tests and reports `new_snapshot_created`, `full_rebuild_completed`, or a clear failure status.

## Per-Ticker Statuses

Supported statuses:

```text
already_current
incrementally_updated
new_snapshot_created
full_rebuild_required
full_rebuild_completed
download_failed
empty_provider_response
merge_failed
validation_failed
stale_after_update
insufficient_history
unsupported_mapping
```

Run 1 status counts:

| Status | Count |
|---|---:|
| `already_current` | 946 |
| `stale_after_update` | 2 |
| `insufficient_history` | 4 |
| Other failure statuses | 0 |

Run 1 blockers:

| Symbol | Status | Explanation |
|---|---|---|
| `BLD` | `stale_after_update` | Provider returned overlap-only data through the existing local end date. |
| `JHG` | `stale_after_update` | Provider returned overlap-only data through the existing local end date. |
| `FDXF` | `insufficient_history` | Current to cutoff but below minimum history rows. |
| `HONA` | `insufficient_history` | Current to cutoff but below minimum history rows. |
| `Q` | `insufficient_history` | Current to cutoff but below minimum history rows. |
| `SOLS` | `insufficient_history` | Current to cutoff but below minimum history rows. |

## Idempotency Run

Run 1:

```text
artifacts/market_engine/data_runs/me-data05-incremental-refresh-20260713T140000Z/
```

Run 2:

```text
artifacts/market_engine/data_runs/me-data05-idempotency-refresh-20260713T141000Z/
```

| Metric | Run 1 | Run 2 | Acceptance | Result |
|---|---:|---:|---|---|
| Histories checked | 952 | 952 | canonical universe | pass |
| Incrementally updated | 0 | 0 | greater than zero where provider has new rows | pass with data-availability exception |
| Already current | 946 | 946 | unchanged set stable | pass |
| Files rewritten | 0 | 0 | 0 in run 2 | pass |
| Rows added | 0 | 0 | no synthetic rows | pass |
| Duplicate dates | 0 | 0 | 0 | pass |
| Existing files damaged on failure | 0 | 0 | 0 | pass |
| Coverage | 946 valid / 6 insufficient | 946 valid / 6 insufficient | no unexplained regression | pass |
| ME-EVAL02 executed | yes | yes | yes | pass |

The strict "incrementally updated greater than zero" criterion could not be demonstrated with the real provider at the chosen cutoff because provider data for the only stale valid histories did not extend beyond the existing local dates. The sprint did not fabricate rows or force resolved outcomes. The incremental append, overlap replacement, new snapshot, and rebuild behavior are covered by deterministic tests.

## Coverage Before and After

Run 1 coverage:

| Metric | Before | After |
|---|---:|---:|
| Total canonical instruments | 952 | 952 |
| Valid histories | 946 | 946 |
| Insufficient histories | 6 | 6 |
| Missing histories | 0 | 0 |
| Invalid histories | 0 | 0 |
| Unsupported mappings | 0 | 0 |
| Completion status | `completed_with_blockers` | `completed_with_blockers` |

Run 2 reproduced the same coverage.

## ME-EVAL02 Before and After

Run 1 ME-EVAL02 refresh:

| Metric | Before | After |
|---|---:|---:|
| Selected outcomes | 12 | 12 |
| Resolved | 0 | 0 |
| Still unresolved | 12 | 12 |
| Newly resolved | 0 | 0 |
| Missing price history blockers | 0 | 0 |
| Insufficient forward data blockers | 12 | 12 |

Run 2 reproduced the same ME-EVAL02 result. This is expected because the advice date and required forward horizons still do not have enough real future trading data.

## Analysis Loader Boundary

ME-DATA05 adds `price_history_loader.py` as the narrow canonical instrument to local history metadata boundary. It centralizes local path derivation and exposes validation/freshness metadata without introducing provider behavior into analysis consumers.

## Failures and Exceptions

The first sandboxed provider attempt could not resolve Yahoo Finance DNS and produced empty provider responses. The same command was rerun with approved network access. Existing files were preserved and the final committed artifacts are from the successful network runs.

The real provider did not supply new rows for `BLD` and `JHG`; both remain explicit `stale_after_update` blockers. Four short but current histories remain `insufficient_history`. These are data-availability limitations, not implementation failures.

## Operator Commands

Run 1:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.data.incremental_market_data_refresh \
  --run-id me-data05-incremental-refresh-20260713T140000Z \
  --cutoff-date 2026-07-10 \
  --overlap-calendar-days 7 \
  --refresh-prices \
  --run-coverage \
  --run-evaluation
```

Run 2:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.data.incremental_market_data_refresh \
  --run-id me-data05-idempotency-refresh-20260713T141000Z \
  --cutoff-date 2026-07-10 \
  --overlap-calendar-days 7 \
  --refresh-prices \
  --run-coverage \
  --run-evaluation
```

## Tests

Executed before commit:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_incremental_market_data_refresh.py tests/market_engine/data/test_complete_local_market_dataset.py tests/market_engine/data/test_local_market_data_universe.py -q
```

Additional full verification is recorded in the PR summary.

## Changed Files

Runtime:

```text
src/market_engine/data/price_history_loader.py
src/market_engine/data/incremental_market_data_refresh.py
```

Tests:

```text
tests/market_engine/data/test_incremental_market_data_refresh.py
```

Documentation:

```text
docs/market_engine/audits/me_data05_incremental_market_data_refresh_and_forward_evaluation_audit.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/market_engine_roadmap.md
docs/market_engine/roadmap/ACTIVE_BASELINE_DIRECTION.md
```

Artifacts:

```text
artifacts/market_engine/data_runs/me-data05-incremental-refresh-20260713T140000Z/
artifacts/market_engine/data_runs/me-data05-idempotency-refresh-20260713T141000Z/
artifacts/market_engine/evaluation_refresh_runs/me-data05-incremental-refresh-20260713T140000Z-evaluation-before/
artifacts/market_engine/evaluation_refresh_runs/me-data05-incremental-refresh-20260713T140000Z-evaluation-after/
artifacts/market_engine/evaluation_refresh_runs/me-data05-idempotency-refresh-20260713T141000Z-evaluation-before/
artifacts/market_engine/evaluation_refresh_runs/me-data05-idempotency-refresh-20260713T141000Z-evaluation-after/
```

Auxiliary full coverage run directories were generated locally and are
referenced by the ME-DATA05 manifests, but they are intentionally not committed
because the primary ME-DATA05 run directories already contain the compact
`coverage_before.json` and `coverage_after.json` artifacts required for audit.

## Acceptance Criteria

| Criterion | Result | Evidence |
|---|---|---|
| Incremental command exists | pass | `market_engine.data.incremental_market_data_refresh` |
| Existing valid current histories avoid full re-download | pass | 946 `already_current`, provider not called in tests |
| Overlap merge and provider correction covered | pass | unit tests |
| Atomic write and failure preservation covered | pass | unit tests |
| Per-ticker statuses emitted | pass | `per_ticker_status.json` |
| Coverage runs automatically | pass | coverage artifacts before and after |
| ME-EVAL02 runs automatically | pass | evaluation refresh artifacts before and after |
| Idempotent same-cutoff run | pass | run 2 `files_rewritten: 0` |
| No synthetic forward data | pass | `newly_resolved: 0`, blockers retained |
| Strict real incremental update in run 1 | pass with data-availability exception | provider had no new rows for stale histories |

## Remaining Risks

Market holiday cutoff handling is still conservative and weekday-based. Provider availability and delisting behavior can leave stale histories unresolved. Short new listings can remain `insufficient_history` until enough real rows accumulate. ME-EVAL02 remains unresolved until enough post-advice trading days exist.

## Recommended Next Sprint

ME-ANALYSIS01 - Broad canonical-universe analysis execution and reporting over the now-operational local market dataset. This should shift the baseline from more data infrastructure toward using the 952-instrument dataset for broad Market Engine analysis, while preserving the established local refresh command for routine updates.
