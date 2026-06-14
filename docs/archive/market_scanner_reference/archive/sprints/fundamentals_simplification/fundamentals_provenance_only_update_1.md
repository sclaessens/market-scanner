# Fundamentals Provenance-Only Update 1

## Status and Scope

This document records the first governed provenance-only fundamentals source-data update for the 15 tickers selected in `docs/sprints/fundamentals_source_data_expansion_preview_1.md`.

This was a local ignored source-data update task. It was not a coding sprint, runtime-logic change, automated ingestion implementation, provider integration, scraping task, credential task, or numerical fundamentals sourcing task.

No numerical fundamentals were collected.

No numerical fundamentals were invented.

No provider APIs, paid or restricted APIs, scraping, credentials, or secrets were used.

No Decision Engine, Reporting, Telegram, scanner, Fundamental Layer, Portfolio Intelligence, watchlist, portfolio source CSV, or runtime logic changes were made.

## Protocol Reference

This update follows `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`.

The protocol classifies `data/raw/fundamentals.csv` as local ignored source data. It may be edited locally only when explicitly approved, must remain ignored and untracked, and must not be committed.

## Preview Artifact Reference

Approved preview artifact:

- `docs/sprints/fundamentals_source_data_expansion_preview_1.md`

The preview selected 15 metadata-complete tickers for a provenance-only fundamentals follow-up.

## Selected Batch

Selected tickers:

- `AMAT`
- `ANET`
- `ASML`
- `COST`
- `DELL`
- `ENPH`
- `EOG`
- `EQIX`
- `EW`
- `EXPD`
- `FDX`
- `FTNT`
- `HAL`
- `HLT`
- `HPE`

No ticker outside this selected batch was added or updated by this task.

## Local Ignored Source-Data Class

Updated local ignored source artifact:

- `data/raw/fundamentals.csv`

This file is ignored by `.gitignore` through `data/raw/`.

It was modified locally for validation only and was not committed.

## Backup

Backup created before editing:

- `data/raw/fundamentals_backup_before_provenance_update_1.csv`

Backup row count: 21 data rows

The backup is also ignored through `data/raw/` and was not committed.

## Raw Fundamentals Update Summary

Raw fundamentals row count before update: 21

Raw fundamentals row count after update: 36

Rows added locally: 15

Rows updated locally: 0

Existing rows were not modified.

The selected rows were added with provenance-only values:

- `source_name`: `provenance_only_manual_seed`
- `source_reference`: `Automated Source Data Steward provenance-only batch 1`
- `source_freshness_date`: `2026-05-24`
- `as_of_date`: `2026-05-24`
- `fundamental_notes`: `Provenance-only placeholder; numerical fundamentals not yet sourced.`

All numerical metric columns were left blank for the selected rows.

The optional `currency` column was also left blank because this task did not authorize additional fundamentals source lookup or value collection.

## Raw File Validation

Local raw file validation confirmed:

- `data/raw/fundamentals.csv` exists.
- Required MVP columns are present.
- Selected 15 tickers each have exactly one raw fundamentals row.
- No duplicate ticker rows were introduced.
- Selected rows have provenance fields populated.
- Selected rows have blank numerical metric fields.
- Selected rows have valid dates that are not later than the local task date.
- No forbidden decision semantics were found in the selected rows.
- `data/raw/fundamentals.csv` remains ignored and untracked.
- The backup file remains ignored and untracked.

## Fundamental Layer Validation Results

Repository command run:

```bash
PYTHONPATH=. .venv/bin/python scripts/core/build_fundamental_layer.py
```

Result:

- Command succeeded.
- `data/processed/fundamental_quality.csv` was written.
- `data/logs/fundamental_layer_log.csv` was written.

Repository-output limitation:

- The current local generated `data/processed/context_strength.csv` contained only six older ticker/date rows at the time this command was run.
- As a result, the direct repository output contained 6 rows and did not include the selected 15 tickers.
- This means the direct command confirmed the builder still runs, but did not validate the selected batch in the current local generated context.

Repository-output distribution from the direct command:

| field | distribution |
|---|---|
| `quality_state` | `PARTIAL_DATA`: 2, `SUFFICIENT_DATA`: 4 |
| `quality_metadata_status` | `complete`: 4, `partial`: 2 |
| `source_data_status` | `partial_data`: 2, `source_available`: 4 |

Focused selected-batch validation:

The existing Fundamental Layer builder was also run against temporary context and output paths for the 15 selected tickers dated `2026-05-24`. This avoided provider/API calls, full pipeline execution, and repository generated-output writes while exercising the current Fundamental Layer source contract against the local ignored raw fundamentals file.

Focused selected-batch result:

| field | distribution |
|---|---|
| `quality_state` | `INSUFFICIENT_DATA`: 15 |
| `quality_metadata_status` | `partial`: 15 |
| `source_data_status` | `partial_data`: 15 |

Interpretation:

- The selected rows moved away from pure `row_missing` / source-missing behavior in the focused source-contract validation.
- They remain `INSUFFICIENT_DATA` because all numerical fundamentals metric fields are intentionally blank.
- This is the expected outcome for a provenance-only update.

## Selected Batch Post-Validation Status

| ticker | raw_row_present | provenance_present | numerical_metrics_present | quality_state_after_validation | source_data_status_after_validation | validation_state |
|---|---|---|---|---|---|---|
| AMAT | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| ANET | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| ASML | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| COST | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| DELL | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| ENPH | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| EOG | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| EQIX | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| EW | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| EXPD | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| FDX | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| FTNT | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| HAL | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| HLT | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |
| HPE | yes | yes | no | INSUFFICIENT_DATA | partial_data | VALIDATED_PROVENANCE_ONLY |

## Generated Artifact Handling

The direct Fundamental Layer builder wrote ignored generated artifacts:

- `data/processed/fundamental_quality.csv`
- `data/logs/fundamental_layer_log.csv`

These files are ignored by repository policy and were not committed.

No tracked generated artifacts were changed.

The focused selected-batch validation wrote only to temporary paths outside the repository and did not create repository generated artifacts.

The full pipeline was not run.

## Git and Ignored Status Confirmation

`data/raw/fundamentals.csv` remains ignored through `data/raw/`.

`data/raw/fundamentals_backup_before_provenance_update_1.csv` remains ignored through `data/raw/`.

Neither the raw fundamentals file nor the backup file was staged or committed.

Only this documentation report is intended for commit.

## Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Existing backlog coverage remains sufficient:

- `BL-0015 — Define and implement approved Fundamental data source and quality classification contract`
- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- `BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract`
- `BL-0011 — Define and repair authoritative active portfolio source`

## Recommended Next Step

If the objective is to move the selected tickers beyond provenance-only `INSUFFICIENT_DATA`, launch a stricter governed numerical fundamentals source lookup task.

That later task must explicitly approve the source method, metric definitions, reporting periods, currency and unit conventions, freshness rules, parseability rules, and validation handling for partial or contradictory values.
