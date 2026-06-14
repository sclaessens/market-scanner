# BL72 — Archive legacy reporting markdown reporter

## Status

Done.

## Purpose

Archive one narrow P3 script-era reporting cleanup candidate from the BL70 canonical Python cleanup registry.

This sprint continues controlled Python cleanup execution without restarting broad inventory work.

## Scope

This is a narrow cleanup implementation sprint.

Only one script-era reporting file is moved from the active `scripts/` path into the legacy runtime archive.

## Selected file

Archived file:

- `scripts/reporting/reporter.py`

Archive destination:

- `archive/legacy_runtime/scripts/reporting/reporter.py`

## Registry basis

BL72 uses the canonical cleanup registry locked by BL70.

The existing cleanup registry classified `scripts/reporting/reporter.py` as:

- `ARCHIVE_CANDIDATE_NOW`;
- reporting / messaging / duplicate responsibility;
- no active reference found;
- no runnable entrypoint;
- low direct risk;
- old legacy markdown report formatter;
- archive-only candidate.

## Delta check

Before archival, BL72 performed a focused delta check for active references in:

- `src/`
- `tests/`
- `.github/`

The delta check remained narrow and did not repeat the broad Python inventory.

## Action taken

`scripts/reporting/reporter.py` was moved out of the active `scripts/` path and into the legacy runtime archive.

No reporting behavior was migrated.

No replacement file was created.

No runtime caller was changed.

No report generation was executed.

## Guardrails

BL72 did not change:

- canonical runtime behavior;
- provider behavior;
- SEC CompanyFacts behavior;
- production data writes;
- report generation behavior;
- Telegram delivery;
- portfolio/watchlist logic;
- Decision Engine authority;
- active workflows.

## Validation

Validation required before merge:

- focused delta reference check;
- active pytest suite.

Validation result:

- focused delta reference check: no blocking active references found;
- `pytest -q`: to be completed before merge.

## BL72 conclusion

BL72 archives one confirmed low-risk legacy reporting file.

Future cleanup batches must remain small and continue to work from the BL70 canonical cleanup registry instead of restarting broad Python inventory work.
