# BL71 — Archive first low-risk script-era Python utility

## Status

Done.

## Purpose

Archive the first narrow low-risk script-era Python cleanup batch from the canonical Python cleanup registry.

This sprint starts real Python cleanup execution without restarting broad inventory work.

## Scope

This is a narrow cleanup implementation sprint.

Only one script-era Python file is moved from the active `scripts/` path into the legacy runtime archive.

## Selected file

Archived file:

- `scripts/utils/utils.py`

Archive destination:

- `archive/legacy_runtime/scripts/utils/utils.py`

## Registry basis

BL71 uses the canonical cleanup registry locked by BL70.

The existing cleanup registry classified `scripts/utils/utils.py` as a low-priority archive candidate:

- utility file;
- no active reference found;
- no runnable entrypoint;
- generic text/CSV write helpers;
- migrate only if still needed;
- otherwise archive candidate.

## Delta check

Before archival, BL71 performed a focused delta check for active references in:

- `src/`
- `tests/`
- `.github/`

Result:

- no active references found.

## Validation

Validation result:

- `pytest -q`: 501 passed

## Guardrails

BL71 did not change:

- canonical runtime behavior;
- provider behavior;
- SEC CompanyFacts behavior;
- production data writes;
- reports;
- Telegram delivery;
- portfolio/watchlist logic;
- Decision Engine authority;
- active workflows.

## BL71 conclusion

BL71 starts real Python cleanup execution by archiving one confirmed low-risk script-era utility file.

Future cleanup batches must remain small and must continue to work from the BL70 canonical cleanup registry instead of restarting broad Python inventory work.
