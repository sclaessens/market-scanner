# BL73 — Archive remaining low-risk script-era cleanup candidates

## Status

Done.

## Purpose

Archive the next small batch of low-risk script-era Python cleanup candidates from the BL70 canonical Python cleanup registry.

This sprint continues controlled Python cleanup execution without restarting broad inventory work.

## Scope

This is a narrow cleanup implementation sprint.

Only confirmed low-risk script-era files are moved from the active `scripts/` path into the legacy runtime archive.

## Selected files

Archived files:

- `scripts/core/regime.py`
- `scripts/portfolio/test_portfolio.py`

Archive destinations:

- `archive/legacy_runtime/scripts/core/regime.py`
- `archive/legacy_runtime/scripts/portfolio/test_portfolio.py`

## Registry basis

BL73 uses the canonical cleanup registry locked by BL70.

The existing cleanup registry classified:

- `scripts/core/regime.py` as a P3 archive candidate after migration or confirmation, with no active test import found, no runnable entrypoint, and low direct write risk.
- `scripts/portfolio/test_portfolio.py` as a P4 delete/archive candidate after confirmation that it is not part of the active test suite or operator procedure.

BL73 archives instead of deleting.

## Blocked candidate carried forward

The earlier BL72 candidate `scripts/fundamentals/__init__.py` remains blocked because active tests still import `scripts.fundamentals`.

It must not be archived until those imports are retired or migrated.

## Delta checks

Before archival, BL73 performed focused delta checks for active references in:

- `src/`
- `tests/`
- `.github/`

The delta checks remained narrow and did not repeat the broad Python inventory.

## Action taken

The selected files were moved out of the active `scripts/` path and into the legacy runtime archive.

No domain logic was migrated.

No replacement files were created.

No runtime callers were changed.

No portfolio mutation was executed.

No report generation, provider calls, Telegram delivery, or production pipeline behavior was executed.

## Guardrails

BL73 did not change:

- canonical runtime behavior;
- scanner behavior;
- provider behavior;
- SEC CompanyFacts behavior;
- production data writes;
- reports;
- Telegram delivery;
- portfolio/watchlist runtime logic;
- Decision Engine authority;
- active workflows.

## Validation

Validation required before merge:

- focused delta reference checks;
- active pytest suite.

Validation result:

- focused delta reference checks: no blocking active references found;
- `pytest -q`: to be completed before merge.

## BL73 conclusion

BL73 archives the next small low-risk script-era cleanup batch.

Future cleanup batches must remain small and continue to work from the BL70 canonical cleanup registry instead of restarting broad Python inventory work.
