# BL77 — Archive BL76 low-risk script-era Python files

Status: COMPLETED

## Purpose

BL77 archives the low-risk script-era Python files identified by BL76 and removes their active runtime-path copies.

## Registry basis

Primary basis:

- `docs/audits/legacy_runtime/bl76_remaining_script_era_python_dependency_classification.md`

BL76 classified the following files as low-risk archive candidates:

- `scripts/core/analyze_validation.py`
- `scripts/diagnostics/audit_data_coverage.py`
- `scripts/analyze_validation.py`

## Changes

Archived:

- `scripts/core/analyze_validation.py`
  - to `archive/legacy_runtime/scripts/core/analyze_validation.py`
- `scripts/diagnostics/audit_data_coverage.py`
  - to `archive/legacy_runtime/scripts/diagnostics/audit_data_coverage.py`
- `scripts/analyze_validation.py`
  - to `archive/legacy_runtime/scripts/analyze_validation.py`

Removed from active runtime paths:

- `scripts/core/analyze_validation.py`
- `scripts/diagnostics/audit_data_coverage.py`
- `scripts/analyze_validation.py`

## Validation

Run locally before merge:

```bash
pytest -q

Result:

PENDING_LOCAL_VALIDATION
Guardrails
No live SEC/EDGAR calls were run.
No yfinance calls were run.
No credentials were read.
No production data was written.
No production reports were generated.
No Telegram messages were sent.
No portfolio/watchlist production state was modified.
No Decision Engine authority was changed.
No script-era Python runtime files were executed.
Remaining blockers

None for the three BL76 low-risk candidates.


Voeg ook onderaan `docs/active/project/backlog.md` toe:

```markdown
### BL77 — Archive BL76 low-risk script-era Python files

Category: Legacy Runtime Cleanup / Repository Hygiene

Status: COMPLETED

BL77 archived the low-risk script-era Python files identified by BL76.

Archived:

- `scripts/core/analyze_validation.py`
- `scripts/diagnostics/audit_data_coverage.py`
- `scripts/analyze_validation.py`

Archive targets:

- `archive/legacy_runtime/scripts/core/analyze_validation.py`
- `archive/legacy_runtime/scripts/diagnostics/audit_data_coverage.py`
- `archive/legacy_runtime/scripts/analyze_validation.py`

Result:

- The three low-risk script-era validation/audit helpers were removed from active runtime paths.
- Historical evidence was preserved under `archive/legacy_runtime/`.

Validation:

- `pytest -q`: pending local validation before merge.

Guardrails:

- no live SEC/EDGAR calls
- no yfinance calls
- no credentials read
- no production data writes
- no reports generated
- no Telegram messages sent
- no portfolio/watchlist state modified
- no Decision Engine authority changed
- no script-era Python runtime files executed