# BL79 — Archive fundamentals compatibility wrappers after active-reference check

Status: COMPLETED

## Purpose

BL79 archives the remaining script-era fundamentals compatibility wrappers under `scripts/core/`.

These files only delegated to `scripts.fundamentals` modules and no longer function as approved canonical runtime owners. BL78 identified them as compatibility wrappers requiring a focused active-reference check before archival.

This sprint archives only the wrappers. It does not archive, delete, move, refactor, or execute the underlying `scripts/fundamentals/*.py` files.

## Registry basis

Primary basis:

- `docs/audits/legacy_runtime/bl76_remaining_script_era_python_dependency_classification.md`
- `docs/audits/legacy_runtime/bl78_fundamentals_script_era_migration_review.md`

Related basis:

- `docs/audits/legacy_runtime/v2_script_era_python_cleanup_inventory.md`
- `docs/audits/legacy_runtime/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/audits/legacy_runtime/v2_fundamentals_script_era_side_effect_migration_review.md`
- `docs/audits/legacy_runtime/bl74_decouple_active_tests_from_script_era_fundamentals.md`

## Archived files

Archived from active runtime paths:

- `scripts/core/build_fundamental_analysis.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamentals_history_intake.py`

Archive targets:

- `archive/legacy_runtime/scripts/core/build_fundamental_analysis.py`
- `archive/legacy_runtime/scripts/core/build_fundamental_layer.py`
- `archive/legacy_runtime/scripts/core/build_fundamental_metrics.py`
- `archive/legacy_runtime/scripts/core/build_fundamentals_history_intake.py`

## Active-reference check

Focused path-reference command:

```bash
grep -R "scripts/core/build_fundamental_analysis.py\|scripts/core/build_fundamental_layer.py\|scripts/core/build_fundamental_metrics.py\|scripts/core/build_fundamentals_history_intake.py" -n . \
  --include="*.py" \
  --include="*.md" \
  --include="*.yml" \
  --include="*.yaml" \
  --exclude-dir=".git" \
  --exclude-dir=".venv" \
  --exclude-dir="archive"

Focused module-reference command:

grep -R "scripts.core.build_fundamental_analysis\|scripts.core.build_fundamental_layer\|scripts.core.build_fundamental_metrics\|scripts.core.build_fundamentals_history_intake" -n . \
  --include="*.py" \
  --include="*.md" \
  --include="*.yml" \
  --include="*.yaml" \
  --exclude-dir=".git" \
  --exclude-dir=".venv" \
  --exclude-dir="archive"

Broad symbol-reference command:

grep -R "build_fundamental_analysis\|build_fundamental_layer\|build_fundamental_metrics\|build_fundamentals_history_intake" -n tests src .github scripts docs/active docs/audits \
  --include="*.py" \
  --include="*.md" \
  --include="*.yml" \
  --include="*.yaml" \
  --exclude-dir=".git" \
  --exclude-dir=".venv" \
  --exclude-dir="archive"

Result summary:

No active runtime imports from scripts.core.build_fundamental_analysis were found.
No active runtime imports from scripts.core.build_fundamental_layer were found.
No active runtime imports from scripts.core.build_fundamental_metrics were found.
No active runtime imports from scripts.core.build_fundamentals_history_intake were found.
Remaining references are static governance, audit, provider-smoke, legacy, or boundary-evidence references.
src/market_scanner/analysis/analysis_boundary.py statically lists scripts/core/build_fundamental_analysis.py as historical boundary evidence.
Active tests reference related names only as static evidence or canonical-analysis boundary checks.
No .github workflow reference was found.
No active runtime script caller was found.
Decision

The four scripts/core/build_fundamental_* files are archive-ready as compatibility wrappers.

They are not canonical runtime owners.

The underlying scripts/fundamentals/*.py files remain in place and retain the BL78 classifications:

migration-required;
source-mapping-required;
review-runner-risk;
provider-side-effect-risk.

BL79 does not change the status of the underlying fundamentals logic.

Validation

Run locally before merge:

pytest -q

Result:

PASTE_LOCAL_RESULT_HERE
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
Final status

BL79 removes the active-path compatibility wrappers from scripts/core/ and preserves their historical contents under archive/legacy_runtime/scripts/core/.

The active scripts/core/ tree is now further reduced without touching provider, SEC, portfolio, watchlist, Telegram, reporting, or Decision Engine runtime behavior.


## 3. Voeg BL79 toe aan de backlog

Voeg onderaan `docs/active/project/backlog.md` toe:

```markdown
### BL79 — Archive fundamentals compatibility wrappers after active-reference check

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL79 archived the `scripts/core/build_fundamental_*` compatibility wrappers after a focused active-reference check.

Archived:

- `scripts/core/build_fundamental_analysis.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamentals_history_intake.py`

Archive targets:

- `archive/legacy_runtime/scripts/core/build_fundamental_analysis.py`
- `archive/legacy_runtime/scripts/core/build_fundamental_layer.py`
- `archive/legacy_runtime/scripts/core/build_fundamental_metrics.py`
- `archive/legacy_runtime/scripts/core/build_fundamentals_history_intake.py`

Result:

- No active runtime imports of these wrappers were found.
- Remaining references are static governance, audit, legacy, provider-smoke, or canonical boundary evidence.
- The underlying `scripts/fundamentals/*.py` files were not archived and remain governed by the BL78 migration/side-effect classifications.

Validation:

- `pytest -q`: 522 passed in 0.58s

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