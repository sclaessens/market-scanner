# ME04-PREP Remaining Legacy Documentation Inventory

Owner role: Governance Auditor / Scrum Master

Status: ME04-PREP-B INVENTORY

## Purpose

This inventory records documentation and reference material that remains outside the active Market Engine documentation root and outside the Market Scanner reference archive path.

This sprint is inspection and documentation only. It does not move, delete, rename, rewrite, or archive files.

## Scope

Inspected documentation-like files and reference areas outside:

- `docs/market_engine/`
- `docs/archive/market_scanner_reference/`

The inspection included root-level documentation-like files, documentation folders, archive folders, legacy folders, reports, and reference text files.

## Active Documentation Rule

From the Market Engine line onward, `docs/market_engine/` is the active Market Engine documentation root.

Old v2, BL, reset, audit, governance, sprint, provider-smoke, and legacy documentation should be treated as reference material only. Future Market Engine work should use old material through explicit extraction.

## Inspection Commands Used

```bash
git checkout main
git pull origin main
git status
git checkout -b me04-prep-inventory-remaining-legacy-docs
find . \
  -path './.git' -prune -o \
  -path './.venv' -prune -o \
  -path './docs/market_engine/*' -prune -o \
  -path './docs/archive/market_scanner_reference/*' -prune -o \
  -type f \( -name '*.md' -o -name '*.txt' -o -name '*.rst' \) \
  -print | sort
find archive legacy reports docs/archive docs/audits docs/legacy docs/resets docs/templates \
  -maxdepth 4 \
  -type f 2>/dev/null | sort
test -f docs/market_engine/audits/me04prep_archive_active_docs_audit.md && sed -n '1,140p' docs/market_engine/audits/me04prep_archive_active_docs_audit.md || true
find docs/archive -type f -name '*.md' | wc -l
find docs/audits -type f -name '*.md' | wc -l
find docs/legacy -type f -name '*.md' | wc -l
find docs/resets -type f -name '*.md' | wc -l
find reports -type f | wc -l
find archive legacy -type f | wc -l
```

## Summary Of Findings

Current `main` still contains `docs/active/`. The ME04-PREP audit file was not present on this branch during inspection, so this sprint does not update it. This inventory treats `docs/active/` as a known predecessor root that should be handled by ME04-PREP rather than by ME04-PREP-B.

Remaining documentation/reference material outside `docs/market_engine/` and outside `docs/archive/market_scanner_reference/` includes:

- root-level repository documentation and text files;
- the former active documentation tree, pending or already handled by ME04-PREP depending on branch state;
- historical archive documentation under `docs/archive/`;
- audit documentation under `docs/audits/`;
- legacy documentation under `docs/legacy/`;
- reset closeout and planning documentation under `docs/resets/`;
- documentation templates under `docs/templates/`;
- generated report markdown under `reports/daily/`;
- runtime or local text artifacts such as `data/logs/telegram_offset.txt`;
- archived legacy runtime code under root-level `archive/`;
- root-level legacy code under `legacy/`.

Observed counts:

| Area | Count observed | Notes |
|---|---:|---|
| `docs/archive/**/*.md` | 126 | Historical sprint, audit, migration, technical, functional, and execution docs. |
| `docs/audits/**/*.md` | 101 | Legacy runtime, provider smoke, reset cleanup, and runtime boundary audits. |
| `docs/legacy/**/*.md` | 104 | Superseded active docs, research, sprints, technical, financial, functional, and vision docs. |
| `docs/resets/**/*.md` | 41 | Reset closeouts, plans, and knowledge extraction maps. |
| `reports/**/*` | 25 | Generated report/output artifacts and Telegram message file. |
| `archive/**/*` plus `legacy/**/*` | 46 | Archived script-era runtime/code files, not documentation roots. |

## Category 1 - Should Remain At Repository Root

These files are legitimate root-level repository control, onboarding, or operation references:

- `README.md` - repository overview/onboarding.
- `AGENTS.md` - institutional agent instructions and governance constraints.
- `requirements.txt` - dependency list used by tooling/runtime setup.
- `tickers.txt` - repository-level ticker list/input reference; not a Market Engine documentation file.

These should not be moved by a documentation archive sprint without a separate repository-structure decision.

## Category 2 - Should Remain Outside Docs Because It Is Runtime, Data, Output, Or Tooling

These files or folders are not candidates for documentation-root cleanup in this sprint:

- `reports/daily/*.md` - generated or historical report outputs.
- `reports/daily/telegram_message.txt` - generated Telegram communication artifact.
- `data/logs/telegram_offset.txt` - runtime/local delivery state.
- `.pytest_cache/README.md` - tool cache file.
- root-level `archive/legacy_runtime/scripts/` - archived runtime/code material, not documentation.
- root-level `legacy/telegram/` and `legacy/watchlist/` - legacy code paths, not documentation.

These areas may need separate data/output/runtime governance, but they should not be moved by a documentation inventory sprint.

## Category 3 - Candidate To Move Into docs/archive/market_scanner_reference/

These areas appear to contain historical market-scanner documentation or reference records and are candidates for later consolidation into `docs/archive/market_scanner_reference/`:

- `docs/archive/` - historical sprint, audit, migration, functional, technical, and execution documentation.
- `docs/audits/` - legacy runtime audits, provider smoke records, reset cleanup records, and runtime boundary migration records.
- `docs/legacy/` - superseded active docs, legacy audits, research, sprint records, technical docs, financial docs, functional docs, and vision docs.
- `docs/resets/` - reset plans, closeouts, knowledge extraction maps, workflow cutover plans, and repository-structure transition records.
- `docs/project_roles_and_responsibilities.md` - standalone documentation file outside Market Engine and outside the reference archive.

Recommended future treatment: move these documentation families only in a separate approved sprint with explicit mapping, preservation of history, and archive README updates.

## Category 4 - Needs Manual Decision

These items are ambiguous or require human judgment before movement:

- `docs/active/` - still present on current `main` during this inventory. It should be handled by ME04-PREP, not by this inventory sprint. If ME04-PREP has already merged elsewhere, reconcile branch state before acting.
- `docs/templates/` - templates may still be useful as shared repository documentation infrastructure. Decide whether templates remain global or move under Market Engine/reference archive.
- `pycharm_test.txt` - root-level local/tooling-looking text file. Purpose is unclear from filename alone.
- `tickers.txt` - root-level ticker input/reference. It may be operational input rather than documentation and should not move without data/input ownership review.
- generated report markdown under `reports/daily/` - clearly output artifacts, but any long-term retention or archive strategy should be decided by data/report governance rather than documentation-root cleanup alone.

## Category 5 - Do Not Touch

These paths should not be moved by documentation archive work:

- `src/`
- `tests/`
- `.github/`
- `.git/`
- `.venv/`
- runtime data directories;
- generated report/output directories;
- archived or legacy code directories unless a future runtime/code archive sprint explicitly scopes them;
- configuration files required by tooling or runtime setup.

## Recommended Follow-Up

1. Confirm whether ME04-PREP has merged into `main`. If not, merge or rebase the inventory work after ME04-PREP so `docs/active/` is not double-counted.
2. Run a dedicated documentation consolidation sprint for `docs/archive/`, `docs/audits/`, `docs/legacy/`, and `docs/resets/`.
3. Create a mapping table before any future move, with source path, target path, rationale, and owner role.
4. Leave `reports/`, runtime data, root-level operational inputs, root-level agent instructions, and code archives untouched unless a future sprint explicitly scopes them.
5. Keep all old documents reference-only. Market Engine implementation must continue from `docs/market_engine/` and explicit extraction decisions.

## Boundaries Confirmed

- No files were moved.
- No files were deleted.
- No files were renamed.
- No Python files were changed.
- No test files were changed.
- No tests were run.
- No provider calls were executed.
- No yfinance calls were executed.
- No SEC or EDGAR calls were executed.
- No scanner or runtime commands were executed.
- No reports were generated.
- No Telegram messages were sent.
- No portfolio data was mutated.
- No watchlist data was mutated.
- No production writes were introduced.
- No Decision Engine behavior was changed.
