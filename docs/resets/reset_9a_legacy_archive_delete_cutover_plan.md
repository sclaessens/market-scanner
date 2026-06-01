# RESET-9A — Legacy Archive/Delete Cutover Plan

## 1. Purpose

RESET-9A defines the legacy archive/delete cutover plan after RESET-0 through RESET-8.

This is a documentation-only governance step. It does not move, delete, archive, refactor, regenerate, or change runtime behavior. It classifies existing legacy surfaces and defines future cleanup batches for Codex/local implementation.

## 2. Executive Decision

Decision: PROCEED_WITH_CONTROLLED_LEGACY_CUTOVER_PLANNING

The repository now has a clean v2 scaffold under `src/market_scanner/` with contracts, fixtures, minimal pipeline core, review-only Decision Engine scaffold, in-memory reporting scaffold, and fixture-only source-data readiness scaffold.

The old runtime remains under `scripts/` and must remain untouched until an approved cutover batch explicitly moves or retires it. The next work should not add more old-architecture features. It should progressively detach legacy surfaces while preserving evidence and fallback until v2 can operate independently.

## 3. Current V2 Status

Completed reset stages:

- RESET-0: controlled rebuild decision.
- RESET-1: canonical documentation baseline.
- RESET-2: repository structure and archive plan.
- Batch A: superseded active documentation moved to legacy.
- RESET-3: v2 package skeleton.
- RESET-4: v2 data contracts and synthetic fixtures.
- RESET-5: minimal v2 pipeline core.
- RESET-6: review-only Decision Engine authority scaffold.
- RESET-7: in-memory reporting communication scaffold.
- RESET-8: fixture-only source-data readiness scaffold.

Current v2 code surface:

```text
src/market_scanner/
  discovery/
  validation/
  context/
  fundamentals/
  timing/
  portfolio/
  decisions/
  reporting/
  orchestration/
  shared/
```

Current v2 is a scaffold. It is not yet a production replacement for the legacy runtime.

## 4. RESET-9A Scope

Allowed:

- classify legacy surfaces;
- define archive/delete candidates;
- define future cutover batches;
- define validation rules for future cleanup PRs;
- define stop conditions;
- create this plan document.

Forbidden:

- code changes;
- test changes;
- CSV/data changes;
- report changes;
- generated outputs;
- workflow changes;
- file moves;
- file deletion;
- physical archive actions;
- production pipeline runs;
- SEC diagnostics;
- live SEC/provider/network calls;
- runtime behavior changes.

## 5. Legacy Surface Classification

### 5.1 Code

| Path family | Classification | Future action | Timing | Notes |
|---|---|---|---|---|
| `src/market_scanner/` | V2_ACTIVE_CODE | Keep and develop | Now | New clean v2 surface |
| `scripts/` | LEGACY_RUNTIME_SURFACE | Keep temporarily, then archive/retire | After v2 acceptance | Do not modify into v2 |
| `scripts/core/` | LEGACY_RUNTIME_REFERENCE | Archive or retire after v2 replacement | Later | Old pipeline core concepts only |
| `scripts/fundamentals/` | LEGACY_SOURCE_DATA_REFERENCE | Archive after source-data strategy matures | Later | Do not reintroduce live behavior without approval |
| `scripts/data_sources/` | LEGACY_PROVIDER_REFERENCE | Archive after v2 source-data contracts | Later | No provider calls in v2 unless approved |
| `scripts/diagnostics/` | LEGACY_DIAGNOSTIC_REFERENCE | Archive or keep as local-only tools | Later | Must not become v2 pipeline authority |
| `scripts/portfolio/` | LEGACY_PORTFOLIO_REFERENCE | Reassess after v2 portfolio contract | Later | Portfolio inputs/generated outputs must be separated |
| `scripts/reporting/` | LEGACY_REPORTING_REFERENCE | Archive after v2 reporting replacement | Later | Reporting must remain communication-only |
| `scripts/telegram/` | LEGACY_DELIVERY_REFERENCE | Reintroduce only after v2 reporting is approved | Later | No Telegram in current v2 scaffold |
| `scripts/watchlist/` | LEGACY_REFERENCE | Archive/delete candidate | Later | No active authority without v2 design |

### 5.2 Tests

| Path family | Classification | Future action | Timing | Notes |
|---|---|---|---|---|
| `tests/contract/` | V2_ACTIVE_TESTS | Keep | Now | Contract tests for v2 |
| `tests/unit/test_v2_*` | V2_ACTIVE_TESTS | Keep | Now | v2 unit tests |
| `tests/integration/test_v2_*` | V2_ACTIVE_TESTS | Keep | Now | v2 integration tests |
| `tests/fixtures/` | V2_FIXTURE_TESTS | Keep | Now | Fixture validation |
| Legacy tests for `scripts/` | LEGACY_TEST_SURFACE | Keep temporarily, then archive/retire | After v2 replacement | Preserve until old runtime is retired |
| Compatibility wrapper tests | DELETE_OR_ARCHIVE_CANDIDATE | Later cleanup batch | Later | Do not preserve wrappers as v2 requirement |
| Generated-output-dependent tests | REASSESS_REQUIRED | Later cleanup batch | Later | Must not force generated data into v2 source-of-truth |

### 5.3 Data and Artifacts

| Path family | Classification | Future action | Timing | Notes |
|---|---|---|---|---|
| `data/fixtures/v2/` | V2_APPROVED_SYNTHETIC_FIXTURES | Keep | Now | Approved minimal fixtures |
| `data/processed/` | LEGACY_GENERATED_OUTPUT | Archive/delete candidate | Later | Not v2 source-of-truth |
| `data/logs/` | LEGACY_GENERATED_LOGS | Archive/delete candidate | Later | Audit reference only |
| `data/intake/` | REAPPROVAL_REQUIRED | Reclassify before v2 use | Later | Candidate input only |
| `data/portfolio/portfolio_transactions.csv` | CANDIDATE_MANUAL_INPUT | Reapprove or migrate | Later | Likely future v2 input |
| `data/portfolio/portfolio_positions.csv` | LEGACY_GENERATED_OUTPUT | Archive/delete candidate | Later | Not source-of-truth by default |
| `data/portfolio/portfolio_review.csv` | LEGACY_GENERATED_OUTPUT | Archive/delete candidate | Later | Not Decision Engine authority |
| `data/raw/` | SOURCE_DATA_REAPPROVAL_REQUIRED | Keep local/ignored unless approved | Later | No v2 source without contract |
| `data/local/` | LOCAL_ONLY | Keep ignored/local | Always | Must not become tracked v2 input |
| `reports/` | LEGACY_GENERATED_COMMUNICATION | Archive/delete candidate | Later | Generated output, not source input |

### 5.4 Documentation

| Path family | Classification | Future action | Timing | Notes |
|---|---|---|---|---|
| `docs/active/` | V2_CANONICAL_DOCS | Keep | Now | Current active source of truth |
| `docs/resets/` | RESET_GOVERNANCE_RECORD | Keep | Now | Reset audit trail |
| `docs/legacy/active_superseded/` | LEGACY_REFERENCE | Keep | Now | Already detached from active authority |
| `docs/sprints/` | LEGACY_SPRINT_RECORD | Move to `docs/legacy/sprints/` | Future docs cleanup | Not v2 authority |
| `docs/audits/` | LEGACY_AUDIT_RECORD | Move to `docs/legacy/audits/` | Future docs cleanup | Preserve traceability |
| `docs/archive/` | EXISTING_LEGACY_ARCHIVE | Keep or normalize | Later | Already historical |

### 5.5 Workflows

| Path family | Classification | Future action | Timing | Notes |
|---|---|---|---|---|
| `.github/workflows/` | LEGACY_CI_OR_OPERATIONAL_SURFACE | Review before v2 cutover | Later | Do not change until v2 CI/commands are defined |
| Legacy scheduled workflows | LEGACY_OPERATIONAL_RISK | Pause/replace only after approval | Later | Avoid running old pipeline against v2 paths |
| Future v2 CI | REQUIRED_BEFORE_RUNTIME_CUTOVER | Add after v2 commands stabilize | Later | Should test v2 package/contracts |

## 6. Future Cleanup Batches

### RESET-9B — Documentation Legacy Consolidation

Executor: ChatGPT/GitHub or Codex, depending on whether only docs move.

Purpose:
- move `docs/sprints/` to `docs/legacy/sprints/`;
- move `docs/audits/` to `docs/legacy/audits/`;
- preserve `docs/resets/` as active reset trail;
- update README/navigation if needed.

Allowed:
- documentation moves only;
- no code/data/workflow changes.

### RESET-9C — Legacy Runtime Inventory and Retirement Decision

Executor: Codex/local implementation.

Purpose:
- inspect old `scripts/` and map each script to v2 replacement status;
- identify temporary keepers, archive candidates, and delete candidates;
- produce a machine-checked inventory before any runtime movement.

Allowed:
- inventory document;
- optional static analysis scripts only if approved;
- no runtime deletion yet.

### RESET-9D — Legacy Test Inventory and Replacement Map

Executor: Codex/local implementation.

Purpose:
- classify legacy tests that protect old runtime;
- identify v2 tests that replace old behavior;
- identify tests that should be archived or deleted after cutover.

Allowed:
- documentation and test inventory only unless separately approved.

### RESET-9E — Data and Generated Artifact Cleanup Plan

Executor: Codex/local implementation with Data Steward review.

Purpose:
- classify tracked generated CSVs, logs, reports, raw data, and backups;
- decide what should become ignored, archived, deleted, or migrated as approved input.

Allowed:
- plan first;
- no CSV deletion until owner approval.

### RESET-9F — Workflow and CI Cutover Plan

Executor: Codex/local implementation.

Purpose:
- inspect `.github/workflows/`;
- determine whether old scheduled/manual workflows still run old pipeline paths;
- define v2 CI path.

Allowed:
- workflow plan only first;
- no workflow changes until v2 run commands are defined.

### RESET-9G — Approved Archive/Delete Execution

Executor: Codex/local implementation.

Purpose:
- perform approved moves/deletes in small PRs;
- keep each PR narrow and reversible.

Allowed only after:
- inventory exists;
- owner approval is recorded;
- v2 replacement or retirement rationale exists;
- validation plan is explicit.

## 7. Stop Conditions

Do not archive or delete a legacy runtime file if:

- v2 replacement does not exist;
- the file is still needed for old fallback runs;
- tests still depend on it without replacement;
- workflows still call it;
- the file contains source-data/provenance knowledge not yet extracted;
- owner approval is missing;
- rollback/reference path is unclear.

Do not archive or delete data/CSV files if:

- ownership is unclear;
- the file may be manual input;
- the file may contain portfolio source records;
- v2 data contract has not classified it;
- the file is required for audit or reproduction;
- generated vs input status is unresolved.

## 8. Validation Rules for Future Cleanup PRs

Every future cleanup PR must report:

- exact files moved/deleted/changed;
- why each file is safe to move/delete;
- whether v2 replacement exists;
- whether legacy fallback is affected;
- tests run;
- `git diff --check` result;
- `git status` result;
- confirmation that no generated output was created unintentionally;
- confirmation that no SEC/provider/network calls were made;
- confirmation that no Decision Engine or Reporting semantics changed unless explicitly scoped.

Runtime cleanup PRs must run:

```bash
.venv/bin/python -m pytest
git diff --check
git status
```

Documentation-only cleanup PRs may skip tests only when no code/tests/data/workflows changed, but must say so explicitly.

## 9. Recommended Next Action

Recommended next action: RESET-9B — Documentation Legacy Consolidation.

Rationale:
- It is low risk.
- It continues removing old documentation from active navigation.
- It does not affect runtime fallback.
- It prepares the repository for later runtime cleanup.

After RESET-9B, proceed to RESET-9C legacy runtime inventory before moving or deleting old code.

## 10. Backlog Impact Assessment

Backlog impact assessment:
- New cleanup batches identified: RESET-9B through RESET-9G.
- No backlog file updated in RESET-9A.
- Future batches should be tracked as separate PRs or backlog items before execution.

## 11. Validation

Documentation-safe validation requested:

```bash
git diff --check
git status
```

Result: Not run through the GitHub connector because RESET-9A was executed through GitHub API file creation rather than a local working tree.

No tests were run because no code, tests, data, reports, workflows, or runtime files were changed. No pipeline was run. No SEC diagnostics were run. No live SEC/provider/network calls were made. No generated outputs were created.

## 12. Final Status

RESET-9A status: COMPLETE FOR REVIEW

Decision: PROCEED_WITH_CONTROLLED_LEGACY_CUTOVER_PLANNING

Recommended next action: RESET-9B — Documentation Legacy Consolidation

Runtime status: legacy runtime remains untouched and available as fallback/reference

V2 status: scaffold remains active under `src/market_scanner/`

Archive/delete status: plan only; no physical archive/delete executed
