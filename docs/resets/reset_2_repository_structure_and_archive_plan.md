# RESET-2 — Repository Structure and Archive Plan

## 1. Purpose

RESET-2 defines how the repository should separate new canonical v2 authority from older legacy material.

This is a documentation-only governance plan. It does not move, remove, archive, regenerate, refactor, or change any runtime files. It creates the classification and cutover rules that later stages must follow.

## 2. Executive Decision

Decision: PROCEED_WITH_STAGED_STRUCTURE_AND_ARCHIVE_PLAN

The repository should keep the RESET-1 canonical documentation as the active v2 planning baseline and progressively detach older documentation, code, tests, data, reports, and workflows from active authority.

RESET-2 authorizes no physical repository restructuring. It only defines the plan for later execution.

## 3. Current Authority Baseline

Current v2 authority order:

1. `AGENTS.md` for repository-level AI governance until explicitly updated.
2. RESET-1 canonical active documents under `docs/active/`.
3. `docs/resets/reset_0_full_repository_knowledge_extraction_and_rebuild_decision.md` for rebuild rationale.
4. `docs/resets/reset_1_canonical_documentation_rewrite_closeout.md` for RESET-1 closeout.
5. Older active, sprint, audit, archive, code, test, data, report, and workflow files as reference material only unless explicitly carried forward.

## 4. Scope

Allowed in RESET-2:

- define future repository structure;
- classify existing path families;
- define legacy and future cutover candidates;
- define movement batches;
- define preconditions for later restructuring;
- define owner approvals required before later changes;
- define validation expectations for future PRs;
- create this reset plan document.

Forbidden in RESET-2:

- code changes;
- test changes;
- CSV/data changes;
- generated outputs;
- report changes;
- workflow changes;
- file moves;
- physical archive actions;
- pipeline runs;
- SEC diagnostics;
- live SEC/network calls;
- runtime behavior changes.

## 5. Target V2 Repository Structure

```text
docs/
  active/
    project_charter.md
    product_vision.md
    roles_and_responsibilities.md
    pm_operating_model.md
    functional_analysis.md
    technical_architecture.md
    data_architecture.md
    source_data_strategy.md
    pipeline_contract.md
    decision_engine_contract.md
    reporting_contract.md
    testing_strategy.md
    repository_structure.md
    backlog.md
    roadmap.md
  legacy/
    active_superseded/
    sprints/
    audits/
    migration/
    source_data/
    technical/
    functional/
    execution/
  resets/
    reset_0_full_repository_knowledge_extraction_and_rebuild_decision.md
    reset_1_canonical_documentation_rewrite_closeout.md
    reset_2_repository_structure_and_archive_plan.md
src/
  market_scanner/
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
tests/
  contract/
  unit/
  integration/
  fixtures/
data/
  input/
    portfolio/
    universe/
  fixtures/
  generated/
    processed/
    logs/
reports/
  generated/
```

The target structure is not executed by RESET-2. It is a plan for later reset stages.

## 6. Documentation Classification

| Path family | Future classification | Timing | Notes |
|---|---|---|---|
| RESET-1 canonical `docs/active/*.md` | KEEP_ACTIVE | Now | Active v2 planning baseline |
| Other old `docs/active/*` | MOVE_TO_LEGACY_ACTIVE_SUPERSEDED | Later approved batch | Should no longer be active authority after cutover |
| `docs/active/analysis/*` | MOVE_TO_LEGACY_ACTIVE_SUPERSEDED | Later approved batch | Source material only |
| `docs/active/contracts/*` | MOVE_TO_LEGACY_ACTIVE_SUPERSEDED | Later approved batch | Replaced conceptually by RESET-1 contracts |
| `docs/active/inventory/*` | MOVE_TO_LEGACY_ACTIVE_SUPERSEDED | Later approved batch | Useful evidence only |
| `docs/active/logic/*` | REWRITE_OR_MOVE_TO_LEGACY | Later approved batch | May need future v2 calculation/logic doc |
| `docs/active/specs/*` | MOVE_TO_LEGACY_ACTIVE_SUPERSEDED | Later approved batch | Not v2 implementation authority |
| `docs/active/runbooks/*` | REWRITE_OR_MOVE_TO_LEGACY | After v2 commands exist | v2 runbooks should be written later |
| `docs/sprints/*` | MOVE_TO_LEGACY_SPRINTS | Later approved batch | Preserve traceability; not active v2 authority |
| `docs/audits/*` | MOVE_TO_LEGACY_AUDITS | Later approved batch | Preserve traceability; not active v2 authority |
| `docs/archive/*` | KEEP_LEGACY_REFERENCE | Later optional normalization | Already historical |
| `docs/resets/*` | KEEP_RESET_RECORD | Now | Active reset trail |
| Old `docs/technical/*` | MOVE_TO_LEGACY_TECHNICAL | Later approved batch | Source material only |
| Old `docs/functional/*` | MOVE_TO_LEGACY_FUNCTIONAL | Later approved batch | Source material only |
| Old `docs/execution/*` | MOVE_TO_LEGACY_EXECUTION | Later approved batch | Source material only |

## 7. Code Classification

| Path family | Future classification | Timing | Notes |
|---|---|---|---|
| `scripts/` | KEEP_TEMPORARILY_FOR_LEGACY_RUNS | Until v2 cutover | Old Python remains reference-only for v2 |
| `scripts/core/` | REFERENCE_ONLY_FOR_V2 | Until v2 replacement exists | Do not modify into v2 code |
| `scripts/fundamentals/` | REFERENCE_ONLY_FOR_V2 | Until RESET-8 | SEC/fundamentals knowledge only |
| `scripts/data_sources/` | REFERENCE_ONLY_FOR_V2 | Until v2 data contracts | Concepts may be reimplemented cleanly |
| `scripts/diagnostics/` | REFERENCE_ONLY_FOR_V2 | Later diagnostics stage | Legacy/reference only |
| `scripts/portfolio/` | REFERENCE_ONLY_FOR_V2 | Until v2 portfolio contract | Source/generated boundaries must be reapproved |
| `scripts/reporting/` | REFERENCE_ONLY_FOR_V2 | Until RESET-7 | Reporting concepts only |
| `scripts/telegram/` | REFERENCE_ONLY_FOR_V2 | After v2 reporting | Network delivery remains isolated |
| `scripts/watchlist/` | LEGACY_REFERENCE | Later cutover review | No active authority without v2 design |
| Future `src/market_scanner/` | V2_ACTIVE_CODE | RESET-3 and later | Newly written code only |

RESET-2 does not create `src/`, move scripts, or change runtime files.

## 8. Test Classification

| Path family | Future classification | Timing | Notes |
|---|---|---|---|
| Existing `tests/` | REFERENCE_ONLY_FOR_V2 | Until v2 tests exist | Preserve behavioral knowledge, not old paths |
| Existing compatibility tests | LEGACY_REFERENCE | Later cutover review | Do not preserve wrappers as v2 requirement |
| Existing fixture-dependent tests | REAPPROVAL_REQUIRED | RESET-4 | Fixtures must be intentionally approved |
| Future `tests/contract/` | V2_ACTIVE_TESTS | RESET-4/5+ | Contract and boundary tests |
| Future `tests/unit/` | V2_ACTIVE_TESTS | RESET-3+ | New unit tests for new modules |
| Future `tests/integration/` | V2_ACTIVE_TESTS | RESET-5+ | Approved pipeline slices only |
| Future `tests/fixtures/` | GOVERNED_FIXTURES | RESET-4 | Small deterministic fixtures only |

RESET-2 does not modify tests.

## 9. Data, CSV, and Artifact Classification

| Path family | Future classification | Timing | Notes |
|---|---|---|---|
| `data/processed/` | GENERATED_LEGACY_OUTPUT | Later cutover review | Not v2 source-of-truth unless reapproved |
| `data/logs/` | GENERATED_LEGACY_OUTPUT | Later cutover review | Audit reference only |
| `data/intake/` | REAPPROVAL_REQUIRED | RESET-4 | Candidate input; contract approval required |
| `data/portfolio/portfolio_transactions.csv` | CANDIDATE_MANUAL_INPUT | RESET-4 | Likely useful but must be reapproved |
| `data/portfolio/portfolio_positions.csv` | GENERATED_LEGACY_OUTPUT | RESET-4 or later | Not source-of-truth by default |
| `data/portfolio/portfolio_review.csv` | GENERATED_LEGACY_OUTPUT | RESET-4 or later | Not Decision Engine authority |
| `data/watchlist/` | LEGACY_REFERENCE | Later cutover review | Not active without design |
| `data/raw/` | SOURCE_DATA_REAPPROVAL_REQUIRED | RESET-4/8 | Real source data remains local/ignored unless approved |
| `data/local/` | LOCAL_ONLY | Always | Should remain local |
| Future `data/input/` | V2_MANUAL_OR_APPROVED_INPUT | RESET-4 | Only approved input files |
| Future `data/fixtures/` | GOVERNED_FIXTURES | RESET-4 | Small deterministic fixtures |
| Future `data/generated/` | GENERATED_OUTPUT | RESET-5+ | Ignored by default |
| `reports/` | GENERATED_LEGACY_OUTPUT | Later cutover review | Not source input |
| Future `reports/generated/` | GENERATED_OUTPUT | RESET-7+ | Ignored by default |

RESET-2 does not modify CSV, data, or reports.

## 10. Workflow and CI Classification

| Path family | Future classification | Timing | Notes |
|---|---|---|---|
| `.github/workflows/` | KEEP_TEMPORARILY_FOR_LEGACY_RUNS | Until v2 CI exists | Do not change in RESET-2 |
| Legacy scheduled workflows | REVIEW_BEFORE_V2_CUTOVER | Later CI stage | Avoid running old pipeline against v2 paths |
| Future v2 CI | REWRITE_FOR_V2 | After RESET-3/4 | Should test new package and contracts |

RESET-2 does not modify workflows.

## 11. Proposed Future Batches

### Batch A — Documentation Authority Detachment

Purpose: ensure only RESET-1 canonical docs are treated as active authority.

Possible later actions:

- move superseded active docs to `docs/legacy/active_superseded/`;
- add short legacy notices if needed;
- update active navigation references.

Preconditions:

- owner approval;
- no code/data/workflow changes in the same PR;
- reference check.

### Batch B — Sprint and Audit Legacy Consolidation

Purpose: preserve sprint and audit evidence outside active authority.

Possible later actions:

- move remaining sprint evidence to `docs/legacy/sprints/`;
- move audit evidence to `docs/legacy/audits/`;
- keep reset documents under `docs/resets/`.

### Batch C — Technical, Functional, and Execution Legacy Consolidation

Purpose: detach old technical, functional, and execution frameworks from v2 authority.

Possible later actions:

- move old technical, functional, and execution content to matching legacy paths.

### Batch D — V2 Codebase Bootstrap

Purpose: create a clean v2 code surface.

Possible later actions:

- create `src/market_scanner/`;
- add minimal package skeleton;
- add smoke tests.

Precondition: old Python files must not be copied as the v2 base.

### Batch E — Data Contract and Fixture Setup

Purpose: approve inputs and fixtures before implementation.

Possible later actions:

- define approved inputs;
- define small deterministic fixtures;
- document schemas and owner approval.

### Batch F — Legacy Runtime Cutover

Purpose: retire old run surfaces after v2 can operate independently.

Possible later actions:

- move or retire legacy scripts, tests, data, reports, and workflows only after replacements exist and owner approval is recorded.

## 12. README and AGENTS Impact

`README.md` and `AGENTS.md` are root-level entry points. They should eventually be updated so that new operators start from the RESET-1 canonical docs and understand that old material is legacy reference.

Recommended timing:

- update `README.md` during Batch A or a dedicated documentation navigation PR;
- update `AGENTS.md` only if the user explicitly approves changing root AI governance wording.

RESET-2 does not modify either file.

## 13. SEC/Fundamentals Impact

SEC-7F remains paused as standalone old-architecture work.

SEC and fundamentals material should be treated as:

- source-data knowledge;
- review-model knowledge;
- local-only data handling knowledge;
- v2 RESET-8 input material.

No SEC code, SEC data, SEC diagnostics, or live SEC call is authorized by RESET-2.

## 14. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Old docs still appear active | HIGH | Execute Batch A after approval |
| Losing historical evidence | HIGH | Move to legacy rather than remove |
| Moving files too early | HIGH | No movement until owner approval and batch plan |
| Breaking local legacy runs | HIGH | Do not move scripts/data/tests until v2 cutover |
| Confusing generated outputs with inputs | HIGH | Enforce RESET-4 data contract approval |
| Codex copying old code into v2 | HIGH | Future prompts must forbid old-code-as-base |
| Link/reference breakage | MEDIUM | Run reference checks in future move PRs |
| Workflow drift | MEDIUM | Do not change workflows until v2 CI plan exists |
| SEC complexity returning too early | HIGH | Keep SEC/fundamentals deferred to RESET-8 |

## 15. RESET-3 Handoff

RESET-3 should bootstrap the v2 codebase only after RESET-2 is reviewed and merged.

Expected RESET-3 output:

- minimal `src/market_scanner/` package skeleton;
- minimal smoke test structure;
- no old Python file copied as v2 base;
- no pipeline behavior unless explicitly scoped;
- no SEC/fundamentals integration;
- no reporting/Telegram implementation;
- no generated outputs.

RESET-3 should be executed by Codex/local implementation unless the owner chooses Batch A as an additional documentation-only step first.

## 16. Backlog Impact Assessment

Backlog impact assessment:
- RESET-2 confirms that active reset backlog direction is already captured in `docs/active/backlog.md`.
- No backlog file change is required in RESET-2.
- Future Batch A through Batch F work should be tracked as separate planning or implementation candidates.

## 17. Validation

Documentation-safe validation requested:

```bash
git diff --check
git status
```

Result: Not run through the GitHub connector because RESET-2 was executed through GitHub API file creation rather than a local working tree.

No tests were run because no code or tests were changed. No pipeline was run. No SEC diagnostics were run. No live SEC calls were made. No generated outputs were created.

## 18. Final Decision

RESET-2 Decision: PROCEED_WITH_STAGED_STRUCTURE_AND_ARCHIVE_PLAN

Recommended next action: Review and merge RESET-2. Then either prepare RESET-3 — V2 Codebase Bootstrap, or execute Batch A documentation authority detachment first if the owner wants old active documents moved out before code bootstrap.

Development status: Pause new feature work on old active architecture.

SEC-7F status: Pause standalone SEC-7F; absorb into v2 source-data roadmap.

Old code status: Reference-only for v2; keep temporarily for legacy runs.

Old documentation status: Legacy source material unless explicitly carried forward.

Old data/CSV status: Not v2 source-of-truth unless reapproved.

Archive status: Plan defined; no physical archive action executed in RESET-2.

Pipeline integration status: Blocked until v2 structure, contracts, fixtures, and skeleton exist.
