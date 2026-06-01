# RESET-0 — Full Repository Knowledge Extraction and Rebuild Decision

## 1. Executive Decision

Decision: PROCEED_WITH_CONTROLLED_REBUILD

RESET-0 recommends a controlled clean rebuild based on the current repository evidence. The repository is valuable and should not be treated as failed work. It has produced certified architecture doctrine, tested behavior, operational lessons, source-data lessons, and governance boundaries that should survive into v2. However, the active surface now contains multiple generations of documentation, compatibility wrappers, generated artifacts, historical sprint evidence, SEC review work, and operational patch layers that make continued incremental development increasingly risky.

Active development on the old architecture should pause for new feature work. Maintenance-only changes may continue only when needed to preserve the current legacy run path or protect existing data. SEC-7F should pause as a standalone old-architecture implementation sprint and be absorbed into the rebuild sequence. Old Python files must not be modified into v2 code. They may be studied as reference material, but v2 Python files should be newly written after new contracts exist. Old documentation remains source material until RESET-1, but it should not remain future canonical authority. No archive, delete, file move, runtime change, pipeline run, SEC diagnostic, or live SEC call is authorized in RESET-0.

## 2. Why RESET-0 Exists

RESET-0 exists because the repository now contains more historical decision material than is healthy for the next phase of development. The project has completed architecture purification, operational sprint planning, active documentation consolidation, SEC source-data exploration, runtime rationalization, and reporting governance. That work created strong knowledge, but it also created fragmentation.

The strategic problem is not that the old repository is wrong. The problem is that many artifacts now preserve partial historical decisions, temporary compatibility decisions, old sprint boundaries, generated outputs, and implementation compromises. Continuing to patch the same active structure may add more files, more exceptions, more wrappers, and more ambiguity.

A clean rebuild is intended to remove ballast, not erase knowledge. The reset should preserve certified knowledge while preventing old implementation details from becoming v2 architecture by accident.

## 3. Repository Knowledge Extraction Summary

The repository currently contains the following path families and knowledge categories:

- Root governance and onboarding files: `AGENTS.md`, `README.md`, dependency/configuration files, and ignore rules.
- Active documentation: `docs/active/`, including architecture, governance, roles, contracts, operating model, roadmap, runbooks, inventory, logic, analysis, and specs.
- Sprint/status documentation: `docs/sprints/`, including current sprint status, project backlog, SEC planning notes, and recent governance work.
- Audit documentation: `docs/audits/`, especially SEC alignment audit material.
- Archive documentation: `docs/archive/`, preserving completed sprints, audits, migration evidence, superseded roadmaps, and older planning material.
- Runtime code: `scripts/`, including orchestration, core layers, fundamentals tools, data-source utilities, diagnostics, portfolio utilities, reporting, Telegram, watchlist utilities, and compatibility wrappers.
- Tests: `tests/`, including contract-like governance tests, runtime unit tests, implementation-specific tests, fixture-dependent tests, SEC/fundamentals tests, and reporting tests.
- Data and artifacts: `data/`, including processed outputs, logs, portfolio inputs/outputs, watchlist files, raw/local fundamentals material, fixtures, backups, and generated CSVs.
- Reports: `reports/`, including generated daily communication artifacts.
- Workflows: `.github/workflows/`, including operational scheduled or manual automation surfaces.

This structure contains substantial institutional knowledge, but its active/reference/archive boundaries are still too easy to confuse during future implementation.

## 4. Still-Valid Certified Knowledge

The following knowledge should survive into v2:

- Classification upstream.
- Allocation downstream.
- Decision Engine as the only allocation, execution, arbitration, and final-action authority.
- Reporting as communication-only.
- Upstream layers must not create tradeability, urgency, conviction, eligibility, allocation, buy/sell, ranking, scoring, or final-action semantics.
- No hidden filtering in scanner, validation, context, fundamentals, timing, portfolio intelligence, or reporting.
- Deterministic outputs for identical inputs.
- Explicit row preservation where contractually required.
- Source row traceability and auditability.
- Explicit data contracts before implementation.
- Source-data classification before analysis.
- Generated outputs are not source-of-truth inputs unless explicitly approved.
- Portfolio state is separate from opportunity classification.
- Portfolio Intelligence may describe portfolio presence and exposure, but must not allocate.
- Stability/persistence metadata must not mutate Decision Engine decisions.
- SEC/fundamentals output is source-data readiness evidence until its review model, provenance, period selection, and data contracts are stable.
- Missing values must not be treated as zero.
- SEC rejected/skipped facts must remain auditable and attached at the narrowest correct evidence level.
- Reporting may group, format, summarize, truncate, and deliver communications, but must not reinterpret source decisions.
- Tests should validate contracts and authority boundaries, not historical implementation accidents.
- ChatGPT may perform documentation-only governance work; Codex/local execution should perform implementation, tests, refactors, file moves, and runtime validation.
- Repository content must remain English-only.

## 5. Decisions That Should Not Survive Into V2

The following patterns should be rejected for v2:

- Compatibility wrappers as permanent architecture.
- Old sprint documents remaining active doctrine.
- Multiple active documents carrying overlapping authority.
- Generated CSVs committed or reused as quasi-source inputs without reapproval.
- Patching old Python files indefinitely.
- Using old Python files as the v2 base.
- Preserving old paths only for backward compatibility.
- Allowing historical docs to overrule new canonical docs.
- Allowing current generated reports/logs to be treated as active project source files.
- Broad SEC outputs being interpreted as pipeline-ready because they are row-preserving.
- SEC review-required states being interpreted as business weakness.
- Filtering SEC periods before classifying source facts.
- Repeating global skipped-fact evidence on every row.
- Hidden upstream filtering disguised as validation, quality, timing, or context logic.
- Reporting summaries that omit rows without explicit representation rules.
- Tests that force v2 to preserve old wrappers, old paths, or old accidental schemas.
- Backlog entries being treated as implementation authorization.

## 6. Documentation Classification

| Documentation family | Classification | Current role | Risk | Recommendation | RESET-1 source material |
|---|---|---|---|---|---|
| `AGENTS.md` | ACTIVE_UNTIL_RESET_1 | Repository AI governance and authority boundary baseline | Current source-of-truth order may conflict with reset direction | Use as mandatory source material; rewrite or preserve its core doctrine during RESET-1 | Yes |
| `README.md` | REWRITE_REQUIRED | Onboarding and current phase summary | Points to current active architecture and operational evolution path, not reset path | Rewrite after canonical reset docs exist | Yes |
| `docs/active/` root documents | ACTIVE_UNTIL_RESET_1 | Current operational doctrine | Too many active documents can preserve overlapping authority | Extract useful content, then replace with smaller canonical v2 set | Yes |
| `docs/active/contracts/` | REWRITE_REQUIRED | Current pipeline contracts | Contracts encode old paths and compatibility surfaces | Use concepts, not paths, as source material for v2 contracts | Yes |
| `docs/active/roles_and_responsibilities.md` | REWRITE_REQUIRED | Active role matrix | Strong role knowledge but needs explicit reset/v2 authority model | Rewrite into one canonical v2 role document | Yes |
| `docs/active/inventory/` | LEGACY_REFERENCE | Runtime/source-data inventories and rationalization plans | May preserve old cleanup direction rather than rebuild direction | Use as evidence for classification; do not make v2 structure depend on it | Yes |
| `docs/active/specs/` | LEGACY_REFERENCE | Implementation specs for current architecture | Can pull v2 toward old implementation | Preserve for knowledge only | Selectively |
| `docs/active/logic/` | REWRITE_REQUIRED | Strategy logic and calculation governance | Useful logic knowledge but may include current-path assumptions | Rewrite as canonical v2 strategy/calculation governance | Yes |
| `docs/active/analysis/` | REWRITE_REQUIRED | Functional/analysis contracts | Useful, but should be consolidated into v2 canonical analysis docs | Rewrite | Yes |
| `docs/sprints/sprint_status_tracker.md` | ACTIVE_UNTIL_RESET_1 | Sprint lifecycle status | Old lifecycle model may no longer fit reset roadmap | Keep untouched in RESET-0; replace or update during RESET-1/RESET-2 | Yes |
| `docs/sprints/project_backlog.md` | ACTIVE_UNTIL_RESET_1 | Deferred-work source | Backlog may become overloaded if reset roadmap is inserted without restructuring | Do not modify in RESET-0; draft reset epic in this document | Yes |
| SEC sprint docs in `docs/sprints/` | LEGACY_REFERENCE | SEC source-data and review-model planning | Standalone SEC implementation could conflict with reset pause | Preserve as reset source material; pause implementation | Yes |
| `docs/audits/` | LEGACY_REFERENCE | Current audit evidence, especially SEC readiness | Can look active because it is outside archive | Use as evidence only; move/archive later by governed plan | Yes |
| `docs/archive/` | LEGACY_REFERENCE | Historical sprint, audit, migration, superseded material | Valuable but too large for daily operational authority | Keep as reference; do not delete in RESET-0 | Selectively |
| Old roadmap documents | ARCHIVE_CANDIDATE | Historical sequencing | Can conflict with reset roadmap | Preserve until RESET-2/RESET-9; do not use as authority | Selectively |
| Source-data docs | REWRITE_REQUIRED | Provenance and readiness policy | Must be aligned with v2 data contracts | Rewrite in RESET-1 | Yes |
| Role/responsibility docs outside active set | LEGACY_REFERENCE | Historical role assumptions | Mixed authority risk | Use only if referenced by active docs or reset extraction | Selectively |

## 7. Role Documentation Rewrite Requirements

| Role | What should remain | What should change | Required v2 documentation | Authority in v2 | Must not do |
|---|---|---|---|---|---|
| Product Owner | Owns priorities, acceptance, risk tolerance, and business direction | Explicitly approve reset scope and cutover timing | `roles_and_responsibilities.md`, `product_vision.md` | Product priority and acceptance authority | Approve hidden allocation semantics or unsupported data assumptions |
| PM | Owns sequencing, scope control, roadmap, backlog hygiene | Shift from sprint ceremony to reset-roadmap control | `pm_operating_model.md`, `roadmap.md`, `backlog.md` | Sprint/reset planning authority | Authorize runtime implementation alone |
| Scrum Master | Owns process discipline, status tracking, closeout, blockers | Use reset lifecycle instead of old sprint lifecycle where appropriate | `pm_operating_model.md` | Process and status authority | Expand scope beyond approved reset stage |
| Functional Analyst | Owns workflows, user requirements, acceptance criteria | Rewrite requirements around v2 operator workflows | `functional_analysis.md` | Functional contract authority | Define formulas or implementation |
| Technical Analyst | Owns architecture, contracts, module placement, validation strategy | Define v2 structure before code | `technical_architecture.md`, `pipeline_contract.md`, `repository_structure.md` | Technical architecture authority | Create product priorities or source-data exceptions |
| Data Steward | Owns provenance, source evidence, local-data handling, readiness | Become mandatory gate before source data enters v2 contracts | `data_architecture.md`, `source_data_strategy.md` | Source acceptability and provenance authority | Interpret financial meaning alone or allocate |
| Developer / Codex | Owns implementation, tests, refactors after approval | Must write new v2 code from scratch after contracts exist | `repository_structure.md`, developer specs in later reset stages | Implementation authority within approved scope | Reuse old code as v2 base or invent architecture |
| Financial Analyst | Owns metric meaning and financial interpretation | Separate financial meaning from raw source readiness and from allocation | `source_data_strategy.md`, `functional_analysis.md`, calculation governance docs | Financial interpretation authority | Implement code or issue final actions |
| Governance Auditor | Owns doctrine compliance and authority-boundary review | Audit reset stages and prevent old assumptions from leaking | `governance_model.md` or section in `pm_operating_model.md` | Compliance and risk authority | Implement changes or allocate |
| ChatGPT / documentation governance assistant | Owns documentation-only governance support when asked | Continue GitHub docs-only work; avoid runtime changes | `roles_and_responsibilities.md` | Documentation drafting and docs-only PR authority | Modify code, tests, data, workflows, or runtime behavior |
| Codex / implementation agent | Owns local/repo implementation after explicit prompt | Implement only after RESET-3+ contracts/specs exist | Developer specs and implementation prompts | Code/test implementation authority | Start from old Python files as active v2 base |

## 8. Code Classification

Old Python files should not be modified into v2 code. They may be studied as reference, but v2 active code should be newly written after new canonical contracts exist.

| Code family | Classification | Current role | Recommendation |
|---|---|---|---|
| Scanner scripts | REFERENCE_ONLY_FOR_V2 | Discovery and opportunity preservation | Rewrite from scratch under v2 contracts; preserve no-hidden-filtering knowledge |
| Validation layer | REFERENCE_ONLY_FOR_V2 | Structure and entry-quality classification | Rewrite from scratch; preserve classification-only semantics |
| Context layer | REFERENCE_ONLY_FOR_V2 | Leadership and relative-strength classification | Rewrite from scratch; preserve non-gating semantics |
| Entry quality/backfill scripts | REFERENCE_ONLY_FOR_V2 | Historical diagnostics/backfills | Study as diagnostics; do not place in v2 core unless approved |
| Fundamentals scripts | REFERENCE_ONLY_FOR_V2 | Current MVP and compatibility fundamental quality logic | Rebuild after source-data contracts exist |
| SEC scripts | REFERENCE_ONLY_FOR_V2 | Local source-data review/transformation | Preserve review-model knowledge; rebuild after v2 data architecture |
| Timing scripts | REFERENCE_ONLY_FOR_V2 | Timing condition and setup-state metadata | Rewrite from scratch against v2 timing contract |
| Portfolio scripts | REFERENCE_ONLY_FOR_V2 | Portfolio source, positions, review, metadata | Rebuild with explicit manual input/generated output boundaries |
| Decision Engine scripts | REWRITE_FROM_SCRATCH | Sole final-action/allocation authority | Rebuild only after v2 pipeline and input contracts exist |
| Reporting scripts | REWRITE_FROM_SCRATCH | Communication-only dashboard/Telegram outputs | Rebuild after v2 final-decision contract exists |
| Telegram scripts | REFERENCE_ONLY_FOR_V2 | Network delivery/command integration | Keep isolated; redesign after reporting contract |
| Orchestration scripts | REWRITE_FROM_SCRATCH | End-to-end run sequencing | New v2 runner after contracts and module layout exist |
| Compatibility wrappers | ARCHIVE_CANDIDATE | Stable old path preservation | Do not preserve in v2 unless a temporary cutover plan explicitly requires it |
| Utility/helper scripts | UNKNOWN_REQUIRES_OWNER_DECISION | Shared helpers and file utilities | Reassess during RESET-3; copy concepts only if clean |
| Watchlist scripts | ARCHIVE_CANDIDATE | Legacy/supporting utilities | Keep as reference; do not restore active authority without v2 design |

KEEP_TEMPORARILY_FOR_LEGACY_RUNS applies to the current old architecture only until a cutover exists. That status does not authorize v2 reuse.

## 9. Test Classification

Current tests are valuable evidence, but they should not be reused directly as v2 constraints unless rewritten against v2 contracts.

| Test type | Classification | Recommendation |
|---|---|---|
| Governance contract tests | Conceptually valuable contract tests | Preserve concepts; rewrite in v2 test suite |
| Decision Engine tests | Conceptually valuable contract tests | Preserve allocation-authority expectations; rewrite against new v2 input/output contracts |
| Reporting tests | Conceptually valuable contract tests | Preserve communication-only and row-representation expectations; rewrite |
| Layer unit tests | Mixed: contract plus implementation-specific | Extract expected semantics; rewrite around v2 modules |
| Compatibility wrapper tests | Implementation-specific old tests | Do not force wrappers into v2 |
| Fixture-dependent tests | Fixture-dependent tests | Reapprove fixtures before v2 use |
| Generated-output-dependent tests | Generated-output-dependent tests | Do not make generated outputs v2 source-of-truth |
| SEC/fundamentals tests | Conceptually valuable but old-path tied | Preserve review-model expectations; rewrite after data contracts |
| Old path/import tests | Archive candidates | Do not preserve unless temporary cutover requires them |

Expected conclusion: old tests should be studied, not copied wholesale. v2 tests should be newly written against v2 contracts and approved fixtures.

## 10. Data / CSV / Artifact Classification

No CSV or artifact should be deleted in RESET-0. No CSV should be treated as v2 source-of-truth unless reapproved. v2 must define data contracts before using CSVs. Generated outputs must not be committed as active source files unless explicitly approved.

| Data family | Classification | Notes and recommendation |
|---|---|---|
| `data/processed/` | generated runtime output | Reference-only for current behavior; not v2 source-of-truth by default |
| `data/logs/` | generated runtime output | Preserve for audit/reference; do not use as v2 input |
| `data/intake/` | active manual input / unknown owner decision | Reapprove each file under v2 source-data strategy |
| `data/portfolio/portfolio_transactions.csv` | active manual input | Candidate v2 input, but must be reapproved under v2 contract |
| `data/portfolio/portfolio_positions.csv` | generated runtime output | Not v2 source unless explicitly reclassified |
| `data/portfolio/portfolio_review.csv` | generated runtime output | Not source of Decision Engine authority |
| `data/watchlist/` | legacy/supporting artifact | Archive/delete decision after cutover |
| `data/local/` | local-only ignored data | Should remain local unless explicitly approved |
| `data/raw/` | source-data / local-data mixed | Requires v2 source-data classification before use |
| `reports/` | generated communication output | Generated, not source-of-truth |
| Test fixtures | governed fixture candidate | Reapprove and relocate under v2 fixture strategy |
| Backups/stale CSVs | stale/backup candidate | Classify in RESET-2; do not delete now |
| SEC cache/review outputs | local-only ignored data | Must remain local-only unless sanitized and approved |

## 11. Workflow / CI Classification

`.github/workflows/` should remain untouched in RESET-0. Existing workflows may still be needed for the legacy operational path, Telegram command handling, scheduled scans, or manual validation. However, there is risk if old workflows continue running against old paths after v2 work starts.

Classification:

- Current workflows: KEEP_TEMPORARILY_FOR_LEGACY_RUNS.
- v2 workflows: REWRITE_FOR_V2 after v2 module layout, data contracts, generated-output policy, and run commands exist.
- Old workflows after cutover: ARCHIVE_OR_DELETE_AFTER_CUTOVER.

Workflows should not be changed until v2 exists. A later reset stage must decide whether to pause scheduled legacy runs during v2 implementation, add branch-specific v2 CI, or keep legacy workflows isolated until cutover.

## 12. SEC/Fundamentals Decision

SEC-7F should pause as a standalone old-architecture implementation sprint. It should be absorbed into the rebuild roadmap.

SEC knowledge that remains valuable:

- SEC output is source-data readiness evidence, not pipeline-ready fundamentals.
- Facts must be classified before period filtering.
- Accepted facts, rejected facts, period review, row evidence, ticker evidence, and run summaries must be separated.
- Row-level evidence must contain only row-relevant evidence.
- Run-level summaries must not be repeated across unrelated rows.
- Generated SEC outputs must remain local and uncommitted unless explicitly approved.
- No live SEC calls should occur in tests.
- Missing values are not zero.
- Conflicts remain review-required unless an approved deterministic policy exists.

SEC code should become reference-only for v2. SEC documents should feed RESET-1 source-data strategy, data architecture, and fundamentals roadmap. SEC/fundamentals should be reintroduced only after v2 core contracts and pipeline skeleton exist. SEC data should remain local-only until a sanitized fixture/source-data policy is approved.

## 13. Proposed V2 Repository Structure

Proposed future structure:

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
    sprints/
    audits/
    superseded/
    source_data/
  resets/
    reset_0_full_repository_knowledge_extraction_and_rebuild_decision.md
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

Tracking policy should be explicit:

- Track canonical documentation.
- Track v2 source code.
- Track approved fixtures only.
- Track minimal configuration needed for reproducible local and CI execution.
- Ignore generated data, logs, local SEC cache, raw provider dumps, and generated reports unless explicitly approved.
- Keep legacy docs in a clearly non-authoritative area.

## 14. Proposed Reset Roadmap

| Stage | Purpose | Allowed changes | Forbidden changes | Expected output | Success criteria |
|---|---|---|---|---|---|
| RESET-0 — Full Repository Knowledge Extraction and Rebuild Decision | Decide whether clean rebuild is justified | One decision document | Code/data/workflow/runtime changes | RESET-0 audit document | Clear proceed/continue/insufficient-evidence decision |
| RESET-1 — Canonical Documentation Rewrite | Write new active v2 docs from extracted knowledge | New canonical docs | Runtime implementation, file deletion | `docs/active/*.md` v2 set | Single source-of-truth documentation exists |
| RESET-2 — Repository Structure and Archive Plan | Define archive/delete/move plan | Documentation and plan only | Actual archive/delete unless explicitly approved | Cutover structure plan | Owner-approved migration map |
| RESET-3 — V2 Codebase Bootstrap | Create empty/new v2 module skeleton | New code scaffolding and minimal tests | Reusing old Python files as base | `src/market_scanner/` skeleton | Importable package, no runtime pipeline logic drift |
| RESET-4 — V2 Data Contracts and Fixtures | Define approved data schemas and fixtures | Contracts, fixture files, tests | Live provider calls, generated-output dependence | Fixture strategy and contract tests | Fixtures are explicit and reproducible |
| RESET-5 — V2 Minimal Pipeline Core | Implement discovery-to-classification skeleton | New v2 code and tests | Allocation/reporting/SEC complexity | Minimal deterministic pipeline | Contract tests pass |
| RESET-6 — V2 Decision Engine | Implement v2 final-action authority | New Decision Engine code/tests | Upstream allocation semantics | Decision Engine contract implementation | Allocation authority isolated and tested |
| RESET-7 — V2 Reporting | Implement communication-only reporting | New reporting code/tests | Decision logic in reporting | Reporting outputs from final decisions | Row representation and neutrality tested |
| RESET-8 — V2 Fundamentals / SEC Reintroduction | Reintroduce source-data reviewed fundamentals | New SEC/fundamentals modules, tests, local fixtures | Live SEC tests, pipeline integration before readiness | SEC review model and approved integration path | Review-model evidence separated and stable |
| RESET-9 — Legacy Archive/Delete Cutover | Archive/delete old files after v2 is stable | Approved moves/deletes | Deleting useful knowledge without archive | Clean legacy cutover PR | Old active surfaces removed or clearly legacy |

## 15. RESET-1 Handoff

RESET-1 should create the new canonical documentation set from extracted repository knowledge. It should not implement runtime behavior.

Expected RESET-1 outputs:

- `docs/active/project_charter.md`
- `docs/active/product_vision.md`
- `docs/active/roles_and_responsibilities.md`
- `docs/active/pm_operating_model.md`
- `docs/active/functional_analysis.md`
- `docs/active/technical_architecture.md`
- `docs/active/data_architecture.md`
- `docs/active/source_data_strategy.md`
- `docs/active/pipeline_contract.md`
- `docs/active/decision_engine_contract.md`
- `docs/active/reporting_contract.md`
- `docs/active/testing_strategy.md`
- `docs/active/repository_structure.md`
- `docs/active/backlog.md`
- `docs/active/roadmap.md`

RESET-1 may use old docs as source material. It should not modify old docs except possibly to mark them as legacy after the new canonical docs are created and approved. It should define the new source-of-truth order explicitly.

## 16. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Losing useful knowledge | HIGH | Extract doctrine, contracts, tests, SEC findings, and role boundaries before archive/delete |
| Overbuilding v2 documentation | MEDIUM | Keep RESET-1 canonical docs concise and operational |
| Pausing coding too long | MEDIUM | Use staged reset roadmap with clear success criteria |
| Accidentally preserving old assumptions | HIGH | Require v2 contracts before code and treat old code as reference-only |
| Deleting files too early | HIGH | No archive/delete until RESET-2/RESET-9 approval |
| Codex mixing old and new files | HIGH | Explicit prompts must forbid reusing old Python files as v2 base |
| Tests forcing legacy behavior | MEDIUM | Rewrite tests against v2 contracts, not old paths/wrappers |
| Generated artifacts mistaken as inputs | HIGH | Define data contracts and tracking policy before v2 data use |
| SEC complexity returning too early | HIGH | Pause standalone SEC-7F and reintroduce after core v2 contracts |
| Big-bang cutover failure | HIGH | Keep legacy run path temporarily and use staged cutover |
| Documentation authority conflict | HIGH | RESET-1 must define one source-of-truth order |
| Workflow runs against wrong paths | MEDIUM | Leave workflows unchanged now; redesign after v2 structure exists |

## 17. Backlog Impact Assessment

RESET-0 identifies reset work that should become a new backlog epic or roadmap sequence. The existing backlog should not be modified in RESET-0 because the task authorized only one new reset document and made backlog changes optional.

Draft backlog items for future insertion:

### Draft backlog item: RESET-EPIC — Controlled Clean Rebuild

- Category: Architecture Candidate / Governance / Documentation
- Source document: `docs/resets/reset_0_full_repository_knowledge_extraction_and_rebuild_decision.md`
- Description: Execute the RESET-1 through RESET-9 roadmap to rebuild the active project from certified knowledge while preserving legacy evidence.
- Rationale: Continued incremental patching risks preserving obsolete assumptions and compatibility surfaces.
- Governance risk rating: HIGH
- Effort points: SPLIT FIRST
- Sprint candidate: YES
- Proposed next step: RESET-1 canonical documentation rewrite
- Status: CAPTURED
- Priority: P0
- Owner role: PM / Scrum Master with Technical Analyst and Governance Auditor

### Draft backlog item: SEC-7F Reclassification

- Category: Data Contract / Source Data / Fundamentals
- Source document: `docs/resets/reset_0_full_repository_knowledge_extraction_and_rebuild_decision.md`
- Description: Pause standalone SEC-7F implementation and reclassify SEC review-model work as part of v2 fundamentals/source-data reintroduction.
- Rationale: SEC review-model knowledge remains valuable, but implementation against old architecture risks wasted work.
- Governance risk rating: HIGH
- Effort points: 5
- Sprint candidate: LATER
- Proposed next step: Feed SEC knowledge into RESET-1 source-data strategy and RESET-8 planning
- Status: CAPTURED
- Priority: P1
- Owner role: Data Steward / Financial Analyst / Technical Analyst

### Draft backlog item: Legacy Archive/Delete Cutover Plan

- Category: Documentation / Technical Debt / Repository Structure
- Source document: `docs/resets/reset_0_full_repository_knowledge_extraction_and_rebuild_decision.md`
- Description: Plan archive/delete/move actions only after v2 canonical docs and structure exist.
- Rationale: Old files should not be deleted before their knowledge has been extracted and v2 replacements exist.
- Governance risk rating: MEDIUM
- Effort points: 5
- Sprint candidate: LATER
- Proposed next step: RESET-2 repository structure and archive plan
- Status: CAPTURED
- Priority: P1
- Owner role: Documentation Steward / PM / Technical Analyst

Backlog impact assessment:
- New backlog items identified, but not added to `project_backlog.md` in RESET-0 because the task limited changes to one new reset document.

## 18. Final Decision

RESET-0 Decision: PROCEED_WITH_CONTROLLED_REBUILD

Recommended next action: RESET-1 — Canonical Documentation Rewrite

Development status: Pause new feature work on old active architecture. Allow only maintenance needed to preserve the current legacy run path until v2 cutover.

SEC-7F status: Pause standalone SEC-7F; absorb SEC review-model knowledge into the rebuild roadmap and reintroduce fundamentals/SEC after v2 core contracts and pipeline skeleton exist.

Old code status: Reference-only for v2; do not modify old Python files into v2 code.

Old documentation status: Source material until RESET-1; not future canonical authority unless rewritten or explicitly carried forward.

Old data/CSV status: Do not use as v2 source-of-truth unless reapproved under explicit v2 data contracts.

Archive/delete status: No archive/delete in RESET-0; decide after RESET-2 and execute only after RESET-9 approval.

Pipeline integration status: Blocked until v2 canonical documentation, contracts, data contracts, and skeleton exist.

## Validation

Validation requested:

```bash
git diff --check
git status
```

Validation result: Not run through the GitHub connector because RESET-0 was executed through GitHub API file creation rather than a local working tree. No tests were run. No pipeline was run. No SEC diagnostics were run. No live SEC calls were made.
