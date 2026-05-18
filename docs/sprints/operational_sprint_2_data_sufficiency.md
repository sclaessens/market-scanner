# Operational Sprint 2 — Data Sufficiency & Historical Storage Baseline

## 1. Sprint Purpose

Define and prepare a baseline for collecting enough historical and operational data to support future analysis of pipeline behavior, decisions, outcomes, and diagnostics.

This sprint is about data sufficiency planning, historical storage boundaries, evidence questions, and observational scope. It does not authorize new runtime data models, generated CSV changes, schema changes, persistent storage behavior, reporting runtime changes, or Decision Engine consumption.

## 2. Business / Operator Value

The operator needs enough historical and operational evidence to evaluate whether the system is improving, where it fails, and which decisions or classifications deserve later review. Without a clear storage baseline, future analysis risks being incomplete, inconsistent, or too dependent on ad hoc local files.

Operational Sprint 1 improved terminal visibility during scans and pipeline execution. Operational Sprint 2 prepares the next baseline: ensuring future analysis can connect runs, artifacts, classifications, decisions, reporting representation, and later outcomes without changing live decision-making.

## 3. Data Sufficiency Definition

Data sufficiency means the minimum reliable, traceable, and deterministic evidence needed to analyze past pipeline behavior, decisions, classifications, and outcomes without affecting live decision-making.

For this project, sufficient evidence should eventually allow a reviewer to reconstruct the observational path from scan to classification to Decision Engine output to reporting representation, and then compare that historical decision or classification with later observed outcomes.

Data sufficiency is not strategy optimization, allocation authority, ranking authority, or live signal generation. It is an evidence baseline for diagnostics, auditability, research, and future governed proposals.

## 4. Observational Boundary

Historical and analytical outputs are observational unless a future governed change explicitly authorizes another role.

Confirmed boundaries:

- historical storage is observational
- diagnostics are observational
- prediction tracking is observational
- learning loops are observational
- historical analysis may support review, diagnostics, research, and future proposals
- historical analysis must not become allocation authority
- historical analysis must not create upstream tradeability
- historical analysis must not create hidden filtering
- historical analysis must not create hidden ranking, scoring, prioritization, conviction, or urgency semantics
- none of these outputs may affect live decisions unless a future governed Decision Engine change explicitly approves it

Any future attempt to route historical or observational outputs into live decisions requires explicit governance review and Decision Engine approval.

## 5. Scope

Included:

- Identify the minimum historical and operational data questions the platform should eventually answer.
- Define what data sufficiency means for future analysis.
- Clarify observational versus decision-making data boundaries.
- Identify candidate artifacts and logs for Codex review in a future implementation sprint.
- Clarify governance classification before implementation.
- Prepare a constrained Codex implementation handoff.
- Preserve current contracts unless a later governed implementation explicitly changes them.

## 6. Out of Scope

Explicitly out of scope:

- allocation logic changes
- Decision Engine semantic changes unless a future governed sprint explicitly targets Decision Engine implementation
- reporting authority changes
- reporting runtime changes
- upstream tradeability
- hidden filtering
- hidden ranking or scoring
- CSV/data changes unless explicitly approved for a later implementation sprint
- code changes by ChatGPT
- test changes by ChatGPT
- generated file changes by ChatGPT
- GitHub Actions changes
- new persistent stores through documentation alone
- schema changes through documentation alone
- making historical analysis actionable without governed Decision Engine approval

## 7. Candidate Evidence Questions

A future governed implementation should be scoped around the smallest safe evidence set needed to answer questions such as:

- Which scan or pipeline run produced this decision?
- Which ticker/date row produced this decision?
- Which upstream layer classifications were present?
- What final Decision Engine action was produced?
- What source artifact and row identity support the decision?
- Was the row represented downstream in reporting?
- What later outcome could be compared with the original decision?
- Which artifacts are needed to audit the path from scan to decision to report?
- What decisions were produced during a run?
- What upstream classifications led to those decisions?
- Which artifacts were written during the run?
- Which rows were represented downstream?
- Which diagnostics are needed to explain missing, invalid, duplicated, or unrepresented rows?

These questions define analysis needs only. They do not define implementation, schemas, new storage, or Decision Engine consumption.

## 8. Candidate Artifact Scope for Codex Inspection

Codex may inspect the following candidate artifacts during implementation planning, but this sprint document does not authorize automatic modification of them:

- `data/processed/scanner_ranked.csv`
- `data/processed/validation_layer.csv`
- `data/processed/context_strength.csv`
- `data/processed/fundamental_quality.csv`
- `data/processed/timing_state_layer.csv`
- `data/processed/portfolio_intelligence.csv`
- `data/processed/final_decisions.csv`
- `data/processed/reporting_dashboard_data.csv`
- `data/logs/*`
- existing run or stability logs if present

Candidate inspection should determine whether existing artifacts already contain enough evidence, where row identity is available, where run identity is missing, and whether any future persistent artifact or contract note is needed.

Generated files must not be committed unless a future implementation plan explicitly approves a source-controlled fixture or contract artifact.

## 9. Governance Boundaries

This sprint must preserve:

- classification upstream
- allocation downstream
- Decision Engine = only allocation authority
- reporting communicates only
- no hidden filtering
- no upstream tradeability
- deterministic architecture
- row preservation
- auditability
- separation of concerns
- governance-first engineering

Historical storage and operational intelligence are observational by default. They may support diagnostics, research, and future proposals, but they must not create hidden allocation, scoring, ranking, filtering, or upstream tradeability semantics.

## 10. Governance Classification

Operational Sprint 2 is documentation preparation until implementation is separately approved.

Likely classification:

- Governance Level 1 if the work only improves documentation, runbooks, or visibility around existing artifacts without changing runtime contracts, schemas, persistent artifacts, row identity, or pipeline behavior.
- Governance Level 2 if implementation creates or changes persistent storage, schemas, row identity contracts, historical artifacts, artifact retention behavior, pipeline observability contracts, or reporting representation contracts.
- Governance Level 3 only if historical or observational outputs are routed into Decision Engine authority, allocation semantics, execution semantics, upstream tradeability, hidden filtering, or live decision influence.

Recommended classification:

`Governance Level 2 required before implementation if any persistent artifact, schema, or historical storage behavior is introduced.`

Level 3 is not approved for this sprint.

## 11. Expected Impacted Areas

Likely candidate files for Codex review, not authorized changes:

- existing processed artifacts listed in section 8
- existing logs under `data/logs/`
- existing pipeline builders that already write logs or processed artifacts
- possible docs: `docs/active/contracts/pipeline_contracts.md`, future design notes, or future runbooks
- possible tests: contract tests if a future implementation adds or changes storage behavior

Any schema or persistent artifact change is at least Governance Level 2 and must be reviewed before implementation.

## 12. Acceptance Criteria

The sprint preparation is successful when it provides:

- an explicit data sufficiency definition
- an explicit observational boundary
- candidate evidence questions for future analysis
- candidate artifact scope for Codex inspection
- governance classification guidance
- a constrained Codex implementation prompt
- confirmation that the sprint does not authorize implementation by itself

A future approved implementation can be considered successful only when it can show:

- data sufficiency requirements are explicit and reviewable
- observational artifacts are clearly separated from allocation authority
- candidate storage outputs preserve source traceability
- new or changed schemas, if any, are documented and governed before implementation
- no historical analysis output is consumed by the Decision Engine unless separately approved
- no hidden filtering, ranking, scoring, or tradeability semantics are introduced
- relevant tests pass if code is changed
- `git diff --check` passes

## 13. Risks and Controls

Risk: Historical performance data could become de facto allocation logic.

Control: Treat all historical storage and analysis as observational unless a future Level 3 governed Decision Engine change explicitly consumes it.

Risk: New storage could weaken row traceability or create duplicate identity ambiguity.

Control: Require explicit ticker/date/run identity and deterministic ordering before implementation.

Risk: CSV/data artifacts could be changed without contract review.

Control: Any new or changed persistent artifact requires Governance v2 classification and contract impact review before implementation.

Risk: Generated artifacts could be committed accidentally during local validation.

Control: Generated data, logs, and reports must not be committed unless a future implementation plan explicitly approves a source-controlled fixture or contract artifact.

Risk: Prediction tracking or learning-loop language could imply live allocation influence.

Control: Keep all prediction tracking, diagnostics, and learning loops observational until a future governed Decision Engine change explicitly approves consumption.

## 14. Backlog Linkage

Related backlog:

- BL-0009 — Define operational intelligence storage and observability scope
- BL-0010 — Frame prediction tracking and feedback loops as observational research

Related categories:

- Operational Intelligence
- Observational Research
- Data Contract
- Contracts

This sprint supports planning for BL-0009 and respects BL-0010, but it does not mark either item as implemented.

## 15. PM Notes

Priority: P1, after Operational Sprint 1.

Sequencing: This should follow scan visibility because better run feedback helps verify what data is actually produced during operational runs.

Preparation recommendation: Operational Sprint 2 should proceed to review as a lightweight planning artifact, but any implementation introducing persistent storage, schemas, row identity contracts, historical artifacts, or retention behavior should be treated as Governance Level 2 before Codex changes runtime files.

Dependencies: Requires the current pipeline contract map and a clear distinction between observational diagnostics and Decision Engine authority.

## 16. Scrum Master Execution Notes

Prepare the sprint by listing the practical questions the operator wants future analysis to answer, such as what was predicted, what decision was emitted, what happened later, which run created the relevant artifact, which rows were represented downstream, and which artifacts prove the path from scan to decision to report.

Codex needs the approved sprint document, active architecture documentation, Governance v2, pipeline contracts, roadmap, backlog items BL-0009 and BL-0010, and the Operational Sprint 1 closeout before implementation.

After Codex implementation, review:

- changed files
- any schema or artifact changes
- contract impact notes
- validation commands
- test results if code changed
- confirmation that the output remains observational and non-authoritative
- confirmation that generated data, logs, and reports were not committed unless explicitly approved

Required validation expectations:

- `git status`
- `git diff --stat`
- `git diff --check`
- relevant tests when code is changed

## 17. Codex Handoff Readiness

This document is ready for review as an Operational Sprint 2 preparation artifact.

It does not approve implementation by itself. Human review should confirm whether Codex should proceed with an implementation proposal and whether the expected implementation is Level 1 or Level 2.

Codex may use section 18 as the implementation handoff only after human approval.

## 18. Codex Implementation Prompt

You are operating inside the institutional `market-scanner` repository.

Implement Operational Sprint 2 — Data Sufficiency & Historical Storage Baseline only after human approval of this sprint document and any required Governance v2 classification.

This is a Codex/local implementation task. ChatGPT must not author code for this sprint.

Goal: propose or implement the smallest approved baseline that improves historical and operational data sufficiency for future analysis while preserving observational-only semantics unless a separate governed Decision Engine change explicitly consumes the data.

Before editing, start from a clean working tree and run:

```bash
git status
```

Read:

- `AGENTS.md`
- `docs/active/architecture_current_state.md`
- `docs/active/governance_v2.md`
- `docs/active/operational_development_model.md`
- `docs/active/contracts/pipeline_contracts.md`
- `docs/active/roadmap_current.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/operational_sprint_1_closeout.md`
- `docs/sprints/operational_sprint_2_data_sufficiency.md`

Inspect existing artifacts and logs as needed:

- `data/processed/scanner_ranked.csv`
- `data/processed/validation_layer.csv`
- `data/processed/context_strength.csv`
- `data/processed/fundamental_quality.csv`
- `data/processed/timing_state_layer.csv`
- `data/processed/portfolio_intelligence.csv`
- `data/processed/final_decisions.csv`
- `data/processed/reporting_dashboard_data.csv`
- `data/logs/*`
- existing run or stability logs if present

Inspection does not authorize modification. Generated files must not be committed unless the approved implementation plan explicitly identifies a source-controlled fixture or contract artifact.

Implementation constraints:

- Propose the smallest safe implementation.
- Do not change allocation logic.
- Do not change Decision Engine semantics.
- Do not route observational data into the Decision Engine unless separately governed and approved.
- Do not introduce upstream tradeability.
- Do not introduce hidden filtering.
- Do not introduce hidden ranking, scoring, prioritization, conviction, or urgency semantics.
- Do not change schemas without explicit approval.
- Do not change CSV/data contracts without explicit Governance v2 classification and approval.
- Do not commit generated data unless explicitly approved.
- Do not make unrelated refactors.
- Preserve deterministic architecture, row preservation, auditability, and separation of concerns.

If the proposed implementation introduces any persistent artifact, schema, row identity contract, historical storage behavior, retention behavior, or reporting representation contract, classify it as Governance Level 2 before implementation.

If the proposed implementation routes observational outputs into live decisions or Decision Engine authority, stop. That would require Level 3 governance and is not approved in this sprint.

After implementation, report:

```bash
git status
git diff --stat
git diff --check
```

Run relevant tests when code is changed. Do not commit until human review approves the diff.