# Operational Sprint 2 — Data Sufficiency & Historical Storage Baseline

## 1. Sprint Purpose

Define and prepare a baseline for collecting enough historical and operational data to support future analysis of pipeline behavior, decisions, outcomes, and diagnostics.

This sprint is about data sufficiency planning and observational storage boundaries. It does not authorize new runtime data models, generated CSV changes, or Decision Engine consumption.

## 2. Business / Operator Value

The operator needs enough historical and operational evidence to evaluate whether the system is improving, where it fails, and which decisions or classifications deserve later review. Without a clear storage baseline, future analysis risks being incomplete, inconsistent, or too dependent on ad hoc local files.

## 3. Scope

Included:

- Identify the minimum historical and operational data questions the platform should eventually answer.
- Define what data sufficiency means for future analysis.
- Clarify observational versus decision-making data boundaries.
- Identify candidate artifacts and logs for Codex review in a future implementation sprint.
- Preserve current contracts unless a later governed implementation explicitly changes them.

## 4. Out of Scope

Explicitly out of scope:

- allocation logic changes
- Decision Engine semantic changes unless a future governed sprint explicitly targets Decision Engine implementation
- reporting authority changes
- upstream tradeability
- hidden filtering
- hidden ranking or scoring
- CSV/data changes unless explicitly approved for a later implementation sprint
- code changes by ChatGPT
- new persistent stores through documentation alone
- schema changes through documentation alone
- making historical analysis actionable without governed Decision Engine approval

## 5. Governance Boundaries

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

## 6. Expected Impacted Areas

Likely candidate files for Codex review, not authorized changes:

- possible scripts: pipeline builders that already write logs or processed artifacts
- possible docs: `docs/active/contracts/pipeline_contracts.md`, future design notes, future runbooks
- possible logs: `data/logs/` artifacts
- possible reporting outputs: none unless communication-only summaries are explicitly approved
- possible tests: contract tests if implementation adds or changes storage behavior

Any schema or persistent artifact change is at least Governance Level 2 and must be reviewed before implementation.

## 7. Acceptance Criteria

The sprint is successful when a future approved implementation can show:

- data sufficiency requirements are explicit and reviewable
- observational artifacts are clearly separated from allocation authority
- candidate storage outputs preserve source traceability
- new or changed schemas, if any, are documented and governed before implementation
- no historical analysis output is consumed by the Decision Engine unless separately approved
- no hidden filtering, ranking, scoring, or tradeability semantics are introduced
- relevant tests pass if code is changed
- `git diff --check` passes

## 8. Risks and Controls

Risk: Historical performance data could become de facto allocation logic.

Control: Treat all historical storage and analysis as observational unless a future Level 3 governed Decision Engine change explicitly consumes it.

Risk: New storage could weaken row traceability or create duplicate identity ambiguity.

Control: Require explicit ticker/date/run identity and deterministic ordering before implementation.

Risk: CSV/data artifacts could be changed without contract review.

Control: Any new or changed persistent artifact requires Governance v2 classification and contract impact review before implementation.

## 9. Backlog Linkage

Related backlog:

- BL-0009 — Define operational intelligence storage and observability scope

Related categories:

- Operational Intelligence
- Data Contract
- Contracts

This sprint supports planning for BL-0009 but does not mark it as implemented.

## 10. PM Notes

Priority: P1, after Sprint 1.

Sequencing: This should follow scan visibility because better run feedback helps verify what data is actually produced during operational runs.

Dependencies: Requires the current pipeline contract map and a clear distinction between observational diagnostics and Decision Engine authority.

## 11. Scrum Master Execution Notes

Prepare the sprint by listing the practical questions the operator wants future analysis to answer, such as what was predicted, what decision was emitted, what happened later, and which step created the relevant artifact.

Codex needs the approved sprint document, active architecture documentation, Governance v2, pipeline contracts, and backlog item BL-0009 before implementation.

After Codex implementation, review:

- changed files
- any schema or artifact changes
- contract impact notes
- validation commands
- test results if code changed
- confirmation that the output remains observational and non-authoritative

Required validation expectations:

- `git status`
- `git diff --stat`
- `git diff --check`
- relevant tests when code is changed

## 12. Codex Implementation Prompt

You are operating inside the institutional `market-scanner` repository.

Implement Operational Sprint 2 — Data Sufficiency & Historical Storage Baseline only after human approval of this sprint document and any required Governance v2 classification.

This is a Codex/local implementation task. ChatGPT must not author code for this sprint.

Goal: prepare or implement the smallest approved baseline that improves historical and operational data sufficiency for future analysis while preserving observational-only semantics unless a separate governed Decision Engine change explicitly consumes the data.

Before editing, run:

```bash
git status
```

Read:

- `AGENTS.md`
- `docs/active/architecture_current_state.md`
- `docs/active/governance_v2.md`
- `docs/active/contracts/pipeline_contracts.md`
- `docs/active/roadmap_current.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/operational_sprint_2_data_sufficiency.md`

Likely candidate areas for review include existing processed artifacts, logs, pipeline contracts, and any existing historical outputs. These are candidates only; do not make unrelated changes.

Hard restrictions:

- Do not change allocation logic.
- Do not change Decision Engine semantics.
- Do not route observational data into the Decision Engine unless separately governed and approved.
- Do not introduce upstream tradeability.
- Do not introduce hidden filtering.
- Do not introduce hidden ranking, scoring, prioritization, conviction, or urgency semantics.
- Do not change CSV/data contracts without explicit Governance v2 classification and approval.
- Do not make unrelated refactors.
- Preserve deterministic architecture, row preservation, auditability, and separation of concerns.

After implementation, report:

```bash
git status
git diff --stat
git diff --check
```

Run relevant tests when code is changed. Do not commit until human review approves the diff.