# Operational Sprint 4 — Prediction Tracking & Learning Loop Preparation

## 1. Sprint Purpose

Prepare observational prediction tracking and learning-loop analysis so earlier decisions, classifications, and outcomes can be reviewed over time without creating new allocation authority.

This sprint is about learning from system behavior. It does not make predictions actionable, change the Decision Engine, or create upstream scoring, ranking, filtering, or tradeability semantics.

## 2. Business / Operator Value

The operator wants to understand whether earlier predictions, decisions, and classifications led to useful outcomes. A governed observational learning loop can reveal patterns, mistakes, missing context, and future improvement opportunities without compromising the certified architecture.

## 3. Scope

Included:

- Define prediction tracking as observational research.
- Identify what earlier decisions or classifications may be compared with later outcomes.
- Clarify what learning-loop outputs may and may not mean.
- Identify candidate artifacts, logs, and future analysis outputs for Codex review.
- Preserve the Decision Engine as the only allocation authority.

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
- making prediction tracking actionable
- changing portfolio actions based on historical outcomes
- creating upstream performance gates
- strategy optimization or threshold tuning

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

Prediction tracking and learning-loop outputs are observational by default. They may inform future research questions, diagnostics, design notes, or governed Decision Engine proposals, but they must not independently affect live decisions.

## 6. Expected Impacted Areas

Likely candidate files for Codex review, not authorized changes:

- possible scripts: future analysis scripts or existing pipeline/log readers if approved
- possible docs: research notes, future design notes, `docs/active/roadmap_current.md`
- possible logs: Decision Engine logs, reporting logs, stability logs, historical scan logs
- possible reporting outputs: research-only summaries, not live decision reporting unless explicitly governed
- possible tests: analysis-output tests or contract tests if implementation creates deterministic research artifacts

Any new analysis artifact or persistent store requires Governance v2 classification before implementation.

## 7. Acceptance Criteria

The sprint is successful when a future approved implementation can show:

- prediction tracking outputs are clearly labeled observational or research-only
- historical decisions/classifications can be traced to later outcomes where data exists
- no live allocation, execution, or final-action semantics are changed
- no upstream hidden scoring, ranking, filtering, tradeability, or urgency is introduced
- any new artifact has documented purpose, source, and non-authoritative meaning
- relevant tests pass if code is changed
- `git diff --check` passes

## 8. Risks and Controls

Risk: Prediction tracking could become hidden Decision Engine input.

Control: Do not route research outputs into the Decision Engine unless a future Level 3 governed change explicitly approves it.

Risk: Research metrics could be interpreted as ranking or tradeability.

Control: Label outputs as observational, avoid action labels, and document that they do not authorize allocation.

Risk: Outcome analysis could create look-ahead bias or misleading conclusions.

Control: Require explicit dates, source artifacts, and audit trail for any future analysis implementation.

## 9. Backlog Linkage

Related backlog:

- BL-0010 — Frame prediction tracking and feedback loops as observational research

Related categories:

- Observational Research
- Operational Intelligence

This sprint supports planning for BL-0010 but does not mark it as implemented.

## 10. PM Notes

Priority: P1, but sequenced after foundational visibility and data sufficiency work.

Sequencing: This sprint should normally happen after Sprint 2 because prediction tracking depends on sufficient historical and operational data. It may also benefit from Sprint 3 if Telegram/reporting examples are used for operator review.

Dependencies: Requires clear data sufficiency boundaries, source traceability, and explicit non-authoritative research semantics.

## 11. Scrum Master Execution Notes

Prepare the sprint by defining the first research questions to answer, such as which earlier decisions were followed, reviewed, or contradicted by later outcomes.

Codex needs the approved sprint document, active architecture documentation, Governance v2, pipeline contracts, roadmap, backlog item BL-0010, and any approved data sufficiency notes before implementation.

After Codex implementation, review:

- changed files
- research-output labels and semantics
- artifact lineage and timestamp handling
- validation commands
- test results if code changed
- confirmation that no live decision pathway consumes the output

Required validation expectations:

- `git status`
- `git diff --stat`
- `git diff --check`
- relevant tests when code is changed

## 12. Codex Implementation Prompt

You are operating inside the institutional `market-scanner` repository.

Implement Operational Sprint 4 — Prediction Tracking & Learning Loop Preparation only after human approval of this sprint document and any required Governance v2 classification.

This is a Codex/local implementation task. ChatGPT must not author code for this sprint.

Goal: prepare or implement the smallest approved observational prediction tracking and learning-loop baseline for reviewing earlier decisions, classifications, and outcomes without changing live allocation behavior.

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
- `docs/sprints/operational_sprint_4_prediction_tracking.md`

Likely candidate areas for review include Decision Engine logs, final decision artifacts, stability logs, historical scan logs, and future research-only analysis outputs. These are candidates only; do not make unrelated changes.

Hard restrictions:

- Do not change allocation logic.
- Do not change Decision Engine semantics.
- Do not route prediction tracking outputs into live decisions unless separately governed and approved.
- Do not give reporting decision authority.
- Do not introduce upstream tradeability.
- Do not introduce hidden filtering.
- Do not introduce hidden ranking, scoring, prioritization, conviction, or urgency semantics.
- Do not tune strategy thresholds.
- Do not make unrelated refactors.
- Preserve deterministic architecture, row preservation, auditability, and separation of concerns.

After implementation, report:

```bash
git status
git diff --stat
git diff --check
```

Run relevant tests when code is changed. Do not commit until human review approves the diff.