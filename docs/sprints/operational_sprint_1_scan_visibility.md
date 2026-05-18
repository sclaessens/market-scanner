# Operational Sprint 1 — Scan Visibility & Operator Feedback

## 1. Sprint Purpose

Improve operator visibility during local and scheduled scan runs so the user can understand what the system is doing, where it is in the pipeline, and whether the run completed cleanly.

This sprint is about operational feedback and run transparency. It is not about changing strategy, allocation, ranking, filtering, or Decision Engine semantics.

## 2. Business / Operator Value

The operator needs better terminal feedback during scans to reduce uncertainty during long or multi-step runs. Clear run progress, summarized outcomes, and visible failure points make daily operation easier and reduce unnecessary debugging time.

## 3. Scope

Included:

- Review current scan and pipeline terminal output from an operator perspective.
- Identify where progress messages, run summaries, or failure diagnostics may be improved.
- Define lightweight operator-facing visibility expectations.
- Preserve deterministic pipeline behavior and all existing authority boundaries.
- Prepare Codex for a later local implementation sprint if approved.

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
- strategy redesign
- threshold tuning
- pipeline behavior changes through documentation alone

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

Terminal visibility may explain what the pipeline is doing, but it must not create new decision semantics, rank opportunities, label urgency, or imply allocation authority outside the Decision Engine.

## 6. Expected Impacted Areas

Likely candidate files for Codex review, not authorized changes:

- possible scripts: `scripts/run_full_pipeline.py`, scanner entry points, core builder scripts that already emit logs or status messages
- possible docs: `docs/active/runbooks/local_development.md`, future operational runbooks
- possible logs: existing files under `data/logs/`
- possible reporting outputs: none expected unless communication-only run summaries are explicitly approved
- possible tests: focused tests only if implementation changes terminal output behavior or helper functions

## 7. Acceptance Criteria

The sprint is successful when a future approved implementation can show:

- `git status` is clean before implementation begins.
- Operator-facing scan output clearly communicates pipeline progress.
- Failures identify the failing step without changing runtime semantics.
- Successful runs provide a concise completion summary.
- No rows are filtered or reinterpreted for visibility purposes.
- No Decision Engine, reporting authority, or allocation semantics are changed.
- Relevant tests pass if code is changed.
- `git diff --check` passes.

## 8. Risks and Controls

Risk: Progress output could accidentally introduce prioritization or urgency language.

Control: Use neutral operational language such as started, completed, skipped, failed, row count, and artifact written. Avoid buy, sell, urgency, tradeable, conviction, rank, score, or priority language outside existing governed outputs.

Risk: Visibility changes could hide failures by converting fail-fast behavior into warnings.

Control: Preserve existing fail-fast behavior unless a separately governed runtime reliability change approves different error handling.

Risk: Terminal summaries could be mistaken for reporting output.

Control: Keep scan visibility operational and diagnostic only. Reporting remains communication-only and downstream of Decision Engine outputs.

## 9. Backlog Linkage

Related backlog:

- BL-0008 — Add operational runbooks beyond local development

Related categories:

- Operational Reliability
- Developer Experience
- Runbooks

This sprint may also support future runbook updates, but it does not mark BL-0008 as implemented.

## 10. PM Notes

Priority: P1 for the next operational phase.

Sequencing: This should be the first operational sprint because better scan visibility improves every later implementation and validation cycle.

Dependencies: Requires review of current local scan workflow and operator pain points. No data model or Decision Engine dependency is expected.

## 11. Scrum Master Execution Notes

Prepare the sprint by confirming the exact operator workflow to observe, including local scan command, full pipeline command, and expected daily usage.

Codex needs the approved sprint document, active architecture documentation, Governance v2, pipeline contracts, and local development runbook before implementation.

After Codex implementation, review:

- changed files
- terminal output examples
- validation commands
- test results if code changed
- confirmation that no allocation, ranking, filtering, or Decision Engine semantics changed

Required validation expectations:

- `git status`
- `git diff --stat`
- `git diff --check`
- relevant tests when code is changed

## 12. Codex Implementation Prompt

You are operating inside the institutional `market-scanner` repository.

Implement Operational Sprint 1 — Scan Visibility & Operator Feedback.

This is a Codex/local implementation task only after human approval of this sprint document. ChatGPT must not author code for this sprint.

Goal: improve terminal/operator visibility during scan and pipeline runs without changing strategy, allocation, Decision Engine semantics, reporting authority, data contracts, row preservation, or pipeline meaning.

Before editing, run:

```bash
git status
```

Read:

- `AGENTS.md`
- `docs/active/architecture_current_state.md`
- `docs/active/governance_v2.md`
- `docs/active/contracts/pipeline_contracts.md`
- `docs/active/runbooks/local_development.md`
- `docs/sprints/operational_sprint_1_scan_visibility.md`

Likely candidate files for review include `scripts/run_full_pipeline.py`, scanner entry points, and existing logging/status output locations. These are candidates only; do not make unrelated changes.

Hard restrictions:

- Do not change allocation logic.
- Do not change Decision Engine semantics.
- Do not change reporting authority.
- Do not introduce upstream tradeability.
- Do not introduce hidden filtering.
- Do not introduce hidden ranking, scoring, prioritization, conviction, or urgency semantics.
- Do not change generated CSV/data files as source changes.
- Do not make unrelated refactors.
- Preserve deterministic architecture, row preservation, auditability, and separation of concerns.

After implementation, report:

```bash
git status
git diff --stat
git diff --check
```

Run relevant tests when code is changed. Do not commit until human review approves the diff.