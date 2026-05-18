# Operational Sprint 3 — Telegram UX & Reporting Usability

## 1. Sprint Purpose

Improve the Telegram user experience and reporting usability while preserving the certified reporting boundary: reporting communicates Decision Engine outputs only.

This sprint is about clarity, readability, grouping, operator comprehension, and delivery experience. It is not about changing decisions, urgency, allocation, ranking, or source semantics.

## 2. Business / Operator Value

The operator needs Telegram messages that are easy to read, actionable only in the sense of communication clarity, and faithful to the underlying Decision Engine outputs. Better UX reduces friction during daily review without giving reporting independent decision authority.

## 3. Scope

Included:

- Review Telegram message structure and operator readability.
- Identify communication-only UX improvements.
- Define clearer grouping, labels, examples, or summaries if they preserve source traceability.
- Keep reporting downstream and non-authoritative.
- Prepare Codex for later implementation if approved.

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
- reporting-based decision modification
- urgency interpretation
- suppressing source decisions for UX convenience

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

Telegram UX may improve presentation, ordering clarity, and readability, but it must not create decision authority, change source final actions, imply hidden prioritization, or omit rows in ways that weaken traceability.

## 6. Expected Impacted Areas

Likely candidate files for Codex review, not authorized changes:

- possible scripts: `scripts/reporting/build_reporting_layer.py`, `scripts/reporting/build_telegram_summary.py`, `scripts/reporting/send_telegram.py`
- possible docs: reporting runbooks, `docs/active/contracts/pipeline_contracts.md` if reporting contracts are clarified
- possible logs: `data/logs/reporting_layer_log.csv`
- possible reporting outputs: `reports/daily/telegram_message.txt`, `data/processed/reporting_dashboard_data.csv`
- possible tests: reporting and Telegram summary tests

## 7. Acceptance Criteria

The sprint is successful when a future approved implementation can show:

- Telegram output is easier for the operator to read and review.
- Source Decision Engine decisions remain unchanged.
- Reporting remains communication-only.
- Source row traceability and represented row counts remain explicit where required.
- No hidden prioritization, ranking, scoring, filtering, or urgency language is introduced.
- Delivery-only Telegram behavior remains isolated from reporting semantics.
- Relevant tests pass if code is changed.
- `git diff --check` passes.

## 8. Risks and Controls

Risk: UX improvements could become implicit prioritization.

Control: Use fixed, documented presentation rules based on source order or explicit source fields, not new hidden scoring or ranking.

Risk: Shorter messages could omit important source decisions.

Control: Preserve traceability through explicit represented row counts, examples, grouping rules, and contract-aware truncation where already governed.

Risk: Telegram wording could imply urgency or execution instructions.

Control: Avoid new actionable language unless it directly reflects Decision Engine output and remains communication-only.

## 9. Backlog Linkage

Related backlog:

- BL-0006 — Remediate legacy Reporting and Telegram semantic drift

Related categories:

- Reporting
- Runbooks
- Operational Reliability

This sprint may refine Telegram usability and reporting communication, but it does not mark BL-0006 as implemented unless a later approved implementation explicitly completes that backlog item.

## 10. PM Notes

Priority: P1, after data sufficiency planning unless Telegram usability becomes the immediate operator bottleneck.

Sequencing: This sprint can run after Sprint 1 and either before or after Sprint 2 if operator feedback shows Telegram UX is the highest-value improvement.

Dependencies: Requires current Reporting Layer contract and examples of current Telegram output.

## 11. Scrum Master Execution Notes

Prepare the sprint by collecting current Telegram message examples and identifying what is confusing, too verbose, missing, or difficult to act on as an operator.

Codex needs the approved sprint document, active architecture documentation, Governance v2, pipeline contracts, and reporting tests before implementation.

After Codex implementation, review:

- changed files
- before/after Telegram examples
- row representation and traceability evidence
- validation commands
- test results if code changed
- confirmation that reporting remains communication-only

Required validation expectations:

- `git status`
- `git diff --stat`
- `git diff --check`
- relevant tests when code is changed

## 12. Codex Implementation Prompt

You are operating inside the institutional `market-scanner` repository.

Implement Operational Sprint 3 — Telegram UX & Reporting Usability only after human approval of this sprint document.

This is a Codex/local implementation task. ChatGPT must not author code for this sprint.

Goal: improve Telegram UX and reporting usability while preserving reporting as communication-only and preserving Decision Engine authority.

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
- `docs/sprints/operational_sprint_3_telegram_ux.md`

Likely candidate files for review include reporting builders, Telegram summary generation, Telegram delivery code, and reporting tests. These are candidates only; do not make unrelated changes.

Hard restrictions:

- Do not change allocation logic.
- Do not change Decision Engine semantics.
- Do not give reporting decision authority.
- Do not introduce upstream tradeability.
- Do not introduce hidden filtering.
- Do not introduce hidden ranking, scoring, prioritization, conviction, or urgency semantics.
- Do not suppress source decisions for presentation convenience unless an existing reporting contract explicitly permits traceable grouping or truncation.
- Do not make unrelated refactors.
- Preserve deterministic architecture, row preservation, auditability, and separation of concerns.

After implementation, report:

```bash
git status
git diff --stat
git diff --check
```

Run relevant tests when code is changed. Do not commit until human review approves the diff.