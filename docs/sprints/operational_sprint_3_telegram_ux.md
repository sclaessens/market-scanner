# Operational Sprint 3 — Telegram UX & Reporting Usability

## 1. Sprint Purpose

Prepare Operational Sprint 3 for governance review and possible future Codex/local implementation.

This sprint improves the Telegram user experience and reporting usability while preserving the certified reporting boundary: reporting communicates Decision Engine outputs only.

This preparation is documentation-only. It does not authorize implementation, code changes, test changes, generated artifact changes, runtime behavior changes, or reporting contract changes.

## 2. Telegram UX Definition

Telegram UX & Reporting Usability means the communication-layer improvement of how existing Decision Engine outputs and Reporting Layer representations are presented to the operator through Telegram or reporting summaries, without changing decisions, allocation semantics, source rows, runtime authority, or pipeline behavior.

Telegram UX means:

- clearer grouping
- clearer wording
- clearer run and context summary
- better readability
- better operator orientation
- safer distinction between decisions, observations, diagnostics, and missing data
- no semantic reinterpretation

Allowed UX work must be derived from existing Reporting Layer and Decision Engine outputs. It must not create new decision semantics, hidden prioritization, urgency interpretation, conviction language, tradeability language, allocation guidance, or filtering behavior.

## 3. Business / Operator Value

The operator needs Telegram messages and reporting summaries that are easy to read, faithful to source decisions, and useful during daily review.

The goal is to reduce operator friction by making the latest run easier to understand:

- what happened
- whether the pipeline succeeded
- which artifacts were produced
- which Decision Engine actions are present
- which rows require review
- whether diagnostics or missing data warnings exist
- where the full reporting artifact can be inspected

This value is communication value only. It does not make Telegram or Reporting an allocation authority.

## 4. Scope

Included as preparation scope:

- Review current Telegram and reporting usability.
- Define communication-only UX objectives.
- Clarify governance classification before implementation.
- Clarify what Telegram UX means in this architecture.
- Clarify the difference between reporting presentation, Telegram message usability, delivery mechanics, inbound Telegram commands, and Decision Engine authority.
- Define operator questions the Telegram/reporting experience should answer.
- Define candidate UX improvements that are allowed only if they preserve existing semantics.
- Define Codex inspection candidates for future implementation planning.
- Prepare a Codex handoff prompt for future human-approved implementation.

Candidate implementation scope, subject to future approval:

- clearer message title and timestamp
- concise run status summary
- artifact path references
- row count summary
- section grouping by existing `final_action` or another reporting-safe group
- clearer distinction between portfolio rows and opportunity rows if existing source fields support it
- diagnostic or warning section
- truncation notice if applicable
- source artifact reference
- communication-only footer or wording if useful
- improved wording for readability
- improved layout for Telegram constraints

All candidate improvements must be derived from existing reporting or Decision Engine outputs and must not create new semantics.

## 5. Out of Scope

Explicitly out of scope:

- allocation logic changes
- Decision Engine semantic changes
- reporting authority changes
- upstream tradeability
- hidden filtering
- hidden ranking or scoring
- hidden prioritization
- conviction or urgency semantics
- CSV/data changes unless explicitly approved in a later implementation plan
- generated report or generated data commits unless a future implementation plan explicitly approves a fixture or contract artifact
- code changes by ChatGPT
- reporting-based decision modification
- suppressing source decisions for UX convenience
- schema changes without explicit governance approval
- row representation changes without explicit governance approval
- delivery behavior changes without explicit governance approval
- inbound Telegram command handling unless explicitly scoped later

Inbound Telegram command handling remains isolated and is not part of this sprint unless explicitly scoped by a future approved implementation plan.

## 6. Communication-Only Boundary

Reporting communicates only.

Telegram delivery communicates only.

Telegram summaries must not:

- create new decisions
- change final actions
- change source row identity
- change row representation rules unless explicitly approved
- rank, score, prioritize, suppress, or filter opportunities unless existing reporting contracts explicitly allow representation rules
- reinterpret Decision Engine outputs as urgency, conviction, tradeability, or allocation advice
- imply buy, sell, trim, hold, review, or remove authority outside the Decision Engine
- introduce hidden grouping, truncation, ordering, or omission rules

If Telegram output is shortened for readability, the shortening must be governed by explicit reporting-safe rules, represented row counts, and source artifact references.

## 7. Distinction Between Related Surfaces

### Reporting Presentation

Reporting presentation is the formatting and representation of existing Decision Engine outputs. It may group, summarize, format, and communicate only within reporting contracts.

### Telegram Message Usability

Telegram message usability is the operator-facing readability of reporting output in Telegram constraints. It may improve wording, section labels, timestamps, diagnostics, and layout, but it must not change source semantics.

### Delivery Mechanics

Delivery mechanics include sending messages, transport behavior, and operational delivery hygiene. Delivery mechanics must remain separate from reporting semantics and must not create or alter decisions.

### Inbound Telegram Commands

Inbound Telegram command handling remains isolated. It is not part of Operational Sprint 3 unless explicitly scoped later. Inbound commands must not be used to bypass Decision Engine authority or introduce reporting-based decision behavior.

### Decision Engine Authority

The Decision Engine remains the only allocation, execution, arbitration, and final action authority. Telegram and Reporting must only communicate Decision Engine outputs.

## 8. Candidate Operator Questions

The Telegram/reporting UX should help the operator answer:

- What happened in the latest run?
- Was the pipeline successful?
- Which artifacts were produced?
- How many rows were represented?
- Which Decision Engine actions are present?
- Which rows require review?
- Which portfolio-related decisions are present?
- Are there diagnostics or missing data warnings?
- Is this a communication summary or an allocation signal?
- Where can the operator inspect the full reporting artifact?

The answer to the allocation-signal question must remain clear: Telegram/reporting output is a communication summary, not an independent allocation signal.

## 9. Candidate Artifact Scope for Codex Inspection

Future Codex implementation planning may inspect the following artifacts and files:

- `data/processed/final_decisions.csv`
- `data/processed/reporting_dashboard_data.csv`
- `data/logs/reporting_layer_log.csv`
- `reports/daily/telegram_message.txt`
- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `scripts/telegram/process_telegram_commands.py`
- `docs/active/contracts/pipeline_contracts.md`

Inspection does not imply authorization to modify.

Generated reports, generated data, and Telegram output artifacts must not be committed unless a future implementation plan explicitly approves a fixture or contract artifact.

## 10. Governance Classification

Operational Sprint 3 can be Level 1 only if implementation improves wording, layout, runbook/docs, or presentation around existing reporting outputs without changing runtime contracts, schemas, grouping rules, row representation, generated output structure, delivery behavior, or pipeline behavior.

Operational Sprint 3 becomes Level 2 if implementation changes any of the following:

- reporting contracts
- Telegram output schema or structure
- grouping rules
- truncation rules
- ordering rules
- row representation behavior
- generated report artifact conventions
- delivery behavior
- operator-facing runtime output conventions
- source artifact references or reporting log conventions
- any behavior affecting row preservation, determinism, or auditability

Operational Sprint 3 would become Level 3 only if implementation introduces or risks introducing:

- decision semantics outside the Decision Engine
- allocation authority outside the Decision Engine
- urgency interpretation
- upstream tradeability
- hidden filtering
- ranking or scoring authority
- reporting-based decision modification
- any weakening of the classification/allocation/reporting boundary

Recommended classification before implementation:

`Governance Level 2 required before implementation if Telegram/reporting changes alter grouping, truncation, ordering, row representation, generated output structure, delivery behavior, operator-facing runtime output conventions, or reporting contracts.`

Level 3 is not approved.

## 11. Acceptance Criteria

Preparation is successful when this document gives reviewers and Codex a clear basis to evaluate future implementation.

Future implementation, if approved, is successful only if it can show:

- Telegram output is easier for the operator to read and review.
- Source Decision Engine decisions remain unchanged.
- Reporting remains communication-only.
- Source row traceability and represented row counts remain explicit where required.
- No hidden prioritization, ranking, scoring, filtering, conviction, tradeability, or urgency language is introduced.
- No source rows are hidden or suppressed without explicit reporting-safe representation rules.
- Delivery-only Telegram behavior remains isolated from reporting semantics.
- Inbound Telegram command handling remains isolated unless explicitly scoped.
- Generated reports/data are not committed unless explicitly approved.
- Relevant tests pass if code is changed.
- `git diff --check` passes.

## 12. Risks and Controls

Risk: UX improvements could become implicit prioritization.

Control: Use fixed, documented presentation rules based on source order or explicit source fields, not new hidden scoring or ranking.

Risk: Shorter messages could omit important source decisions.

Control: Preserve traceability through explicit represented row counts, examples, grouping rules, source artifact references, and contract-aware truncation where already governed.

Risk: Telegram wording could imply urgency or execution instructions.

Control: Avoid actionable language unless it directly reflects Decision Engine output and remains communication-only.

Risk: Delivery mechanics could become confused with reporting semantics.

Control: Keep sending behavior isolated from message semantics and Decision Engine authority.

Risk: Inbound Telegram command handling could expand scope.

Control: Keep inbound command handling out of scope unless explicitly approved in a future implementation plan.

## 13. Backlog Linkage

Related backlog:

- BL-0006 — Remediate legacy Reporting and Telegram semantic drift

Related categories:

- Reporting
- Runbooks
- Operational Reliability

This sprint may refine Telegram usability and reporting communication, but it does not mark BL-0006 as implemented unless a later approved implementation explicitly completes that backlog item.

No new backlog item is identified by this preparation. Existing backlog coverage is sufficient.

## 14. PM Notes

Priority: P1 after Operational Sprint 2 closeout.

Sequencing: Operational Sprint 3 is the recommended next sprint after the certified closeout of Operational Sprint 2.

Dependencies:

- certified Reporting Layer contract
- current Telegram output example
- current `reporting_dashboard_data.csv`
- current `reporting_layer_log.csv`
- current `final_decisions.csv`
- Governance v2 classification before implementation

Implementation should start only after human review approves the prepared scope and governance classification.

## 15. Scrum Master Execution Notes

Before future implementation:

1. Confirm clean working tree.
2. Review current Telegram message examples.
3. Identify what is confusing, too verbose, missing, or difficult to orient around as an operator.
4. Classify the implementation level under Governance v2.
5. Confirm whether the implementation remains Level 1 or requires Level 2.
6. If Level 2, create the required lightweight design note and contract impact review before code changes.
7. Keep Codex implementation small and targeted.
8. Require before/after output examples after implementation.
9. Require validation evidence before merge.

After Codex implementation, review:

- changed files
- before/after Telegram examples
- row representation and traceability evidence
- validation commands
- test results if code changed
- confirmation that reporting remains communication-only
- confirmation that Decision Engine authority is preserved
- confirmation that no generated data/report artifacts were committed unless explicitly approved

Required validation expectations:

```bash
git status
git diff --stat
git diff --check
```

Relevant tests are required when code changes.

## 16. Codex Implementation Prompt

You are operating inside the institutional `market-scanner` repository.

Implement Operational Sprint 3 — Telegram UX & Reporting Usability only after human approval of the prepared sprint document and governance classification.

This is a Codex/local implementation task. ChatGPT must not author code for this sprint.

Goal: improve Telegram UX and reporting usability while preserving Reporting as communication-only and preserving Decision Engine authority.

Start from a clean working tree.

Before editing, run:

```bash
git status
```

Read:

- `AGENTS.md`
- `docs/active/architecture_current_state.md`
- `docs/active/governance_v2.md`
- `docs/active/operational_development_model.md`
- `docs/active/contracts/pipeline_contracts.md`
- `docs/active/contracts/historical_evidence_contracts.md`
- `docs/active/roadmap_current.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/operational_sprint_1_closeout.md`
- `docs/sprints/operational_sprint_2_closeout.md`
- `docs/sprints/operational_sprint_3_telegram_ux.md`

Inspect as needed:

- `data/processed/final_decisions.csv`
- `data/processed/reporting_dashboard_data.csv`
- `data/logs/reporting_layer_log.csv`
- `reports/daily/telegram_message.txt`
- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `scripts/telegram/process_telegram_commands.py`
- `docs/active/contracts/pipeline_contracts.md`

These are candidate inspection files only. Do not make unrelated changes.

Hard restrictions:

- Do not change allocation logic.
- Do not change Decision Engine semantics.
- Do not give Reporting decision authority.
- Do not introduce upstream tradeability.
- Do not introduce hidden filtering.
- Do not introduce hidden ranking, scoring, prioritization, conviction, or urgency semantics.
- Do not change row representation rules without explicit approval.
- Do not hide or suppress rows.
- Do not alter reporting neutrality.
- Do not change schemas without explicit approval.
- Do not change reporting contracts without explicit approval.
- Do not change grouping, truncation, ordering, generated output structure, or delivery behavior without confirming Governance Level 2 handling.
- Do not commit generated reports/data unless explicitly approved.
- Keep inbound Telegram command handling isolated unless explicitly scoped.
- Do not make unrelated refactors.

Propose and implement the smallest safe change that improves operator readability.

If code changes are made, run relevant tests.

After implementation, report:

```bash
git status
git diff --stat
git diff --check
```

Also report:

- files changed
- whether any generated data/report artifacts changed
- before/after Telegram examples
- row representation and traceability evidence
- tests run and results
- confirmation that Decision Engine authority is preserved
- confirmation that Reporting remains communication-only

Do not commit until human review approves the diff.

## 17. Preparation Status

Preparation status: PREPARATION COMPLETE / READY FOR REVIEW

Current phase: PREPARATION

Governance status: GOVERNANCE REVIEW REQUIRED

This document does not mark Operational Sprint 3 as implementation, closed, or certified complete.
