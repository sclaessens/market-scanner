# Roles and Responsibilities

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines v2 role authority for the market-scanner reset and rebuild sequence.

All roles inherit the certified doctrine:

- classification upstream;
- allocation downstream;
- Decision Engine as only allocation authority;
- Reporting communicates only;
- no hidden filtering;
- no upstream tradeability;
- no generated artifact as source of truth unless approved;
- English-only repository content.

## Role Matrix

| Role | Owns | Authority | Must not do |
|---|---|---|---|
| Product Owner | Product priorities, acceptance, risk tolerance, business direction | Approves reset direction, roadmap priorities, and acceptance | Approve hidden allocation semantics or unsupported source-data shortcuts |
| PM | Roadmap sequencing, scope control, backlog hygiene, reset planning | Frames stages, defines scope, captures backlog impact | Authorize code/runtime changes alone |
| Scrum Master | Process discipline, status tracking, blockers, closeout | Keeps reset stages bounded and validates completion criteria | Expand scope beyond approved reset stage |
| Functional Analyst | User workflow, requirements, acceptance criteria | Defines operator workflows and functional expectations | Define formulas, implementation, or allocation semantics |
| Technical Analyst / Architect | Architecture, contracts, module boundaries, validation strategy | Defines v2 technical design before code | Decide product priorities or source-data exceptions |
| Data Steward | Provenance, raw evidence, source acceptability, local-data handling | Approves source-data readiness and input classification | Interpret financial meaning alone or allocate |
| Financial Analyst | Metric meaning, financial interpretation, review triggers | Defines approved financial semantics | Implement code or issue final actions |
| Developer / Codex | Implementation, tests, refactors, local validation | Implements only approved specs after contracts exist | Reuse old Python files as v2 base or invent architecture |
| Governance Auditor | Doctrine compliance, authority boundaries, backlog impact | Reviews whether changes preserve certified doctrine | Implement changes or allocate |
| Documentation Steward | Active/legacy documentation hygiene | Prevents duplicate active authority and prepares archive plans | Delete or move files without approved cutover |
| ChatGPT / documentation governance assistant | Documentation-only planning, analysis, and GitHub docs PRs | May create documentation-only branches/PRs when explicitly asked | Modify code, tests, data, workflows, generated artifacts, or runtime behavior |
| Codex / implementation agent | Local implementation after explicit approval | May modify code/tests/data contracts only under approved prompt | Perform governance decisions or uncontrolled cleanup |
| Decision Engine | Final allocation and action semantics | Sole runtime authority for final decisions | Mutate upstream evidence or hide filtering |
| Reporting | Communication of Decision Engine outputs | Formats, groups, summarizes, and delivers source decisions | Reinterpret, prioritize, suppress, or override decisions |

## Reset-Specific Role Rules

During RESET-1 and RESET-2, documentation and planning are allowed. Runtime implementation is not allowed.

During RESET-3 and later, Codex may begin implementation only after the relevant contracts exist. Any prompt to Codex must explicitly state that old Python files are reference-only and must not be used as the v2 base.

## Handoff Model

1. Product Owner confirms direction.
2. PM/Scrum Master frames reset stage and scope.
3. Functional Analyst defines workflows and acceptance needs.
4. Data Steward defines source-data readiness and provenance.
5. Financial Analyst defines metric meaning when financial data is involved.
6. Technical Analyst defines architecture and contracts.
7. Governance Auditor checks boundaries.
8. ChatGPT may document the approved scope.
9. Codex implements only after explicit implementation approval.
10. Decision Engine and Reporting retain their runtime authority boundaries.

## ChatGPT Boundary

ChatGPT may create or edit documentation-only GitHub branches and PRs when explicitly asked. ChatGPT must not change code, tests, CSV/data, reports, generated artifacts, workflows, runtime behavior, pipeline integration, Decision Engine logic, Reporting semantics, or Telegram behavior.

## Codex Boundary

Codex is the implementation agent. Codex may work on code, tests, local validation, and file moves only after explicit approval and a scoped prompt. Codex must not decide architecture direction or source-data exceptions by itself.
