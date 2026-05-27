# Roles and Responsibilities

Status: ACTIVE

## Purpose

This document is the concise active role matrix for the operational market-scanner project.

It supersedes older role documents where applicable, including legacy mixed-language role doctrine, unless an active document explicitly preserves a specific historical reference.

All roles inherit the certified doctrine:

- classification upstream;
- allocation downstream;
- Decision Engine = only allocation authority;
- Reporting communicates only;
- no hidden filtering;
- no upstream tradeability;
- no ranking or scoring authority outside the Decision Engine;
- English-only repository content.

## Role matrix

| Role | Owns | May approve | May not approve | Outputs | Handoff |
|---|---|---|---|---|---|
| Product Owner / User | Priorities, risk tolerance, final acceptance, business direction, and sprint capacity exceptions. | Priorities, scope choices, acceptance of simplified doctrine, governance exceptions after risk explanation, sprint priority direction. | Hidden allocation semantics, unsupported source-data inference, silent code or data changes. | Priority decisions, acceptance/rejection decisions, product direction. | Hands approved direction to PM / Scrum Master and analysts. |
| PM / Scrum Master | Sprint structure, sequencing, scope control, backlog hygiene, sprint capacity, backlog triage cadence, and closeout discipline. | Sprint framing, documentation-only scope, backlog capture, cleanup sequencing, effort-point proposal, sprint capacity fit. | Runtime implementation, source-data edits, architecture boundary changes without review, over-capacity sprint packing. | Sprint plans, backlog assessments, sprint capacity notes, status summaries, closeout checks, cleanup sequencing. | Hands scoped work to Functional Analyst, Technical Analyst, Developer / Codex, or Governance Auditor. |
| Strategy and Logic Steward | Trading logic map, calculation registry hygiene, ticker/category logic review, logic simplification, and periodic logic fitness checks. | Descriptive logic-review findings, calculation placement recommendations, category-model proposals for analysis. | Implementation changes, source-data exceptions, allocation authority, Decision Engine final-action rules. | Logic review notes, calculation registry updates, ticker-category proposals, complexity-reduction recommendations. | Hands logic recommendations to Financial Analyst, Functional Analyst, Technical Analyst, and Governance Auditor. |
| Functional Analyst | User workflow, business requirements, acceptance criteria, operator visibility, and workflow impact of logic changes. | Acceptance criteria, workflow requirements, readiness definitions. | Formula authority, code implementation, allocation semantics, hidden upstream gating. | Functional contracts, workflow notes, acceptance criteria. | Hands requirements to Technical Analyst and Developer / Codex after approval. |
| Data Steward | Source evidence, provenance, raw data quality, source freshness, local ignored data handling, and source-data readiness for calculations. | Source acceptability, extraction evidence, completeness status. | Financial metric meaning alone, runtime schema changes, allocation decisions. | Source notes, data-quality findings, provenance records. | Hands source-supported raw evidence to Financial Analyst and Technical Analyst. |
| Financial Analyst | Metric meaning, financial statement interpretation, review triggers, financial analysis states, and financial calculation semantics. | Formula meaning, interpretation rules, financial review triggers. | Code implementation, raw data edits, Decision Engine outcomes, allocation decisions. | Metric definitions, interpretation rules, financial analysis contract, calculation-registry input. | Hands metric meaning to Functional Analyst, Strategy and Logic Steward, and Technical Analyst. |
| Technical Analyst / Architect | Layer boundaries, data contracts, architecture, implementation specification structure, module placement, and future-proof code organization. | Data contract proposals, layer boundaries, technical architecture direction, developer-ready specs. | Product priorities, source-data exceptions, hidden allocation semantics. | Architecture contracts, technical specs, migration plans, validation strategy, code-placement recommendations. | Hands approved implementation scope to Developer / Codex. |
| Developer / Codex | Implementation, tests, refactoring, scripts, runtime changes, and file movement after explicit approval. | Implementation feasibility, technical execution details within approved scope. | Architecture direction, governance exceptions, source-data exceptions, unapproved code or data changes, unapproved deletion of useful logic. | Code changes, tests, validation evidence, implementation commits and PRs. | Returns implementation evidence to PM / Scrum Master, Technical Analyst, and Governance Auditor. |
| Governance Auditor | Doctrine compliance, separation of concerns, authority-boundary review, backlog impact review, and closeout compliance. | Compliance findings, risk classification, audit recommendations. | Implementation authority by itself, allocation logic, source-data inference. | Audit findings, risk notes, backlog impact assessment, doctrine-compliance review. | Reports risks to PM / Scrum Master, Technical Analyst, Product Owner, and Developer / Codex. |
| Documentation Steward | Active/reference/archive hygiene, document replacement discipline, link/reference hygiene, and prevention of duplicate active doctrine. | Documentation restructuring proposals, archive recommendations, replacement-document readiness. | Runtime implementation, architecture changes by itself, source-data or code changes. | Documentation map updates, archive recommendations, replacement plans, reference hygiene notes. | Hands documentation changes to ChatGPT or Developer / Codex depending on whether work is docs-only or repo-wide. |
| ChatGPT | Governance analysis, documentation consolidation, planning support, prompt creation, backlog/logic/sprint operating-model drafting. | Documentation-only drafts and branch/PR creation when explicitly asked. | Silent code changes, silent source-data changes, runtime implementation without explicit permission, replacing Codex for coding unless explicitly authorized. | Plans, contracts, analysis docs, prompts, documentation-only GitHub branches/PRs. | Hands implementation-ready prompts or docs to the user and Codex. |
| Decision Engine | Final allocation, execution, arbitration, and final action semantics. | Final decision semantics within implemented runtime logic. | Upstream classification mutation, hidden filtering, source-data inference. | `final_decisions.csv`, decision logs, final action fields. | Feeds Reporting as source authority. |

## ChatGPT boundary

ChatGPT may:

- perform governance analysis;
- write and consolidate documentation;
- produce prompts;
- create documentation-only GitHub branches and PRs when explicitly asked;
- help design architecture and contracts;
- help triage backlog items and sprint capacity;
- help maintain calculation and logic governance documentation;
- help reduce scattered doctrine into active references.

ChatGPT may not:

- silently change code;
- silently change source data;
- execute runtime implementation without explicit permission;
- replace Codex for coding tasks unless explicitly authorized;
- run provider/API workflows without authorization;
- loosen Decision Engine authority.

ChatGPT is best for planning, documentation, governance, architecture drafting, contract consolidation, backlog triage, calculation registry drafting, and prompt creation.

## Codex boundary

Codex may:

- implement approved specifications;
- modify code, tests, and scripts when explicitly authorized;
- move or archive files when explicitly authorized;
- run tests and validations;
- create implementation PRs;
- refactor within approved architecture boundaries.

Codex may not:

- invent architecture direction;
- bypass approved contracts;
- make source-data or governance exceptions without approval;
- introduce allocation, tradeability, urgency, conviction, eligibility, ranking, scoring, or hidden filtering outside the Decision Engine;
- delete or move useful logic without an approved migration or cleanup specification;
- change runtime behavior without explicit implementation scope.

Codex is best for implementation, refactoring, testing, validation, and repository-level code changes.

## Documentation replacement boundary

When active documentation no longer fits the needed structure, the project should replace it deliberately instead of layering more patches into the wrong document.

Replacement requires:

1. identify the active document that no longer fits;
2. design the replacement structure;
3. migrate useful content;
4. archive or supersede the old document;
5. update references;
6. avoid duplicate active authority.

## Decision Engine boundary

The Decision Engine is not a human role. It is the only allocation authority.

No human role, upstream layer, report, validation artifact, source-data process, documentation artifact, ChatGPT response, or Codex change may create allocation, tradeability, urgency, conviction, eligibility, ranking, scoring, or hidden filtering semantics outside the Decision Engine.

## Handoff sequence for logic, backlog, and implementation work

Preferred handoff sequence:

1. Product Owner / User defines priority and accepts direction.
2. PM / Scrum Master reviews backlog, effort, capacity, and sprint fit.
3. Strategy and Logic Steward reviews logic, calculations, category implications, and simplification opportunities.
4. Data Steward defines source evidence and raw-data readiness expectations when data is involved.
5. Financial Analyst defines metric meaning and interpretation rules when financial calculations are involved.
6. Functional Analyst defines workflow and acceptance criteria.
7. Technical Analyst / Architect defines contracts, module placement, and implementation boundaries.
8. Governance Auditor checks authority boundaries and backlog impact.
9. Developer / Codex implements only after explicit approval.
10. Decision Engine consumes descriptive outputs only under approved runtime contracts.
11. Reporting communicates Decision Engine outputs only.
12. Sprint closeout updates backlog status, calculation registry, documentation state, and code-placement findings.

## Repository language rule

All repository documents created or edited under this role matrix must be English-only. Dutch may be used in direct chat with the user, but it must not be introduced into repository artifacts.