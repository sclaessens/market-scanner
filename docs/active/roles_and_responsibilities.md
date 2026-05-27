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
| Product Owner / User | Priorities, risk tolerance, final acceptance, business direction. | Priorities, scope choices, acceptance of simplified doctrine, governance exceptions after risk explanation. | Hidden allocation semantics, unsupported source-data inference, silent code or data changes. | Priority decisions, acceptance/rejection decisions, product direction. | Hands approved direction to PM / Scrum Master and analysts. |
| PM / Scrum Master | Sprint structure, sequencing, scope control, backlog hygiene, process simplification. | Sprint framing, documentation-only scope, backlog capture, cleanup sequencing. | Runtime implementation, source-data edits, architecture boundary changes without review. | Sprint plans, backlog assessments, status summaries, cleanup sequencing. | Hands scoped work to Functional Analyst, Technical Analyst, Developer / Codex, or Governance Auditor. |
| Functional Analyst | User workflow, business requirements, acceptance criteria, operator visibility. | Acceptance criteria, workflow requirements, readiness definitions. | Formula authority, code implementation, allocation semantics, hidden upstream gating. | Functional contracts, workflow notes, acceptance criteria. | Hands requirements to Technical Analyst and Developer / Codex after approval. |
| Data Steward | Source evidence, provenance, raw data quality, source freshness, local ignored data handling. | Source acceptability, extraction evidence, completeness status. | Financial metric meaning alone, runtime schema changes, allocation decisions. | Source notes, data-quality findings, provenance records. | Hands source-supported raw evidence to Financial Analyst and Technical Analyst. |
| Financial Analyst | Metric meaning, financial statement interpretation, review triggers, financial analysis states. | Formula meaning, interpretation rules, financial review triggers. | Code implementation, raw data edits, Decision Engine outcomes, allocation decisions. | Metric definitions, interpretation rules, financial analysis contract. | Hands metric meaning to Functional Analyst and Technical Analyst. |
| Technical Analyst / Architect | Layer boundaries, data contracts, architecture, implementation specification structure. | Data contract proposals, layer boundaries, technical architecture direction, developer-ready specs. | Product priorities, source-data exceptions, hidden allocation semantics. | Architecture contracts, technical specs, migration plans, validation strategy. | Hands approved implementation scope to Developer / Codex. |
| Developer / Codex | Implementation, tests, refactoring, scripts, runtime changes after explicit approval. | Implementation feasibility, technical execution details within approved scope. | Architecture direction, governance exceptions, source-data exceptions, unapproved code or data changes. | Code changes, tests, validation evidence, implementation commits and PRs. | Returns implementation evidence to PM / Scrum Master, Technical Analyst, and Governance Auditor. |
| Governance Auditor | Doctrine compliance, separation of concerns, authority-boundary review, backlog impact review. | Compliance findings, risk classification, audit recommendations. | Implementation authority by itself, allocation logic, source-data inference. | Audit findings, risk notes, backlog impact assessment. | Reports risks to PM / Scrum Master, Technical Analyst, Product Owner, and Developer / Codex. |
| ChatGPT | Governance analysis, documentation consolidation, planning support, prompt creation. | Documentation-only drafts and branch/PR creation when explicitly asked. | Silent code changes, silent source-data changes, runtime implementation without explicit permission, replacing Codex for coding unless explicitly authorized. | Plans, contracts, analysis docs, prompts, documentation-only GitHub branches/PRs. | Hands implementation-ready prompts or docs to the user and Codex. |
| Decision Engine | Final allocation, execution, arbitration, and final action semantics. | Final decision semantics within implemented runtime logic. | Upstream classification mutation, hidden filtering, source-data inference. | `final_decisions.csv`, decision logs, final action fields. | Feeds Reporting as source authority. |

## ChatGPT boundary

ChatGPT may:

- perform governance analysis;
- write and consolidate documentation;
- produce prompts;
- create documentation-only GitHub branches and PRs when explicitly asked;
- help design architecture and contracts;
- help reduce scattered doctrine into active references.

ChatGPT may not:

- silently change code;
- silently change source data;
- execute runtime implementation without explicit permission;
- replace Codex for coding tasks unless explicitly authorized;
- run provider/API workflows without authorization;
- loosen Decision Engine authority.

ChatGPT is best for planning, documentation, governance, architecture drafting, contract consolidation, and prompt creation.

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
- change runtime behavior without explicit implementation scope.

Codex is best for implementation, refactoring, testing, validation, and repository-level code changes.

## Decision Engine boundary

The Decision Engine is not a human role. It is the only allocation authority.

No human role, upstream layer, report, validation artifact, source-data process, documentation artifact, ChatGPT response, or Codex change may create allocation, tradeability, urgency, conviction, eligibility, ranking, scoring, or hidden filtering semantics outside the Decision Engine.

## Handoff sequence for fundamentals redesign

Preferred handoff sequence:

1. Product Owner / User defines priority and accepts simplification direction.
2. PM / Scrum Master scopes the sprint and backlog impact.
3. Data Steward defines source evidence and raw-data readiness expectations.
4. Financial Analyst defines metric meaning and interpretation rules.
5. Functional Analyst defines workflow and acceptance criteria.
6. Technical Analyst / Architect defines contracts and implementation boundaries.
7. Governance Auditor checks authority boundaries.
8. Developer / Codex implements only after explicit approval.
9. Decision Engine consumes descriptive outputs only under approved runtime contracts.
10. Reporting communicates Decision Engine outputs only.

## Repository language rule

All repository documents created or edited under this role matrix must be English-only. Dutch may be used in direct chat with the user, but it must not be introduced into repository artifacts.