# Market Engine

Owner role: PM / Product Owner

Status: ACTIVE STRATEGIC DIRECTION

## Purpose

Market Engine is the new strategic product direction for this repository.

`ME` means `Market Engine`. Market Engine sprint identifiers begin at `ME01`.

Market Engine is not a blind continuation of the old script-era implementation. The existing codebase, documents, audits, tests, backlog items, provider findings, and runtime behavior remain important reference sources, but they are not the implementation foundation for the new product direction.

Old repository assets must be inspected for knowledge, lessons, risks, governance decisions, source-readiness findings, test rules, and implementation implications. Market Engine must later be built from clean specifications.

## Strategic Rules

- Preserve old evidence and lessons.
- Do not delete, archive, rename, or ignore old files as part of Market Engine documentation reset work.
- Do not blindly copy old code or script-era runtime patterns.
- Do not continue legacy cleanup as the active implementation path.
- Treat old material as reference input for Market Engine specifications.
- Keep classification upstream and allocation downstream.
- Preserve Decision Engine authority as the only allocation authority.

## Anti-Iteration Rule

Every future Market Engine documentation sprint must follow this loop:

1. Inspect existing documentation, code, tests, audits, and backlog items.
2. Extract useful logic and lessons.
3. Decide what Market Engine keeps, rejects, or defers.
4. Write implementation implications.
5. Move to the next sprint.

A document is complete when it is good enough to steer implementation and tests. It does not need to be theoretically perfect.

## Roadmap

| Sprint | Goal |
|---|---|
| ME01 | Reset Market Engine documentation structure and knowledge extraction policy. |
| ME02 | Extract and write Market Engine functional flow. |
| ME03 | Extract and write Market Engine financial, scanner, and fundamental logic. |
| ME04 | Extract and write Market Engine technical, coding, and testing architecture. |
| ME05 | Build all-ticker source intake smoke. |
| ME06 | Run all-ticker source coverage and triage failures. |
| ME07 | Build first analysis pass on collected data. |
| ME08 | Produce local operator review output. |

## Role Ownership Principle

Each Market Engine document must name its owner role near the top and must be written from that role's responsibility boundary.

Core roles to preserve:

- PM / Product Owner
- Scrum Master
- Functional Analyst
- Financial Analyst
- Technical Architect
- Data Steward
- Development Lead
- QA / Test Lead
- Governance Auditor
- Operator / User

Role ownership does not authorize runtime changes. Implementation authority remains bound to later approved sprints and repository governance.

