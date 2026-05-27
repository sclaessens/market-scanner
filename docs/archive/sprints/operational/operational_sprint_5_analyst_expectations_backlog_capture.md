# Operational Sprint 5 Analyst Expectations Backlog Capture

## 1. Status and Scope

This document is a documentation-only backlog capture note.

It captures the new backlog item identified by `docs/research/analyst_expectations_and_backtesting_research_plan.md` for governed analyst expectations research and historical validation/backtesting.

This document does not implement code, tests, CSV changes, generated artifacts, provider integration, runtime orchestration, daily ingestion, Reporting changes, Telegram changes, scanner changes, Decision Engine changes, portfolio changes, watchlist changes, or fundamentals source changes.

No sprint is closed or certified complete by this document.

## 2. Backlog Item Capture

The following backlog item is captured for insertion into `docs/sprints/project_backlog.md`:

| ID | Title | Category | Source Document | Source Sprint | Description | Rationale | Governance Risk | Proposed Next Step | Status | Priority | Owner Role | Created Date | Related Documents | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BL-0018 | Define governed analyst expectations and historical validation research strategy | Observational Research / Operational Intelligence / Data Contract | `docs/research/analyst_expectations_and_backtesting_research_plan.md` | Operational Sprint 5 research planning | Define a governed research strategy for analyst consensus, analyst ratings, price targets, estimate data, point-in-time storage, historical validation, and future backtesting pipeline integration. | The operator identified analyst expectations and historical outcome validation as potentially valuable for improving analysis reliability, but this data must be source-supported, point-in-time controlled, and validated before any future Decision Engine consideration. | Analyst expectations could be misused as hidden buy/sell signals, ranking authority, conviction, tradeability, allocation pressure, or Decision Engine bypass if integrated without research governance and historical validation controls. | Create a documentation-only source-policy and validation-design sprint for analyst expectations before any data collection, provider/API integration, runtime ingestion, backtesting code, or Decision Engine integration. | CAPTURED | P1 | PM / Data Steward / Research / Technical Lead / Governance | 2026-05-21 | `docs/research/analyst_expectations_and_backtesting_research_plan.md`; `docs/sprints/project_backlog.md`; `docs/sprints/operational_sprint_5_data_steward_role.md`; `docs/sprints/operational_sprint_5_source_data_expansion_plan.md` | Backlog capture only. Does not authorize implementation, provider/API calls, daily ingestion, runtime changes, generated files, CSV edits, analyst consensus scoring, backtesting code, Decision Engine changes, Reporting changes, or Telegram changes. |

## 3. Backlog Update Requirement

This item should be inserted into the main backlog table in `docs/sprints/project_backlog.md` as `BL-0018`.

The backlog item remains `CAPTURED` only. It is not active sprint scope and does not authorize implementation.

## 4. Governance Constraints

This capture preserves project doctrine:

- classification upstream;
- allocation downstream;
- Decision Engine is the only allocation authority;
- upstream layers classify and enrich only;
- reporting communicates only;
- no hidden filtering;
- no ranking authority outside approved Decision Engine logic;
- no scoring authority outside approved Decision Engine logic.

Analyst expectations and backtesting remain research-only unless a future governed design explicitly approves a limited integration path.

## 5. Backlog Impact Assessment

Backlog impact assessment:
- New backlog items identified and captured for project_backlog.md
