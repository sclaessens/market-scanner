# Source Inventory

Owner role: Scrum Master / Governance Auditor

Status: ME01 INVENTORY

## Purpose

This inventory lists source areas that must be inspected in ME02, ME03, and ME04. It does not perform deep extraction.

| Category | What to inspect | Why it matters | Likely owner role | Target sprint |
|---|---|---|---|---|
| Existing active documentation | `docs/active/` project, architecture, data, pipeline, portfolio, reporting, governance, and testing documents. | Captures the current institutional source of truth and operating boundaries. | PM / Product Owner, Governance Auditor | ME02, ME03, ME04 |
| Old backlog and sprint records | `docs/active/project/backlog.md`, `docs/archive/sprints/`, `docs/legacy/sprints/`, and reset planning records. | Preserves prior decisions, deferred work, and scope boundaries without continuing the old sprint line. | Scrum Master, PM / Product Owner | ME02 |
| Audits and provider smoke reports | `docs/audits/`, especially provider smoke, runtime boundary, reset cleanup, and legacy runtime audits. | Captures known risks, source-readiness findings, operational failures, and governance evidence. | Governance Auditor, Data Steward | ME03, ME04 |
| Scanner logic | Scanner modules, scanner tests, scanner docs, and related legacy records. | Identifies discovery and classification lessons while preventing allocation or recommendation leakage. | Functional Analyst, Technical Architect | ME03 |
| Fundamental logic | Fundamentals modules, fundamentals tests, provider adapter records, SEC-related records, and data contracts. | Preserves financial data lessons, missing-data handling, provenance, and source normalization risks. | Financial Analyst, Data Steward | ME03 |
| Source/provider readiness findings | Provider approval documents, smoke policies, source data strategy, provider integration designs, and provider audit records. | Defines how Market Engine should separate source intake from analysis and decisions. | Data Steward, Governance Auditor | ME03 |
| Portfolio/watchlist logic | Portfolio and watchlist modules, docs, data files, tests, and operational records. | Captures exposure and timing-state lessons while preventing lower-layer mutation or allocation decisions. | Functional Analyst, Technical Architect | ME02, ME04 |
| Reporting and Telegram boundaries | Reporting contracts, Telegram UX docs, reporting modules, and notification tests. | Preserves communication-only boundaries and side-effect controls. | Technical Architect, Governance Auditor | ME04 |
| Tests and test-family conventions | Unit, contract, integration, fixture, smoke, and governance test families. | Preserves where future tests should live and how fake provider responses should replace live calls. | QA / Test Lead | ME04 |
| Coding standards already learned | Python file creation policy, runtime boundary audits, active architecture, and cleanup findings. | Prevents unnecessary file proliferation, hidden runtime behavior, and import-time side effects. | Development Lead, Technical Architect | ME04 |
| Data files and CSV outputs | Data directories, CSV schemas, generated outputs, fixtures, and report inputs. | Captures source contracts, generated artifact risks, and missing-data semantics without mutating data. | Data Steward, QA / Test Lead | ME03 |
| Runtime entrypoints | CLI, scripts, orchestration modules, scheduled workflows, and operational command docs. | Identifies future entrypoint design needs and side-effect boundaries. | Technical Architect, Development Lead | ME04 |
| Archive/legacy material | `docs/archive/`, `docs/legacy/`, archived technical analysis, historical functional analysis, and old execution records. | Preserves lessons and evidence while preventing blind reuse of old implementation. | Governance Auditor, Functional Analyst | ME02, ME03, ME04 |

## ME01 Boundary

ME01 creates the inventory and method only. Deep extraction begins in ME02.

