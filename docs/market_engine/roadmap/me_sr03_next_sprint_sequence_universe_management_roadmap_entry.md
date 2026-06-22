# ME-SR03 Follow-up Roadmap Entry - Universe Management Sprint Sequence

Owner roles: Product Owner / Operator / Technical Architect / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR / ME-UNI / ME-RUN / ME-OUT planning

Status: PLANNED SEQUENCE AFTER ME-SR03

## Roadmap Decision

ME-SR03 proved that the current cached-source pipeline can complete all discovered supported canonical-universe snapshots after resolving ASML and TSM foreign-issuer field mapping.

The only remaining small canonical-universe blocker is `HO`.

The operator clarified that the future production-intent scan universe is an editable professional swing universe of roughly 250 to 300 tickers, not a fixed 12-to-13 ticker test universe.

Therefore, after ME-SR04 resolves the `HO` source-identity/exclusion decision, the roadmap must move into editable universe management before broader source support classification, large supported-universe runs, readable output, candidate classification, or entry analysis.

## Ordered Next Sprint Sequence

| Order | Sprint | Job family | Purpose |
| --- | --- | --- | --- |
| 1 | ME-SR04 | Source Refresh / Source Coverage | Resolve `HO` source identity or exclusion so the current small canonical SEC CompanyFacts run no longer carries an unresolved unsupported ticker blocker. |
| 2 | ME-UNI04 | Ticker Universe | Define the editable Professional Swing Universe contract. |
| 3 | ME-UNI05 | Ticker Universe | Import and normalize the operator-provided professional swing universe seed list. |
| 4 | ME-UNI06 | Ticker Universe | Implement editable universe loading, validation, active-only selection, duplicate handling, and fail-closed config behavior. |
| 5 | ME-SR05 | Source Refresh / Source Coverage | Classify source support for all active professional swing universe tickers before large scans. |
| 6 | ME-RUN20 | Run / orchestration | Execute a clean local cached-source scan against the supported active universe subset. |
| 7 | ME-OUT01 | Operator Output | Define readable, non-actionable operator reporting from dry-run artifacts. |
| 8 | ME-CANDIDATE01 | Candidate Classification | Define controlled, non-actionable candidate classification before any buy/entry semantics. |

## Why This Sequence Comes Before Broader Scans

The Market Engine can no longer treat ticker membership as a fixed Python/runtime assumption. The operator must be able to add, remove, disable, or reclassify tickers in a managed universe file.

Editable universe governance must exist before large-universe source coverage and before any report or candidate layer. Otherwise every future scan would be tied to brittle hardcoded or one-off ticker lists.

## Boundary

This roadmap entry records planning only.

It does not authorize provider calls, live SEC/EDGAR calls, yfinance calls, live market data calls, broker calls, production portfolio/watchlist writes, Telegram/email delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation advice, target prices, ranking, scoring, conviction, urgency, tradeability, or execution advice.
