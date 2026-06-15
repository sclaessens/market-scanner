# Market Engine Backlog

Owner role: Scrum Master / PM / Product Owner

Status: ACTIVE MARKET ENGINE BACKLOG

## Purpose

This backlog captures the Market Engine sprint line from ME01 through ME08.

Backlog items do not authorize implementation unless the sprint scope explicitly does so and repository governance allows it.

## Backlog Rules

* Preserve old repository assets as reference material.
* Do not blindly copy old script-era code.
* Do not use old quick scripts as canonical runtime.
* Do not continue legacy cleanup as the active implementation path.
* Do not delete, archive, rename, or ignore old files as part of Market Engine backlog work.
* Keep classification upstream and allocation downstream.
* Preserve Decision Engine authority as the only allocation authority.
* Keep source readiness separate from investment quality.
* Keep missing data explicit.
* Do not convert missing numeric values to zero.
* Do not introduce BUY / SELL / HOLD, recommendation, allocation, urgency, conviction, tradeability, or hidden ranking semantics outside an approved Decision Engine boundary.
* Do not introduce Telegram, reporting, portfolio, watchlist, provider, or runtime side effects unless a sprint explicitly authorizes them.

## Sprint Roadmap

### ME01 - Reset Market Engine documentation structure and knowledge extraction policy

Owner roles: PM / Product Owner, Scrum Master, Governance Auditor, Technical Architect

Status: COMPLETED BY ME01

Goal: Create the Market Engine documentation root, knowledge extraction policy, source inventory, baseline coding and testing standards, placeholders, audit record, and backlog.

Scope: Documentation and backlog only.

Not in scope: Python code changes, tests, runtime logic, provider calls, reports, Telegram, portfolio mutation, watchlist mutation, old file deletion, archive actions, file renames, quick scripts.

Acceptance criteria:

* `docs/market_engine/` exists with required subareas.
* Knowledge extraction policy exists.
* Reference extraction template exists.
* Source inventory exists for ME02 through ME04.
* Coding standards and testing strategy baselines exist.
* ME02, ME03, and ME04 placeholders exist.
* ME01 audit exists.
* Market Engine backlog exists.
* Only documentation/backlog files are changed.

### ME02 - Extract and write Market Engine functional flow

Owner roles: Functional Analyst, Product Owner, Scrum Master, Governance Auditor

Status: COMPLETED BY ME02

Goal: Extract the Market Engine functional flow from existing documentation, code, tests, audits, and backlog items.

Scope: Functional flow specification, role responsibilities, user/operator workflows, classification flow, state boundaries, and implementation/testing implications.

Not in scope: Coding, provider calls, test changes, runtime execution, report generation, Telegram, portfolio/watchlist mutation.

Acceptance criteria:

* Functional flow specification exists.
* Extraction decisions are recorded as keep / reject / defer.
* Implementation and testing implications are recorded.
* Open questions are narrow and do not block ME03.
* Source intake, analysis, operator review, and deferred downstream layers are clearly separated.
* Early-layer recommendation and side-effect exclusions are documented.

### ME03 - Extract and write Market Engine financial, scanner, and fundamental logic

Owner roles: Financial Analyst, Data Steward, Functional Analyst, Governance Auditor

Status: COMPLETED BY ME03

Goal: Extract financial, scanner, fundamental, and source-readiness logic for Market Engine specifications.

Scope: Financial logic, scanner classification lessons, fundamental data lessons, provider/source readiness, data implications, missing-data rules, quality-state rules, ticker failure handling, source-intake boundaries, analysis boundaries, and failure modes.

Not in scope: BUY / SELL / HOLD behavior, allocation, urgency, conviction, tradeability, hidden recommendation ranking, portfolio mutation, watchlist mutation, Telegram, reporting behavior, provider calls, runtime execution, Python implementation, or test implementation.

Acceptance criteria:

* Financial/scanner/fundamental specification exists.
* ME03 extraction document exists.
* ME03 audit document exists.
* Source intake is clearly separated from recommendation and allocation logic.
* Analysis is clearly separated from Decision Engine authority.
* Missing-data and source-failure implications are recorded.
* Missing numeric values remain missing and must not be converted to zero.
* Testing implications use fake or synthetic provider responses.
* Scanner context concepts are classified as keep / reject / defer.
* Fundamental metrics and source-readiness states are classified as keep / reject / defer.
* Data/source implications support future all-ticker source intake smoke.
* The specification uses ME02 functional stage boundaries.
* Only documentation/backlog files are changed.

### ME04-PREP - Archive old active documentation and make Market Engine the only active docs root

Owner roles: Scrum Master, Governance Auditor, Technical Architect

Status: COMPLETED BY ME04-PREP

Goal: Preserve the former active v2, BL, and reset documentation as historical reference material while making `docs/market_engine/` the only active Market Engine documentation root.

Scope: Documentation structure only.

Not in scope: Python code changes, test changes, provider calls, yfinance, SEC or EDGAR calls, scanner/runtime commands, report generation, Telegram, portfolio/watchlist mutation, production writes, Decision Engine behavior changes, or moving `docs/market_engine/`.

Outcome: The former `docs/active/` tree is preserved under `docs/archive/market_scanner_reference/active/`. Future Market Engine work should cite the archived path for old v2, BL, and reset documents and use them only through explicit extraction.

### ME04-PREP-B - Inventory remaining legacy documentation outside Market Engine

Owner roles: Scrum Master, Governance Auditor

Status: COMPLETED BY ME04-PREP-B

Goal: Inventory remaining documentation and reference material outside `docs/market_engine/` and outside the Market Scanner reference archive.

Scope: Documentation inventory only.

Outcome: Remaining legacy documentation candidates were inventoried before consolidation.

### ME04-PREP-C - Consolidate remaining legacy documentation under Market Scanner reference archive

Owner roles: Scrum Master, Governance Auditor

Status: COMPLETED BY ME04-PREP-C

Goal: Move clear legacy documentation candidates under `docs/archive/market_scanner_reference/` while keeping `docs/market_engine/` as the only active Market Engine documentation root.

Scope: Documentation structure only.

Not in scope: Code, tests, provider calls, runtime behavior, production writes, reports, Telegram, portfolio/watchlist mutation, or Decision Engine behavior changes.

Outcome: Legacy documentation/reference areas were preserved under `docs/archive/market_scanner_reference/`. `docs/templates/` remains in place pending manual decision.

### ME04-PREP-D - Inventory legacy runtime, tests, and data before Market Engine cutover

Owner roles: Technical Architect, Development Lead, Data Steward, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME04-PREP-D

Goal: Inventory legacy runtime, scripts, tests, data, reports, and root-level files before Market Engine cutover.

Scope: Documentation-only inventory.

Not in scope: Moving, deleting, or renaming files; Python code changes; test changes; data, CSV, or report changes; provider calls; runtime commands; production writes; reports; Telegram; portfolio/watchlist mutation; Decision Engine behavior changes.

Acceptance criteria:

* Inventory exists.
* Audit exists.
* Old runtime, scripts, tests, data, reports, and root-level files are classified.
* No files are moved, deleted, or renamed.
* No Python, test, data, CSV, or report files are changed.
* No provider or runtime commands are run.
* ME04 extraction needs are recorded.

### ME04 - Extract and write Market Engine technical, coding, and testing architecture

Owner roles: Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME04

Goal: Extract technical architecture, coding rules, and testing architecture for Market Engine.

Scope: Module ownership, provider/data/analysis/decision separation, runtime boundaries, side-effect controls, test-family conventions, manual smoke harness standards, forbidden field policy, and file strategy.

Not in scope: Implementation unless separately authorized, provider calls, production writes, report generation, Telegram, portfolio/watchlist mutation, Decision Engine behavior changes, recommendation behavior, or runtime execution.

Acceptance criteria:

* Technical architecture specification exists.
* ME04 extraction document exists.
* ME04 audit document exists.
* ME04 architecture is explicitly based on ME01, ME02, ME03, and ME04-PREP-D.
* Coding and testing standards are tied to implementation implications.
* Module ownership maps to the ME02 functional stages.
* Financial/scanner/fundamental boundaries from ME03 are translated into technical ownership.
* Provider/source access is isolated from scanner, fundamental, analysis, reporting, Telegram, portfolio, watchlist, and Decision Engine behavior.
* Provider/data/analysis/decision/reporting/delivery boundaries are explicit.
* Portfolio/watchlist mutation and Telegram/reporting side effects are excluded from lower layers.
* Forbidden authority fields are documented for source, scanner, fundamental, and analysis layers.
* Automated test boundaries prohibit live provider calls.
* Test-family placement rules are clear enough to avoid unnecessary new test files.
* Manual smoke harness policy is explicit and bounded before ME05 begins.
* File/module strategy prevents unnecessary new Python files and avoids temporary quick scripts as canonical runtime.
* ME05 readiness gate is defined.
* Only documentation/backlog files are changed.

### ME05 - Build all-ticker source intake smoke

Owner roles: Development Lead, Data Steward, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME05

Goal: Build an explicit all-ticker source intake smoke harness after ME02 through ME04 specifications authorize the boundary.

Scope: Bounded manual source intake smoke harness, source availability capture, per-ticker failure capture, raw evidence feasibility, normalized data feasibility, missingness preservation, and source-readiness states.

Not in scope: Production pipeline integration, normal automated tests with live calls, BUY / SELL / HOLD logic, allocation, urgency, conviction, tradeability, portfolio/watchlist mutation, reporting, Telegram, or Decision Engine behavior.

Acceptance criteria:

* Smoke harness is explicit and non-canonical unless promoted by architecture.
* Provider access is bounded and reviewable.
* Source intake stops at coverage, raw evidence, normalized view, and readiness state.
* Source coverage is captured per ticker.
* Failures are captured without stopping the whole batch.
* Missing data remains missing.
* Missing numeric values are not converted to zero.
* Provider errors and unsupported tickers are distinguishable where possible.
* No recommendation or allocation behavior is introduced.
* No production reports, Telegram delivery, portfolio/watchlist mutation, or Decision Engine behavior occurs.
* Smoke evidence can feed ME06 triage without becoming source truth by default.

Outcome: ME05 added a clean `src/market_engine/source_intake/` package, fake provider scenarios, source readiness statuses, per-ticker intake results, batch summary, missing-field frequency tracking, targeted tests, a fake-provider manual smoke entrypoint, and audit/documentation updates.

### ME06 - Add bounded real provider source intake smoke and coverage review

Owner roles: Data Steward, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME06

Goal: Add a bounded real-provider source intake smoke and review coverage evidence without entering analysis or recommendation behavior.

Scope: First real provider selection, explicit manual invocation, ticker limit, source coverage evidence, failure triage, source-readiness implications, missing-data observations, provider/source limitations, data-owner review, generated-output/archive decision inputs, and backlog follow-up.

Not in scope: Analysis, recommendation behavior, BUY / SELL / HOLD behavior, allocation, ranking, urgency, conviction, tradeability, position sizing, execution advice, production writes, normal automated tests with live calls, report generation, Telegram, portfolio/watchlist mutation, or Decision Engine behavior changes.

Acceptance criteria:

* First real provider is selected and documented.
* Live execution is explicit, manual, bounded, and uses a ticker limit.
* Coverage results are documented as evidence.
* Ticker failures are triaged.
* Provider/source limitations are recorded.
* Missing-field patterns are recorded.
* Partial, stale, invalid, unsupported, and provider-error states are distinguished where possible.
* Data-owner review is recorded.
* Generated output and archive implications are recorded without treating smoke evidence as source truth by default.
* Follow-up work is added without changing runtime decision behavior.
* Smoke evidence remains evidence for triage and does not become source truth by default.

Outcome: ME06 added a SEC CompanyFacts provider adapter, mocked provider tests, explicit real-provider manual smoke flags, ticker limit enforcement, and local source coverage review. The bounded real-provider smoke executed safely but returned provider errors for the sampled tickers in this environment.

### ME07 - Review real-provider coverage and define source-data owner decisions

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME07

Goal: Review ME06 real-provider coverage behavior and define source-data owner decisions before building first fundamental source context.

Scope: Provider availability review, SEC access/user-agent/network follow-up, ticker-to-CIK ownership decision, smoke evidence retention policy, required-field alias review, source artifact handling, and readiness criteria for first fundamental source context.

Not in scope: BUY / SELL / HOLD behavior, allocation, urgency, conviction, tradeability, hidden ranking, score behavior, portfolio mutation, watchlist mutation, Telegram, production reporting, Decision Engine behavior changes, broad provider execution, or first analysis implementation unless separately authorized.

Acceptance criteria:

* ME06 bounded SEC smoke result is reviewed.
* Provider-error cause is triaged without forcing unbounded provider calls.
* Ticker-to-CIK ownership is decided for future source intake.
* Smoke evidence retention policy is documented.
* Required SEC field aliases are reviewed.
* Generated smoke artifact handling is documented.
* Readiness criteria for first fundamental source context are recorded.
* No analysis, recommendation, score, allocation, urgency, conviction, tradeability, or Decision Engine behavior is introduced.
* No reporting, Telegram, portfolio, watchlist, production data, or production report side effects occur.

Outcome: ME07 identified the bounded SEC smoke failure as a controlled network/DNS access failure in this environment, improved provider error categories, kept SEC CompanyFacts approved for bounded smoke only, and deferred all-ticker coverage or analysis until access and source-data ownership decisions are resolved.

### ME08 - Repair SEC CompanyFacts network access and rerun bounded coverage review

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME08

Goal: Resolve the SEC CompanyFacts network/request access issue and rerun bounded source coverage before approving source context work.

Scope: SEC access diagnostics, User-Agent/contact policy review, environment/network review, bounded manual SEC smoke rerun, ticker-to-CIK ownership decision, source evidence retention decision, and coverage review documentation.

Not in scope: Analysis, recommendation behavior, BUY / SELL / HOLD behavior, allocation, ranking, score, urgency, conviction, tradeability, position sizing, execution advice, production writes, normal automated tests with live calls, report generation, Telegram, portfolio/watchlist mutation, or Decision Engine behavior changes.

Acceptance criteria:

* SEC network/request failure cause is resolved or documented as an environment blocker.
* Bounded SEC smoke is rerun with explicit ticker limit.
* Readiness counts and provider error categories are documented.
* Ticker-to-CIK source ownership is decided.
* Smoke evidence retention policy is decided.
* Required SEC alias coverage is reviewed for first source context readiness.
* No source evidence is treated as source truth by default.
* No analysis, recommendation, score, allocation, urgency, conviction, tradeability, or Decision Engine behavior is introduced.
* No reporting, Telegram, portfolio, watchlist, production data, or production report side effects occur.

Outcome: ME08 confirmed that sandboxed network access still fails for `data.sec.gov`, but escalated local runtime DNS and HTTPS access succeeds. The bounded SEC CompanyFacts smoke reached `AVAILABLE=4` for `NVDA`, `AMD`, `META`, and `COST` with no missing fields or provider errors. SEC CompanyFacts is approved for bounded coverage review only.

### ME09 - Run bounded multi-ticker SEC CompanyFacts coverage artifact review

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME09

Goal: Run a bounded multi-ticker SEC CompanyFacts coverage review and evaluate isolated smoke artifacts before any source context or analysis sprint.

Scope: Explicit bounded ticker set, SEC CompanyFacts coverage review, isolated non-production smoke artifacts if explicitly requested, missing-field evidence, provider-error evidence, ticker-to-CIK source ownership review, artifact retention review, and readiness criteria for first source context.

Not in scope: Analysis, recommendation behavior, BUY / SELL / HOLD behavior, allocation, ranking, score, urgency, conviction, tradeability, position sizing, execution advice, production writes, normal automated tests with live calls, report generation, Telegram, portfolio/watchlist mutation, or Decision Engine behavior changes.

Acceptance criteria:

* Multi-ticker SEC coverage run is explicitly bounded.
* Any artifact writing is explicitly requested and isolated under `data/market_engine/smokes/source_intake/sec_companyfacts/<run_id>/`.
* No old data/report paths are written.
* Readiness counts are documented.
* Missing-field frequency is documented.
* Provider-error categories are documented.
* Ticker-to-CIK ownership decision is refined.
* Smoke artifacts are reviewed as evidence only, not source truth.
* No analysis, recommendation, score, allocation, urgency, conviction, tradeability, or Decision Engine behavior is introduced.
* No reporting, Telegram, portfolio, watchlist, production data, or production report side effects occur.

Outcome: ME09 ran a bounded 10-ticker SEC CompanyFacts coverage review for `NVDA`, `AMD`, `META`, `COST`, `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`, and `AVGO`. Escalated SEC coverage returned `AVAILABLE=10`, no missing fields, and no provider errors. Non-production smoke artifacts were written under `data/market_engine/smokes/source_intake/sec_companyfacts/20260615T103333Z/` and intentionally not committed.

### ME10 - Define approved SEC CompanyFacts field mapping and source coverage contract

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME10

Goal: Convert bounded SEC CompanyFacts smoke evidence into an approved source-field mapping and source coverage contract before source context or analysis work.

Scope: SEC field alias review, required-field contract, ticker-to-CIK ownership decision, source coverage contract, artifact retention policy, missing-field semantics, provider-error semantics, and readiness criteria for first fundamental source context.

Not in scope: Analysis, recommendation behavior, BUY / SELL / HOLD behavior, allocation, ranking, score, urgency, conviction, tradeability, position sizing, execution advice, production writes, normal automated tests with live calls, report generation, Telegram, portfolio/watchlist mutation, or Decision Engine behavior changes.

Acceptance criteria:

* SEC required-field mapping is documented.
* SEC alias selection rules are documented.
* Ticker-to-CIK ownership is decided.
* Source coverage contract is defined.
* Artifact retention policy is decided.
* Missing-field semantics are documented.
* Provider-error semantics are documented.
* Readiness criteria for first source context are recorded.
* No analysis, recommendation, score, allocation, urgency, conviction, tradeability, or Decision Engine behavior is introduced.
* No reporting, Telegram, portfolio, watchlist, production data, or production report side effects occur.

Outcome: ME10 approved the first SEC CompanyFacts field mapping and source coverage contract for `revenue`, `net_income`, `operating_cash_flow`, and `capital_expenditures`. SEC CompanyFacts is approved for field mapping implementation, but not for analysis.

### ME11 - Implement SEC field mapping and first fundamental source context

Owner roles: Financial Analyst, Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME11

Goal: Implement the approved SEC CompanyFacts field mapping contract and create the first source-only Market Engine fundamental context.

Scope: Deterministic SEC alias priority, canonical source field mapping, raw source value preservation, SEC fact provenance, missing-data preservation, source readiness, and source-only context objects.

Not in scope: Financial analysis, free cash flow, growth, margins, valuation metrics, recommendation behavior, BUY / SELL / HOLD behavior, allocation, ranking, score, urgency, conviction, tradeability, position sizing, execution advice, production writes, live provider calls in automated tests, report generation, Telegram, portfolio/watchlist mutation, or Decision Engine behavior changes.

Acceptance criteria:

* Approved SEC CompanyFacts mappings are implemented for `revenue`, `net_income`, `operating_cash_flow`, and `capital_expenditures`.
* Alias priority is deterministic and selects one approved tag only.
* Unapproved substitutions are not selected.
* Selected SEC tag, unit, period metadata, filing metadata, raw value, selection reason, and fallback alias metadata are preserved.
* Missing values remain missing and are not converted to zero.
* Source context exposes readiness states without analysis output.
* Automated tests use mocked/synthetic SEC payloads only.
* No legacy runtime imports are introduced.
* No old data/report paths are changed.
* No Decision Engine, reporting, Telegram, portfolio, or watchlist side effects occur.

Outcome: ME11 added SEC CompanyFacts contract mapping, a source-only fundamental context, tests for approved mappings and forbidden substitutions, provenance checks, missing-data checks, and documentation/audit updates.

### ME12 - Build first non-decision fundamental analysis pass

Owner roles: Financial Analyst, Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME12

Goal: Build the first non-decision fundamental analysis pass from approved Market Engine source context.

Scope: Source-backed financial observations, explicit missing-data handling, source limitation flags, and deterministic non-decision context suitable for later operator review.

Not in scope: BUY / SELL / HOLD behavior, allocation, ranking, score, urgency, conviction, tradeability, position sizing, execution advice, portfolio/watchlist mutation, Telegram, production reporting, or Decision Engine behavior changes.

Acceptance criteria:

* Analysis consumes only approved Market Engine source context.
* Missing data remains explicit.
* Output is non-decision and non-allocation.
* No recommendation, score, rank, tradeability, or action language is emitted.
* Tests prove no Decision Engine, reporting, Telegram, portfolio, or watchlist side effects.
* Old runtime code remains reference only.

Outcome: ME12 added the first non-decision fundamental analysis pass. It consumes the ME11 source context and emits source-grounded observations for source readiness, revenue presence, net income sign, operating cash flow sign, capital expenditures presence, and cash-generation source completeness. It does not calculate free cash flow, growth, margins, ratios, valuation metrics, scores, rankings, recommendations, or Decision Engine behavior.

### ME13 - Add first derived cash-generation observation layer

Owner roles: Financial Analyst, Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: RECOMMENDED NEXT

Goal: Add the first derived but still non-decision cash-generation observation layer.

Scope: Explicitly approved derived cash-generation observation, likely free cash flow presence/sign based on operating cash flow and capital expenditures, source reference preservation, missing-data handling, and non-decision boundary tests.

Not in scope: BUY / SELL / HOLD behavior, allocation, ranking, score, recommendation behavior, urgency, conviction, tradeability, position sizing, execution advice, valuation metrics, portfolio/watchlist mutation, Telegram, production reporting, or Decision Engine behavior changes.

Acceptance criteria:

* Any derived calculation is explicitly documented and source-grounded.
* Missing operating cash flow or capital expenditures blocks the derived observation.
* Output remains observational and non-decision.
* No score, ranking, recommendation, tradeability, allocation, or operator action language is emitted.
* Tests prove no Decision Engine, reporting, Telegram, portfolio, or watchlist side effects.
* Old runtime code remains reference only.

### Future - Produce local operator review output

Owner roles: Operator / User, Product Owner, Development Lead, QA / Test Lead, Governance Auditor

Status: DEFERRED AFTER ME07

Goal: Produce local operator review output for human review without creating allocation authority outside the Decision Engine.

Scope: Local review output, communication formatting, operator usability, side-effect controls, auditability, source-readiness visibility, missing-data visibility, and limitation flags.

Not in scope: Telegram delivery unless separately authorized, production report generation, portfolio/watchlist mutation, recommendation behavior, allocation behavior, or final-action behavior outside the Decision Engine.

Acceptance criteria:

* Local operator review output is produced from approved data and analysis boundaries.
* Output is communication only.
* Output clearly shows source coverage, missingness, failures, limitations, and review-required states.
* Output does not create BUY / SELL / HOLD, allocation, urgency, conviction, tradeability, or hidden ranking semantics.
* No lower layer mutates portfolio or watchlist state.
* No hidden provider calls or production side effects occur.
* No Telegram delivery or production reporting occurs unless a later sprint explicitly authorizes it.
* Operator limitations and next steps are documented.

Deferral note: This was the original ME08 roadmap target. After ME07, the next recommended sprint is the revised ME08 source-access repair and bounded coverage review. Local operator review output remains important but depends on usable source intake evidence.
