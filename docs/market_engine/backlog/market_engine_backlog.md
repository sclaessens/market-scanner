# Market Engine Backlog

Owner role: Scrum Master / PM / Product Owner

Status: ACTIVE MARKET ENGINE BACKLOG

## Purpose

This backlog captures the Market Engine sprint line from ME01 through ME08.

Backlog items do not authorize implementation unless the sprint scope explicitly does so and repository governance allows it.

## Backlog Rules

- Preserve old repository assets as reference material.
- Do not blindly copy old script-era code.
- Do not use old quick scripts as canonical runtime.
- Do not continue legacy cleanup as the active implementation path.
- Do not delete, archive, rename, or ignore old files as part of Market Engine backlog work.
- Keep classification upstream and allocation downstream.
- Preserve Decision Engine authority as the only allocation authority.

## Sprint Roadmap

### ME01 - Reset Market Engine documentation structure and knowledge extraction policy

Owner roles: PM / Product Owner, Scrum Master, Governance Auditor, Technical Architect

Goal: Create the Market Engine documentation root, knowledge extraction policy, source inventory, baseline coding and testing standards, placeholders, audit record, and backlog.

Scope: Documentation and backlog only.

Not in scope: Python code changes, tests, runtime logic, provider calls, reports, Telegram, portfolio mutation, watchlist mutation, old file deletion, archive actions, file renames, quick scripts.

Acceptance criteria:

- `docs/market_engine/` exists with required subareas.
- Knowledge extraction policy exists.
- Reference extraction template exists.
- Source inventory exists for ME02 through ME04.
- Coding standards and testing strategy baselines exist.
- ME02, ME03, and ME04 placeholders exist.
- ME01 audit exists.
- Market Engine backlog exists.
- Only documentation/backlog files are changed.

### ME02 - Extract and write Market Engine functional flow

Owner roles: Functional Analyst, Product Owner, Scrum Master, Governance Auditor

Goal: Extract the Market Engine functional flow from existing documentation, code, tests, audits, and backlog items.

Scope: Functional flow specification, role responsibilities, user/operator workflows, classification flow, state boundaries, and implementation/testing implications.

Not in scope: Coding, provider calls, test changes, runtime execution, report generation, Telegram, portfolio/watchlist mutation.

Acceptance criteria:

- Functional flow specification exists.
- Extraction decisions are recorded as keep / reject / defer.
- Implementation and testing implications are recorded.
- Open questions are narrow and do not block ME03.

### ME03 - Extract and write Market Engine financial, scanner, and fundamental logic

Owner roles: Financial Analyst, Data Steward, Functional Analyst, Governance Auditor

Goal: Extract financial, scanner, fundamental, and source-readiness logic for Market Engine specifications.

Scope: Financial logic, scanner classification lessons, fundamental data lessons, provider/source readiness, data implications, missing-data rules, failure modes.

Not in scope: BUY / SELL / HOLD behavior, allocation, urgency, conviction, tradeability, portfolio mutation, watchlist mutation, Telegram, reporting behavior, provider calls, implementation.

Acceptance criteria:

- Financial/scanner/fundamental specification exists.
- Source intake is clearly separated from recommendation and allocation logic.
- Missing-data and source-failure implications are recorded.
- Testing implications use fake or synthetic provider responses.

### ME04 - Extract and write Market Engine technical, coding, and testing architecture

Owner roles: Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Goal: Extract technical architecture, coding rules, and testing architecture for Market Engine.

Scope: Module ownership, provider/data/analysis/decision separation, runtime boundaries, side-effect controls, test-family conventions, manual smoke harness standards.

Not in scope: Implementation unless separately authorized, provider calls, production writes, report generation, Telegram, portfolio/watchlist mutation.

Acceptance criteria:

- Technical architecture specification exists.
- Coding and testing standards are tied to implementation implications.
- Automated test boundaries prohibit live provider calls.
- Manual smoke harness rules are explicit and bounded.

### ME05 - Build all-ticker source intake smoke

Owner roles: Development Lead, Data Steward, QA / Test Lead, Governance Auditor

Goal: Build an explicit all-ticker source intake smoke harness after ME02 through ME04 specifications authorize the boundary.

Scope: Bounded manual source intake smoke harness, source availability capture, failure capture, no recommendation logic.

Not in scope: Production pipeline integration, normal automated tests with live calls, BUY / SELL / HOLD logic, allocation, portfolio/watchlist mutation, reporting, Telegram.

Acceptance criteria:

- Smoke harness is explicit and non-canonical unless promoted by architecture.
- Provider access is bounded and reviewable.
- Failures are captured without stopping the whole batch.
- Missing data remains missing.
- No recommendation or allocation behavior is introduced.

### ME06 - Run all-ticker source coverage and triage failures

Owner roles: Data Steward, QA / Test Lead, Governance Auditor, Operator / User

Goal: Run the approved all-ticker source coverage smoke and triage failures.

Scope: Manual smoke execution, source coverage evidence, failure triage, source-readiness implications, backlog follow-up.

Not in scope: Production writes, normal automated tests with live calls, report generation, Telegram, portfolio/watchlist mutation, recommendation or allocation behavior.

Acceptance criteria:

- Coverage results are documented.
- Ticker failures are triaged.
- Provider/source limitations are recorded.
- Follow-up work is added without changing runtime decision behavior.

### ME07 - Build first analysis pass on collected data

Owner roles: Development Lead, Financial Analyst, Data Steward, QA / Test Lead

Goal: Build the first analysis pass on collected source data after source intake boundaries are proven.

Scope: Analysis-only transformation of collected data, missing-data preservation, source-readiness awareness, synthetic tests.

Not in scope: BUY / SELL / HOLD behavior, allocation, urgency, conviction, tradeability, portfolio mutation, watchlist mutation, Telegram, production reporting.

Acceptance criteria:

- Analysis pass consumes approved collected data boundaries.
- Missing data remains missing.
- Ticker-level failures do not stop batch analysis.
- Tests use fake or fixture data.
- No Decision Engine leakage occurs.

### ME08 - Produce local operator review output

Owner roles: Operator / User, Product Owner, Development Lead, QA / Test Lead, Governance Auditor

Goal: Produce local operator review output for human review without creating allocation authority outside the Decision Engine.

Scope: Local review output, communication formatting, operator usability, side-effect controls, auditability.

Not in scope: Telegram delivery unless separately authorized, production report generation, portfolio/watchlist mutation, recommendation or allocation behavior outside Decision Engine.

Acceptance criteria:

- Local operator review output is produced from approved data and analysis boundaries.
- Output is communication only.
- No lower layer mutates portfolio or watchlist state.
- No hidden provider calls or production side effects occur.
- Operator limitations and next steps are documented.

