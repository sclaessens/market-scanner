# Backlog

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document captures the reset-oriented backlog direction for v2. It does not authorize implementation.

## Backlog Rules

A backlog item may identify future work, but it does not authorize code, tests, file moves, archive/delete actions, data changes, workflows, or runtime behavior changes.

Each item should define:

- title;
- category;
- rationale;
- governance risk;
- owner role;
- proposed next step;
- status.

## Active Reset Epic

### RESET-EPIC — Controlled Clean Rebuild

Category: Architecture Candidate / Governance / Documentation

Rationale: RESET-0 concluded that continued incremental patching risks preserving obsolete assumptions, compatibility surfaces, generated artifact confusion, and fragmented authority. The project should rebuild from certified knowledge.

Governance risk: HIGH

Owner role: PM / Scrum Master with Technical Analyst and Governance Auditor

Status: ACTIVE PLANNING

Proposed next step: Complete RESET-1, then proceed to RESET-2 repository structure and archive plan.

## Captured Reset Items

### RESET-2 — Repository Structure and Archive Plan

Category: Repository Structure / Documentation

Rationale: Old active, sprint, audit, code, test, data, report, and workflow paths must be classified before any archive/delete/move action.

Governance risk: HIGH

Owner role: Documentation Steward / Technical Analyst

Status: CANDIDATE NEXT STAGE

### RESET-3 — V2 Codebase Bootstrap

Category: Technical Architecture / Developer Experience

Rationale: New v2 code must start from clean contracts and should not reuse old Python files as the implementation base.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Codex

Status: BLOCKED BY RESET-2

### RESET-4 — V2 Data Contracts and Fixtures

Category: Data Contract / Testing

Rationale: v2 requires approved inputs, generated-output rules, and fixtures before meaningful runtime implementation.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: BLOCKED BY RESET-2

### SEC/Fundamentals Reclassification

Category: Source Data / Fundamentals

Rationale: SEC-7F should pause as a standalone old-architecture sprint and return under v2 source-data contracts.

Governance risk: HIGH

Owner role: Data Steward / Financial Analyst / Technical Analyst

Status: DEFERRED TO RESET-8

### Legacy Archive/Delete Cutover

Category: Repository Structure / Technical Debt

Rationale: No legacy file should be deleted before its knowledge has been extracted and v2 replacements exist.

Governance risk: MEDIUM

Owner role: Documentation Steward / PM / Technical Analyst

Status: DEFERRED TO RESET-9

## RESET-10L Provider Approval Backlog

These backlog items are documentation and governance items. They do not authorize provider integration, provider calls, data fetching, runtime code, report generation, Telegram delivery, Decision Engine behavior, or production pipeline execution.

### RESET-10L-BL1 — Provider Approval Decision

Category: Source Data / Governance

Rationale: The first real fundamentals provider or provider strategy must be selected and approved before implementation begins.

Governance risk: HIGH

Owner role: Data Steward / Financial Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL1

Decision: The first approved v2 fundamentals provider strategy is primary-first and provenance-first.

Decision record: `docs/active/v2_provider_approval_decision.md`

Proposed next step: Proceed to `RESET-10L-BL2 — Provider Integration Design`.

### RESET-10L-BL2 — Provider Integration Design

Category: Source Data / Architecture

Rationale: Raw capture and normalization mapping must be designed for the approved provider before implementation begins.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Financial Analyst

Status: COMPLETED BY RESET-10L-BL2

Design record: `docs/active/v2_provider_integration_design.md`

Proposed next step: Proceed to `RESET-10L-BL3 — Synthetic Provider Contract Tests`.

### RESET-10L-BL3 — Synthetic Provider Contract Tests

Category: Source Data / Testing

Rationale: Approved provider mapping must be translated into synthetic contract tests before live provider integration.

Governance risk: HIGH

Owner role: Technical Analyst / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL3

Test record: `tests/contract/test_v2_provider_synthetic_contracts.py`

Proposed next step: Proceed to `RESET-10L-BL4 — Real Provider Implementation` only after the synthetic provider contract tests pass.

### RESET-10L-BL4 — Real Provider Implementation

Category: Source Data / Implementation

Rationale: Provider integration may begin only after provider approval, mapping design, and synthetic provider contract tests exist.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL4

Implementation records:

- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `tests/contract/test_v2_fundamentals_provider_contracts.py`
- `tests/unit/test_v2_fundamentals_provider_adapter.py`

Implementation result: A v2-only, dependency-injected provider boundary now accepts governed provider/source responses, preserves raw source evidence, maps supported raw fields into normalized program-ready fundamentals records, and emits neutral source-data readiness. Tests use fake provider responses only.

Proposed next step: Proceed to `RESET-10L-BL5 — Real Provider Dry-Run Fixture Review`.

Guardrails:

- no Decision Engine investment logic;
- no reporting or Telegram delivery side effects;
- no production pipeline execution in tests;
- no missing-to-zero conversion;
- no allocation, tradeability, conviction, urgency, or recommendation behavior outside approved authority.

### RESET-10L-BL5 — Real Provider Dry-Run Fixture Review

Category: Source Data / Verification

Rationale: The provider boundary should be reviewed with a governed dry-run fixture before any live provider client, file writing, production pipeline integration, or downstream Decision Engine use is considered.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL5

Review records:

- `tests/fixtures/fundamentals/provider_dry_run_fixture.json`
- `tests/contract/test_v2_provider_dry_run_fixture_review.py`
- `docs/active/v2_provider_dry_run_fixture_review.md`

Review result: A static ASML-shaped official/regulatory source fixture can move through the v2 provider boundary from provider/source response to raw evidence, normalized fundamentals, and neutral source-data readiness without live calls, data writes, reports, Telegram, pipeline execution, or investment logic.

Proposed next step: Proceed to `RESET-10L-BL6 — Controlled Real-Source Smoke Test`.

Guardrails:

- no live provider, SEC, EDGAR, broker, network, or Telegram calls;
- no production data file creation or modification;
- no report generation;
- no Decision Engine authority expansion;
- no missing-to-zero conversion;
- no BUY, SELL, HOLD, allocation, tradeability, conviction, urgency, or recommendation behavior.

### RESET-10L-BL6 — Controlled Real-Source Smoke Test

Category: Source Data / Verification

Rationale: After the dry-run fixture review, the next step is a separately approved, manually invoked smoke test for controlled real-source access using the v2 provider boundary.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL6

Smoke-test records:

- `src/market_scanner/fundamentals/fundamentals_real_source_smoke.py`
- `tests/unit/test_v2_fundamentals_real_source_smoke.py`
- `docs/active/v2_controlled_real_source_smoke_test.md`

Smoke-test result: A manual-only, dependency-injected v2 smoke harness can pass an explicitly invoked source response through raw evidence capture, normalized fundamentals, neutral source-data readiness, and an in-memory smoke result without automatic execution, live calls in tests, data writes, reports, Telegram, production pipeline behavior, or investment logic.

Proposed next step: Proceed to `RESET-10L-BL7 — Manual Real-Source Smoke Execution Review`.

Guardrails:

- no automatic provider execution;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no Decision Engine authority expansion;
- no credentials committed;
- no data writes unless separately approved;
- no BUY, SELL, HOLD, allocation, tradeability, conviction, urgency, scoring, or recommendation behavior.

### RESET-10L-BL7 — Manual Real-Source Smoke Execution Review

Category: Source Data / Verification

Rationale: After the controlled smoke harness exists, a separate review must define and, if approved, manually execute one controlled real-source smoke run without committing credentials, live output, data files, reports, or production behavior.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL7

Review record: `docs/active/v2_manual_real_source_smoke_execution_review.md`

Review result: The local-only manual execution procedure, pre-run checklist, safe one-ticker/source target pattern, review checklist, no-write policy, allowed local-only output rules, pass/fail criteria, and post-run validation checks are now documented. No live provider/source execution was performed or committed.

Proposed next step: Proceed to `RESET-10L-BL8 — Manual Real-Source Smoke Execution`.

Guardrails:

- manual-only;
- local-only;
- no committed credentials;
- no committed live output;
- no data writes unless separately approved;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, tradeability, conviction, urgency, scoring, or recommendation behavior.

### RESET-10L-BL8 — Manual Real-Source Smoke Execution

Category: Source Data / Verification

Rationale: After the manual execution review procedure exists, one local-only controlled real-source smoke execution may be reviewed under explicit no-write and no-commit guardrails.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL8

Execution guide: `docs/active/v2_manual_real_source_smoke_execution.md`

Execution enablement result: The manual local invocation path is now documented for both explicit `ProviderSourceResponse` review and injected source-client review. The guidance includes terminal-only examples, operator commands, review checklist, post-run safety checks, and no-commit/no-write guardrails. No live provider/source call was made or committed.

Proposed next step: Proceed to `RESET-10L-BL9 — Local Real-Source Smoke Result Review`.

Guardrails:

- manual-only;
- local-only;
- one ticker and one source;
- no committed credentials;
- no committed live output;
- no data writes unless separately approved;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, tradeability, conviction, urgency, scoring, or recommendation behavior.

### RESET-10L-BL9 — Local Real-Source Smoke Result Review

Category: Source Data / Verification

Rationale: After the manual execution path is documented, one local-only smoke result may be reviewed as a governance-safe summary without committing credentials, raw live payloads, generated files, reports, Telegram artifacts, or production behavior.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL9

Review record: `docs/active/v2_local_real_source_smoke_result_review.md`

Review result: The governance-safe local smoke result review process is now documented. It defines the summary-only review scope, allowed redacted fields, pass/fail criteria, post-review safety checks, and no-commit/no-write guardrails. No live provider/source call was made or committed.

Proposed next step: Proceed to `RESET-10L-BL10 — First Local Real-Source Smoke Execution Summary`.

Guardrails:

- manual-only;
- local-only;
- one ticker and one source;
- no committed credentials;
- no committed raw live payload;
- no data writes unless separately approved;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, tradeability, conviction, urgency, scoring, or recommendation behavior;
- summary-only review if anything is committed.

### RESET-10L-BL10 — First Local Real-Source Smoke Execution Summary

Category: Source Data / Verification

Rationale: After the review template exists, one local-only real-source smoke execution may be summarized manually without committing credentials, raw live payloads, generated files, reports, Telegram artifacts, or production behavior.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL10

Summary record: `docs/active/v2_first_local_real_source_smoke_execution_summary.md`

Summary result: A local manual `ProviderSourceResponse` smoke execution was reviewed through the existing v2 smoke harness. The result confirmed raw evidence capture, normalized program-ready fundamentals, neutral source-data readiness, explicit missing values, no missing-to-zero conversion, and no side effects. The result remained `review_required` and `partial` because the manually supplied response intentionally used redacted placeholder values and explicit missing fields. No live provider call, credentials, raw live payload, data files, reports, Telegram artifacts, production pipeline behavior, or Decision Engine investment logic were committed.

Proposed next step: Proceed to `RESET-10L-BL11 — Real-Source Capture Persistence Design`.

Guardrails:

- manual-only;
- local-only;
- one ticker and one source;
- no committed credentials;
- no committed raw live payload;
- no data writes unless separately approved;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, tradeability, conviction, urgency, scoring, target-price, or recommendation behavior;
- redacted summary-only review if anything is committed.

### RESET-10L-BL11 — Real-Source Capture Persistence Design

Category: Source Data / Architecture

Rationale: After the first local smoke execution summary, the project must design how real source capture and normalized fundamentals may be persisted without violating raw-to-normalized separation, provenance requirements, missing-value behavior, data-write guardrails, or Decision Engine authority boundaries.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL11

Design record: `docs/active/v2_real_source_capture_persistence_design.md`

Design result: The proposed persistence boundary is now documented for raw source evidence, normalized program-ready fundamentals, and neutral source-data readiness. The design preserves raw-to-normalized separation, provenance linkage, explicit missing values, neutral readiness, write authorization requirements, failure handling, and Decision Engine authority boundaries. No implementation, provider calls, production data writes, reports, Telegram artifacts, or investment logic were authorized.

Proposed next step: Proceed to `RESET-10L-BL12 — Persistence Contract and Fixture Design`.

Guardrails:

- design-only;
- no automatic provider execution;
- no committed credentials;
- no committed raw live payload;
- no production data writes unless separately approved;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, tradeability, conviction, urgency, scoring, target-price, or recommendation behavior.

### RESET-10L-BL12 — Persistence Contract and Fixture Design

Category: Source Data / Testing

Rationale: After persistence design, the project must translate the raw evidence, normalized fundamentals, and readiness persistence boundaries into explicit schemas, fixtures, and contract-test requirements before any production write function is implemented.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL12

Design record: `docs/active/v2_persistence_contract_and_fixture_design.md`

Design result: The persistence contract and fixture design now defines raw evidence, normalized fundamentals, and readiness schema expectations; synthetic fixture families; future fixture and contract-test path candidates; acceptance criteria for synthetic persistence contract tests; and guardrails that prohibit provider calls, production data writes, reports, Telegram artifacts, pipeline execution, Decision Engine behavior, and investment semantics.

Proposed next step: Proceed to `RESET-10L-BL13 — Synthetic Persistence Contract Tests`.

Guardrails:

- design and test-contract planning only;
- no automatic provider execution;
- no committed credentials;
- no committed raw live payload;
- no production data writes unless separately approved;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, tradeability, conviction, urgency, scoring, target-price, or recommendation behavior.

### RESET-10L-BL13 — Synthetic Persistence Contract Tests

Category: Source Data / Testing

Rationale: After persistence contract and fixture design, the project may add synthetic persistence fixtures and contract tests that prove raw evidence, normalized fundamentals, and readiness persistence expectations without production writes.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL13

Fixture records:

- `tests/fixtures/fundamentals/persistence/raw_complete_source.json`
- `tests/fixtures/fundamentals/persistence/raw_partial_source.json`
- `tests/fixtures/fundamentals/persistence/raw_invalid_source.json`
- `tests/fixtures/fundamentals/persistence/raw_stale_source.json`
- `tests/fixtures/fundamentals/persistence/raw_provenance_gap_source.json`
- `tests/fixtures/fundamentals/persistence/raw_forbidden_semantics_source.json`

Test records:

- `tests/contract/test_v2_persistence_raw_evidence_contracts.py`
- `tests/contract/test_v2_persistence_normalized_fundamentals_contracts.py`
- `tests/contract/test_v2_persistence_readiness_contracts.py`
- `tests/contract/test_v2_persistence_fixture_contracts.py`

Test result: Synthetic persistence contract tests now verify raw evidence schema expectations, normalized fundamentals expectations, neutral readiness expectations, raw-to-normalized provenance linkage, explicit missing-value preservation, no missing-to-zero conversion, neutral partial/invalid/stale/provenance-gap handling, controlled forbidden-semantics isolation, fixture safety metadata, and no-side-effect guardrails. Tests use synthetic static fixture inspection only.

Proposed next step: Proceed to `RESET-10L-BL14 — Controlled Persistence Implementation Design`.

Guardrails:

- synthetic fixtures and contract tests only;
- no live provider calls;
- no SEC, EDGAR, broker, or network calls;
- no committed credentials;
- no committed raw live payload;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, tradeability, conviction, urgency, scoring, target-price, or recommendation behavior.

### RESET-10L-BL14 — Controlled Persistence Implementation Design

Category: Source Data / Architecture

Rationale: After synthetic persistence fixtures and contract tests exist, the next safer step is to design controlled persistence implementation boundaries before any production write function is approved.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Governance Auditor

Status: CANDIDATE NEXT STAGE

Proposed next step: Define design-only controlled persistence implementation boundaries, write authorization rules, no-side-effect requirements, rollback expectations, and schema-to-test traceability.

Guardrails:

- design-only unless separately approved;
- no live provider calls;
- no SEC, EDGAR, broker, or network calls;
- no committed credentials;
- no committed raw live payload;
- no production data writes unless separately approved;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

## Relationship to Existing Backlog

The historical `docs/sprints/project_backlog.md` remains preserved as legacy planning evidence until RESET-2 decides how to reconcile or archive it. This document is the v2 reset-facing backlog baseline.
