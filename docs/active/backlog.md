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

Status: COMPLETED BY RESET-10L-BL14

Design record: `docs/active/v2_controlled_persistence_implementation_design.md`

Design result: The controlled persistence implementation boundary is now documented. The design defines a future v2-only persistence module boundary, pure validators, synthetic-only temporary write support, forbidden production paths, missing-value behavior, forbidden semantics guardrails, failure behavior, write authorization rules, no-side-effect requirements, rollback expectations, schema-to-test traceability, and required future test coverage. No code, tests, fixtures, provider calls, production data writes, reports, Telegram artifacts, pipeline behavior, or Decision Engine investment logic were authorized.

Proposed next step: Proceed to `RESET-10L-BL15 — Controlled Synthetic Persistence Implementation`.

Guardrails:

- design-only;
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

### RESET-10L-BL15 — Controlled Synthetic Persistence Implementation

Category: Source Data / Implementation

Rationale: After controlled persistence implementation design, the project may implement a v2-only synthetic persistence boundary that validates and writes only synthetic/test records to pytest-managed temporary directories.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL15

Implementation records:

- `src/market_scanner/fundamentals/fundamentals_persistence.py`
- `tests/unit/test_v2_fundamentals_persistence.py`

Implementation result: The controlled v2 synthetic persistence boundary now validates raw evidence, normalized fundamentals, and readiness records; preserves provenance linkage and explicit missing-value behavior; rejects forbidden production/report/Telegram paths; supports synthetic-only temporary writes; and remains disconnected from provider calls, production data paths, reports, Telegram, pipeline execution, and Decision Engine investment logic.

Proposed next step: Proceed to `RESET-10L-BL16 — Persistence Integration Review`.

Guardrails:

- synthetic-only implementation;
- tests must use pytest temporary directories only;
- no live provider calls;
- no SEC, EDGAR, broker, or network calls;
- no committed credentials;
- no committed raw live payload;
- no production data writes;
- no writes under `data/`;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL16 — Persistence Integration Review

Category: Source Data / Review

Rationale: After the controlled synthetic persistence boundary exists, the next step should review integration readiness, guardrail coverage, and any remaining approval requirements before production persistence or runtime integration is considered.

Governance risk: HIGH

Owner role: Technical Analyst / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL16

Review record: `docs/active/v2_persistence_integration_review.md`

Review result: The controlled synthetic persistence boundary is suitable for a future separately approved synthetic provider-to-persistence integration contract step. The review confirms that BL15 remains a persistence safety boundary only and is not yet approved for production data writes, live provider execution, reports, Telegram delivery, pipeline integration, portfolio/watchlist integration, or Decision Engine investment use.

Proposed next step: Proceed to `RESET-10L-BL17 — Synthetic Provider-to-Persistence Integration Contracts`.

Guardrails:

- review-only;
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

### RESET-10L-BL17 — Synthetic Provider-to-Persistence Integration Contracts

Category: Source Data / Testing

Rationale: After persistence integration review, the project may add synthetic integration contracts proving that fake provider-boundary output can be handed to the controlled persistence boundary while preserving provenance, explicit missing-value behavior, neutral readiness, and no-side-effect guardrails.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL17

Test record: `tests/contract/test_v2_provider_to_persistence_integration_contracts.py`

Result summary: Synthetic provider-to-persistence integration contracts now prove that fake provider-boundary output can be handed to the controlled persistence boundary while preserving provenance linkage, explicit missing-value behavior, neutral readiness, synthetic tmp_path writes, forbidden-path rejection, and no-side-effect guardrails. The contracts remain synthetic-only and do not authorize production data writes, live provider execution, reports, Telegram, pipeline integration, portfolio/watchlist integration, or Decision Engine investment use.

Proposed next step: Proceed to `RESET-10L-BL18 — Production Persistence Readiness Review`.

Guardrails:

- synthetic integration contracts only;
- fake provider output only;
- pytest temporary directories only;
- no live provider calls;
- no SEC, EDGAR, broker, or network calls;
- no committed credentials;
- no committed raw live payload;
- no production data writes;
- no writes under `data/`;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist integration;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL18 — Approve One-Ticker Real-Source Data Capture

Category: Source Data / Governance

Rationale: After synthetic provider-to-persistence integration contracts, the project should stop adding broad readiness layers and approve one controlled real-source data capture target so the application can start learning from real fundamentals.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL18

Approval record: `docs/active/v2_one_ticker_real_source_data_capture_approval.md`

Approval result: One controlled real-source data capture target is approved for the next sprint. The approved ticker is `NVDA` / NVIDIA Corporation. The approval is limited to one ticker, one source family, and one controlled local execution path. It does not authorize broad provider integration, automatic execution, multi-ticker capture, production pipeline execution, reports, Telegram delivery, portfolio/watchlist updates, or Decision Engine investment behavior.

Proposed next step: Proceed to `RESET-10L-BL19 — Execute One-Ticker Real-Source Persistence Smoke`.

Guardrails:

- one ticker only: `NVDA`;
- one source family only;
- controlled local smoke only;
- no committed credentials;
- no committed raw live payload unless separately redacted and explicitly approved;
- no broad production persistence;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist integration;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL19 — Execute One-Ticker Real-Source Persistence Smoke

Category: Source Data / Verification

Rationale: After one-ticker real-source data capture approval, the project may execute a controlled NVDA real-source persistence smoke to determine whether real fundamentals can move through the provider/source, normalization, readiness, and persistence boundaries without triggering downstream behavior.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL19

Smoke record: `docs/active/v2_nvda_real_source_persistence_smoke.md`

Smoke result: A controlled NVDA one-ticker real-source persistence smoke was executed. The run produced a governance-safe redacted summary showing source family, observed fields, missing fields, neutral readiness, persistence-boundary result, and side-effect checks. No credentials, raw unredacted live payloads, production data writes, reports, Telegram artifacts, production pipeline behavior, portfolio/watchlist updates, or Decision Engine investment behavior were committed.

Proposed next step: Proceed to `RESET-10L-BL20 — Run First One-Ticker Real Fundamental Analysis`.

Guardrails:

- one ticker only: `NVDA`;
- one source family only;
- explicit local invocation only;
- no committed credentials;
- no committed raw unredacted live payload;
- no multi-ticker capture;
- no automatic provider scheduling;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist integration;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL20 — Run First One-Ticker Real Fundamental Analysis

Category: Fundamentals / Analysis Review

Rationale: After the controlled NVDA real-source persistence smoke succeeded with explicit partial readiness, the project may run a first one-ticker real fundamental analysis review to learn how real source-data limitations affect analysis behavior.

Governance risk: HIGH

Owner role: Technical Analyst / Data Steward / Financial Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL20

Analysis record: `docs/active/v2_nvda_first_real_fundamental_analysis_review.md`

Analysis result: A controlled NVDA one-ticker real fundamental analysis review was executed using the BL19 real-source smoke findings. The review carried FreeCashFlow missingness forward explicitly, observed readiness behavior, documented analysis behavior, and did not produce final BUY/SELL/HOLD, portfolio action, reports, Telegram artifacts, production pipeline execution, or Decision Engine investment behavior.

Proposed next step: Proceed to `RESET-10L-BL21 — Govern FreeCashFlow Derivation or Missingness Policy`.

Guardrails:

- one ticker only: `NVDA`;
- controlled local execution only;
- no multi-ticker run;
- no automated scheduling;
- no reports unless separately approved;
- no Telegram unless separately approved;
- no portfolio or watchlist modification;
- no production data writes unless separately approved;
- no BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior unless explicitly part of an approved analysis review.

### RESET-10L-BL21 — Govern FreeCashFlow Derivation or Missingness Policy

Category: Fundamentals / Data Governance

Rationale: The first controlled NVDA real fundamental analysis review showed that useful cash-flow analysis remains limited when direct FreeCashFlow is absent, even when operating cash flow and capital expenditures are available. The project needs a governed rule before any derivation may be added.

Governance risk: HIGH

Owner role: Data Steward / Financial Analyst / Governance Auditor / Developer

Status: COMPLETED BY RESET-10L-BL21

Policy record: `docs/active/v2_free_cash_flow_derivation_policy.md`

Policy result: Option C is approved. V2 may support both directly sourced FreeCashFlow and governed derived FreeCashFlow. Derived FreeCashFlow may only be produced from operating cash flow minus capital expenditures when both inputs are valid, same-period, same-currency, same-unit, provenance-linked, and clearly marked as derived. Missing values must remain explicit, no missing value may be converted to zero, and ambiguous sign conventions or mismatched inputs must fail closed as not derivable.

Proposed next step: Proceed to `RESET-10L-BL22 — Implement Governed FreeCashFlow Derivation`.

Guardrails:

- governed derivation only;
- no silent FreeCashFlow derivation;
- no missing-to-zero conversion;
- derived metrics must preserve provenance for both input fields;
- derived metrics must be visibly marked as derived;
- mismatched currency, unit, period, fiscal context, missing provenance, or ambiguous sign convention must fail closed;
- implementation must update existing modules wherever possible;
- Python file creation policy applies;
- no committed credentials;
- no committed raw unredacted live payload;
- no production data writes unless separately approved;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist integration;
- no Decision Engine investment logic;
- no BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL22 — Implement Governed FreeCashFlow Derivation

Category: Fundamentals / Implementation

Rationale: After FreeCashFlow derivation governance approval, the project may implement governed FreeCashFlow derivation so real-source analysis can progress beyond CASH_FLOW_UNKNOWN when operating cash flow and capital expenditures are available and valid.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL22

Implementation records:

- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_real_source_smoke.py`
- `tests/contract/test_v2_provider_dry_run_fixture_review.py`
- `tests/contract/test_v2_provider_to_persistence_integration_contracts.py`
- `docs/active/v2_free_cash_flow_derivation_implementation.md`

Result summary: Governed FreeCashFlow derivation is implemented. The v2 fundamentals mapping now supports directly sourced FreeCashFlow as source_reported and derived FreeCashFlow as source_derived when operating cash flow and capital expenditures are valid, same-period, same-currency, same-unit, provenance-linked, and sign-convention safe. Missing, invalid, mismatched, or ambiguous inputs fail closed without missing-to-zero conversion or investment behavior.

Proposed next step: Proceed to `RESET-10L-BL23 — Re-run NVDA One-Ticker Real Fundamental Analysis with Derived FreeCashFlow`.

Guardrails:

- update existing Python files first;
- no new Python file unless formally justified under `docs/active/v2_python_file_creation_policy.md`;
- no one-off ticker-specific Python files;
- no silent derivation;
- no missing-to-zero conversion;
- no investment recommendation behavior;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist integration;
- no BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL23 — Re-run NVDA One-Ticker Real Fundamental Analysis with Derived FreeCashFlow

Category: Fundamentals / Analysis Review

Rationale: After governed FreeCashFlow derivation is implemented, the project should re-run the controlled one-ticker NVDA analysis review to determine whether NVDA can move beyond CASH_FLOW_UNKNOWN while preserving source-data readiness, provenance, and non-recommendation guardrails.

Governance risk: HIGH

Owner role: Technical Analyst / Data Steward / Financial Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL23

Analysis record: `docs/active/v2_nvda_real_analysis_rerun_with_derived_fcf.md`

Analysis result: The controlled NVDA one-ticker real fundamental analysis re-run used governed source_derived FreeCashFlow and moved beyond the previous CASH_FLOW_UNKNOWN blocker. The review documented updated readiness, remaining limitations, and did not produce final BUY/SELL/HOLD, portfolio action, reports, Telegram artifacts, production pipeline execution, production data writes, or Decision Engine investment behavior.

Proposed next step: Proceed to `RESET-10L-BL24 — Real Analysis Output Defect Review`.

Guardrails:

- one ticker only: `NVDA`;
- controlled local execution only;
- no multi-ticker run;
- no automated scheduling;
- no committed credentials;
- no committed raw unredacted live payload;
- no production data writes unless separately approved;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL24 — Real Analysis Output Defect Review

Category: Fundamentals / Analysis Review

Rationale: After governed derived FreeCashFlow resolved the NVDA CASH_FLOW_UNKNOWN blocker, the next limitation is incomplete real analysis metric coverage, especially prior-year growth evidence and clearer review signaling for partial metrics.

Governance risk: HIGH

Owner role: Technical Analyst / Data Steward / Financial Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL24

Review record: `docs/active/v2_real_analysis_output_defect_review.md`

Review result: The NVDA real analysis path improved after governed derived FreeCashFlow. `CASH_FLOW_UNKNOWN` is resolved, the cash-flow profile moved to `CASH_FLOW_POSITIVE`, source-data readiness is available, and `missing_fundamentals_count = 0`. The remaining blocker is `LIMITED_ANALYSIS` caused by missing governed prior-year growth evidence. The blocker is classified as `MISSING_GOVERNED_PRIOR_YEAR_GROWTH_EVIDENCE`.

Proposed next step: Proceed to `RESET-10L-BL25 — Implement Governed Prior-Year Growth Evidence`.

Guardrails:

- review/documentation-only;
- no code changes;
- no test changes;
- no provider calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist integration;
- no Decision Engine investment logic;
- no final BUY, SELL, or HOLD recommendation;
- no missing-to-zero conversion;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL25 — Implement Governed Prior-Year Growth Evidence

Category: Fundamentals / Implementation

Rationale: After the real analysis output defect review, the next blocker is governed prior-year growth evidence. The project may implement a narrow evidence layer so real analysis can compare current and prior comparable periods without missing-to-zero conversion or recommendation behavior.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Financial Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL25

Implementation records:

- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_provider_adapter.py`
- `tests/contract/test_v2_fundamentals_provider_contracts.py`
- `docs/active/v2_prior_year_growth_evidence_implementation.md`

Result summary: Governed prior-year growth evidence is implemented. The v2 fundamentals/analysis path now supports explicit growth evidence states for comparable current/prior metric values, preserves current and prior provenance, computes growth only when values are valid and comparable, and fails closed for missing, zero, invalid, not-parseable, mismatched, or provenance-gap inputs without missing-to-zero conversion or investment behavior.

Proposed next step: Proceed to `RESET-10L-BL26 — Re-run NVDA Real Analysis with Governed Growth Evidence`.

Guardrails:

- update existing Python files first;
- no new Python file unless formally justified under `docs/active/v2_python_file_creation_policy.md`;
- no one-off ticker-specific Python files;
- no silent growth inference;
- no missing-to-zero conversion;
- prior-year growth evidence must preserve current and prior value provenance;
- zero, missing, invalid, not-parseable, mismatched, or provenance-gap inputs must fail closed;
- no investment recommendation behavior;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist integration;
- no BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL26 — Re-run NVDA Real Analysis with Governed Growth Evidence

Category: Fundamentals / Analysis Review

Rationale: After governed prior-year growth evidence is implemented, the project should re-run the controlled NVDA real analysis path to determine whether the remaining LIMITED_ANALYSIS blocker is resolved.

Governance risk: HIGH

Owner role: Technical Analyst / Data Steward / Financial Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL26

Analysis record: `docs/active/v2_nvda_real_analysis_rerun_with_growth_evidence.md`

Analysis result: The controlled NVDA real analysis re-run used governed source_derived FreeCashFlow and governed prior-year growth evidence, but LIMITED_ANALYSIS remains. The remaining blocker is documented in `docs/active/v2_nvda_real_analysis_rerun_with_growth_evidence.md`. No final recommendation, portfolio action, reports, Telegram artifacts, production pipeline execution, production data writes, or Decision Engine investment behavior were produced.

Proposed next step: Proceed to `RESET-10L-BL27 — Python Architecture Cleanup and Legacy Decoupling Review`.

Guardrails:

- one ticker only: `NVDA`;
- controlled local execution only;
- no multi-ticker run;
- no automated scheduling;
- no committed credentials;
- no committed raw unredacted live payload;
- no production data writes unless separately approved;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no missing-to-zero conversion;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL27 — Python Architecture Cleanup and Legacy Decoupling Review

Category: Architecture / Cleanup Review

Rationale: After governed real-analysis evidence improved the NVDA path, the project must understand Python runtime ownership, legacy dependencies, duplicate entrypoints, scanner/report/Telegram coupling, and cleanup sequencing before adding more real-analysis features.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL27

Review record: `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`

Result summary: The Python architecture cleanup and legacy decoupling review inventoried committed Python files, identified multiple runnable legacy entrypoints, classified canonical v2 modules and script-era dependencies, mapped duplicate responsibilities across scanner, fundamentals analysis, Decision Engine, reporting, Telegram, portfolio, watchlist, configuration, and shared utilities, and proposed a canonical v2 ownership map and cleanup sequence. The review confirms that still-used legacy files are temporary dependencies, not automatically approved long-term owners. No Python files, tests, runtime behavior, data files, reports, workflows, portfolio/watchlist files, or Telegram artifacts were changed.

Proposed next step: Proceed to `RESET-10L-BL28 — Define Canonical V2 Runtime Architecture`.

Guardrails:

- review-only unless separately approved;
- no Python code changes;
- no test changes;
- no file moves or deletions;
- no provider calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist updates;
- no Decision Engine investment behavior changes;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL28 — Define Canonical V2 Runtime Architecture

Category: Architecture / Design Review

Rationale: The BL27 cleanup review found multiple unclear runners, script-era scanner and analysis owners, reporting/Telegram coupling, and legacy dependencies that should not be moved or deleted until canonical v2 ownership is defined.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL28

Architecture record: `docs/active/v2_canonical_runtime_architecture.md`

Architecture result: The canonical v2 runtime architecture is now defined. The project must converge toward a single official application entrypoint, scanner/universe flow, fundamentals/provider/evidence layer, analysis layer, decision/review boundary, message composition layer, report artifact boundary, delivery/Telegram boundary, configuration owner, and shared utilities boundary. Legacy runners such as `scripts/run_scan.py` and `scripts/run_full_pipeline.py` are not approved as permanent canonical runtime authorities and must be treated as migration targets.

Proposed next step: Proceed to `RESET-10L-BL29 — Migrate Legacy Runtime Entrypoint Logic`.

Guardrails:

- documentation/governance-only;
- no Python code changes;
- no test changes;
- no file moves or deletions;
- no provider calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist updates;
- no Decision Engine investment behavior changes;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL29 — Migrate Legacy Runtime Entrypoint Logic

Category: Architecture / Cleanup Implementation

Rationale: After canonical v2 runtime ownership is defined, the project may begin decoupling runtime entrypoint authority from legacy runners and migrating required logic toward the canonical v2 application entrypoint.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL29

Implementation records:

- `src/market_scanner/app.py`
- `tests/unit/test_v2_canonical_app.py`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`

Result summary: Runtime entrypoint authority migration has started. The canonical v2 application boundary is now established under `src/market_scanner/app.py`, with tests proving a deterministic, side-effect-free dry-run boundary. Legacy runners remain present but are no longer the only runtime authority and remain migration/archive candidates under the legacy decoupling policy.

Proposed next step: Proceed to `RESET-10L-BL30 — Migrate Scanner Runtime Logic to Canonical V2 Boundary`.

Guardrails:

- update existing Python files first;
- no new Python file unless formally justified under `docs/active/v2_python_file_creation_policy.md`;
- no one-off migration helper files committed to the repository;
- no file deletion or archival in this sprint unless separately approved;
- no scanner, analysis, reporting, Telegram, or Decision Engine behavior expansion;
- no provider calls;
- no production data writes;
- no production pipeline execution unless explicitly safe and approved;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior.

### RESET-10L-BL30 — Migrate Scanner Runtime Logic to Canonical V2 Boundary

Category: Architecture / Cleanup Implementation

Rationale: After the canonical v2 application boundary is established, the next migration target is scanner and universe-selection runtime logic currently concentrated in legacy scripts. Scanner migration should start with a side-effect-free canonical boundary before any production data writes or provider execution are connected.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL30

Implementation records:

- `src/market_scanner/app.py`
- `src/market_scanner/scanner/__init__.py`
- `src/market_scanner/scanner/scanner_contracts.py`
- `src/market_scanner/scanner/scanner_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_scanner.py`
- `docs/active/v2_scanner_runtime_boundary_migration.md`

Result summary: Scanner runtime migration has started. The canonical v2 scanner/universe boundary is now established and integrated into the side-effect-free canonical app dry-run plan. Legacy runtime scripts remain present but are not canonical scanner authorities and remain migration/archive candidates under the legacy decoupling policy.

Proposed next step: Proceed to `RESET-10L-BL31 — Migrate Analysis Runtime Logic to Canonical V2 Boundary`.

Guardrails:

- update existing Python files first unless BL28-approved canonical ownership requires a new file;
- no one-off migration helper files committed to the repository;
- no file deletion or archival unless separately approved;
- no live provider calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- legacy runner authority must not be expanded.

### RESET-10L-BL31 — Migrate Analysis Runtime Logic to Canonical V2 Boundary

Category: Architecture / Cleanup Implementation

Rationale: After the canonical app and scanner boundaries are established, the next migration target is analysis runtime logic currently concentrated in script-era fundamentals analysis files. The analysis boundary should start as side-effect-free and evidence/review-oriented before any production runner integration is approved.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: CANDIDATE NEXT STAGE

Proposed next step: Establish the canonical v2 analysis boundary without production data writes, report generation, Telegram delivery, portfolio/watchlist updates, or Decision Engine investment behavior.

Guardrails:

- update existing Python files first unless BL28-approved canonical ownership requires a new file;
- no one-off migration helper files committed to the repository;
- no file deletion or archival unless separately approved;
- no live provider calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- legacy runner authority must not be expanded.

## Relationship to Existing Backlog

The historical `docs/sprints/project_backlog.md` remains preserved as legacy planning evidence until RESET-2 decides how to reconcile or archive it. This document is the v2 reset-facing backlog baseline.
