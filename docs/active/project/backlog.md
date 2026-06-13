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

Status: COMPLETED BY RESET-10L-BL31

Implementation records:

- `src/market_scanner/app.py`
- `src/market_scanner/analysis/__init__.py`
- `src/market_scanner/analysis/analysis_contracts.py`
- `src/market_scanner/analysis/analysis_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_analysis.py`
- `docs/active/v2_analysis_runtime_boundary_migration.md`

Result summary: Analysis runtime migration has started. The canonical v2 analysis boundary is now established and integrated into the side-effect-free canonical app dry-run plan. Legacy runtime scripts and scattered analysis logic remain present but are not canonical analysis authorities and remain migration/archive candidates under the legacy decoupling policy.

Proposed next step: Proceed to `RESET-10L-BL32 — Migrate Decision Review Runtime Logic to Canonical V2 Boundary`.

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

### RESET-10L-BL32 — Migrate Decision Review Runtime Logic to Canonical V2 Boundary

Category: Architecture / Cleanup Implementation

Rationale: After canonical app, scanner, and analysis boundaries are established, the next migration target is the decision/review runtime boundary. This step should keep Decision Engine allocation authority isolated while defining the canonical review boundary that receives analysis evidence without producing unapproved investment behavior.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL32

Implementation records:

- `src/market_scanner/app.py`
- `src/market_scanner/decision/__init__.py`
- `src/market_scanner/decision/decision_contracts.py`
- `src/market_scanner/decision/decision_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_decision.py`
- `docs/active/v2_decision_review_runtime_boundary_migration.md`

Result summary: Decision/review runtime migration has started. The canonical v2 decision/review boundary is now established and integrated into the side-effect-free canonical app dry-run plan. Legacy runtime scripts and legacy Decision Engine logic remain present but are not canonical decision/review authorities and remain migration/archive candidates under the legacy decoupling policy. The new boundary explicitly blocks final recommendation semantics.

Proposed next step: Proceed to `RESET-10L-BL33 — Migrate Message Composition Runtime Logic to Canonical V2 Boundary`.

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
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior outside the approved Decision Engine authority;
- legacy runner and legacy analysis authority must not be expanded.

### RESET-10L-BL33 — Migrate Message Composition Runtime Logic to Canonical V2 Boundary

Category: Architecture / Cleanup Implementation

Rationale: After canonical app, scanner, analysis, and decision/review boundaries are established, the next migration target is message composition. This step should separate message composition from report generation and delivery while keeping Telegram delivery and production reports disconnected unless separately approved.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL33

Completion record:

- `src/market_scanner/app.py`
- `src/market_scanner/messaging/__init__.py`
- `src/market_scanner/messaging/message_contracts.py`
- `src/market_scanner/messaging/message_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_messaging.py`
- `docs/active/v2_message_composition_runtime_boundary_migration.md`

Result summary: Message composition runtime migration has started. The canonical v2 message composition boundary is now established and integrated into the side-effect-free canonical app dry-run plan. Legacy runtime scripts, report builders, Telegram delivery files, and scattered message composition logic remain present but are not canonical message composition authorities and remain migration/archive candidates under the legacy decoupling policy. The new boundary explicitly separates message composition from delivery.

Proposed next step: Proceed to `RESET-10L-BL34 — Migrate Report Artifact Runtime Logic to Canonical V2 Boundary`.

Guardrails:

- update existing Python files first unless BL28-approved canonical ownership requires a new file;
- no one-off migration helper files committed to the repository;
- no file deletion or archival unless separately approved;
- no live provider calls;
- no production data writes;
- no production pipeline execution;
- no report generation unless separately approved;
- no Telegram delivery unless separately approved;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- legacy runner, legacy Decision Engine, legacy reporting, and legacy Telegram authority must not be expanded.

### RESET-10L-BL34 — Migrate Report Artifact Runtime Logic to Canonical V2 Boundary

Category: Architecture / Cleanup Implementation

Rationale: After canonical message composition is established, report artifact ownership should be separated from message composition and Telegram delivery. This step should define the canonical report artifact boundary without enabling production report generation unless separately approved.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL34

Completion record:

- `src/market_scanner/app.py`
- `src/market_scanner/reporting/__init__.py`
- `src/market_scanner/reporting/report_contracts.py`
- `src/market_scanner/reporting/report_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_reporting.py`
- `docs/active/v2_report_artifact_runtime_boundary_migration.md`

Result summary: Report artifact runtime migration has started. The canonical v2 report artifact boundary is now established and integrated into the side-effect-free canonical app dry-run plan. Legacy runtime scripts, report builders, Telegram delivery files, and scattered report artifact logic remain present but are not canonical report artifact authorities and remain migration/archive candidates under the legacy decoupling policy. The new boundary explicitly separates report artifact planning from file writing, message composition, and delivery.

Proposed next step: Proceed to `RESET-10L-BL35 — Migrate Delivery and Telegram Runtime Logic to Canonical V2 Boundary`.

Guardrails:

- update existing Python files first unless BL28-approved canonical ownership requires a new file;
- no one-off migration helper files committed to the repository;
- no file deletion or archival unless separately approved;
- no live provider calls;
- no production data writes;
- no production pipeline execution;
- no report generation unless separately approved;
- no Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- message composition, report artifact generation, and delivery must remain separate;
- legacy runner, legacy Decision Engine, legacy reporting, and legacy Telegram authority must not be expanded.

### RESET-10L-BL35 — Migrate Delivery and Telegram Runtime Logic to Canonical V2 Boundary

Category: Architecture / Cleanup Implementation

Rationale: After canonical message composition and report artifact planning are established, delivery ownership should be separated into its own canonical boundary. Telegram delivery must remain disconnected from message composition and report artifact planning unless a future sprint explicitly approves delivery behavior.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL35

Completion record:

- `src/market_scanner/app.py`
- `src/market_scanner/delivery/__init__.py`
- `src/market_scanner/delivery/delivery_contracts.py`
- `src/market_scanner/delivery/delivery_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_delivery.py`
- `docs/active/v2_delivery_runtime_boundary_migration.md`

Result summary: Delivery runtime migration has started. The canonical v2 delivery boundary is now established and integrated into the side-effect-free canonical app dry-run plan. Legacy runtime scripts, Telegram senders, delivery files, report builders, and scattered notification logic remain present but are not canonical delivery authorities and remain migration/archive candidates under the legacy decoupling policy. The new boundary explicitly separates delivery planning from Telegram execution, message composition, report artifact planning, network calls, and credential access.

Proposed next step: Proceed to `RESET-10L-BL36 — Legacy Runtime Script Archive Readiness Review`.

Guardrails:

- update existing Python files first unless BL28-approved canonical ownership requires a new file;
- no one-off migration helper files committed to the repository;
- no file deletion or archival unless separately approved;
- no live provider calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram delivery unless separately approved;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- message composition, report artifact planning, and delivery must remain separate;
- legacy runner, legacy Decision Engine, legacy reporting, and legacy Telegram authority must not be expanded.

### RESET-10L-BL36 — Legacy Runtime Script Archive Readiness Review

Category: Architecture / Cleanup Review

Rationale: After the canonical app, scanner, analysis, decision/review, message composition, report artifact, and delivery boundaries are established, the repository needs a controlled archive-readiness review for legacy runtime scripts before any move, deletion, or behavioral migration.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL36

Review record:

- `docs/active/v2_legacy_runtime_script_archive_readiness_review.md`

Result summary: Legacy runtime script archive readiness was reviewed for `scripts/run_scan.py` and `scripts/run_full_pipeline.py`. Neither script is archive-ready. `scripts/run_scan.py` remains referenced by an active workflow and active tests, still owns broad executable runtime sequencing and side effects, and still invokes scanner execution, production data/report writes, reporting, Telegram delivery, portfolio/intelligence layers, and legacy Decision Engine behavior. `scripts/run_full_pipeline.py` remains an active test dependency and subprocess wrapper around `scripts/run_scan.py`. Canonical v2 boundaries exist but remain planning-only for most runtime responsibilities, so dependencies and remaining logic must be decoupled before archive.

Proposed next step: Proceed to `RESET-10L-BL37 — Decouple Remaining Legacy Runtime Dependencies`.

Guardrails:

- review-only unless a future sprint separately approves implementation;
- no Python runtime changes;
- no file deletion, move, or archival in the review sprint;
- no live provider calls;
- no network calls;
- no credential reads;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- canonical delivery, message composition, report artifact planning, and legacy runtime responsibilities must remain separate;
- legacy runner, legacy Decision Engine, legacy reporting, legacy message, and legacy Telegram authority must not be expanded.

### RESET-10L-BL37 — Decouple Remaining Legacy Runtime Dependencies

Category: Architecture / Cleanup Implementation

Rationale: BL36 found that the legacy runtime scripts are not archive-ready because active workflow and test dependencies remain, and broad executable runtime logic has not yet been migrated, replaced, or retired through canonical v2 owners.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL37

Implementation records:

- `.github/workflows/daily-market-scan.yml`
- `src/market_scanner/app.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/core/test_fundamentals_runtime_organization.py`
- `docs/active/v2_legacy_runtime_dependency_decoupling.md`

Result summary: Remaining legacy runtime dependencies were partially decoupled. The active workflow now targets the canonical v2 app dry-run instead of `scripts/run_scan.py`, and one narrow fundamentals namespace test no longer imports the legacy runner. The canonical app now exposes a guarded dry-run CLI that fails closed for non-dry-run execution. The remaining test, wrapper, and logic dependencies are documented in `docs/active/v2_legacy_runtime_dependency_decoupling.md`. Legacy scripts remain present and are not yet archive-ready.

Proposed next step: Proceed to `RESET-10L-BL38 — Decouple Remaining Legacy Runtime Blockers`.

Guardrails:

- no archive, delete, move, or rename until dependencies are removed and separately approved;
- no live provider calls unless separately approved;
- no credential reads;
- no network calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- canonical delivery, message composition, report artifact planning, Decision Engine behavior, and runtime entrypoint responsibilities must remain separate;
- legacy runner, legacy Decision Engine, legacy reporting, legacy message, and legacy Telegram authority must not be expanded.

### RESET-10L-BL38 — Decouple Remaining Legacy Runtime Blockers

Category: Architecture / Cleanup Implementation

Rationale: BL37 removed the active workflow dependency on `scripts/run_scan.py` and added a certified canonical app dry-run CLI, but active tests still import and monkeypatch `scripts/run_scan.py` and `scripts/run_full_pipeline.py`, and `scripts/run_full_pipeline.py` still shells into `scripts/run_scan.py`.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL38

Implementation records:

- `scripts/run_full_pipeline.py`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- `docs/active/v2_legacy_runtime_blocker_decoupling.md`

Result summary: Remaining legacy runtime blockers were decoupled. Workflow and tests no longer depend on legacy runtime scripts, and the legacy wrapper dependency has been removed by making `scripts/run_full_pipeline.py` fail closed instead of invoking the legacy scan runtime. The legacy runtime scripts remain present but are ready for a new archive-readiness recheck.

Proposed next step: Proceed to `RESET-10L-BL39 — Legacy Runtime Script Archive Readiness Recheck`.

Guardrails:

- no archive, delete, move, or rename until archive-readiness is rechecked and separately approved;
- no live provider calls unless separately approved;
- no credential reads;
- no network calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- canonical delivery, message composition, report artifact planning, Decision Engine behavior, and runtime entrypoint responsibilities must remain separate;
- legacy runner, legacy Decision Engine, legacy reporting, legacy message, and legacy Telegram authority must not be expanded.

### RESET-10L-BL39 — Legacy Runtime Script Archive Readiness Recheck

Category: Architecture / Cleanup Review

Rationale: BL38 removed active workflow/test dependencies and the wrapper invocation from the primary legacy runtime scripts. A controlled recheck is needed before any archive, delete, or move decision.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL39

Review record:

- `docs/active/v2_legacy_runtime_script_archive_readiness_recheck.md`

Result summary: Legacy runtime script archive readiness was rechecked after BL37 and BL38. No active workflow, source import, test import, test monkeypatch, or wrapper dependency remains for `scripts/run_scan.py` or `scripts/run_full_pipeline.py`. `scripts/run_full_pipeline.py` is fail-closed and archive-ready. `scripts/run_scan.py` is archive-ready with manual invocation risk because it still contains side-effectful legacy runtime logic while present. Static governance, metadata, historical documentation, and static assertion references remain and should be handled by the controlled archive sprint.

Proposed next step: Proceed to `RESET-10L-BL40 — Archive Confirmed Legacy Runtime Scripts`.

Guardrails:

- no archive, delete, move, or rename until separately approved in BL40;
- no live provider calls unless separately approved;
- no credential reads;
- no network calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- canonical delivery, message composition, report artifact planning, Decision Engine behavior, and runtime entrypoint responsibilities must remain separate;
- legacy runner, legacy Decision Engine, legacy reporting, legacy message, and legacy Telegram authority must not be expanded.

### RESET-10L-BL40 — Archive Confirmed Legacy Runtime Scripts

Category: Architecture / Cleanup Implementation

Rationale: BL39 found both primary legacy runtime scripts ready for a controlled archive sprint. `scripts/run_full_pipeline.py` is fail-closed and `scripts/run_scan.py` has no remaining active workflow, source import, test import, test monkeypatch, or wrapper dependency, but still carries manual invocation risk until removed from active script paths.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL40

Archive records:

- `archive/legacy_runtime/scripts/run_scan.py`
- `archive/legacy_runtime/scripts/run_full_pipeline.py`

Implementation records:

- `src/market_scanner/app.py`
- `src/market_scanner/scanner/scanner_boundary.py`
- `src/market_scanner/delivery/delivery_boundary.py`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- `tests/unit/test_v2_canonical_scanner.py`
- `tests/unit/test_v2_canonical_analysis.py`
- `tests/unit/test_v2_canonical_decision.py`
- `tests/unit/test_v2_canonical_messaging.py`
- `tests/unit/test_v2_canonical_reporting.py`
- `tests/unit/test_v2_canonical_delivery.py`
- `docs/active/v2_legacy_runtime_script_archive_execution.md`

Result summary: Confirmed legacy runtime scripts were archived. `scripts/run_scan.py` and `scripts/run_full_pipeline.py` were removed from the active `scripts/` runtime path and moved to the legacy runtime archive. The canonical v2 runtime authority remains `src/market_scanner/app.py`. Workflow and tests remain decoupled from legacy runtime scripts, and no provider calls, production data writes, reports, Telegram delivery, portfolio/watchlist updates, or investment recommendation behavior were added.

Proposed next step: Proceed to `RESET-10L-BL41 — Legacy Runtime Archive Validation and Active Entrypoint Certification`.

Guardrails:

- no live provider calls unless separately approved;
- no credential reads;
- no network calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- canonical delivery, message composition, report artifact planning, Decision Engine behavior, and runtime entrypoint responsibilities must remain separate;
- archived legacy runner, legacy Decision Engine, legacy reporting, legacy message, and legacy Telegram authority must not be expanded.

### RESET-10L-BL41 — Legacy Runtime Archive Validation and Active Entrypoint Certification

Category: Architecture / Cleanup Validation

Rationale: BL40 archived the confirmed legacy runtime scripts. A follow-up validation sprint should certify that active runtime entrypoint authority is now only `src/market_scanner/app.py`, no workflow/source/test path depends on archived scripts, and historical references remain evidence-only.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL41

Validation record: `docs/active/v2_legacy_runtime_archive_validation_and_entrypoint_certification.md`

Result summary: Legacy runtime archive validation is complete. The confirmed legacy runtime scripts are absent from the active `scripts/` runtime path, present only in `archive/legacy_runtime/scripts/`, and are not referenced by active workflow/source/test runtime paths. The certified active runtime entrypoint is `src/market_scanner/app.py`.

Proposed next step: Proceed to `RESET-10L-BL42 — Script-Era Python Cleanup Inventory`.

Guardrails:

- no live provider calls unless separately approved;
- no credential reads;
- no network calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- canonical delivery, message composition, report artifact planning, Decision Engine behavior, and runtime entrypoint responsibilities must remain separate;
- legacy runner, legacy Decision Engine, legacy reporting, legacy message, and legacy Telegram authority must not be expanded.

### RESET-10L-BL42 — Script-Era Python Cleanup Inventory

Category: Architecture / Cleanup Review

Rationale: With the primary legacy runtime scripts archived and the canonical app certified as the active runtime entrypoint, the remaining script-era Python surface should be inventoried for migration, compatibility, archive, or deletion planning before further cleanup implementation.

Governance risk: HIGH

Owner role: Technical Analyst / Architecture Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL42

Inventory record: `docs/active/v2_script_era_python_cleanup_inventory.md`

Result summary: Script-era Python cleanup inventory is complete for `scripts/`. There are 52 remaining script-era Python files, 43 runnable script-era entrypoints, broad active test imports, and high-risk side-effect surfaces across scanner, fundamentals, Decision Engine, reporting, Telegram, portfolio, watchlist, data-source, diagnostics, and maintenance utilities. The next safest cleanup step is to remove the remaining archived-script execution pattern from tests before archiving additional script-era files.

Proposed next step: Proceed to `RESET-10L-BL43 — Remove Archived Script Execution from Tests`.

Guardrails:

- review-only unless separately approved;
- no live provider calls unless separately approved;
- no credential reads;
- no network calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- canonical delivery, message composition, report artifact planning, Decision Engine behavior, and runtime entrypoint responsibilities must remain separate;
- archived legacy runner, legacy Decision Engine, legacy reporting, legacy message, and legacy Telegram authority must not be expanded.

### RESET-10L-BL43 — Remove Archived Script Execution from Tests

Category: Architecture / Cleanup Testing

Rationale: BL41 and BL42 confirmed that `tests/core/test_run_full_pipeline.py` still executes `archive/legacy_runtime/scripts/run_full_pipeline.py` to validate fail-closed behavior. Archived scripts should remain historical references, not executable test targets, before additional script-era archive/delete work proceeds.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Governance Auditor

Status: COMPLETED BY RESET-10L-BL43

Cleanup record: `docs/active/v2_archived_script_execution_test_cleanup.md`

Result summary: Archived script execution has been removed from active tests. Tests now validate archived legacy runtime scripts through static archive-status checks and continue to validate the canonical app dry-run as the active runtime path. Archived scripts remain historical references only and are not imported, monkeypatched, or executed by active tests.

Proposed next step: Proceed to `RESET-10L-BL44 — High-Risk Script-Era Side-Effect Cleanup Review`.

Guardrails:

- no live provider calls unless separately approved;
- no credential reads;
- no network calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- archived scripts must not be executed;
- archived legacy runner, legacy Decision Engine, legacy reporting, legacy message, and legacy Telegram authority must not be expanded.

### RESET-10L-BL44 — High-Risk Script-Era Side-Effect Cleanup Review

Category: Architecture / Side-Effect Cleanup Review

Rationale: BL42 inventoried 52 remaining script-era Python files under `scripts/`, including many runnable entrypoints and high-risk side-effect zones. BL43 removed archived-script execution from active tests, so the next cleanup step is to classify the highest-risk active script-era side-effect files before migration, archive, or deletion work begins.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Governance Auditor

Status: COMPLETED BY RESET-10L-BL44

Review record: `docs/active/v2_high_risk_script_era_side_effect_cleanup_review.md`

Result summary: High-risk script-era side-effect review is complete. The review confirmed 52 remaining script-era Python files under `scripts/`, 43 runnable entrypoints, no script-era subprocess/shell execution patterns, and high-risk side-effect groups involving provider access, SEC download support, data writes, report writes, Telegram delivery, credential reads, portfolio/watchlist mutation, and Decision Engine final/allocation semantics. Active tests still import and exercise many high-risk script-era modules, so test execution should be decoupled before migration or archive work.

Proposed next step: Proceed to `RESET-10L-BL45 — Remove High-Risk Script-Era Test Execution`.

Guardrails:

- no live provider calls unless separately approved;
- no credential reads;
- no network calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- script-era files must not be executed;
- archived scripts must not be executed;
- no script-era migration, archive, delete, or refactor is authorized by this review.

### RESET-10L-BL45 — Remove High-Risk Script-Era Test Execution

Category: Architecture / Cleanup Testing

Rationale: BL44 confirmed that active tests still import and exercise many high-risk script-era modules. These tests make script-era modules harder to migrate, archive, or delete safely. Active coverage should move toward canonical v2 boundaries, static policy checks, and documented migration blockers.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Governance Auditor

Status: COMPLETED BY RESET-10L-BL45

Cleanup record: `docs/active/v2_high_risk_script_era_test_execution_cleanup.md`

Implementation records:

- `tests/conftest.py`
- `tests/test_operator_visibility.py`

Result summary: High-risk script-era test execution has been removed from active pytest collection. Active tests no longer import or execute the highest-risk script-era behavior suites and instead rely on canonical boundary coverage, static policy checks, and documented blockers. No script-era production files were changed.

Proposed next step: Proceed to `RESET-10L-BL46 — Fundamentals Script-Era Side-Effect Migration Review`.

Guardrails:

- no live provider calls unless separately approved;
- no credential reads;
- no network calls;
- no production data writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- script-era files must not be executed;
- archived scripts must not be executed;
- no script-era production files, archived scripts, workflows, data files, report files, portfolio/watchlist files, or runtime behavior were changed.

### RESET-10L-BL46 — Fundamentals Script-Era Side-Effect Migration Review

Category: Architecture / Fundamentals Cleanup Review

Rationale: BL44 identified fundamentals/provider/source-data scripts as a high-risk side-effect domain with SEC/EDGAR download support, yfinance/provider access, data writes, production path pressure, and old fundamentals builders. Before migration or archive work begins, the useful logic and canonical parity gaps must be mapped.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Governance Auditor

Status: COMPLETED BY RESET-10L-BL46

Review record: `docs/active/v2_fundamentals_script_era_side_effect_migration_review.md`

Result summary: Fundamentals script-era side-effect migration review is complete. The review classified 22 fundamentals/provider/source-data related script-era files, confirmed that canonical v2 already covers injected provider responses, provenance, explicit missingness, governed derived FreeCashFlow, prior-year growth evidence, readiness, and tmp-path persistence, and identified remaining gaps around live provider/SEC governance, SEC Company Facts bulk intake/cache ownership, SEC fact transformation parity, ticker/CIK mapping, production path policy, and old quality/metrics/profile outputs.

Proposed next step: Proceed to `RESET-10L-BL47 — Govern Canonical Fundamentals Live Provider Boundary`.

Guardrails:

- review-only unless separately approved;
- no live provider calls unless separately approved;
- no SEC/EDGAR calls;
- no yfinance calls;
- no credential reads;
- no network calls;
- no production data writes;
- no raw payload writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- no Python files, tests, workflows, script-era files, archived scripts, data files, reports, or production artifacts were changed.

### RESET-10L-BL47 — Govern Canonical Fundamentals Live Provider Boundary

Category: Source Data / Fundamentals Governance

Rationale: BL46 confirmed that canonical fundamentals already owns injected provider-shaped responses, provenance, explicit missingness, governed derived FreeCashFlow, prior-year growth evidence, readiness, and tmp-path persistence, but live SEC/EDGAR, cache, raw-payload, production path, ticker-to-CIK, and fact-selection governance must be defined before any live-provider implementation begins.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL47

Governance record: `docs/active/v2_canonical_fundamentals_live_provider_boundary_policy.md`

Result summary: The canonical fundamentals live-provider boundary is now governed. BL47 approves policy only, not implementation. The first future implementation candidate is a one-ticker SEC CompanyFacts canonical live-provider smoke boundary, with strict single-ticker, explicit-local-invocation, no-production-write, no-raw-payload-commit, provenance, missingness, and fail-closed guardrails.

Proposed next step: Proceed to `RESET-10L-BL48 — Implement Canonical Fundamentals SEC CompanyFacts Smoke Boundary`.

Guardrails:

- governance-only unless separately approved;
- no implementation of live provider execution;
- no SEC/EDGAR calls;
- no yfinance calls;
- no provider calls;
- no credential reads;
- no network calls;
- no production data writes;
- no raw payload writes;
- no cache writes;
- no production pipeline execution;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- no Python files, tests, workflows, script-era files, archived scripts, data files, reports, or production artifacts were changed.

### RESET-10L-BL48 — Implement Canonical Fundamentals SEC CompanyFacts Smoke Boundary

Category: Source Data / Fundamentals Implementation

Rationale: BL47 governed the first future canonical fundamentals live-provider boundary as a one-ticker SEC CompanyFacts smoke boundary. The implementation must remain injected-input-only, side-effect-free by default, and compliant with the no-live-call, no-production-write, no-cache, no-raw-payload, and no-investment-semantics guardrails.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL48

Implementation record: `docs/active/v2_canonical_sec_companyfacts_smoke_boundary_implementation.md`

Implementation records:

- `src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py`
- `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`

Result summary: The canonical fundamentals SEC CompanyFacts smoke boundary is implemented for injected one-ticker SEC-shaped input only. The boundary preserves ticker/CIK/source provenance, performs deterministic fact selection, supports governed source-derived FreeCashFlow when inputs are valid, fails closed on ambiguity/missing provenance/context mismatches, and does not perform live SEC/EDGAR calls, network calls, production writes, raw payload/cache commits, reports, Telegram delivery, workflow execution, scanner-triggered execution, or multi-ticker capture.

Proposed next step: Proceed to `RESET-10L-BL49 — Validate NVDA SEC CompanyFacts Smoke Boundary Against Redacted Source-Shaped Evidence`.

Guardrails:

- no live SEC/EDGAR calls;
- no yfinance calls;
- no provider network calls;
- no credential reads;
- no production data writes;
- no raw payload writes;
- no cache writes;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no workflow execution;
- no scanner-triggered execution;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- no missing values converted to zero;
- no script-era files imported or executed;
- no archived scripts executed;
- no script-era files, archived scripts, workflows, data files, reports, portfolio/watchlist files, raw payload files, cache files, or production artifacts were changed.

### RESET-10L-BL49 — Validate NVDA SEC CompanyFacts Smoke Boundary Against Redacted Source-Shaped Evidence

Category: Source Data / Fundamentals Validation

Rationale: BL48 implemented the canonical SEC CompanyFacts smoke boundary using minimal synthetic input. Before any controlled live-source governance can be considered, the boundary must be validated against a more realistic but still redacted and injected NVDA SEC CompanyFacts-shaped fixture.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL49

Validation record: `docs/active/v2_nvda_sec_companyfacts_smoke_boundary_validation.md`

Validation records:

- `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`

Result summary: The canonical SEC CompanyFacts smoke boundary was validated against injected redacted/source-shaped NVDA evidence. Validation confirmed deterministic SEC-like fact selection, ticker/CIK/source provenance, source-derived FreeCashFlow, comparable prior-year growth evidence, explicit missingness, and fail-closed behavior without live SEC/EDGAR calls, provider calls, network access, raw payload commits, cache writes, production persistence, reports, Telegram, workflow execution, scanner-triggered execution, portfolio/watchlist integration, or recommendation behavior.

Proposed next step: Proceed to `RESET-10L-BL50 — Govern Controlled Live SEC CompanyFacts One-Ticker Smoke`.

Guardrails:

- no live SEC/EDGAR calls;
- no yfinance calls;
- no provider calls;
- no network calls;
- no credential reads;
- no environment variable reads;
- no production data writes;
- no raw payload writes;
- no raw SEC payload commits;
- no cache writes;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no workflow execution;
- no scanner-triggered execution;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- no missing values converted to zero;
- no script-era files imported or executed;
- no archived scripts executed;
- no script-era files, archived scripts, workflows, data files, reports, portfolio/watchlist files, raw payload files, cache files, or production artifacts were changed.

### RESET-10L-BL50 — Govern Controlled Live SEC CompanyFacts One-Ticker Smoke

Category: Source Data / Fundamentals Governance

Rationale: BL49 validated the canonical SEC CompanyFacts smoke boundary against injected redacted/source-shaped NVDA evidence. Before the first live SEC CompanyFacts smoke can be executed, the repository must define exact pre-flight rules for SEC User-Agent handling, explicit local invocation, single-ticker scope, network containment, raw payload handling, cache handling, tmp/local-only output, redacted documentation, fail-closed behavior, and forbidden production paths.

Governance risk: HIGH

Owner role: Data Steward / Technical Analyst / Governance Auditor

Status: COMPLETED BY RESET-10L-BL50

Governance record: `docs/active/v2_controlled_live_sec_companyfacts_one_ticker_smoke_policy.md`

Result summary: The controlled live SEC CompanyFacts one-ticker smoke is now governed. BL50 approves policy only, not implementation or execution. The only approved first live-smoke target is NVDA / CIK 0001045810. A future BL51 may execute one explicit local SEC CompanyFacts smoke only under strict no-production-write, no-raw-payload-commit, no-cache-commit, no-workflow, no-scanner-trigger, no-portfolio/watchlist, no-report, no-Telegram, provenance, missingness, and fail-closed guardrails.

Proposed next step: Proceed to `RESET-10L-BL51 — Execute Controlled Live SEC CompanyFacts One-Ticker Smoke`.

Guardrails:

- governance-only;
- no implementation of a live SEC provider client;
- no live SEC/EDGAR calls;
- no yfinance calls;
- no provider calls;
- no network calls;
- no credential reads;
- no environment variable reads;
- no production data writes;
- no raw payload writes;
- no cache writes;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no workflow execution;
- no scanner-triggered execution;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- no missing values converted to zero;
- no Python files, tests, workflows, script-era files, archived scripts, data files, reports, raw payload files, cache files, portfolio/watchlist files, or production artifacts were changed.

### RESET-10L-BL51 — Execute Controlled Live SEC CompanyFacts One-Ticker Smoke

Category: Source Data / Fundamentals Validation

Rationale: BL50 governed the exact pre-flight conditions for the first controlled live SEC CompanyFacts one-ticker smoke. BL51 must execute the smoke only if those pre-flight guardrails are satisfied, or fail closed and document the blocker.

Governance risk: HIGH

Owner role: Technical Analyst / Developer / Data Steward / Governance Auditor

Status: COMPLETED BY RESET-10L-BL51

Result record: `docs/active/v2_live_nvda_sec_companyfacts_one_ticker_smoke_result.md`

Implementation records:

- `src/market_scanner/fundamentals/sec_companyfacts_live_smoke.py`
- `tests/unit/test_v2_sec_companyfacts_live_smoke.py`

Result summary: The first controlled live SEC CompanyFacts one-ticker smoke was attempted for NVDA / CIK 0001045810 and failed closed under BL50 guardrails. The failure category, readiness/missingness result, and next remediation step are documented. No retry loop, fallback provider, raw payload, cache, production data, reports, Telegram artifacts, workflow integration, scanner integration, portfolio/watchlist update, or recommendation behavior was committed.

Proposed next step: Proceed to `RESET-10L-BL52 — Resolve Live SEC CompanyFacts Smoke Failure`.

Guardrails:

- one-ticker scope only: `NVDA`;
- one CIK scope only: `0001045810`;
- no live SEC request was executed because the local `SEC_USER_AGENT` pre-flight was missing;
- no more than one SEC CompanyFacts request performed;
- no endpoint other than NVDA / CIK0001045810 CompanyFacts used;
- no yfinance calls;
- no fallback provider calls;
- no retry loop;
- no production data writes;
- no raw payload writes or commits;
- no cache writes or commits;
- no report generation;
- no Telegram artifacts or Telegram delivery;
- no workflow execution;
- no scanner-triggered execution;
- no portfolio or watchlist updates;
- no final BUY, SELL, or HOLD recommendation;
- no allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior;
- no missing values converted to zero;
- no script-era files imported or executed;
- no archived scripts executed;
- no script-era files, archived scripts, workflows, data files, reports, portfolio/watchlist files, raw payload files, cache files, credential files, or production artifacts were changed.

## Relationship to Existing Backlog

The historical `docs/sprints/project_backlog.md` remains preserved as legacy planning evidence until RESET-2 decides how to reconcile or archive it. This document is the v2 reset-facing backlog baseline.

## BL69 — Document SEC CompanyFacts capex alias validation and one-ticker live source-readiness result

Status: Done

Type: Documentation / Governance

Summary:

Documented the completed SEC CompanyFacts capex alias validation after BL68B and the controlled BL68C live retry.

The documentation records that `PaymentsToAcquireProductiveAssets` is now accepted as a safe SEC CompanyFacts alias for canonical `capital_expenditures`, while acquisition and intangible concepts remain excluded.

Controlled BL68C live retry result for NVDA / CIK `0001045810`:

* request executed: true
* request count: 1
* HTTP status category: 2xx
* status: passed
* readiness state: available
* canonical fields missing: none
* `capital_expenditures:missing_fact`: resolved
* free cash flow: source-derived
* growth evidence: available
* git status after retry: clean

Governance conclusion:

The SEC CompanyFacts provider now supports the productive-assets capex alias generally, but live source-readiness is only validated for the approved one-ticker NVDA smoke. Broader ticker validation remains out of scope.

Files changed:

* `docs/audits/provider_smokes/bl69_sec_companyfacts_capex_alias_validation.md`
* `docs/active/project/backlog.md`

No code, tests, data files, reports, Telegram output, portfolio/watchlist logic, or Decision Engine behavior were changed.

## BL71 — Archive first low-risk script-era Python utility

Status: Done

Type: Python Cleanup / Legacy Runtime Archive

Summary:

Archived the first narrow low-risk script-era Python cleanup batch from the BL70 canonical cleanup registry.

Selected file:

- `scripts/utils/utils.py`

Archive destination:

- `archive/legacy_runtime/scripts/utils/utils.py`

Registry basis:

The existing cleanup registry classified `scripts/utils/utils.py` as a low-priority archive candidate with no active reference found, no runnable entrypoint, and generic utility/write-helper behavior. BL71 archived the file instead of deleting it.

Validation:

- focused delta reference check: no active references found in `src/`, `tests/`, or `.github/`
- `pytest -q`: 501 passed

Guardrails:

- no broad Python inventory was repeated;
- no runtime behavior was changed;
- no provider behavior was changed;
- no SEC CompanyFacts behavior was changed;
- no production data writes were added;
- no reports were generated;
- no Telegram behavior was changed;
- no portfolio/watchlist logic was changed;
- no Decision Engine behavior was changed.

Files changed:

- `scripts/utils/utils.py` moved to `archive/legacy_runtime/scripts/utils/utils.py`
- `docs/audits/legacy_runtime/bl71_archive_low_risk_script_era_utils.md`
- `docs/active/project/backlog.md`

## BL72 — Archive legacy reporting markdown reporter

Status: Done

Type: Python Cleanup / Legacy Runtime Archive

Summary:

Archived one narrow P3 script-era reporting cleanup candidate from the BL70 canonical cleanup registry.

Selected file:

- `scripts/reporting/reporter.py`

Archive destination:

- `archive/legacy_runtime/scripts/reporting/reporter.py`

Registry basis:

The existing cleanup registry classified `scripts/reporting/reporter.py` as `ARCHIVE_CANDIDATE_NOW`, with no active reference found, no runnable entrypoint, low direct risk, and old legacy markdown report formatting responsibility. BL72 archived the file instead of deleting it.

Validation:

- focused delta reference check required before merge;
- `pytest -q` required before merge.

Guardrails:

- no broad Python inventory was repeated;
- no runtime behavior was changed;
- no provider behavior was changed;
- no SEC CompanyFacts behavior was changed;
- no production data writes were added;
- no reports were generated;
- no Telegram behavior was changed;
- no portfolio/watchlist logic was changed;
- no Decision Engine behavior was changed.

Files changed:

- `scripts/reporting/reporter.py` moved to `archive/legacy_runtime/scripts/reporting/reporter.py`
- `docs/audits/legacy_runtime/bl72_archive_legacy_reporting_reporter.md`
- `docs/active/project/backlog.md`

## BL73 — Archive remaining low-risk script-era cleanup candidates

Status: Done

Type: Python Cleanup / Legacy Runtime Archive

Summary:

Archived the next small low-risk script-era Python cleanup batch from the BL70 canonical cleanup registry.

Selected files:

- `scripts/core/regime.py`
- `scripts/portfolio/test_portfolio.py`

Archive destinations:

- `archive/legacy_runtime/scripts/core/regime.py`
- `archive/legacy_runtime/scripts/portfolio/test_portfolio.py`

Registry basis:

The existing cleanup registry classified `scripts/core/regime.py` as a P3 archive candidate with no active test import found, no runnable entrypoint, and low direct write risk.

The existing cleanup registry classified `scripts/portfolio/test_portfolio.py` as a P4 delete/archive candidate after confirmation that it is not part of the active test suite or operator procedure. BL73 archived the file instead of deleting it.

Blocked candidate:

- `scripts/fundamentals/__init__.py` remains blocked because active tests still import `scripts.fundamentals`.

Validation:

- focused delta reference checks required before merge;
- `pytest -q` required before merge.

Guardrails:

- no broad Python inventory was repeated;
- no runtime behavior was changed;
- no scanner behavior was changed;
- no provider behavior was changed;
- no SEC CompanyFacts behavior was changed;
- no production data writes were added;
- no reports were generated;
- no Telegram behavior was changed;
- no portfolio/watchlist runtime logic was changed;
- no Decision Engine behavior was changed.

Files changed:

- `scripts/core/regime.py` moved to `archive/legacy_runtime/scripts/core/regime.py`
- `scripts/portfolio/test_portfolio.py` moved to `archive/legacy_runtime/scripts/portfolio/test_portfolio.py`
- `docs/audits/legacy_runtime/bl73_archive_remaining_low_risk_script_era_candidates.md`
- `docs/active/project/backlog.md`

## BL74 — Decouple active tests from script-era fundamentals imports

Status: Done

Type: Python Cleanup / Active Test Decoupling

Summary:

Active tests were decoupled from script-era fundamentals imports. The former `scripts.fundamentals` behavior tests were converted into active static legacy-policy/evidence tests where they only preserved governance expectations and no longer execute old script-era fundamentals modules.

Archive pytest collection behavior was addressed by excluding `archive/` from pytest recursion in `pyproject.toml`. Future cleanup can retry archiving `scripts/fundamentals/__init__.py` and `scripts/portfolio/test_portfolio.py` if focused validation passes.

Files changed:

- `pyproject.toml`
- `tests/conftest.py`
- `tests/test_operator_visibility.py`
- `tests/core/test_build_fundamental_analysis.py`
- `tests/core/test_build_fundamental_layer.py`
- `tests/core/test_build_fundamental_metrics.py`
- `tests/core/test_build_fundamentals_history_intake.py`
- `tests/core/test_fundamentals_operational_validation.py`
- `tests/core/test_fundamentals_runtime_organization.py`
- `tests/fundamentals/test_run_sec_transformation_review.py`
- `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`
- `tests/fundamentals/test_sec_companyfacts_transform.py`
- `tests/fundamentals/test_sec_ticker_cik_index.py`
- `docs/audits/legacy_runtime/bl74_decouple_active_tests_from_script_era_fundamentals.md`
- `docs/active/project/backlog.md`

Validation:

- active `scripts.fundamentals` import grep across `tests`, `src`, and `.github`: no output
- `source .venv/bin/activate && pytest -q`: `522 passed in 0.78s`

### BL75 — Retry archiving BL74-unblocked script-era Python files

Category: Legacy Runtime Cleanup / Repository Hygiene

Status: COMPLETED

BL75 retried the two script-era Python cleanup candidates that were unblocked by BL74.

Archived or removed from active runtime paths:

- `scripts/fundamentals/__init__.py`
- `scripts/portfolio/test_portfolio.py`

Archive targets:

- `archive/legacy_runtime/scripts/fundamentals/__init__.py`
- `archive/legacy_runtime/scripts/portfolio/test_portfolio.py`

Result: active runtime/test paths are further decoupled from legacy `scripts/` files. BL74’s archive pytest exclusion allows archived `test_*.py` files to remain under `archive/` without active pytest collection.

Validation:

- `pytest -q`: pending local validation before merge.

### BL76 — Static dependency classification of remaining script-era Python files

Category: Legacy Runtime Cleanup / Repository Hygiene

Status: COMPLETED

BL76 classified the remaining script-era Python files under `scripts/` after BL74 and BL75.

No runtime Python files were archived, deleted, moved, refactored, or executed. The sprint only added a static classification document for the remaining script-era tree.

Classification record:

- `docs/audits/legacy_runtime/bl76_remaining_script_era_python_dependency_classification.md`

Result:

- Initial BL77 archive candidates identified:
  - `scripts/core/analyze_validation.py`
  - `scripts/diagnostics/audit_data_coverage.py`
  - `scripts/analyze_validation.py`
- High-risk categories were separated:
  - scanner/runtime entrypoints
  - provider/data-fetch/intake scripts
  - prefill/backfill scripts
  - portfolio/watchlist state writers
  - Telegram delivery and command handlers
  - Decision Engine authority files
  - fundamentals logic that may require canonical migration

Guardrails:

- no live SEC/EDGAR calls
- no yfinance calls
- no credentials read
- no production data writes
- no reports generated
- no Telegram messages sent
- no portfolio/watchlist state modified
- no Decision Engine authority changed
- no script-era Python runtime files executed
### BL77 — Archive BL76 low-risk script-era Python files

Category: Legacy Runtime Cleanup / Repository Hygiene

Status: COMPLETED

BL77 archived the low-risk script-era Python files identified by BL76.

Archived:

- `scripts/core/analyze_validation.py`
- `scripts/diagnostics/audit_data_coverage.py`
- `scripts/analyze_validation.py`

Archive targets:

- `archive/legacy_runtime/scripts/core/analyze_validation.py`
- `archive/legacy_runtime/scripts/diagnostics/audit_data_coverage.py`
- `archive/legacy_runtime/scripts/analyze_validation.py`

Result:

- The three low-risk script-era validation/audit helpers were removed from active runtime paths.
- Historical evidence was preserved under `archive/legacy_runtime/`.
- The active `scripts/` tree is further reduced without touching provider, portfolio, Telegram, watchlist, reporting, or Decision Engine runtime behavior.

Validation:

- `pytest -q`: `522 passed in 0.59s`

Guardrails:

- no live SEC/EDGAR calls
- no yfinance calls
- no credentials read
- no production data writes
- no reports generated
- no Telegram messages sent
- no portfolio/watchlist state modified
- no Decision Engine authority changed
- no script-era Python runtime files executed

### BL78 — Fundamentals script-era migration and archive-readiness review

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL78 reviewed the remaining script-era fundamentals Python files after BL74 removed active runtime imports from `scripts.fundamentals`.

No runtime Python files were archived, deleted, moved, refactored, or executed. The sprint only classified the remaining fundamentals scripts for future cleanup.

Review record:

- `docs/audits/legacy_runtime/bl78_fundamentals_script_era_migration_review.md`

Result:

- No remaining `scripts/fundamentals/*.py` file is declared archive-ready by BL78.
- Remaining fundamentals scripts still contain migration-required or side-effect-risk logic.
- `scripts/core/build_fundamental_*` files were identified as compatibility wrappers over `scripts.fundamentals`.
- Canonical modules exist under `src/market_scanner/analysis/` and `src/market_scanner/fundamentals/`, but complete behavioral parity is not proven by BL78.
- The recommended next sprint is to review/archive the `scripts/core/build_fundamental_*` compatibility wrappers only after a focused active-reference check.

Recommended next sprint:

- BL79 — Archive fundamentals compatibility wrappers after active-reference check

Validation:

- `pytest -q`: `522 passed in 0.59s`

Guardrails:

- no live SEC/EDGAR calls
- no yfinance calls
- no credentials read
- no production data writes
- no reports generated
- no Telegram messages sent
- no portfolio/watchlist state modified
- no Decision Engine authority changed
- no script-era Python runtime files executed
### BL80 — Remaining fundamentals migration blocker review

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL80 reviewed the remaining `scripts/fundamentals/*.py` files after BL79 archived the `scripts/core/build_fundamental_*` compatibility wrappers.

No runtime Python files were archived, deleted, moved, refactored, or executed. This sprint only classified the remaining blockers that prevent safe archive of the fundamentals modules.

Review record:

- `docs/audits/legacy_runtime/bl80_remaining_fundamentals_migration_blocker_review.md`

Result:

- No remaining `scripts/fundamentals/*.py` file is archive-ready as of BL80.
- Eight fundamentals files remain under `scripts/fundamentals/`.
- No active test, source, or workflow runtime imports were found.
- The remaining imports are internal dependencies inside the legacy fundamentals group.
- The files still contain migration knowledge, internal dependency coupling, provider/data-write risk, or review-runner behavior.
- The remaining fundamentals cleanup is split into four governed lanes:
  - pure contract extraction;
  - analysis and quality parity;
  - SEC transform and identifier mapping;
  - provider and review-runner retirement.

Recommended next sprint:

- BL81 — Extract fundamentals history and metrics contracts from script-era modules

Validation:

- `pytest -q`: `522 passed in 0.58s`

Guardrails:

- no live SEC/EDGAR calls
- no yfinance calls
- no credentials read
- no production data writes
- no reports generated
- no Telegram messages sent
- no portfolio/watchlist state modified
- no Decision Engine authority changed
- no script-era Python runtime files executed

### BL81 — Extract fundamentals history and metrics contracts from script-era modules

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL81 extracted the stable history-schema and metrics-calculation contracts from the remaining script-era fundamentals modules:

- `scripts/fundamentals/build_history_intake.py`
- `scripts/fundamentals/build_metrics.py`

No runtime Python files were archived, deleted, moved, refactored, or executed. This sprint was documentation-only.

Review record:

- `docs/audits/legacy_runtime/bl81_fundamentals_history_and_metrics_contract_extraction.md`

Result:

- Extracted the fundamentals history required-column contract.
- Extracted the key/duplicate policy.
- Extracted fiscal-year, fiscal-period, date, numeric, and required-value validation policies.
- Extracted forbidden investment-semantics column policy.
- Extracted metrics identity columns, metric columns, helper columns, ratio formulas, YoY formulas, missing-input policy, zero-denominator policy, and metric-status policy.
- Confirmed that script-era CLI/write behavior is not automatically approved for canonical migration.
- Confirmed that `build_history_intake.py` and `build_metrics.py` are not archive-ready until canonical parity or explicit retirement is completed.

Recommended next sprint:

- BL82 — Implement canonical fundamentals history and metrics contract tests

Validation:

- `pytest -q`: `522 passed in 0.57s`

Guardrails:

- no live SEC/EDGAR calls
- no yfinance calls
- no credentials read
- no production data writes
- no reports generated
- no Telegram messages sent
- no portfolio/watchlist state modified
- no Decision Engine authority changed
- no script-era Python runtime files executed

### BL83 — Add canonical fundamentals history and metrics contract tests

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL83 added canonical fundamentals metrics contract coverage based on BL81 and BL82.

Added canonical module:

- `src/market_scanner/fundamentals/fundamentals_metrics_contracts.py`

Added contract tests:

- `tests/contract/test_v2_fundamentals_metrics_contracts.py`

Result:

- Added explicit canonical ownership for BL81 metrics identity fields, input fields, derived metric fields, and helper fields.
- Added canonical tests for ratio formulas:
  - `gross_margin`
  - `operating_margin`
  - `net_margin`
  - `free_cash_flow_margin`
  - `debt_to_equity`
  - `return_on_equity`
- Added canonical tests for YoY formulas:
  - `revenue_yoy_growth`
  - `eps_yoy_growth`
  - `free_cash_flow_yoy_growth`
- Added canonical tests for missing-input behavior.
- Added canonical tests for zero-denominator behavior.
- Added canonical tests for missing-prior-year behavior.
- Added canonical tests for absolute prior-year denominator policy.
- Added no-investment-authority guardrails.
- Added no-legacy/network/provider import guardrails.
- Added no-file-side-effect guardrails.

Validation:

- `pytest tests/contract/test_v2_fundamentals_metrics_contracts.py -q`: `12 passed in 0.02s`
- focused related tests: `54 passed in 0.03s`
- `pytest -q`: `534 passed in 0.56s`

Archive readiness:

- `scripts/fundamentals/build_history_intake.py`: `NOT_ARCHIVE_READY`
- `scripts/fundamentals/build_metrics.py`: `CANDIDATE_FOR_REVIEWED_ARCHIVE_AFTER_PARITY_CHECK`

Recommended next sprint:

- BL84 — Add canonical fundamentals history validation contract tests

Guardrails:

- no live SEC/EDGAR calls
- no yfinance calls
- no credentials read
- no production data writes
- no reports generated
- no Telegram messages sent
- no portfolio/watchlist state modified
- no Decision Engine authority changed
- no script-era Python runtime files executed


### BL84 — Add canonical fundamentals history validation contract tests

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL84 added canonical fundamentals history validation contract coverage based on BL81 and BL82.

Updated canonical module:

* `src/market_scanner/fundamentals/fundamental_contracts.py`

Added contract tests:

* `tests/contract/test_v2_fundamental_history_validation_contracts.py`

Result:

* Added canonical ownership for fundamentals history numeric fields.
* Added canonical ownership for fundamentals history date fields.
* Added supported fiscal-period policy.
* Added fiscal-period validation.
* Added fiscal-year validation with accepted range `1900 <= fiscal_year <= 2200`.
* Added date validation for non-empty date values.
* Added duplicate-key validation for `ticker + fiscal_year + fiscal_period`.
* Preserved required-value validation.
* Preserved numeric validation.
* Preserved forbidden investment-authority guardrails.
* Added no-legacy/network/provider import guardrails.
* Added no-file-side-effect guardrails.

Validation:

* `pytest tests/contract/test_v2_fundamental_history_validation_contracts.py -q`: `13 passed in 0.02s`
* focused related tests: `27 passed in 0.02s`
* `pytest -q`: `547 passed in 0.59s`

Archive readiness:

* `scripts/fundamentals/build_history_intake.py`: `CANDIDATE_FOR_REVIEWED_ARCHIVE_AFTER_PARITY_CHECK`
* `scripts/fundamentals/build_metrics.py`: `CANDIDATE_FOR_REVIEWED_ARCHIVE_AFTER_PARITY_CHECK`

Recommended next sprint:

* BL85 — Review archive-readiness of script-era fundamentals history and metrics modules

Guardrails:

* no live SEC/EDGAR calls
* no yfinance calls
* no credentials read
* no production data writes
* no reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed


### BL109 — Decouple historical backfill tests from script-era modules

Category: Legacy Runtime Cleanup / Test Decoupling

Status: COMPLETED

BL109 decoupled historical backfill tests from script-era modules.

Updated tests:

* `tests/core/test_build_entry_quality_backfill.py`
* `tests/core/test_build_context_backfill.py`

Test-harness updates:

* `tests/conftest.py`
* `tests/test_operator_visibility.py`

Targeted script-era modules:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

Result:

* the target tests no longer import or execute the script-era backfill modules;
* the target tests now use test-local contract helpers and synthetic fixtures;
* the target tests were removed from the high-risk script-era blocker registry and are active in the full suite;
* no archive move was performed;
* no script-era module was modified or deleted;
* no provider calls, production data writes, reports, Telegram delivery, portfolio state changes, watchlist state changes, scan validation runtime changes, Decision Engine changes, or portfolio intelligence changes were performed.

Validation:

* focused suite: `17 passed in 0.53s`
* full suite: `628 passed in 1.13s`

Full-suite count note:

* the total increased from the prior `610 passed` baseline because the two previously ignored historical backfill tests are active again and operator visibility now includes one additional decoupling guard.

Decision:

* `BL110_ARCHIVE_READINESS_REVIEW_APPROVED_FOR_DECOUPLED_HISTORICAL_BACKFILL_MODULES`

Recommended next sprint:

* BL110 — Archive-readiness review for decoupled historical backfill modules

BL110 goal:

* review archive-readiness for `scripts/core/build_entry_quality_backfill.py` and `scripts/core/build_context_backfill.py`;
* do not archive the modules unless a separate review proves readiness;
* preserve historical source content;
* keep the sprint review-only unless archive readiness is explicitly established.

High-risk areas still out of scope:

* Decision Engine
* portfolio intelligence
* portfolio source contract
* trade command parser
* scanner/provider runtime
* SEC/EDGAR
* yfinance
* Telegram
* production data writes
* portfolio state
* watchlist state
* scan validation runtime
* no script-era Python runtime files executed


### BL85 — Review archive-readiness of script-era fundamentals history and metrics modules

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL85 reviewed archive-readiness for:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

Result:

* Both files still exist in `scripts/fundamentals/`.
* Canonical coverage from BL83 and BL84 is present.
* The files are still not ready for archive.
* Remaining blockers are active references and internal script-era dependencies, not missing canonical contract coverage.

Static active-reference review found references in:

* `src/market_scanner/analysis/analysis_boundary.py`
* `tests/unit/test_v2_canonical_analysis.py`
* `tests/core/test_fundamentals_runtime_organization.py`
* `tests/core/test_fundamentals_operational_validation.py`
* `tests/core/test_build_fundamental_metrics.py`
* `tests/core/test_build_fundamentals_history_intake.py`

Internal script-era dependency review found imports from:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/build_metrics.py`

Side-effect review confirmed:

* `build_history_intake.py` has CSV read, CLI entrypoint, and optional report-write behavior.
* `build_metrics.py` has CSV read, CLI entrypoint, optional directory creation, and optional CSV-write behavior.

BL85 also corrected a BL84 focused-test-order instability:

* removed unnecessary `reload(fundamental_contracts)` from `tests/contract/test_v2_fundamental_history_validation_contracts.py`;
* this prevented dataclass/enum identity mismatch in focused combined test runs.

Validation:

* focused related tests: `39 passed in 0.03s`
* full suite: `547 passed in 0.52s`

Archive decision:

* `scripts/fundamentals/build_history_intake.py`: `NOT_READY_FOR_ARCHIVE_YET`
* `scripts/fundamentals/build_metrics.py`: `NOT_READY_FOR_ARCHIVE_YET`

Recommended next sprint:

* BL86 — Decouple active tests and metadata references from script-era fundamentals history and metrics files

Guardrails:

* no live SEC/EDGAR calls
* no yfinance calls
* no credentials read
* no production data writes
* no reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime files executed
* no script-era runtime files edited
* no script-era runtime files archived
* no script-era runtime files deleted


### BL87 — Review internal script-era dependencies on fundamentals history and metrics modules

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL87 reviewed internal `scripts/fundamentals/` dependencies that still block archiving:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

Result:

* `build_history_intake.py` and `build_metrics.py` are no longer blocked by missing canonical contract coverage.
* `build_history_intake.py` and `build_metrics.py` are no longer blocked by active positive references from `src`, `tests`, or `.github`.
* They remain blocked by internal script-era dependency clustering.

Remaining script-era fundamentals files:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`
* `scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

Internal dependency findings:

* `build_analysis.py` imports from `build_metrics.py`
* `build_quality.py` imports from `build_metrics.py`
* `build_quality.py` imports from `build_history_intake.py`
* `build_metrics.py` imports from `build_history_intake.py`
* `run_sec_transformation_review.py` imports from `build_history_intake.py`
* `run_sec_transformation_review.py` imports from `sec_companyfacts_transform.py`
* `run_sec_transformation_review.py` imports from `sec_ticker_cik_index.py`
* `sec_companyfacts_transform.py` imports from `build_history_intake.py`
* `sec_companyfacts_transform.py` imports from `sec_ticker_cik_index.py`

Archive decision after BL87:

* `scripts/fundamentals/build_history_intake.py`: `CLUSTER_DEPENDENCY_BLOCKED`
* `scripts/fundamentals/build_metrics.py`: `CLUSTER_DEPENDENCY_BLOCKED`
* `scripts/fundamentals/build_analysis.py`: `ACTIVE_REFERENCE_AND_OPTIONAL_WRITE_RISK`
* `scripts/fundamentals/build_quality.py`: `HIGH_RISK_PRODUCTION_WRITE_BLOCKER`
* `scripts/fundamentals/run_sec_transformation_review.py`: `SEC_REVIEW_RUNNER_BLOCKER`
* `scripts/fundamentals/sec_companyfacts_transform.py`: `SEC_TRANSFORM_BLOCKER`
* `scripts/fundamentals/sec_ticker_cik_index.py`: `SEC_MAPPING_DEPENDENCY`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`: `PROVIDER_SIDE_EFFECT_RISK`

Validation:

* focused related tests: `44 passed in 0.04s`
* full suite: `550 passed in 0.55s`

Recommended next sprint:

* BL88 — Decouple active tests from remaining script-era fundamentals analysis and quality modules

Recommended follow-up:

* BL89 — Decouple active SEC transform/review tests from script-era SEC transformation modules

Guardrails:

* no live SEC/EDGAR calls
* no yfinance calls
* no credentials read
* no production data writes
* no reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules executed
* no script-era runtime modules edited
* no script-era runtime files archived
* no script-era runtime files deleted


### BL88 — Decouple active tests from remaining script-era fundamentals analysis and quality modules

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL88 decoupled active tests and analysis metadata from:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_quality.py`

Updated source metadata:

* `src/market_scanner/analysis/analysis_boundary.py`

Updated tests:

* `tests/unit/test_v2_canonical_analysis.py`
* `tests/core/test_fundamentals_runtime_organization.py`
* `tests/core/test_fundamentals_operational_validation.py`
* `tests/core/test_build_fundamental_analysis.py`
* `tests/core/test_build_fundamental_layer.py`

Result:

* `LEGACY_ANALYSIS_AUTHORITIES` is now empty.
* Canonical analysis metadata now tracks migrated analysis contract authorities:

  * `src/market_scanner/analysis/analysis_boundary.py`
  * `src/market_scanner/analysis/analysis_contracts.py`
* Canonical analysis metadata continues to track migrated fundamentals contract authorities:

  * `src/market_scanner/fundamentals/fundamental_contracts.py`
  * `src/market_scanner/fundamentals/fundamentals_metrics_contracts.py`
* Active tests no longer depend on `scripts/fundamentals/build_analysis.py`.
* Active tests no longer depend on `scripts/fundamentals/build_quality.py`.
* Remaining grep hits are canonical function names or negative guardrail assertions only.

Validation:

* focused tests: `20 passed in 0.03s`
* full suite: `551 passed in 0.55s`

Archive decision after BL88:

* `scripts/fundamentals/build_analysis.py`: `ACTIVE_REFERENCE_DECOUPLED_BUT_CLUSTER_BLOCKED`
* `scripts/fundamentals/build_quality.py`: `ACTIVE_REFERENCE_DECOUPLED_BUT_HIGH_RISK_CLUSTER_BLOCKED`
* `scripts/fundamentals/build_history_intake.py`: `CLUSTER_DEPENDENCY_BLOCKED`
* `scripts/fundamentals/build_metrics.py`: `CLUSTER_DEPENDENCY_BLOCKED`

Recommended next sprint:

* BL89 — Decouple active SEC transform/review tests from script-era SEC transformation modules

Guardrails:

* no live SEC/EDGAR calls
* no yfinance calls
* no credentials read
* no production data writes
* no reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules executed
* no script-era runtime modules edited
* no script-era runtime files archived
* no script-era runtime files deleted


### BL89 — Decouple active SEC transform/review tests from script-era SEC transformation modules

Category: Legacy Runtime Cleanup / SEC CompanyFacts Governance

Status: COMPLETED

BL89 decoupled active SEC transform/review tests from script-era SEC transformation module paths.

Updated tests:

* `tests/fundamentals/test_sec_companyfacts_transform.py`
* `tests/fundamentals/test_run_sec_transformation_review.py`
* `tests/fundamentals/test_sec_ticker_cik_index.py`

Reviewed related tests:

* `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`
* `tests/test_operator_visibility.py`

Target script-era modules:

* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

Result:

* active positive test dependency on `run_sec_transformation_review.py` removed;
* active positive test dependency on `sec_companyfacts_transform.py` removed;
* active positive test dependency on `sec_ticker_cik_index.py` removed;
* remaining references are limited to negative guardrails or operator-visibility references to test files.

Validation:

* focused SEC-related tests: `40 passed in 0.05s`
* full suite: `551 passed in 0.58s`

Archive decision after BL89:

* `scripts/fundamentals/run_sec_transformation_review.py`: `ACTIVE_REFERENCE_DECOUPLED_BUT_CLUSTER_BLOCKED`
* `scripts/fundamentals/sec_companyfacts_transform.py`: `ACTIVE_REFERENCE_DECOUPLED_BUT_CLUSTER_BLOCKED`
* `scripts/fundamentals/sec_ticker_cik_index.py`: `ACTIVE_REFERENCE_DECOUPLED_BUT_CLUSTER_BLOCKED`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`: `PROVIDER_SIDE_EFFECT_RISK`
* `scripts/fundamentals/build_analysis.py`: `ACTIVE_REFERENCE_DECOUPLED_BUT_CLUSTER_BLOCKED`
* `scripts/fundamentals/build_quality.py`: `ACTIVE_REFERENCE_DECOUPLED_BUT_HIGH_RISK_CLUSTER_BLOCKED`
* `scripts/fundamentals/build_history_intake.py`: `CLUSTER_DEPENDENCY_BLOCKED`
* `scripts/fundamentals/build_metrics.py`: `CLUSTER_DEPENDENCY_BLOCKED`

Recommended next sprint:

* BL90 — Final archive-readiness review of remaining scripts/fundamentals cluster

Guardrails:

* no live SEC/EDGAR calls
* no yfinance calls
* no credentials read
* no production data writes
* no reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules executed
* no script-era runtime modules edited
* no script-era runtime files archived
* no script-era runtime files deleted


### BL90 — Final archive-readiness review of remaining scripts/fundamentals cluster

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL90 performed a final archive-readiness review of the remaining `scripts/fundamentals/` cluster.

Reviewed files:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`
* `scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

Decision:

* `NOT_ARCHIVE_READY_YET`

Reason:

* one active positive test reference remains:

  * `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`
* this test still references:

  * `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* `sec_companyfacts_bulk_intake.py` is provider/network/cache-risk.

File-level decisions:

* `scripts/fundamentals/build_analysis.py`: `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`
* `scripts/fundamentals/build_history_intake.py`: `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`
* `scripts/fundamentals/build_metrics.py`: `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`
* `scripts/fundamentals/build_quality.py`: `HIGH_RISK_CLUSTER_ARCHIVE_CANDIDATE_AFTER_BULK_INTAKE_DECOUPLING`
* `scripts/fundamentals/run_sec_transformation_review.py`: `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`
* `scripts/fundamentals/sec_companyfacts_transform.py`: `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`
* `scripts/fundamentals/sec_ticker_cik_index.py`: `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`: `NOT_ARCHIVE_READY_PROVIDER_RISK_ACTIVE_TEST_REFERENCE`

Validation:

* focused review suite: `85 passed in 0.08s`
* full suite: `551 passed in 0.67s`

Recommended next sprint:

* BL91 — Decouple active bulk SEC CompanyFacts intake test from provider-risk script-era module

Likely follow-up:

* BL92 — Archive remaining scripts/fundamentals cluster after final no-active-reference check

Guardrails:

* no live SEC/EDGAR calls
* no yfinance calls
* no credentials read
* no production data writes
* no reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules executed
* no script-era runtime modules edited
* no script-era runtime files archived
* no script-era runtime files deleted


### BL91 — Decouple active bulk SEC CompanyFacts intake test from provider-risk script-era module

Category: Legacy Runtime Cleanup / SEC CompanyFacts Governance

Status: COMPLETED

BL91 decoupled the final active positive test reference to the provider-risk script-era bulk SEC CompanyFacts intake module:

* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`

Updated test:

* `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`

Result:

* removed positive path reference to `scripts/fundamentals/sec_companyfacts_bulk_intake.py`;
* preserved SEC CompanyFacts provider-governance policy;
* preserved explicit operator-action requirement for network/cache/manifest behavior;
* preserved no-network/no-cache-write test guardrails;
* preserved no-investment-authority policy.

Validation:

* focused bulk-intake test: `4 passed in 0.02s`
* BL90/BL91 focused suite: `89 passed in 0.07s`
* full suite: `553 passed in 0.57s`

Archive-readiness impact:

* `sec_companyfacts_bulk_intake.py` active positive path reference is now decoupled;
* provider/network/cache-risk remains but is no longer actively referenced by tests as a runtime path;
* the remaining `scripts/fundamentals/` cluster is now candidate for archive after one final no-active-positive-reference check.

Recommended next sprint:

* BL92 — Archive remaining scripts/fundamentals cluster after final no-active-reference check

Candidate archive targets:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`
* `scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

Guardrails:

* no live SEC/EDGAR calls
* no yfinance calls
* no credentials read
* no production data writes
* no reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules executed
* no script-era runtime modules edited
* no script-era runtime files archived
* no script-era runtime files deleted


### BL92 — Archive remaining scripts/fundamentals cluster after final no-active-reference check

Category: Legacy Runtime Cleanup / Fundamentals Governance

Status: COMPLETED

BL92 archived the remaining `scripts/fundamentals/` script-era runtime cluster after final no-active-reference checks confirmed that active `src`, `tests`, and `.github` code no longer depends on these files.

Archived files:

* `scripts/fundamentals/build_analysis.py` -> `archive/legacy_runtime/scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_history_intake.py` -> `archive/legacy_runtime/scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py` -> `archive/legacy_runtime/scripts/fundamentals/build_metrics.py`
* `scripts/fundamentals/build_quality.py` -> `archive/legacy_runtime/scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/run_sec_transformation_review.py` -> `archive/legacy_runtime/scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py` -> `archive/legacy_runtime/scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* `scripts/fundamentals/sec_companyfacts_transform.py` -> `archive/legacy_runtime/scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py` -> `archive/legacy_runtime/scripts/fundamentals/sec_ticker_cik_index.py`

Final checks:

* no active positive file-path references from `src`, `tests`, or `.github`;
* no active positive `scripts.fundamentals` imports from `src`, `tests`, or `.github`;
* no active `scripts/fundamentals/*.py` files remain.

Validation:

* focused regression suite: `89 passed in 0.07s`
* full suite: `553 passed in 0.58s`

Decision:

* `ARCHIVED`

Impact:

* active fundamentals runtime is now canonical under `src/market_scanner/`;
* script-era fundamentals runtime implementation is preserved under `archive/legacy_runtime/`;
* provider-risk SEC CompanyFacts bulk-intake implementation is no longer active runtime and remains historical legacy evidence only.

Recommended next sprint:

* BL93 — Review remaining active scripts/ tree after fundamentals archive

Guardrails:

* no live SEC/EDGAR calls
* no yfinance calls
* no credentials read
* no production data writes
* no reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules executed
* no script-era runtime behavior modified
* files archived, not deleted


### BL93 — Review remaining active scripts tree after fundamentals archive

Category: Legacy Runtime Cleanup / Repository Governance

Status: COMPLETED

BL93 reviewed the remaining active `scripts/` tree after BL92 archived the full `scripts/fundamentals/` runtime cluster.

Result:

* 32 active `scripts/**/*.py` files remain.
* The active `scripts/fundamentals/` runtime cluster is no longer present.
* Active tests still import script-era modules from:

  * `scripts.reporting`
  * `scripts.core`
  * `scripts.portfolio`
  * `scripts.ops`
  * `scripts.data_sources`
  * `scripts.diagnostics`
* Canonical boundary metadata still references script-era files in:

  * reporting
  * messaging
  * delivery
  * decision
  * scanner
* Side-effect markers remain across:

  * reporting and Telegram delivery
  * scanner/provider access
  * core layer builders and Decision Engine
  * data-source prefill utilities
  * portfolio and watchlist mutation scripts
  * ops evidence capture

Decision:

* `REMAINING_SCRIPTS_TREE_NOT_ARCHIVE_READY_AS_A_WHOLE`

Reason:

* broad active test coupling remains;
* positive script-era metadata references remain;
* high-risk side-effect surfaces remain;
* domain-specific decoupling is required before further archive.

Validation:

* full suite: `553 passed in 0.58s`

Recommended next sprint:

* BL94 — Decouple active reporting, messaging, and delivery tests from script-era reporting and Telegram modules

Candidate targets:

* `scripts/reporting/build_reporting_layer.py`
* `scripts/reporting/build_telegram_summary.py`
* `scripts/reporting/send_telegram.py`
* `scripts/telegram/process_telegram_commands.py`

Likely follow-up:

* BL95 — Archive reporting/messaging/delivery script-era modules after final no-active-reference check

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules archived
* no script-era runtime modules edited
* no script-era runtime modules executed directly


### BL94 — Decouple reporting, messaging, and delivery from script-era modules

Category: Legacy Runtime Cleanup / Reporting, Messaging & Delivery Governance

Status: COMPLETED

BL94 decoupled active reporting, messaging, and delivery tests and canonical boundary metadata from script-era reporting and Telegram modules.

Targeted script-era modules:

* `scripts/reporting/build_reporting_layer.py`
* `scripts/reporting/build_telegram_summary.py`
* `scripts/reporting/send_telegram.py`
* `scripts/telegram/process_telegram_commands.py`

Changed files:

* `src/market_scanner/delivery/delivery_boundary.py`
* `src/market_scanner/messaging/message_boundary.py`
* `src/market_scanner/reporting/report_boundary.py`
* `tests/conftest.py`
* `tests/reporting/test_build_reporting_layer.py`
* `tests/reporting/test_build_telegram_summary.py`
* `tests/test_operator_visibility.py`
* `tests/unit/test_v2_canonical_delivery.py`
* `tests/unit/test_v2_canonical_messaging.py`
* `tests/unit/test_v2_canonical_reporting.py`

Result:

* active reporting tests no longer import `scripts.reporting`;
* active tests no longer import `scripts.telegram`;
* canonical reporting/messaging/delivery tests no longer statically read the targeted script-era files;
* canonical boundary metadata no longer lists the targeted active script-era paths as legacy authorities;
* operator visibility blockers were updated because the reporting tests are no longer script-era blocker tests.

Validation:

* focused suite: `45 passed in 0.05s`
* full suite: `560 passed in 0.57s`

Active import check:

* no active positive imports remain from `scripts.reporting`;
* no active positive imports remain from `scripts.telegram`.

Active path reference check:

* remaining references to targeted script-era paths are negative guardrail assertions only;
* they are not imports, static file reads, runtime dependencies, or execution paths.

Decision:

* `REPORTING_MESSAGING_DELIVERY_ACTIVE_DEPENDENCIES_DECOUPLED`

Recommended next sprint:

* BL95 — Archive reporting, messaging, and delivery script-era modules after final no-active-reference check

Candidate BL95 archive targets:

* `scripts/reporting/build_reporting_layer.py`
* `scripts/reporting/build_telegram_summary.py`
* `scripts/reporting/send_telegram.py`
* `scripts/telegram/process_telegram_commands.py`

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era reporting module executed
* no script-era Telegram module executed
* no script-era runtime modules archived
* no script-era runtime behavior modified


### BL95 — Archive reporting, messaging, and delivery script-era modules

Category: Legacy Runtime Cleanup / Reporting, Messaging & Delivery Governance

Status: COMPLETED

BL95 archived the remaining script-era reporting, messaging, and delivery modules after BL94 decoupled active tests and canonical boundary metadata from them.

Archived files:

* `scripts/reporting/build_reporting_layer.py` -> `archive/legacy_runtime/scripts/reporting/build_reporting_layer.py`
* `scripts/reporting/build_telegram_summary.py` -> `archive/legacy_runtime/scripts/reporting/build_telegram_summary.py`
* `scripts/reporting/send_telegram.py` -> `archive/legacy_runtime/scripts/reporting/send_telegram.py`
* `scripts/telegram/process_telegram_commands.py` -> `archive/legacy_runtime/scripts/telegram/process_telegram_commands.py`

Pre-archive checks:

* no active positive imports remain from `scripts.reporting`;
* no active positive imports remain from `scripts.telegram`;
* remaining targeted path references are negative guardrail assertions only.

Validation:

* focused suite: `45 passed in 0.03s`
* full suite: `560 passed in 0.53s`

Decision:

* `ARCHIVED`

Impact:

* active `scripts/reporting/*.py` no longer exists;
* active `scripts/telegram/*.py` no longer exists;
* historical script-era reporting and Telegram implementation remains preserved under `archive/legacy_runtime/`.

Operator note:

* archived file paths were accidentally pasted into the terminal as shell commands;
* the shell produced permission/syntax errors;
* no Python module execution, Telegram delivery, credential read, production data write, or report generation occurred.

Recommended next sprint:

* BL96 — Review remaining active scripts tree after reporting and Telegram archive

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* files archived, not deleted


### BL96 — Review remaining active scripts tree after reporting and Telegram archive

Category: Legacy Runtime Cleanup / Repository Governance

Status: COMPLETED

BL96 reviewed the remaining active `scripts/` tree after BL92 archived `scripts/fundamentals/` and BL95 archived `scripts/reporting/` and `scripts/telegram/`.

Result:

* 28 active `scripts/**/*.py` files remain.
* Active `scripts/fundamentals/` Python runtime files no longer exist.
* Active `scripts/reporting/` Python runtime files no longer exist.
* Active `scripts/telegram/` Python runtime files no longer exist.
* Active tests still import script-era modules from:

  * `scripts.core`
  * `scripts.portfolio`
  * `scripts.ops`
  * `scripts.data_sources`
  * `scripts.diagnostics`
* Canonical boundary metadata still references script-era files in:

  * decision
  * scanner
* Side-effect markers remain across:

  * core layer builders
  * scanner/provider access
  * Decision Engine
  * data-source prefill utilities
  * portfolio state
  * watchlist state
  * ops evidence capture
  * diagnostics

Decision:

* `REMAINING_SCRIPTS_TREE_NOT_ARCHIVE_READY_AS_A_WHOLE`

Reason:

* broad active test coupling remains;
* positive script-era metadata references remain;
* high-risk side-effect surfaces remain;
* domain-specific decoupling is required before further archive.

Validation:

* full suite: `560 passed in 0.57s`

Recommended next sprint:

* BL97 — Decouple active data-source tests from script-era data_sources modules

Candidate targets:

* `scripts/data_sources/common.py`
* `scripts/data_sources/prefill_fundamentals.py`
* `scripts/data_sources/prefill_portfolio_metadata.py`

Candidate tests:

* `tests/data_sources/test_prefill_common.py`
* `tests/data_sources/test_prefill_fundamentals.py`
* `tests/data_sources/test_prefill_portfolio_metadata.py`

Likely follow-up:

* BL98 — Archive data_sources script-era modules after final no-active-reference check

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules archived
* no script-era runtime modules edited
* no script-era runtime modules executed directly


### BL97 — Decouple active data-source tests from script-era data_sources modules

Category: Legacy Runtime Cleanup / Data Source Governance

Status: COMPLETED

BL97 decoupled active data-source tests from script-era `scripts/data_sources` modules.

Targeted script-era modules:

* `scripts/data_sources/common.py`
* `scripts/data_sources/prefill_fundamentals.py`
* `scripts/data_sources/prefill_portfolio_metadata.py`

Changed files:

* `tests/conftest.py`
* `tests/data_sources/test_prefill_common.py`
* `tests/data_sources/test_prefill_fundamentals.py`
* `tests/data_sources/test_prefill_portfolio_metadata.py`
* `tests/test_operator_visibility.py`

Result:

* active tests no longer import `scripts.data_sources`;
* active tests no longer require the `scripts` package during focused collection;
* data-source tests now validate static/canonical source-data contracts instead of script-era runtime behavior;
* decoupled data-source tests were removed from high-risk script-era blocker registries.

Validation:

* focused suite: `15 passed in 0.05s`
* full suite: `569 passed in 0.59s`

Active import/path check:

* no `scripts.data_sources` imports or `scripts/data_sources` path references remain in active `src`, `tests`, or `.github`.

Decision:

* `DATA_SOURCE_ACTIVE_TEST_DEPENDENCIES_DECOUPLED`

Recommended next sprint:

* BL98 — Archive data_sources script-era modules after final no-active-reference check

Candidate BL98 archive targets:

* `scripts/data_sources/common.py`
* `scripts/data_sources/prefill_fundamentals.py`
* `scripts/data_sources/prefill_portfolio_metadata.py`

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era data-source module executed
* no script-era data-source module edited
* no script-era runtime module archived


### BL98 — Archive data_sources script-era modules

Category: Legacy Runtime Cleanup / Data Source Governance

Status: COMPLETED

BL98 archived the script-era `scripts/data_sources` modules after BL97 decoupled active tests from them.

Archived files:

* `scripts/data_sources/common.py` -> `archive/legacy_runtime/scripts/data_sources/common.py`
* `scripts/data_sources/prefill_fundamentals.py` -> `archive/legacy_runtime/scripts/data_sources/prefill_fundamentals.py`
* `scripts/data_sources/prefill_portfolio_metadata.py` -> `archive/legacy_runtime/scripts/data_sources/prefill_portfolio_metadata.py`

Pre-archive checks:

* no active positive imports remain from `scripts.data_sources`;
* no active positive path references remain to `scripts/data_sources` in `src`, `tests`, or `.github`.

Post-archive result:

* active `scripts/data_sources/*.py` no longer exists;
* historical script-era data-source implementation remains preserved under `archive/legacy_runtime/scripts/data_sources/`.

Validation:

* focused suite: `15 passed in 0.03s`
* full suite: `569 passed in 0.57s`

Decision:

* `ARCHIVED`

Recommended next sprint:

* BL99 — Review remaining active scripts tree after data_sources archive

Goal:

* inspect remaining active `scripts/**/*.py` files after BL92, BL95, and BL98;
* confirm that `scripts/fundamentals`, `scripts/reporting`, `scripts/telegram`, and `scripts/data_sources` are now archived;
* identify the safest next decoupling domain.

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era data-source module modified
* no script-era data-source module executed
* files archived, not deleted


### BL99 — Review remaining active scripts tree after data_sources archive

Category: Legacy Runtime Cleanup / Repository Governance

Status: COMPLETED

BL99 reviewed the remaining active `scripts/` tree after BL92 archived `scripts/fundamentals/`, BL95 archived `scripts/reporting/` and `scripts/telegram/`, and BL98 archived `scripts/data_sources/`.

Result:

* 25 active `scripts/**/*.py` files remain.
* Active `scripts/fundamentals/` Python runtime files no longer exist.
* Active `scripts/reporting/` Python runtime files no longer exist.
* Active `scripts/telegram/` Python runtime files no longer exist.
* Active `scripts/data_sources/` Python runtime files no longer exist.
* Active tests still import script-era modules from:

  * `scripts.core`
  * `scripts.portfolio`
  * `scripts.ops`
  * `scripts.diagnostics`
* Canonical boundary metadata still references script-era files in:

  * decision
  * scanner
* Side-effect markers remain across:

  * core layer builders
  * scanner/provider access
  * Decision Engine
  * ops evidence capture
  * portfolio state
  * watchlist state
  * scanner validation

Decision:

* `REMAINING_SCRIPTS_TREE_NOT_ARCHIVE_READY_AS_A_WHOLE`

Reason:

* active test coupling remains;
* positive script-era metadata references remain;
* high-risk side-effect surfaces remain;
* domain-specific decoupling is required before further archive.

Validation:

* full suite: `569 passed in 0.58s`

Recommended next sprint:

* BL100 — Decouple ops and diagnostics tests from script-era modules

Candidate targets:

* `scripts/ops/capture_historical_evidence.py`
* `archive/legacy_runtime/scripts/diagnostics/audit_data_coverage.py`

Candidate tests:

* `tests/ops/test_capture_historical_evidence.py`
* `tests/diagnostics/test_audit_data_coverage.py`

Likely follow-up:

* BL101 — Archive ops capture script after final no-active-reference check

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules archived
* no script-era runtime modules edited
* no script-era runtime modules executed directly


### BL100 — Decouple ops and diagnostics tests from script-era modules

Category: Legacy Runtime Cleanup / Test Decoupling

Status: COMPLETED

BL100 decoupled active ops and diagnostics tests from script-era modules.

Targeted script-era modules:

* `scripts/ops/capture_historical_evidence.py`
* `archive/legacy_runtime/scripts/diagnostics/audit_data_coverage.py`

Updated tests:

* `tests/ops/test_capture_historical_evidence.py`
* `tests/diagnostics/test_audit_data_coverage.py`

Updated blocker registries:

* `tests/conftest.py`
* `tests/test_operator_visibility.py`

Result:

* `tests/ops/test_capture_historical_evidence.py` no longer imports `scripts.ops.capture_historical_evidence`.
* `tests/diagnostics/test_audit_data_coverage.py` no longer imports `scripts.diagnostics.audit_data_coverage`.
* Both tests now validate static/canonical contracts instead of script-era runtime behavior.
* The two tests were removed from the high-risk script-era blocker registries.

Active import check:

* no active positive imports remain from `scripts.ops`;
* no active positive imports remain from `scripts.diagnostics`.

Validation:

* focused suite: `18 passed in 0.06s`
* full suite: `581 passed in 0.62s`

Decision:

* `OPS_AND_DIAGNOSTICS_ACTIVE_TEST_DEPENDENCIES_DECOUPLED`

Remaining archive-readiness note:

* `scripts/ops/capture_historical_evidence.py` still physically exists and is not archived by BL100.
* `archive/legacy_runtime/scripts/diagnostics/audit_data_coverage.py` remains an archived historical reference.

Recommended next sprint:

* BL101 — Archive ops capture script after final no-active-reference check

Candidate archive target:

* `scripts/ops/capture_historical_evidence.py`

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime module archived
* no script-era runtime module edited
* no script-era runtime module executed directly


### BL101 — Archive ops capture script

Category: Legacy Runtime Cleanup / Ops Archive

Status: COMPLETED

BL101 archived the remaining active ops script-era module after BL100 decoupled active ops and diagnostics tests from script-era modules.

Archived file:

* `scripts/ops/capture_historical_evidence.py` -> `archive/legacy_runtime/scripts/ops/capture_historical_evidence.py`

Pre-archive checks:

* no active positive imports remain from `scripts.ops`;
* no active runtime import, workflow invocation, or source dependency remains for `scripts/ops/capture_historical_evidence.py`;
* remaining references are static/negative test guardrails only.

Post-archive result:

* active `scripts/ops/*.py` no longer exists;
* historical ops evidence-capture implementation remains preserved under `archive/legacy_runtime/scripts/ops/`.

Validation:

* focused suite: `12 passed in 0.03s`
* full suite: `581 passed in 0.58s`

Decision:

* `ARCHIVED`

Remaining cleanup status:

* `scripts/fundamentals/` archived
* `scripts/reporting/` archived
* `scripts/telegram/` archived
* `scripts/data_sources/` archived
* `scripts/ops/` archived

Recommended next sprint:

* BL102 — Review remaining active scripts tree after ops archive

Goal:

* inspect remaining active `scripts/**/*.py` files after BL92, BL95, BL98, and BL101;
* confirm that archived domains are no longer active Python runtime paths;
* identify the safest next decoupling domain.

Likely next candidates:

* selected `scripts/core` layer builders

High-risk domains to avoid archiving casually:

* Decision Engine
* scanner/provider access
* portfolio
* watchlist

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era ops module modified
* no script-era ops module executed
* files archived, not deleted


### BL102 — Review remaining active scripts tree after ops archive

Category: Legacy Runtime Cleanup / Repository Governance

Status: COMPLETED

BL102 reviewed the remaining active `scripts/` tree after BL92, BL95, BL98, and BL101 archived the fundamentals, reporting, telegram, data_sources, and ops script-era domains.

Result:

* 24 active `scripts/**/*.py` files remain.
* Active `scripts/fundamentals/` Python runtime files no longer exist.
* Active `scripts/reporting/` Python runtime files no longer exist.
* Active `scripts/telegram/` Python runtime files no longer exist.
* Active `scripts/data_sources/` Python runtime files no longer exist.
* Active `scripts/ops/` Python runtime files no longer exist.

Remaining active domains:

* `scripts/core/`
* `scripts/portfolio/`
* `scripts/watchlist/`
* `scripts/validate_scans.py`

Active test coupling remains in:

* core layer-builder tests;
* Decision Engine tests;
* portfolio source contract tests.

Canonical metadata still references:

* `scripts/core/decision_engine.py`
* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`

Side-effect and runtime-risk markers remain across:

* core layer builders;
* scanner/provider access;
* Decision Engine;
* portfolio state;
* watchlist state;
* scan validation.

Validation:

* full suite: `581 passed in 0.58s`

Decision:

* `REMAINING_SCRIPTS_TREE_NOT_ARCHIVE_READY_AS_A_WHOLE`

Recommended next sprint:

* BL103 — Decouple selected core layer tests from script-era modules

Candidate tests:

* `tests/core/test_build_context_layer.py`
* `tests/core/test_build_validation_layer.py`
* `tests/core/test_entry_quality.py`
* `tests/core/test_build_timing_state_layer.py`
* `tests/core/test_build_stability_layer.py`

Candidate script-era modules:

* `scripts/core/build_context_layer.py`
* `scripts/core/build_validation_layer.py`
* `scripts/core/build_timing_state_layer.py`
* `scripts/core/build_stability_layer.py`

High-risk areas to avoid in BL103:

* Decision Engine
* scanner/provider access
* portfolio
* watchlist
* scan validation

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules archived
* no script-era runtime modules edited
* no script-era runtime modules executed directly


### BL103 — Decouple selected core layer tests from script-era modules

Category: Legacy Runtime Cleanup / Test Decoupling

Status: COMPLETED

BL103 decoupled selected active core-layer tests from script-era modules.

Updated tests:

* `tests/core/test_build_context_layer.py`
* `tests/core/test_build_validation_layer.py`
* `tests/core/test_entry_quality.py`
* `tests/core/test_build_timing_state_layer.py`
* `tests/core/test_build_stability_layer.py`

Updated blocker registries:

* `tests/conftest.py`
* `tests/test_operator_visibility.py`

Targeted script-era modules:

* `scripts/core/build_context_layer.py`
* `scripts/core/build_validation_layer.py`
* `scripts/core/build_timing_state_layer.py`
* `scripts/core/build_stability_layer.py`

Result:

* the selected tests no longer import script-era modules;
* the tests now validate static/canonical contracts;
* the selected tests were removed from the high-risk script-era blocker registries;
* the full suite now includes these tests again.

Validation:

* focused suite: `35 passed in 0.07s`
* full suite: `610 passed in 0.64s`

Decision:

* `SELECTED_CORE_LAYER_ACTIVE_TEST_DEPENDENCIES_DECOUPLED`

Remaining active positive `scripts.core` test imports:

* `tests/core/test_build_entry_quality_backfill.py`
* `tests/core/test_build_context_backfill.py`
* `tests/core/test_decision_engine.py`
* `tests/core/test_build_portfolio_intelligence.py`

Recommended next sprint:

* BL104 — Review archive-readiness of decoupled core layer modules

Candidate modules for review only:

* `scripts/core/build_context_layer.py`
* `scripts/core/build_validation_layer.py`
* `scripts/core/build_timing_state_layer.py`
* `scripts/core/build_stability_layer.py`

High-risk areas still out of scope:

* Decision Engine
* scanner/provider access
* portfolio
* watchlist
* scan validation

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules archived
* no script-era runtime modules edited
* no script-era runtime modules executed directly


### BL104 — Review archive-readiness of decoupled core layer modules

Category: Legacy Runtime Cleanup / Archive Readiness Review

Status: COMPLETED

BL104 reviewed archive-readiness of the selected core layer modules that were decoupled from active tests in BL103.

Targeted modules:

* `scripts/core/build_context_layer.py`
* `scripts/core/build_validation_layer.py`
* `scripts/core/build_timing_state_layer.py`
* `scripts/core/build_stability_layer.py`

Result:

* no active positive imports remain for the four targeted modules;
* remaining references are static contract-test references and negative import guardrails;
* active `scripts.core` imports remain elsewhere, outside BL104 scope;
* all four targeted modules still contain side-effect/runnable markers.

Remaining side-effect/runnable markers include:

* fixed `data/processed` paths;
* fixed `data/logs` paths;
* `pd.read_csv(...)`;
* `to_csv(...)`;
* `mkdir(...)`;
* `if __name__ == "__main__"`;
* `main()` in the stability layer.

Validation:

* focused suite: `35 passed in 0.07s`
* full suite: `610 passed in 0.62s`

Decision:

* `DECOUPLED_CORE_LAYER_MODULES_NOT_ARCHIVE_READY_DUE_TO_MANUAL_RUN_AND_WRITE_RISK`

Recommended next sprint:

* BL105 — Fail-close or de-run selected decoupled core layer modules before archive

BL105 goal:

* remove or guard manual execution risk from the four selected modules;
* preserve historical implementation for later archive;
* avoid changing canonical runtime behavior;
* avoid executing script-era modules.

High-risk areas still out of scope:

* Decision Engine
* scanner/provider access
* portfolio
* watchlist
* scan validation
* portfolio intelligence

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules archived
* no script-era runtime modules edited
* no script-era runtime modules executed directly


### BL105 — Fail-close selected decoupled core layer modules before archive

Category: Legacy Runtime Cleanup / De-run / Fail-close

Status: COMPLETED

BL105 fail-closed manual execution surfaces for selected decoupled core layer script-era modules.

Updated modules:

* `scripts/core/build_context_layer.py`
* `scripts/core/build_validation_layer.py`
* `scripts/core/build_timing_state_layer.py`
* `scripts/core/build_stability_layer.py`

Result:

* direct manual execution through `if __name__ == "__main__"` is now fail-closed;
* `scripts/core/build_stability_layer.py` now has a fail-closed `main()` function;
* historical function bodies were preserved;
* no modules were archived;
* no canonical runtime behavior was changed.

Validation:

* focused suite: `35 passed in 0.07s`
* full suite: `610 passed in 0.64s`

Decision:

* `SELECTED_DECOUPLED_CORE_LAYER_MANUAL_ENTRYPOINTS_FAIL_CLOSED`

Remaining archive-readiness status:

* internal `pd.read_csv(...)`, `to_csv(...)`, `mkdir(...)`, and fixed data paths remain;
* the modules require one final archive-readiness review before controlled archive;
* broader `scripts/core` remains blocked by active positive imports outside BL105 scope.

Recommended next sprint:

* BL106 — Final archive-readiness review for fail-closed core layer modules

Candidate modules:

* `scripts/core/build_context_layer.py`
* `scripts/core/build_validation_layer.py`
* `scripts/core/build_timing_state_layer.py`
* `scripts/core/build_stability_layer.py`

High-risk areas still out of scope:

* Decision Engine
* scanner/provider access
* portfolio
* watchlist
* scan validation
* portfolio intelligence

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed
* no script-era runtime modules archived
* no script-era runtime modules executed directly
* only manual entrypoint fail-close behavior changed


### BL106 — Final archive-readiness review for fail-closed core layer modules

Category: Legacy Runtime Cleanup / Archive Readiness Review

Status: COMPLETED

BL106 reviewed final archive-readiness for the selected fail-closed core layer script-era modules.

Reviewed modules:

* `scripts/core/build_context_layer.py`
* `scripts/core/build_validation_layer.py`
* `scripts/core/build_timing_state_layer.py`
* `scripts/core/build_stability_layer.py`

Result:

* no active positive imports remain for the four targeted modules;
* remaining target-module references are static tests, negative import guardrails, backlog entries, audit notes, historical archive documents, and legacy documentation;
* active `scripts.core` imports remain elsewhere, outside BL106 scope;
* manual execution remains fail-closed;
* historical read/write bodies remain preserved for archive.

Validation:

* focused suite: `35 passed in 0.06s`
* full suite: `610 passed in 0.62s`

Decision:

* `BL107_ARCHIVE_SPRINT_APPROVED`


### BL107 — Controlled archive of fail-closed core layer modules

Category: Legacy Runtime Cleanup / Controlled Archive

Status: COMPLETED

BL107 archived the four fail-closed core layer script-era modules.

Archived modules:

* `archive/legacy_runtime/scripts/core/build_context_layer.py`
* `archive/legacy_runtime/scripts/core/build_validation_layer.py`
* `archive/legacy_runtime/scripts/core/build_timing_state_layer.py`
* `archive/legacy_runtime/scripts/core/build_stability_layer.py`

Result:

* the four-module core-layer archive cluster is completed;
* historical source content was preserved through `git mv`;
* no canonical runtime under `src/market_scanner/` was changed;
* remaining active `scripts.core` imports reference other modules outside BL107 scope.

Validation:

* post-archive focused suite: `35 passed in 0.05s`
* post-archive full suite: `610 passed in 0.64s`

Decision:

* `FAIL_CLOSED_CORE_LAYER_ARCHIVE_CLUSTER_COMPLETED`

Recommended next sprint:

* BL108 — Review remaining active `scripts/` dependencies after core layer archive

BL108 goal:

* perform a review-only dependency scan of remaining active `scripts/` dependencies;
* do not archive additional files without a separate archive-readiness decision;
* preserve Decision Engine authority and all canonical runtime boundaries.

High-risk areas still out of scope:

* Decision Engine
* scanner/provider access
* portfolio
* watchlist
* scan validation
* portfolio intelligence

Guardrails:

* no live provider calls
* no yfinance calls
* no SEC/EDGAR calls
* no credentials read
* no production data writes
* no production reports generated
* no Telegram messages sent
* no portfolio/watchlist state modified
* no Decision Engine authority changed


### BL111 — Fail-close decoupled historical backfill modules

Category: Legacy Runtime Cleanup / Fail-close

Status: COMPLETED

BL111 fail-closed manual execution surfaces for the decoupled historical backfill modules.

Updated modules:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

Result:

* both modules now define a `FAIL_CLOSED_MESSAGE`;
* direct `if __name__ == "__main__"` execution raises `SystemExit` with the fail-closed message;
* public `main()` functions raise `SystemExit` with the fail-closed message;
* historical `main()` bodies were preserved as private `_legacy_main_impl(...)` helpers;
* no archive move was performed;
* no historical helper logic was refactored;
* no provider calls, production data writes, reports, Telegram delivery, portfolio state changes, watchlist state changes, scan validation runtime changes, Decision Engine changes, or portfolio intelligence changes were performed.

Validation:

* focused suite: `24 passed in 0.37s`
* full suite: `628 passed in 0.99s`
* direct execution safety check: both target scripts exited immediately with the fail-closed message and no writes.

Decision:

* `BL112_ARCHIVE_READINESS_REVIEW_APPROVED_FOR_FAIL_CLOSED_HISTORICAL_BACKFILL_MODULES`

Recommended next sprint:

* BL112 — Archive-readiness review for fail-closed historical backfill modules

BL112 goal:

* review archive-readiness for `scripts/core/build_entry_quality_backfill.py` and `scripts/core/build_context_backfill.py`;
* do not archive the modules unless a separate review proves readiness;
* preserve historical source content;
* keep the sprint review-only unless archive readiness is explicitly established.

High-risk areas still out of scope:

* Decision Engine
* portfolio intelligence
* portfolio source contract
* trade command parser
* scanner/provider runtime
* SEC/EDGAR
* yfinance
* credentials
* production data writes
* report generation
* Telegram
* watchlist state
* portfolio state

## BL113 — Controlled archive of fail-closed historical backfill modules

Status: COMPLETED

Context:
BL112 completed archive-readiness review for the two fail-closed historical backfill modules:

* `scripts/core/build_entry_quality_backfill.py`
* `scripts/core/build_context_backfill.py`

BL112 confirmed:

* no active imports from `src`, `tests`, or `.github`;
* public `main()` functions fail closed;
* direct script execution fails closed;
* historical bodies are preserved under `_legacy_main_impl(...)`;
* direct execution prints the `FAIL_CLOSED` message and performs no writes;
* focused tests passed;
* full suite passed.

Decision:
BL113 is approved as a controlled archive sprint.

Scope:

* archive `scripts/core/build_entry_quality_backfill.py`
* archive `scripts/core/build_context_backfill.py`

Required archive destination:

* `archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py`
* `archive/legacy_runtime/scripts/core/build_context_backfill.py`

Required constraints:

* use `git mv`;
* do not modify runtime behavior;
* do not modify tests unless required by collection/import failures;
* do not touch Decision Engine, portfolio intelligence, portfolio source contract, trade command parser, scanner/provider runtime, SEC/EDGAR, yfinance, credentials, production data, reports, Telegram, portfolio state, or watchlist state;
* run focused and full pytest suites after the archive.

Result:

* `scripts/core/build_entry_quality_backfill.py` was moved with `git mv` to `archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py`;
* `scripts/core/build_context_backfill.py` was moved with `git mv` to `archive/legacy_runtime/scripts/core/build_context_backfill.py`;
* no historical code inside the archived files was modified;
* no tests required changes;
* no canonical runtime behavior changed.

Validation:

* focused suite: `24 passed in 0.35s`
* full suite: `628 passed in 1.23s`

Decision:

* `BL113_COMPLETED_FAIL_CLOSED_HISTORICAL_BACKFILL_MODULES_ARCHIVED`


## BL114 — Review remaining active scripts/core tree after historical backfill archive

Status: proposed

Context:
BL113 archived the two fail-closed historical backfill modules:

- `scripts/core/build_entry_quality_backfill.py`
- `scripts/core/build_context_backfill.py`

The modules were moved with `git mv` to:

- `archive/legacy_runtime/scripts/core/build_entry_quality_backfill.py`
- `archive/legacy_runtime/scripts/core/build_context_backfill.py`

Required next step:
Perform a review-only inventory of the remaining active `scripts/core/` tree after the historical backfill archive.

BL114 must not archive anything. It must only classify remaining active script-era dependencies, active imports, side-effect risks, and candidate next sprints.


## BL115 — Decouple active portfolio intelligence tests from script-era module

Status: COMPLETED

Context:
BL114 reviewed the remaining active `scripts/core/` tree after BL113 archived the fail-closed historical backfill modules.

BL114 found 8 remaining active `scripts/core/` Python files:

* `scripts/core/build_portfolio_intelligence.py`
* `scripts/core/data_fetcher.py`
* `scripts/core/decision_engine.py`
* `scripts/core/indicators.py`
* `scripts/core/log_scans.py`
* `scripts/core/scanner.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`

BL114 found three remaining active positive `scripts.core` imports:

* `tests/core/test_decision_engine.py` imports `scripts.core.decision_engine`
* `tests/core/test_build_portfolio_intelligence.py` imports `scripts.core.build_portfolio_intelligence`
* `tests/portfolio/test_portfolio_source_contract.py` imports `scripts.core.build_portfolio_intelligence`

Decision:
BL115 must decouple only the portfolio intelligence tests from the script-era module.

Scope:

* `tests/core/test_build_portfolio_intelligence.py`
* `tests/portfolio/test_portfolio_source_contract.py`
* their import dependency on `scripts/core/build_portfolio_intelligence.py`

Required outcome:

* active tests no longer import `scripts.core.build_portfolio_intelligence`;
* test-local or canonical contract helpers preserve the existing tested behavior;
* `scripts/core/build_portfolio_intelligence.py` remains in place;
* no archive action is performed;
* Decision Engine remains untouched;
* scanner/provider runtime remains untouched;
* portfolio state and watchlist state remain untouched;
* focused and full pytest suites remain green.

Result:

* `tests/core/test_build_portfolio_intelligence.py` no longer imports `scripts.core.build_portfolio_intelligence`;
* `tests/portfolio/test_portfolio_source_contract.py` no longer imports `scripts.core.build_portfolio_intelligence`;
* the tested portfolio intelligence contract is preserved with test-local helpers and synthetic fixtures;
* the two tests were removed from the high-risk script-era blocker registry and are active in the full suite;
* `scripts/core/build_portfolio_intelligence.py` was not modified;
* no archive action was performed;
* Decision Engine, scanner/provider runtime, portfolio state, watchlist state, and scan validation runtime were untouched.

Validation:

* focused suite: `46 passed in 0.53s`
* full suite: `667 passed in 1.13s`

Full-suite count note:

* the total increased from the prior `628 passed` baseline because the two previously ignored portfolio-intelligence tests are active again and operator visibility now includes one additional decoupling guard.

Decision:

* `BL116_ARCHIVE_READINESS_REVIEW_FOR_DECOUPLED_PORTFOLIO_INTELLIGENCE_MODULE_APPROVED`


## BL116 — Archive-readiness review for decoupled portfolio intelligence module

Status: proposed

Context:
BL115 decoupled active portfolio intelligence tests from the script-era module:

- `scripts/core/build_portfolio_intelligence.py`

The following tests no longer import `scripts.core.build_portfolio_intelligence`:

- `tests/core/test_build_portfolio_intelligence.py`
- `tests/portfolio/test_portfolio_source_contract.py`

BL115 preserved the tested portfolio intelligence contract through test-local or canonical contract helpers.

Decision:
BL116 must be review-only.

Required outcome:
- verify no active imports from `src`, `tests`, or `.github` to `scripts.core.build_portfolio_intelligence`;
- inspect `scripts/core/build_portfolio_intelligence.py` for manual-run/write-risk markers;
- classify whether it is archive-ready, fail-close-required, or blocked;
- do not archive anything in BL116.


## BL117 — Fail-close decoupled portfolio intelligence module

Status: COMPLETED

Context:
BL116 reviewed archive-readiness for:

* `scripts/core/build_portfolio_intelligence.py`

BL116 confirmed that no active tests, `src`, or `.github` paths import `scripts.core.build_portfolio_intelligence`.

However, the module is not archive-ready because it still contains manual-run and write-risk markers:

* fixed `data/processed` paths;
* fixed `data/logs` paths;
* fixed portfolio source paths;
* `pd.read_csv(...)`;
* `mkdir(...)`;
* `to_csv(...)`;
* direct execution via `if __name__ == "__main__"`;
* no `FAIL_CLOSED` marker.

Decision:
BL117 must be a fail-close sprint, not an archive sprint.

Scope:

* `scripts/core/build_portfolio_intelligence.py`

Required outcome:

* manual execution fails closed;
* historical function body/content remains preserved;
* no archive action is performed;
* no runtime behavior is enhanced;
* Decision Engine remains untouched;
* portfolio state and watchlist state remain untouched;
* focused and full pytest suites remain green.

Result:

* `FAIL_CLOSED_MESSAGE` was added to `scripts/core/build_portfolio_intelligence.py`;
* the historical `build_portfolio_intelligence()` body was preserved as `_legacy_build_portfolio_intelligence_impl()`;
* public `build_portfolio_intelligence()` now raises `SystemExit(FAIL_CLOSED_MESSAGE)`;
* direct execution via `if __name__ == "__main__"` raises `SystemExit(FAIL_CLOSED_MESSAGE)`;
* no archive action was performed;
* no runtime behavior was enhanced;
* Decision Engine, scanner/provider runtime, portfolio state, watchlist state, scan validation runtime, trade command parser, reports, Telegram, and credentials were untouched.

Validation:

* focused suite: `46 passed in 0.51s`
* full suite: `667 passed in 1.06s`
* direct execution safety check: exited non-zero with the fail-closed message and no production writes.

Decision:

* `BL118_ARCHIVE_READINESS_REVIEW_FOR_FAIL_CLOSED_PORTFOLIO_INTELLIGENCE_MODULE_APPROVED`


## BL118 — Archive-readiness review for fail-closed portfolio intelligence module

Status: proposed

Context:
BL117 fail-closed the decoupled script-era module:

- `scripts/core/build_portfolio_intelligence.py`

BL115 had already removed active test imports from `scripts.core.build_portfolio_intelligence`.
BL116 confirmed the module was not archive-ready because manual/runtime execution and write-risk markers remained.
BL117 disabled public/manual execution while preserving the historical implementation body for audit purposes.

Decision:
BL118 must be review-only.

Required outcome:
- verify no active imports from `src`, `tests`, or `.github` to `scripts.core.build_portfolio_intelligence`;
- verify public/manual execution fails closed;
- verify historical implementation is preserved;
- verify no production data writes occur through direct execution;
- classify whether BL119 controlled archive is approved or blocked;
- do not archive anything in BL118.


## BL119 — Controlled archive of fail-closed portfolio intelligence module

Status: completed

Context:
BL118 reviewed archive-readiness for the fail-closed script-era module:

* `scripts/core/build_portfolio_intelligence.py`

BL115 had already removed active test imports from `scripts.core.build_portfolio_intelligence`.
BL116 confirmed the module was not archive-ready because manual/runtime execution and write-risk markers remained.
BL117 fail-closed the public/manual execution path while preserving the historical implementation body.
BL118 confirmed that the module is now archive-ready for a controlled archive sprint.

Decision:
BL119 must be a controlled archive sprint.

Scope:

* `scripts/core/build_portfolio_intelligence.py`

Required action:

* move `scripts/core/build_portfolio_intelligence.py` to `archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py` using `git mv`;
* preserve historical file content;
* do not modify the archived source body;
* do not archive any other module.

Required validation:

* verify `scripts/core/build_portfolio_intelligence.py` no longer exists;
* verify `archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py` exists;
* verify the archived file still contains `FAIL_CLOSED_MESSAGE`;
* verify the archived file still contains `_legacy_build_portfolio_intelligence_impl`;
* verify no active import from `src`, `tests`, or `.github` to `scripts.core.build_portfolio_intelligence`;
* run focused portfolio-intelligence/operator visibility tests;
* run full pytest suite.

Strict exclusions:

* no Decision Engine changes;
* no scanner/provider changes;
* no SEC/EDGAR calls;
* no yfinance calls;
* no credentials;
* no production data writes;
* no reports;
* no Telegram;
* no portfolio state changes;
* no watchlist state changes;
* no trade command parser changes;
* no portfolio command processing changes;
* no other `scripts/core/` archive actions.

Result:

* `scripts/core/build_portfolio_intelligence.py` was moved with `git mv` to `archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py`;
* the archived file content was not edited;
* the archived file still contains `FAIL_CLOSED_MESSAGE`;
* the archived file still contains `_legacy_build_portfolio_intelligence_impl()`;
* public/manual execution remains fail-closed in the archived file;
* no active import from `src`, `tests`, or `.github` references `scripts.core.build_portfolio_intelligence`;
* `scripts/core/build_portfolio_intelligence.py` is no longer present in the active `scripts/core/` tree;
* Decision Engine, scanner/provider runtime, SEC/EDGAR, yfinance, credentials, production data, reports, Telegram, portfolio state, watchlist state, scan validation runtime, trade command parser, portfolio command processing, and all other `scripts/core/` modules were untouched.

Validation:

* focused suite: `46 passed in 0.88s`
* full suite: `667 passed in 1.19s`

Decision:

* `BL120_REMAINING_ACTIVE_SCRIPTS_CORE_REVIEW_APPROVED`


## BL120 — Review remaining active scripts/core tree after portfolio intelligence archive

Status: proposed

Context:
BL119 archived the fail-closed portfolio intelligence module:

- `scripts/core/build_portfolio_intelligence.py`

The file was moved with `git mv` to:

- `archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py`

BL115 removed active test imports.
BL117 fail-closed public/manual execution.
BL118 approved controlled archive.

Decision:
BL120 must be review-only.

Required outcome:
- inventory the remaining active `scripts/core/` Python files;
- verify active `scripts.core` imports from `src`, `tests`, and `.github`;
- classify remaining modules by risk:
  - Decision Engine authority;
  - scanner/provider access;
  - log/write behavior;
  - validation/runtime behavior;
  - pure helper candidates;
- recommend the next cleanup sprint;
- do not archive anything in BL120.


## BL121 — Scanner/provider boundary review for remaining script-era core scanner modules

Status: proposed

Context:
BL120 reviewed the remaining active `scripts/core/` tree after BL119 archived the fail-closed portfolio intelligence module.

Remaining active `scripts/core/` files:

* `scripts/core/data_fetcher.py`
* `scripts/core/decision_engine.py`
* `scripts/core/indicators.py`
* `scripts/core/log_scans.py`
* `scripts/core/scanner.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`

BL120 confirmed that the only remaining active positive `scripts.core` import is:

* `tests/core/test_decision_engine.py` imports `scripts.core.decision_engine`

BL120 also confirmed that:

* `scripts/core/data_fetcher.py` contains yfinance/provider access;
* `scripts/core/scanner.py` contains yfinance/provider access;
* `scripts/core/indicators.py` appears scanner-adjacent and has no active test import or write-risk marker in the BL120 scan.

Decision:
BL121 must be a review-only scanner/provider boundary sprint.

Scope:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`
* `scripts/core/indicators.py`

Required outcome:

* classify whether each file is:

  * active scanner/provider dependency;
  * canonical metadata-only reference;
  * pure logic migration candidate;
  * fail-close candidate;
  * archive candidate;
  * blocked by scanner/provider policy;
* verify active imports from `src`, `tests`, `.github`, and `scripts`;
* inspect static yfinance/provider/network markers without executing provider code;
* inspect canonical scanner boundary references;
* recommend BL122 as either:

  * scanner/provider fail-close sprint;
  * scanner/provider test/canonical decoupling sprint;
  * controlled archive sprint for pure unused helper;
  * blocker/governance sprint.

Strict exclusions:

* no yfinance execution;
* no live provider calls;
* no SEC/EDGAR calls;
* no credentials;
* no production data writes;
* no scanner runtime behavior changes;
* no Decision Engine changes;
* no portfolio/watchlist changes;
* no report generation;
* no Telegram;
* no archive action in BL121.


## BL122 — Archive-readiness review for script-era indicators helper

Status: proposed

Context:
BL121 reviewed the scanner/provider-adjacent script-era core modules:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`
* `scripts/core/indicators.py`

BL121 found that:

* `scripts/core/data_fetcher.py` contains yfinance provider calls and is not archive-ready;
* `scripts/core/scanner.py` contains yfinance sector lookup plus scanner/scoring/trade-plan semantics and is not archive-ready;
* `scripts/core/indicators.py` appears to be a pure pandas helper with no active import, no provider/network marker, and no write-risk marker in the BL121 scan.

Decision:
BL122 must be a review-only archive-readiness sprint for the indicators helper.

Scope:

* `scripts/core/indicators.py`

Required outcome:

* verify active references from `src`, `tests`, `.github`, and `scripts`;
* verify whether `scripts.core.indicators` is imported anywhere;
* inspect static provider/network markers;
* inspect runtime/write markers;
* inspect function-level behavior;
* determine whether canonical scanner coverage already includes equivalent indicator behavior;
* decide whether BL123 should be:

  * controlled archive of `scripts/core/indicators.py`;
  * fail-close first;
  * canonical parity/test extraction first;
  * blocked by scanner policy.

Strict exclusions:

* no archive action in BL122;
* no yfinance execution;
* no live provider calls;
* no scanner runtime execution;
* no Decision Engine changes;
* no production data writes;
* no report generation;
* no Telegram;
* no portfolio/watchlist changes;
* no changes to `scripts/core/data_fetcher.py`;
* no changes to `scripts/core/scanner.py`;
* no runtime behavior changes.

## BL123 — Controlled archive of script-era indicators helper

Status: proposed

Context:
BL122 reviewed archive-readiness for:

* `scripts/core/indicators.py`

BL122 found that:

* no active import from `src`, `tests`, `.github`, or `scripts` references `scripts.core.indicators`;
* the only remaining active positive `scripts.core` import is `tests/core/test_decision_engine.py` importing `scripts.core.decision_engine`;
* `scripts/core/indicators.py` has no yfinance/provider/network markers;
* `scripts/core/indicators.py` has no runtime entrypoint;
* `scripts/core/indicators.py` has no production data/log write behavior;
* `scripts/core/indicators.py` is a pure pandas helper that computes moving averages, ATR14, 20-day high/low, and 20-day average volume.

Decision:
BL123 must be a controlled archive sprint.

Scope:

* `scripts/core/indicators.py`

Required action:

* move `scripts/core/indicators.py` to `archive/legacy_runtime/scripts/core/indicators.py` using `git mv`;
* preserve historical source content;
* do not modify the archived file body;
* do not archive any other module.

Required validation:

* verify `scripts/core/indicators.py` no longer exists;
* verify `archive/legacy_runtime/scripts/core/indicators.py` exists;
* verify no active import from `src`, `tests`, `.github`, or `scripts` references `scripts.core.indicators`;
* verify remaining active `scripts/core/` inventory;
* run operator visibility tests;
* run full pytest suite.

Strict exclusions:

* no changes to `scripts/core/data_fetcher.py`;
* no changes to `scripts/core/scanner.py`;
* no scanner/provider runtime changes;
* no yfinance execution;
* no live provider calls;
* no SEC/EDGAR calls;
* no credentials;
* no production data writes;
* no Decision Engine changes;
* no report generation;
* no Telegram;
* no portfolio/watchlist changes;
* no runtime behavior changes.

## BL124 — Review logging/validation/bootstrap core helpers after indicators archive

Status: proposed

Context:
BL123 archived the script-era indicators helper:

* `scripts/core/indicators.py`

to:

* `archive/legacy_runtime/scripts/core/indicators.py`

using `git mv`.

After BL123, the remaining active `scripts/core/` Python files are:

* `scripts/core/data_fetcher.py`
* `scripts/core/decision_engine.py`
* `scripts/core/log_scans.py`
* `scripts/core/scanner.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`

The scanner/provider modules remain blocked:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`

The Decision Engine remains P0 and out of scope:

* `scripts/core/decision_engine.py`

Decision:
BL124 must be a review-only sprint for the remaining logging/validation/bootstrap helpers.

Scope:

* `scripts/core/log_scans.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`

Required outcome:

* verify active imports from `src`, `tests`, `.github`, and `scripts`;
* inspect static read/write markers;
* inspect runtime entrypoints;
* inspect production data/log paths;
* classify each file as:

  * archive-ready;
  * fail-close required before archive;
  * canonical migration required;
  * blocked by validation/logging policy;
* recommend the next sprint.

Strict exclusions:

* no archive action in BL124;
* no runtime behavior changes;
* no production data writes;
* no scanner/provider changes;
* no yfinance execution;
* no live provider calls;
* no Decision Engine changes;
* no portfolio/watchlist changes;
* no report generation;
* no Telegram delivery.

## BL125 — Clean up logging/validation/bootstrap core helpers

Status: completed

Context:
BL124 reviewed the remaining logging/validation/bootstrap helpers under `scripts/core/`:

* `scripts/core/log_scans.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`

BL124 found that no active import from `src`, `tests`, `.github`, or `scripts` references:

* `scripts.core.log_scans`
* `scripts.core.validate_scans`
* `scripts.core.validator`

The only remaining active positive `scripts.core` import is still:

* `tests/core/test_decision_engine.py` importing `scripts.core.decision_engine`

BL124 classified:

* `scripts/core/log_scans.py` as `FAIL_CLOSE_REQUIRED_BEFORE_ARCHIVE`;
* `scripts/core/validate_scans.py` as `FAIL_CLOSE_REQUIRED_BEFORE_ARCHIVE`;
* `scripts/core/validator.py` as `CONTROLLED_ARCHIVE_APPROVED`.

Decision:
BL125 must be a cleanup execution sprint.

Scope:

* `scripts/core/log_scans.py`
* `scripts/core/validate_scans.py`
* `scripts/core/validator.py`

Required actions:

* fail-close `scripts/core/log_scans.py`;
* fail-close `scripts/core/validate_scans.py`;
* archive `scripts/core/validator.py` to `archive/legacy_runtime/scripts/core/validator.py` using `git mv`;
* preserve historical source bodies;
* do not modify the archived `validator.py` body.

Required validation:

* verify no active imports for the three target modules;
* verify direct execution of fail-closed modules exits with the fail-closed message;
* verify `scripts/core/validator.py` no longer exists;
* verify `archive/legacy_runtime/scripts/core/validator.py` exists;
* verify remaining active `scripts/core/` inventory;
* run operator visibility tests;
* run full pytest suite.

Strict exclusions:

* no scanner/provider runtime changes;
* no changes to `scripts/core/data_fetcher.py`;
* no changes to `scripts/core/scanner.py`;
* no Decision Engine changes;
* no yfinance execution;
* no live provider calls;
* no SEC/EDGAR calls;
* no credentials;
* no production data writes;
* no report generation;
* no Telegram;
* no portfolio/watchlist changes;
* no trade command parser changes.

Result:

* `scripts/core/log_scans.py` was fail-closed;
* the historical `log_scans()` body was preserved as `_legacy_log_scans_impl()`;
* public `log_scans()` now raises `SystemExit(FAIL_CLOSED_MESSAGE)`;
* direct execution of `scripts/core/log_scans.py` exits with the fail-closed message before writing to `data/logs/scans_log.csv`;
* `scripts/core/validate_scans.py` was fail-closed;
* the historical `validate_scans()` body was preserved as `_legacy_validate_scans_impl()`;
* the historical CLI body was preserved as `_legacy_main_impl()`;
* public `validate_scans()` and `main()` now raise `SystemExit(FAIL_CLOSED_MESSAGE)`;
* direct execution of `scripts/core/validate_scans.py` exits with the fail-closed message before writing to `data/processed/validation_results.csv`;
* `scripts/core/validator.py` was moved with `git mv` to `archive/legacy_runtime/scripts/core/validator.py`;
* the archived `validator.py` body was not modified;
* no active import references `scripts.core.log_scans`, `scripts.core.validate_scans`, or `scripts.core.validator`.

Validation:

* operator visibility: `8 passed in 0.02s`
* full suite: `667 passed in 1.16s`

Decision:

* `BL126_ARCHIVE_READINESS_REVIEW_FOR_FAIL_CLOSED_LOGGING_AND_VALIDATION_HELPERS_APPROVED`


## BL126 — Archive-readiness review for fail-closed logging and validation helpers

Status: proposed

Context:
BL125 fail-closed the decoupled script-era logging and validation helpers:

* `scripts/core/log_scans.py`
* `scripts/core/validate_scans.py`

BL125 also archived the bootstrap helper:

* `scripts/core/validator.py`

to:

* `archive/legacy_runtime/scripts/core/validator.py`

using `git mv`.

Decision:
BL126 must be review-only.

Required outcome:

* verify no active imports from `src`, `tests`, `.github`, or `scripts` reference `scripts.core.log_scans` or `scripts.core.validate_scans`;
* verify public/manual execution of both active files fails closed;
* verify historical implementation bodies remain preserved under private legacy implementation names;
* verify direct execution performs no production data writes;
* verify `scripts/core/validator.py` remains archived and absent from the active tree;
* classify whether a later controlled archive sprint is approved or blocked for the fail-closed logging and validation helpers;
* do not archive anything in BL126.


## BL127 — Review remaining scanner/provider core modules after logging validation archive

Status: proposed

Context:
BL126 archived the fail-closed logging/validation helpers:

* `scripts/core/log_scans.py`
* `scripts/core/validate_scans.py`

to:

* `archive/legacy_runtime/scripts/core/log_scans.py`
* `archive/legacy_runtime/scripts/core/validate_scans.py`

After BL126, active `scripts/core/` contains only:

* `scripts/core/data_fetcher.py`
* `scripts/core/decision_engine.py`
* `scripts/core/scanner.py`

The Decision Engine remains P0 and out of scope.

Decision:
BL127 must be a review-only sprint for the remaining scanner/provider core modules.

Scope:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`

Required outcome:

* verify active imports from `src`, `tests`, `.github`, and `scripts`;
* inspect yfinance/provider/network behavior;
* inspect scanner/scoring/trade-plan semantics;
* inspect runtime entrypoints and write-risk markers;
* compare against canonical scanner/provider boundary documentation;
* classify each file as:

  * fail-close required before archive;
  * canonical migration required;
  * blocked by provider/source policy;
  * not archive-ready;
* recommend the next sprint.

Strict exclusions:

* no archive action in BL127;
* no runtime behavior changes;
* no yfinance execution;
* no live provider calls;
* no SEC/EDGAR calls;
* no credentials;
* no production data writes;
* no Decision Engine changes;
* no report generation;
* no Telegram delivery;
* no portfolio/watchlist changes.


## BL128 — Define canonical scanner/provider migration path

Status: completed

Context:
BL127 reviewed the remaining scanner/provider core modules:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`

BL127 classified both modules as not archive-ready because provider/source-access behavior and scanner/scoring semantics remain.

Decision:
BL128 was documentation/governance/migration-planning only.

Result:

* approved a canonical migration path only;
* did not approve provider execution;
* did not approve yfinance execution;
* did not approve source-access implementation;
* did not approve archiving;
* did not approve fail-closing;
* kept `scripts/core/decision_engine.py` out of scope;
* confirmed active `scripts/core` inventory remains:
  * `scripts/core/data_fetcher.py`
  * `scripts/core/decision_engine.py`
  * `scripts/core/scanner.py`

Migration lanes:

* provider/source-access lane for `scripts/core/data_fetcher.py`;
* scanner semantics lane for `scripts/core/scanner.py`;
* canonical scanner boundary lane;
* archive-readiness lane.

Safety:

* no live provider calls were run;
* no yfinance calls were run;
* no SEC/EDGAR calls were run;
* no production data writes were performed;
* no report generation was performed;
* no Telegram delivery was performed;
* no portfolio/watchlist state was changed;
* no Decision Engine behavior was changed.


## BL129 — Establish canonical scanner semantics contracts before script-era scanner migration

Status: completed

Context:
BL128 approved a migration path only for the remaining script-era scanner/provider modules:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`

BL128 did not approve implementation, archive, fail-close, provider execution, yfinance execution, source-access implementation, production writes, tests, or Python code changes.

Decision:
BL129 must be documentation/test-contract planning only unless later explicitly approved.

Required scope:

* identify pure scanner semantics to contract-test;
* define planned contract coverage for setup classification, score components, liquidity state, discovery state, `rank_setups`, A/B/C grading, and trade-plan fields;
* keep provider/source-access behavior disconnected;
* do not execute providers;
* do not execute yfinance;
* do not move files to archive;
* do not fail-close active files;
* do not change runtime behavior;
* do not change Decision Engine behavior.

Result:

* documented the current canonical scanner boundary as planning-only and not yet a semantic scanner implementation;
* documented that existing canonical scanner tests prove boundary and side-effect guarantees only;
* inventoried script-era scanner semantics from `scripts/core/scanner.py`;
* defined planned future scanner input, classification, scoring, trade-plan, ranking/grading, and side-effect safety contract families;
* documented the migration gate before `scripts/core/scanner.py` can be migrated, archived, or fail-closed;
* approved scanner semantics contract planning only;
* did not approve implementation, tests, Python code changes, provider execution, yfinance execution, archive, or fail-close;
* kept `scripts/core/data_fetcher.py` provider/source-access lane separate;
* kept `scripts/core/decision_engine.py` out of scope.

Safety:

* no live provider calls were run;
* no yfinance calls were run;
* no SEC/EDGAR calls were run;
* no production data writes were performed;
* no report generation was performed;
* no Telegram delivery was performed;
* no portfolio/watchlist state was changed;
* no Decision Engine behavior was changed;
* no Python code was changed;
* no tests were changed.


## BL130 — Implement canonical scanner semantics contract tests with synthetic inputs

Status: proposed

Context:
BL129 established the scanner semantics contract plan for future migration of `scripts/core/scanner.py`.

BL129 did not approve implementation, tests, Python code changes, provider execution, yfinance execution, archive, fail-close, runtime behavior changes, or Decision Engine changes.

Decision:
BL130 requires explicit approval before implementation.

Required scope:

* implement canonical scanner semantics contract tests only after explicit approval;
* use synthetic inputs only;
* cover scanner input, classification, score component, trade-plan, ranking/grading, and side-effect safety contract families;
* prohibit provider execution and yfinance;
* prohibit archive moves;
* prohibit fail-close changes;
* prohibit runtime behavior changes unless separately approved;
* keep Decision Engine out of scope.
