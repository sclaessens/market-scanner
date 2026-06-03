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

Status: CANDIDATE NEXT STAGE

Proposed next step: Define the local-only manual execution procedure, required parameters, credential handling, no-write policy, and review checklist for one approved source path.

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

## Relationship to Existing Backlog

The historical `docs/sprints/project_backlog.md` remains preserved as legacy planning evidence until RESET-2 decides how to reconcile or archive it. This document is the v2 reset-facing backlog baseline.
