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

## Relationship to Existing Backlog

The historical `docs/sprints/project_backlog.md` remains preserved as legacy planning evidence until RESET-2 decides how to reconcile or archive it. This document is the v2 reset-facing backlog baseline.
