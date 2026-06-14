# Roadmap

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines the v2 reset roadmap. It replaces operational feature sequencing as the current planning baseline until the rebuild reaches an approved cutover point.

## Roadmap Principle

The project should move from certified knowledge to clean implementation through explicit stages. Each stage must remain narrow enough to validate safely.

## Reset Roadmap

| Stage | Purpose | Allowed changes | Forbidden changes | Success criteria |
|---|---|---|---|---|
| RESET-0 | Full repository knowledge extraction and rebuild decision | One reset decision document | Runtime changes, code, tests, data, archive/delete | Decision recorded |
| RESET-1 | Canonical documentation rewrite | New or updated canonical docs | Code, tests, data, workflows, archive/delete | v2 source-of-truth docs exist |
| RESET-2 | Repository structure and archive plan | Classification and migration plan | Actual delete/move unless explicitly approved | Owner-approved structure/cutover plan |
| RESET-3 | V2 codebase bootstrap | New package skeleton and minimal smoke tests | Old code reuse as v2 base | Clean importable v2 skeleton |
| RESET-4 | V2 data contracts and fixtures | Contracts and approved fixtures | Live provider calls, generated-output dependence | Approved deterministic fixture strategy |
| RESET-5 | V2 minimal pipeline core | Discovery and classification skeleton | Decision/reporting/SEC complexity | Deterministic contract-tested core |
| RESET-6 | V2 Decision Engine | New Decision Engine implementation | Allocation outside Decision Engine | Final-action authority isolated |
| RESET-7 | V2 Reporting | Communication-only reporting | Reporting decision logic | Reporting preserves source decisions |
| RESET-8 | V2 fundamentals / SEC reintroduction | Source-data review model and approved integration | Live SEC tests, premature pipeline integration | Review evidence stable and local-first |
| RESET-9 | Legacy archive/delete cutover | Approved archive/delete/move actions | Deleting unextracted knowledge | Legacy surfaces retired safely |

## Current Recommended Next Action

After RESET-1 is reviewed and merged, proceed to RESET-2.

## Paused Work

- New feature work on the old active architecture.
- Standalone SEC-7F implementation.
- Legacy cleanup/delete/archive actions.
- v2 code implementation before contracts and structure are approved.

## Future Feature Reintroduction

Operational visibility, Telegram UX, prediction tracking, historical learning, and SEC/fundamentals should return only after the v2 core structure is stable.
