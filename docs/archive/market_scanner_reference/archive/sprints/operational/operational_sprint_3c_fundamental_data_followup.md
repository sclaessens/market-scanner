# Operational Sprint 3C Fundamental Data Follow-up

## 1. Status

Status: INVESTIGATION FOLLOW-UP

Operational Sprint 3C is not closed, not certified complete, and not authorized for implementation by this document.

This document records the governance finding that the current pipeline now runs fresh end-to-end, but the Fundamental Layer still has no approved real fundamental data source. The resulting all-`REVIEW` Decision Engine output is expected current behavior, not a path bug.

No runtime implementation, provider integration, schema change, test change, generated artifact update, CSV edit, or Decision Engine loosening is authorized by this document.

## 2. Investigation Context

After OS3A, BL-0011, and OS3B, the operator pipeline now runs fresh end-to-end through the governed operational sequence:

```text
scanner -> validation -> context -> fundamental -> timing -> portfolio state -> portfolio review -> portfolio intelligence -> final decisions -> reporting -> Telegram delivery
```

The current flow is technically correct. However, final decisions remain `REVIEW` because the Fundamental Layer emits missing-data fallback rows rather than approved real fundamental quality classifications.

## 3. Investigation Finding

The current `data/processed/fundamental_quality.csv` investigation state is:

- row count: 291
- all rows have `quality_state = INSUFFICIENT_DATA`
- all rows have `quality_metadata_status = source_missing`
- all rows have `source_data_status = source_missing`
- `missing_fundamentals_count = 291`
- `partial_data_count = 0`
- `stale_data_count = 0`

The current `build_fundamental_layer.py` behavior reads `data/processed/context_strength.csv` and intentionally emits fallback rows when approved source data is missing.

There is currently:

- no configured raw fundamentals artifact
- no approved provider integration
- no current real-data classification contract
- no governed quality metadata source that can support allocation decisions downstream

## 4. Root Cause

The root cause is not stale orchestration, row loss, Telegram formatting, Reporting behavior, or a Decision Engine path bug.

The root cause is that the Fundamental Layer contract currently supports source-missing fallback classification only. The layer has no approved real fundamental data source and no governed real-data classification contract.

As a result, all opportunities reach the Decision Engine with missing fundamental quality metadata. The Decision Engine correctly keeps those rows in `REVIEW` instead of authorizing allocation decisions without approved quality data.

## 5. Current Fundamental Layer Contract

The current Fundamental Layer remains classification and enrichment only.

It may:

- read the governed upstream context artifact
- preserve the input opportunity universe
- emit one descriptive fundamental-quality row per upstream row
- classify missing or insufficient fundamental data
- expose quality metadata describing source availability
- remain deterministic and auditable

It must not:

- allocate capital
- create trade actions
- rank opportunities
- score tradeability
- express urgency or conviction
- filter opportunities from the universe
- bypass the Decision Engine
- loosen Decision Engine behavior
- infer approved quality from absent data

The current fallback output is therefore governance-safe but operationally incomplete.

## 6. Test Coverage Gap

Existing tests cover fallback and missing-data behavior. That coverage is useful and should remain.

The current test gap is real-data classification coverage. There is no approved real-data fixture, provider contract, raw fundamentals artifact contract, freshness contract, partial-data classification contract, stale-data classification contract, or test suite proving how real fundamental inputs should become descriptive quality classifications.

This gap should not be solved by weakening missing-data behavior. It should be solved by a governed design and implementation plan for approved fundamental data integration.

## 7. Governance Classification for Possible Fixes

Potential fixes are governance-sensitive because fundamental quality can affect Decision Engine outcomes.

Allowed direction under future governed planning:

- define an approved raw fundamentals artifact or provider integration
- define descriptive quality states and metadata fields
- define source availability, partial-data, stale-data, and insufficient-data rules
- define deterministic row identity and freshness behavior
- define fixtures and tests for real-data classification
- preserve upstream classification-only semantics
- keep allocation authority inside the Decision Engine

Forbidden direction:

- loosen the Decision Engine because fundamentals are missing
- permit allocation decisions without approved fundamental quality data
- add upstream ranking, scoring, tradeability, urgency, or conviction semantics
- introduce hidden filtering
- allow Reporting or Telegram to compensate for missing quality data
- infer real quality from absent source data
- treat provider availability as implementation authorization without a Level 2 plan

Governance classification: Level 2 design and implementation planning is required before any implementation. Escalate to Level 3 if a proposed solution changes allocation authority, Decision Engine semantics, ranking authority, tradeability semantics, or row-universe authority.

## 8. Recommended Next Step

Create a governed Level 2 design and implementation plan for fundamental data source integration.

The plan should define:

1. the approved fundamental data source or raw artifact
2. the Fundamental Layer real-data classification contract
3. source freshness and missing/partial/stale-data rules
4. required schema and metadata fields
5. deterministic row identity and preservation requirements
6. test fixtures for real-data, partial-data, stale-data, and source-missing scenarios
7. governance checks preventing ranking, tradeability, urgency, conviction, allocation, hidden filtering, or Decision Engine bypass

Do not loosen the Decision Engine. Do not authorize allocation decisions without approved fundamental quality data.

## 9. Backlog Impact Assessment

New backlog item identified and added to `docs/sprints/project_backlog.md`:

- BL-0015 — Define and implement approved Fundamental data source and quality classification contract

Backlog impact assessment:
- New backlog items identified and added to project_backlog.md

## 10. Sprint Status Recommendation

Do not mark any sprint closed or certified complete based on this investigation follow-up.

Recommended current interpretation:

- flow status: technically correct
- final-decision status: correctly constrained to `REVIEW`
- root cause: no approved real Fundamental data source or classification contract
- governance status: Level 2 design required before implementation
- implementation status: not authorized by this document

This document captures the finding only. It does not change runtime behavior.