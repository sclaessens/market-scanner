# Decision Engine Contract

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines the canonical v2 authority boundary for final decisions.

## Core Rule

The Decision Engine is the only allocation, execution, arbitration, and final-action authority.

No upstream layer, report, source-data process, research artifact, role, or documentation artifact may create final decision semantics.

## Inputs

The v2 Decision Engine may consume only approved descriptive inputs:

- opportunity identity;
- validation classifications;
- context classifications;
- source-data readiness and approved metric metadata;
- timing-state metadata;
- portfolio context;
- explicit provenance and freshness metadata.

## Outputs

The v2 Decision Engine should emit:

- final action;
- rationale;
- decision state;
- required review state where applicable;
- provenance references;
- input freshness state;
- audit fields required by the final contract.

Exact fields must be defined in RESET-6 before implementation.

## Forbidden Behavior

The Decision Engine must not:

- mutate upstream input evidence;
- silently drop input rows without an approved contract;
- depend on reporting behavior;
- use generated artifacts as source inputs unless approved;
- infer unavailable source-data values as zero;
- hide unresolved source-data review behind final confidence.

## Authority Protection

Any change that affects final action semantics, allocation rules, execution semantics, or arbitration is governance-critical and requires explicit review before implementation.

## Testing Expectations

v2 tests must prove that:

- upstream layers do not allocate;
- Decision Engine preserves required row identity;
- final decisions are deterministic for identical inputs;
- Reporting cannot alter final decisions;
- source-data insufficiency remains explicit.
