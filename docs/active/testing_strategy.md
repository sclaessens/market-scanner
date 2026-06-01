# Testing Strategy

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines the v2 testing strategy before implementation begins.

## Core Principle

Tests should validate contracts, authority boundaries, determinism, row preservation, and source-data readiness behavior. Tests should not force v2 to preserve old paths, wrappers, accidental schemas, or generated-output quirks.

## Test Classes

| Test class | Purpose |
|---|---|
| Contract tests | Validate layer contracts and authority boundaries |
| Unit tests | Validate isolated functions and modules |
| Integration tests | Validate approved pipeline slices |
| Fixture tests | Validate approved deterministic data examples |
| Governance tests | Prevent hidden allocation, upstream tradeability, reporting decision logic, and generated-output-as-source drift |
| Source-data tests | Validate provenance, missing values, conflicts, and review-required states |

## Fixture Policy

Fixtures must be:

- small;
- deterministic;
- approved;
- tracked intentionally;
- independent from current generated runtime outputs unless explicitly selected;
- designed around contracts rather than old implementation details.

## Old Test Policy

Old tests are reference material. Their concepts may be preserved, but v2 tests should be newly written.

Old tests must not force:

- old file paths;
- compatibility wrappers;
- old CSV schemas;
- generated-output dependence;
- legacy import structure;
- old reporting artifacts;
- old SEC diagnostics.

## Required V2 Test Themes

v2 must test:

- no allocation outside Decision Engine;
- no upstream tradeability;
- no hidden filtering;
- Reporting communicates only;
- deterministic outputs;
- explicit failure on missing required inputs;
- row identity and traceability where required;
- generated outputs are not inputs unless approved;
- SEC/source-data missing values are not zero;
- source-data review-required states remain explicit.

## CI Policy

CI should be rewritten after v2 structure exists. RESET-1 does not modify workflows.
