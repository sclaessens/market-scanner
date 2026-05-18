# Operational Development Model

Status: ACTIVE

This document defines how development should proceed in the operational intelligence platform phase.

## Phase Definition

The repository has completed architecture purification and is now focused on operational evolution.

Operational evolution means improving reliability, visibility, maintainability, orchestration, historical learning, and production readiness without weakening certified architecture boundaries.

## Development Priorities

Future work should prioritize:

- runtime reliability
- orchestration stability
- GitHub Actions recovery
- operational visibility
- portfolio intelligence improvements
- historical decision storage
- self-analysis
- prediction tracking
- feedback loops
- reporting usability
- production readiness

## Non-Goals

Operational development must not casually redesign:

- allocation authority
- Decision Engine semantics
- upstream classification doctrine
- reporting neutrality
- runtime output contracts
- strategy logic

## Default Workflow

1. Define the operational problem.
2. Identify the governance level.
3. Confirm architecture boundaries are unchanged.
4. Implement the smallest maintainable change.
5. Validate with tests or explicit non-runtime checks.
6. Summarize operational impact in the PR.
7. Update active documentation only when the source of truth changes.

## Documentation Discipline

Do not create new doctrine files unless they become active sources of truth.

Prefer updating existing active documents over creating parallel governance documents.

Use reference documents for rationale and archive documents for history.

## Engineering Principle

The architecture is certified. Future work should make it easier to operate, observe, test, recover, and evolve.