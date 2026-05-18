# Simplified Sprint Lifecycle

Status: ACTIVE

This document defines the operational sprint lifecycle for the post-purification phase.

## Background

The previous lifecycle was intentionally heavy because the project was performing architecture purification, governance migration, and institutional stabilization.

That phase is complete. Sprints 0 through 8 are certified complete.

Future work should use a lighter operational lifecycle unless an architecture review trigger is activated.

## Standard Operational Sprint

A standard operational sprint requires four artifacts or sections:

1. Objective
2. Scope
3. Implementation summary
4. Validation summary

These may live in one concise sprint note, issue, PR description, or release note.

## Required Sprint Content

### Objective

Define the operational problem being solved.

### Scope

State what is included and explicitly what is not included.

### Implementation Summary

Summarize changed files, behavior, contracts, and operational impact.

### Validation Summary

Summarize tests, checks, manual validation, or reasoned non-runtime validation.

## Optional Architecture Review

An architecture review is required only when Governance v2 triggers are met.

Examples:

- data contract changes
- pipeline sequencing changes
- Decision Engine authority impact
- reporting neutrality impact
- allocation semantics impact
- hidden filtering risk
- row preservation risk

## Optional Formal Audit

A formal audit is reserved for Governance Level 3 changes.

It is not required for ordinary documentation cleanup, orchestration work, test improvements, operational runbooks, or presentation-only reporting changes.

## Sprint Closeout v2

A sprint is closed when:

- implementation is complete
- required validation has passed or exceptions are documented
- architecture safety check is answered
- backlog impact is recorded if applicable
- PR or merge summary explains operational impact

No separate closeout document is required unless the sprint is Governance Level 3.

## Lightweight Sprint Template

```markdown
# Sprint: <name>

## Objective

## Scope

### In Scope

### Out of Scope

## Governance Level

Level 1 / Level 2 / Level 3

## Architecture Safety Check

- Allocation authority changed: no
- Upstream tradeability introduced: no
- Hidden filtering introduced: no
- Reporting semantics changed: no
- Runtime contracts changed: no

## Implementation Summary

## Validation Summary

## Backlog Impact
```

## Principle

Governance must protect the architecture while preserving delivery speed. The default path is lightweight. Escalation is triggered by risk, not ceremony.