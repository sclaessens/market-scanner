# Technical Architecture

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines the canonical v2 architecture direction. It is a documentation contract, not an implementation plan.

## Architecture Doctrine

```text
classification upstream
allocation downstream
reporting communicates
```

The Decision Engine is the only layer allowed to produce final allocation, execution, arbitration, and final-action semantics.

## Proposed V2 Pipeline

```text
input contracts
-> discovery
-> validation classification
-> context classification
-> source-data readiness and fundamentals classification
-> timing classification
-> portfolio context
-> Decision Engine
-> reporting
-> optional research feedback loop
```

## Layer Responsibilities

| Layer | Allowed responsibility | Forbidden responsibility |
|---|---|---|
| Input contracts | Validate approved manual/source inputs | Infer missing business meaning |
| Discovery | Preserve candidate opportunities | Hidden filtering or tradeability |
| Validation | Classify structure and data coherence | Allocation eligibility or urgency |
| Context | Classify leadership and market/sector context | Blocking or ranking opportunities |
| Fundamentals/source data | Classify readiness, provenance, and approved metrics | Treat review output as investment quality by default |
| Timing | Describe timing condition and setup state | Execution gating or tradeability |
| Portfolio context | Describe holdings, exposure, and provenance | BUY/SELL authority |
| Decision Engine | Allocate and emit final decision semantics | Mutate upstream evidence |
| Reporting | Communicate decisions and traceability | Override, prioritize, or suppress decisions |
| Research feedback loop | Analyze historical outcomes | Feed allocation without governance |

## V2 Code Rule

Old Python files are reference-only. v2 code must be newly written from v2 contracts. Useful old logic may be studied, but not used as the active v2 base.

## Compatibility Rule

v2 should avoid permanent compatibility wrappers. Temporary cutover adapters may be approved only when they have an expiration path.

## Runtime Determinism

For the same approved inputs, v2 must produce deterministic outputs. Any external provider, timestamp, ordering rule, or cache must be explicit and testable.

## Contract-First Implementation

No v2 implementation stage may begin until its input/output contracts and fixture strategy are defined.
