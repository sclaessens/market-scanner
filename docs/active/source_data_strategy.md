# Source Data Strategy

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines how v2 should treat external, raw, transformed, and reviewed source data.

## Core Principle

Source-data readiness is not the same as investment quality. A source may be incomplete, noisy, stale, conflicting, or review-required without implying that the underlying company or opportunity is weak.

## Source-Data Lifecycle

```text
raw source evidence
-> provenance capture
-> source-data validation
-> source-data classification
-> approved metric or readiness output
-> downstream classification layer
```

## Approved Source Requirements

A source may feed v2 only when it has:

- known origin;
- known refresh behavior;
- clear license or access boundary;
- documented schema;
- deterministic transformation policy;
- missing-value policy;
- conflict policy;
- review-required policy;
- fixture strategy;
- owner approval.

## SEC/Fundamentals Strategy

SEC and fundamentals work should pause as standalone old-architecture implementation. SEC knowledge remains valuable and should return after the v2 core pipeline and data contracts exist.

The future SEC review model must distinguish:

- accepted facts;
- rejected or skipped facts;
- period-level evidence;
- row-level evidence;
- ticker-level evidence;
- run-level diagnostic summaries;
- selected review-mode outputs.

Filtering must happen after fact classification. Missing values are not zero. Conflicts remain review-required unless an approved deterministic policy exists.

## Review Modes

Future SEC/fundamentals review may support:

- broad local diagnostic review;
- recent annual review;
- explicitly requested mixed annual/quarterly review.

None of these modes are pipeline-ready by default.

## Local-Only Data

Real SEC cache files, provider dumps, broad review outputs, and run-level diagnostic summaries remain local-only unless explicitly approved and sanitized.

## Integration Rule

No source-data output enters the v2 pipeline until the Data Steward, Financial Analyst, Technical Analyst, and Governance Auditor confirm readiness under an approved contract.

## RESET-9C2C Contract Translation Boundary

RESET-9C2C establishes fundamentals/source-data contract metadata without reintroducing SEC or provider runtime behavior.

The translated contract boundary is:

- source-data readiness is a governance state only;
- raw source capture, normalized input, generated classification output, generated analysis output, and local-only review artifacts are separate dataset roles;
- missing, source-missing, row-missing, partial, stale, invalid, unavailable, and review-required states must remain explicit;
- missing numeric values must not be converted to zero;
- source availability must not imply investment quality;
- contract helpers must not read legacy generated CSVs, write files, import legacy `scripts`, call SEC/provider/network services, or create final decisions.

Formula knowledge, SEC fact selection, provider ingestion, raw-to-normalized transformation, and investment-quality analysis require later explicit approval.
