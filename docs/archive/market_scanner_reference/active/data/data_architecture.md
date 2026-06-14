# Data Architecture

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines the v2 data architecture principles before implementation begins.

## Data Classes

| Class | Meaning | Git policy |
|---|---|---|
| Manual input | Human-maintained approved input | Track only when approved |
| Source-data input | Raw or transformed external data with provenance | Track only if explicitly approved and sanitized |
| Fixture | Small deterministic test data | Track after approval |
| Generated output | Runtime product of scripts | Ignore unless explicitly approved |
| Log | Runtime evidence or diagnostics | Ignore unless explicitly approved |
| Local cache | Provider cache, SEC cache, raw dumps | Ignore |
| Report | Human communication output | Generated; ignore unless explicitly approved |

## Source-of-Truth Rule

Generated outputs must not become source-of-truth inputs unless explicitly reclassified by a data contract.

## Portfolio Data

Portfolio transactions may become a v2 manual input candidate. Positions, portfolio review, and portfolio intelligence outputs are generated unless reapproved.

## Processed Data

Processed CSVs from the old architecture are reference-only. They must not be used as v2 source-of-truth unless the v2 data contract explicitly approves them.

## Logs and Reports

Logs and reports are generated evidence. They may support debugging or audit review, but they must not drive v2 decisions directly.

## Fixtures

Fixtures must be small, deterministic, approved, and designed to validate contracts rather than reproduce old generated outputs.

## SEC and Provider Data

SEC caches, real provider data, and generated SEC review outputs remain local-only by default. Sanitized examples may become fixtures only after approval.

## Data Contract Requirements

Every v2 data artifact must define:

- owner;
- path;
- source;
- tracked or ignored status;
- required columns or fields;
- row identity expectations;
- freshness expectations;
- allowed semantics;
- forbidden semantics;
- validation rules;
- downstream consumers.
