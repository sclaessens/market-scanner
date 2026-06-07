# Pipeline Contract

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines the v2 pipeline contract at the level required before implementation.

## Contract Doctrine

- Layers classify before allocation.
- Row identity must be preserved where the contract requires it.
- Generated outputs are not source inputs unless explicitly reclassified.
- Each layer owns only its allowed semantics.
- Decision Engine is the only final-action authority.
- Reporting communicates only.

## V2 Pipeline Contract Map

| Stage | Input | Output | Allowed semantics | Forbidden semantics |
|---|---|---|---|---|
| Input contracts | Approved manual/source inputs | validated input records | schema and readiness validation | investment meaning or allocation |
| Discovery | validated universe/source records | opportunity candidates | opportunity discovery | hidden filtering, tradeability |
| Validation | opportunity candidates | structure classifications | structural validity and data coherence | allocation eligibility, urgency |
| Context | validated candidates | context classifications | leadership and relative context | blocking, ranking, allocation |
| Fundamentals/source readiness | context rows and approved source data | readiness and quality metadata | source-data readiness and approved metric states | final action, conviction, tradeability |
| Timing | classified rows | timing-state metadata | setup condition and timing description | execution gating, urgency |
| Portfolio context | timing rows and portfolio input | descriptive portfolio context | exposure, presence, provenance | BUY/SELL, allocation gating |
| Decision Engine | fully classified rows | final decisions | allocation, execution, arbitration, final action | upstream mutation |
| Reporting | final decisions | communication artifacts | formatting, grouping, summarizing, traceability | decision logic, suppression, override |

## Row Preservation

A v2 layer must state whether it preserves row count, row identity, order, and source traceability. Any layer that intentionally changes row count must document why and how the change remains auditable.

## Freshness

A normal v2 run must rebuild required upstream generated artifacts before downstream decision and reporting artifacts are produced. A stale artifact must not silently drive final decisions.

## Failure Policy

Required missing inputs should fail clearly. Optional missing inputs should produce explicit readiness or unavailable states, not hidden defaults.

## Forbidden Terms Outside Decision Engine

Outside the Decision Engine, v2 code and contracts must not introduce:

- tradeable;
- conviction;
- BUY logic;
- SELL logic;
- REMOVE logic;
- allocation gating;
- execution urgency;
- hidden filtering;
- final action semantics.
