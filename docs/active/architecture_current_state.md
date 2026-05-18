# Current-State Architecture

Status: ACTIVE

This document is the current-state architecture source of truth for the market-scanner operational phase.

## Architecture Phase

Certified Sprints 0 through 8 completed the architecture purification phase. The system now operates as a classification-first, deterministic institutional pipeline with a single downstream allocation authority.

The project is transitioning from architecture purification to operational intelligence platform evolution.

## Certified Doctrine

The following principles are mandatory and remain unchanged:

- classification upstream
- allocation downstream
- Decision Engine = only allocation authority
- reporting communicates only
- no hidden filtering
- no upstream tradeability
- deterministic architecture
- row preservation
- auditability
- governance-first engineering
- separation of concerns

## Active Pipeline

The certified runtime architecture is:

```text
scanner
  -> validation_layer
  -> context_layer
  -> fundamental_layer
  -> timing_state_layer
  -> portfolio_intelligence_layer
  -> decision_engine
  -> reporting
```

Legacy watchlist and portfolio workflows may exist as supporting inputs or compatibility surfaces, but final allocation authority remains downstream in the Decision Engine.

## Layer Responsibilities

| Layer | Allowed responsibility | Forbidden responsibility | Authoritative downstream relationship |
|---|---|---|---|
| Scanner | Discover and preserve opportunity rows | Hidden filtering, tradeability, allocation, conviction, final action | Feeds classification layers without allocation authority |
| Validation | Classify structure and data coherence | Allocation eligibility, execution quality, urgency, final action | Provides structure metadata for downstream classification and Decision Engine interpretation |
| Context | Classify leadership and relative context | Blocking opportunities, tradeability, allocation simulation | Provides context metadata only |
| Fundamental | Classify quality metadata | Ranking, allocation priority, final action, conviction | Provides quality metadata only |
| Timing State | Classify timing condition and setup state | Execution gating, urgency, tradeability, allocation readiness | Provides timing metadata only |
| Portfolio Intelligence | Describe portfolio presence, exposure, and portfolio context | BUY/SELL decisions, allocation gating, opportunity destruction | Provides descriptive portfolio metadata to the Decision Engine |
| Decision Engine | Allocate, arbitrate, and emit final decision semantics | Upstream classification mutation, hidden filtering | Sole authority for allocation, execution, arbitration, and final action outputs |
| Reporting | Communicate Decision Engine outputs | Reinterpreting urgency, prioritizing allocation, injecting decision logic | Presents source decisions without changing them |

Upstream layers classify only. Portfolio Intelligence is descriptive. The Decision Engine is the only allocation authority. Reporting communicates only.

### Scanner

The scanner identifies and preserves opportunities. It must not silently remove valid opportunity rows through hidden strategic filtering.

### Validation Layer

The validation layer classifies structural setup quality. It emits descriptive structure classification only.

It must not determine tradeability, allocation eligibility, urgency, conviction, or final action.

### Context Layer

The context layer classifies market, sector, leadership, and relative-strength context.

It must not act as an allocation gate.

### Fundamental Layer

The fundamental layer enriches opportunities with descriptive fundamental quality classification.

It must not rank opportunities for allocation or produce final action semantics.

### Timing State Layer

The timing state layer describes timing condition and setup state.

It must remain descriptive, non-mutating, and distribution-preserving.

### Portfolio Intelligence Layer

The portfolio intelligence layer adds descriptive portfolio state. It may read portfolio inputs but must not perform allocation.

### Decision Engine

The Decision Engine is the only allocation, execution, arbitration, and final action authority.

It owns final decision semantics and produces final decision outputs.

No upstream layer may pre-empt, override, or duplicate Decision Engine authority.

### Reporting Layer

Reporting communicates Decision Engine outputs. It may group, format, truncate, and summarize for presentation, but it must not create allocation logic, alter source decisions, or imply independent decision authority.

## Authority Boundary

The authority boundary is simple:

```text
Upstream layers classify.
Decision Engine allocates.
Reporting communicates.
```

## Runtime Preservation Rules

All operational changes must preserve:

- deterministic outputs
- row identity where contractually required
- source row traceability
- explicit data contracts
- fail-fast behavior for invalid required inputs
- separation of classification and allocation
- reporting neutrality

## Prohibited Architecture Regressions

The following are forbidden unless a formal Level 3 governance review explicitly approves an architecture change:

- allocation logic outside the Decision Engine
- BUY, SELL, HOLD, TRIM, or REVIEW authority outside the Decision Engine
- hidden filtering in upstream layers
- tradeability semantics upstream
- reporting-based decision modification
- runtime output contract changes without architecture review
- duplicated final-action semantics outside the Decision Engine

## Operational Evolution Scope

Future development should focus on operational intelligence platform evolution, including:

- runtime reliability
- orchestration
- GitHub Actions recovery
- operational visibility
- portfolio intelligence
- historical decision storage
- self-analysis
- prediction tracking
- feedback loops
- reporting redesign
- production readiness

These changes must evolve the certified architecture without weakening its authority boundaries.
