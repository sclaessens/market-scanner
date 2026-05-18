# Sprint 4 Governance Constraints — Timing State Layer

## 1. Constraint Status

Status: READY FOR SPRINT 4 GOVERNANCE AUDIT

This document is a governance-preparation artifact only. It does not authorize execution planning, developer specification, implementation, runtime changes, tests, generated data changes, threshold tuning, strategy optimization, or Decision Engine changes.

## 2. Binding Doctrine

Sprint 4 inherits the certified doctrine from Sprint 0 through Sprint 3:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority
- upstream layers classify only
- reporting communicates only
- no upstream tradeability
- no hidden filtering
- no hidden allocation semantics outside Decision Engine
- no decision semantics outside Decision Engine
- no ranking authority outside Decision Engine
- no scoring authority outside Decision Engine
- distribution preservation is mandatory

## 3. Timing State Layer Constraint

The Timing State Layer may describe timing conditions only.

It may not determine, imply, or simulate:

- tradeability
- actionability
- allocation
- conviction
- urgency
- execution readiness
- priority
- ranking
- score authority
- recommendation
- opportunity quality
- expected return
- expected alpha
- portfolio desirability

Timing metadata is information, not instruction.

## 4. Forbidden Output Language

Forbidden field or value language includes:

- `tradeable`
- `approved`
- `rejected`
- `high_conviction`
- `conviction_score`
- `priority`
- `actionable`
- `execution_ready`
- `best_opportunity`
- `buy_candidate`
- `sell_candidate`
- `ranking_score`
- `timing_score`
- `final_score`
- `allocation_weight`
- `expected_return`
- `alpha_score`
- `opportunity_rank`
- `preferred_setup`
- `BUY`
- `SELL`
- `REMOVE`
- `urgency`

Negative tests or governance documentation may mention forbidden terms only as prohibited examples.

## 5. Descriptive Metadata Constraint

Governance-safe timing metadata must be:

- descriptive
- deterministic
- non-mutating
- non-preferential
- non-allocative
- non-executory
- non-decisional

Examples of directionally acceptable descriptive concepts:

- pullback state
- breakout state
- consolidation state
- volatility contraction state
- extension state
- compression state
- momentum continuation state
- timing environment metadata
- timing pattern classification
- trend participation state

These concepts remain candidates only. They are not approved implementation schema.

## 6. Distribution Constraint

Timing must preserve:

- input row count
- ticker/date identity
- upstream ordering
- full upstream opportunity visibility
- upstream distribution shape

Timing may never:

- drop rows
- remove tickers
- filter stale, extended, weak, incomplete, or missing-data rows
- sort by timing condition
- collapse multiple opportunities into a preferred subset
- hide opportunities from downstream layers

## 7. Cross-Layer Constraint

Timing may not synthesize a cross-layer opportunity judgment from:

- Validation structure
- Context leadership
- Fundamental quality
- Portfolio state
- Decision Engine output

Timing may not produce composite intelligence such as:

- best opportunity
- preferred setup
- high-quality breakout
- leadership-confirmed timing
- quality-adjusted timing
- allocation-ready timing
- execution-ready timing

Composite interpretation belongs only to the Decision Engine.

## 8. Audit Constraint

The Sprint 4 governance audit must verify:

- this constraint document is aligned with `docs/sprints/sprint_4_timing_state_layer.md`
- forbidden semantics are absent from proposed schema direction except as prohibited examples
- Decision Engine authority remains exclusive
- no implementation authority is implied
- execution planning remains future work after audit certification

## 9. Scrum Master Recommendation

READY FOR SPRINT 4 GOVERNANCE AUDIT
