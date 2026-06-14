# Sprint 4 — Timing State Layer

## 1. Sprint Status

Status: READY FOR SPRINT 4 GOVERNANCE AUDIT

Sprint 4 is in governance preparation only. This document does not authorize implementation, runtime code changes, test changes, generated data changes, architecture redesign, strategy optimization, allocation logic, filtering logic, ranking logic, scoring authority, Decision Engine logic, or developer execution.

Sprint 4 may proceed only through the audit-first sequence:

1. governance preparation
2. governance audit
3. architecture validation
4. execution planning
5. developer specification
6. implementation

Execution planning, developer specification, and implementation must not begin from this document.

## 2. Governance Inheritance From Sprint 0 Through Sprint 3

Sprint 4 inherits the certified Sprint 0 doctrine:

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

Sprint 4 inherits Sprint 1 Validation certification:

- Validation Layer = structure classification only
- Validation may not invalidate extended momentum
- Validation may not simulate execution quality
- Validation may not determine allocation eligibility
- Validation output may not be reinterpreted by Timing as tradeability, urgency, conviction, ranking, scoring, or execution readiness

Sprint 4 inherits Sprint 2 Context certification:

- Context Layer = leadership and relative-strength classification only
- weak context is not rejection
- strong or leading context is not tradeability
- sector-relative context data is enrichment only
- Context output may not be reinterpreted by Timing as allocation, urgency, conviction, ranking, scoring, priority, or execution readiness

Sprint 4 inherits Sprint 3 Fundamental certification:

- Fundamental Layer = quality classification and enrichment only
- high quality is not tradeability
- low quality is not rejection
- missing fundamental data is not removal
- Fundamental output may not be reinterpreted by Timing as allocation, urgency, conviction, ranking, scoring, priority, or execution readiness

## 3. Sprint Objective

Sprint 4 prepares the Timing State Layer as a pure timing-condition classification and enrichment layer.

The objective is to define how timing and technical state metadata can exist without introducing:

- allocation logic
- filtering-first behavior
- tradeability semantics
- urgency semantics
- conviction semantics
- execution semantics
- hidden filtering
- Decision Engine leakage
- ranking authority
- scoring authority
- opportunity suppression
- composite opportunity intelligence
- multi-factor opportunity synthesis
- composite scoring
- signal aggregation authority
- cross-layer weighted interpretation

Core Sprint 4 doctrine:

The Timing State Layer exists to preserve informational richness without creating execution authority.

## 4. Timing State Layer Responsibilities

The Timing State Layer may classify and enrich timing conditions only.

Allowed responsibilities:

- classify pullback state
- classify breakout state
- classify consolidation state
- classify volatility contraction state
- classify extension state
- classify compression state
- classify momentum continuation state
- classify timing structure observations
- classify timing environment metadata
- classify timing pattern state
- classify trend participation state
- append descriptive timing metadata
- log timing-state distributions

The layer must remain descriptive, deterministic, non-mutating, and non-decisional.

## 5. Explicitly Forbidden Timing State Responsibilities

The Timing State Layer may not determine:

- tradeability
- allocation eligibility
- conviction
- urgency
- actionability
- execution readiness
- final action
- portfolio action
- BUY/SELL/HOLD/TRIM/REMOVE behavior
- opportunity priority
- opportunity rank
- opportunity score
- capital weight
- approval or rejection
- expected return
- expected alpha
- portfolio desirability

The Timing State Layer may not:

- remove opportunities
- suppress opportunities
- reorder opportunities
- prioritize opportunities
- narrow opportunity distribution
- become a hard gate
- become a hidden filter
- override scanner output
- override Validation output
- override Context output
- override Fundamental output
- create portfolio semantics
- create recommendation semantics
- create execution preference semantics
- simulate Decision Engine behavior

## 6. Classification-Only Doctrine

Timing classification means describing observed timing conditions only.

Allowed:

- descriptive state labels
- structural timing observations
- extension and compression classifications
- participation classifications
- pattern classifications
- metadata availability status
- missing-data status
- deterministic audit logging

Forbidden:

- trade filtering
- execution filtering
- allocation filtering
- ranking authority
- scoring authority
- signal aggregation authority
- final-action logic
- conviction logic
- urgency logic
- BUY/SELL/HOLD/TRIM/REMOVE behavior
- recommendation logic
- expected-performance logic

Extended means extended timing condition only. It does not mean invalid.

Compressed means compressed timing condition only. It does not mean preferred.

Pullback means pullback timing condition only. It does not mean ready.

Breakout means breakout timing condition only. It does not mean buy.

## 7. Distribution-Preservation Doctrine

The Timing State Layer must preserve the full upstream opportunity universe.

The layer may enrich opportunities with timing metadata only.

It must preserve:

- row count
- ticker universe
- upstream ordering
- upstream distribution shape
- upstream opportunity visibility

It may never:

- suppress rows
- remove tickers
- reorder opportunities
- prioritize opportunities
- narrow the universe
- gatekeep opportunities
- reduce visibility of upstream classifications

Distribution changes are audit findings, not permission to add hidden filters.

## 8. Non-Mutating Enrichment Doctrine

The Timing State Layer may append descriptive metadata only.

It may not:

- mutate upstream classifications
- rewrite upstream outputs
- alter upstream decisions
- overwrite upstream metadata
- normalize away upstream signals
- reinterpret Validation Layer outputs
- reinterpret Context Layer outputs
- reinterpret Fundamental Layer outputs
- reinterpret Decision Engine authority

Any future implementation must treat upstream fields as read-only source classifications.

## 9. Governance-Safe Schema Direction

Sprint 4 governance audit should evaluate schema direction before any execution planning or developer specification.

Possible governance-safe schema direction candidates include:

- `ticker`
- `date`
- `timing_state`
- `timing_reason`
- `breakout_state`
- `pullback_state`
- `compression_state`
- `extension_state`
- `participation_state`
- `timing_environment`
- `timing_metadata_status`
- `timing_pattern_state`
- `trend_participation_state`
- `timing_structure_state`
- `source_data_status`
- `source_timestamp`
- `generated_at`

No schema candidate is finalized by this preparation document.

Forbidden schema semantics include:

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

## 10. Descriptive Metadata Policy

Timing metadata must remain:

- descriptive
- classificatory
- deterministic
- non-allocative
- non-executory
- non-decisional
- non-preferential

Timing metadata must not imply:

- actionability
- urgency
- priority
- conviction
- allocation
- execution readiness
- good trade
- bad trade
- approval
- rejection
- opportunity preference
- expected outperformance
- expected alpha
- portfolio desirability

Descriptive timing states must never be interpreted as recommendation, preference, opportunity quality, execution preference, portfolio preference, expected performance, or alpha expectation.

## 11. Forbidden Semantics Matrix

| Forbidden Semantic | Forbidden Timing Output Examples | Governance Reason |
|---|---|---|
| Tradeability | `tradeable`, `approved`, `buy_candidate` | Decision Engine owns tradeability |
| Rejection | `rejected`, `failed_trade`, `blocked` | Timing may not suppress opportunities |
| Conviction | `high_conviction`, `conviction_score` | Decision Engine owns conviction semantics |
| Priority | `priority`, `best_opportunity`, `preferred_setup` | Decision Engine owns prioritization |
| Actionability | `actionable`, `execution_ready`, `trigger_ready` | Decision Engine owns execution semantics |
| Allocation | `allocation_weight`, `capital_weight` | Decision Engine owns allocation |
| Ranking | `opportunity_rank`, `timing_rank` | Decision Engine owns opportunity ranking authority |
| Scoring authority | `timing_score`, `composite_score`, `final_score` | Upstream layers may not become scoring authorities |
| Final actions | `BUY`, `SELL`, `HOLD`, `TRIM`, `REMOVE` | Decision Engine owns final actions |
| Urgency | `urgency`, `act_now`, `immediate` | Decision Engine owns urgency semantics |
| Expected performance | `expected_return`, `alpha_score`, `outperformance_expected` | Timing may not create alpha expectation |

## 12. Layer Responsibility Matrix

| Layer | Certified Responsibility | Sprint 4 Boundary |
|---|---|---|
| Scanner | discovery | Timing may not suppress scanner opportunities |
| Validation | structure classification | Timing may not override `structure_state` or reinterpret `valid_setup` |
| Context | leadership classification | Timing may not override leadership classification or reinterpret strength as actionability |
| Fundamentals | quality classification | Timing may not override quality metadata or synthesize quality with timing |
| Watchlist / Timing | timing-state tracking | Timing may classify timing conditions only |
| Portfolio | exposure/risk-state modelling | Timing may not create portfolio semantics |
| Decision Engine | allocation decisions | Only Decision Engine may allocate, rank, prioritize, score decisions, create conviction, or create final actions |
| Reporting | communication only | Timing may not create reporting priorities or execution framing |

## 13. Interaction Boundaries With Certified Upstream Layers

Validation interaction boundary:

- Timing may read structural fields as context for timing observation only after governance approval.
- Timing may not invalidate, override, downgrade, upgrade, or repair Validation classifications.
- Timing may not convert `valid_setup` into timing actionability.

Context interaction boundary:

- Timing may preserve Context classifications as upstream metadata.
- Timing may not combine leadership with timing into a composite opportunity interpretation.
- Timing may not treat strong leadership as execution preference or weak leadership as rejection.

Fundamental interaction boundary:

- Timing may preserve Fundamental classifications as upstream metadata.
- Timing may not combine quality with timing into a composite opportunity interpretation.
- Timing may not treat high quality as preferred timing or low quality as degraded timing.

## 14. Decision Engine Exclusivity Inheritance

Decision Engine exclusivity remains fully active.

Only `scripts/core/decision_engine.py` may determine:

- tradeability
- conviction
- allocation eligibility
- allocation priority
- execution aggressiveness
- urgency
- BUY logic
- SELL logic
- REMOVE logic
- final action
- portfolio-aware capital allocation

Sprint 4 may create no direct or indirect substitute for Decision Engine authority.

## 15. Risks And Controls

| Risk | Example | Required Control |
|---|---|---|
| Execution-readiness leakage | `breakout_ready`, `execution_ready` | Use descriptive state names only |
| Hidden filtering | dropping extended or stale rows | Row-count and key-preservation audit |
| Composite intelligence | combining timing, context, and quality into one label | Prohibit multi-factor synthesis outside Decision Engine |
| Ranking drift | sorting by timing state | Preserve upstream ordering |
| Scoring drift | numeric timing score becomes preference | Prohibit scoring-authority fields |
| Urgency drift | `act_now` style timing language | Ban urgency semantics outside Decision Engine |
| Recommendation drift | "preferred setup" timing states | Ban preference and recommendation semantics |
| Upstream mutation | replacing Validation or Context values | Treat upstream data as read-only |

## 16. Governance Leakage Scenarios

The following scenarios must be rejected during audit, execution planning, developer specification, and implementation:

- extended timing rows are removed
- pullback rows are marked as better opportunities
- breakout rows are marked as actionable
- compressed rows are promoted as preferred
- stale rows are hidden or deprioritized
- timing state changes output ordering
- timing metadata becomes a score, rank, or priority
- timing metadata implies urgency or readiness
- timing combines leadership and quality into a composite recommendation
- timing overwrites upstream structure, leadership, or quality classifications

## 17. In-Scope And Out-of-Scope Boundaries

In scope for Sprint 4 preparation:

- define governance-safe Timing State Layer responsibilities
- define forbidden responsibilities
- define classification-only doctrine
- define possible schema direction candidates
- define descriptive metadata policy
- define distribution-preservation doctrine
- define boundary controls
- define interaction boundaries
- define risks and controls
- define acceptance criteria for future governance audit and execution planning

Out of scope for Sprint 4 preparation:

- runtime implementation
- test implementation
- generated CSV/data changes
- architecture redesign
- strategy optimization
- thresholds
- filters
- ranking logic
- scoring authority
- allocation logic
- Decision Engine logic
- execution semantics
- developer specification
- implementation sequencing

## 18. Acceptance Criteria For Governance Audit

Sprint 4 may proceed to governance audit when this preparation package confirms:

- Sprint 0 through Sprint 3 doctrine is explicitly inherited
- Timing State Layer is descriptive only
- Decision Engine exclusivity is preserved
- distribution preservation is mandatory
- non-mutating enrichment is mandatory
- forbidden schema semantics are documented
- interaction boundaries with Validation, Context, and Fundamentals are documented
- governance leakage scenarios are documented
- no implementation is authorized
- no execution planning has begun

## 19. Acceptance Criteria For Future Execution Planning

Future execution planning may begin only after governance audit certification.

Future execution planning must define, without implementing:

- exact upstream input source
- exact row-key contract
- exact output artifact direction
- exact schema after audit approval
- deterministic ordering rules that preserve upstream order
- missing-data handling
- logging expectations
- forbidden-field checks
- distribution-preservation checks
- implementation file boundaries
- validation commands

Future execution planning must not authorize developer execution.

## 20. Scrum Master Recommendation

READY FOR SPRINT 4 GOVERNANCE AUDIT
