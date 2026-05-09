# Sprint 3 — Fundamental Quality Layer

## 1. Sprint Status

Status: READY FOR SPRINT 3 GOVERNANCE AUDIT

Sprint 3 is in preparation only. This document does not authorize implementation, runtime code changes, test changes, generated data changes, architecture redesign, strategy optimization, allocation logic, filtering logic, ranking logic, scoring authority, or Decision Engine changes.

Sprint 3 may proceed only through the certified delivery workflow:

audit  
→ impact analysis  
→ scoped Technical Lead specification  
→ developer execution  
→ validation  
→ governance review  
→ closeout

## 2. Governance Inheritance From Sprint 0, Sprint 1, And Sprint 2

Sprint 3 inherits the certified Sprint 0 doctrine:

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

Sprint 3 inherits Sprint 1 Validation certification:

- Validation Layer = structure classification only
- `structure_state` and `structure_reason` are authoritative
- `valid_setup` and `validation_reason` are compatibility-only
- Validation output may not be reinterpreted as tradeability, allocation eligibility, conviction, urgency, ranking, or execution readiness

Sprint 3 inherits Sprint 2 Context certification:

- Context Layer = leadership and relative-strength classification only
- weak context is not rejection
- strong or leading context is not tradeability
- sector-relative context data is enrichment only
- Context output may not be reinterpreted as allocation, execution readiness, conviction, ranking, or priority

## 3. Sprint Objective

Sprint 3 prepares the Fundamental Quality Layer as a pure classification and enrichment layer.

The sprint objective is to define how business quality and financial quality can be classified without introducing:

- allocation logic
- filtering-first behavior
- tradeability semantics
- urgency semantics
- conviction semantics
- execution semantics
- hidden filtering
- Decision Engine leakage
- ranking semantics
- scoring authority
- decision semantics

Sprint 3 must preserve upstream opportunity distribution.

## 4. Fundamental Layer Responsibilities

The Fundamental Layer may classify and enrich:

- profitability quality
- balance-sheet quality
- earnings quality
- capital efficiency
- cash-flow quality
- stability metrics
- quality-factor metadata
- Piotroski-style quality classification
- Greenblatt-style quality metadata
- sector-relative quality metadata

The Fundamental Layer may emit descriptive metadata that helps later layers understand business quality. It may not decide what should happen with capital.

## 5. Explicitly Forbidden Fundamental Layer Responsibilities

The Fundamental Layer may not determine:

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

The Fundamental Layer may not:

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
- create portfolio semantics
- create execution semantics
- simulate Decision Engine behavior

## 6. Classification-Only Doctrine

Sprint 3 must keep Fundamentals as quality classification only.

Allowed:

- classify quality profiles
- classify profitability profiles
- classify balance-sheet profiles
- classify earnings quality profiles
- classify capital-efficiency profiles
- classify cash-flow profiles
- expose deterministic quality metadata
- expose missing-data metadata
- expose sector-relative quality metadata where available
- log quality-state distributions

Forbidden:

- trade filtering
- execution filtering
- allocation filtering
- ranking authority
- scoring authority
- final-action logic
- conviction logic
- urgency logic
- BUY/SELL/HOLD/TRIM/REMOVE behavior

High quality means high quality classification only. It does not mean tradeability.

Low quality means low quality classification only. It does not mean rejection.

Missing fundamental data means missing-data classification only. It does not mean removal.

## 7. Distribution-Preservation Doctrine

The Fundamental Layer must preserve upstream opportunity distribution.

Fundamentals may enrich every available opportunity with quality metadata, but may never:

- suppress
- remove
- reorder
- prioritize
- narrow
- gatekeep

the upstream opportunity universe.

Distribution changes are observability findings, not permission to add hidden filters.

## 8. Governance-Safe Schema Direction

Sprint 3 governance audit should evaluate schema direction before any developer specification.

Governance-safe output names may include:

- `ticker`
- `date`
- `quality_profile`
- `profitability_profile`
- `balance_sheet_profile`
- `earnings_quality_profile`
- `capital_efficiency_profile`
- `cashflow_profile`
- `quality_reason`
- `quality_metadata`
- `quality_classification`
- `quality_state`
- `missing_fundamental_data`
- `sector_quality_profile`

Any numeric metrics used as source data must remain descriptive metadata. Numeric source metrics may not become ranking authority, scoring authority, allocation weight, priority, conviction, or execution readiness.

## 9. Descriptive Metadata Policy

Fundamental metadata must remain:

- descriptive
- classificatory
- deterministic
- non-allocative
- non-executory
- non-decisional

Fundamental metadata may be consumed later by Decision Engine-owned logic, but the Fundamental Layer itself may not decide capital allocation, tradeability, conviction, urgency, ranking, priority, or final actions.

Sector-relative quality metadata may enrich classification where available. Missing sector-quality data must be handled as missing metadata, not as a reason to block or remove an opportunity.

## 10. Forbidden Semantics Matrix

| Forbidden Semantic | Forbidden Fundamental Output Examples | Governance Reason |
|---|---|---|
| Tradeability | `tradeable`, `approved`, `buy_candidate` | Decision Engine owns tradeability |
| Rejection | `rejected`, `invalid_quality`, `blocked` | Upstream layers may not remove opportunities |
| Conviction | `high_conviction`, `conviction_score` | Decision Engine owns conviction semantics |
| Priority | `priority`, `allocation_priority`, `best_opportunity` | Decision Engine owns prioritization |
| Actionability | `actionable`, `execution_ready` | Decision Engine owns execution readiness |
| Allocation | `allocation_weight`, `capital_weight` | Decision Engine owns allocation |
| Ranking | `quality_rank`, `opportunity_rank` | Decision Engine owns opportunity ranking authority |
| Scoring authority | `quality_score`, `composite_score`, `final_score` | Upstream layers may not become scoring authorities |
| Final actions | `BUY`, `SELL`, `HOLD`, `TRIM`, `REMOVE` | Decision Engine owns final actions |
| Urgency | `urgency`, `act_now`, `fast_track` | Decision Engine owns urgency semantics |

## 11. Layer Responsibility Matrix

| Layer | Certified Responsibility | Sprint 3 Boundary |
|---|---|---|
| Scanner | discovery | Fundamentals may not suppress scanner opportunities |
| Validation | structure classification | Fundamentals may not override `structure_state` |
| Context | leadership classification | Fundamentals may not override leadership classification |
| Fundamentals | quality classification | Fundamentals may classify business and financial quality only |
| Watchlist | timing-state tracking | Fundamentals may not determine timing state |
| Portfolio | exposure/risk-state modelling | Fundamentals may not create portfolio semantics |
| Decision Engine | allocation decisions | Only Decision Engine may allocate, rank, prioritize, score decisions, or create actions |
| Reporting | communication only | Fundamentals may not create reporting priorities or execution framing |

## 12. Layer Boundary Enforcement

Sprint 3 must protect certified boundaries:

- Validation remains structure-only
- Context remains leadership-only
- Fundamentals remain quality-only
- Watchlist remains timing-state-only
- Portfolio remains exposure/risk-state-only
- Decision Engine remains the only allocation authority
- Reporting remains presentation-only

No Sprint 3 work may create cross-layer shortcuts or hidden interpretations.

## 13. Separation-Of-Concerns Enforcement

Fundamental quality classification must be separable from:

- setup structure
- leadership context
- timing state
- portfolio risk
- final decisions
- allocation
- urgency
- ranking
- scoring authority

The Fundamental Layer may classify quality independently, but it may not collapse these classifications into a decision.

## 14. Hidden-Filter Prevention

Sprint 3 audit and any future implementation spec must search for and forbid:

- row drops caused by weak fundamentals
- missing-data exclusion
- sector-quality exclusion
- minimum quality thresholds that remove rows
- low-quality rejection
- high-quality fast tracks
- hard gates on profitability or balance sheet fields
- quality-derived allocation readiness
- quality-derived execution readiness
- quality-derived priority or ranking
- quality-derived conviction labels

If quality data is unavailable, output should preserve the opportunity with a missing-data classification.

## 15. Decision Engine Protection

The Decision Engine remains the ONLY layer allowed to:

- allocate
- prioritize
- rank
- filter for execution
- generate actionable decisions
- create portfolio semantics
- create urgency semantics
- create conviction semantics
- create tradeability semantics
- create final action semantics

Sprint 3 must not create fields that pre-package these decisions for the Decision Engine.

## 16. Audit Requirements Before Future Implementation

Before any Sprint 3 developer specification, perform a governance audit of:

- existing fundamentals-related code, if present
- existing fundamentals-related tests, if present
- current processed data schemas, if present
- financial data sources and missing-data behavior
- old documentation references to quality scoring, conviction, ranking, or filtering
- any proposed `fundamental_profile.csv` schema
- any proposed logging or observability schema

Required audit questions:

- Does any existing fundamentals logic filter opportunities?
- Does any existing fundamentals logic rank or score opportunities?
- Does any existing fundamentals logic imply tradeability, conviction, urgency, actionability, or allocation eligibility?
- Does missing fundamental data remove opportunities?
- Are sector-relative quality metrics descriptive only?
- Are proposed output fields governance-clean?
- Are tests prepared to enforce classification-only behavior?

## 17. Governance Risks And Controls

| Risk | Severity | Control |
|---|---|---|
| Fundamentals become a trade filter | HIGH | Forbid rejection, approval, tradeability, and gate fields |
| Quality metrics become ranking authority | HIGH | Forbid rank, priority, composite score, and final score outputs |
| Quality classifications become conviction labels | HIGH | Keep conviction semantics Decision Engine-only |
| Missing fundamentals remove rows | HIGH | Require missing-data classification and row preservation |
| Sector-relative quality becomes blocking | MEDIUM | Treat sector-relative data as nullable enrichment |
| Legacy docs imply downstream conviction enrichment from Fundamentals | MEDIUM | Clarify that only Decision Engine may create conviction semantics |
| Developer introduces thresholds during implementation | HIGH | Audit-first and Technical Lead specification required before execution |
| Fundamental Layer overrides Validation or Context | HIGH | Enforce layer responsibility matrix and boundary tests |

## 18. Controls And Enforcement

Future Sprint 3 execution planning should require:

- exact schema assertions
- forbidden-field assertions
- row-preservation tests
- missing-data non-blocking tests
- sector-relative data non-blocking tests
- deterministic output validation
- governance grep checks
- Technical Lead review
- Functional Analyst review

Potential forbidden grep terms for future implementation:

```bash
grep -R "tradeable" scripts tests
grep -R "approved" scripts tests
grep -R "rejected" scripts tests
grep -R "conviction" scripts tests
grep -R "priority" scripts tests
grep -R "actionable" scripts tests
grep -R "execution_ready" scripts tests
grep -R "allocation" scripts tests
grep -R "ranking_score" scripts tests
grep -R "composite_score" scripts tests
grep -R "final_score" scripts tests
grep -R "BUY" scripts tests
grep -R "SELL" scripts tests
grep -R "REMOVE" scripts tests
```

Interpretation must distinguish Decision Engine-owned logic, valid portfolio/Telegram command parsing, and test-only absence assertions from upstream governance leakage.

## 19. Acceptance Criteria For Sprint 3 Execution Approval

Sprint 3 may move to Technical Lead specification only when:

- Sprint 3 governance audit is complete
- Fundamental Layer responsibilities are limited to quality classification
- forbidden responsibilities are explicit
- proposed schema direction is governance-clean
- no ranking, scoring authority, tradeability, allocation, conviction, urgency, actionability, or final-action semantics are present upstream
- distribution preservation is explicit
- missing-data behavior is non-blocking
- sector-relative quality metadata is enrichment-only
- implementation scope is narrow and inspection-first
- Technical Lead approves developer-spec readiness

## 20. Out-Of-Scope Items

Out of scope for Sprint 3 preparation:

- runtime implementation
- test implementation
- generated data changes
- architecture redesign
- strategy optimization
- threshold tuning
- new filters
- allocation logic
- Decision Engine changes
- ranking logic
- scoring authority
- execution semantics
- BUY/SELL/HOLD/TRIM/REMOVE behavior

Out of scope for Fundamental Layer generally:

- tradeability
- allocation eligibility
- conviction
- urgency
- final actions
- priority
- ranking
- scoring authority
- execution readiness

## 21. Scrum Master Recommendation

READY FOR SPRINT 3 GOVERNANCE AUDIT
