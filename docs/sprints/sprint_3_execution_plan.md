# Sprint 3 Execution Plan — Fundamental Quality Layer

## 1. Sprint Status

Status: READY FOR TECHNICAL LEAD EXECUTION REVIEW

Sprint 3 preparation, governance audit, and re-audit are complete. This execution plan is a Scrum planning artifact only. It does not authorize implementation or developer execution.

## 2. Certified Governance Baseline

Sprint 3 inherits the certified baseline:

- Sprint 0 = CERTIFIED COMPLETE
- Sprint 1 = CERTIFIED COMPLETE
- Sprint 2 = CERTIFIED COMPLETE
- Sprint 3 preparation = CERTIFIED
- Sprint 3 governance audit = COMPLETE
- Sprint 3 re-audit = COMPLETE

Certified doctrine:

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

## 3. Sprint Objective

Sprint 3 prepares implementation of a Fundamental Quality Layer as a pure classification and enrichment layer.

The layer may classify:

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

The layer must not create or imply allocation, tradeability, conviction, urgency, priority, ranking, scoring authority, execution readiness, BUY/SELL semantics, hard gating, hidden filtering, opportunity suppression, opportunity reordering, portfolio logic, or Decision Engine leakage.

## 4. Execution Scope

Future Sprint 3 implementation may be scoped to:

- inspect existing fundamentals-related code and data, if present
- create a Fundamental Quality Layer runtime only after Technical Lead specification
- create a governance-clean processed output, such as `data/processed/fundamental_quality.csv`
- create a governance-clean runtime log, such as `data/logs/fundamental_layer_log.csv`
- create focused tests for schema, missing data, distribution preservation, and forbidden-field absence
- document any data-source assumptions and missing-data behavior

This execution plan does not create those files.

## 5. Explicit Out-of-Scope Items

Out of scope:

- runtime implementation during planning
- test implementation during planning
- generated CSV/data changes during planning
- architecture redesign
- strategy optimization
- threshold tuning
- filters
- ranking logic
- scoring authority
- allocation logic
- Decision Engine logic
- execution semantics
- BUY/SELL/HOLD/TRIM/REMOVE behavior
- portfolio logic
- reporting behavior
- scanner, validation, context, watchlist, portfolio, Decision Engine, or reporting changes unless explicitly approved in a future Technical Lead specification

## 6. Target Layer Placement

Target certified pipeline placement:

scanner → validation_layer → context_layer → fundamental_layer → watchlist → portfolio → decision_engine → reporting

The Fundamental Layer must sit upstream of the Decision Engine as quality classification only.

Responsibilities:

| Layer | Responsibility | Sprint 3 Boundary |
|---|---|---|
| Scanner | discovery | Fundamentals may not suppress scanner output |
| Validation | structure classification | Fundamentals may not override `structure_state` |
| Context | leadership classification | Fundamentals may not override leadership classification |
| Fundamentals | quality classification | Fundamentals classify business and financial quality only |
| Watchlist | timing-state tracking | Fundamentals may not determine timing |
| Portfolio | exposure/risk-state modelling | Fundamentals may not create portfolio semantics |
| Decision Engine | allocation decisions | Only Decision Engine may allocate, rank, prioritize, score decisions, create conviction, or create final actions |
| Reporting | communication only | Fundamentals may not create reporting priorities or execution framing |

## 7. Data Contract Direction

Future Technical Lead specification may define files such as:

- `scripts/core/build_fundamental_layer.py`
- `data/processed/fundamental_quality.csv`
- `data/logs/fundamental_layer_log.csv`
- `tests/core/test_build_fundamental_layer.py`

These are planning directions only and are not authorized for creation by this document.

The data contract must:

- preserve one output row per eligible upstream opportunity where possible
- preserve ticker/date identity
- expose quality classifications and metadata only
- expose source-data status clearly
- keep raw numeric financial inputs descriptive, if used
- avoid fields that imply decisions, rankings, scoring authority, allocation, priority, conviction, urgency, actionability, or execution readiness

## 8. Schema Governance Direction

Governance-safe schema direction may include:

- `ticker`
- `date`
- `quality_state`
- `quality_reason`
- `profitability_profile`
- `balance_sheet_profile`
- `earnings_quality_profile`
- `capital_efficiency_profile`
- `cashflow_profile`
- `stability_profile`
- `quality_metadata_status`
- `source_data_status`
- `sector_quality_profile`
- `missing_fundamental_data`

Forbidden schema semantics:

- `tradeable`
- `approved`
- `rejected`
- `high_conviction`
- `conviction_score`
- `priority`
- `actionable`
- `buy_candidate`
- `execution_ready`
- `best_opportunity`
- `allocation_weight`
- `urgency`
- `ranking_score`
- `composite_score`
- `final_score`
- `quality_rank`
- `quality_score` when used as scoring authority

Any future schema proposal must be reviewed by the Technical Lead before implementation.

## 9. Missing Data Governance

Missing or partial fundamentals must never suppress rows.

Missing data may produce descriptive metadata states only, such as:

- `insufficient_data`
- `unavailable`
- `partial_data`
- `source_missing`
- `stale_data`

Missing data must never produce:

- rejection
- exclusion
- tradeability failure
- priority downgrade
- conviction downgrade
- ranking penalty
- allocation impact

Future implementation must test missing, stale, partial, and unavailable source data.

## 10. Distribution Preservation Controls

The Fundamental Layer must preserve the upstream opportunity universe.

It may enrich opportunities with quality metadata only.

It may never:

- suppress rows
- remove tickers
- reorder opportunities
- prioritize opportunities
- narrow the universe
- gatekeep opportunities

Future implementation must include row-count and identity-preservation tests comparing input opportunity keys to output opportunity keys.

## 11. Logging and Audit Trail Expectations

Future implementation may add a runtime log such as `data/logs/fundamental_layer_log.csv`.

Logging may include:

- run timestamp
- input row count
- output row count
- quality-state distribution
- source-data status distribution
- missing-data count
- stale-data count
- partial-data count
- sector-quality metadata availability count
- forbidden-field drift checks, if implemented

Logging must not:

- change runtime output eligibility
- create priority
- create ranking
- create scoring authority
- create conviction
- create urgency
- create allocation semantics
- create execution readiness

## 12. Testing and Regression Expectations

Future Sprint 3 implementation must include tests for:

- exact schema expectations
- forbidden-field absence
- row preservation
- ticker/date identity preservation
- missing-data non-blocking behavior
- partial-data non-blocking behavior
- stale-data non-blocking behavior
- sector-relative quality metadata as enrichment only
- deterministic output
- no mutation of upstream scanner, validation, or context outputs
- no Decision Engine, portfolio, watchlist, reporting, or Telegram changes

Focused tests should run before full test suite.

## 13. Forbidden Field / Forbidden Semantic Controls

Future Technical Lead specification must include forbidden-field checks for active Fundamental source and tests.

Required forbidden semantics:

- tradeability
- approval/rejection
- conviction
- priority
- actionability
- execution readiness
- allocation
- ranking authority
- scoring authority
- final actions
- urgency

Potential grep checks:

```bash
grep -R "tradeable" scripts/core tests/core
grep -R "approved" scripts/core tests/core
grep -R "rejected" scripts/core tests/core
grep -R "conviction" scripts/core tests/core
grep -R "priority" scripts/core tests/core
grep -R "actionable" scripts/core tests/core
grep -R "execution_ready" scripts/core tests/core
grep -R "allocation" scripts/core tests/core
grep -R "ranking_score" scripts/core tests/core
grep -R "composite_score" scripts/core tests/core
grep -R "final_score" scripts/core tests/core
grep -R "BUY" scripts/core tests/core
grep -R "SELL" scripts/core tests/core
grep -R "REMOVE" scripts/core tests/core
```

Interpretation must distinguish Decision Engine-owned logic, existing valid portfolio/Telegram command parsing, and test-only absence assertions.

## 14. Implementation Sequencing

Required sequence before any developer implementation:

1. Technical Lead reviews this execution plan.
2. Technical Lead inspects current codebase for existing fundamentals-related code, data, and tests.
3. Technical Lead creates a developer specification.
4. Developer executes inspection-first.
5. Developer changes only approved files.
6. Developer adds schema, distribution, missing-data, deterministic-output, and forbidden-field tests.
7. Developer runs focused tests.
8. Developer runs full test suite.
9. Developer runs governance grep checks.
10. Developer runs pipeline only if runtime source or generated artifacts change and Technical Lead specification requires it.
11. Technical Lead performs implementation audit.
12. Sprint closeout documentation is completed only after approval.

## 15. Audit Checkpoints

Required audit checkpoints:

- Technical Lead execution-plan review
- Technical Lead developer-spec review
- developer inspection checkpoint
- schema contract review
- missing-data behavior review
- distribution-preservation review
- forbidden-field grep review
- generated-artifact review
- Technical Lead implementation audit
- Scrum closeout review

## 16. Developer Handoff Prerequisites

Developer handoff may not occur until:

- this execution plan is reviewed
- a Technical Lead developer specification exists
- allowed files are explicitly listed
- forbidden files are explicitly listed
- proposed schema is approved
- missing-data behavior is approved
- distribution-preservation tests are specified
- forbidden-field checks are specified
- validation commands are specified
- artifact handling rules are specified

## 17. Acceptance Criteria for Execution Approval

Technical Lead may approve Sprint 3 execution only when:

- Fundamental Layer scope is quality classification only
- schema direction is governance-clean
- missing-data behavior is non-blocking
- distribution preservation is mandatory
- no ranking authority is introduced
- no scoring authority is introduced
- no allocation logic is introduced
- no tradeability semantics are introduced
- no conviction semantics are introduced
- no urgency semantics are introduced
- no execution readiness is introduced
- no BUY/SELL/HOLD/TRIM/REMOVE behavior is introduced
- no hard gates or hidden filters are introduced
- Decision Engine authority is explicitly protected
- implementation is inspection-first and minimal
- all expected tests and grep checks are defined

## 18. Scrum Master Recommendation

READY FOR TECHNICAL LEAD EXECUTION REVIEW
