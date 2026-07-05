# ME-GV05 - Governor Recommendation-State Mapping Audit

Sprint ID: ME-GV05
Status: COMPLETED BY ME-GV05
Job family: ME-GV / The Governor
Date: 2026-07-05
Branch: `me-gv05-recommendation-state-mapping`

## Purpose and Implementation

ME-GV05 implements:

```text
market-engine-governor-recommendation-state-v1
```

Runtime:

```text
src/market_engine/governor/recommendation.py
src/market_engine/governor/evaluation.py
```

Runner and fixture:

```text
scripts/market_engine/me_gv05_governor_recommendation_mapping.py
tests/fixtures/market_engine/governor/me_gv05_governor_recommendation_case.json
```

The pure mapping API consumes the Governor contract identity, top-level
evaluation state, factor evaluations, and explicit Recommendation Review
boundary. It reads no filesystem, clock, or network resource.

## Contract Decisions

Allowed eligibility states:

```text
eligible
ineligible
```

Allowed recommendation states:

```text
blocked
insufficient_evidence
avoid
watch
consider
preferred
```

The four ME-GV04-scored factors are critical: fundamentals, growth, risk, and
data confidence. All must be evaluable, scored under the approved scoring
contract, and free of score limitations.

The top-level Governor evaluation must be
`evaluation_completed_non_actionable`. This preserves valuation and contextual
factor sufficiency without inventing unsupported scores.

## Mapping and Guardrails

The mapping uses direct per-factor comparisons. It creates no average, overall
score, weighting, weighted score, or rank.

Data confidence below 75 blocks eligibility. Risk uses the verified ME-GV04
direction: a higher score means a more favorable lower-risk profile. Risk below
40 contributes to `avoid`; risk below 60 prevents `consider` or `preferred`.

Hard conflict references block eligibility. Soft conflict references remain
visible and cap otherwise favorable mapping at `watch`. No conflict is
averaged.

Missing valuation remains unavailable and prevents a complete evaluation.
Portfolio fit remains blocked without approved context and is disclosed as a
non-authoritative limitation.

## Recommendation Review and Reserved Readiness

Mapping requires the existing
`sec-companyfacts-recommendation-review-v1` boundary with
`human_review_required`, a deterministic reference, and `non_actionable=true`.

ME-GV05 sets `recommendation_mapping_authorized=true`, but leaves
`recommendation_state_ready=false`. Existing governance reserves the readiness
state and does not separately authorize it in this sprint. Actionable review,
decision readiness, DE readiness, and Decision Engine readiness remain false.

## Preserved Invariants

```text
factor.weight = null
factor.weighted_score = null
overall_evaluation.score = null
overall_evaluation.weighted_score = null
overall_evaluation.rank = null
recommendation_state.actionable = false
recommendation_state.decision_engine_ready = false
buy_zone_explanation.state = blocked_not_authorized
position_management_explanation.state = blocked_not_authorized
```

No entry, buy-under level, breakout trigger, stop, target, buy zone,
position-management instruction, allocation, target weight, position size,
order, execution advice, delivery, broker behavior, or portfolio/watchlist
mutation is introduced.

## Deterministic Run

```text
run_id: me-gv05-governor-recommendation-mapping-20260705T160000Z
evaluation_timestamp: 2026-07-05T16:00:00Z
artifact_root: artifacts/market_engine/me-gv05-governor-recommendation-mapping-20260705T160000Z
```

Artifacts:

```text
governor_recommendation_mapping.json
governor_recommendation_mapping_report.md
```

## Validation

```text
68 passed - tests/market_engine/governor
134 passed - tests/market_engine/run
688 passed - tests/market_engine
1355 passed - full pytest
5 passed - focused ME-GV05 runner tests
PASS - deterministic runner execution
PASS - git diff --check
```

Runner command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/market_engine/me_gv05_governor_recommendation_mapping.py --input tests/fixtures/market_engine/governor/me_gv05_governor_recommendation_case.json --run-id me-gv05-governor-recommendation-mapping-20260705T160000Z --evaluation-timestamp 2026-07-05T16:00:00Z --artifact-root artifacts/market_engine/me-gv05-governor-recommendation-mapping-20260705T160000Z
```

Observed artifact summary:

```text
evaluations_total: 1
recommendation_eligible_count: 1
recommendation_ineligible_count: 0
counts_by_recommendation_state: preferred=1
actionable_count: 0
actionable_review_count: 0
recommendation_state_ready_count: 0
decision_ready_count: 0
de_ready_count: 0
non_null_weight_count: 0
non_null_weighted_score_count: 0
non_null_overall_score_count: 0
non_null_rank_count: 0
```

Governance grep interpretation:

* no audited production ticker literal occurs in new ME-GV05 runtime,
  fixture, tests, or documentation; the fixture uses synthetic `GR001` data;
* new runtime authority-term hits are contract fields, fixed-false
  declarations, zero-count calculations/assertions, and explicit boundary
  prose;
* test dependency hits are negative provider/network/Telegram guardrails;
* documentation occurrences are state semantics, non-goals, and authority
  restrictions;
* repository-wide hits include pre-existing historical documentation,
  fixtures, tests, legacy providers, portfolio utilities, and Decision Engine
  behavior;
* mandatory legacy script greps report only pre-existing portfolio and
  bytecode BUY/SELL/tradeability hits; no ME-GV05 path appears;
* `/dev/tty` is unavailable in the managed environment, so both requested
  repository-wide `tee /dev/tty` commands emitted results and then reported
  the environment restriction. Direct ME-GV05-scoped checks found no
  production ticker, provider call, network call, allocation logic, order
  logic, execution instruction, or Decision Engine decision.

## Next Sprint

```text
ME-GV06 - Implement buy-zone and position-management explanation contract
```
