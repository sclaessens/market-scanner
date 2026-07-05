# ME-GV04 - Governor Factor Scoring Audit

Sprint ID: ME-GV04
Status: COMPLETED BY ME-GV04
Job family: ME-GV / The Governor
Date: 2026-07-05
Branch: `me-gv04-factor-scoring-approved-analysis-evidence`

## Purpose and Implementation

ME-GV04 implements the versioned contract:

```text
market-engine-governor-factor-scoring-v1
```

Runtime:

```text
src/market_engine/governor/scoring.py
src/market_engine/governor/evaluation.py
```

Runner and committed evidence:

```text
scripts/market_engine/me_gv04_governor_factor_scoring.py
tests/fixtures/market_engine/governor/me_gv04_governor_scoring_cases.json
```

The implementation scores only explicit approved component evidence after the
ME-GV03 factor state is `evaluable`.

## Scale, Direction, and Precision

The common factor scale is 0.0 to 100.0 with midpoint 50.0. Higher means a
more favorable factor assessment. For risk, higher explicitly means a more
favorable lower-risk profile. Decimal calculations use two-place half-up
rounding.

Implemented scorers:

```text
fundamentals
growth
risk
data_confidence
```

Deliberately unimplemented scorers:

```text
valuation
trend
momentum
technical_setup
portfolio_fit
```

No scorer derives new upstream metrics. Normalization rules and clamp
boundaries are named, versioned by the scoring contract, documented, and
tested at boundaries.

## Evidence and Failure Behavior

Every component output preserves its component identity, approved evidence
reference, input value, normalization rule, normalized value, contribution,
and limitations.

Missing components are not zero-filled. Invalid inputs, incomplete components,
unapproved contracts, mismatched references, unexpected rules, malformed
growth periods, and non-evaluable factor states produce null scores with
reason-coded limitations.

Conflict references remain visible. The existing ME-GV03 conflict downgrade
makes the factor `partial`; ME-GV04 therefore leaves its score null and never
silently averages the conflict.

Data confidence is scored separately and is never multiplied into or otherwise
used to alter another factor score.

## Aggregation and Authority Invariants

Runtime and runner assertions preserve:

```text
factor.weight = null
factor.weighted_score = null
overall_evaluation.score = null
overall_evaluation.weighted_score = null
overall_evaluation.rank = null
```

Recommendation, buy-zone, and position-management sections remain
`blocked_not_authorized`. Actionable, actionable-review,
recommendation-state-ready, decision-ready, and DE-ready counts remain zero.

ME-GV04 adds no recommendation state, ranking, urgency, conviction,
tradeability, price target, entry level, buy zone, allocation, position size,
order, execution instruction, delivery, portfolio/watchlist mutation, broker
behavior, or Decision Engine authority.

## Deterministic Run

```text
run_id: me-gv04-governor-factor-scoring-20260705T140000Z
evaluation_timestamp: 2026-07-05T14:00:00Z
artifact_root: artifacts/market_engine/me-gv04-governor-factor-scoring-20260705T140000Z
```

Artifacts:

```text
governor_factor_scoring.json
governor_factor_scoring_report.md
```

The local artifact root is ignored. Caller-supplied run identity and timestamp
make JSON and Markdown output byte-for-byte reproducible.

## Validation

```text
41 passed - tests/market_engine/governor
129 passed - tests/market_engine/run
656 passed - tests/market_engine
1323 passed - full pytest
5 passed - focused ME-GV04 runner tests
PASS - deterministic runner execution
PASS - git diff --check
```

Runner command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/market_engine/me_gv04_governor_factor_scoring.py --input tests/fixtures/market_engine/governor/me_gv04_governor_scoring_cases.json --run-id me-gv04-governor-factor-scoring-20260705T140000Z --evaluation-timestamp 2026-07-05T14:00:00Z --artifact-root artifacts/market_engine/me-gv04-governor-factor-scoring-20260705T140000Z
```

Observed artifact summary:

```text
evaluations_total: 2
scored_factor_count: 4
unscored_factor_count: 14
score_null_count: 14
conflict_blocked_score_count: 1
non_null_weight_count: 0
non_null_weighted_score_count: 0
non_null_overall_score_count: 0
non_null_rank_count: 0
all reserved state counts: 0
```

Governance grep interpretation:

* no audited production ticker literal occurs in new ME-GV04 runtime, fixture,
  tests, or documentation; the fixture uses only synthetic `GS001` and
  `GS002` values as data;
* new runtime authority-term hits are fixed-false side-effect declarations,
  zero-count calculations/assertions, and explicit non-authority text;
* test hits for provider, network, Telegram, and similar imports are negative
  dependency guardrails;
* documentation hits are explicit boundaries and non-goals;
* repository-wide hits include pre-existing historical documentation,
  fixtures, tests, legacy providers, and Decision Engine behavior;
* the mandatory legacy `scripts/` BUY and SELL checks report only pre-existing
  portfolio utilities and bytecode; the tradeability check reports pre-existing
  bytecode and no ME-GV04 path;
* `/dev/tty` is unavailable in the managed environment, so both requested
  repository-wide `tee /dev/tty` checks emitted their results and reported the
  environment restriction. Direct ME-GV04-scoped checks confirmed no ticker
  branch, provider call, network call, recommendation logic, allocation logic,
  or action semantics.

## Next Sprint

```text
ME-GV05 - Implement recommendation-state mapping under approved boundary
```
