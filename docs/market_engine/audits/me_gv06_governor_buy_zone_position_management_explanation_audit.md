# ME-GV06 - Governor Explanation Audit

Sprint ID: ME-GV06
Status: COMPLETED BY ME-GV06
Job family: ME-GV / The Governor
Date: 2026-07-05
Branch: `me-gv06-buy-zone-position-management-explanation`

## Purpose and Implementation

ME-GV06 implements:

```text
market-engine-governor-buy-zone-position-management-explanation-v1
```

Runtime:

```text
src/market_engine/governor/explanation.py
src/market_engine/governor/evaluation.py
```

Runner and fixture evidence:

```text
scripts/market_engine/me_gv06_governor_explanation.py
tests/fixtures/market_engine/governor/me_gv05_governor_recommendation_case.json
```

The existing recommendation fixture is extended with versioned approved
price/setup context. This preserves the full ME-GV03 -> ME-GV04 -> ME-GV05 ->
ME-GV06 pipeline in one deterministic input.

## Evidence Decisions

The current SEC CompanyFacts Setup Detection contract does not provide market
price levels. ME-GV06 therefore requires explicit caller-supplied:

```text
market-engine-governor-approved-price-setup-context-v1
```

The fixture levels are synthetic approved fixture evidence and are copied
verbatim. Runtime contains no price calculation, percentage offset, target
generation, or current-price lookup.

Position explanation uses the existing:

```text
market-engine-portfolio-context-v1
```

Missing position context stays missing. Explicit `not_held` remains valid
context. Hold/add/reduce/exit review states require explicit `held` context.

## Eligibility and Mapping

Buy-zone eligibility checks the completed Governor state, recommendation
eligibility, technical/trend/momentum states, risk, data confidence, price
contract, freshness, provenance, structure, conflicts, invalidation, and
condition-specific evidence.

Position-management eligibility checks approved position context,
recommendation state, setup state, risk, invalidation, and conflicts.

Hard conflicts block. Soft conflicts remain visible and conservatively limit
zone/add-review interpretation. Risk uses the verified higher-is-more-favorable
ME-GV04 direction.

Missing valuation remains a limitation and never creates a target.

## Explanation-Only Boundary

The runtime and runner assert:

```text
execution_authorized = false
stop_order_authorized = false
portfolio_mutation_authorized = false
order_generation_authorized = false
actionable = false
recommendation_state_ready = false
decision_engine_ready = false
actionable_review = false
decision_ready = false
de_ready = false
```

Factor weights, weighted scores, overall score, overall weighted score, and
rank remain null.

No provider/network call, portfolio/watchlist mutation, allocation, target
weight, position sizing, order generation/routing, execution scheduling,
automatic stop placement, automatic profit taking, broker action, or Decision
Engine decision is introduced.

## Deterministic Run

```text
run_id: me-gv06-governor-explanation-20260705T180000Z
evaluation_timestamp: 2026-07-05T18:00:00Z
artifact_root: artifacts/market_engine/me-gv06-governor-explanation-20260705T180000Z
```

Artifacts:

```text
governor_explanation.json
governor_explanation_report.md
```

## Validation

```text
102 passed - tests/market_engine/governor
139 passed - tests/market_engine/run
727 passed - tests/market_engine
1394 passed - full pytest
5 passed - focused ME-GV06 runner tests
PASS - deterministic runner execution
PASS - git diff --check
```

Runner command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/market_engine/me_gv06_governor_explanation.py --input tests/fixtures/market_engine/governor/me_gv05_governor_recommendation_case.json --run-id me-gv06-governor-explanation-20260705T180000Z --evaluation-timestamp 2026-07-05T18:00:00Z --artifact-root artifacts/market_engine/me-gv06-governor-explanation-20260705T180000Z
```

Observed artifact summary:

```text
evaluations_total: 1
buy_zone_eligible_count: 1
buy_zone_ineligible_count: 0
counts_by_buy_zone_state: acceptable_zone_context=1
position_management_eligible_count: 0
position_management_ineligible_count: 1
counts_by_position_management_state: no_position_context=1
missing_position_context_count: 1
execution_authorized_count: 0
portfolio_mutation_authorized_count: 0
order_generation_authorized_count: 0
actionable_count: 0
recommendation_state_ready_count: 0
actionable_review_count: 0
decision_ready_count: 0
de_ready_count: 0
non_null_weight_count: 0
non_null_weighted_score_count: 0
non_null_overall_score_count: 0
non_null_rank_count: 0
```

Governance grep interpretation:

* no audited production ticker literal occurs in new ME-GV06 runtime, tests,
  or documentation; runtime and batch tests are ticker-agnostic;
* the extended committed fixture retains its synthetic `GR001` identifier as
  data only;
* new authority-term runtime hits are explanation contract fields,
  fixed-false booleans, zero-count calculations/assertions, and explicit
  non-authority prose;
* test dependency hits are negative provider/network/Telegram guardrails;
* documentation occurrences are evidence semantics and explicit non-goals;
* repository-wide results include pre-existing historical documentation,
  fixtures, legacy providers, portfolio utilities, and Decision Engine code;
* mandatory legacy script greps report only pre-existing portfolio and
  bytecode BUY/SELL/tradeability hits; no ME-GV06 path appears;
* `/dev/tty` is unavailable in the managed environment, so both requested
  repository-wide `tee /dev/tty` commands emitted results and then reported
  the environment restriction. Direct ME-GV06-scoped checks found no
  production ticker branch, provider/network call, allocation, sizing, order
  generation, portfolio mutation, broker action, or Decision Engine decision.

## Next Sprint

```text
ME-DS01 - Define Dispatch Station output contract for Governor reports
```
