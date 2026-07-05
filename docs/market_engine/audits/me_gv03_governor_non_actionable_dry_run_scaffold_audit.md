# ME-GV03 - Governor Non-Actionable Dry-Run Scaffold Audit

Sprint ID: ME-GV03
Status: COMPLETED BY ME-GV03
Job family: ME-GV / The Governor
Date: 2026-07-05
Branch: `me-gv03-governor-non-actionable-dry-run-scaffold`

## Purpose

ME-GV03 implements the first Governor runtime scaffold after:

```text
ME-SA12 -> ME-SA13 -> ME-SA14 -> ME-RUN29 -> ME-GV01 -> ME-GV02
```

The scaffold consumes approved deterministic evidence, classifies evidence
sufficiency with the ME-GV02 taxonomy, and emits the ME-GV01 contract shape.
It does not evaluate investment quality or produce action semantics.

## Contract Implementation

Runtime:

```text
src/market_engine/governor/evaluation.py
src/market_engine/governor/__init__.py
```

Runner:

```text
scripts/market_engine/me_gv03_governor_non_actionable_dry_run.py
```

Implemented contract identities:

```text
market-engine-governor-investment-evaluation-v1
market-engine-governor-factor-taxonomy-v1
market-engine-governor-approved-evidence-v1
```

The output includes every ME-GV01 top-level section:

```text
contract_version
evaluation_id
ticker
market
company_name
input_references
evidence_readiness
evaluation_state
factor_evaluations
overall_evaluation
recommendation_state
buy_zone_explanation
position_management_explanation
risk_and_limitations
missing_evidence
blocked_reasons
authority_boundary
provenance
```

## Canonical Taxonomy

The runtime contains exactly the nine ME-GV02 families and seven states:

```text
fundamentals        not_started
growth              blocked
valuation           unavailable
trend               insufficient_evidence
momentum            partial
risk                qualitative_only
technical_setup     evaluable
portfolio_fit
data_confidence
```

No positive/negative or other investment-quality state exists.

## Input Evidence

Committed deterministic fixture:

```text
tests/fixtures/market_engine/governor/me_gv03_governor_evidence_cases.json
```

The six cases cover company-profile-only evidence, partial fundamentals, stale
market/setup evidence, unprovenanced evidence, conflicting evidence, missing
valuation and portfolio evidence, and broad evaluable evidence. Ticker values
are synthetic fixture data only and never select behavior.

## Factor-State Rules

Common fail-closed precedence:

1. unapproved evidence contract -> `blocked`;
2. invalid manifest -> `blocked`;
3. missing provenance -> `blocked`;
4. stale factor evidence -> `blocked`;
5. non-consumable evidence -> `blocked`;
6. malformed evidence -> `blocked`;
7. absent approved evidence -> `unavailable`;
8. descriptive evidence -> `qualitative_only`;
9. limited evidence -> `partial`, except incomplete valuation becomes
   `insufficient_evidence`;
10. complete approved evidence -> `evaluable`.

Data confidence preserves inspectable stale evidence as `partial`; it cannot
upgrade another factor. Portfolio fit is always blocked without approved
portfolio context:

```text
blocked_missing_approved_portfolio_context
```

## Top-Level Evaluation Mapping

| Condition | Evaluation state |
| --- | --- |
| Unapproved contract or global hard gate failure | `blocked` |
| Per-factor invalid manifest, missing provenance, non-consumable, or malformed evidence | `blocked` |
| Only descriptive/unavailable core evidence | `descriptive_only` |
| Any partial, insufficient, or mixed evaluable core evidence | `partial_evaluation` |
| All seven core factors and data confidence evaluable | `evaluation_completed_non_actionable` |

`evaluation_ready` remains available in the contract enum but is not needed by
this completed scaffold mapping. No actionable state exists.

## Conflicting Evidence

Conflicting references remain verbatim. A complete factor with conflict is
downgraded from `evaluable` to `partial` and records:

```text
conflicting_evidence_preserved_without_averaging
```

The scaffold does not average, reconcile, score, or discard the conflict.

## Score-Null and Authority Invariants

Every factor emits null `score`, `score_scale`, `weight`, and
`weighted_score`. Overall output emits null `score`, `score_scale`,
`weighted_score`, and `rank`.

Recommendation, buy-zone, and position-management sections remain:

```text
state = blocked_not_authorized
```

The runner asserts that actionable, actionable-review,
recommendation-state-ready, decision-ready, DE-ready, non-null score, non-null
weight, and non-null rank counts remain zero.

## Deterministic Run

```text
run_id: me-gv03-governor-non-actionable-dry-run-20260705T120000Z
evaluation_timestamp: 2026-07-05T12:00:00Z
artifact_root: artifacts/market_engine/me-gv03-governor-non-actionable-dry-run-20260705T120000Z
```

Artifacts:

```text
governor_evaluation.json
governor_evaluation_report.md
```

The artifact root is local and ignored. The runner, fixture, and tests
reproduce it.

Observed states:

```text
blocked: 1
descriptive_only: 1
partial_evaluation: 3
evaluation_completed_non_actionable: 1
```

## Governance Boundary

ME-GV03 adds no provider/network call, acquisition, import, upstream semantic
change, factor scoring, scale, weight, weighted aggregation, ranking, urgency,
conviction, tradeability, BUY/SELL/HOLD semantics, recommendation mapping,
target price, buy zone, stop, position-management instruction, allocation,
position sizing, order generation, execution advice, delivery,
portfolio/watchlist mutation, scheduler/UI/broker behavior, production report,
or Decision Engine authority.

## Validation

```text
15 passed - tests/market_engine/governor
5 passed - focused ME-GV03 runner tests
124 passed - tests/market_engine/run
625 passed - tests/market_engine
1292 passed - full pytest
PASS - deterministic runner execution
PASS - git diff --check
```

Governance grep interpretation:

* no audited ticker literal appears in new Governor runtime, runner, fixture,
  tests, or documentation;
* synthetic `GX001` through `GX006` values exist only in fixture/test evidence;
* new authority-term runtime hits are blocked defaults, fixed-false booleans,
  zero-count assertions, and non-authority boundary text;
* test hits for prohibited dependencies and the invented `positive` state are
  negative guardrails;
* documentation hits are explicit non-goals and zero-state evidence;
* repository-wide hits include pre-existing contracts, historical docs,
  fixtures, and legacy runtime outside the ME-GV03 diff;
* the mandatory legacy `scripts/` BUY/SELL grep still reports pre-existing
  portfolio and bytecode hits; ME-GV03 does not modify those paths;
* `/dev/tty` is unavailable in the managed environment, so the requested
  `tee /dev/tty` commands emitted `rg` results and then reported an environment
  error. Direct scoped fallback greps confirmed the interpretations above.

## Next Sprint

```text
ME-GV04 - Implement factor scoring from approved analysis evidence
```
