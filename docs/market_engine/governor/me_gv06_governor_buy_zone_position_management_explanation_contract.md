# ME-GV06 - Governor Buy-Zone and Position-Management Explanation Contract

Owner roles: Product Owner / Technical Architect / Financial Analyst / Data Steward / QA Lead / Governance Auditor

Job family: ME-GV - The Governor

Status: IMPLEMENTED

Contract version: `market-engine-governor-buy-zone-position-management-explanation-v1`

## Purpose

ME-GV06 adds evidence-backed explanation after governed recommendation mapping:

```text
factor states and scores
  -> recommendation state
  -> approved price/setup context
  -> buy-zone explanation
  -> approved position context
  -> position-management explanation
```

This is an explanation contract. It is not an execution, order, allocation,
portfolio mutation, or Decision Engine contract.

## Approved Input Contracts

### Price and setup context

ME-GV06 consumes only caller-supplied:

```text
market-engine-governor-approved-price-setup-context-v1
```

The context must declare:

```text
contract_version
reference
ticker
fresh
provenance_valid
structurally_valid
condition_state
invalidation_context
```

Optional approved condition evidence may include:

```text
support_zone
acceptable_zone
breakout_trigger
extension_reference
current_position_relative_to_zone
additional_setup_confirmation
additional_setup_confirmation_reference
```

The contract is necessary because the current SEC CompanyFacts Setup Detection
contract describes fundamental evidence patterns and does not itself provide
market support, resistance, trigger, or invalidation price levels.

### Position context

Position-management explanation consumes the existing approved contract:

```text
market-engine-portfolio-context-v1
```

ME-GV06 uses only its explicit `held` or `not_held` position state, ticker,
reference, freshness, and provenance validity. Price/setup context and
position context must match the Governor evaluation ticker. It does not infer
holdings from reports, watchlists, broker files, quantities, or market values.

## No Invented Price Rule

Numeric price fields are copied only from valid approved context. ME-GV06 does
not calculate:

* percentage pullbacks;
* support from current price;
* resistance from latest close;
* score-derived zones;
* valuation-derived targets;
* stop distances;
* risk-adjusted price levels.

Missing or malformed numeric evidence remains null or blocks the applicable
condition. Missing current price is represented by:

```text
current_position_relative_to_zone = unavailable
```

No live current-price lookup is required.

## Buy-Zone Eligibility

Eligibility requires:

1. `evaluation_completed_non_actionable`;
2. eligible ME-GV05 recommendation mapping;
3. evaluable technical setup, trend, and momentum factors;
4. numeric approved risk and data-confidence scores;
5. data confidence of at least 75;
6. risk score of at least 40;
7. valid, fresh, provenanced, structurally valid approved price context;
8. no unresolved hard price conflict;
9. valid approved invalidation context;
10. condition-specific approved evidence.

Failure is reason-coded and fail-closed.

## Buy-Zone States

Allowed states are exactly:

```text
blocked
insufficient_evidence
wait_for_pullback
wait_for_breakout_confirmation
acceptable_zone_context
extended_avoid_chasing
no_favorable_zone_identified
```

They are conditional explanations, not order instructions.

| Approved context state | Explanation state |
| --- | --- |
| `pullback_preferred` | `wait_for_pullback` |
| `breakout_confirmation_required` | `wait_for_breakout_confirmation` |
| `acceptable_zone` | `acceptable_zone_context` |
| `extended` | `extended_avoid_chasing` |
| `no_favorable_zone` | `no_favorable_zone_identified` |

`acceptable_zone_context` requires an approved bounded zone reference.
`wait_for_pullback` requires an approved bounded support-zone reference.
`wait_for_breakout_confirmation` requires an approved trigger reference and
level. `extended_avoid_chasing` requires an explicit extension reference.

## Recommendation, Risk, and Setup Limiters

An eligible `avoid` recommendation maps any otherwise favorable condition to
`no_favorable_zone_identified`.

An eligible `watch` recommendation cannot produce
`acceptable_zone_context`; it is limited to
`no_favorable_zone_identified`.

Risk below 40 blocks buy-zone explanation. Risk from 40 up to 60 is preserved
as a guardrail and prevents an acceptable-zone state.

Explicit `deteriorating` or `invalidated` setup context prevents a pullback,
breakout, or acceptable-zone explanation from remaining favorable.

Risk direction follows ME-GV04:

```text
higher risk-factor score = more favorable lower-risk profile
```

## Pullback and Breakout Conditions

The pullback payload preserves:

```text
condition_type
evidence_reference
lower_bound
upper_bound
explanation
limitations
```

The breakout payload preserves:

```text
condition_type
evidence_reference
level
explanation
limitations
```

Fields are copied from approved evidence only. Neither payload instructs an
entry, order type, quantity, or timing.

## Invalidation Context

Allowed invalidation states are:

```text
intact
deteriorating
invalidated
```

The output preserves the evidence reference, optional approved level, reason,
and limitations. It always emits:

```text
stop_order_authorized = false
```

An invalidation level is evidence context, not an automatically placed stop.

## Price Conflict Policy

`hard_conflict_references` block buy-zone and held-position explanation.
References remain visible and are never averaged.

`soft_conflict_references` remain visible and conservatively replace
pullback, breakout, or acceptable-zone states with
`no_favorable_zone_identified`. They also prevent `add_review_context`.

## Valuation Boundary

ME-GV04 does not implement valuation scoring. ME-GV06 therefore emits:

```text
valuation_score_unavailable_no_target_inference
```

It never substitutes neutral 50 and never produces a target price.

## Position-Management Eligibility

Missing approved position context produces ineligible
`no_position_context`.

Approved `not_held` context produces eligible `no_position_context`.

Hold/add/reduce/exit/monitor states require approved fresh provenanced `held`
context, eligible recommendation output, valid fresh provenanced structurally
valid ticker-aligned price/setup context, no hard conflict, and a valid risk
score.

## Position-Management States

Allowed states are exactly:

```text
blocked
insufficient_evidence
no_position_context
hold_context
add_review_context
reduce_review_context
exit_review_context
monitor_context
```

Mapping precedence for an existing position:

1. invalidated setup plus `avoid` -> `exit_review_context`;
2. deteriorating setup, risk below 60, or `avoid` ->
   `reduce_review_context`;
3. approved additional confirmation, `preferred` or `consider`, risk at least
   70, and no soft conflict -> `add_review_context`;
4. intact setup plus `preferred`, `consider`, or `watch` -> `hold_context`;
5. otherwise -> `monitor_context`.

These are review contexts. `add_review_context` does not mean buy more;
`reduce_review_context` does not specify a quantity; `exit_review_context` does
not create a sell order.

## Output Contracts

Buy-zone output includes:

```text
contract_version
eligibility_state
state
reason_codes
approved_price_references
pullback_condition
breakout_condition
invalidation_context
current_position_relative_to_zone
conflict_references
limitations
execution_authorized
stop_order_authorized
decision_engine_ready
```

Position-management output includes:

```text
contract_version
eligibility_state
state
reason_codes
position_context_reference
supporting_recommendation_state
supporting_factor_scores
invalidation_context
conflict_references
limitations
portfolio_mutation_authorized
order_generation_authorized
decision_engine_ready
```

## Preserved Authority and Aggregation Boundaries

The following remain fixed:

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

factor.weight = null
factor.weighted_score = null
overall_evaluation.score = null
overall_evaluation.weighted_score = null
overall_evaluation.rank = null
```

ME-GV06 adds no provider/network call, target weight, position sizing, order
generation, order routing, execution scheduling, automatic stop placement,
automatic profit taking, portfolio/watchlist mutation, broker behavior, or
Decision Engine decision.

## Next Sprint

```text
ME-DS01 - Define Dispatch Station output contract for Governor reports
```
