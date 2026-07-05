# ME-GV05 - Governor Recommendation-State Mapping Contract

Owner roles: Product Owner / Technical Architect / Financial Analyst / Data Steward / QA Lead / Governance Auditor

Job family: ME-GV - The Governor

Status: IMPLEMENTED

Contract version: `market-engine-governor-recommendation-state-v1`

## Purpose

ME-GV05 maps approved Governor evidence into an explainable, governed,
non-actionable recommendation state:

```text
approved Governor evidence
  -> factor states
  -> ME-GV04 factor scores
  -> recommendation eligibility
  -> recommendation direction
```

Eligibility and direction are separate decisions. An ineligible evaluation
cannot receive a directional state.

## Allowed States

Eligibility states are exactly:

```text
eligible
ineligible
```

Recommendation states are exactly:

```text
blocked
insufficient_evidence
avoid
watch
consider
preferred
```

These are interpretive Governor states, not trade actions. `preferred` does not
mean buy. `avoid` does not mean sell or remove. No state authorizes execution,
allocation, portfolio mutation, watchlist mutation, or delivery.

## Recommendation Review Boundary

Recommendation mapping may proceed only when the input preserves an approved
non-actionable Recommendation Review boundary:

```text
contract_version = sec-companyfacts-recommendation-review-v1
review_state = human_review_required
non_actionable = true
reference = non-empty deterministic reference
```

Missing or malformed boundary metadata blocks mapping. A valid Recommendation
Review state other than `human_review_required` is insufficient for mapping.
The Governor does not bypass or change Recommendation Review semantics.

## Critical Factor Policy

ME-GV04 implements numeric scoring for exactly:

```text
fundamentals
growth
risk
data_confidence
```

ME-GV05 therefore requires all four factors to:

* have state `evaluable`;
* have a numeric score under
  `market-engine-governor-factor-scoring-v1`;
* have no score limitation.

Valuation, trend, momentum, and technical setup must still satisfy the
ME-GV03 top-level complete-evaluation policy, but ME-GV05 does not invent
scores for them. Missing valuation remains missing and makes the Governor
evaluation incomplete.

Portfolio fit remains blocked without approved portfolio context. It is not a
critical factor for this portfolio-agnostic recommendation interpretation and
is disclosed as:

```text
portfolio_fit_not_used_without_approved_context
```

This policy creates no portfolio-aware state or allocation authority.

## Eligibility Precedence

The deterministic precedence is:

1. invalid Governor contract -> `blocked`;
2. globally blocked Governor evaluation -> `blocked`;
3. missing or invalid Recommendation Review boundary -> `blocked`;
4. valid but non-reviewable Recommendation Review state ->
   `insufficient_evidence`;
5. unresolved hard conflict -> `blocked`;
6. incomplete top-level Governor evaluation -> `insufficient_evidence`;
7. missing or invalid critical factor score -> `insufficient_evidence`;
8. critical score limitation -> `insufficient_evidence`;
9. data-confidence score below 75 -> `insufficient_evidence`;
10. eligible directional mapping.

Every failure produces a stable reason code and never substitutes missing
evidence with zero or a neutral value.

## Data-Confidence Boundary

Data confidence is an explicit gate:

```text
data_confidence >= 75.0
```

It is also an explicit directional condition at higher state thresholds. It is
never multiplied into another score and never changes another factor score.

## Directional Mapping

Mapping reads the four critical scores directly. It does not calculate an
average, weighted score, overall score, or rank.

### `avoid`

Map to `avoid` when any of these conditions is true:

```text
fundamentals < 40
growth < 40
risk < 40
```

For risk, higher means a more favorable lower-risk profile. A risk score below
40 is therefore an explicit unfavorable-risk guardrail.

### `preferred`

All conditions are required:

```text
fundamentals >= 75
growth >= 70
risk >= 70
data_confidence >= 85
```

### `consider`

All conditions are required:

```text
fundamentals >= 60
growth >= 55
risk >= 60
data_confidence >= 80
```

### `watch`

Eligible patterns that match none of the preceding rules map to `watch`.
Risk below 60 is disclosed through
`recommendation_limited_by_risk_guardrail`.

Threshold comparisons are inclusive at the favorable boundary and exclusive
at the unfavorable boundary.

## Conflict Policy

Hard conflicts use `conflicting_evidence_references`. They remain visible,
downgrade the factor under ME-GV03, and block recommendation eligibility
before the incomplete-evaluation check.

Soft conflicts use `soft_conflicting_evidence_references`. They remain visible
without downgrading an otherwise evaluable factor. An eligible `consider` or
`preferred` result is capped at `watch` and records:

```text
recommendation_limited_by_soft_conflict
```

Neither conflict type is silently averaged.

## Output Shape

Each recommendation output contains:

```text
contract_version
eligibility_state
state
reason_codes
supporting_factor_scores
supporting_factor_states
blocking_factors
conflict_references
limitations
actionable
recommendation_state_ready
decision_engine_ready
```

Supporting scores preserve factor identity, numeric score, and score-contract
identity. Supporting states preserve every canonical factor state.

## Reason Codes

Eligibility and mapping reason codes are deterministic and ordered by
evaluation precedence. The primary taxonomy is:

```text
blocked_invalid_governor_contract
blocked_governor_evaluation
blocked_recommendation_review_boundary_missing
blocked_recommendation_review_boundary_invalid
ineligible_recommendation_review_state
blocked_unresolved_hard_conflict
ineligible_governor_evaluation_incomplete
ineligible_critical_factor_coverage
ineligible_critical_score_limitations
ineligible_data_confidence_below_threshold
recommendation_eligible
mapped_unfavorable_critical_factor_pattern
mapped_mixed_critical_factor_pattern
mapped_moderately_favorable_critical_factor_pattern
mapped_favorable_critical_factor_pattern
recommendation_limited_by_soft_conflict
recommendation_limited_by_risk_guardrail
```

## Preserved Boundaries

ME-GV05 preserves:

```text
factor.weight = null
factor.weighted_score = null
overall_evaluation.score = null
overall_evaluation.weighted_score = null
overall_evaluation.rank = null
actionable = false
decision_engine_ready = false
buy_zone_explanation.state = blocked_not_authorized
position_management_explanation.state = blocked_not_authorized
```

`recommendation_state_ready` remains false. ME-GV05 authorizes mapping, but no
existing contract separately authorizes that reserved readiness state.
`actionable_review`, `decision_ready`, and `de_ready` also remain false.

ME-GV05 adds no price level, buy zone, stop, target, position-management
instruction, allocation, target weight, position size, order, broker action,
execution instruction, or Decision Engine decision.

## Next Sprint

```text
ME-GV06 - Implement buy-zone and position-management explanation contract
```
