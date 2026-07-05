# ME-GV04 - Governor Factor Scoring Contract

Owner roles: Product Owner / Technical Architect / Financial Analyst / Data Steward / QA Lead / Governance Auditor

Job family: ME-GV - The Governor

Status: IMPLEMENTED

Contract version: `market-engine-governor-factor-scoring-v1`

## Purpose

ME-GV04 adds deterministic, transparent factor scoring to the ME-GV03
non-actionable Governor scaffold:

```text
approved analysis evidence
  -> ME-GV03 evidence readiness and factor state
  -> ME-GV04 factor-specific scoring
  -> score, evidence components, and limitations
  -> non-actionable Governor artifact
```

Factor state remains evidence sufficiency. Factor score is the result of
explicit approved scoring logic. The two concepts are not interchangeable.

## Score Eligibility and Scale

Only a factor whose state is exactly `evaluable` may receive a numeric score.
Every other canonical state must retain `score = null`.

The uniform scale is:

```text
minimum: 0.0
midpoint: 50.0
maximum: 100.0
```

Higher always means a more favorable factor assessment. For `risk`, a higher
score therefore means a more favorable, lower-risk profile. Scores are rounded
to two decimal places using decimal half-up rounding.

This is not a recommendation scale. A high or low factor score has no BUY,
SELL, HOLD, urgency, conviction, tradeability, ranking, or allocation meaning.

## Approved Score Input Shape

Scored evidence declares:

```text
score_inputs.contract_version
score_inputs.components
```

Every component contains:

```text
component_id
evidence_reference
input_value
normalization_rule
limitations
```

The evidence reference must also occur in the factor's approved
`evidence_references`. Inputs must be finite numbers; booleans, strings, nulls,
NaN, and infinity are rejected. Missing components remain missing and are
never zero-filled.

## Implemented Factor Scorers

ME-GV04 deliberately implements only four scorers for which the local approved
fixture contract supplies explicit interpretable numerical evidence.

### Fundamentals

All three dimensions are required and contribute equally:

| Component | Normalization |
| --- | --- |
| `profitability_margin` | Linear from -10% to 30%, clamped |
| `operating_cash_flow_margin` | Linear from -10% to 30%, clamped |
| `return_on_assets` | Linear from -5% to 20%, clamped |

No single metric can produce the factor score.

### Growth

All three aligned multi-period dimensions are required:

| Component | Normalization |
| --- | --- |
| `revenue_growth_rate` | Linear from -10% to 30%, clamped |
| `earnings_growth_rate` | Linear from -20% to 40%, clamped |
| `cash_flow_growth_rate` | Linear from -20% to 40%, clamped |

`period_alignment` must equal `aligned_multi_period` and `period_count` must be
at least three. An isolated period or malformed alignment cannot score.

### Risk

All three dimensions are required:

| Component | Normalization |
| --- | --- |
| `debt_to_assets_ratio` | Inverse linear from 20% to 80%, clamped |
| `net_debt_to_cash_flow` | Inverse linear from 0 to 5, clamped |
| `cash_coverage_ratio` | Linear from 0 to 2, clamped |

The inverse rules make greater leverage less favorable. Missing risk evidence
does not imply either safety or danger.

### Data Confidence

All three ratios are required and normalized linearly from 0 to 1:

```text
source_support_ratio
provenance_completeness_ratio
evidence_completeness_ratio
```

Data confidence is an independent factor. It is not a multiplier, booster, or
penalty applied to any investment factor.

## Deliberately Unimplemented Scorers

The following factor scorers remain unavailable:

```text
valuation
trend
momentum
technical_setup
portfolio_fit
```

Their ME-GV02 state evaluation remains active, but an `evaluable` state alone
does not manufacture a score. No valuation ratio, peer group, technical
indicator, lookback, threshold, or portfolio context is invented.
`portfolio_fit` remains blocked without an approved Portfolio Context contract.

## Component Contributions

Each required component is normalized to 0-100 and contributes an equal share
to its factor score. Outputs expose the input, named normalization rule,
normalized value, normalized contribution, evidence reference, and
limitations. The factor score is the arithmetic mean of the unrounded
normalized component values and is then rounded under the precision policy.

These component contributions are factor-internal calculations. They are not
factor weights and do not aggregate factors.

## Fail-Closed and Conflict Behavior

A score remains null when any of the following applies:

* factor state is not `evaluable`;
* scoring contract identity is absent or unapproved;
* a required component is missing or duplicated;
* an input is missing, malformed, or non-finite;
* a normalization rule is absent or unexpected;
* an evidence reference is not approved for the factor;
* growth period alignment is invalid;
* evidence is stale, unprovenanced, malformed, or non-consumable;
* conflicting evidence prevents an evaluable state;
* no factor-specific scorer is implemented.

Conflict references remain visible. ME-GV03 downgrades complete conflicting
evidence to `partial`, which makes the score ineligible. ME-GV04 never silently
averages conflicting evidence.

## Aggregation and Authority Boundary

ME-GV04 preserves:

```text
factor.weight = null
factor.weighted_score = null
overall_evaluation.score = null
overall_evaluation.weighted_score = null
overall_evaluation.rank = null
```

Recommendation state, buy-zone explanation, and position-management
explanation remain `blocked_not_authorized`. Actionable, actionable-review,
recommendation-state-ready, decision-ready, and DE-ready states remain
unreachable.

ME-GV04 adds no recommendation mapping, allocation, target weight, position
sizing, order, execution instruction, provider/network call, delivery, or
portfolio/watchlist mutation. The Decision Engine remains the only allocation
authority.

## Next Sprint

```text
ME-GV05 - Implement recommendation-state mapping under approved boundary
```
