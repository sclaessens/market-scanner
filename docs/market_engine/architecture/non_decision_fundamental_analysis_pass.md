# Non-Decision Fundamental Analysis Pass

Owner role: Technical Architect / Financial Analyst / Data Steward / QA / Test Lead

Status: ME12 IMPLEMENTED

## Purpose

ME12 adds the first Market Engine fundamental analysis pass.

This pass converts a fundamental source context into source-grounded observations. It does not produce scores, rankings, recommendations, allocation behavior, execution advice, reporting output, Telegram output, portfolio mutation, or watchlist mutation.

## Strict Non-Decision Scope

The ME12 pass is observational only.

It may describe whether approved source fields are present, missing, positive, negative, or zero. It may describe whether cash-generation source fields are complete.

It must not decide what an operator should do.

## Input

Input:

```text
FundamentalSourceContext
```

The source context provides:

- ticker;
- provider;
- source readiness;
- canonical source fields;
- missing canonical fields;
- provenance;
- period metadata;
- provider error category and message when applicable.

The analysis pass does not fetch provider data.

## Output

Output:

```text
FundamentalAnalysisPass
```

The pass contains source-grounded observations only.

Each observation preserves:

- ticker;
- provider;
- observation category;
- observation state;
- message;
- source readiness;
- canonical field references;
- source values where applicable;
- source references derived from provenance where available.

## Observation States

Allowed observation states:

- `POSITIVE`
- `NEGATIVE`
- `NEUTRAL`
- `MISSING_DATA`
- `NOT_ASSESSED`

These states describe source-backed observations only. They are not recommendations.

## Observation Categories

Allowed ME12 categories:

- `SOURCE_READINESS`
- `REVENUE_PRESENCE`
- `PROFITABILITY_PRESENCE`
- `OPERATING_CASH_FLOW_PRESENCE`
- `CAPEX_PRESENCE`
- `CASH_GENERATION_SOURCE_COMPLETENESS`
- `DATA_QUALITY`

ME12 does not add scoring, ranking, valuation, or allocation categories.

## Missing-Data Behavior

Missing data remains explicit.

If a field is missing, the observation uses `MISSING_DATA` or `NOT_ASSESSED`.

The pass does not convert missing data to zero, infer missing values from another field, use previous periods, or derive replacement metrics.

## Forbidden Outputs

ME12 does not emit:

- free cash flow;
- revenue growth;
- margins;
- profitability ratios;
- valuation metrics;
- quality scores;
- risk scores;
- rankings;
- recommendations;
- BUY / SELL / HOLD;
- allocation;
- conviction;
- urgency;
- tradeability;
- position sizing;
- execution advice.

## Future Layers

A later sprint may add derived observations only if the source context remains intact and the output remains non-decision. ME13 is expected to decide whether a first derived cash-generation observation layer is appropriate.
