# Financial Analysis Contract

Status: ACTIVE ANALYSIS CONTRACT

## Purpose

This document defines what financial analysis means in the market-scanner fundamentals platform.

Financial analysis interprets raw financial facts and calculated metrics descriptively. It does not create allocation decisions, buy/sell advice, ranking authority, tradeability, urgency, conviction, eligibility, or hidden filtering.

The Financial Analyst owns metric meaning and financial interpretation. Financial analysis remains descriptive until consumed by the Decision Engine under approved downstream contracts.

## Scope

Financial analysis covers:

- growth;
- margins;
- profitability;
- debt and balance-sheet leverage;
- cash-flow generation;
- consistency and trend interpretation;
- source-supported financial statement interpretation;
- review triggers when reported values or calculated metrics are ambiguous.

Financial analysis does not cover:

- code implementation;
- source-data entry;
- provider/API integration;
- runtime validation;
- portfolio allocation;
- final action;
- reporting prioritization;
- Telegram decision language.

## Approved financial concepts

The following concepts are approved for descriptive financial analysis:

| Concept | Meaning |
|---|---|
| Growth | Revenue, EPS, and free cash flow expansion or contraction over comparable periods. |
| Margins | Profitability or cash-flow intensity relative to revenue. |
| Profitability | Net income and return on equity characteristics. |
| Debt | Financial leverage and equity denominator quality. |
| Cash flow | Free cash flow level, margin, and durability. |
| Consistency | Multi-year stability or variability of financial characteristics. |
| Trend | Directional change over comparable fiscal years. |

## Metrics belonging to financial analysis

Financial analysis may interpret metrics defined by the technical calculation specification, including:

- `gross_margin`
- `operating_margin`
- `net_margin`
- `debt_to_equity`
- `return_on_equity`
- `free_cash_flow_margin`
- `revenue_growth_yoy`
- `eps_growth_yoy`
- `free_cash_flow_growth_yoy`
- `revenue_cagr_3y`
- `eps_cagr_3y`
- `free_cash_flow_cagr_3y`
- `average_gross_margin_3y`
- `average_operating_margin_3y`
- `operating_margin_trend_3y`
- `debt_to_equity_trend_3y`
- `free_cash_flow_margin_trend_3y`

Metrics may be interpreted only when the quality layer marks the required data as suitable or reviewable with explicit caveats.

## Raw facts, calculated metrics, and interpretation

The fundamentals platform separates three levels:

| Level | Description | Example | Authority |
|---|---|---|---|
| Raw financial facts | Source-supported reported statement values. | Revenue, net income, total debt. | Data Steward and source evidence. |
| Calculated metrics | Deterministic formulas applied to raw facts. | Net margin, revenue growth, debt-to-equity. | Technical calculation specification. |
| Financial interpretation | Descriptive meaning assigned to facts and metrics. | Margin pressure, improving cash generation, leverage review. | Financial Analyst. |

No level may create allocation authority.

## Growth interpretation

Growth analysis may describe:

- positive, flat, negative, volatile, or review-required growth;
- year-over-year growth;
- multi-year CAGR where mathematically meaningful;
- ambiguity caused by negative or zero starting values;
- gaps caused by missing or non-comparable periods.

Growth analysis must not state that a stock is attractive, buyable, actionable, or eligible for allocation. It may only describe the business growth profile.

## Margin interpretation

Margin analysis may describe:

- gross margin level and direction;
- operating margin level and direction;
- net margin level and direction;
- free cash flow margin level and direction;
- margin compression, expansion, volatility, or review-required conditions.

Margin levels must be interpreted relative to source-supported financial context. Cross-sector comparison requires later governance if it becomes systematic.

## Debt interpretation

Debt analysis may describe:

- leverage level using debt-to-equity;
- increasing or decreasing leverage trends;
- review-required cases caused by zero or negative equity;
- possible capital structure ambiguity.

Debt analysis must not create automatic exclusion, eligibility, rejection, or allocation decisions.

## Profitability interpretation

Profitability analysis may describe:

- positive or negative net income;
- return on equity when equity is positive and meaningful;
- review-required cases caused by negative equity or extreme ratios;
- consistency of profitability across periods.

Negative profitability may be descriptive evidence only. It is not a Decision Engine action.

## Cash-flow interpretation

Cash-flow analysis may describe:

- positive or negative free cash flow;
- free cash flow margin;
- free cash flow growth;
- sign changes;
- volatility;
- source-definition ambiguity.

Free cash flow definitions must be source-supported. If the source definition is unclear, the analysis state must be review-required.

## Source interpretation rules

The Financial Analyst may interpret financial values only when:

- values are source-supported;
- period metadata is clear;
- currency is clear;
- source freshness is documented;
- extraction evidence exists;
- calculated metrics follow the active technical specification.

If source evidence conflicts, the output must be review-required rather than inferred.

## IFRS vs US GAAP considerations

IFRS and US GAAP differences can affect metric comparability. Analysis must be careful around:

- operating income definition;
- free cash flow source definition;
- equity presentation;
- debt classification;
- restatements;
- currency and reporting-period differences.

A future implementation may add accounting-standard metadata. Until then, accounting-standard ambiguity should be handled with notes or review-required classification.

## Negative value handling

Negative values are not automatically errors.

| Case | Handling |
|---|---|
| Negative EPS | Valid raw value; growth and CAGR interpretation may require review. |
| Negative equity | Leverage and ROE metrics should be null or review-required. |
| Negative free cash flow | Valid raw value; growth and CAGR may require review when signs change. |
| Negative revenue | Unusual; review-required before metric interpretation. |
| Negative net income | Valid raw value; profitability state may be negative or review-required depending on context. |

## Human review requirements

Human review is required when:

- source definitions are unclear;
- values conflict across sources;
- period metadata is incomplete;
- currency changes across years;
- EPS or FCF sign changes make growth interpretation ambiguous;
- equity is zero or negative;
- metrics produce extreme values;
- reported values include obvious restatement or one-off caveats;
- data was manually extracted and has not been validated.

## Approved descriptive outputs

Financial analysis may produce descriptive states such as:

- `GROWTH_POSITIVE`
- `GROWTH_FLAT`
- `GROWTH_NEGATIVE`
- `GROWTH_VOLATILE`
- `MARGIN_EXPANDING`
- `MARGIN_STABLE`
- `MARGIN_COMPRESSING`
- `PROFITABLE`
- `LOSS_MAKING`
- `LEVERAGE_LOW`
- `LEVERAGE_ELEVATED`
- `CASHFLOW_POSITIVE`
- `CASHFLOW_NEGATIVE`
- `REVIEW_REQUIRED`

These labels are descriptive only and must not be treated as allocation instructions.

## Handoff

Financial Analyst handoff to Functional Analyst:

- metric meanings;
- review triggers;
- user-facing interpretation requirements;
- analysis-state definitions.

Financial Analyst handoff to Technical Analyst:

- formulas requiring implementation;
- edge cases;
- data requirements;
- validation expectations.

Financial Analyst handoff to Decision Engine governance:

- descriptive fundamental states only;
- no final decision semantics.