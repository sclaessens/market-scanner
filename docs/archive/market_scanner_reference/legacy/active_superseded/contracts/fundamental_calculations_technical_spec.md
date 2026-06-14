# Fundamental Calculations Technical Specification

Status: ACTIVE TECHNICAL REFERENCE

## Purpose and authority

This document defines calculation logic for the simplified fundamentals platform. It exists so formulas can later be implemented in small deterministic algorithms, optimized, compared, tested, and maintained outside scattered sprint documents.

This document defines calculations only. No calculation creates buy/sell advice. No calculation has allocation authority. Analysis layers may consume calculated values descriptively only. The Decision Engine remains the only allocation, execution, arbitration, and final-action authority.

## General calculation rules

All calculations must be deterministic for the same inputs.

Inputs come from raw historical statement data, primarily `data/raw/fundamentals_history.csv`, unless a future approved implementation spec defines a compatible source.

Outputs belong in `data/processed/fundamental_metrics.csv` or helper fields used by later quality/analysis layers.

Missing, invalid, zero-denominator, sign-change, period-mismatch, or currency-mismatch cases must not be guessed. They must produce null metric output plus a review or quality helper signal.

Metric output must not imply ranking, scoring, urgency, conviction, eligibility, tradeability, allocation, filtering, or final action.

## Standard calculation template

Each calculation must define:

- calculation name;
- purpose;
- required raw inputs;
- formula;
- period requirement;
- output field;
- output type;
- unit convention;
- missing data behavior;
- edge cases;
- review triggers;
- possible alternative formulas;
- optimization notes;
- test cases needed.

## Single-period metrics

### gross_margin

| Field | Specification |
|---|---|
| Purpose | Measure gross profitability relative to revenue. |
| Required raw inputs | `gross_profit`, `revenue`. |
| Formula | `gross_profit / revenue`. |
| Period requirement | Same ticker and same fiscal period. |
| Output field | `gross_margin`. |
| Output type | Decimal float. |
| Unit convention | Ratio, e.g. `0.42` means 42%. |
| Missing data behavior | Null if either input is missing. |
| Edge cases | Null if revenue is zero. Negative revenue requires review. |
| Review triggers | Missing inputs, zero revenue, negative revenue, unusual gross profit sign. |
| Possible alternative formulas | Use directly reported gross margin only if explicitly approved and source-defined. |
| Optimization notes | Vectorizable row-level calculation. |
| Test cases needed | Normal positive inputs, missing input, zero revenue, negative revenue. |

### operating_margin

| Field | Specification |
|---|---|
| Purpose | Measure operating profitability relative to revenue. |
| Required raw inputs | `operating_income`, `revenue`. |
| Formula | `operating_income / revenue`. |
| Period requirement | Same ticker and same fiscal period. |
| Output field | `operating_margin`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if either input is missing. |
| Edge cases | Null if revenue is zero. Negative operating income is valid but may inform analysis. |
| Review triggers | Missing inputs, zero revenue, negative revenue. |
| Possible alternative formulas | Use EBIT margin only if operating income is unavailable and a future contract approves EBIT substitution. |
| Optimization notes | Keep operating income definition stable across sources. |
| Test cases needed | Positive margin, negative margin, missing data, zero revenue. |

### net_margin

| Field | Specification |
|---|---|
| Purpose | Measure net profitability relative to revenue. |
| Required raw inputs | `net_income`, `revenue`. |
| Formula | `net_income / revenue`. |
| Period requirement | Same ticker and same fiscal period. |
| Output field | `net_margin`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if either input is missing. |
| Edge cases | Null if revenue is zero. Negative net income is valid but may trigger descriptive analysis. |
| Review triggers | Missing inputs, zero revenue, negative revenue, major unusual items if noted. |
| Possible alternative formulas | Directly reported net margin only when source-defined and approved. |
| Optimization notes | This removes the need to add `net_margin` to raw data when `net_income` and `revenue` exist. |
| Test cases needed | Positive net income, negative net income, missing inputs, zero revenue. |

### debt_to_equity

| Field | Specification |
|---|---|
| Purpose | Measure financial leverage. |
| Required raw inputs | `total_debt`, `total_equity`. |
| Formula | `total_debt / total_equity`. |
| Period requirement | Same ticker and same fiscal period. |
| Output field | `debt_to_equity`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if either input is missing. |
| Edge cases | Null and review-required if total equity is zero or negative. Negative debt requires review. |
| Review triggers | Missing inputs, zero equity, negative equity, negative debt. |
| Possible alternative formulas | Net debt to equity, total liabilities to equity, or debt to capital in future approved specs. |
| Optimization notes | Keep total debt definition source-stable. |
| Test cases needed | Positive equity, zero equity, negative equity, missing debt, missing equity. |

### return_on_equity

| Field | Specification |
|---|---|
| Purpose | Measure profitability relative to shareholder equity. |
| Required raw inputs | `net_income`, `total_equity`. |
| Formula | `net_income / total_equity`. |
| Period requirement | Same ticker and same fiscal period. |
| Output field | `return_on_equity`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if either input is missing. |
| Edge cases | Null and review-required if equity is zero or negative. Negative net income is valid but may inform analysis. |
| Review triggers | Missing inputs, zero equity, negative equity, extreme ratio. |
| Possible alternative formulas | Average equity denominator if opening equity becomes available. |
| Optimization notes | Current formula uses period-end equity until average equity is explicitly supported. |
| Test cases needed | Positive ROE, negative net income, zero equity, negative equity. |

### free_cash_flow_margin

| Field | Specification |
|---|---|
| Purpose | Measure cash generation relative to revenue. |
| Required raw inputs | `free_cash_flow`, `revenue`. |
| Formula | `free_cash_flow / revenue`. |
| Period requirement | Same ticker and same fiscal period. |
| Output field | `free_cash_flow_margin`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if either input is missing. |
| Edge cases | Null if revenue is zero. Negative free cash flow is valid but may inform analysis. |
| Review triggers | Missing inputs, zero revenue, negative revenue, ambiguous FCF definition. |
| Possible alternative formulas | Operating cash flow minus capex if both become raw fields. |
| Optimization notes | Free cash flow must remain source-supported and consistently defined. |
| Test cases needed | Positive FCF, negative FCF, missing input, zero revenue. |

## Year-over-year metrics

### revenue_growth_yoy

| Field | Specification |
|---|---|
| Purpose | Measure annual revenue growth. |
| Required raw inputs | Current `revenue`, previous comparable fiscal-year `revenue`. |
| Formula | `(current_revenue - previous_revenue) / previous_revenue`. |
| Period requirement | Same ticker, comparable fiscal period, consecutive fiscal years. |
| Output field | `revenue_growth_yoy`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if current or previous value is missing. |
| Edge cases | Null if previous revenue is zero or negative. |
| Review triggers | Missing year, non-consecutive years, zero or negative previous revenue, currency mismatch. |
| Possible alternative formulas | Constant-currency growth if source-supported. |
| Optimization notes | Use fiscal-year ordering, not extraction-date ordering. |
| Test cases needed | Normal growth, decline, missing previous year, zero previous revenue, non-consecutive years. |

### eps_growth_yoy

| Field | Specification |
|---|---|
| Purpose | Measure annual diluted EPS growth while handling negative bases more transparently. |
| Required raw inputs | Current `diluted_eps`, previous comparable fiscal-year `diluted_eps`. |
| Formula | `(current_diluted_eps - previous_diluted_eps) / abs(previous_diluted_eps)`. |
| Period requirement | Same ticker, comparable fiscal period, consecutive fiscal years. |
| Output field | `eps_growth_yoy`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if current or previous EPS is missing. |
| Edge cases | Null if previous EPS is zero. Sign changes are review-required. |
| Review triggers | Missing values, zero previous EPS, EPS sign change, non-consecutive years. |
| Possible alternative formulas | Standard denominator `previous_eps`; adjusted EPS growth if adjusted EPS source is approved. |
| Optimization notes | Store sign-change helper metadata for analysis review. |
| Test cases needed | Positive-to-positive, positive-to-negative, negative-to-positive, zero previous EPS. |

### free_cash_flow_growth_yoy

| Field | Specification |
|---|---|
| Purpose | Measure annual free cash flow growth. |
| Required raw inputs | Current `free_cash_flow`, previous comparable fiscal-year `free_cash_flow`. |
| Formula | `(current_free_cash_flow - previous_free_cash_flow) / abs(previous_free_cash_flow)`. |
| Period requirement | Same ticker, comparable fiscal period, consecutive fiscal years. |
| Output field | `free_cash_flow_growth_yoy`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if current or previous FCF is missing. |
| Edge cases | Null if previous FCF is zero. Sign changes are review-required. |
| Review triggers | Missing values, zero previous FCF, sign change, ambiguous FCF definition. |
| Possible alternative formulas | CAGR or rolling-average FCF growth for volatile businesses. |
| Optimization notes | Preserve raw FCF signs and separate interpretation from calculation. |
| Test cases needed | Positive growth, decline, negative-to-positive, positive-to-negative, zero previous FCF. |

## Multi-year metrics

### revenue_cagr_3y

| Field | Specification |
|---|---|
| Purpose | Measure 3-year compound revenue growth. |
| Required raw inputs | Revenue for start year and end year. |
| Formula | `(end_revenue / start_revenue) ** (1 / 3) - 1`. |
| Period requirement | Same ticker, comparable fiscal period, start and end years exactly 3 fiscal years apart. |
| Output field | `revenue_cagr_3y`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if required years are missing. |
| Edge cases | Null if start revenue is zero or negative. |
| Review triggers | Missing start/end year, currency mismatch, non-comparable fiscal period. |
| Possible alternative formulas | 5-year CAGR when enough history exists. |
| Optimization notes | Use fiscal-year index and period consistency helper. |
| Test cases needed | Normal CAGR, negative growth, missing start year, zero start value. |

### eps_cagr_3y

| Field | Specification |
|---|---|
| Purpose | Measure 3-year compound EPS growth when interpretation is mathematically safe. |
| Required raw inputs | Diluted EPS for start year and end year. |
| Formula | `(end_diluted_eps / start_diluted_eps) ** (1 / 3) - 1`. |
| Period requirement | Same ticker, comparable fiscal period, start and end years exactly 3 fiscal years apart. |
| Output field | `eps_cagr_3y`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if required years are missing. |
| Edge cases | Null and review-required if start EPS is zero or negative. Sign changes are review-required. |
| Review triggers | Missing years, zero start EPS, negative start EPS, sign change, non-comparable periods. |
| Possible alternative formulas | Absolute-change trend, median EPS growth, or review-only classification for sign-change cases. |
| Optimization notes | Do not force CAGR when sign behavior makes interpretation ambiguous. |
| Test cases needed | Positive start/end, zero start, negative start, sign change. |

### free_cash_flow_cagr_3y

| Field | Specification |
|---|---|
| Purpose | Measure 3-year compound free cash flow growth when interpretation is safe. |
| Required raw inputs | Free cash flow for start year and end year. |
| Formula | `(end_free_cash_flow / start_free_cash_flow) ** (1 / 3) - 1`. |
| Period requirement | Same ticker, comparable fiscal period, start and end years exactly 3 fiscal years apart. |
| Output field | `free_cash_flow_cagr_3y`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if required years are missing. |
| Edge cases | Null and review-required if start FCF is zero or negative. Sign changes are review-required. |
| Review triggers | Missing years, zero or negative start FCF, sign change, ambiguous FCF definition. |
| Possible alternative formulas | 3-year average FCF margin trend or cumulative FCF growth. |
| Optimization notes | Treat volatile or sign-changing FCF conservatively. |
| Test cases needed | Positive CAGR, negative CAGR, zero start, negative start, sign change. |

### average_gross_margin_3y

| Field | Specification |
|---|---|
| Purpose | Smooth gross margin over 3 fiscal years. |
| Required raw inputs | `gross_margin` for three comparable fiscal years. |
| Formula | Arithmetic average of available 3-year gross margin values when all required years exist. |
| Period requirement | Three consecutive comparable fiscal years. |
| Output field | `average_gross_margin_3y`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if fewer than three comparable values exist. |
| Edge cases | Review if any underlying year has zero or negative revenue. |
| Review triggers | Missing years, missing metric inputs, period inconsistency. |
| Possible alternative formulas | Median margin or weighted average by revenue. |
| Optimization notes | Calculate after single-period margin outputs. |
| Test cases needed | Complete 3-year set, one missing year, one invalid revenue year. |

### average_operating_margin_3y

| Field | Specification |
|---|---|
| Purpose | Smooth operating profitability over 3 fiscal years. |
| Required raw inputs | `operating_margin` for three comparable fiscal years. |
| Formula | Arithmetic average of three comparable operating margin values. |
| Period requirement | Three consecutive comparable fiscal years. |
| Output field | `average_operating_margin_3y`. |
| Output type | Decimal float. |
| Unit convention | Ratio. |
| Missing data behavior | Null if fewer than three comparable values exist. |
| Edge cases | Negative margins are valid but descriptive analysis may classify them separately. |
| Review triggers | Missing years, invalid revenue, period mismatch. |
| Possible alternative formulas | Median margin or revenue-weighted operating margin. |
| Optimization notes | Keep smoothing separate from trend direction. |
| Test cases needed | Complete values, negative margin, missing year, invalid revenue. |

### operating_margin_trend_3y

| Field | Specification |
|---|---|
| Purpose | Describe 3-year operating margin direction. |
| Required raw inputs | Operating margin for start, middle, and end years. |
| Formula | Preferred initial method: latest-vs-earliest comparison with optional direction label. |
| Period requirement | Three consecutive comparable fiscal years. |
| Output field | `operating_margin_trend_3y`. |
| Output type | Descriptive metric value or label, not score. |
| Unit convention | Ratio delta or direction label. |
| Missing data behavior | Null if required years are missing. |
| Edge cases | Mixed signs are valid but require careful analysis. |
| Review triggers | Missing years, period mismatch, large discontinuities. |
| Possible alternative formulas | Linear slope over 3 points; monotonic direction check. |
| Optimization notes | Direction is preferred over score/ranking semantics. |
| Test cases needed | Improving, deteriorating, flat, missing year, mixed signs. |

### debt_to_equity_trend_3y

| Field | Specification |
|---|---|
| Purpose | Describe leverage direction over 3 years. |
| Required raw inputs | Debt-to-equity values for three comparable fiscal years. |
| Formula | Preferred initial method: latest-vs-earliest comparison. |
| Period requirement | Three consecutive comparable fiscal years. |
| Output field | `debt_to_equity_trend_3y`. |
| Output type | Descriptive metric value or label, not score. |
| Unit convention | Ratio delta or direction label. |
| Missing data behavior | Null if required years are missing. |
| Edge cases | Negative or zero equity years produce null metric and review-required status. |
| Review triggers | Missing years, invalid equity, major capital structure event. |
| Possible alternative formulas | Linear slope; debt-to-capital trend. |
| Optimization notes | Higher or lower debt is not automatically good or bad; analysis remains descriptive. |
| Test cases needed | Increasing leverage, decreasing leverage, invalid equity, missing year. |

### free_cash_flow_margin_trend_3y

| Field | Specification |
|---|---|
| Purpose | Describe cash-flow margin direction over 3 years. |
| Required raw inputs | Free cash flow margin for three comparable fiscal years. |
| Formula | Preferred initial method: latest-vs-earliest comparison. |
| Period requirement | Three consecutive comparable fiscal years. |
| Output field | `free_cash_flow_margin_trend_3y`. |
| Output type | Descriptive metric value or label, not score. |
| Unit convention | Ratio delta or direction label. |
| Missing data behavior | Null if required years are missing. |
| Edge cases | Sign changes are valid but review-required for interpretation. |
| Review triggers | Missing years, sign changes, invalid revenue, FCF definition ambiguity. |
| Possible alternative formulas | Linear slope; rolling average comparison. |
| Optimization notes | Keep trend output descriptive and separate from investment conclusions. |
| Test cases needed | Improving, deteriorating, sign change, missing year, invalid revenue. |

## Data quality helper calculations

### fiscal_year_count_available

| Field | Specification |
|---|---|
| Purpose | Count available fiscal years per ticker. |
| Required raw inputs | `ticker`, `fiscal_year`, `fiscal_period`. |
| Formula | Count distinct fiscal years for approved period type. |
| Period requirement | Same ticker and comparable fiscal period. |
| Output field | `fiscal_year_count_available`. |
| Output type | Integer. |
| Unit convention | Count. |
| Missing data behavior | Zero if no rows exist. |
| Edge cases | Duplicate fiscal-year rows require review. |
| Review triggers | Duplicate periods, unknown fiscal period. |
| Possible alternative formulas | Count completed annual periods only. |
| Optimization notes | Useful for raw history readiness. |
| Test cases needed | No rows, one row, three years, duplicate year. |

### consecutive_years_available

| Field | Specification |
|---|---|
| Purpose | Determine consecutive fiscal-year coverage. |
| Required raw inputs | `ticker`, `fiscal_year`, `fiscal_period`. |
| Formula | Longest consecutive fiscal-year run for comparable period type. |
| Period requirement | Same ticker and comparable fiscal period. |
| Output field | `consecutive_years_available`. |
| Output type | Integer. |
| Unit convention | Count. |
| Missing data behavior | Zero if no rows exist. |
| Edge cases | Duplicate years or fiscal calendar changes require review. |
| Review triggers | Gaps, duplicate years, mixed period types. |
| Possible alternative formulas | Boolean `has_3_consecutive_years`. |
| Optimization notes | Supports 3-year metric readiness. |
| Test cases needed | 2022-2024, 2020/2022/2024 gaps, duplicates. |

### missing_required_raw_fields_count

| Field | Specification |
|---|---|
| Purpose | Count missing required raw fields per row. |
| Required raw inputs | Required raw fields from the platform contract. |
| Formula | Count null or empty required fields. |
| Period requirement | Row-level. |
| Output field | `missing_required_raw_fields_count`. |
| Output type | Integer. |
| Unit convention | Count. |
| Missing data behavior | Count missing fields. |
| Edge cases | Distinguish zero values from missing values. |
| Review triggers | Any required source or period metadata missing. |
| Possible alternative formulas | Separate metadata-missing and value-missing counts. |
| Optimization notes | Supports quality states. |
| Test cases needed | Complete row, missing source, zero numeric value, missing numeric value. |

### missing_metric_inputs_count

| Field | Specification |
|---|---|
| Purpose | Count missing inputs required to calculate approved metrics. |
| Required raw inputs | Metric input dependency list. |
| Formula | Count missing dependencies for a metric set. |
| Period requirement | Row-level and ticker-period-level. |
| Output field | `missing_metric_inputs_count`. |
| Output type | Integer. |
| Unit convention | Count. |
| Missing data behavior | Count missing dependencies. |
| Edge cases | Zero denominator is invalid, not merely missing. |
| Review triggers | Missing required inputs or invalid denominators. |
| Possible alternative formulas | Per-metric missing input flags. |
| Optimization notes | Should be generated from dependency metadata where possible. |
| Test cases needed | Complete dependencies, one missing input, zero denominator. |

### period_consistency_flag

| Field | Specification |
|---|---|
| Purpose | Detect whether periods are comparable. |
| Required raw inputs | `fiscal_year`, `fiscal_period`, `period_end_date`. |
| Formula | Boolean flag based on comparable fiscal period sequence. |
| Period requirement | Ticker-level across years. |
| Output field | `period_consistency_flag`. |
| Output type | Boolean or descriptive label. |
| Unit convention | Flag. |
| Missing data behavior | False or review-required if required metadata is missing. |
| Edge cases | Fiscal year-end changes, annual vs quarterly mix. |
| Review triggers | Mixed period types, missing period end dates, fiscal calendar shifts. |
| Possible alternative formulas | Period consistency category: `CONSISTENT`, `MIXED_PERIODS`, `REVIEW_REQUIRED`. |
| Optimization notes | Keep logic transparent and testable. |
| Test cases needed | Annual sequence, mixed quarterly/annual, missing dates. |

### currency_consistency_flag

| Field | Specification |
|---|---|
| Purpose | Detect whether multi-year values use the same currency. |
| Required raw inputs | `currency`. |
| Formula | Boolean flag based on distinct currency count per ticker and comparable period sequence. |
| Period requirement | Ticker-level across years. |
| Output field | `currency_consistency_flag`. |
| Output type | Boolean or descriptive label. |
| Unit convention | Flag. |
| Missing data behavior | False or review-required if currency is missing. |
| Edge cases | Source restatements, reporting currency changes, ADRs. |
| Review triggers | Multiple currencies, missing currency, changed reporting currency. |
| Possible alternative formulas | Source-supported normalized currency conversion in a future spec. |
| Optimization notes | Do not convert currencies unless a future approved contract defines conversion policy. |
| Test cases needed | Same currency, mixed currency, missing currency. |

## CAGR interpretation rules

CAGR requires a valid start year and end year separated by the exact period length. For 3-year CAGR, the end year must be exactly three fiscal years after the start year and must use comparable fiscal periods.

If the starting value is zero or negative, CAGR output must be null and review-required. EPS and free cash flow CAGR must be review-required when sign changes make interpretation ambiguous.

## Trend interpretation rules

The preferred first implementation for 3-year trends is simple latest-vs-earliest comparison or descriptive direction labels. Linear slope may be evaluated later as an alternative. No trend formula may create score, ranking, eligibility, urgency, conviction, tradeability, allocation, or buy/sell semantics.

## Testing requirements

A future implementation spec should include focused tests for:

- normal positive calculations;
- missing inputs;
- zero denominators;
- negative denominators;
- sign changes;
- duplicate periods;
- non-consecutive fiscal years;
- period inconsistency;
- currency inconsistency;
- exact output column names;
- no forbidden allocation or ranking fields.