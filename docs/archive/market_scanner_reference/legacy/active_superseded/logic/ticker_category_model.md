# Ticker Category Model Specification

Status: ACTIVE SPECIFICATION
Backlog driver: BL-0022
Related analysis: `docs/active/logic/strategy_logic_rationalization.md`

## 1. Purpose

This document defines the future ticker-category model for the market-scanner project.

The model exists because not every sector, business model, or financial profile reacts to the same signals in the same way. A semiconductor company, software company, energy producer, bank, retailer, biotech company, defensive compounder, and cyclical growth company should not all be interpreted through exactly the same analytical lens.

This document specifies the categories, allowed semantics, required source evidence, calculation relevance, pipeline placement, and future implementation boundaries.

This document does not authorize implementation, code changes, tests, data changes, generated artifact updates, provider/API usage, scraping, pipeline execution, allocation changes, or Decision Engine changes.

## 2. Core Rule

Ticker-category information is descriptive upstream.

It may later help decide:

- which calculations are relevant;
- which interpretation notes are useful;
- which data-quality checks matter;
- which analysis states require review;
- which category-specific context should be shown to the operator.

It may not create:

- buy/sell decisions;
- allocation authority;
- tradeability;
- urgency;
- conviction;
- eligibility;
- ranking authority;
- scoring authority;
- hidden filtering;
- Reporting-based decision semantics;
- Decision Engine bypass.

Only the Decision Engine may use category information for final action or allocation logic, and only after explicit future approval.

## 3. Model Design

A ticker should not have one overloaded category field. It should have multiple descriptive dimensions.

Proposed dimensions:

| Dimension | Required? | Purpose |
|---|---|---|
| `sector_group` | Yes | Broad economic sector context. |
| `business_model_group` | Yes | How the company earns money and which metrics are relevant. |
| `cycle_sensitivity` | Optional | How sensitive the business is to cycles, rates, or commodities. |
| `growth_profile` | Optional | Growth and maturity interpretation. |
| `financial_maturity` | Optional | Profitability/cash-flow stage. |
| `data_profile` | Optional | Whether category analysis needs special source data. |
| `portfolio_role` | Optional | Descriptive portfolio/watchlist role only. |

A future implementation may begin with only `sector_group` and `business_model_group` if that is safer.

## 4. Sector Groups

| Sector group | Description | Notes |
|---|---|---|
| `TECHNOLOGY` | Technology hardware, software, semiconductors, IT services. | Often needs business-model split. |
| `COMMUNICATIONS` | Platforms, media, telecom, digital advertising. | Revenue quality differs strongly by submodel. |
| `CONSUMER_DISCRETIONARY` | Retail, autos, travel, luxury, consumer cyclicals. | Often cycle-sensitive. |
| `CONSUMER_STAPLES` | Food, household products, staples retail. | Often defensive, margin and stability focused. |
| `HEALTHCARE` | Pharma, biotech, medtech, services. | Biotech requires special data model. |
| `FINANCIALS` | Banks, insurers, asset managers, fintech. | Standard industrial debt metrics may not apply directly. |
| `INDUSTRIALS` | Machinery, aerospace, defense, transport, industrial services. | Order cycle and macro context can matter. |
| `ENERGY` | Oil, gas, energy infrastructure, services. | Commodity sensitivity must be explicit. |
| `MATERIALS` | Mining, chemicals, metals, packaging. | Often commodity/cycle sensitive. |
| `UTILITIES` | Regulated utilities and energy distribution. | Rate sensitivity and debt context matter. |
| `REAL_ESTATE` | REITs and real estate operators. | Needs special metrics if implemented later. |
| `MULTI_SECTOR` | Conglomerates or mixed exposure. | Requires manual or source-supported classification. |
| `UNKNOWN` | Insufficient source-supported category data. | Must not be inferred silently. |

## 5. Business Model Groups

| Business model group | Description | Useful calculation focus | Caution |
|---|---|---|---|
| `SEMICONDUCTOR` | Chip designers, manufacturers, equipment, suppliers. | Revenue growth, gross margin, operating margin, sector leadership, cyclicality. | Cycles can distort short-term growth. |
| `SOFTWARE` | Software, SaaS, platforms, cloud tools. | Revenue durability, margin expansion, free cash flow, operating leverage. | Recurring revenue requires explicit source support. |
| `HARDWARE` | Devices, equipment, systems, electronics. | Margins, revenue trend, inventory/cycle context if source-supported. | Hardware economics differ from software. |
| `RETAIL` | Physical or digital retail. | Gross margin, operating margin, revenue consistency, consumer context. | Margin comparisons are category-specific. |
| `CONSUMER_BRAND` | Brand-led consumer companies. | Margin durability, revenue consistency, pricing power if source-supported. | Brand strength may need qualitative source data. |
| `ENERGY_PRODUCER` | Oil, gas, commodity energy producers. | Free cash flow, debt, commodity sensitivity, cycle context. | Commodity price exposure is context, not a direct signal. |
| `BANK` | Banks and lending institutions. | Requires future bank-specific metrics. | Standard debt/equity logic is not directly comparable. |
| `INSURANCE` | Insurance companies. | Requires insurance-specific metrics. | Industrial profitability metrics may mislead. |
| `ASSET_MANAGER` | Asset managers, exchanges, brokers. | Cash flow, revenue trend, market sensitivity. | AUM or market exposure may need source contract. |
| `BIOTECH` | Clinical-stage or innovation-driven healthcare. | Cash runway, catalysts, pipeline stage. | Requires separate source-data model. |
| `PHARMA_MEDTECH` | Mature pharma, medtech, healthcare products. | Revenue durability, margins, cash flow. | Pipeline exposure may still matter. |
| `INDUSTRIAL_CYCLICAL` | Machinery, transport, aerospace suppliers, capital goods. | Cycle context, margin trend, revenue trend. | Orders/backlog may require future source data. |
| `DEFENSE_INFRASTRUCTURE` | Defense, infrastructure, long-cycle contracts. | Revenue stability, backlog if source-supported, margin durability. | Backlog data needs source contract. |
| `UTILITY_REGULATED` | Regulated utilities. | Debt, cash flow, rate sensitivity. | Valuation/rate context may matter later. |
| `REIT_REAL_ESTATE` | REITs and property operators. | Requires real-estate-specific metrics. | FFO/AFFO require separate source contract. |
| `COMMODITY_MATERIALS` | Miners, metals, chemicals, commodity materials. | Cash flow, debt, cycle sensitivity. | Commodity context should be explicit. |
| `UNKNOWN` | No reliable business-model classification. | Basic general analysis only. | Do not infer category silently. |

## 6. Cycle Sensitivity

| Cycle sensitivity | Meaning |
|---|---|
| `DEFENSIVE` | Less sensitive to economic cycle; stability and cash-flow consistency matter. |
| `CYCLICAL` | Strongly tied to economic, demand, or capex cycles. |
| `COMMODITY_SENSITIVE` | Directly affected by commodity prices. |
| `RATE_SENSITIVE` | Directly affected by interest rates or financing conditions. |
| `HIGH_BETA_GROWTH` | Strongly affected by growth appetite, liquidity, or risk sentiment. |
| `EVENT_DRIVEN` | Catalysts, regulatory events, clinical events, or contract events matter. |
| `UNKNOWN` | Not classified yet. |

## 7. Growth Profile

| Growth profile | Meaning |
|---|---|
| `EARLY_GROWTH` | Growth is high but profitability may be immature. |
| `DURABLE_GROWTH` | Growth appears more established and repeatable. |
| `MATURE_COMPOUNDER` | Growth is moderate but durable with quality/stability focus. |
| `CYCLICAL_RECOVERY` | Growth may reflect recovery from cycle trough. |
| `TURNAROUND` | Improvement is possible but business quality is not yet stable. |
| `EX_DIVIDEND_STABILITY` | Stability, income, and balance sheet may matter more than growth. |
| `UNKNOWN` | Not classified yet. |

## 8. Financial Maturity

| Financial maturity | Meaning |
|---|---|
| `PROFITABLE_CASH_GENERATIVE` | Profitable and free-cash-flow positive. |
| `PROFITABLE_LOW_CASH_CONVERSION` | Profitable but cash conversion requires review. |
| `LOSS_MAKING_GROWTH` | Loss-making but growth-oriented. |
| `CASH_BURN_STAGE` | Cash runway and financing risk matter. |
| `BALANCE_SHEET_HEAVY` | Capital structure and debt context are central. |
| `FINANCIAL_SPECIFIC` | Needs sector-specific financial interpretation. |
| `UNKNOWN` | Not classified yet. |

## 9. Calculation Relevance Matrix

| Business model group | High relevance | Medium relevance | Special future metrics |
|---|---|---|---|
| `SEMICONDUCTOR` | Revenue YoY, gross margin, operating margin, free cash flow margin, sector relative strength. | Debt to equity, ROE. | Cycle phase, inventory, capex exposure. |
| `SOFTWARE` | Revenue YoY, operating margin, free cash flow margin, margin expansion. | ROE, debt to equity. | Recurring revenue, retention, rule-of-40 style metrics only if source-supported. |
| `RETAIL` | Gross margin, operating margin, revenue consistency, free cash flow margin. | Debt to equity, ROE. | Inventory, same-store sales if source-supported. |
| `ENERGY_PRODUCER` | Free cash flow, debt, cash-flow margin, commodity sensitivity. | Revenue YoY, net margin. | Commodity price context, reserves only if source-supported. |
| `BANK` | Future bank-specific metrics. | General revenue/profit trend with caution. | Net interest margin, credit quality, capital ratios. |
| `BIOTECH` | Cash runway, catalyst status, cash burn. | Revenue only if commercial-stage. | Pipeline stage, trial/regulatory events. |
| `DEFENSE_INFRASTRUCTURE` | Revenue durability, operating margin, cash flow. | Debt, ROE. | Backlog, contract duration if source-supported. |
| `REIT_REAL_ESTATE` | Future REIT-specific metrics. | Debt and cash flow with caution. | FFO/AFFO, occupancy, rate sensitivity. |
| `UNKNOWN` | General metrics only. | Category-specific metrics disabled. | Requires review. |

## 10. Pipeline Placement

Ticker-category metadata should eventually live in a descriptive source artifact or metadata layer, not inside the Decision Engine.

Candidate future artifact:

```text
data/reference/ticker_categories.csv
```

Potential fields:

- `ticker`
- `sector_group`
- `business_model_group`
- `cycle_sensitivity`
- `growth_profile`
- `financial_maturity`
- `data_profile`
- `portfolio_role`
- `category_source_name`
- `category_source_reference`
- `category_review_date`
- `category_confidence`
- `category_notes`

This artifact is proposed only. It is not authorized for implementation by this document.

## 11. Source and Review Policy

Ticker category assignment must be source-supported or manually reviewed.

Allowed future source types may include:

- company profile provider;
- exchange or sector classification source;
- manually reviewed local reference file;
- portfolio research note with source reference;
- official company description;
- approved data provider metadata.

Forbidden behavior:

- infer category from ticker symbol alone;
- infer category from price behavior alone;
- silently assign high-confidence categories without source evidence;
- use category to filter opportunities upstream;
- use category to create action labels outside the Decision Engine.

## 12. Category Confidence

Potential confidence values:

| Confidence | Meaning |
|---|---|
| `HIGH` | Clear source-supported classification. |
| `MEDIUM` | Source-supported but some business-model ambiguity remains. |
| `LOW` | Category requires review before category-specific logic is used. |
| `UNKNOWN` | No reliable category evidence available. |

Category-specific calculations should not be activated for `LOW` or `UNKNOWN` confidence without explicit review.

## 13. Implementation Boundaries

Future implementation should be split into small steps.

Recommended future sequence:

1. Documentation-only category model approval.
2. Define source artifact schema and fixture examples.
3. Add category validation helper and tests.
4. Add category metadata pass-through to analysis layer only.
5. Add calculation relevance mapping as descriptive metadata.
6. Decide whether and how the Decision Engine may later consume category context.

Do not combine this with raw-history fundamentals implementation unless explicitly scoped and capacity-checked.

## 14. Testing Requirements for Future Implementation

Future tests should verify:

- required category columns;
- allowed enum values;
- unknown category handling;
- source reference required for non-unknown category;
- no forbidden decision/action fields;
- no row filtering based on category;
- deterministic row preservation;
- category-specific calculation relevance is descriptive only;
- Decision Engine output does not change unless explicitly governed.

## 15. Recommended Next Step

Recommended next step after this specification:

```text
Choose between:
1. narrow Sprint E1 raw-history implementation;
2. narrow BL-0023 cleanup implementation scope;
3. category source-artifact schema specification.
```

If the goal is to make fundamentals implementation useful quickly, choose option 1.

If the goal is to make future category-aware analysis robust before code, choose option 3.

If the goal is to reduce codebase complexity before new logic, choose option 2.

Do not combine all three in one sprint.

## 16. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

BL-0022 covers ticker-category model rationalization and specification. BL-0015 covers fundamentals implementation. BL-0023 covers narrow Python cleanup scope.

## 17. Validation

Documentation-only validation for this change should confirm:

- only documentation files changed;
- no code files changed;
- no tests changed;
- no CSV files changed;
- no raw data changed;
- no generated files changed;
- no workflow files changed;
- no provider APIs called;
- no scraping performed;
- no pipeline run;
- no tests run unless explicitly needed.