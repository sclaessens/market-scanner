# Calculation Registry

Status: ACTIVE REGISTRY

## 1. Purpose

This registry documents calculations, formulas, derived states, and algorithmic logic used or planned in the market-scanner project.

The goal is to preserve useful analytical logic, expose misplaced or duplicated calculations, prevent silent logic drift, and make future implementation safer.

This document does not authorize implementation, code changes, runtime behavior changes, source-data changes, generated artifact changes, provider/API usage, scraping, or pipeline execution.

## 2. Registry Rule

A meaningful calculation should not be implemented or materially changed unless it is documented here or in a linked active calculation specification.

A calculation entry must clarify:

- what the calculation means;
- which layer owns it;
- whether it is descriptive or decision-authoritative;
- which inputs it uses;
- which output field it creates;
- whether it is ticker-category or sector dependent;
- which tests protect it;
- which downstream consumers rely on it.

## 3. Authority Boundary

Most calculations in the pipeline are descriptive. They classify, enrich, summarize, or prepare evidence.

Only the Decision Engine may convert evidence into allocation, execution, arbitration, final action, or portfolio decision semantics.

No calculation outside the Decision Engine may silently create:

- buy/sell decisions;
- final actions;
- allocation authority;
- tradeability;
- urgency;
- conviction;
- eligibility;
- ranking authority;
- scoring authority that changes opportunity treatment;
- hidden filtering.

## 4. Registry Entry Template

Use this template for future additions:

| Field | Required content |
|---|---|
| Calculation name | Stable name. |
| Owner layer | Scanner, Validation, Context, Fundamental Metrics, Fundamental Quality, Fundamental Analysis, Timing State, Portfolio Intelligence, Decision Engine, Reporting, or other approved layer. |
| Purpose | Why the calculation exists. |
| Inputs | Required input fields and artifacts. |
| Formula or algorithm | Exact formula, rule, or method. |
| Output artifact | Target artifact. |
| Output fields | Fields created or updated. |
| Semantics | Descriptive, diagnostic, quality classification, analysis classification, or Decision Engine authority. |
| Ticker-category relevance | General, sector-specific, category-specific, or requires future mapping. |
| Edge cases | Missing values, zero denominators, sign changes, stale data, duplicates, etc. |
| Review triggers | When human or governance review is required. |
| Tests required | Tests that should protect the calculation. |
| Current implementation | Current file or not yet implemented. |
| Status | Current, planned, candidate, deprecated, or requires review. |

## 5. Current Fundamental Calculation Family

The active detailed specification for fundamental calculations is:

- `docs/active/contracts/fundamental_calculations_technical_spec.md`

That document remains the formula authority for:

- gross margin;
- operating margin;
- net margin;
- debt to equity;
- return on equity;
- free cash flow margin;
- revenue year-over-year growth;
- EPS year-over-year growth;
- free cash flow year-over-year growth;
- 3-year CAGR calculations;
- 3-year average margins;
- 3-year trend helpers;
- raw data quality helper calculations.

This registry references that specification rather than duplicating all formulas.

## 6. Current Runtime Calculation Inventory

The table below is intentionally high level. It identifies calculation families and where detailed formulas or implementation references should live.

| Calculation family | Owner layer | Current implementation | Current artifact | Semantics | Status | Notes |
|---|---|---|---|---|---|---|
| Structure/setup classification | Validation | `scripts/core/build_validation_layer.py` and related validation logic | `data/processed/validation_layer.csv` | Descriptive structure classification | Current | Must not become tradeability or allocation authority. |
| Entry-quality metrics | Validation / Entry Quality | Entry quality builder/backfill logic | `data/processed/entry_quality_metrics.csv` | Descriptive entry-quality metrics | Current | Metrics such as extension, volume, and ATR-style checks must remain descriptive unless Decision Engine consumes them under governed scope. |
| Relative strength and leadership | Context | `scripts/core/build_context_layer.py` | `data/processed/context_strength.csv` | Descriptive context classification | Current | Useful for market leadership; no blocking or allocation authority upstream. |
| Fundamental quality MVP | Fundamental Layer | `scripts/core/build_fundamental_layer.py` | `data/processed/fundamental_quality.csv` | Data-readiness classification | Current / compatibility surface | Current layer combines source presence, metadata, sufficiency, and quality. Future implementation should separate raw history, metrics, quality, and analysis. |
| Fundamental metrics | Fundamental Metrics | `scripts/core/build_fundamental_metrics.py` | Future `data/processed/fundamental_metrics.csv` | Deterministic calculations only | Current | Calculates single-period margins, leverage, return on equity, and same-period prior-year growth from validated raw fundamentals history. Formula authority remains `fundamental_calculations_technical_spec.md`. |
| Fundamental analysis states | Fundamental Analysis | Not yet implemented as separate layer | Future `data/processed/fundamental_analysis.csv` | Descriptive business/fundamental classification | Planned | Must not create allocation, ranking, conviction, urgency, or tradeability semantics. |
| Timing state classification | Timing State | `scripts/core/build_timing_state_layer.py` | `data/processed/timing_state_layer.csv` | Descriptive timing/setup state | Current | Must not become execution gating outside Decision Engine. |
| Portfolio presence/exposure metadata | Portfolio Intelligence | `scripts/core/build_portfolio_intelligence.py` and portfolio builders | `data/processed/portfolio_intelligence.csv` | Descriptive portfolio metadata | Current | Must not create buy/sell authority. |
| Final decision arbitration | Decision Engine | `scripts/core/decision_engine.py` | `data/processed/final_decisions.csv` | Allocation/final-action authority | Current | Only authoritative decision layer. |
| Reporting grouping/summarization | Reporting | `scripts/reporting/build_reporting_layer.py` | `data/processed/reporting_dashboard_data.csv`; Telegram message | Communication-only | Current | May group and summarize but may not reinterpret decisions. |

## 7. Ticker Category Calculation Relevance

Future logic may need ticker-category-aware calculation mapping because different sectors and business types react differently to the same metrics.

Candidate future mapping dimensions:

| Category | Potentially relevant calculation focus | Notes |
|---|---|---|
| Semiconductors | Revenue growth, gross margin, operating margin, sector leadership, cycle sensitivity. | Must avoid hardcoded allocation conclusions. |
| Software | Revenue durability, margin expansion, free cash flow, recurring revenue if source-supported. | Recurring revenue requires new source-data contract before use. |
| Retail | Revenue consistency, gross margin, operating margin, free cash flow, consumer context. | Margin interpretation differs from software or semiconductors. |
| Energy | Free cash flow, debt, commodity sensitivity, cyclicality. | Commodity exposure requires separate context/source rules. |
| Biotech / healthcare innovation | Cash runway, catalysts, revenue maturity, pipeline stage. | Requires new source-data model before use. |
| Defensive compounders | Stability, drawdown behavior, cash flow consistency, margin durability. | Must remain descriptive upstream. |
| Cyclical growth | Revenue trend, margin trend, sector cycle, relative strength. | Requires careful regime/context handling. |

This mapping is a candidate direction only. It does not authorize implementation.

## 8. Calculation Placement Rules

Use these placement rules when reviewing or implementing calculations:

| Calculation type | Correct location |
|---|---|
| Raw market data transformations | Scanner or market-data utility layer. |
| Setup/structure classification | Validation Layer. |
| Entry-quality descriptive metrics | Validation / Entry Quality layer. |
| Market leadership and relative strength | Context Layer. |
| Raw financial statement data | Raw fundamentals history source artifact. |
| Deterministic financial metrics | Fundamental Metrics layer. |
| Data readiness/completeness | Fundamental Quality layer. |
| Business/fundamental characteristics | Fundamental Analysis layer. |
| Timing condition/state | Timing State layer. |
| Portfolio presence/exposure metadata | Portfolio Intelligence layer. |
| Allocation/final action | Decision Engine only. |
| Grouping and communication | Reporting only. |

## 9. Calculation Review Triggers

A calculation requires review when:

- it affects a data contract;
- it is used by more than one layer;
- it is duplicated in multiple files;
- it has hidden thresholds;
- it changes opportunity treatment;
- it is sector/category-specific;
- it uses stale or local ignored source data;
- it changes downstream artifact shape;
- it risks allocation semantics outside the Decision Engine;
- it lacks tests;
- it is not documented in this registry or a linked active spec.

## 10. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

This registry records how calculation logic should be governed. It does not add implementation scope by itself.

## 11. Validation

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
