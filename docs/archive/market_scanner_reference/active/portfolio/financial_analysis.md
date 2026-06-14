# Financial Analysis

## 1. Purpose

This document defines the financial analysis scope for the v2 market scanner.

It explains what the application must eventually understand about companies, financial data, portfolio relevance, and investment context. It is a PM and analyst document, not an implementation specification and not a trading model.

This document is canonical for financial analysis intent. It must be read together with:

- `docs/active/product_vision.md`
- `docs/active/project_charter.md`
- `docs/active/functional_analysis.md`
- `docs/active/source_data_strategy.md`
- `docs/active/data_contracts.md`
- `docs/active/decision_engine_contract.md`
- `docs/active/reporting_contract.md`

## 2. Product Goal Connection

The market scanner exists to support disciplined investment review.

The application must help identify, structure, and explain investment-relevant information so that portfolio decisions can be made consistently and with traceability.

The scanner must not behave as an uncontrolled recommendation engine. It must separate information gathering, classification, financial interpretation, decision authority, and communication.

The financial analysis layer exists to answer questions such as:

- What kind of business is this company?
- Is the company financially healthy enough to keep under review?
- Are profitability, growth, balance sheet, and cash generation directionally understandable?
- Is valuation context available and interpretable?
- Is the source data complete enough to support further analysis?
- Which uncertainties must remain visible?
- Which findings should be passed downstream as classification, not as final actions?

## 3. Governance Doctrine

Financial analysis is upstream classification.

The following doctrine is mandatory:

- Classification upstream.
- Allocation downstream.
- Decision Engine is the only allocation, execution, arbitration, and final-action authority.
- Reporting communicates only.
- No hidden filtering.
- No upstream tradeability.
- Missing source-data values are not zero.
- Source-data readiness is not investment quality.
- Generated outputs are not source-of-truth unless explicitly approved.

Financial analysis may describe conditions, risks, uncertainty, and context. It must not create final actions.

## 4. Financial Analysis Is Not a Decision Engine

Financial analysis may eventually produce structured observations such as:

- business quality classification;
- financial health classification;
- growth profile classification;
- profitability profile classification;
- valuation context classification;
- balance sheet risk classification;
- cash generation classification;
- dividend or capital allocation context;
- source-data completeness and uncertainty notes.

Financial analysis must not produce:

- buy actions;
- sell actions;
- hold actions;
- position sizing;
- allocation amounts;
- execution instructions;
- tradeability states;
- urgency rankings;
- conviction scores;
- hidden filtering decisions;
- portfolio transaction advice.

Any future final-action field belongs exclusively to the Decision Engine.

## 5. Core Financial Questions

The v2 market scanner must eventually support structured review across the following financial question groups.

### 5.1 Business Understanding

The scanner should help identify what the company does, how it makes money, and which economic drivers matter.

Possible future dimensions:

- business model clarity;
- sector and industry context;
- revenue drivers;
- cyclicality;
- competitive position indicators;
- dependency on macro, regulation, technology, or commodities;
- geographic exposure;
- customer or supplier concentration if available.

This must remain descriptive. Business understanding does not equal investment approval.

### 5.2 Growth Profile

The scanner should help classify whether a company shows evidence of growth, stagnation, decline, or insufficient data.

Possible future dimensions:

- revenue trend;
- earnings trend;
- free cash flow trend;
- margin expansion or compression;
- backlog or order growth if relevant and available;
- segment-level growth where available;
- analyst-independent source-data trend context.

Growth classification must preserve uncertainty. Missing periods must not be filled with zero.

### 5.3 Profitability Profile

The scanner should help classify whether a company appears structurally profitable, temporarily pressured, improving, deteriorating, or insufficiently documented.

Possible future dimensions:

- gross margin;
- operating margin;
- net margin;
- return on equity;
- return on invested capital;
- free cash flow margin;
- profitability consistency;
- exceptional or one-off effects if source data supports them.

Profitability analysis is not a final quality score unless a future approved model defines such a classification. It remains an upstream observation.

### 5.4 Balance Sheet and Financial Risk

The scanner should help identify financial fragility or resilience.

Possible future dimensions:

- net debt or net cash position;
- debt maturity risk if available;
- interest coverage;
- liquidity position;
- leverage trend;
- working capital stress;
- dilution or share issuance risk;
- financial covenant or refinancing risk if explicitly available.

Balance sheet risk must be explicit and traceable. It must not be hidden behind a single opaque score.

### 5.5 Cash Generation

The scanner should help distinguish accounting earnings from cash generation.

Possible future dimensions:

- operating cash flow;
- capital expenditures;
- free cash flow;
- free cash flow conversion;
- cash generation stability;
- working capital impact;
- reinvestment intensity.

Cash generation classification must state whether data is available, partial, stale, missing, or review-required.

### 5.6 Valuation Context

The scanner may eventually provide valuation context, but valuation must be handled carefully.

Possible future dimensions:

- price-to-earnings context;
- enterprise value to EBITDA context;
- price-to-sales context;
- free cash flow yield context;
- comparison with own history;
- comparison with peers if peer data is approved;
- valuation versus growth context.

Valuation context must not become an automatic buy or sell rule. Cheap is not automatically attractive. Expensive is not automatically unacceptable. Valuation requires interpretation and Decision Engine governance before any action can be produced.

### 5.7 Capital Allocation and Shareholder Returns

The scanner may classify how a company uses capital.

Possible future dimensions:

- dividends;
- buybacks;
- reinvestment;
- acquisitions;
- debt reduction;
- dilution;
- capital intensity.

This domain is informational. It must not create portfolio allocation decisions.

### 5.8 Source-Data Readiness

The scanner must always distinguish source-data readiness from financial quality.

Source-data readiness may describe:

- available data;
- missing data;
- partial data;
- stale data;
- review-required data;
- source provenance;
- period coverage;
- missing field names;
- explicit missing-value policy.

Source-data readiness does not mean the company is good or bad. It only describes whether the available data is usable for the next analytical step.

## 6. Classification States

Financial analysis may eventually use descriptive states, subject to future implementation approval.

Allowed conceptual examples:

- `AVAILABLE`
- `MISSING`
- `PARTIAL`
- `STALE`
- `REVIEW_REQUIRED`
- `IMPROVING`
- `DETERIORATING`
- `STABLE`
- `VOLATILE`
- `INSUFFICIENT_DATA`

These are descriptive states. They must not imply final action.

Forbidden in financial analysis output:

- buy;
- sell;
- hold;
- tradeable;
- position size;
- execution instruction;
- allocation recommendation;
- final portfolio action.

## 7. Data Requirements

The financial analysis model depends on approved source data.

Possible future data groups:

- income statement data;
- balance sheet data;
- cash flow statement data;
- shares outstanding;
- market price context;
- sector and industry metadata;
- portfolio exposure metadata;
- fiscal period metadata;
- source provenance metadata.

No external provider or SEC-derived data may become active source-of-truth unless it is explicitly approved by the source-data strategy and data contracts.

Local-only or cached provider data must not silently become canonical input.

## 8. Missing Data Policy

Missing financial data must remain explicit.

Mandatory rules:

- missing numeric values must not be converted to zero;
- missing periods must not be silently interpolated;
- unavailable metrics must be marked as unavailable, partial, stale, missing, or review-required;
- source-data gaps must remain visible downstream;
- uncertainty must be communicated, not hidden.

This protects the scanner from false precision.

## 9. Portfolio Relevance

Financial analysis may eventually provide portfolio context, but it must not control allocation.

Possible future portfolio context:

- whether the company is already held;
- current exposure category;
- sector concentration context;
- single-position concentration context;
- ETF overlap context if available;
- watchlist relevance;
- review reason.

Portfolio relevance is not the same as a portfolio action. The Decision Engine remains the only final-action authority.

## 10. Reporting Expectations

Reporting must communicate financial analysis outputs without changing them.

Reporting may eventually show:

- financial classification summaries;
- source-data readiness notes;
- missing data warnings;
- rationale text;
- uncertainty and review reasons;
- Decision Engine output.

Reporting must not create, suppress, prioritize, filter, override, or reinterpret financial decisions.

## 11. Future Implementation Boundaries

When this document is later translated into code, implementation must proceed through controlled sprints.

Expected future implementation sequence:

1. extend source-data contracts;
2. add approved financial fixtures;
3. add source-data validation;
4. add descriptive financial classification records;
5. add contract tests;
6. only later connect classified observations to the Decision Engine;
7. only the Decision Engine may produce final actions.

Any implementation must remain deterministic, auditable, testable, and side-effect free unless explicitly approved.

## 12. Out of Scope

This document does not define:

- trading rules;
- buy/sell/hold rules;
- valuation thresholds;
- portfolio allocation policy;
- position sizing;
- execution timing;
- broker integration;
- live provider integration;
- SEC integration;
- generated reporting output;
- Telegram delivery.

Those require separate approved documentation and implementation steps.

## 13. Relationship to Legacy Documentation

Legacy financial, functional, technical, research, and execution documents may contain useful historical context. They are not active authority unless explicitly carried forward into `docs/active/`.

This document replaces scattered legacy financial-analysis intent with a single canonical v2 financial-analysis reference.

## 14. Final Statement

The v2 market scanner must support disciplined financial review, not uncontrolled recommendations.

Financial analysis must make relevant information clearer, preserve uncertainty, and prepare structured classifications for downstream governance.

It must never bypass the Decision Engine, hide missing data, or turn source-data availability into investment quality.
