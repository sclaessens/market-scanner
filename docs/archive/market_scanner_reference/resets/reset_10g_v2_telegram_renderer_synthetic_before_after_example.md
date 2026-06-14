# RESET-10G — V2 Telegram Renderer Synthetic Before/After Example

## Purpose

This document records a synthetic before/after example for the v2 Telegram renderer design.

It demonstrates that the approved portfolio-first Telegram UX can be rendered from explicit in-memory display records without reading real data, generating report artifacts, or sending Telegram messages.

## Legacy-Style Example

RESET-10E0 found that the legacy Telegram output starts with technical reporting metadata and artifact references before showing operator-useful information.

Synthetic legacy-style shape:

```text
Daily Reporting Summary
Reporting contract: REPORTING_CONTRACT_V1
Source artifact: data/processed/final_decisions.csv
Dashboard artifact: data/processed/reporting_dashboard_data.csv
Source row count: ...
Represented row count: ...
omitted_row_count: 0
Input status: ...
Stability status: ...

Decision output: HOLD
Group count: 1
- AAA: action=HOLD; allocation=SOURCE_HOLD; execution=SOURCE_NONE

Traceability
Grouping rule: GROUP_BY_SOURCE_FINAL_ACTION_THEN_SOURCE_ORDER
Truncation rule: TELEGRAM_GROUP_SUMMARY_WITH_SOURCE_ORDER_EXAMPLES
Ordering rule: SOURCE_ORDER_WITH_FIXED_SECTION_ORDER
All source rows are represented in the dashboard artifact.
```

This shape preserves governance evidence, but it is not the approved v2 Telegram baseline because technical artifact details dominate the top of the message.

## V2 Synthetic Example

Synthetic v2 output:

```text
Market Scanner

Portfolio
ASML: +8.4% | price €XXX | target €XXX | REVIEW
Thales: +14.2% | price €XXX | target €XXX | HOLD
Costco: -3.1% | price $XXX | target $XXX | REVIEW
ETFs: +2-8% | keep accumulating

Buy now
No candidates today.

Buy on pullback
AMD below $XXX
ASML below €XXX

Buy on breakout
NVIDIA above $XXX
Meta above $XXX

Data status
Fundamental data incomplete -> many REVIEW.
```

## Improvements

- Portfolio information appears before new candidate sections.
- Empty candidate sections remain explicit and compact.
- Pullback and breakout sections display upstream-supplied thresholds without calculating them.
- Data status remains visible without turning source-data readiness into investment quality.
- Technical artifact paths, grouping rules, and truncation rules do not dominate the Telegram message.
- The message remains communication-only and does not create decisions.

## Synthetic Data Confirmation

The v2 example is synthetic.

It is not generated from real portfolio data, real market data, provider data, SEC data, generated reports, or legacy CSV outputs.

## Renderer Boundary

The renderer must not calculate:

- decisions;
- target prices;
- buy thresholds;
- breakout thresholds;
- data status;
- rankings;
- urgency;
- conviction;
- tradeability;
- allocation;
- execution instructions.

All rendered values must be supplied explicitly by approved upstream records.
