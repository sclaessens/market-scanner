# User-Friendly Reporting and Telegram Output Specification

Status: ACTIVE
Reset stage: RESET-10F
Owner role: PM / UX / Analyst

## 1. Purpose

This document defines the target v2 user experience for the market scanner Telegram summary and reporting communication.

It is based on the RESET-10E0 review and on the agreed compact Telegram message structure.

The goal is to make Telegram useful as a daily operator summary without turning Telegram or Reporting into a decision engine.

## 2. Core UX Decision

Legacy Telegram output must not be translated directly into v2.

The target Telegram message must be compact, portfolio-first, and action-grouped.

The agreed baseline structure is:

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

This is the baseline UX shape. It may be refined later, but future v2 reporting contracts and tests should align with this structure unless explicitly superseded.

## 3. Message Objective

Telegram must help the operator answer quickly:

- What is happening with my current portfolio?
- What is the current profit or loss per position?
- What is the target price per position where available?
- What action/status is currently attached to each position?
- Are there any immediate buy candidates?
- Which candidates should be watched on a price drop?
- Which candidates should be watched on a breakout?
- Is the data complete enough to trust the signal?

Telegram is not meant to show every pipeline field.

## 4. Required Message Sections

The v2 Telegram message should contain these sections in this order.

### 4.1 Header

Purpose:

- identify the message as the market scanner summary;
- optionally include date or run timestamp if available.

Preferred label:

```text
Market Scanner
```

A date may be added later:

```text
Market Scanner — 2026-06-02
```

### 4.2 Portfolio

The Portfolio section must come first.

Purpose:

- show current holdings before new opportunities;
- make current exposure visible;
- show current profit/loss, current price, target price, and current action/status.

Preferred row format:

```text
TICKER: P/L | price VALUE | target VALUE | ACTION
```

Examples:

```text
ASML: +8.4% | price €XXX | target €XXX | REVIEW
Thales: +14.2% | price €XXX | target €XXX | HOLD
Costco: -3.1% | price $XXX | target $XXX | REVIEW
```

ETF positions may be aggregated if the message would otherwise become too long:

```text
ETFs: +2-8% | keep accumulating
```

If ETF-level detail is later required, it should be introduced explicitly and tested separately.

### 4.3 Buy now

Purpose:

- show candidates that are immediately buyable according to approved Decision Engine output.

Preferred empty-state wording:

```text
No candidates today.
```

Rules:

- Reporting must not place a ticker in this section unless the upstream Decision Engine or approved v2 decision contract provides that state.
- Reporting must not infer immediate buyability from score, ranking, price movement, or message formatting.

### 4.4 Buy on pullback

Purpose:

- show candidates that may become attractive below a defined price threshold.

Preferred row format:

```text
TICKER below PRICE
```

Examples:

```text
AMD below $XXX
ASML below €XXX
```

Rules:

- The threshold must come from an approved upstream field.
- Reporting must not calculate or invent the threshold unless a future reporting contract explicitly permits display-only formatting of an existing field.

### 4.5 Buy on breakout

Purpose:

- show candidates that may become attractive above a defined breakout threshold.

Preferred row format:

```text
TICKER above PRICE
```

Examples:

```text
NVIDIA above $XXX
Meta above $XXX
```

Rules:

- The breakout threshold must come from an approved upstream field.
- Reporting must not infer breakout status from raw market data unless that is already produced by an approved upstream layer.

### 4.6 Data status

Purpose:

- explain why many results may remain `REVIEW`;
- expose missing, partial, stale, or unavailable source-data states;
- prevent false confidence.

Preferred compact wording:

```text
Fundamental data incomplete -> many REVIEW.
```

Alternative examples:

```text
Source data missing -> all final actions remain REVIEW.
```

```text
Partial source data -> review before acting.
```

Rules:

- Data status must describe data readiness only.
- Data status must not imply investment quality.
- Missing data must remain visible.

## 5. Compactness Rules

Telegram must remain compact.

Default rules:

- Portfolio section should show current holdings first.
- Candidate sections should show only the most relevant rows allowed by the reporting contract.
- Empty sections should use one short line.
- Technical artifact paths must not appear near the top.
- Full traceability details should remain in logs or full reporting artifacts.
- Avoid long rationale text in Telegram.
- Use short labels: `price`, `target`, `REVIEW`, `HOLD`, `below`, `above`.

## 6. Data Fields Required for the Target UX

Future v2 reporting contracts will need upstream fields for this UX.

### 6.1 Portfolio row fields

Required display concepts:

- ticker;
- position type or holding indicator;
- current profit/loss percentage;
- current price;
- target price;
- action/status;
- currency;
- optional instrument group, such as ETF or equity.

### 6.2 Candidate row fields

Required display concepts:

- ticker;
- candidate group:
  - buy now;
  - buy on pullback;
  - buy on breakout;
- threshold price, where relevant;
- threshold direction:
  - below;
  - above;
- action/status;
- currency.

### 6.3 Data status fields

Required display concepts:

- source-data readiness state;
- missing data count or status if available;
- partial data count or status if available;
- stale data count or status if available;
- short review reason.

## 7. Authority Boundaries

Telegram and Reporting may display only upstream-approved fields.

Telegram and Reporting must not:

- create buy/sell/hold decisions;
- calculate target prices unless explicitly approved elsewhere;
- calculate buy thresholds unless explicitly approved elsewhere;
- calculate breakout thresholds unless explicitly approved elsewhere;
- rank opportunities;
- create urgency;
- create conviction language;
- create tradeability language;
- hide source rows without governed representation rules;
- convert missing values to zero;
- convert source-data readiness into investment quality.

The Decision Engine remains the only final-action authority.

## 8. Language and Formatting

Current repository governance generally prefers English-only tracked documentation and generated contract text.

The v2 implementation may keep internal field names in English while still allowing future operator-facing Telegram text to be reviewed separately.

For now, the target baseline is written in English for repository consistency.

Future Dutch operator-facing wording may be approved later as a separate UX/localization decision.

## 9. Handling Missing Values

Missing values must remain explicit.

Examples:

```text
ASML: +8.4% | price €XXX | target unavailable | REVIEW
```

```text
Costco: P/L unavailable | price $XXX | target $XXX | REVIEW
```

Do not display missing numeric fields as zero.

## 10. Handling Empty Sections

If a section has no rows, show a compact empty-state line.

Examples:

```text
Buy now
No candidates today.
```

```text
Buy on breakout
No breakout candidates today.
```

Do not omit the section silently if omission could confuse the operator.

## 11. Traceability Requirements

Telegram may be compact, but traceability must remain available.

Minimum requirement:

- Telegram may include a short reference to the full report if available.
- Full reporting artifacts must preserve source row identity, representation rules, and row counts.
- Telegram must not become the only auditable output.

A compact footer may be added later:

```text
Full details in reporting dashboard. Reporting only; decisions from Decision Engine.
```

This footer is optional if it makes the message too long, but the same principle must be preserved in the full reporting contract.

## 12. Acceptance Criteria for Future Implementation

A future implementation is acceptable only if:

- Telegram begins with portfolio status;
- each portfolio row can show profit/loss, current price, target price, and action/status;
- buy candidates are grouped into buy now, buy on pullback, and buy on breakout;
- empty states are compact and explicit;
- data-status warning remains visible;
- no reporting-side decision authority is introduced;
- missing values remain explicit;
- no real Telegram call is required for tests;
- tests use synthetic fixtures or approved deterministic records;
- full traceability remains available outside Telegram.

## 13. Recommended Next Action

Recommended next action:

```text
RESET-9C2D — Translate Reporting and Telegram UX Requirements to V2 Contracts
```

This should be a Codex/local contract-translation sprint.

It should translate this UX specification into v2 reporting contract metadata and tests without implementing Telegram delivery.
