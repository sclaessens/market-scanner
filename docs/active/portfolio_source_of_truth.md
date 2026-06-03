# Portfolio Source-of-Truth Contract

Status: ACTIVE
Reset stage: RESET-10C

## Purpose

This document defines the v2 portfolio source-of-truth boundary.

It exists because v2 Reporting and Telegram need portfolio display fields, but Reporting and Telegram must not become portfolio data owners.

## Core Decision

Manual portfolio source records are the only approved portfolio source-of-truth role in this contract.

Generated portfolio review, generated portfolio intelligence, reporting display input, and Telegram output are not source-of-truth.

## Portfolio Dataset Roles

| Role | Meaning | Source-of-truth status |
|---|---|---|
| `MANUAL_SOURCE_TRANSACTIONS` | Manual transaction source records | Source record |
| `MANUAL_SOURCE_POSITIONS` | Manual holding or position source records | Source record |
| `NORMALIZED_POSITIONS` | Program-ready position records derived from approved source data | Normalized input, not manual source |
| `GENERATED_PORTFOLIO_REVIEW` | Generated portfolio review output | Generated output only |
| `GENERATED_PORTFOLIO_INTELLIGENCE` | Generated portfolio classification or intelligence output | Generated output only |
| `REPORTING_DISPLAY_INPUT` | Display-ready portfolio fields supplied to Reporting or Telegram | Communication input only |

## Manual Source Principle

Manual source records own portfolio holdings and transaction evidence until a later approved contract introduces another source owner.

Manual source records must be protected from generated-output overwrite.

Generated review output must never overwrite manual source records.

Reporting display input must never be written back as portfolio source data.

Telegram text must never become portfolio source data.

## Required Manual Position Fields

Manual position source records require:

- `portfolio_id`
- `ticker`
- `quantity`
- `currency`
- `source_type`
- `as_of_date`

These fields describe source ownership and holding identity. They do not create final actions, allocation instructions, or Reporting text.

## Required Manual Transaction Fields

Manual transaction source records require:

- `portfolio_id`
- `transaction_id`
- `ticker`
- `transaction_type`
- `quantity`
- `price`
- `currency`
- `transaction_date`
- `source_type`

These fields describe transaction evidence. RESET-10C does not implement transaction ingestion, transaction-to-position calculation, broker parsing, or portfolio review generation.

## Reporting Display Input

Portfolio display input for Reporting and Telegram requires:

- `ticker`
- `profit_loss_percent_display`
- `current_price_display`
- `target_price_display`
- `action_status`
- `currency`
- `source_reference`

These are display-ready fields. Reporting and Telegram may format them, but must not calculate them.

Ownership:

- profit/loss display must come from an approved upstream record;
- current price display must come from an approved upstream record;
- target price display must come from an approved upstream record;
- action/status must come from the Decision Engine or an approved downstream display contract that preserves Decision Engine authority;
- source references must preserve traceability to source or normalized records.

## Missing Value Policy

Missing portfolio values must remain explicit.

Examples:

- `P/L unavailable`
- `price unavailable`
- `target unavailable`

Missing numeric or display values must not be converted to zero.

## Forbidden Authority

Portfolio source records must not contain:

- final action authority;
- allocation instructions;
- execution instructions;
- urgency;
- conviction;
- tradeability;
- ranking;
- score;
- recommendation text;
- reporting-only display authority;
- Telegram message text;
- generated review overwrite markers.

Portfolio source contracts may describe holdings and transactions. They must not create buy, sell, hold, allocation, execution, urgency, conviction, tradeability, ranking, or recommendation behavior.

## Future Implementation Implications

Future portfolio implementation must preserve this separation:

```text
manual portfolio source
  -> normalized portfolio input
  -> generated portfolio classification/review
  -> Decision Engine
  -> reporting display input
  -> Telegram/report communication
```

RESET-10C does not authorize:

- broker integration;
- transaction ingestion;
- real profit/loss calculation;
- target price calculation;
- report generation;
- Telegram delivery;
- portfolio allocation behavior;
- generated portfolio output creation.
