# Reporting Input Aggregation Contract

Status: ACTIVE
Reset stage: RESET-10H

## Purpose

This document defines the v2 Reporting Input Aggregation Contract.

It exists because Reporting and Telegram need display-ready inputs from several approved upstream surfaces, but Reporting must not become a source-of-truth layer, decision layer, portfolio engine, price calculator, threshold calculator, or report generator.

## Core Decision

Reporting input aggregation may assemble approved upstream display records for communication.

Reporting input aggregation must not create source truth, decisions, target prices, thresholds, profit/loss values, rankings, scores, allocation instructions, execution instructions, urgency, conviction, or tradeability.

Telegram renderer input is downstream of reporting input aggregation.

## Reporting Input Roles

| Role | Meaning |
|---|---|
| `PORTFOLIO_DISPLAY_INPUT` | Portfolio display fields supplied by approved upstream portfolio or decision records |
| `CANDIDATE_DISPLAY_INPUT` | Candidate display fields supplied by approved upstream decision or watch records |
| `DECISION_STATUS_INPUT` | Decision/action status supplied by the Decision Engine boundary |
| `SOURCE_DATA_STATUS_INPUT` | Source-data readiness or data-status text supplied by approved source-data records |
| `DATA_WARNING_INPUT` | Compact data warning text supplied by approved upstream records |
| `TELEGRAM_RENDERER_INPUT` | Final in-memory input shape consumed by the Telegram renderer |

## Boundary Roles

Reporting input aggregation distinguishes:

- upstream source input;
- derived display input;
- generated report input;
- renderer input.

This distinction protects source-of-truth records from downstream communication records.

## Required Traceability

Every reporting input record must preserve:

- `source_role`
- `source_reference`
- `aggregation_contract_version`

The source reference must trace back to an approved upstream record. It must not point to Telegram message text or generated report text as source authority.

## Portfolio Display Input

Portfolio display input requires:

- `ticker`
- `profit_loss_percent_display`
- `current_price_display`
- `target_price_display`
- `action_status`
- `currency`
- `source_reference`

These values are display-ready. Reporting may aggregate and format them, but must not calculate them.

Portfolio source-of-truth remains governed by `docs/active/portfolio_source_of_truth.md`.

## Candidate Display Input

Candidate display input requires:

- `ticker`
- `candidate_group`
- `threshold_price_display`
- `threshold_direction`
- `action_status`
- `currency`
- `source_reference`

Threshold values and threshold direction must be supplied by approved upstream records. Reporting must not calculate, infer, rank, or prioritize candidates.

## Decision Status Input

Decision status input requires:

- `row_id`
- `action_status`
- `decision_rationale`
- `source_reference`

Decision status must preserve Decision Engine authority. Reporting input aggregation may carry the status forward, but must not create, override, or reinterpret it.

## Source-Data Status and Warning Input

Source-data status input requires:

- `data_status`
- `review_reason`
- `source_reference`

Data warning input requires:

- `warning_type`
- `warning_text`
- `source_reference`

Source-data readiness is not investment quality. Missing, partial, stale, unavailable, or review-required source-data states must remain visible and must not be converted into scores or recommendations.

## Missing Value Policy

Missing display values must remain explicit.

Examples:

- `P/L unavailable`
- `price unavailable`
- `target unavailable`
- `threshold unavailable`

Missing numeric or display values must not be converted to zero.

## Forbidden Authority

Reporting input aggregation must not contain or create:

- source-of-truth overwrite authority;
- portfolio source overwrite authority;
- Telegram message text as source truth;
- report artifact paths as source truth;
- allocation instructions;
- execution instructions;
- urgency;
- conviction;
- tradeability;
- ranking;
- score;
- recommendation text;
- target price calculation authority;
- buy threshold calculation authority;
- breakout threshold calculation authority;
- profit/loss calculation authority;
- current price fetching authority;
- decision override authority.

## Relation to Telegram Renderer

The Telegram renderer consumes explicit in-memory records.

Reporting input aggregation may prepare renderer input, but the renderer remains downstream and communication-only.

Telegram output must never be written back as portfolio source data, normalized input, Decision Engine input, or source-data readiness input.

## Future Implementation Implications

Future implementation must preserve this separation:

```text
approved upstream records
  -> reporting input aggregation
  -> renderer input
  -> Telegram/report communication
```

RESET-10H does not authorize:

- real aggregation runtime;
- file loading;
- file writing;
- report generation;
- Telegram delivery;
- portfolio calculations;
- price fetching;
- target price calculation;
- threshold calculation;
- Decision Engine behavior changes;
- broker/provider integration.
