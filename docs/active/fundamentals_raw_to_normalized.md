# Fundamentals Raw-to-Normalized Contract

## Purpose

This document defines the v2 boundary for fundamentals data moving from raw source capture into normalized program-ready records.

It is a contract and governance document. It does not authorize provider integration, SEC or EDGAR ingestion, real raw-data loading, runtime normalization, financial scoring, target calculation, reporting generation, Telegram delivery, or Decision Engine behavior.

## Core Decision

Raw fundamentals source data, normalized fundamentals input, source-data readiness, generated fundamentals outputs, and reporting display input are separate lifecycle roles.

```text
raw source capture
-> normalized fundamentals input
-> source-data readiness
-> downstream classification and communication inputs
```

Raw source capture is immutable evidence. Normalized fundamentals input is program-ready data. Source-data readiness describes availability, completeness, freshness, and review needs. None of these roles creates investment quality, final actions, allocation authority, reporting authority, or Telegram output.

## Dataset Roles

| Role | Purpose | Source-of-truth status |
|---|---|---|
| `RAW_SOURCE_CAPTURE` | Captured source evidence with provenance. | Source evidence only. |
| `NORMALIZED_FUNDAMENTALS_INPUT` | Program-ready fundamentals metric records. | Input contract, not a generated conclusion. |
| `SOURCE_DATA_READINESS` | Availability, completeness, freshness, and review metadata. | Data governance state only. |
| `GENERATED_FUNDAMENTAL_QUALITY` | Future generated classification output if approved. | Not raw or normalized source-of-truth. |
| `GENERATED_FUNDAMENTAL_ANALYSIS` | Future generated analysis output if approved. | Not raw or normalized source-of-truth. |
| `REPORTING_DISPLAY_INPUT` | Downstream display-ready communication input. | Not source-of-truth. |

## Raw Source Capture

Raw source capture records must preserve provenance and identity. Required fields are:

- `source_provider`
- `source_record_id`
- `ticker`
- `fiscal_period`
- `fiscal_year`
- `captured_at`
- `source_reference`
- `raw_payload_hash`

Raw capture records are evidence. They must not be rewritten by generated outputs, reporting outputs, or normalization helpers.

## Normalized Fundamentals Input

Normalized fundamentals input records must be program-ready but must not become investment conclusions.

Required fields are:

- `ticker`
- `fiscal_period`
- `fiscal_year`
- `metric_name`
- `metric_value`
- `metric_unit`
- `currency`
- `normalized_at`
- `source_provider`
- `source_reference`
- `source_record_identity`

Every normalized fundamentals record must remain traceable to raw source provenance through provider, source reference, and source record identity.

## Source-Data Readiness

Source-data readiness records describe whether source data is usable for the next analytical step.

Required fields are:

- `ticker`
- `fiscal_period`
- `readiness_state`
- `source_data_status`
- `missing_fundamentals_count`
- `partial_data_count`
- `stale_data_count`
- `source_reference`

Readiness states are descriptive only. Missing, partial, stale, unavailable, invalid, source-missing, and review-required states must remain explicit.

Source-data readiness is not investment quality.

## Missing-Value Policy

Missing source values must remain explicit.

Mandatory rules:

- missing metric values must not be converted to zero;
- missing count fields must not be converted to zero;
- partial source data must remain partial;
- stale source data must remain stale;
- unavailable source data must remain unavailable;
- invalid source data must remain invalid;
- source-missing conditions must remain source-missing.

This protects downstream layers from false precision.

## Forbidden Authority

Fundamentals raw-to-normalized contracts must not include:

- investment quality scores;
- final-action fields;
- allocation fields;
- execution instructions;
- urgency fields;
- conviction fields;
- tradeability fields;
- ranking or score fields;
- recommendation fields;
- target prices;
- threshold prices;
- Telegram or report message text.

The Decision Engine remains the only final-action and allocation authority. Reporting and Telegram remain communication-only.

## Overwrite Protection

Generated fundamentals outputs must not overwrite raw source capture or normalized fundamentals input.

Reporting display input must not become source-of-truth. Telegram output must not become source-of-truth.

## Future Implementation Implications

Future raw-to-normalized implementation must proceed through a separate approved sprint.

That future work must define explicit fixture strategy, provenance checks, missing-value behavior, deterministic transformation rules, and local-data boundaries before provider or SEC integration can be considered.
