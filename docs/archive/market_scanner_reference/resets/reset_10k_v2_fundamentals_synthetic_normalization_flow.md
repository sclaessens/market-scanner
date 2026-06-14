# RESET-10K — V2 Fundamentals Synthetic Normalization Flow

## Purpose

This document records the synthetic in-memory flow proven by RESET-10K.

The flow demonstrates the RESET-10D contract boundary without provider integration, SEC or EDGAR ingestion, real raw-data parsing, file loading, file writing, generated CSV output, financial scoring, Decision Engine behavior, reporting runtime, or Telegram delivery.

## Synthetic Raw Input

Example synthetic raw source record:

```text
source_provider = SYNTHETIC_PROVIDER
source_record_id = raw-asml-fy-2025
ticker = ASML
fiscal_period = FY
fiscal_year = 2025
captured_at = 2026-06-03T00:00:00Z
source_reference = synthetic/raw/asml/fy-2025
raw_payload_hash = sha256:synthetic-asml
metrics = revenue, gross_margin, free_cash_flow
```

The synthetic record is supplied in memory by tests. It is not read from `data/raw/`, provider files, SEC files, or legacy generated data.

## Normalized Output

The adapter emits one normalized metric record per supplied metric.

For the example above, the synthetic output is:

```text
ASML FY 2025 revenue
ASML FY 2025 gross_margin
ASML FY 2025 free_cash_flow
```

Each normalized record preserves:

- ticker;
- fiscal period;
- fiscal year;
- metric name;
- metric value exactly as supplied;
- metric unit;
- currency;
- normalized timestamp;
- source provider;
- source reference;
- source record identity.

## Readiness Output

The adapter emits one source-data readiness record per raw record.

Readiness describes only availability, completeness, freshness, validity, and provenance.

Examples:

- complete synthetic metrics produce `available`;
- missing metric values produce `partial` or `missing`;
- incomplete raw provenance produces `invalid`;
- empty metric sets produce `source_missing`;
- explicitly stale synthetic metric names produce `stale`.

Readiness is not investment quality.

## Deliberate Non-Implementation

RESET-10K does not implement:

- provider ingestion;
- SEC or EDGAR parsing;
- real raw-to-normalized runtime transformation;
- file loading;
- file writing;
- generated fundamentals outputs;
- financial scoring;
- investment-quality classification;
- target price calculation;
- threshold calculation;
- ranking;
- recommendation logic;
- Decision Engine behavior;
- reporting aggregation;
- Telegram delivery.

## Governance Confirmation

All data in this flow is synthetic and supplied in memory.

The adapter does not calculate investment quality, decisions, targets, thresholds, rankings, scores, recommendations, allocation, execution, urgency, conviction, or tradeability.
