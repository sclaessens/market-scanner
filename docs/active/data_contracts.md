# Data Contracts

Status: ACTIVE
Reset stage: RESET-4, updated by RESET-10B

## Purpose

This document defines the minimal v2 data contract baseline for approved inputs, fixtures, generated outputs, legacy data, local-only data, and reporting artifacts.

RESET-4 does not implement pipeline behavior. It only defines the first governed data and fixture surface for later implementation stages.

## Doctrine

- Classification upstream.
- Allocation downstream.
- Generated outputs are not source-of-truth inputs unless explicitly approved.
- Source-data readiness is not investment quality.
- Missing source-data values are not zero.
- Fixtures validate contracts, not old implementation paths.

## Data Classifications

| Classification | Meaning | Tracking policy | V2 source-of-truth status |
|---|---|---|---|
| Manual input | Human-maintained approved input, such as future v2 portfolio or universe inputs | Track only after explicit approval | May be source-of-truth when contract-approved |
| Approved fixture | Small deterministic synthetic or explicitly approved test data | Track intentionally | Test-only contract evidence |
| Generated output | Runtime product of v2 processing | Ignore by default | Not source-of-truth unless reclassified |
| Legacy generated output | Runtime product from old architecture under legacy paths | Preserve as reference only | Not v2 source-of-truth |
| Local-only data | Provider caches, raw dumps, local diagnostics, secrets, and machine-specific files | Ignore | Not source-of-truth |
| Raw/source data requiring reapproval | External data or transformed source evidence without a v2 contract | Ignore unless sanitized and approved | Not source-of-truth until approved |
| Report artifact | Generated communication artifact | Ignore by default | Communication only |

## Lifecycle Stage Alignment

RESET-10B aligns data contracts with the canonical v2 lifecycle:

```text
external source
  -> raw source data
  -> normalized program-ready input
  -> analytical classification output
  -> Decision Engine output
  -> reporting output
```

Mandatory separation:

```text
raw source data != normalized input != generated output != report
```

Contract metadata may use these lifecycle stages:

| Lifecycle stage | Meaning | Default repository policy |
|---|---|---|
| `RAW_SOURCE` | Original or near-original source evidence | Ignore by default except placeholders unless explicitly sanitized and approved |
| `NORMALIZED_INPUT` | Program-ready contract-compliant input | Ignore by default except placeholders until an approved normalized input policy exists |
| `FIXTURE_INPUT` | Small deterministic synthetic or explicitly approved fixture input | Track intentionally under `data/fixtures/v2/` |
| `GENERATED_OUTPUT` | Runtime output from analysis, pipeline, or Decision Engine execution | Ignore by default; not source-of-truth |
| `REPORTING_OUTPUT` | Communication artifact derived from approved records | Ignore by default; communication only |
| `LOCAL_ONLY` | Private local data, provider caches, diagnostics, secrets-adjacent artifacts, or machine-specific files | Ignore; not canonical |

Rules:

- raw source data is evidence, not program-ready input;
- normalized input is contract-compliant program input;
- fixtures are synthetic and tracked for tests;
- generated output is not source-of-truth unless explicitly reclassified and approved;
- reports are communication only and must not become input authority;
- local data is private or machine-specific and is not canonical.

Approved RESET-4 fixture contracts are lifecycle stage `FIXTURE_INPUT`.

## Approved RESET-4 Fixture Baseline

RESET-4 approves only these synthetic fixtures:

| Fixture | Path | Purpose | Required columns |
|---|---|---|---|
| Synthetic universe candidates | `data/fixtures/v2/universe_candidates.csv` | Validate minimal candidate identity and provenance | `candidate_id`, `symbol`, `company_name`, `source_kind`, `source_reference`, `discovered_at`, `row_identity`, `inclusion_reason` |
| Synthetic portfolio transactions | `data/fixtures/v2/portfolio_transactions.csv` | Validate manual portfolio input shape without final-action semantics | `transaction_id`, `portfolio_account`, `symbol`, `transaction_kind`, `quantity_delta`, `cash_amount`, `currency`, `occurred_at`, `source_reference` |
| Synthetic source-data readiness | `data/fixtures/v2/source_data_readiness.csv` | Validate source readiness states and missing-value handling | `source_record_id`, `symbol`, `source_name`, `metric_name`, `metric_value`, `metric_unit`, `as_of_date`, `readiness_state`, `missing_value_policy`, `review_required_reason` |

## Fixture Policy

Fixtures must be:

- small;
- deterministic;
- tracked intentionally;
- synthetic or explicitly approved;
- independent from generated runtime outputs;
- designed around v2 contracts, not old implementation paths.

RESET-4 fixtures must not depend on `data/processed/`, `data/logs/`, `reports/`, SEC caches, provider caches, or generated review output.

## Manual Input Policy

Manual input paths are not approved by RESET-4 for runtime use. Future manual inputs must define:

- owner;
- path;
- schema;
- row identity;
- allowed semantics;
- forbidden semantics;
- freshness expectations;
- validation rules.

## Source-Data Policy

SEC and provider data remain local-only unless explicitly sanitized and approved.

Approved source-data contracts must distinguish readiness from investment quality. Missing source-data values must remain missing or explicit review-required states. Missing values must not be interpreted as zero unless a future approved source-specific contract states that zero is the observed value.

## Generated and Legacy Data Policy

Generated v2 outputs, old processed CSVs, logs, and reports are not approved v2 source-of-truth inputs.

Legacy generated files may be read for historical understanding, but v2 implementation must not consume them as active input unless a future contract explicitly reclassifies the artifact.

## Reporting Policy

Reports are generated communication artifacts. They may summarize traceability and decisions emitted by the Decision Engine, but they must not become allocation inputs or override decision semantics.

## RESET-5 Readiness

RESET-5 may use the approved fixtures to build minimal contract validation and pipeline-core scaffolding. It must not expand fixture semantics into allocation, execution, or final-action behavior.

## V2 Validation Contract Alignment

RESET-9C2A translates safe legacy validation-test knowledge into v2 contract metadata only.

The v2 validation contract defines:

- required candidate identity fields: `ticker`, `date`;
- required candidate input fields needed for structure classification;
- explicit issue metadata for missing fields, missing values, invalid numeric values, non-positive required metrics, and forbidden upstream decision fields;
- validation classification fields that preserve row identity and communicate structure metadata;
- no file input/output, no generated-output dependency, no legacy `scripts` import, and no provider access.

Validation contract helpers must preserve row identity conceptually and report issues explicitly. They must not filter rows, treat missing numeric values as zero, create final actions, create allocation or execution semantics, or read from legacy generated CSVs.

RESET-9C2A does not implement a full validation runtime. It establishes the first v2 validation contract surface for later pipeline work.

## V2 Portfolio Contract Alignment

RESET-9C2B translates safe legacy portfolio-test knowledge into v2 contract metadata only.

The v2 portfolio contract defines:

- manual portfolio transaction input fields aligned with the approved synthetic `portfolio_transactions.csv` fixture;
- normalized portfolio position identity fields: `portfolio_account`, `symbol`;
- generated portfolio review and generated portfolio classification dataset types as generated-output roles, not source records;
- explicit issue metadata for missing fields, missing values, invalid numeric values, and forbidden portfolio authority fields;
- no file input/output, no generated-output dependency, no legacy `scripts` import, no broker access, and no provider access.

Portfolio contracts must distinguish manual source input from generated portfolio outputs. They may describe portfolio presence, source provenance, and metadata readiness in later layers, but they must not create final actions, allocation instructions, execution semantics, tradeability, conviction, urgency, rankings, or hidden filtering.

RESET-9C2B does not implement transaction ingestion, broker integration, portfolio review generation, or a full portfolio runtime. It establishes the first v2 portfolio contract surface for later pipeline work.

## V2 Portfolio Source-of-Truth Contract Alignment

RESET-10C defines the active portfolio source-of-truth boundary in `docs/active/portfolio_source_of_truth.md`.

The portfolio source-of-truth contract separates:

- manual source transactions;
- manual source positions;
- normalized portfolio positions;
- generated portfolio review and intelligence outputs;
- reporting display input;
- Telegram/report communication.

Manual source transactions and manual source positions are the only source-of-truth roles approved by RESET-10C.

Generated portfolio review, generated portfolio intelligence, reporting display input, and Telegram output are not source-of-truth.

Portfolio display fields such as profit/loss, current price, target price, and action/status must be supplied by approved upstream records. Reporting and Telegram may format these fields, but must not calculate them or write them back as portfolio source data.

Missing portfolio values must remain explicit and must not be converted to zero.

RESET-10C does not implement broker integration, transaction ingestion, real profit/loss calculation, target price calculation, generated portfolio outputs, report generation, Telegram delivery, or portfolio allocation behavior.

## V2 Reporting Input Aggregation Contract Alignment

RESET-10H defines the active reporting input aggregation boundary in `docs/active/reporting_input_aggregation.md`.

The reporting input aggregation contract separates:

- portfolio display input;
- candidate display input;
- Decision Engine status input;
- source-data status input;
- data warning input;
- Telegram renderer input;
- source-of-truth records.

Reporting input aggregation may assemble approved upstream display records for communication, but it must not create source truth, decisions, target prices, thresholds, profit/loss values, rankings, scores, allocation instructions, execution instructions, urgency, conviction, tradeability, or recommendations.

Every reporting input record must preserve traceability through source role, source reference, and aggregation contract version.

Telegram renderer input is downstream of reporting input aggregation and must not become source-of-truth.

RESET-10H does not implement real aggregation runtime, file loading, file writing, report generation, Telegram delivery, provider integration, broker integration, portfolio calculations, target price calculation, threshold calculation, or Decision Engine behavior.

## V2 Fundamentals and Source-Data Contract Alignment

RESET-9C2C translates safe legacy fundamentals and source-data test knowledge into v2 contract metadata only.

The v2 fundamentals/source-data contract defines:

- source-readiness fields aligned with the approved synthetic `source_data_readiness.csv` fixture;
- normalized fundamentals-history identity fields: `ticker`, `fiscal_year`, `fiscal_period`;
- raw source capture, normalized source-readiness, normalized fundamentals-history, generated classification, generated analysis, and local-only review dataset roles;
- explicit readiness states for available, missing, source-missing, row-missing, partial, stale, invalid, unavailable, and review-required source data;
- explicit issue metadata for missing fields, missing values, invalid numeric values, invalid readiness states, and forbidden fundamentals authority fields;
- no file input/output, no generated-output dependency, no legacy `scripts` import, no SEC access, no provider access, and no network access.

Fundamentals and source-data contracts must distinguish source-data readiness from investment quality. They may describe source provenance, period metadata, freshness, missing values, partial data, stale data, unavailable data, and review-required conditions, but they must not create quality scores, final actions, allocation instructions, execution semantics, tradeability, conviction, urgency, rankings, or hidden filtering.

RESET-9C2C does not implement SEC ingestion, provider integration, raw-to-normalized transformation, financial scoring, investment-quality analysis, generated fundamentals outputs, or a full fundamentals runtime. It establishes the next v2 contract surface for later source-data work.
