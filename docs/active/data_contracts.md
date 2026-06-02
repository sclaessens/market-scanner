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
