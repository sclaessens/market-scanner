# Fundamentals Source Data Operating Workflow

Status: ACTIVE OPERATING WORKFLOW
Backlog driver: BL-0015
Date: 2026-05-30

## 1. Purpose

This document defines how real fundamentals data should be sourced, prepared, validated, stored, and used by the fundamentals platform.

The goal is to support repeatable, auditable source-data work without provider/API misuse, scraping problems, generated-output commits, or hidden decision logic.

This document governs the first real-data operating workflow for:

```text
data/raw/fundamentals_history.csv
```

It does not authorize code changes, test changes, CSV commits, generated-output commits, provider/API integration, scraping, credentials, pipeline runs, Decision Engine changes, Reporting changes, Telegram changes, portfolio changes, ticker-category runtime logic, or runtime behavior changes.

## 2. Current Status

The fundamentals platform is technically implemented and synthetically validated.

Current platform chain:

```text
raw fundamentals history
-> calculated fundamental metrics
-> fundamental quality compatibility
-> fundamental analysis classification
```

Current runtime surface:

```text
scripts/fundamentals/
  build_history_intake.py
  build_metrics.py
  build_quality.py
  build_analysis.py
```

Compatibility wrappers remain available under:

```text
scripts/core/
```

The protected downstream artifact remains:

```text
data/processed/fundamental_quality.csv
```

`fundamental_analysis.csv` remains descriptive and is not a downstream-required dependency.

BL-0015 should remain active until the source-data workflow is operationally proven with real source-supported data.

## 3. Operating Principles

The fundamentals source-data workflow must follow these principles:

- raw source facts before calculated metrics;
- source evidence before interpretation;
- manual or controlled semi-manual workflow before automation;
- no provider/API automation in this sprint;
- no scraping in this sprint;
- no inferred values;
- no generated-output commits unless repository policy explicitly allows them;
- no Decision Engine authority outside `scripts/core/decision_engine.py`;
- no hidden filtering, ranking, scoring, allocation, eligibility, tradeability, urgency, or conviction semantics;
- repeatable preparation steps;
- traceable source references;
- local ignored raw source-data handling unless repository policy changes;
- analyst review before broad data expansion.

## 4. Approved Initial Workflow

The approved initial workflow is manual or controlled semi-manual.

Approved initial steps:

1. Select a small controlled ticker sample.
2. Identify approved public or licensed source material manually.
3. Extract raw reported statement facts manually or from a controlled human-reviewed export.
4. Record source evidence for every row.
5. Prepare a local `fundamentals_history.csv` draft.
6. Validate the draft with the raw-history validator.
7. Generate metrics, quality, and analysis outputs locally using explicit paths.
8. Review structural validation results.
9. Review analyst-facing quality and analysis outputs.
10. Decide whether the sample is acceptable for broader controlled expansion.

The initial workflow may use local spreadsheet preparation or a controlled local CSV draft. It must not use unapproved automated provider calls, scraping scripts, browser scraping, credential creation, or hidden ingestion shortcuts.

## 5. Manual vs Semi-Automated vs Automated Phases

| Phase | Status | Description | Allowed now? |
|---|---|---|---|
| Manual | Approved initial phase | Human selects sources, extracts facts, records evidence, prepares local CSV. | Yes. |
| Controlled semi-manual | Approved with review | Human-reviewed export or spreadsheet workflow feeds local CSV, with traceable source evidence. | Yes, if no unapproved provider/API calls or scraping occur. |
| Automated provider/API ingestion | Not approved | Runtime or helper scripts call providers to collect data automatically. | No. |
| Scraping | Not approved | Runtime or helper scripts scrape websites or pages for fundamentals data. | No. |
| Fully automated refresh | Not approved | Scheduled or unattended source-data refresh. | No. |

Automation may be revisited later under BL-0017 or a separate governed provider/source-data sprint.

## 6. Required Source Evidence

Every real fundamentals history row must have traceable source evidence.

Minimum evidence:

- source name;
- source reference;
- fiscal year;
- fiscal period;
- period end date;
- report date;
- currency;
- extraction date;
- source freshness date;
- raw reported values used;
- notes for caveats, missing fields, restatements, or source-specific definitions.

Acceptable source reference examples:

- annual report name and year;
- filing accession or filing page reference;
- provider export filename plus row/source identifier, if provider use is separately approved;
- public company investor-relations report reference;
- internal evidence pointer, if the source is licensed and cannot be publicly linked.

Unacceptable source reference examples:

- blank reference;
- generic text such as `internet`, `manual`, or `website` without a specific document or evidence pointer;
- undocumented spreadsheet values;
- inferred or estimated values without source support.

## 7. `fundamentals_history.csv` Preparation Rules

The local raw-history draft should follow the target schema:

```text
ticker
fiscal_year
fiscal_period
period_end_date
report_date
currency
revenue
gross_profit
operating_income
net_income
diluted_eps
total_debt
total_equity
free_cash_flow
source_name
source_reference
source_freshness_date
extraction_date
notes
```

Preparation rules:

- one row per `ticker` + `fiscal_year` + `fiscal_period`;
- use raw reported values only;
- leave missing numeric values blank when source data is incomplete;
- do not infer missing values;
- do not calculate ratios in raw history;
- do not add quality states;
- do not add analysis states;
- do not add allocation, decision, ranking, scoring, eligibility, urgency, conviction, tradeability, buy/sell, entry, stop, or target fields;
- preserve source-specific caveats in `notes`;
- keep date semantics separate across `period_end_date`, `report_date`, `source_freshness_date`, and `extraction_date`;
- validate before generating downstream outputs.

## 8. Required Fields and `source_reference` Rules

Required non-empty identity and source fields:

| Field | Rule |
|---|---|
| `ticker` | Required. Must identify the security consistently with the project ticker universe. |
| `fiscal_year` | Required. Must be parseable as an integer year. |
| `fiscal_period` | Required. Supported initial values include `FY`, `Q1`, `Q2`, `Q3`, `Q4`, and `TTM`. |
| `period_end_date` | Required when available from the source. Must represent the fiscal period end. |
| `report_date` | Required when available from the source. Must represent company report, filing, or publication date. |
| `currency` | Required. Must reflect the reporting currency for raw values. |
| `source_name` | Required. Must identify the source or source family. |
| `source_reference` | Required. Must point to specific evidence. |
| `source_freshness_date` | Required. Date the source evidence was checked. |
| `extraction_date` | Required. Date the data was extracted into the local workflow. |

`source_reference` must be specific enough that a later reviewer can find or verify the source evidence. If a source is licensed and cannot be linked publicly, use a stable internal evidence pointer and describe the constraint in `notes`.

## 9. What May Be Committed

Allowed commits:

- source-data workflow documentation;
- schema documentation;
- validation procedure documentation;
- synthetic test fixtures that are clearly marked as synthetic;
- code or tests only in a separately approved implementation sprint;
- closeout documents and governance reviews.

Real raw fundamentals data must not be committed unless repository policy explicitly changes.

Generated operational outputs must not be committed unless repository policy explicitly allows them.

## 10. What Must Remain Local or Generated

The following should remain local, ignored, or generated unless repository policy explicitly allows committing them:

- real `data/raw/fundamentals_history.csv`;
- real source-data staging files;
- real provider exports;
- real local analyst working spreadsheets;
- generated `data/processed/fundamental_metrics.csv`;
- generated `data/processed/fundamental_quality.csv`;
- generated `data/processed/fundamental_analysis.csv`;
- generated logs from validation or pipeline runs;
- generated reports from operational validation.

If any generated output is proposed for commit later, that decision must be documented separately with repository policy, review scope, and rationale.

## 11. Validation Procedure

Initial validation procedure:

1. Prepare local `fundamentals_history.csv` draft.
2. Confirm the file contains only approved raw-history columns.
3. Run raw-history validation using the implemented validator.
4. Fix missing required columns, duplicate keys, invalid dates, invalid numeric fields, or forbidden fields.
5. Generate metrics locally to an explicit temporary or local ignored path.
6. Generate quality compatibility output locally to an explicit temporary or local ignored path.
7. Generate analysis output locally to an explicit temporary or local ignored path.
8. Review row counts and failure messages.
9. Confirm no generated outputs are staged for commit.
10. Record validation findings in a sprint report or closeout document.

Validation must confirm:

- required columns are present;
- `ticker` + `fiscal_year` + `fiscal_period` keys are unique;
- dates are parseable when present;
- numeric fields are parseable when present;
- missing numeric values remain blank, not inferred;
- forbidden semantic columns are absent;
- generated metrics are deterministic;
- quality output is row-preserving for the selected upstream universe;
- analysis output remains descriptive;
- `fundamental_analysis.csv` is not required downstream.

## 12. Analyst Review Procedure

Analyst review should occur after structural validation and before broad data expansion.

Review questions:

1. Are source references traceable?
2. Are fiscal periods and report dates correct?
3. Are raw statement values copied from the source without inference?
4. Are missing fields documented honestly?
5. Are currencies consistent within comparable rows?
6. Are restatements, adjusted values, or source-specific definitions noted?
7. Are calculated metrics plausible for the source facts?
8. Are quality classifications descriptive and reviewable?
9. Are analysis states descriptive and reviewable?
10. Are there any hidden allocation, ranking, scoring, eligibility, tradeability, urgency, or conviction semantics?

Analyst review may approve a controlled sample for broader source-data expansion. It may not authorize Decision Engine consumption or allocation logic.

## 13. Governance Boundaries

The workflow must preserve the project doctrine:

```text
classification upstream
allocation downstream
Decision Engine = ONLY allocation authority
```

Source-data workflow may:

- gather source evidence;
- record raw statement facts;
- validate schema and completeness;
- identify missing, stale, partial, or review-required source data;
- support deterministic metric generation;
- support descriptive analysis review.

Source-data workflow may not:

- create allocation decisions;
- create buy/sell decisions;
- create final actions;
- create eligibility gates;
- create tradeability semantics;
- create urgency or conviction;
- rank or score opportunities;
- filter rows;
- override the Decision Engine;
- make `fundamental_analysis.csv` a downstream-required dependency.

`fundamental_analysis.csv` remains descriptive and optional. Any Decision Engine consumption must be specified and approved in a separate future sprint.

## 14. Risks and Controls

| Risk | Control |
|---|---|
| Source values are copied without traceable evidence. | Require specific `source_reference` and reviewer traceability. |
| Raw history becomes mixed with ratios or analysis states. | Validate allowed schema and reject forbidden columns. |
| Missing values are inferred silently. | Require blanks for missing numeric values and notes for caveats. |
| Generated outputs are accidentally committed. | Keep real raw and generated outputs local or ignored unless policy changes. |
| Provider/API usage begins without approval. | Keep initial workflow manual or controlled semi-manual; defer automation to BL-0017 or a separate sprint. |
| Scraping creates legal, reliability, or maintenance risk. | No scraping is approved in this sprint. |
| Analyst interpretation becomes hidden Decision Engine logic. | Keep quality and analysis descriptive; require separate Decision Engine consumption specification. |
| Local ignored data drifts from documentation. | Record validation findings and source-data sample decisions in sprint reports. |
| Currency or period mismatches distort metrics. | Require currency, period, report date, and notes review before expansion. |

## 15. Recommended Next Sprint

Recommended next sprint:

```text
Controlled Real Fundamentals Sample Sprint
```

Purpose:

- prepare a small real source-supported local `fundamentals_history.csv` sample;
- validate it with the implemented raw-history validator;
- generate local metrics, quality, and analysis outputs;
- review output structure and analyst usefulness;
- document whether the workflow is ready to scale.

Alternative:

```text
Decision Engine Consumption Specification
```

This should occur only after the source-data workflow is operationally proven. It must remain a separate decision and must preserve the Decision Engine as the only allocation authority.

## 16. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

The remaining work is covered by BL-0015 and, for future provider/API automation, BL-0017. This document does not add a new backlog item.

## Validation

Validation commands:

```bash
git status
git diff --check
```

Validation expectation:

- only documentation files changed;
- no scripts changed;
- no tests changed;
- no data changed;
- no reports changed;
- no generated files changed;
- no CSV files changed;
- no workflow files changed;
- no runtime behavior changed;
- no provider/API calls performed;
- no scraping performed.
