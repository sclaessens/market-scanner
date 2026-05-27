# Functional Analysis Contract

Status: ACTIVE ANALYSIS CONTRACT

## Purpose

This document defines how fundamentals outputs should support user workflows, acceptance criteria, and future sprint readiness.

The Functional Analyst owns business requirements and acceptance criteria. The Functional Analyst does not own formulas or code implementation. No functional requirement may add hidden allocation semantics upstream.

## Scope

Functional analysis defines:

- expected operator workflow;
- data creation workflow requirements;
- validation workflow requirements;
- dashboard and reporting needs;
- acceptance criteria for raw data, metrics, quality, and analysis layers;
- readiness definitions for implementation and data creation;
- user-facing clarity requirements.

Functional analysis does not authorize:

- code changes;
- formula changes;
- runtime implementation;
- raw data edits;
- provider/API usage;
- scraping;
- allocation logic;
- Decision Engine loosening;
- reporting-based decision logic.

## Expected user workflow

The intended operator workflow is:

1. Maintain or collect raw historical fundamentals for the relevant ticker universe.
2. Validate source evidence, fiscal periods, dates, and currency metadata.
3. Calculate deterministic metrics from raw history.
4. Classify data quality and readiness.
5. Classify descriptive financial characteristics.
6. Review dashboard or report summaries for visibility.
7. Allow downstream layers to consume descriptive outputs under approved contracts.

The operator should be able to see what is missing, what is ready, what requires review, and why.

## Operator visibility needs

The operator needs clear visibility into:

- which tickers have raw history;
- how many fiscal years are available;
- whether 3-year and 5-year coverage exists;
- which source references support each row;
- which metrics could be calculated;
- which metrics are missing and why;
- which rows require human review;
- which descriptive analysis states are available;
- whether downstream consumption is blocked by quality issues.

Operator visibility must not become ranking, priority, tradeability, urgency, conviction, eligibility, allocation, or buy/sell guidance.

## Batch data creation workflow

The future batch workflow should prioritize raw history coverage rather than narrow per-3-ticker metric pilots.

Target workflow:

1. Select the relevant ticker universe from approved scanner or metadata sources.
2. Collect raw historical statement values for each ticker.
3. Target at least 3 completed fiscal years per ticker.
4. Prefer 5 fiscal years when source availability allows.
5. Store source references, period metadata, source freshness, extraction date, and notes for every row.
6. Validate raw rows before calculating metrics.
7. Calculate metrics only after enough source-supported raw data exists.

Manual pilots may still be used for learning or validation, but they must not be the main operating model.

## Validation workflow

Validation should happen in focused steps:

| Step | Functional purpose |
|---|---|
| Raw history validation | Confirm source evidence, period metadata, currency, and required fields. |
| Metrics validation | Confirm formulas can be calculated deterministically. |
| Quality validation | Confirm readiness and review-required classifications. |
| Analysis validation | Confirm descriptive states match approved financial meanings. |
| Downstream validation | Confirm descriptive outputs are consumed without authority leakage. |

Validation should be timeout-safe and should avoid unnecessary full pipeline execution when documentation or focused contract checks are enough.

## Dashboard and reporting needs

Future dashboards and reports should communicate:

- raw coverage by ticker;
- fiscal-year count;
- consecutive-year availability;
- missing raw fields;
- missing metric inputs;
- calculated metric availability;
- quality state;
- analysis state;
- review reasons;
- source freshness.

Dashboards and reports must remain communication-only. They may group and summarize, but they must not create prioritization, allocation, ranking, urgency, or decision logic.

## Acceptance criteria: raw data layer

Raw history is acceptable when:

- required columns exist;
- ticker and fiscal period identity are present;
- period end date, report date, source freshness date, and extraction date are distinct where available;
- currency is present;
- all numeric values are raw reported values or explicitly approved source-supported fields;
- every row has a source reference;
- notes capture caveats without inventing values;
- no quality, analysis, scoring, ranking, allocation, or buy/sell fields exist.

## Acceptance criteria: metrics layer

Metrics are acceptable when:

- formulas match the active technical calculation specification;
- missing inputs produce null or review-required outputs rather than guessed values;
- zero denominators are handled deterministically;
- negative and sign-change cases are preserved and flagged;
- period and currency consistency are checked;
- outputs remain calculation-only;
- no metric output creates allocation, scoring, ranking, tradeability, urgency, conviction, eligibility, or hidden filtering semantics.

## Acceptance criteria: quality layer

Quality classification is acceptable when:

- it distinguishes source missing, raw history completeness, metric readiness, and review-required conditions;
- it does not interpret whether a business is good or bad;
- it explains why data is incomplete or ready;
- it preserves row traceability;
- it avoids buy/sell, allocation, ranking, scoring, urgency, conviction, tradeability, or eligibility semantics.

## Acceptance criteria: analysis layer

Fundamental analysis is acceptable when:

- it classifies business and financial characteristics descriptively;
- it uses approved metrics and quality states;
- it preserves review-required caveats;
- it distinguishes raw facts, calculations, and interpretation;
- it avoids allocation, ranking, scoring, tradeability, urgency, conviction, eligibility, buy/sell, or hidden filtering semantics.

## Ready for data creation

A fundamentals workflow is ready for data creation when:

- the raw history schema is approved;
- source evidence requirements are clear;
- date semantics are clear;
- ticker universe selection is defined;
- minimum fiscal-year target is defined;
- validation checks are documented;
- review-required handling is documented;
- local ignored data policy is understood.

Ready for data creation does not mean ready for runtime implementation.

## Ready for implementation

A fundamentals workflow is ready for implementation when:

- platform contract is approved;
- calculation specification is approved;
- financial analysis contract is approved;
- functional acceptance criteria are approved;
- technical architecture contract is approved;
- migration strategy is documented;
- test expectations are documented;
- implementation scope is explicitly authorized.

Ready for implementation does not authorize implementation by itself. Developer / Codex implementation requires explicit approval.

## Functional anti-patterns

Avoid:

- adding all responsibilities to one artifact;
- treating stale generated outputs as active source truth;
- expanding raw schemas for every desired ratio;
- using per-3-ticker pilots as the main workflow;
- allowing dashboard summaries to imply priority or action;
- hiding review-required cases;
- creating upstream fields that sound like Decision Engine decisions.

## Handoff

Functional Analyst handoff to Technical Analyst:

- user workflow;
- acceptance criteria;
- operator visibility needs;
- validation expectations;
- readiness definitions.

Functional Analyst handoff to Financial Analyst:

- user-facing interpretation needs;
- review cases requiring financial meaning;
- descriptive state requirements.

Functional Analyst handoff to PM / Scrum Master:

- future sprint readiness;
- scope boundaries;
- acceptance risks;
- backlog impact assessment input.