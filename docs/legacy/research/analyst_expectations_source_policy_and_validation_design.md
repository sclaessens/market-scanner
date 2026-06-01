# Analyst Expectations Source Policy and Validation Design

## 1. Status and Scope

This document is a documentation-only source-policy and validation-design document for analyst expectations research.

It prepares governance, source-evaluation, field-definition, point-in-time, and validation rules before any data collection, automation, or integration is authorized.

This document does not implement:

- code;
- tests;
- CSV files;
- generated artifacts;
- reports;
- workflows;
- provider integration;
- provider/API calls;
- scraping;
- credentials or secrets;
- runtime orchestration;
- daily ingestion;
- backtesting code;
- Reporting changes;
- Telegram changes;
- scanner changes;
- Decision Engine changes;
- portfolio files;
- watchlist files;
- fundamentals files;
- runtime behavior changes.

No sprint is closed or certified complete by this document.

## 2. Background

This document follows `docs/research/analyst_expectations_and_backtesting_research_plan.md`.

It also follows backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy` in `docs/sprints/project_backlog.md`.

The research plan established analyst expectations, analyst consensus, price targets, earnings expectations, revenue expectations, point-in-time storage, and historical validation as future research topics only.

The purpose of this document is to define source policy and validation rules before:

- analyst expectations data is collected;
- source comparison is treated as operational input;
- provider/API integration is proposed;
- daily ingestion is designed;
- historical validation is implemented;
- Decision Engine integration is considered.

## 3. Research-Only Boundary

Analyst expectations remain research-only.

Analyst expectations must not become buy/sell advice.

Analyst consensus must not become:

- ranking;
- scoring;
- allocation;
- tradeability;
- urgency;
- conviction;
- eligibility;
- hidden filtering;
- Reporting recommendation;
- Telegram recommendation;
- Decision Engine bypass.

Analyst expectation research may describe source-supported observations, data quality, coverage, conflicts, freshness, and historical validation outcomes. It must not create production decision authority.

Any future Decision Engine use requires separate governance, separate design approval, explicit authority boundaries, implementation scope, tests, audit rules, and acceptance criteria.

## 4. Candidate Source Categories

No analyst expectations source is approved by this document.

All sources listed here are candidates pending evaluation.

A future source investigation may evaluate the following candidate source categories.

### 4.1 Official company investor relations materials

Official company investor relations materials may provide company guidance, investor presentations, earnings releases, transcripts, or management outlook.

These materials may be useful for source-supported expectations context, but they are not automatically analyst consensus data.

Evaluation must distinguish company guidance from third-party analyst estimates.

### 4.2 SEC filings or equivalent regulatory filings

SEC filings or equivalent regulatory filings may provide official historical disclosures, financial statements, risk factors, management discussion, and reported results.

Regulatory filings may support validation context and financial history, but they generally do not provide third-party analyst consensus directly.

Where relevant, jurisdiction-specific regulatory sources must be evaluated for stability, availability, and comparable disclosure structure.

### 4.3 Exchange or market-profile pages

Exchange or market-profile pages may provide ticker profiles, company metadata, sector classification, market data summaries, or links to research-related data.

If analyst ratings, price targets, or estimates are present, their source definition, update frequency, and terms of use must be evaluated before use.

### 4.4 Market data platforms

Market data platforms that publish analyst ratings, consensus values, price targets, earnings estimates, or revenue estimates may be candidate sources.

They must be evaluated for coverage, licensing, point-in-time support, stability, consistency, and suitability for automation.

Public web display of a current consensus value does not automatically make the source suitable for historical validation.

### 4.5 Paid, licensed, or API-based providers

Paid, licensed, or API-based providers may be considered only if later approved through separate provider governance.

Future provider approval must cover licensing, credentials, secrets handling, rate limits, caching, audit logs, source provenance, data-quality states, failure modes, and storage boundaries.

This document does not authorize account creation, credentials, API calls, ingestion, or automation.

### 4.6 Historical point-in-time data providers

Historical point-in-time data providers may be required for valid backtesting.

Candidate providers must demonstrate whether they can reconstruct what analyst expectations were known on a specific historical evaluation date.

Current snapshots are not sufficient for historical truth unless the provider offers point-in-time access or versioned historical snapshots.

## 5. Source Evaluation Criteria

Every candidate source should be evaluated against the following criteria before approval.

| Criterion | Required evaluation |
|---|---|
| Coverage by ticker | Determine which project tickers are covered and where coverage gaps exist. |
| Field availability | Identify which analyst expectation fields are available per ticker. |
| Definition clarity | Confirm how each source defines consensus, rating categories, price targets, estimates, revisions, and freshness dates. |
| Historical point-in-time availability | Determine whether the source supports reconstruction of values available on historical evaluation dates. |
| Freshness | Identify the latest source freshness date and whether it is distinct from collection date. |
| Update frequency | Determine whether values update daily, weekly, after earnings, after analyst actions, or irregularly. |
| Licensing and terms of use | Confirm whether manual research, storage, redistribution, automation, or API usage is permitted. |
| Rate limits | Identify usage limits for manual access, automated access, API usage, or bulk retrieval. |
| Stability of references/URLs | Determine whether source URLs, identifiers, or references are stable enough for audit trails. |
| Consistency across tickers | Determine whether fields and definitions are consistent across markets, sectors, and exchanges. |
| Conflict handling | Determine how source values should be compared when candidate sources disagree. |
| Auditability | Confirm whether raw values, normalized values, source references, and collection metadata can be preserved. |
| Suitability for automation | Evaluate whether future automation is technically and legally appropriate after separate approval. |

## 6. Candidate Fields

The following fields are research candidate fields only.

They are not runtime contract fields.

They do not authorize CSV creation, schema changes, pipeline integration, Reporting changes, Telegram changes, or Decision Engine changes.

| Candidate field | Purpose |
|---|---|
| `ticker` | Identifies the security. |
| `as_of_date` | Date the source value represents or was valid as of. |
| `collection_date` | Date the value was collected by the project or research process. |
| `source_name` | Name of the source or provider. |
| `source_url_or_reference` | Stable source URL, identifier, filing reference, provider reference, or audit reference. |
| `source_freshness_date` | Date shown by the source as its latest update, freshness date, or estimate date. |
| `consensus_rating` | Source-published or later governed internally calculated consensus rating. |
| `analyst_count` | Number of analysts supporting the consensus or estimate set. |
| `buy_count` | Count of buy-equivalent ratings if source-supported. |
| `hold_count` | Count of hold-equivalent ratings if source-supported. |
| `sell_count` | Count of sell-equivalent ratings if source-supported. |
| `average_price_target` | Average source-supported price target. |
| `low_price_target` | Lowest source-supported price target. |
| `high_price_target` | Highest source-supported price target. |
| `current_price_at_collection` | Market price observed at collection time, if source-supported or separately governed. |
| `implied_upside_pct` | Difference between current price and average price target, if later approved and formula-defined. |
| `current_year_eps_estimate` | Current fiscal-year EPS estimate. |
| `next_year_eps_estimate` | Next fiscal-year EPS estimate. |
| `current_year_revenue_estimate` | Current fiscal-year revenue estimate. |
| `next_year_revenue_estimate` | Next fiscal-year revenue estimate. |
| `estimate_revision_direction` | Source-supported direction of estimate changes, if available. |
| `raw_source_value` | Original source value before normalization. |
| `normalized_value` | Later governed normalized representation of the source value. |
| `normalization_note` | Explanation of mapping, assumptions, or limitations. |
| `data_quality_state` | Research data quality label such as complete, partial, stale, conflict, missing, or unsupported, subject to future definition. |
| `data_quality_notes` | Human-readable notes about gaps, conflicts, source issues, or interpretation limits. |

## 7. Consensus Definition Policy

Consensus may be represented in one of two ways.

### 7.1 Source-published consensus

A source-published consensus is a consensus value directly published by a candidate source.

The research process must preserve the source definition where available and must avoid reinterpreting the value as project-defined authority.

### 7.2 Internally calculated consensus

An internally calculated consensus may be considered only if a later document approves the formula and supporting rules.

Internal consensus calculation requires a later approved formula and must define:

- rating category mapping;
- minimum analyst count;
- stale estimate handling;
- missing estimate handling;
- conflict rules;
- whether price-target consensus and rating consensus are separate signals;
- whether raw and normalized values are both stored.

Until such a formula is approved, internally calculated consensus must not be used as scoring, ranking, tradeability, conviction, urgency, eligibility, allocation, or Decision Engine input.

## 8. Point-in-Time Requirements

Historical validation must prevent look-ahead bias.

The following rules are mandatory for future analyst expectations validation design:

- every value must have an `as_of_date`;
- every value must have a `collection_date`;
- future-published values must not be used for simulated past decisions;
- revised analyst data must not overwrite historical snapshots without preserving versioning;
- unavailable historical data must remain unavailable;
- current web pages must not be treated as historical truth unless point-in-time support exists;
- source freshness must be distinguishable from project collection timing;
- source corrections and provider revisions must be auditable;
- missing data must not be silently filled with later values;
- stale values must remain visible rather than converted into current signals.

A valid future research dataset must support reconstruction of what the project could have known at each historical evaluation date.

## 9. Historical Validation Design

A future historical validation design should follow a controlled research sequence.

1. Choose historical evaluation dates.
2. Reconstruct the scanner universe for those dates.
3. Collect only source data available at that time.
4. Store analyst expectation snapshots.
5. Preserve source references, freshness dates, collection dates, and data-quality states.
6. Measure forward outcomes from each signal date.
7. Compare analyst expectation buckets against future returns.
8. Compare analyst expectations against baseline scanner, context, timing, and fundamental classifications.
9. Evaluate whether analyst expectations add incremental value beyond existing project classifications.
10. Document limitations, missing data, survivorship risk, look-ahead risk, sample-size limits, and source conflicts.

Analyst expectation buckets should remain research groupings only. They must not become production ranking or allocation buckets unless a later governed Decision Engine design explicitly authorizes limited use.

## 10. Suggested Forward Windows

Future validation may evaluate forward outcomes over the following windows:

- 2 weeks;
- 1 month;
- 3 months;
- 6 months;
- 12 months.

The validation design should define whether returns are measured from close-to-close, next-open-to-close, next-close-to-close, or another explicitly governed method.

The return method must be consistent across samples.

## 11. Suggested Validation Metrics

Future validation may include the following metrics:

- forward return by consensus bucket;
- median forward return;
- average forward return;
- hit rate;
- downside frequency;
- drawdown after signal date;
- sector-adjusted return;
- market-adjusted return;
- comparison against scanner grade alone;
- comparison against scanner plus fundamentals;
- sample size;
- confidence limitations.

Validation must explicitly disclose sample-size limitations, missing-data patterns, survivorship bias risk, point-in-time limitations, and source-definition differences.

Positive historical results must not automatically authorize production integration.

Negative or inconclusive results must remain documented and should not be hidden.

## 12. Exception-Based Human Review

The operator should not be required to review every ticker.

Human review should focus on exception-based controls, including:

- approving source policy;
- approving field definitions;
- reviewing conflicts;
- reviewing missing data exceptions;
- reviewing stale data exceptions;
- reviewing a limited sample;
- approving future automation proposals.

Routine ticker-by-ticker review should be avoided unless required for a limited audit sample or an exception queue.

## 13. Future Daily Ingestion Design Boundary

Daily analyst expectations ingestion is future scope only.

It requires separate governance for:

- provider choice;
- credentials and secrets;
- rate limits;
- caching;
- freshness thresholds;
- audit logs;
- data-quality states;
- failure handling;
- tests;
- CI checks;
- storage location;
- integration boundaries.

No daily ingestion, provider/API integration, scraping, credentials, runtime orchestration, or storage artifact is authorized by this document.

## 14. Pipeline Integration Boundary

A future research pipeline may include:

- analyst expectation snapshot storage;
- point-in-time source data;
- historical scanner reconstruction;
- forward outcome measurement;
- research reporting;
- exception reporting.

This remains separate from production Decision Engine authority.

Research reporting may communicate validation findings, source gaps, exception queues, and confidence limits. It must not create production recommendations, Telegram action language, hidden ranking, hidden filtering, or allocation semantics.

## 15. Backlog Impact Assessment

Existing backlog item `BL-0018` is sufficient.

`BL-0018` already captures governed analyst expectations, point-in-time storage, historical validation, backtesting research, provider/API restrictions, and future Decision Engine boundary controls.

Backlog impact assessment:
- No new backlog items identified.

## 16. Recommended Next Step

The recommended next step is a future documentation-only source comparison matrix for candidate providers and sources.

That future step should compare candidate sources without:

- collecting runtime data;
- calling provider APIs;
- scraping websites;
- creating credentials or secrets;
- creating CSV files;
- changing runtime artifacts;
- changing scanner logic;
- changing Decision Engine logic;
- changing Reporting logic;
- changing Telegram logic.

The comparison matrix should evaluate candidate sources against the criteria in this document and identify which sources, if any, are suitable for later governed research collection.
