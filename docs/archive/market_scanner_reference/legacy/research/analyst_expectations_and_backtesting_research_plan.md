# Analyst Expectations and Historical Validation Research Plan

## 1. Status and Scope

This document is a documentation-only research and governance plan.

It records a future research path for analyst expectations, analyst consensus data, price targets, earnings expectations, and historical validation/backtesting inside the `market-scanner` project.

This document does not implement:

- code changes
- tests
- CSV changes
- generated artifacts
- reports
- workflows
- provider integration
- provider/API calls
- credentials or secrets
- runtime orchestration
- daily ingestion
- Reporting changes
- Telegram changes
- scanner changes
- Decision Engine changes
- portfolio changes
- watchlist changes
- fundamentals source changes

No sprint is closed or certified complete by this document.

This document is research-only. It does not authorize analyst expectations to influence allocation, tradeability, urgency, conviction, eligibility, ranking, scoring, filtering, Reporting language, Telegram language, or Decision Engine behavior.

## 2. Background

Operational Sprint 5 established a controlled path for source-supported metadata and fundamentals coverage.

The current OS5 flow has already clarified that:

- scanner A/B rows are the preferred next source-data coverage target;
- scanner A/B selection is coverage-prioritization only;
- source data must preserve provenance and freshness metadata;
- ChatGPT may assist with research planning and source comparison but must not be treated as the factual source of record;
- human review should focus on source policy, exception review, and sampling rather than ticker-by-ticker mass validation;
- automated ingestion is a future governed design topic, not an ad hoc runtime change.

The operator has now identified an additional potential research signal: analyst expectations, including analyst consensus, ratings, price targets, and forward-looking estimates published by market data platforms or company/market sources.

This document captures that idea as a future observation and validation path.

## 3. Research Objective

The objective is to investigate whether analyst expectations can become a useful, source-supported, historically validated input to future market-scanner analysis.

The research must answer:

- which analyst expectation sources are acceptable;
- which analyst expectation fields are stable, available, and comparable;
- how source provenance and freshness should be stored;
- how consensus values should be calculated or preserved;
- whether analyst expectations have measurable predictive value in historical tests;
- whether the signal adds value beyond scanner, context, timing, fundamentals, and portfolio metadata;
- whether any future integration should remain research-only or be proposed for governed Decision Engine consideration.

Until that research is completed and separately approved, analyst expectations remain observation-only.

## 4. Candidate Data Domain

Analyst expectations may include, subject to future source evaluation:

- consensus rating;
- rating distribution, such as buy, overweight, hold, underweight, and sell counts;
- analyst count;
- average price target;
- low price target;
- high price target;
- current price at collection time;
- implied upside or downside percentage;
- current-year earnings estimate;
- next-year earnings estimate;
- current-year revenue estimate;
- next-year revenue estimate;
- estimate revision direction where available;
- source freshness date;
- source name;
- source URL or stable reference;
- collection timestamp;
- data quality notes.

This list is not a runtime contract. It is a candidate research field set.

No new runtime field is introduced by this document.

## 5. Candidate Source Investigation

A future source investigation should compare candidate analyst expectation sources before any data collection or automation is authorized.

Candidate source categories include:

1. market data platforms that publish analyst ratings, price targets, or estimates;
2. exchange or market-profile pages where analyst consensus is available;
3. company investor-relations materials when forward-looking expectations are explicitly provided;
4. licensed or API-based providers if later approved through provider governance;
5. historical market data providers capable of point-in-time analyst estimate reconstruction.

The source investigation must evaluate:

- availability by ticker;
- historical availability;
- point-in-time support;
- update frequency;
- definition clarity;
- licensing and terms of use;
- stability of URLs or identifiers;
- data consistency across tickers;
- coverage gaps;
- rate limits;
- whether manual collection, provider-assisted collection, or automated ingestion is appropriate.

No provider/API call is authorized by this document.

## 6. Source Provenance and Freshness Rules

Every future analyst expectation record must be source-supported.

Every record must preserve:

- ticker;
- as-of date;
- collection date;
- source name;
- source URL or stable reference where possible;
- source freshness date or last-updated date where available;
- raw source value where possible;
- normalized value if normalization is required;
- normalization note if applicable;
- data quality note if the source is partial, stale, ambiguous, or inconsistent.

If a source does not provide sufficient evidence, the value must remain empty or be marked unavailable.

ChatGPT output must never be stored as the factual source of record.

## 7. Consensus Handling

Analyst consensus may be stored in two ways, subject to future design:

1. preserve source-published consensus directly with source metadata;
2. calculate an internal consensus from source-supported component values.

Internal consensus calculation must not be implemented until definitions are approved.

Any future consensus calculation must define:

- included rating categories;
- how ratings are mapped to numeric or categorical values;
- treatment of missing ratings;
- treatment of stale ratings;
- minimum analyst count;
- source hierarchy;
- conflict handling across sources;
- whether price-target consensus and rating consensus are separate signals;
- whether consensus is stored as raw observation, normalized feature, or both.

Consensus must not be treated as a buy/sell decision. It may become a research feature only after validation.

## 8. Historical Validation and Backtesting Objective

Historical validation is required before analyst expectations may influence any analysis pipeline.

The project should investigate whether analyst expectations available at a past date had predictive value over future return windows.

The research should use a walk-forward or point-in-time methodology.

Example validation design:

1. select a historical evaluation date;
2. reconstruct the scanner universe and relevant source-supported data as of that date;
3. collect analyst expectations that were available as of that date only;
4. create observation records without future information;
5. measure subsequent price behavior across forward windows;
6. compare results with baseline scanner/context/timing/fundamental classifications;
7. evaluate whether analyst expectations add incremental explanatory or predictive value.

Suggested forward windows:

- 2 weeks;
- 1 month;
- 3 months;
- 6 months;
- 12 months.

Backtesting must remain research-only until separately governed.

## 9. Look-Ahead Bias Controls

The research must explicitly prevent look-ahead bias.

Controls should include:

- point-in-time source data only;
- as-of dates for all observations;
- collection dates for all records;
- no use of values published after the simulated decision date;
- no use of future price movement when creating signals;
- no survivorship-only universe unless explicitly documented;
- no manual reconstruction from future-known winners only;
- separate training, validation, and evaluation windows if model-like logic is later considered;
- explicit record of unavailable historical data.

If point-in-time analyst expectations cannot be obtained reliably, the research must document that limitation before any conclusions are drawn.

## 10. Validation Metrics

Future research may evaluate analyst expectations using metrics such as:

- forward return by analyst consensus bucket;
- hit rate versus forward return threshold;
- median and average forward return by bucket;
- downside frequency by bucket;
- maximum drawdown after signal date;
- sector-adjusted forward return;
- market-adjusted forward return;
- comparison against scanner grade alone;
- comparison against scanner plus fundamentals;
- incremental value over existing descriptive layers;
- stability across time windows;
- sample size and confidence limitations.

Metrics must be reported as research observations. They must not be converted into trading authority without separate governance.

## 11. Pipeline Research Path

The operator explicitly wants backtesting to become part of the future pipeline.

This document records that goal as future research and platform scope.

A future governed research pipeline may include:

1. historical scanner universe reconstruction;
2. point-in-time source-data ingestion;
3. analyst expectations snapshot storage;
4. fundamentals and metadata snapshot storage;
5. forward outcome measurement;
6. historical validation metrics;
7. research reporting;
8. exception reporting;
9. decision on whether a candidate signal merits future Decision Engine governance review.

This future research pipeline must remain separate from production Decision Engine authority until explicitly approved.

## 12. Daily Ingestion Future Scope

Daily ingestion of analyst expectations may be useful later, but it is not authorized by this document.

A future daily ingestion design would need to define:

- approved providers;
- credentials and secrets policy;
- licensing and terms-of-use compliance;
- rate-limit handling;
- retry behavior;
- caching;
- freshness thresholds;
- provenance fields;
- audit logs;
- exception handling;
- data-quality states;
- integration boundaries with existing layers;
- whether data is stored in intake, raw, research, or processed locations;
- tests and CI validation.

Daily ingestion should be proposed only after source investigation and historical validation design are complete.

## 13. Governance Constraints

This research plan preserves certified project doctrine:

- classification upstream;
- allocation downstream;
- Decision Engine is the only allocation authority;
- upstream layers classify and enrich only;
- reporting communicates only;
- no hidden filtering;
- no ranking authority outside approved Decision Engine logic;
- no scoring authority outside approved Decision Engine logic.

Analyst expectations must not introduce:

- allocation semantics;
- tradeability semantics;
- urgency semantics;
- conviction semantics;
- eligibility semantics;
- hidden filtering;
- Decision Engine bypass;
- Reporting-based recommendations;
- Telegram-based recommendations.

Any future integration into the Decision Engine requires a separate governed design, implementation plan, test plan, and approval.

## 14. Human Review Model

Human review should focus on method governance, exception handling, and sampling.

The operator should not be required to manually validate every ticker.

The preferred model is:

1. approve source policy;
2. approve field definitions;
3. approve freshness thresholds;
4. review exceptions and conflicts;
5. perform limited sampling review;
6. approve or reject future automation proposals.

This supports the operator's goal of building a system that is more reliable than ad hoc human judgment.

## 15. Relationship to Existing Backlog

Existing backlog item `BL-0017` covers governed automated data ingestion for fundamentals and portfolio metadata.

Analyst expectations and historical validation are related but distinct enough to require explicit backlog capture.

This plan therefore introduces a new backlog item for future analyst expectations and historical validation research strategy.

Backlog impact assessment:
- New backlog items identified and added to project_backlog.md

## 16. Recommended Next Step

The next recommended step is a documentation-only source-policy and validation-design sprint for analyst expectations.

That step should:

- compare candidate sources;
- define acceptable source hierarchy;
- define candidate analyst expectation fields;
- define point-in-time requirements;
- define historical validation methodology;
- define storage locations for research snapshots;
- define exception-review rules;
- decide whether daily ingestion should be proposed as a future implementation sprint.

No data collection, provider/API integration, runtime ingestion, backtesting code, or Decision Engine integration should begin until that source-policy and validation-design step is reviewed and approved.
