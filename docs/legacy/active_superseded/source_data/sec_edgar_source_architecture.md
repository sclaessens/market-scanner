# SEC EDGAR Source Architecture Finalization

Status: ACTIVE SPECIFICATION
Backlog driver: BL-0015
Related strategy: `docs/active/source_data/sec_edgar_fundamentals_source_strategy.md`
Sprint: SEC-1 — SEC EDGAR Source Architecture Finalization
Date: 2026-05-31

## 1. Purpose

This document finalizes the documentation-only SEC EDGAR source architecture for future fundamentals source-data implementation.

SEC-1 exists to define source architecture, boundaries, storage expectations, update expectations, and Codex handoff requirements before any runtime implementation begins.

SEC-1 does not authorize code changes, test changes, generated data commits, CSV commits, provider/API execution, SEC downloads, scraping, pipeline runs, Decision Engine changes, Reporting changes, Telegram changes, portfolio changes, ticker-category runtime logic, or runtime behavior changes.

## 2. Source Decision

SEC EDGAR is approved as the primary source candidate for the first governed real fundamentals source-data implementation sequence.

The approved initial source family is:

```text
SEC EDGAR Company Facts bulk data
```

The SEC EDGAR APIs provide JSON-formatted data through `data.sec.gov` without authentication or API keys. SEC documentation identifies submissions history and XBRL financial statement data as available API surfaces and identifies bulk archive ZIP files as the efficient way to fetch large amounts of API data.

Source references:

```text
https://www.sec.gov/search-filings/edgar-application-programming-interfaces
https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip
https://www.sec.gov/about/webmaster-frequently-asked-questions
```

## 3. Architecture Decision

The project should use a bulk-first SEC source architecture.

Approved target source sequence for future implementation:

```text
SEC Company Facts bulk archive
-> local ignored SEC cache
-> SEC CIK/ticker index
-> SEC coverage report
-> SEC XBRL tag mapping review
-> SEC-to-fundamentals raw history transformation
-> internal fundamentals history validation
-> fundamental metrics
-> fundamental quality compatibility
-> fundamental analysis classification
```

Only the later implementation sprints may build this sequence. SEC-1 only defines the architecture.

## 4. Why Bulk First

The bulk-first approach is approved because it:

- supports broad-data-first inspection instead of five-ticker manual entry;
- reduces repeated request pressure on SEC endpoints;
- enables local cache validation before transformation;
- supports coverage analysis for scanner, watchlist, and portfolio universes;
- supports systematic XBRL tag mapping investigation;
- makes refresh/update behavior easier to govern;
- keeps the five review tickers as validation samples rather than the entire strategy.

SEC documentation describes the bulk archive ZIP files as the efficient way to fetch large amounts of API data, and describes the `companyfacts.zip` archive as containing data from the XBRL Frame API and XBRL Company Facts API.

## 5. Approved Access Method For Future Implementation

Future implementation may use only the following approved source access pattern:

1. retrieve the official SEC Company Facts bulk archive from the SEC archive URL;
2. store the downloaded archive or extracted cache in a local ignored cache path;
3. read from the local cache for downstream indexing and transformation;
4. record source freshness and extraction metadata;
5. avoid committing real downloaded SEC data unless repository policy explicitly changes.

Future implementation should not start with repeated per-company API calls unless a later sprint documents a narrow exception.

## 6. Not Approved Access Methods

The following are not approved by SEC-1:

- browser scraping;
- HTML scraping;
- ad hoc page parsing;
- uncontrolled repeated per-company API calls;
- scheduled unattended SEC refresh;
- committing full SEC archive files;
- committing extracted generated SEC source caches;
- using non-SEC providers as a replacement source without separate approval;
- inferring missing financial values from unrelated sources;
- using ChatGPT as a factual source for financial statement values.

## 7. Fair Access and User-Agent Requirements

Future implementation must comply with SEC programmatic-access expectations.

Minimum implementation requirements:

- use a descriptive User-Agent identifying the project or operator contact path where required;
- use HTTPS official SEC endpoints only;
- prefer bulk archive access for large data intake;
- avoid unnecessary repeated requests;
- avoid parallel request bursts;
- support local caching;
- fail clearly when the SEC endpoint is unavailable;
- document extraction date and source freshness date;
- keep provider/API execution out of documentation-only sprints.

SEC-1 does not perform any provider/API calls and does not test SEC access.

## 8. Local Cache Policy

Future implementation should use a local ignored SEC cache.

Recommended path family:

```text
data/local/sec_edgar/
```

Potential internal structure:

```text
data/local/sec_edgar/companyfacts/companyfacts.zip
data/local/sec_edgar/companyfacts/extracted/
data/local/sec_edgar/submissions/submissions.zip
data/local/sec_edgar/submissions/extracted/
data/local/sec_edgar/metadata/sec_cache_manifest.json
```

This path is a proposed implementation direction only. SEC-2 must confirm whether this path is already ignored or must be added to `.gitignore` as part of implementation.

SEC-1 does not create these directories or modify `.gitignore`.

## 9. Generated Data Commit Policy

The approved policy direction is:

- do not commit SEC bulk archives;
- do not commit extracted SEC cache files;
- do not commit generated real raw fundamentals history by default;
- do not commit generated metrics, quality, or analysis outputs by default;
- allow documentation-only summaries only when they contain no uncontrolled generated operational dataset;
- allow synthetic fixtures only when clearly marked and minimized;
- allow future generated artifact commits only after a specific repository-policy decision.

SEC-2 must explicitly validate that no generated SEC archive, cache, CSV, log, report, or processed output is staged for commit.

## 10. Source Freshness and Refresh Policy

Future implementation should separate freshness concepts.

Required metadata concepts:

| Field | Meaning |
|---|---|
| `source_freshness_date` | Date the SEC source archive or source facts were checked. |
| `extraction_date` | Date the local project extracted or transformed the SEC data. |
| `period_end_date` | Fiscal period end date from the SEC fact context or internal mapping. |
| `report_date` | Filing/report date when available through SEC source metadata. |

Future refresh should be manually triggered or explicitly commanded at first. Fully automated scheduled refresh is not approved by SEC-1.

## 11. Submissions Data Role

The primary data source is Company Facts, but SEC submissions data may be needed for metadata and coverage.

Potential submissions use cases:

- CIK identity review;
- company names;
- ticker symbols;
- exchanges;
- filing history metadata;
- filing dates;
- coverage completeness review.

Future implementation may use the official `submissions.zip` bulk archive only if SEC-2 or SEC-3 scope explicitly includes it.

## 12. CIK/Ticker Mapping Requirements

Future implementation must handle the difference between ticker-first project inputs and CIK-first SEC data.

Requirements:

- normalize CIKs to SEC 10-digit format with leading zeros when constructing SEC paths;
- map tickers to CIKs using an approved SEC-supported or source-supported mapping path;
- preserve original project ticker identity;
- record mapping status;
- handle missing mapping explicitly;
- handle ambiguous mapping explicitly;
- handle ticker changes and class-share suffixes conservatively;
- do not infer mappings silently;
- do not drop rows because CIK mapping is missing unless a later validation contract explicitly allows fail-fast behavior for a configured input.

Suggested future mapping states:

```text
CIK_MATCHED
CIK_MISSING
CIK_AMBIGUOUS
CIK_REVIEW_REQUIRED
CIK_NOT_SEC_REPORTER
```

These states are descriptive source-data readiness states only. They do not create allocation, eligibility, ranking, scoring, tradeability, urgency, conviction, buy/sell, or hidden filtering semantics.

## 13. Coverage Report Requirements

Future SEC coverage work should produce a coverage report before transformation is trusted.

The coverage report should answer:

- which project tickers have a CIK match;
- which tickers are missing or ambiguous;
- which tickers have SEC Company Facts data;
- which tickers lack sufficient facts for internal fundamentals fields;
- which fields are consistently available;
- which fields are missing or tag-dependent;
- which review tickers are fully supported;
- whether the scanner universe and portfolio universe are sufficiently covered.

The coverage report should remain documentation-safe or generated-local unless a later policy approves committing it.

## 14. XBRL Tag Mapping Requirements

SEC XBRL data does not guarantee a simple one-to-one mapping into the internal fundamentals fields.

Future XBRL mapping must identify primary and alternate tags for:

```text
revenue
gross_profit
operating_income
net_income
diluted_eps
total_debt
total_equity
free_cash_flow
```

Mapping review must consider:

- taxonomy namespace;
- tag meaning;
- unit type;
- fiscal period duration;
- annual versus quarterly data;
- amended facts;
- duplicate facts;
- company-specific extensions;
- missing values;
- sign conventions;
- comparability across companies;
- whether the field is reliable enough to remain core.

SEC-1 does not approve final tag mappings. SEC-4 should perform the mapping investigation.

## 15. Internal Transformation Boundary

Future SEC-to-fundamentals transformation must produce internal raw history shaped like the existing target fundamentals history contract.

Target internal fields remain:

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

Rules:

- raw history stores source-supported facts only;
- no ratios in raw history;
- no analysis states in raw history;
- no quality states in raw history;
- no allocation, ranking, scoring, eligibility, urgency, conviction, tradeability, buy/sell, entry, stop, target, or final action fields;
- missing numeric values remain blank/null rather than inferred;
- source reference must be specific enough for later review;
- transformation must preserve row identity and source evidence.

## 16. Failure Behavior Requirements

Future implementation should use explicit failure behavior.

Recommended failure policy:

| Situation | Behavior |
|---|---|
| SEC cache missing when explicitly required | Fail clearly before transformation. |
| SEC archive unavailable during refresh | Fail clearly; preserve existing local cache if safe and documented. |
| SEC archive downloaded but invalid | Fail before extraction or transformation. |
| CIK mapping missing for a ticker | Mark descriptive mapping status; do not infer. |
| XBRL field missing | Leave internal value blank/null and record review note. |
| Duplicate or conflicting facts | Mark review-required unless deterministic rule is later approved. |
| Unsupported unit | Leave value blank/null and record review note. |
| Non-US or non-SEC reporter | Mark descriptive source status; do not infer source data. |

Fail-fast behavior should protect invalid configured source files. Descriptive missing-data behavior should preserve opportunity rows where missing data is expected or source coverage is incomplete.

## 17. Validation Requirements For SEC-2

SEC-2 implementation must validate:

- official source URL configuration;
- local cache path creation or existence;
- no generated SEC data committed;
- no generated CSV outputs committed;
- User-Agent configuration documented or required;
- archive existence and readable structure;
- deterministic local extraction behavior;
- clear failure when archive is missing or invalid;
- no Decision Engine changes;
- no Reporting changes;
- no downstream dependency on `fundamental_analysis.csv`;
- no allocation, ranking, scoring, tradeability, urgency, conviction, eligibility, buy/sell, or hidden filtering semantics.

SEC-2 tests should use minimized fixtures rather than real full SEC bulk data.

## 18. Fixture Policy

Future tests should not depend on live SEC downloads.

Approved fixture approach:

- use small synthetic SEC-like JSON fixtures;
- include only fields needed to test parsing and validation paths;
- avoid storing real full SEC data in tests;
- mark fixtures clearly as synthetic or minimized;
- test failure paths with small invalid fixture files;
- keep network access out of automated tests unless a separate integration-test policy is approved.

## 19. Security and Reliability Controls

Future implementation must include controls for:

- official SEC host allowlist;
- HTTPS-only access;
- safe local file paths;
- no credential requirements;
- no unbounded extraction into uncontrolled paths;
- cache manifest review;
- deterministic extraction;
- fail-clear error messages;
- no silent fallback to unofficial sources;
- no web scraping fallback.

## 20. Review Ticker Set

The SEC review ticker set remains:

```text
NVDA
TWLO
ON
AMD
KEYS
```

These tickers should be used after SEC bulk intake and CIK mapping exist.

They are not the full intake scope and should not drive the architecture by themselves.

## 21. Sprint Split Confirmation

SEC work must remain split across smaller sprints.

Approved sequence:

| Sprint | Type | Purpose |
|---|---|---|
| SEC-1 | Documentation/specification | Finalize SEC source architecture. |
| SEC-2 | Codex implementation | Implement controlled local SEC bulk intake/cache. |
| SEC-3 | Codex implementation plus documentation | Build CIK/ticker index and coverage report. |
| SEC-4 | Documentation plus optional fixture tests | Investigate XBRL tag mapping. |
| SEC-5 | Documentation-first | Rationalize fundamental analysis against available SEC data. |
| SEC-6 | Codex implementation | Transform SEC facts to internal fundamentals history. |
| SEC-7 | Execution plus analyst review | Validate controlled real fundamentals analysis. |

Do not combine SEC-2 implementation with SEC-3 coverage, SEC-4 mapping, SEC-5 analysis rationalization, or SEC-6 transformation.

## 22. Codex Handoff For SEC-2

SEC-2 may be handed to Codex only after SEC-1 is accepted.

SEC-2 developer prompt must include:

- documentation-only source references from this document;
- approved official SEC Company Facts bulk URL;
- local ignored cache path requirement;
- no generated data commits;
- no pipeline integration beyond approved intake/cache scope;
- fixture-based tests only;
- no live SEC dependency in tests;
- no Decision Engine changes;
- no Reporting changes;
- no transformation into internal fundamentals history yet unless SEC-2 scope is explicitly expanded, which is not recommended;
- validation evidence requirements.

SEC-2 should be a narrow 3-point or smaller implementation sprint if it only builds controlled local bulk intake/cache. If it also includes CIK/ticker indexing or coverage reporting, it should be split.

## 23. Out Of Scope For SEC-1

SEC-1 does not:

- update the backlog;
- update role definitions;
- update the fundamentals operating workflow;
- update calculation registry entries;
- implement SEC downloads;
- create local cache directories;
- modify `.gitignore`;
- modify code;
- modify tests;
- modify data;
- modify reports;
- modify CSV files;
- modify generated files;
- modify workflow files;
- execute scripts;
- run the pipeline;
- call SEC APIs;
- scrape websites;
- perform live source-data validation;
- map final XBRL tags;
- transform SEC data into internal raw fundamentals history;
- approve Decision Engine consumption of `fundamental_analysis.csv`.

These are later-sprint concerns.

## 24. Backlog Impact Assessment

Backlog impact assessment:

- No backlog changes are made in SEC-1.

Rationale:

- BL-0015 remains the primary driver for approved fundamentals source data and quality classification.
- BL-0017 remains the future driver for governed automated ingestion strategy.
- SEC-1 finalizes source architecture but does not need to add a new backlog item.
- Any backlog status alignment should be handled in a separate backlog-alignment sprint if needed.

## 25. Closeout Decision

SEC-1 is complete when this document is reviewed and merged.

Completion means:

- SEC EDGAR Company Facts bulk data is the approved initial source architecture direction;
- SEC-2 can be prepared as a narrow Codex implementation sprint for controlled local SEC bulk intake/cache;
- CIK/ticker index, coverage reporting, XBRL mapping, analysis rationalization, and SEC-to-internal transformation remain separate later sprints.

## 26. No-Runtime-Change Confirmation

SEC-1 confirms:

- no scripts changed;
- no tests changed;
- no data changed;
- no reports changed;
- no CSV files changed;
- no generated files changed;
- no workflow files changed;
- no runtime behavior changed;
- no provider/API calls performed by this sprint;
- no SEC data downloaded by this sprint;
- no scraping performed;
- no Decision Engine behavior changed;
- no Reporting behavior changed;
- no Telegram behavior changed;
- no portfolio behavior changed;
- no ticker-category runtime logic implemented.