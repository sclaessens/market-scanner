# Analyst Expectations Named Source Shortlist Review

## 1. Status and Scope

This document is a documentation-only shortlist review of named candidate sources for analyst expectations research.

It follows:

- `docs/research/analyst_expectations_source_policy_and_validation_design.md`;
- `docs/research/analyst_expectations_source_comparison_matrix.md`;
- backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy`.

This document names candidate sources for future governance review. Naming a source does not approve that source.

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

No source is approved by this document.

No data collection is authorized by this document.

## 2. Purpose

The purpose of this shortlist is to identify named sources that may be worth later documentation review before any analyst expectations data is collected.

The shortlist prepares the next governance discussion by separating possible sources into categories:

- current public market-data pages;
- company and regulatory context sources;
- paid or API-based market-data providers;
- historical point-in-time estimate providers;
- sources that may be useful only as supporting context.

This document does not evaluate live data, current field availability, current pricing, current licensing, or current terms of use.

Any future review of a named source must verify current documentation, licensing, field definitions, and point-in-time capability at the time of review.

## 3. Research-Only Boundary

Analyst expectations remain research-only.

Analyst expectations must not become buy/sell advice.

Named-source review must not create or imply:

- ranking authority;
- scoring authority;
- allocation authority;
- tradeability;
- urgency;
- conviction;
- eligibility;
- hidden filtering;
- Reporting recommendations;
- Telegram recommendations;
- Decision Engine bypass.

A source can be promising for research while still being unsuitable for runtime use, automation, historical validation, or Decision Engine integration.

Future Decision Engine use requires separate governance, separate approval, explicit design, tests, and audit controls.

## 4. Shortlist Governance Rules

The following rules apply to this shortlist:

1. Named sources are candidates only.
2. No provider is approved.
3. No source field is accepted as authoritative.
4. No source is approved for collection.
5. No source is approved for automation.
6. No source is approved for backtesting.
7. No source is approved for Decision Engine use.
8. No source is approved for Reporting or Telegram use.
9. Public pages must not be treated as point-in-time historical truth.
10. Paid/API providers require separate provider governance before any access.
11. Licensing and terms of use must be reviewed before storing or automating data.
12. Historical validation requires true point-in-time support or preserved historical snapshots.

## 5. Named Candidate Source Shortlist

The following sources are named candidates for later documentation review.

The notes below are planning hypotheses only and must be verified before any source is used.

| Source | Category | Candidate use | Initial governance view | Main risks to review | Current approval status |
|---|---|---|---|---|---|
| Company investor relations pages | Official company source | Company guidance, earnings materials, investor presentations, management outlook, and official context. | Useful supporting context, but not a direct independent analyst consensus source. | Inconsistent structure, limited comparability, no analyst consensus, changing URLs, selective guidance. | Candidate only; not approved. |
| SEC EDGAR | Regulatory filing source | US regulatory filings, reported results, historical disclosures, risk factors, and management discussion. | Useful for official disclosure context and historical facts, not direct analyst expectations. | Jurisdiction limitation, no analyst ratings, no price-target consensus, not enough for analyst expectations alone. | Candidate only; not approved. |
| Equivalent non-US regulatory filing portals | Regulatory filing source | Official filings for non-US securities where SEC EDGAR does not apply. | Useful for official disclosure context if the project covers relevant markets. | Jurisdiction fragmentation, inconsistent formats, language differences, limited analyst expectations fields. | Candidate only; not approved. |
| Nasdaq market profile pages | Exchange or market-profile source | Ticker profile, market context, and possibly analyst-related displayed fields. | Possible current snapshot review candidate if definitions and terms are suitable. | Terms of use, field definitions, point-in-time weakness, URL stability, coverage variation. | Candidate only; not approved. |
| Yahoo Finance | Public market-data platform | Public-facing analyst rating, price target, estimate, and ticker-summary research candidate. | Possible current snapshot comparison source, not sufficient for point-in-time validation without historical support. | Terms of use, field definitions, API ambiguity, scraping restrictions, point-in-time limitations, coverage differences. | Candidate only; not approved. |
| MarketWatch | Public market-data platform | Public-facing analyst estimates, ratings, price targets, and quote/profile context candidate. | Possible documentation-review candidate for current analyst expectations visibility. | Terms of use, field definitions, freshness, historical support, page stability, redistribution limits. | Candidate only; not approved. |
| Investing.com | Public market-data platform | Public-facing analyst estimates, financial calendar, market profile, or consensus-related candidate. | Possible current snapshot review candidate only after terms and definitions review. | Terms of use, scraping restrictions, regional coverage, field definitions, point-in-time limitations. | Candidate only; not approved. |
| TipRanks | Analyst-ratings platform | Analyst ratings, analyst counts, price targets, and consensus-style presentation candidate. | Potentially relevant for analyst expectation fields if licensing and definitions are acceptable. | Paywall/licensing, terms of use, methodology opacity, historical availability, redistribution restrictions. | Candidate only; not approved. |
| Koyfin | Market-data platform | Analyst estimates, price targets, fundamentals, and market-data workspace candidate. | Potential paid platform candidate for structured research review. | Licensing, export rights, automation rights, historical access, cost, credentials. | Candidate only; not approved. |
| Finviz | Public market-data platform | Snapshot-style analyst recommendation and target-price context candidate. | Possible lightweight current snapshot candidate, not enough for historical validation by itself. | Terms of use, field definitions, no clear point-in-time history, scraping restrictions, limited detail. | Candidate only; not approved. |
| Alpha Vantage | API-based market-data provider | API-assisted fundamentals and market-data candidate; may offer some analyst-related or estimate-adjacent data depending on product scope. | Candidate only if provider documentation confirms needed analyst expectations fields. | API terms, rate limits, field availability, historical depth, licensing, credentials/secrets. | Candidate only; not approved. |
| Financial Modeling Prep | API-based market-data provider | API-assisted analyst estimates, price targets, ratings, financial statements, and fundamentals candidate if documented. | Candidate for structured source review and possible later provider governance. | Licensing, subscription tier, rate limits, field definitions, point-in-time support, redistribution, credentials. | Candidate only; not approved. |
| Finnhub | API-based market-data provider | API-assisted analyst recommendations, price targets, estimates, and company fundamentals candidate if documented. | Candidate for structured source review and possible later provider governance. | Licensing, subscription tier, rate limits, field definitions, historical support, credentials/secrets. | Candidate only; not approved. |
| Polygon.io | API-based market-data provider | Market data, reference data, and possible financial-data provider candidate depending on available products. | Candidate only if analyst expectations fields and history are documented. | Product coverage, analyst field availability, licensing, cost, historical support, credentials. | Candidate only; not approved. |
| IEX Cloud or equivalent API data platforms | API-based market-data provider | API-assisted fundamentals, estimates, or market-data candidate depending on current product availability. | Candidate only after confirming current platform availability, data rights, and analyst field support. | Product changes, availability, licensing, field definitions, historical support, credentials. | Candidate only; not approved. |
| FactSet | Licensed institutional provider | Institutional analyst estimates, consensus, financials, and historical datasets candidate. | Strong potential fit for point-in-time and institutional data if available and licensed. | Cost, licensing, access, credentials, redistribution, implementation complexity, governance burden. | Candidate only; not approved. |
| LSEG / Refinitiv | Licensed institutional provider | Institutional estimates, analyst consensus, ratings, financial data, and historical datasets candidate. | Strong potential fit for institutional analyst expectations if licensed and point-in-time capable. | Cost, licensing, access model, data redistribution, credentials, historical contract details. | Candidate only; not approved. |
| Bloomberg | Licensed institutional provider | Institutional estimates, consensus, company data, analyst context, and historical market-data candidate. | Strong research candidate where access exists, but integration and rights require strict governance. | Cost, terminal/API licensing, redistribution limits, automation limits, credentials, point-in-time controls. | Candidate only; not approved. |
| S&P Capital IQ | Licensed institutional provider | Institutional estimates, company financials, consensus, and historical data candidate. | Strong potential fit for structured historical validation if licensed. | Cost, licensing, access, redistribution, credentials, historical snapshot semantics. | Candidate only; not approved. |
| Visible Alpha | Historical estimates provider | Analyst models, consensus estimates, and institutional estimate history candidate. | Potentially strong point-in-time analyst estimates candidate if licensed and suitable. | Cost, licensing, coverage, access model, redistribution, implementation complexity. | Candidate only; not approved. |
| Estimize | Alternative estimates platform | Crowd-sourced or alternative earnings/revenue estimate candidate. | Potential supplementary estimates source, not a direct substitute for sell-side analyst consensus. | Methodology differences, comparability, historical access, licensing, field definitions. | Candidate only; not approved. |
| Zacks | Estimates and ratings provider | Analyst estimates, revisions, and rating-style research candidate. | Potential source for estimate revision research if definitions and licensing are suitable. | Proprietary methodology, licensing, field transparency, historical access, redistribution. | Candidate only; not approved. |
| Morningstar | Research and data provider | Analyst research, fair value, ratings, and fundamentals context candidate. | Useful supporting context, but proprietary ratings may not equal analyst consensus. | Licensing, methodology, redistribution, field definitions, historical availability. | Candidate only; not approved. |

## 6. Preliminary Shortlist Grouping

This grouping is for research planning only.

It does not rank securities, sources, or providers for operational use.

| Planning group | Sources | Reason for grouping | Governance implication |
|---|---|---|---|
| Official context sources | Company investor relations pages; SEC EDGAR; equivalent non-US regulatory portals | Useful for official context and audit-backed company facts. | Supporting context only; not sufficient analyst consensus sources. |
| Public current-snapshot candidates | Nasdaq market profile pages; Yahoo Finance; MarketWatch; Investing.com; Finviz | May expose visible analyst-related fields for current research comparison. | Must not be treated as historical truth without point-in-time support. |
| Analyst-ratings focused platforms | TipRanks; Zacks; Morningstar | May provide rating, price target, or revision-style views. | Requires methodology, licensing, and historical availability review. |
| API-assisted candidates | Alpha Vantage; Financial Modeling Prep; Finnhub; Polygon.io; IEX Cloud or equivalent API platforms | May support structured access if product fields match project needs. | Requires provider governance, credentials/secrets, rate limits, and tests before any use. |
| Institutional licensed providers | FactSet; LSEG / Refinitiv; Bloomberg; S&P Capital IQ; Visible Alpha | May offer stronger field definitions, historical data, and point-in-time support. | Likely high governance burden due to cost, licensing, redistribution, and access controls. |
| Alternative estimate source | Estimize | May offer supplementary crowd or alternative estimates. | Must remain separate from sell-side analyst consensus unless explicitly defined. |

## 7. Named Source Review Checklist

A future source-by-source documentation review should answer these questions for each named candidate.

| Review question | Required answer before collection |
|---|---|
| What exact analyst expectation fields are documented? | Must be explicit. |
| Are consensus ratings source-published or internally derived by the provider? | Must be documented. |
| Are analyst counts available for every consensus value? | Must be verified. |
| Are buy, hold, and sell components available? | Must be verified. |
| Are price targets available as average, low, and high values? | Must be verified. |
| Are EPS and revenue estimates available by fiscal period? | Must be verified. |
| Is source freshness shown? | Must be verified. |
| Is historical point-in-time access available? | Must be verified before backtesting. |
| Are revised values versioned or overwritten? | Must be understood. |
| Are terms of use compatible with storage and research? | Must be reviewed. |
| Are API calls, exports, or automation allowed? | Must be reviewed before any integration. |
| Are credentials or secrets required? | Must be governed before access. |
| Are stable URLs or provider identifiers available? | Must be verified for auditability. |
| Is field coverage consistent across tickers and regions? | Must be tested only after collection approval. |
| How are missing, stale, and conflicting values represented? | Must be defined. |
| Does use of this source risk hidden ranking, scoring, or Decision Engine bypass? | Must be explicitly assessed. |

## 8. Shortlist Suitability Notes

### 8.1 Current snapshot research

Public market-data platforms may be useful for understanding which analyst expectation fields are visible and how source definitions differ.

They are not automatically suitable for historical validation because current pages generally cannot prove what was known on a historical date unless point-in-time support exists.

### 8.2 Historical validation

Historical validation requires sources that can reconstruct analyst expectations as they existed on historical evaluation dates.

Institutional or dedicated historical estimate providers may be better candidates for this requirement, but they require separate licensing and provider governance.

### 8.3 API-assisted research

API-assisted sources may reduce manual effort and improve structure, but they introduce governance requirements around provider choice, credentials, secrets, rate limits, caching, audit logs, tests, failure handling, and storage location.

No API-assisted source is approved by this document.

### 8.4 Company and regulatory context

Company and regulatory sources may support context, verification, and official disclosure history.

They should not be treated as independent analyst consensus sources unless they explicitly contain third-party analyst expectation data and the definition is governed.

## 9. Review Priority Proposal

The following review sequence is proposed for documentation-only follow-up.

| Priority | Candidate review area | Rationale | Allowed next action |
|---|---|---|---|
| 1 | Public current-snapshot candidates | They can clarify visible field definitions and coverage expectations without committing to provider integration. | Documentation review of terms, field labels, methodology pages, and point-in-time limitations only. |
| 2 | API-assisted candidates | They may offer structured fields if documentation confirms analyst expectations support. | Documentation review of provider docs, rate limits, field lists, and licensing model only. |
| 3 | Historical point-in-time and institutional providers | They are most relevant for proper backtesting but likely have higher cost and governance overhead. | Documentation review of point-in-time capability and licensing prerequisites only. |
| 4 | Official context sources | They are useful for context and audit support but not direct analyst consensus sources. | Documentation review of how context references could support future research notes only. |

This priority proposal does not approve source use or data collection.

## 10. Not Approved by This Shortlist

This shortlist does not approve:

- any source or provider;
- runtime data collection;
- manual ticker-by-ticker analyst data collection;
- scraping;
- provider/API calls;
- credentials or secrets;
- automated ingestion;
- CSV creation;
- historical backtesting code;
- source-derived scoring;
- source-derived ranking;
- source-derived tradeability;
- source-derived conviction;
- source-derived urgency;
- source-derived eligibility;
- hidden filtering;
- Decision Engine integration;
- Reporting recommendations;
- Telegram recommendations.

## 11. Backlog Impact Assessment

Existing backlog item `BL-0018` remains sufficient.

This document executes the recommended documentation-only named-source shortlist review already implied by the analyst expectations source comparison matrix.

It does not identify additional deferred work beyond the existing governed analyst expectations research path.

Backlog impact assessment:
- No new backlog items identified.

## 12. Recommended Next Step

The recommended next step is a documentation-only named-source due-diligence template.

That template should define the exact review form to use for one source at a time before any data collection is allowed.

The template should capture:

- source name;
- source category;
- documented fields;
- field definitions;
- source freshness semantics;
- point-in-time support;
- licensing and terms-of-use review status;
- automation restrictions;
- credential or secrets requirements;
- auditability notes;
- missing or unclear items;
- research-only conclusion;
- whether limited manual sample collection should be proposed later.

The next step must remain documentation-only and must not collect source data, call APIs, scrape websites, create CSVs, or alter runtime behavior.
