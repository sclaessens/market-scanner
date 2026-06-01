# Financial Modeling Prep Analyst Expectations Due Diligence Review

## 1. Status and Scope

This document is a documentation-only source-specific due diligence review for Financial Modeling Prep as a named analyst expectations candidate source.

It applies the template defined in `docs/research/analyst_expectations_named_source_due_diligence_template.md`.

It also follows:

- `docs/research/analyst_expectations_source_policy_and_validation_design.md`;
- `docs/research/analyst_expectations_source_comparison_matrix.md`;
- `docs/research/analyst_expectations_named_source_shortlist_review.md`;
- backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy`.

This document reviews public documentation only.

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

Financial Modeling Prep is not approved by this document.

No data collection is authorized by this document.

## 2. Source-Specific Review Header

| Field | Value |
|---|---|
| Source name | Financial Modeling Prep |
| Source category | API-based market-data provider |
| Review document date | 2026-05-21 |
| Reviewer role | PM / Functional Analyst / Technical Analyst / Research / Governance |
| Review type | Documentation-only |
| Source approval requested | No |
| Runtime collection requested | No |
| API access requested | No |
| Scraping requested | No |
| Credentials or secrets requested | No |
| Decision Engine integration requested | No |
| Reporting or Telegram integration requested | No |
| Related backlog item | `BL-0018` |
| Related policy document | `docs/research/analyst_expectations_source_policy_and_validation_design.md` |

## 3. Documentation Reviewed

Public Financial Modeling Prep developer documentation was reviewed at a documentation level only.

The review did not call any endpoint, did not use an API key, did not create credentials, did not collect ticker data, and did not store source values.

Documentation observations:

- The developer documentation lists an `Analyst Estimates & Price Target` dataset category.
- The developer documentation states that API requests require an API key.
- The documented analyst section includes a Financial Estimates API for projected revenue, EPS, and related analyst financial metrics.
- The documented analyst section includes Price Target Summary and Price Target Consensus APIs.
- The documented analyst section includes Stock Grades, Historical Stock Grades, and Grades Summary APIs.
- The documented bulk section includes Price Target Summary Bulk and Upgrades Downgrades Consensus Bulk APIs.

These observations confirm potential documented field relevance only. They do not approve source use.

## 4. Source Identity and Access Review

| Review question | Answer | Notes / evidence reference |
|---|---|---|
| What is the named source? | Financial Modeling Prep | Public developer documentation reviewed. |
| What source category does it belong to? | API-based market-data provider | Candidate provider for structured analyst expectations fields. |
| Is the source public, paid, licensed, API-based, or institutional? | API-based provider with documented API-key authorization | Licensing, pricing, plan tier, and permitted use require separate review. |
| Is source documentation publicly available? | Yes | Public developer documentation was reviewed. |
| Is a login required for documentation review? | Not required for the reviewed public documentation page | This does not imply data access rights. |
| Is a paid account required for field confirmation? | Unknown | Plan-level access and endpoint availability require later provider review. |
| Does the source expose stable provider identifiers? | Unknown | Must be verified before audit design. |
| Does the source expose stable URLs or references? | Partial | Documentation exposes endpoint paths, but runtime audit references require later design. |
| Does the source cover the project universe? | Unknown | No ticker-level coverage test was performed. |
| Does the source cover non-US securities if needed? | Unknown / partial | Documentation includes global indicators in some sections, but ticker-level coverage requires later review. |

## 5. Analyst Expectations Field Review

This section records documented field relevance only. No ticker-level values were collected.

| Candidate field | Documented availability | Definition clarity | Notes |
|---|---|---|---|
| `consensus_rating` | Partial | Unclear / provider-defined | Grades Summary and upgrades/downgrades consensus may be relevant, but exact mapping requires source-specific methodology review. |
| `analyst_count` | Unknown / partial | Unclear | Grades Summary mentions counts by rating category, but analyst-count semantics require later verification. |
| `buy_count` | Partial | Unclear / provider-defined | Grades Summary documentation mentions strong buy and buy categories. Exact field names and semantics require later review. |
| `hold_count` | Partial | Unclear / provider-defined | Grades Summary documentation mentions hold category. Exact field names and semantics require later review. |
| `sell_count` | Partial | Unclear / provider-defined | Grades Summary documentation mentions sell and strong sell categories. Exact field names and semantics require later review. |
| `average_price_target` | Yes / partial | Unclear / provider-defined | Price Target Summary documentation refers to average price targets. Exact timeframes and field names require later review. |
| `low_price_target` | Yes / partial | Unclear / provider-defined | Price Target Consensus documentation refers to low price target. Exact semantics require later review. |
| `high_price_target` | Yes / partial | Unclear / provider-defined | Price Target Consensus documentation refers to high price target. Exact semantics require later review. |
| `current_price_at_collection` | Unknown | Unclear | May require a separate approved price source or FMP quote endpoint; not reviewed here. |
| `implied_upside_pct` | Unknown | Unclear | If calculated internally, formula governance is required. |
| `current_year_eps_estimate` | Partial | Unclear / provider-defined | Financial Estimates API is documented for analyst EPS and revenue estimates. Fiscal-period mapping must be verified. |
| `next_year_eps_estimate` | Partial | Unclear / provider-defined | Financial Estimates API period handling must be reviewed before use. |
| `current_year_revenue_estimate` | Partial | Unclear / provider-defined | Financial Estimates API documentation mentions revenue estimates. Fiscal-period mapping must be verified. |
| `next_year_revenue_estimate` | Partial | Unclear / provider-defined | Financial Estimates API period handling must be reviewed before use. |
| `estimate_revision_direction` | Unknown / partial | Unclear | Upgrades/downgrades and historical grades may support direction-like analysis, but estimate revision semantics are not approved. |
| `source_freshness_date` | Unknown | Unclear | Must be verified from endpoint metadata or documentation before any use. |
| `as_of_date` | Unknown / partial | Unclear | Historical endpoints may include dates, but point-in-time meaning must be verified. |
| `raw_source_value` storage | Unknown | N/A | Terms and licensing review required. |
| `normalized_value` derivation | Possible later | Requires formula governance | Normalization must not create project scoring/ranking authority without approval. |

## 6. Consensus Definition Review

| Review question | Answer | Notes |
|---|---|---|
| Is consensus source-published? | Partial / likely for price targets and grades summary, pending verification | Price Target Consensus and Grades Summary APIs are documented, but exact consensus definitions require methodology review. |
| Is consensus internally calculated by the provider? | Unknown | Must be verified before use. |
| Are rating categories defined? | Partial | Documentation mentions strong buy, buy, hold, sell, and strong sell in grades summary context. |
| Is a numeric consensus score exposed? | Unknown | Numeric values must not become project scoring authority. |
| Is analyst count tied to the consensus value? | Unknown / partial | Category counts may exist, but analyst count semantics must be verified. |
| Are price-target consensus and rating consensus separate? | Likely separate, pending verification | Price target and grades endpoints appear separate in documentation. |
| Are stale estimates excluded by the source? | Unknown | Must be verified before historical or current research use. |
| Are missing estimates represented explicitly? | Unknown | Missing values must not be silently inferred. |
| Are conflicting analyst actions visible? | Partial | Stock grades and historical grades may expose actions, but conflict semantics require review. |

## 7. Point-in-Time and Historical Validation Review

| Review question | Answer | Notes |
|---|---|---|
| Does the source provide historical point-in-time analyst expectations? | Partial / unclear | Historical Ratings and Historical Stock Grades are documented, but true point-in-time analyst expectation reconstruction is not confirmed. |
| Does the source distinguish as-of date from collection/access date? | Unknown | Required before backtesting. |
| Are historical revisions preserved? | Unknown | Required before look-ahead-safe validation. |
| Can the source reconstruct what was known on a past evaluation date? | Unknown | Must be confirmed before historical validation. |
| Are unavailable historical values preserved as unavailable? | Unknown | Future-filled values are not allowed. |
| Are source corrections versioned? | Unknown | Important for auditability. |
| Does the source include delisted or renamed securities? | Unknown / possible for broader FMP docs | Must be reviewed separately for survivorship-bias control. |
| Are historical price targets and ratings available separately? | Partial / unclear | Historical grades are documented; historical price-target point-in-time support requires further review. |
| Are historical EPS/revenue estimates available by fiscal period? | Unknown | Analyst estimates endpoint may provide dated rows, but point-in-time semantics require verification. |
| Is current-page content clearly unsuitable as historical truth? | Yes | Current documentation or current endpoint outputs must not be treated as historical truth without point-in-time proof. |

## 8. Licensing, Terms, and Usage Review

This section must be completed before any collection or automation is proposed.

| Review question | Answer | Notes |
|---|---|---|
| Have terms of use been reviewed? | Partial / not sufficient | Public developer docs were reviewed, but full licensing and terms review remains required. |
| Is manual research use allowed? | Unknown | Must be verified. |
| Is storage of raw values allowed? | Unknown | Must be verified. |
| Is storage of normalized values allowed? | Unknown | Must be verified. |
| Is redistribution prohibited? | Unknown | Must be verified. |
| Is API use allowed? | Potentially yes with API key, not approved here | API authorization is documented, but project use requires provider governance. |
| Is scraping prohibited? | Unknown | Scraping is not authorized. |
| Are exports allowed? | Unknown | Must be verified. |
| Are rate limits documented? | Unknown / plan-dependent | Must be reviewed before automation. |
| Are attribution requirements documented? | Unknown | Must be reviewed. |
| Is a commercial license required? | Unknown / plan-dependent | Must be assessed. |
| Is legal or formal license review required? | Yes | Required before any collection, storage, or automation proposal. |

## 9. Automation and Integration Review

No automation or integration is approved by this review.

| Review question | Answer | Notes |
|---|---|---|
| Would API credentials be required? | Yes | FMP documentation states API requests require an API key. |
| Would a paid subscription be required? | Unknown / likely plan-dependent | Must be reviewed before any proposal. |
| Would rate-limit handling be required? | Yes if automated | Requires design and tests. |
| Would caching be required? | Yes if automated | Requires storage and freshness policy. |
| Would audit logs be required? | Yes | Required for future automation. |
| Would failure handling be required? | Yes | Required for future automation. |
| Would CI checks be required? | Yes if implemented | Required for future implementation. |
| Would runtime storage be required? | Yes if collected | Requires separate data contract. |
| Would generated files be created? | Not approved | No generated artifacts are authorized. |
| Would the source affect scanner output? | No | Not authorized. |
| Would the source affect Decision Engine output? | No | Not authorized. |
| Would the source affect Reporting or Telegram output? | No | Not authorized. |

## 10. Data Quality and Exception Review

| Review area | Required assessment | Notes |
|---|---|---|
| Missing data | Must determine whether missing fields are explicit or silent. | Unknown. |
| Stale data | Must determine whether source freshness dates exist and are usable. | Unknown. |
| Conflicting values | Must determine whether grades, price targets, and estimates can conflict and how to flag them. | Unknown. |
| Partial coverage | Must determine coverage by ticker, market, and endpoint. | Unknown. |
| Low analyst count | Must determine whether analyst counts are exposed and whether minimum count rules are possible. | Unknown / partial. |
| Currency issues | Must determine whether price targets and revenue estimates include currency metadata. | Unknown. |
| Fiscal-period ambiguity | Must determine whether annual and quarterly estimate periods map cleanly to project fields. | Unknown. |
| Methodology opacity | Must determine whether FMP definitions are transparent enough for governance. | Unknown / partial. |
| Regional inconsistency | Must determine whether endpoint support differs by region or exchange. | Unknown. |
| Survivorship bias | Must determine whether historical validation can include delisted or renamed securities. | Unknown. |

## 11. Human Review and Sampling Proposal

The operator should not review every ticker.

| Review question | Answer | Notes |
|---|---|---|
| Is limited manual sample collection proposed? | Yes, later governance required | This review does not approve collection. |
| What would the sample size be? | Proposed later: small sample only | Candidate sample could include a few portfolio/scanner tickers after governance approval. |
| What would the sample purpose be? | Field verification only | Confirm endpoint fields, freshness, missing data, and consistency. |
| Would the sample include portfolio tickers? | TBD | Requires explicit scope. |
| Would the sample include scanner tickers? | TBD | Requires explicit scope. |
| Would the sample be stored? | TBD | Storage requires approval. |
| Would sample values influence decisions? | No | Must remain research-only. |
| What exceptions would require human review? | Missing, stale, conflicting, unclear, restricted, low analyst count, unclear dates, or licensing constraints | Exception queue only. |

## 12. Source-Specific Due Diligence Conclusion

Source-specific due diligence conclusion:
- Source reviewed: Financial Modeling Prep
- Review type: Documentation-only
- Runtime data collected: No
- API calls made: No
- Scraping performed: No
- Credentials or secrets used: No
- Source approved: No
- Runtime use approved: No
- Decision Engine use approved: No
- Reporting or Telegram use approved: No
- Limited manual sample collection proposed: Yes, requires later governance
- Main unresolved issues: licensing and terms of use; plan-level endpoint access; field definitions; analyst count semantics; source freshness; point-in-time support; historical revision handling; raw-value storage rights; rate limits; credential/secrets handling; auditability.
- Recommended next step: Create a documentation-only limited manual sample collection proposal for FMP that defines sample size, candidate fields, no-storage versus temporary-review handling, approval criteria, and explicit research-only boundaries before any values are collected.

## 13. Not Approved by This Review

This review does not approve:

- Financial Modeling Prep as a project source;
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

## 14. Backlog Impact Assessment

Existing backlog item `BL-0018` remains sufficient.

This document executes the recommended documentation-only source-specific review step already implied by the named-source due diligence template.

It does not identify additional deferred work beyond the existing governed analyst expectations research path.

Backlog impact assessment:
- No new backlog items identified.

## 15. Recommended Next Step

The recommended next step is a documentation-only limited manual sample collection proposal for Financial Modeling Prep.

That proposal must define:

- whether sample collection is allowed;
- the exact maximum sample size;
- the exact candidate fields to inspect;
- whether source values may be stored or only visually reviewed;
- whether screenshots, raw values, or normalized values are prohibited;
- required licensing and terms-of-use checks;
- API-key and secrets handling requirements;
- no-runtime-change boundaries;
- no-Decision-Engine-use boundaries;
- acceptance criteria for deciding whether FMP remains a candidate source.

No sample collection is authorized until that proposal is separately approved.
