# Analyst Expectations Named Source Due Diligence Template

## 1. Status and Scope

This document is a documentation-only due diligence template for reviewing one named analyst expectations candidate source at a time.

It follows:

- `docs/research/analyst_expectations_source_policy_and_validation_design.md`;
- `docs/research/analyst_expectations_source_comparison_matrix.md`;
- `docs/research/analyst_expectations_named_source_shortlist_review.md`;
- backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy`.

This document defines a repeatable review format only.

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

The purpose of this template is to create a consistent due diligence format for reviewing named candidate analyst expectations sources before any collection, automation, or integration is proposed.

The template should be copied into a future documentation-only review document for each named source.

The template supports uniform assessment of:

- documented analyst expectation fields;
- field definitions;
- source freshness semantics;
- point-in-time support;
- licensing and terms-of-use status;
- automation restrictions;
- credentials or secrets requirements;
- auditability;
- unresolved questions;
- research-only conclusion;
- whether limited manual sample collection should be proposed later.

The template does not approve sample collection by itself.

## 3. Research-Only Boundary

Analyst expectations remain research-only.

Analyst expectations must not become buy/sell advice.

Source due diligence must not create or imply:

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

A source may pass documentation review and still remain unsuitable for collection, automation, historical validation, or runtime use.

Future Decision Engine use requires separate governance, separate approval, explicit design, tests, and audit controls.

## 4. Due Diligence Review Header

Use the following header for each future named-source review.

| Field | Value |
|---|---|
| Source name | TBD |
| Source category | TBD |
| Review document date | TBD |
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

## 5. Source Identity and Access Review

| Review question | Answer | Notes / evidence reference |
|---|---|---|
| What is the named source? | TBD | Use official source name. |
| What source category does it belong to? | TBD | Public platform, API provider, institutional provider, regulatory source, company source, or alternative estimate source. |
| Is the source public, paid, licensed, API-based, or institutional? | TBD | Identify access model only. |
| Is source documentation publicly available? | TBD | Link or reference may be recorded in a future review if allowed. |
| Is a login required for documentation review? | TBD | Do not create credentials under this template. |
| Is a paid account required for field confirmation? | TBD | Paid access requires later governance. |
| Does the source expose stable provider identifiers? | TBD | Important for auditability. |
| Does the source expose stable URLs or references? | TBD | Important for audit trails. |
| Does the source cover the project universe? | TBD | Do not collect ticker-level data unless later approved. |
| Does the source cover non-US securities if needed? | TBD | Record jurisdiction limitations. |

## 6. Analyst Expectations Field Review

Record only documented field availability. Do not collect ticker-level values.

| Candidate field | Documented availability | Definition clarity | Notes |
|---|---|---|---|
| `consensus_rating` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `analyst_count` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `buy_count` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `hold_count` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `sell_count` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `average_price_target` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `low_price_target` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `high_price_target` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `current_price_at_collection` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `implied_upside_pct` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `current_year_eps_estimate` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `next_year_eps_estimate` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `current_year_revenue_estimate` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `next_year_revenue_estimate` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `estimate_revision_direction` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `source_freshness_date` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `as_of_date` | Unknown / yes / no / partial | Unknown / clear / unclear / proprietary | TBD |
| `raw_source_value` storage | Unknown / permitted / not permitted / unclear | N/A | Requires terms review. |
| `normalized_value` derivation | Unknown / possible / not applicable / unclear | Unknown / clear / unclear / proprietary | Requires later formula governance if used. |

## 7. Consensus Definition Review

| Review question | Answer | Notes |
|---|---|---|
| Is consensus source-published? | TBD | If yes, preserve source definition. |
| Is consensus internally calculated by the provider? | TBD | Record methodology limits. |
| Are rating categories defined? | TBD | Example: buy, outperform, hold, underperform, sell. |
| Is a numeric consensus score exposed? | TBD | Numeric values must not become project scoring authority. |
| Is analyst count tied to the consensus value? | TBD | Minimum analyst count rules require later governance. |
| Are price-target consensus and rating consensus separate? | TBD | Must not blend signals unless later approved. |
| Are stale estimates excluded by the source? | TBD | Record source rule if documented. |
| Are missing estimates represented explicitly? | TBD | Missing must not be silently inferred. |
| Are conflicting analyst actions visible? | TBD | Conflicts require review policy. |

## 8. Point-in-Time and Historical Validation Review

| Review question | Answer | Notes |
|---|---|---|
| Does the source provide historical point-in-time analyst expectations? | TBD | Required for valid backtesting. |
| Does the source distinguish as-of date from collection/access date? | TBD | Required to prevent look-ahead bias. |
| Are historical revisions preserved? | TBD | Overwritten values are not sufficient for reconstruction. |
| Can the source reconstruct what was known on a past evaluation date? | TBD | Core historical validation requirement. |
| Are unavailable historical values preserved as unavailable? | TBD | Future-filled values are not allowed. |
| Are source corrections versioned? | TBD | Important for auditability. |
| Does the source include delisted or renamed securities? | TBD | Important for survivorship bias review. |
| Are historical price targets and ratings available separately? | TBD | Important for signal separation. |
| Are historical EPS/revenue estimates available by fiscal period? | TBD | Important for estimate validation. |
| Is current-page content clearly unsuitable as historical truth? | TBD | Must be explicit if no point-in-time support exists. |

## 9. Licensing, Terms, and Usage Review

This section must be completed before any collection or automation is proposed.

| Review question | Answer | Notes |
|---|---|---|
| Have terms of use been reviewed? | No / yes / partial | Required before collection. |
| Is manual research use allowed? | Unknown / yes / no / restricted | Must be verified. |
| Is storage of raw values allowed? | Unknown / yes / no / restricted | Must be verified. |
| Is storage of normalized values allowed? | Unknown / yes / no / restricted | Must be verified. |
| Is redistribution prohibited? | Unknown / yes / no / restricted | Must be verified. |
| Is API use allowed? | Unknown / yes / no / restricted | Requires provider governance. |
| Is scraping prohibited? | Unknown / yes / no / restricted | Scraping is not authorized by this template. |
| Are exports allowed? | Unknown / yes / no / restricted | Must be verified. |
| Are rate limits documented? | Unknown / yes / no / not applicable | Required before automation. |
| Are attribution requirements documented? | Unknown / yes / no / not applicable | Required before use. |
| Is a commercial license required? | Unknown / yes / no / unclear | Must be assessed. |
| Is legal or formal license review required? | Yes / no / unclear | Default to yes for paid/API providers. |

## 10. Automation and Integration Review

No automation or integration is approved by this template.

| Review question | Answer | Notes |
|---|---|---|
| Would API credentials be required? | TBD | Requires secrets governance. |
| Would a paid subscription be required? | TBD | Requires provider approval. |
| Would rate-limit handling be required? | TBD | Requires design and tests. |
| Would caching be required? | TBD | Requires storage and freshness policy. |
| Would audit logs be required? | Yes | Required for future automation. |
| Would failure handling be required? | Yes | Required for future automation. |
| Would CI checks be required? | Yes | Required for future implementation. |
| Would runtime storage be required? | TBD | Requires separate data contract. |
| Would generated files be created? | TBD | Not approved here. |
| Would the source affect scanner output? | No | Not authorized. |
| Would the source affect Decision Engine output? | No | Not authorized. |
| Would the source affect Reporting or Telegram output? | No | Not authorized. |

## 11. Data Quality and Exception Review

| Review area | Required assessment | Notes |
|---|---|---|
| Missing data | Define whether missing fields are explicit, silent, or unclear. | TBD |
| Stale data | Define how stale values are identified. | TBD |
| Conflicting values | Define whether conflicts are visible and how they would be reviewed. | TBD |
| Partial coverage | Define whether some fields are unavailable for some tickers. | TBD |
| Low analyst count | Define whether analyst count is sufficient for interpretation. | TBD |
| Currency issues | Define whether targets and estimates include currency metadata. | TBD |
| Fiscal-period ambiguity | Define whether EPS/revenue estimate periods are clear. | TBD |
| Methodology opacity | Define whether proprietary methods limit interpretation. | TBD |
| Regional inconsistency | Define whether coverage differs by market or exchange. | TBD |
| Survivorship bias | Define whether historical universe issues are visible. | TBD |

## 12. Human Review and Sampling Proposal

The operator should not review every ticker.

A future source-specific review should propose whether limited sampling is needed.

| Review question | Answer | Notes |
|---|---|---|
| Is limited manual sample collection proposed? | No / yes, later governance required | This template does not approve collection. |
| What would the sample size be? | TBD | Must be small and justified. |
| What would the sample purpose be? | TBD | Example: field verification, coverage check, freshness check. |
| Would the sample include portfolio tickers? | TBD | Requires explicit scope. |
| Would the sample include scanner tickers? | TBD | Requires explicit scope. |
| Would the sample be stored? | TBD | Storage requires approval. |
| Would sample values influence decisions? | No | Must remain research-only. |
| What exceptions would require human review? | TBD | Missing, stale, conflicting, unclear, restricted, or low-confidence data. |

## 13. Source-Specific Conclusion Template

Use the following conclusion block in every future source-specific review.

```text
Source-specific due diligence conclusion:
- Source reviewed: [SOURCE NAME]
- Review type: Documentation-only
- Runtime data collected: No
- API calls made: No
- Scraping performed: No
- Credentials or secrets used: No
- Source approved: No
- Runtime use approved: No
- Decision Engine use approved: No
- Reporting or Telegram use approved: No
- Limited manual sample collection proposed: [No / Yes, requires later governance]
- Main unresolved issues: [LIST]
- Recommended next step: [NEXT STEP]
```

## 14. Not Approved by This Template

This template does not approve:

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

## 15. Backlog Impact Assessment

Existing backlog item `BL-0018` remains sufficient.

This document executes the recommended documentation-only named-source due diligence template step already implied by the named-source shortlist review.

It does not identify additional deferred work beyond the existing governed analyst expectations research path.

Backlog impact assessment:
- No new backlog items identified.

## 16. Recommended Next Step

The recommended next step is to apply this template to one named source in a documentation-only source-specific review.

A suitable first review candidate should be a source where public documentation can be assessed without:

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

The first source-specific review should remain research-only and should conclude whether limited manual sample collection should be proposed later under separate governance.
