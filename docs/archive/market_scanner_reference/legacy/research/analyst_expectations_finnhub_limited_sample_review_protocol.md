# Finnhub Limited Sample Review Protocol

## 1. Status and Scope

This document is a documentation-only protocol for a possible future limited sample review of Finnhub analyst expectations data.

It follows:

- `docs/research/analyst_expectations_source_policy_and_validation_design.md`;
- `docs/research/analyst_expectations_source_comparison_matrix.md`;
- `docs/research/analyst_expectations_named_source_shortlist_review.md`;
- `docs/research/analyst_expectations_named_source_due_diligence_template.md`;
- `docs/research/analyst_expectations_fmp_execution_readiness_reconciliation.md`;
- `docs/research/analyst_expectations_finnhub_combined_source_review.md`;
- `docs/research/analyst_expectations_finnhub_limited_manual_sample_collection_proposal.md`;
- `docs/research/analyst_expectations_finnhub_sample_proposal_governance_note.md`;
- backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy`.

This document defines the future review protocol only.

This document does not implement code, tests, CSV files, generated artifacts, reports, workflows, provider integration, provider/API calls, scraping, account creation, API-key creation, credentials or secrets, runtime orchestration, daily ingestion, backtesting code, Reporting changes, Telegram changes, scanner changes, Decision Engine changes, portfolio files, watchlist files, fundamentals files, or runtime behavior changes.

No sprint is closed or certified complete by this document.

Finnhub is not approved as a project source by this document.

No sample values are collected, viewed, copied, stored, screenshotted, summarized, or normalized by this document.

No API access is authorized by this document.

## 2. Protocol Purpose

The purpose of this protocol is to define exactly how a later limited Finnhub sample review may be performed if separately approved.

The protocol defines:

- the maximum ticker sample;
- the proposed sample-selection logic;
- the endpoint families that may be inspected;
- the qualitative observations that may be recorded;
- strict storage prohibitions;
- account, API-key, and secrets rules;
- rejection and revision triggers;
- research-only boundaries.

This protocol is a control document. It is not an execution document.

## 3. Approval Boundary

This protocol is approved for documentation design only.

Before any sample review occurs, a separate execution approval note must confirm:

- terms-of-use review completion;
- manual research permission;
- account and API-key handling approach if access is required;
- secrets-handling approach;
- endpoint access permission;
- final ticker list;
- final endpoint family list;
- final qualitative-observation checklist;
- storage prohibitions;
- reviewer role;
- review date.

Without that separate execution approval note, no values may be viewed, collected, copied, screenshotted, stored, summarized, normalized, or used.

## 4. Research-Only Boundary

Analyst expectations remain research-only.

Analyst expectations must not become buy/sell advice.

The Finnhub limited sample review must not create or imply:

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

The review may only assess field presence, field absence, date availability, freshness clarity, permission constraints, plan constraints, and source-quality questions.

The review must not conclude whether any ticker is attractive, actionable, tradable, undervalued, overvalued, or portfolio-relevant.

Future Decision Engine use requires separate governance, separate approval, explicit design, tests, and audit controls.

## 5. Proposed Maximum Sample

A later approved review must not exceed:

- maximum 5 tickers;
- maximum 3 endpoint families;
- maximum 1 review session;
- maximum 1 source provider: Finnhub;
- no repeated sampling without separate governance approval.

The sample must remain small enough to avoid becoming an informal dataset.

The sample exists only to test whether documented fields appear usable for future governance review.

## 6. Proposed Ticker Sample Protocol

The exact ticker list must be finalized in a later execution approval note.

The protocol permits the following sample structure:

| Slot | Ticker type | Purpose | Selection rule |
|---|---|---|---|
| 1 | Current portfolio ticker | Test whether Finnhub covers an owned security. | Must be selected before viewing Finnhub values. |
| 2 | Current portfolio ticker or recent portfolio-review ticker | Test coverage for a second known project security. | Must be selected before viewing Finnhub values. |
| 3 | Recent scanner ticker | Test coverage for a current opportunity-universe security. | Must be selected before viewing Finnhub values. |
| 4 | Recent scanner ticker from a different sector if possible | Test basic cross-sector consistency. | Must be selected before viewing Finnhub values. |
| 5 | Non-US, cross-listed, or expected partial-coverage ticker if relevant | Test coverage limitations. | Must be selected before viewing Finnhub values. |

Forbidden ticker-selection logic:

- do not select tickers because they appear attractive;
- do not select tickers because they appear likely to be buys;
- do not select tickers because of expected upside;
- do not select tickers to validate an existing trade idea;
- do not select tickers to improve portfolio decisions;
- do not select tickers to optimize scanner output;
- do not change the sample after viewing values.

Ticker selection must be documented before any Finnhub source value is viewed.

## 7. Proposed Endpoint Family Protocol

A later approved review may inspect only the following endpoint families if access is allowed and terms permit.

| Endpoint family | Allowed review purpose | Disallowed use |
|---|---|---|
| Recommendation Trends | Check whether recommendation category fields exist and whether date or period fields are visible. | Do not store recommendation values; do not convert categories into project scores; do not treat recommendations as actions. |
| Price Target | Check whether average, low, high, or consensus-style price-target fields exist and whether date/freshness fields are visible. | Do not store price targets; do not calculate implied upside; do not treat targets as buy/sell evidence. |
| EPS Estimates | Check whether EPS estimate fields exist and whether fiscal-period/date fields are visible. | Do not store estimate values; do not compare estimates across tickers; do not evaluate attractiveness. |

No other endpoint family is included in this protocol.

If any endpoint family requires an account, API key, subscription, credential, or terms that are not approved, it must be skipped.

## 8. Allowed Qualitative Observations

A later approved sample review may record only qualitative observations.

Allowed observation values:

- `field_present`;
- `field_absent`;
- `field_unclear`;
- `date_present`;
- `date_absent`;
- `date_unclear`;
- `freshness_present`;
- `freshness_absent`;
- `freshness_unclear`;
- `permission_blocked`;
- `plan_blocked`;
- `endpoint_unavailable`;
- `definition_clear`;
- `definition_unclear`;
- `point_in_time_supported_claimed`;
- `point_in_time_not_supported`;
- `point_in_time_unclear`.

The review may also record short governance notes, but those notes must not contain raw source values.

## 9. Prohibited Observations and Stored Material

A later approved sample review must not store:

- raw numeric values;
- raw text values from endpoint responses;
- copied API responses;
- screenshots;
- CSV files;
- JSON files;
- spreadsheet files;
- downloaded provider payloads;
- normalized values;
- calculated implied upside;
- analyst rating values;
- price target values;
- EPS estimate values;
- revenue estimate values;
- ticker-level recommendations;
- comparisons between tickers;
- rankings or scores.

The protocol allows only qualitative field-availability and governance observations.

## 10. Account, API-Key, and Secrets Rules

No Finnhub account or API key is authorized by this protocol.

If a later execution approval permits account or API-key use, the following rules are mandatory:

- no key may be committed to the repository;
- no key may be placed in documentation;
- no key may be placed in CSV files;
- no key may be placed in scripts;
- no key may be pasted into GitHub issues;
- no key may be pasted into GitHub pull requests;
- no key may be included in screenshots;
- no key may be exposed in terminal logs;
- no workflow may use the key;
- no runtime script may use the key;
- the key must be handled only through approved local secret handling;
- if exposure is suspected, the key must be revoked or rotated.

If safe key handling cannot be guaranteed, the sample review must not proceed.

## 11. Required Review Output Format

A later approved sample review must produce a documentation-only review with this structure.

| Section | Required content |
|---|---|
| Approval reference | Reference to the execution approval note. |
| Review date | Date the sample review occurred. |
| Reviewer role | PM / Functional Analyst / Technical Analyst / Research / Governance. |
| Source | Finnhub. |
| Ticker sample | Ticker symbols only if permitted by the approval note. |
| Endpoint families inspected | Must match this protocol. |
| API calls made | Yes or no, only if separately approved. |
| Credentials used | Yes or no; never record credential values. |
| Raw values stored | Must be no. |
| Screenshots stored | Must be no unless separately approved. |
| CSVs created | Must be no. |
| Runtime files changed | Must be no. |
| Qualitative findings | Allowed observation values only. |
| Terms or permission issues | Required if encountered. |
| Field-definition issues | Required if encountered. |
| Point-in-time issues | Required if encountered. |
| Research-only conclusion | Required. |
| Runtime integration conclusion | Must be no. |
| Decision Engine conclusion | Must be no. |
| Reporting or Telegram conclusion | Must be no. |

## 12. Protocol Rejection or Revision Triggers

This protocol must be rejected or revised before execution if any of the following occur:

- terms of use are unclear;
- terms prohibit the intended review;
- manual research use is not confirmed;
- account or API-key handling is unresolved;
- safe secret handling is not available;
- endpoint access requires unapproved subscription changes;
- raw values would need to be stored;
- screenshots would need to be stored;
- qualitative observations are insufficient for the decision;
- sample size needs to exceed 5 tickers;
- more than 3 endpoint families are needed;
- results would be used for trade decisions;
- results would affect Decision Engine behavior;
- results would affect Reporting or Telegram language;
- results would require runtime files, CSVs, generated artifacts, scripts, or tests.

If any trigger is met, the sample review must not proceed until governance resolves it.

## 13. Completion Criteria for a Future Sample Review

A future approved sample review may be considered complete only if it answers these questions qualitatively:

1. Are Finnhub analyst expectations fields visible in the reviewed endpoint families?
2. Are field definitions clear enough for future governance discussion?
3. Are date, freshness, or fiscal-period fields visible?
4. Is point-in-time support clear, unclear, or unsupported?
5. Are missing or unavailable fields visible as exceptions?
6. Are terms, plan access, account, or API-key constraints blocking further review?
7. Can future research proceed without storing raw values?
8. Should Finnhub remain a candidate source, be revised for further review, or be deprioritized?

The future sample review must not answer whether any ticker should be bought, sold, held, ranked, scored, or prioritized.

## 14. Not Approved by This Protocol

This protocol does not approve:

- Finnhub as an approved project source;
- sample execution;
- runtime data collection;
- manual ticker-by-ticker analyst data collection;
- source value viewing;
- source value collection;
- source value storage;
- screenshots;
- scraping;
- provider/API calls;
- account creation;
- API-key creation;
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

This document executes the recommended documentation-only Finnhub limited sample review protocol step already authorized for protocol design by the Finnhub sample proposal governance note.

It does not identify additional deferred work beyond the existing governed analyst expectations research path.

Backlog impact assessment:
- No new backlog items identified.

## 16. Recommended Next Step

The recommended next step is a separate execution approval note deciding whether this protocol may be executed.

That note must either:

- approve execution of the limited sample review under this protocol;
- require protocol revision;
- or reject or defer sample execution.

No sample values may be viewed or collected until that execution approval note exists.
