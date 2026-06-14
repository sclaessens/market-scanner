# Finnhub Limited Manual Sample Collection Proposal

## 1. Status and Scope

This document is a documentation-only proposal for a possible future limited manual sample review of Finnhub analyst expectations data.

It follows:

- `docs/research/analyst_expectations_source_policy_and_validation_design.md`;
- `docs/research/analyst_expectations_source_comparison_matrix.md`;
- `docs/research/analyst_expectations_named_source_shortlist_review.md`;
- `docs/research/analyst_expectations_named_source_due_diligence_template.md`;
- `docs/research/analyst_expectations_fmp_execution_readiness_reconciliation.md`;
- `docs/research/analyst_expectations_finnhub_combined_source_review.md`;
- backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy`.

This document proposes conditions for a later limited manual sample review. It does not perform that review.

This document does not implement code, tests, CSV files, generated artifacts, reports, workflows, provider integration, provider/API calls, scraping, account creation, API-key creation, credentials or secrets, runtime orchestration, daily ingestion, backtesting code, Reporting changes, Telegram changes, scanner changes, Decision Engine changes, portfolio files, watchlist files, fundamentals files, or runtime behavior changes.

No sprint is closed or certified complete by this document.

Finnhub is not approved as a project source by this document.

No sample values are collected, viewed, copied, stored, screenshotted, summarized, or normalized by this document.

No API access is authorized by this document.

No sample collection is authorized until this proposal is separately reviewed and approved through governance.

## 2. Purpose

The purpose of this proposal is to define the minimum governance controls required before any limited manual sample review of Finnhub analyst expectations data could occur.

The proposal defines:

- whether limited sample collection may be proposed later;
- the maximum allowed sample size if later approved;
- which Finnhub endpoint families may be visually inspected;
- which qualitative observations may be recorded;
- whether values may be stored;
- which terms and access checks are mandatory first;
- how account, API-key, and secrets questions must be handled;
- how research-only boundaries are preserved;
- which acceptance criteria decide whether Finnhub remains a candidate source.

This document intentionally stops before data collection.

## 3. Research-Only Boundary

Analyst expectations remain research-only.

Analyst expectations must not become buy/sell advice.

A future Finnhub sample review must not create or imply:

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

A limited sample review may only evaluate source structure, documented field presence, date availability, freshness semantics, access constraints, missing-data behavior, and source-quality questions.

It must not evaluate whether any ticker is attractive, actionable, tradable, undervalued, overvalued, or portfolio-relevant.

Future Decision Engine use requires separate governance, separate approval, explicit design, tests, and audit controls.

## 4. Pre-Approval Requirements

Before any Finnhub sample values are viewed, collected, copied, screenshotted, stored, summarized, normalized, or used, the following requirements must be completed.

| Requirement | Status before collection | Notes |
|---|---|---|
| Confirm full Finnhub terms of use | Required | Public documentation review is not enough. |
| Confirm permitted manual research use | Required | Must establish whether viewing limited field availability is allowed. |
| Confirm whether raw values may be stored | Required | If unclear, raw values must not be stored. |
| Confirm whether normalized values may be stored | Required | If unclear, normalized values must not be stored. |
| Confirm whether screenshots are permitted | Required | If unclear, screenshots must not be taken. |
| Confirm endpoint access permission | Required | Documentation visibility is not endpoint access approval. |
| Confirm API-key requirements | Required | No key may be placed in repository files. |
| Confirm secrets handling policy | Required | Any future key must use approved local or secret-store handling only. |
| Confirm rate-limit constraints | Required if endpoint access is proposed | Any future access must respect provider limits. |
| Confirm subscription or plan requirements | Required | Endpoint availability may depend on plan. |
| Confirm field definitions from documentation | Required | Sample review must know what it is checking. |
| Confirm no runtime integration | Required | Sample review must remain outside production pipeline. |
| Confirm no Decision Engine use | Required | Sample values must not affect decisions. |
| Confirm no Reporting or Telegram use | Required | Sample values must not be communicated as recommendations. |

If any required item remains unresolved, sample collection must not proceed.

## 5. Proposed Sample Size

If later approved, the sample must be intentionally small.

Recommended maximum sample:

- maximum 5 tickers;
- maximum 3 endpoint families;
- maximum 1 review session;
- maximum 1 source provider: Finnhub;
- no repeated collection until a later governance step approves it.

The purpose is field verification, not analysis.

The sample must not be large enough to become an informal dataset.

The sample must not become a substitute for governed ingestion.

## 6. Proposed Sample Ticker Selection Rules

If later approved, sample tickers should be selected for coverage diversity only.

Allowed selection principles:

- include at most 2 current portfolio tickers;
- include at most 2 recent scanner tickers;
- include at most 1 non-US or cross-listed candidate if relevant;
- include different sectors if possible;
- include at least 1 ticker where missing or partial data would be acceptable as a test case.

Forbidden selection principles:

- do not select tickers because they look attractive;
- do not select tickers because they are likely buys;
- do not select tickers because of expected upside;
- do not select tickers to influence portfolio decisions;
- do not select tickers to validate an existing trade idea;
- do not select tickers to optimize the scanner.

Ticker selection must be documented before viewing values.

## 7. Proposed Finnhub Endpoint Family Scope

If later approved, the sample may inspect only the minimum endpoint families needed to answer due-diligence questions.

Proposed endpoint families:

| Endpoint family | Review purpose | Approved by this document? |
|---|---|---|
| Recommendation Trends | Verify whether rating-category trend fields are available and whether date or period fields are visible. | No |
| Price Target | Verify whether average, low, high, or consensus-style price-target fields are available and whether date/freshness fields are visible. | No |
| EPS Estimates | Verify whether EPS estimate fields are available and whether fiscal-period/date fields are visible. | No |

This document does not approve calling these endpoints.

The endpoint list is a proposal only.

If any endpoint family requires a subscription, credentials, or terms that are not approved, it must be skipped.

## 8. Proposed Field Inspection Scope

If later approved, visual field inspection should be limited to the following candidate fields.

| Candidate field | Allowed inspection purpose | Storage allowed by this proposal? |
|---|---|---|
| `ticker` | Confirm symbol mapping. | No |
| `as_of_date` | Confirm whether values have a date. | No |
| `source_freshness_date` | Confirm whether source freshness is exposed. | No |
| `consensus_rating` | Confirm whether a source-published or provider-derived consensus field exists. | No |
| `analyst_count` | Confirm whether analyst count is exposed. | No |
| `buy_count` | Confirm whether buy-equivalent count is exposed. | No |
| `hold_count` | Confirm whether hold-equivalent count is exposed. | No |
| `sell_count` | Confirm whether sell-equivalent count is exposed. | No |
| `average_price_target` | Confirm whether average target field exists. | No |
| `low_price_target` | Confirm whether low target field exists. | No |
| `high_price_target` | Confirm whether high target field exists. | No |
| `current_year_eps_estimate` | Confirm whether current-year EPS estimate exists. | No |
| `next_year_eps_estimate` | Confirm whether next-year EPS estimate exists. | No |
| `current_year_revenue_estimate` | Confirm whether current-year revenue estimate exists if endpoint support is documented. | No |
| `next_year_revenue_estimate` | Confirm whether next-year revenue estimate exists if endpoint support is documented. | No |
| `estimate_revision_direction` | Confirm whether revision-like or trend-like data exists. | No |

This proposal does not allow storing raw values, normalized values, screenshots, downloaded responses, copied endpoint outputs, generated CSVs, JSON files, spreadsheet files, or provider payloads.

## 9. Allowed Qualitative Observations

If later approved, the sample review may record only qualitative observations.

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

The review may record short governance notes, but those notes must not contain raw source values.

## 10. Storage and Handling Rules

Default rule: do not store sample values.

A later approved sample review may record only qualitative observations unless terms explicitly permit more, and a separate governance note explicitly approves it.

Allowed qualitative observations, if later approved:

- field present;
- field absent;
- field unclear;
- date present;
- date absent;
- endpoint requires plan upgrade;
- endpoint returns permission limitation;
- field definition unclear;
- source freshness unclear;
- historical semantics unclear.

Not allowed unless separately approved:

- raw numeric values;
- raw text values;
- copied API responses;
- screenshots;
- CSV files;
- JSON files;
- spreadsheet files;
- normalized values;
- calculated implied upside;
- ticker-level investment conclusions;
- comparisons between tickers;
- ranking or scoring of securities.

## 11. Account, API-Key, and Secrets Preconditions

No Finnhub account or API key is authorized by this document.

If a future approved sample review requires a Finnhub account or API key:

- the key must not be committed to the repository;
- the key must not be placed in documentation;
- the key must not be placed in CSV files;
- the key must not be placed in scripts;
- the key must not be pasted into issue or PR text;
- the key must be handled through approved local environment or secret-management procedures only;
- the key must be rotated or revoked if exposure is suspected;
- no automated workflow may use the key unless separate workflow governance approves it.

Credential handling must be resolved before any endpoint access occurs.

## 12. Review Output Format

If later approved, the limited sample review should produce a documentation-only review using this structure.

| Section | Required content |
|---|---|
| Approval reference | Link or reference to the approved sample collection governance note. |
| Sample date | Date of review. |
| Source | Finnhub. |
| Reviewer role | PM / Functional Analyst / Technical Analyst / Research / Governance. |
| Runtime data collected | No. |
| API calls made | Yes or no, only if separately approved. |
| Credentials used | Yes or no, never recorded. |
| Sample ticker count | Maximum 5 if approved. |
| Endpoint families inspected | Must match approved scope. |
| Raw values stored | No unless separately approved. |
| Screenshots stored | No unless separately approved. |
| Qualitative field availability notes | Present / absent / unclear only. |
| Licensing unresolved items | Required if any. |
| Point-in-time unresolved items | Required if any. |
| Research-only conclusion | Required. |
| Runtime integration conclusion | Must remain no. |

## 13. Acceptance Criteria for Keeping Finnhub as a Candidate

A later approved sample review may keep Finnhub as a candidate source only if the review can answer enough due-diligence questions without violating governance.

Minimum acceptance criteria:

- terms of use do not clearly prohibit the intended research review;
- account and API-key handling can be governed safely if needed;
- endpoint access path is documented;
- candidate analyst expectation fields are identifiable;
- source freshness or date semantics can be evaluated;
- missing-data behavior can be observed qualitatively;
- no raw source values need to be stored for the next governance decision;
- no runtime integration is required;
- no Decision Engine, Reporting, or Telegram use is implied.

If these criteria are not met, Finnhub should remain unresolved or be deprioritized as a candidate source.

## 14. Rejection or Deprioritization Triggers

Finnhub should be rejected or deprioritized as a candidate analyst expectations source if later review finds:

- terms prohibit the intended research use;
- data storage is not allowed and qualitative review is insufficient;
- endpoint definitions are too unclear;
- required fields are unavailable;
- point-in-time support is absent for historical validation;
- historical revision handling cannot be understood;
- account or API-key handling cannot be governed safely;
- rate limits or subscription constraints prevent responsible use;
- source data would create pressure for hidden scoring or Decision Engine bypass;
- source use would require runtime changes before research is complete.

Rejection or deprioritization must be documented without deleting the review history.

## 15. Explicitly Not Approved

This proposal does not approve:

- Finnhub as an approved project source;
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

## 16. Backlog Impact Assessment

Existing backlog item `BL-0018` remains sufficient.

This document executes the recommended documentation-only Finnhub limited manual sample collection proposal already implied by the combined Finnhub source review.

It does not identify additional deferred work beyond the existing governed analyst expectations research path.

Backlog impact assessment:
- No new backlog items identified.

## 17. Recommended Next Step

The recommended next step is a documentation-only Finnhub sample proposal governance note.

That note should review this proposal and decide whether it is:

- approved for protocol design only;
- revision required before protocol design;
- rejected or deferred.

No sample collection, API calls, account creation, API-key creation, credentials, screenshots, CSVs, or runtime changes are authorized until later governance explicitly approves them.
