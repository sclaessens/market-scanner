# Finnhub Combined Analyst Expectations Source Review

## 1. Status and Scope

This document is a documentation-only combined source review for Finnhub as a candidate analyst expectations source.

It combines:

- source-specific due diligence;
- terms and access-permission review;
- account, API-key, and secrets governance design;
- execution-readiness conclusion.

This document does not implement code, tests, CSV files, generated artifacts, reports, workflows, provider integration, API calls, scraping, account creation, API-key creation, credentials or secrets, runtime orchestration, daily ingestion, backtesting code, Reporting changes, Telegram changes, scanner changes, Decision Engine changes, portfolio files, watchlist files, fundamentals files, or runtime behavior changes.

No sprint is closed or certified complete by this document.

Finnhub is not approved as a project source by this document.

No sample values are viewed or collected by this document.

No API access is authorized by this document.

## 2. Background

The project previously completed a documentation-only Financial Modeling Prep research path.

The FMP execution-readiness reconciliation deferred FMP for now because too many access, terms, credential, and execution-readiness controls remained unresolved.

The reconciliation recommended evaluating a second candidate source before continuing to resolve FMP-specific blockers.

Finnhub is reviewed here as that second candidate source for comparison.

This document supports backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy`.

Analyst expectations, analyst consensus, price targets, estimates, and source-provider reviews remain research-only.

## 3. Research-Only Boundary

Analyst expectations remain research-only.

Analyst expectations must not become buy/sell advice.

Finnhub data or documentation must not become:

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

Any future Decision Engine use requires separate governance, separate approval, explicit design, tests, audit controls, and acceptance criteria.

## 4. Review Method

This review used public documentation review only.

This review did not:

- call Finnhub APIs;
- create or request an API key;
- create a Finnhub account;
- inspect ticker-level values;
- store source values;
- download provider data;
- take screenshots;
- create CSV files;
- change runtime behavior.

If public documentation could not conclusively answer a question, the question is marked as `UNRESOLVED`.

Public documentation pages reviewed at a documentation level included the Finnhub API documentation landing page and documentation pages that appear to correspond to recommendation trends, price target, and EPS estimate endpoint families.

## 5. Finnhub Source Identity and Access Model

| Review area | Status | Notes |
|---|---|---|
| Source name | Finnhub | Candidate only. |
| Source category | API-oriented market-data provider | Candidate source from the existing named-source shortlist. |
| Public developer documentation | `DOCUMENTED` | Public API documentation is visible. |
| Analyst-relevant endpoint indication | `PARTIALLY DOCUMENTED` | Public documentation pages appear to exist for recommendation trends, price target, and EPS estimate endpoint families. |
| API-key or account requirement | `UNRESOLVED` | Public documentation visibility does not establish access permission. Endpoint access requirements must be confirmed before any use. |
| Source approval | `NOT APPROVED` | This document does not approve Finnhub as a project source. |
| Data collection permission | `NOT APPROVED` | Documentation visibility is not permission to collect data. |

Finnhub is therefore an API-oriented candidate source only.

## 6. Candidate Analyst Expectations Field Review

This section assesses documented availability only.

It is not ticker-level verification.

No values were viewed, collected, copied, stored, or normalized.

| Candidate field | Status | Notes |
|---|---|---|
| `consensus_rating` | `partially documented` | Recommendation trend documentation suggests rating-category trend data may exist, but consensus definition is not execution-verified. |
| `analyst_count` | `unclear` | Public documentation reviewed here does not conclusively establish analyst-count semantics. |
| `buy_count` | `partially documented` | Recommendation trend documentation may expose buy-like recommendation categories, but exact mapping is unresolved. |
| `hold_count` | `partially documented` | Recommendation trend documentation may expose hold-like categories, but exact mapping is unresolved. |
| `sell_count` | `partially documented` | Recommendation trend documentation may expose sell-like categories, but exact mapping is unresolved. |
| `average_price_target` | `partially documented` | Price-target documentation appears relevant, but field semantics and date/freshness handling remain unresolved. |
| `low_price_target` | `partially documented` | Price-target documentation appears relevant, but field semantics remain unresolved. |
| `high_price_target` | `partially documented` | Price-target documentation appears relevant, but field semantics remain unresolved. |
| `current_price_at_collection` | `unclear` | May require a separate quote or market-data endpoint; not approved or reviewed for collection. |
| `implied_upside_pct` | `not documented` | If calculated internally later, formula governance would be required. |
| `current_year_eps_estimate` | `partially documented` | EPS estimate documentation appears relevant, but fiscal-period mapping remains unresolved. |
| `next_year_eps_estimate` | `partially documented` | EPS estimate documentation appears relevant, but fiscal-period mapping remains unresolved. |
| `current_year_revenue_estimate` | `unclear` | Revenue-estimate support was not conclusively established in this review. |
| `next_year_revenue_estimate` | `unclear` | Revenue-estimate support was not conclusively established in this review. |
| `estimate_revision_direction` | `unclear` | Upgrade/downgrade or trend-style support was not conclusively reviewed here. |
| `source_freshness_date` | `unresolved` | Source freshness semantics were not conclusively established. |
| `as_of_date` | `unresolved` | Date semantics require endpoint-level review before use. |
| `raw_source_value` | `unresolved` | Storage rights are not established. |
| `normalized_value` | `unresolved` | Normalization is not approved and would require later governance. |

## 7. Consensus Definition Review

| Consensus topic | Status | Notes |
|---|---|---|
| Source-published consensus | `UNRESOLVED` | Public documentation reviewed here does not conclusively define whether consensus is source-published or derived. |
| Provider-calculated consensus | `UNRESOLVED` | Provider calculation methodology is not established by this review. |
| Analyst count | `UNRESOLVED` | Analyst-count semantics are not confirmed. |
| Rating categories | `PARTIALLY DOCUMENTED` | Recommendation trend endpoint family appears relevant, but exact category mapping requires later review. |
| Buy/hold/sell distribution | `PARTIALLY DOCUMENTED` | Possible category distribution exists, but mapping and count semantics remain unresolved. |
| Price target consensus | `PARTIALLY DOCUMENTED` | Price-target endpoint family appears relevant, but consensus rules remain unresolved. |
| Estimate periods | `UNRESOLVED` | EPS estimate period semantics require later review. |
| Stale estimate handling | `UNRESOLVED` | Not conclusively documented in this review. |
| Missing estimate handling | `UNRESOLVED` | Not conclusively documented in this review. |

No consensus field may be treated as project scoring, ranking, allocation, tradeability, urgency, conviction, eligibility, or Decision Engine input.

## 8. Point-in-Time and Historical Validation Review

| Historical validation topic | Status | Notes |
|---|---|---|
| Historical analyst expectations | `UNRESOLVED` | Historical support for analyst expectations was not conclusively established. |
| Point-in-time reconstruction | `UNRESOLVED` | Documentation reviewed here does not prove look-ahead-safe reconstruction. |
| As-of dates | `UNRESOLVED` | Date fields may exist in endpoint outputs, but semantics were not execution-verified. |
| Revision history | `UNRESOLVED` | Revision/versioning behavior is not established. |
| Unavailable historical values | `UNRESOLVED` | Treatment of unavailable historical data is not established. |
| Delisted or renamed securities | `UNRESOLVED` | Survivorship-bias support is not established. |
| Current snapshots vs historical truth | `UNRESOLVED` | Current documentation or current endpoint availability must not be treated as historical truth. |

Finnhub cannot be used for historical validation unless a later governed review confirms point-in-time support.

## 9. Terms and Access-Permission Review

This review used public documentation only.

It did not establish legal permission, account permission, endpoint permission, or storage rights.

| Permission or access control | Status | Notes |
|---|---|---|
| Full terms-of-use compatibility | `UNRESOLVED` | Complete terms compatibility for project research use was not established. |
| Manual research permission | `UNRESOLVED` | Permission for limited manual field-availability review is not confirmed. |
| API endpoint access permission | `UNRESOLVED` | Endpoint access is not approved. |
| Raw-value storage rights | `UNRESOLVED` | Raw values must not be stored. |
| Normalized-value storage rights | `UNRESOLVED` | Normalized values must not be stored. |
| Screenshot permission | `UNRESOLVED` | Screenshots must remain prohibited unless separately approved. |
| Redistribution/publication rights | `UNRESOLVED` | Provider output must not be published or committed. |
| Rate-limit constraints | `UNRESOLVED` | Rate limits and usage constraints require later review. |
| Subscription or plan requirements | `UNRESOLVED` | Plan-level endpoint access is not confirmed. |
| Commercial vs personal-use restrictions | `UNRESOLVED` | Intended-use restrictions require later review. |

Because these controls remain unresolved, Finnhub sample execution is not approved.

## 10. Account, API-Key, and Secrets Governance Design

This document does not authorize Finnhub account creation or API-key creation.

Before any Finnhub account or API key could be created, the project must document:

- full terms-of-use compatibility;
- manual research permission;
- endpoint access permission;
- plan or subscription requirements;
- rate-limit constraints;
- raw-value storage policy;
- normalized-value storage policy;
- screenshot policy;
- redistribution and publication policy;
- local protected-credential handling;
- exposure response;
- rotation or revocation procedure;
- reviewer role;
- review date;
- exact ticker list;
- exact endpoint families;
- allowed qualitative observations;
- new execution approval after all controls are resolved.

Finnhub credentials must never be stored in:

- repository files;
- Markdown documentation;
- Python files;
- shell scripts;
- notebooks;
- CSV files;
- JSON files;
- YAML files;
- committed environment files;
- generated artifacts;
- reports;
- screenshots;
- committed logs;
- GitHub issues;
- GitHub pull requests;
- PR comments;
- commit messages;
- branch names;
- filenames;
- test fixtures;
- CI output.

Local protected-credential handling rules, if later approved, must include:

| Rule | Requirement |
|---|---|
| Local only | Use only in an approved local review environment. |
| No repository storage | Do not commit, stage, paste, or document credentials. |
| No saved output | Do not save logs or screenshots containing credentials. |
| No runtime dependency | Do not make credentials part of runtime processes. |
| No workflow dependency | Do not use credentials in GitHub Actions unless separately governed. |
| Temporary use | Limit use to an approved review session. |
| Exposure response | Define revocation or rotation before use. |

This design defines conditions for future approval only. It does not approve credential use.

## 11. Data Quality and Exception Review

Expected data-quality and exception risks:

| Risk | Status | Notes |
|---|---|---|
| Missing data | `EXPECTED RISK` | Field coverage may differ by ticker, market, endpoint, or plan. |
| Stale data | `EXPECTED RISK` | Freshness semantics are unresolved. |
| Partial coverage | `EXPECTED RISK` | Coverage by region and security type is not verified. |
| Unclear field definitions | `EXPECTED RISK` | Consensus, estimate, and recommendation definitions require further review. |
| Low analyst count | `EXPECTED RISK` | Analyst-count availability and minimum-count rules are unresolved. |
| Fiscal-period ambiguity | `EXPECTED RISK` | EPS/revenue estimate period mapping is unresolved. |
| Currency issues | `EXPECTED RISK` | Price target and revenue currency metadata are unresolved. |
| Methodology opacity | `EXPECTED RISK` | Provider methodology is not fully established. |
| Regional inconsistency | `EXPECTED RISK` | Global coverage behavior is unresolved. |
| Survivorship bias | `EXPECTED RISK` | Historical universe and delisting support are unresolved. |
| Point-in-time uncertainty | `EXPECTED RISK` | Look-ahead-safe historical validation support is unresolved. |

Any future sample review must remain exception-based and qualitative only.

## 12. Execution-Readiness Conclusion

Execution-readiness decision: continue Finnhub to a documentation-only limited manual sample collection proposal.

Rationale:

- Finnhub appears to have publicly visible API documentation and analyst-relevant endpoint families;
- the review can be advanced more efficiently than the prior FMP micro-step path by combining due diligence outputs;
- enough candidate field relevance exists to justify defining a limited sample proposal;
- execution is still not approved;
- terms, access, API-key, storage, point-in-time, and permission controls remain unresolved.

This conclusion authorizes only the creation of a documentation-only limited manual sample collection proposal for Finnhub.

It does not authorize sample execution.

It does not authorize API calls.

It does not authorize account creation.

It does not authorize API-key creation.

It does not authorize sample value viewing, sample value collection, screenshots, CSV creation, runtime integration, Decision Engine use, Reporting use, or Telegram use.

## 13. Explicitly Not Approved

This document does not approve:

- Finnhub as an approved project source;
- account creation;
- API-key creation;
- endpoint access;
- sample execution;
- runtime data collection;
- manual ticker-by-ticker analyst data collection;
- sample value viewing;
- sample value collection;
- sample value storage;
- screenshots;
- scraping;
- provider/API calls;
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

This document creates a combined source review for Finnhub under the existing governed analyst expectations research path and does not authorize implementation, source access, sample execution, or runtime behavior change.

Backlog impact assessment:
- No new backlog items identified.

## 15. Recommended Next Step

The recommended next step is a documentation-only Finnhub limited manual sample collection proposal.

That proposal should define:

- whether limited sample collection may be proposed later;
- maximum ticker sample size;
- exact endpoint families to inspect;
- qualitative observations allowed;
- storage prohibitions;
- terms and access preconditions;
- account/API-key/secrets preconditions;
- no-runtime-change boundary;
- no-Decision-Engine-use boundary;
- no-Reporting-or-Telegram-use boundary.

No sample values may be viewed or collected until a later execution approval note explicitly approves execution.
