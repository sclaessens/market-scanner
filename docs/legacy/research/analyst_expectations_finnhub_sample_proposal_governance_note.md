# Finnhub Sample Proposal Governance Note

## 1. Status and Scope

This document is a documentation-only governance note reviewing the Finnhub limited manual sample collection proposal.

It follows:

- `docs/research/analyst_expectations_source_policy_and_validation_design.md`;
- `docs/research/analyst_expectations_source_comparison_matrix.md`;
- `docs/research/analyst_expectations_named_source_shortlist_review.md`;
- `docs/research/analyst_expectations_named_source_due_diligence_template.md`;
- `docs/research/analyst_expectations_fmp_execution_readiness_reconciliation.md`;
- `docs/research/analyst_expectations_finnhub_combined_source_review.md`;
- `docs/research/analyst_expectations_finnhub_limited_manual_sample_collection_proposal.md`;
- backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy`.

This document reviews the proposal only.

This document does not implement code, tests, CSV files, generated artifacts, reports, workflows, provider integration, provider/API calls, scraping, account creation, API-key creation, credentials or secrets, runtime orchestration, daily ingestion, backtesting code, Reporting changes, Telegram changes, scanner changes, Decision Engine changes, portfolio files, watchlist files, fundamentals files, or runtime behavior changes.

No sprint is closed or certified complete by this document.

Finnhub is not approved as a project source by this document.

No sample values are collected, viewed, copied, stored, screenshotted, summarized, or normalized by this document.

No API access is authorized by this document.

## 2. Governance Decision

The Finnhub limited manual sample collection proposal is approved for protocol design only.

This means the project may create a future documentation-only Finnhub limited sample review protocol.

This does not approve sample collection.

This does not approve API calls.

This does not approve account creation.

This does not approve API-key creation.

This does not approve credentials or secrets.

This does not approve storing source values.

This does not approve screenshots.

This does not approve CSV creation.

This does not approve runtime ingestion.

This does not approve Decision Engine, Reporting, Telegram, scanner, portfolio, watchlist, or fundamentals integration.

## 3. Decision Rationale

The proposal is accepted for protocol design because it defines strict guardrails before any sample review could occur.

The proposal correctly requires:

- terms-of-use review before collection;
- confirmation of permitted manual research use;
- no default storage of raw values;
- no default storage of normalized values;
- no screenshots unless separately approved;
- no repository-stored credentials;
- no runtime integration;
- no Decision Engine use;
- no Reporting or Telegram use;
- small sample size only if later approved;
- qualitative field verification only;
- explicit rejection or deprioritization triggers.

The proposal is not approved for execution because several unresolved governance questions remain.

## 4. Required Conditions Before Any Future Sample Review

Before any sample values may be viewed, copied, collected, screenshotted, stored, summarized, normalized, or used, a separate protocol must define and confirm the following controls.

| Control | Required status before sample review |
|---|---|
| Exact ticker list | Must be defined before collection. |
| Exact endpoint families | Must be defined before collection. |
| Maximum sample size | Must remain no more than 5 tickers unless separately governed. |
| Terms-of-use review | Must be completed before collection. |
| Manual research permission | Must be confirmed before collection. |
| Account or API-key handling | Must be resolved before any endpoint access. |
| Secrets handling | Must prohibit repository storage and PR/issue exposure. |
| Raw value storage | Must remain prohibited unless explicitly approved. |
| Normalized value storage | Must remain prohibited unless explicitly approved. |
| Screenshots | Must remain prohibited unless explicitly approved. |
| CSV or generated files | Must remain prohibited. |
| Qualitative observations | Must be limited to present / absent / unclear style findings. |
| Runtime integration | Must remain prohibited. |
| Decision Engine use | Must remain prohibited. |
| Reporting or Telegram use | Must remain prohibited. |
| Backtesting use | Must remain prohibited. |

If any required control cannot be satisfied, sample review must not proceed.

## 5. Approved Next Step

The only approved next step is a documentation-only Finnhub limited sample review protocol.

That protocol may define:

- the exact ticker sample;
- the exact endpoint families;
- the exact qualitative observations allowed;
- the exact storage prohibitions;
- the approval reference;
- the no-runtime-change boundary;
- the no-Decision-Engine-use boundary;
- the no-Reporting-or-Telegram-use boundary;
- the no-backtesting-use boundary;
- the conditions under which the protocol must be rejected or revised.

The protocol must not collect sample values.

The protocol must not call provider APIs.

The protocol must not introduce accounts, API keys, credentials, or secrets.

The protocol must not create CSV files or generated artifacts.

## 6. Explicitly Not Approved

This governance note does not approve:

- Finnhub as an approved project source;
- runtime data collection;
- manual ticker-by-ticker analyst data collection;
- sample value collection;
- sample value viewing;
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

## 7. Governance Status Summary

| Item | Status |
|---|---|
| Finnhub combined source review | Completed as documentation-only. |
| Finnhub limited sample proposal | Reviewed. |
| Finnhub sample proposal decision | Approved for protocol design only. |
| Finnhub source approval | Not approved. |
| Finnhub sample collection | Not approved. |
| Finnhub API access | Not approved. |
| Finnhub account creation | Not approved. |
| Finnhub API-key creation | Not approved. |
| Finnhub credentials or secrets | Not approved. |
| Finnhub runtime integration | Not approved. |
| Finnhub Decision Engine integration | Not approved. |
| Finnhub Reporting or Telegram integration | Not approved. |

## 8. Backlog Impact Assessment

Existing backlog item `BL-0018` remains sufficient.

This document reviews the Finnhub limited manual sample collection proposal and approves only the next documentation-only protocol design step.

It does not identify additional deferred work beyond the existing governed analyst expectations research path.

Backlog impact assessment:
- No new backlog items identified.

## 9. Recommended Next Step

Create a documentation-only Finnhub limited sample review protocol.

The protocol should define the exact ticker sample, endpoint families, qualitative observations, storage prohibitions, approval reference, and research-only boundaries.

No sample values may be collected until the protocol is separately reviewed and approved.
