# Analyst Expectations Provider Strategy Closure

## 1. Status and Scope

This document is a documentation-only strategy closure note for the analyst expectations provider research path.

It consolidates the Financial Modeling Prep and Finnhub documentation-only review paths and intentionally stops the current micro-step provider workflow.

This document does not implement code, tests, CSV files, generated artifacts, reports, workflows, provider integration, provider/API calls, scraping, account creation, API-key creation, credentials or secrets, runtime orchestration, daily ingestion, backtesting code, Reporting changes, Telegram changes, scanner changes, Decision Engine changes, portfolio files, watchlist files, fundamentals files, or runtime behavior changes.

No sprint is closed or certified complete by this document.

No analyst expectations source is approved by this document.

No provider is approved by this document.

No sample values are collected, viewed, copied, stored, screenshotted, summarized, or normalized by this document.

No API access is authorized by this document.

## 2. Background

The analyst expectations research path was created under backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy`.

The path first defined source-policy, comparison, shortlist, and due-diligence templates for analyst expectations data.

The project then investigated Financial Modeling Prep as a first candidate provider and Finnhub as a second candidate provider.

Both investigations remained documentation-only and research-only.

The FMP path became too granular, producing many small governance documents before reaching an execution-readiness decision.

The Finnhub path initially attempted to combine steps, but later also started drifting back into micro-step execution notes and checklists.

This document closes that pattern and records a broader strategy decision.

## 3. Research-Only Boundary

Analyst expectations remain research-only.

Analyst expectations must not become buy/sell advice.

Analyst expectations provider research must not create or imply:

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

Any future Decision Engine use requires separate governance, separate approval, explicit design, tests, audit controls, and acceptance criteria.

## 4. Financial Modeling Prep Summary

Financial Modeling Prep was reviewed as an API-oriented market-data provider candidate.

The FMP documentation-only research path established that FMP may expose analyst expectations-related endpoint families or data categories.

However, FMP was deferred because execution-readiness remained blocked by unresolved controls.

Key unresolved FMP blockers included:

- complete terms-of-use compatibility;
- limited manual research permission;
- raw-value storage rights;
- normalized-value storage rights;
- screenshot permission;
- redistribution or publication rights;
- rate-limit constraints;
- subscription or plan requirements;
- endpoint access permission;
- account and credential creation approval;
- local protected-credential handling approval;
- exact ticker sample;
- exact endpoint families;
- execution approval.

FMP remains candidate-only.

FMP is not approved for sample collection, provider calls, account creation, credential use, runtime ingestion, Decision Engine use, Reporting use, or Telegram use.

## 5. Finnhub Summary

Finnhub was reviewed as a second API-oriented market-data provider candidate.

The Finnhub combined source review established that Finnhub public documentation appears relevant to recommendation trends, price targets, and EPS estimate-style endpoint families.

The Finnhub path showed that a combined review can reduce some documentation overhead compared with the FMP path.

However, the Finnhub path still began to recreate the same micro-step pattern through proposal, governance note, protocol, execution note, and pre-execution checklist documents.

Finnhub execution-readiness also remains blocked by unresolved controls.

Key unresolved Finnhub blockers include:

- complete terms-of-use compatibility;
- limited manual research permission;
- raw-value storage rights;
- normalized-value storage rights;
- screenshot permission;
- redistribution or publication rights;
- rate-limit constraints;
- subscription or plan requirements;
- endpoint access permission;
- account and API-key approval;
- local secrets handling approval;
- exact ticker sample;
- exact endpoint families;
- final qualitative-observation checklist;
- execution approval.

Finnhub remains candidate-only.

Finnhub is not approved for sample collection, provider calls, account creation, API-key creation, credential use, runtime ingestion, Decision Engine use, Reporting use, or Telegram use.

## 6. Provider Comparison Summary

| Area | Financial Modeling Prep | Finnhub | Strategy implication |
|---|---|---|---|
| Candidate type | API-oriented provider | API-oriented provider | Both require access and credential governance before execution. |
| Analyst expectations relevance | Candidate endpoint families identified through documentation review. | Candidate endpoint families identified through documentation review. | Both remain potentially relevant but unapproved. |
| Terms clarity | Unresolved. | Unresolved. | Neither can proceed to execution without terms review. |
| Access permission | Unresolved. | Unresolved. | Documentation visibility is not permission to collect data. |
| Account/API-key need | Unresolved / access-dependent. | Unresolved / access-dependent. | Both introduce credential governance before sample review. |
| Storage rights | Unresolved. | Unresolved. | Raw and normalized value storage remain prohibited. |
| Screenshot rights | Unresolved. | Unresolved. | Screenshots remain prohibited. |
| Point-in-time support | Unresolved. | Unresolved. | Neither is approved for historical validation. |
| Runtime integration | Not approved. | Not approved. | No implementation path exists. |
| Decision Engine integration | Not approved. | Not approved. | No allocation authority exists. |
| Reporting or Telegram use | Not approved. | Not approved. | No communication output use exists. |

The comparison shows that API-oriented analyst expectations providers create substantial governance overhead before even a small field-availability sample can be considered.

## 7. Strategy Decision

Strategy decision: stop the current provider microflow and keep analyst expectations provider execution blocked.

The project should not continue producing separate micro-documents for individual provider terms reviews, account reviews, execution notes, and checklists unless a later strategic decision explicitly resumes provider evaluation.

The current provider path should be consolidated as follows:

- FMP: deferred;
- Finnhub: deferred;
- API-provider sample execution: blocked;
- analyst expectations source access: not approved;
- analyst expectations runtime integration: not approved;
- analyst expectations Decision Engine integration: not approved;
- analyst expectations Reporting or Telegram use: not approved.

This decision is not a permanent rejection of analyst expectations.

It is a governance pause on provider execution and micro-documentation overhead.

## 8. Rationale for Stopping the Microflow

Continuing the provider microflow now would likely produce additional documentation without resolving the core blockers.

The repeated blocker pattern is clear:

- terms and licensing remain unresolved;
- manual research permission remains unresolved;
- credentials or API-key governance remains unresolved;
- storage rights remain unresolved;
- point-in-time support remains unresolved;
- execution remains blocked;
- no runtime integration is allowed.

The project should avoid spending additional governance effort on provider-level execution preparation until there is a clearer strategic reason to do so.

## 9. Recommended Provider Strategy

The recommended strategy is to keep analyst expectations as backlog-only research for now.

The project should prioritize more immediate source-data foundations first, especially approved real fundamental data source work.

Analyst expectations provider research may be resumed later only if one of the following conditions is met:

1. a provider with clear licensing, field definitions, and point-in-time support is identified;
2. a formal decision is made to budget for a licensed provider review;
3. fundamentals/source-data governance is mature enough to justify analyst expectations as a next research layer;
4. a later sprint explicitly reopens analyst expectations provider evaluation with combined-document rules to prevent microflow overhead.

## 10. Future Documentation Rule

If analyst expectations provider research is resumed, future provider reviews should use combined documents only.

A future provider review should combine at least:

- source-specific due diligence;
- terms and access review;
- account and credential governance;
- execution-readiness conclusion;
- backlog impact assessment.

Separate micro-documents should be avoided unless a specific governance risk requires separation.

Any future provider review must continue to prohibit:

- source value collection;
- API calls;
- account creation;
- credential creation;
- scraping;
- screenshots;
- CSV files;
- runtime integration;
- Decision Engine use;
- Reporting use;
- Telegram use.

## 11. Explicitly Not Approved

This strategy closure does not approve:

- Financial Modeling Prep as an approved project source;
- Finnhub as an approved project source;
- any other analyst expectations provider;
- account creation;
- API-key creation;
- credential creation;
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

## 12. Backlog Impact Assessment

Existing backlog item `BL-0018` remains sufficient.

This document closes the current analyst expectations provider microflow and keeps provider execution blocked while preserving analyst expectations as a future research path.

It does not identify additional deferred work beyond the existing governed analyst expectations research path.

Backlog impact assessment:
- No new backlog items identified.

## 13. Recommended Next Step

Recommended next step: stop analyst expectations provider execution work for now and return focus to higher-priority source-data foundations.

The most relevant next governance action is to continue work on approved real fundamental data sourcing before reopening analyst expectations provider execution.

If analyst expectations are revisited later, the next document should be a combined provider strategy restart note rather than another provider-specific micro-step.
