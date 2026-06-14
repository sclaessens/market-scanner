# Financial Modeling Prep Account and Access Governance Design

## 1. Status and Scope

This document is a documentation-only governance design for possible future Financial Modeling Prep account access and protected credential handling in the analyst expectations research path.

It follows the existing analyst expectations research documents and backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy`.

This document defines a possible future governance model only.

This document does not implement code, tests, CSV files, generated artifacts, reports, workflows, provider integration, provider/API calls, scraping, account creation, credential creation, runtime orchestration, daily ingestion, backtesting code, Reporting changes, Telegram changes, scanner changes, Decision Engine changes, portfolio files, watchlist files, fundamentals files, or runtime behavior changes.

No sprint is closed or certified complete by this document.

Financial Modeling Prep is not approved as a project source by this document.

No account may be created by this document.

No credential may be created by this document.

No sample values are viewed or collected by this document.

No provider access is authorized by this document.

## 2. Purpose

The purpose of this document is to define the governance required before the project could consider creating or using Financial Modeling Prep access credentials for a later approved limited sample review.

This design covers:

- whether account access could be governance-compatible later;
- where credentials must never be stored;
- local protected-credential handling rules;
- possible future GitHub secret boundary rules;
- exposure, rotation, and revocation expectations;
- preconditions before any account or credential can be created;
- continued sample-execution blocking status.

This design does not approve account creation, credential creation, endpoint access, or sample execution.

## 3. Research-Only Boundary

Analyst expectations remain research-only.

Analyst expectations must not become buy/sell advice.

Any future FMP access must not create or imply ranking authority, scoring authority, allocation authority, tradeability, urgency, conviction, eligibility, hidden filtering, Reporting recommendations, Telegram recommendations, or Decision Engine bypass.

The only possible future use case under this research path is a tightly controlled field-availability and source-governance review.

Future Decision Engine use requires separate governance, separate approval, explicit design, tests, and audit controls.

## 4. Current Approval Status

| Item | Current status | Notes |
|---|---|---|
| FMP account creation | Not approved | This document does not authorize account creation. |
| FMP credential creation | Not approved | This document does not authorize credential generation. |
| FMP endpoint calls | Not approved | This document does not authorize provider calls. |
| FMP sample execution | Blocked | Prior execution approval required revision before execution. |
| FMP source approval | Not approved | FMP remains a candidate only. |
| FMP runtime integration | Not approved | No runtime path may use FMP. |
| FMP GitHub Actions use | Not approved | No workflow use is allowed. |
| FMP Decision Engine use | Not approved | No decision authority is authorized. |
| FMP Reporting or Telegram use | Not approved | No communication output use is authorized. |

## 5. Preconditions Before Any Account or Credential Can Be Created

Before any account or credential can be created, the project must document all of the following:

- complete terms-of-use review;
- confirmation that limited manual research review is permitted;
- confirmation that endpoint access is permitted;
- confirmation of subscription or plan requirements;
- review of rate-limit constraints;
- raw-value storage policy, defaulting to prohibited;
- normalized-value storage policy, defaulting to prohibited;
- screenshot policy, defaulting to prohibited;
- redistribution and publication policy, defaulting to prohibited;
- local protected-credential handling process;
- exposure response process;
- rotation or revocation process;
- reviewer role;
- review date;
- exact ticker list;
- exact endpoint families;
- allowed qualitative observations;
- new execution approval after all controls are resolved.

If any precondition remains unresolved, account creation and credential creation must remain prohibited.

## 6. Places Where Credentials Must Never Appear

Financial Modeling Prep credentials must never appear in:

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
- Telegram messages;
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

If credentials appear in any prohibited location, they must be treated as exposed.

## 7. Local Protected-Credential Handling Design

If a later governance note approves limited local credential use, the following rules are mandatory:

| Rule | Requirement |
|---|---|
| Local only | Use only in a local environment approved for the review. |
| No repository storage | Do not commit, stage, paste, or document the credential. |
| No saved output | Do not save logs or screenshots containing the credential. |
| No runtime dependency | Do not make the credential part of any runtime process. |
| No workflow dependency | Do not use it in GitHub Actions unless separately governed. |
| Temporary use | Limit use to the approved review session. |
| Exposure response | Define revocation or rotation before use. |

This design does not approve local credential use. It only defines conditions for a later approval decision.

## 8. Future GitHub Secrets Boundary

GitHub Secrets are not approved for FMP use by this document.

A future GitHub Secrets design would be required only if a later governed implementation proposes automated ingestion or CI-based access.

That future design must define why GitHub-hosted access is necessary, which workflow would use the secret, who can manage it, how logs avoid exposure, how rate limits are enforced, how failures are handled, how access is rotated or revoked, and how use remains separate from Decision Engine authority unless separately approved.

No such workflow or GitHub Secret use is approved here.

## 9. Exposure Response Requirements

If a credential is ever approved later and exposure is suspected, the response must be:

1. stop using the credential immediately;
2. revoke or rotate it through the provider account;
3. remove it from local files or uncommitted artifacts;
4. if committed, treat repository history as contaminated and follow a separate incident response;
5. review logs, screenshots, pull requests, issues, and documents for exposure;
6. document the incident without repeating the credential value;
7. confirm that no runtime path depends on the exposed credential;
8. confirm that no collected values influence decisions.

## 10. Current Blocking Status

Execution remains blocked.

| Control area | Status | Reason |
|---|---|---|
| Terms-of-use review | `UNRESOLVED` | Complete terms compatibility has not been documented. |
| Manual research permission | `UNRESOLVED` | Permission has not been confirmed. |
| Credential creation | `UNRESOLVED` | No account or credential approval exists. |
| Local protected handling | `DESIGNED / NOT APPROVED` | This document defines rules but does not approve use. |
| GitHub Secrets handling | `NOT APPROVED` | Not needed unless future automation is proposed. |
| Rate limits | `UNRESOLVED` | Provider limits are not reviewed. |
| Plan requirements | `UNRESOLVED` | Subscription constraints are not confirmed. |
| Exact ticker list | `UNRESOLVED` | No sample list is approved. |
| Exact endpoint families | `UNRESOLVED` | No execution endpoint list is approved. |
| Qualitative observations | `PARTIAL` | Candidate values exist in protocol but are not execution-approved. |
| Sample execution | `BLOCKED` | Mandatory controls remain unresolved. |

## 11. Explicitly Not Approved

This governance design does not approve Financial Modeling Prep as an approved project source, account creation, credential creation, endpoint access, sample execution, runtime data collection, manual ticker-by-ticker analyst data collection, sample value viewing, sample value collection, sample value storage, screenshots, scraping, provider calls, automated ingestion, CSV creation, historical backtesting code, source-derived scoring, source-derived ranking, source-derived tradeability, source-derived conviction, source-derived urgency, source-derived eligibility, hidden filtering, Decision Engine integration, Reporting recommendations, or Telegram recommendations.

## 12. Backlog Impact Assessment

Existing backlog item `BL-0018` remains sufficient.

This document designs possible future account, access, and protected-credential governance for the FMP analyst expectations research path while keeping execution blocked.

It does not identify additional deferred work beyond the existing governed analyst expectations research path.

Backlog impact assessment:
- No new backlog items identified.

## 13. Recommended Next Step

The recommended next step is a documentation-only FMP execution-readiness reconciliation note.

That note should reconcile the current status of terms/access review, account/access governance, sample protocol, pre-execution checklist, and remaining unresolved blockers.

The note should decide whether the project should continue resolving FMP blockers, defer FMP, or evaluate a second candidate source instead.

No account, credential, endpoint access, or sample execution is authorized until a later approval note explicitly allows it.
