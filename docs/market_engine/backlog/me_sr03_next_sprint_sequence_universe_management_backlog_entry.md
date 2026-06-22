# ME-SR03 Follow-up Backlog Entry - Universe Management Sprint Sequence

Owner roles: Product Owner / Operator / Technical Architect / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR / ME-UNI / ME-RUN / ME-OUT planning

Status: PLANNED SEQUENCE AFTER ME-SR03

## Context

ME-SR03 resolved the ASML and TSM cached-source blockers using existing approved cached source payloads and narrow SEC CompanyFacts field mapping remediation.

The remaining canonical-universe blocker is `HO`, which has no approved local SEC CompanyFacts snapshot in the bounded source bundle.

The operator also clarified that the future minimum scan universe is not the small canonical test universe, but an editable professional swing universe of roughly 250 to 300 liquid leaders and high-quality movers. That universe must support future ticker additions, removals, exclusions, category changes, and unsupported-source classification without requiring Python code changes or hardcoded ticker lists.

## Planning Decision

The next sprint sequence is updated so Market Engine does not jump directly from the small canonical-universe blocker to large-scale scans, reporting, candidate selection, or entry logic.

The project must first establish an editable universe-management layer.

## Required Sprint Sequence

### ME-SR04 - Resolve HO canonical-universe source identity or exclusion decision

Status: NEXT CANDIDATE AFTER ME-SR03

Job family: ME-SR - Source Refresh / Source Coverage

Goal: decide whether `HO` has a valid approved source identity and local cached source path, or whether it must be excluded from default SEC CompanyFacts cached-source execution until a valid source exists.

Expected outcome: the current small canonical SEC CompanyFacts cached-source universe should no longer treat unsupported `HO` as an unresolved runtime blocker.

Scope boundaries: source identity and universe/source governance only. No provider calls, live SEC/EDGAR calls, yfinance calls, broker calls, portfolio/watchlist writes, Telegram/email delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation, target prices, ranking, scoring, conviction, urgency, tradeability, or execution advice.

### ME-UNI04 - Define editable Professional Swing Universe contract

Status: PLANNED AFTER ME-SR04

Job family: ME-UNI - Ticker Universe

Goal: define the editable universe contract for the operator's professional swing universe so tickers can later be added, disabled, removed, categorized, source-classified, and audited without hardcoding ticker membership in Python.

Required contract direction:

* support a stable `universe_id`, format version, description, update metadata, and active/inactive ticker state;
* preserve category or segment metadata where useful;
* support source policy fields such as `sec_companyfacts_preferred`, `foreign_issuer_sec_companyfacts_supported`, `manual_review_only`, `unsupported_sec_companyfacts`, or future approved values;
* support explicit exclusion/deactivation reasons;
* preserve duplicate handling and normalization rules;
* distinguish universe membership from source readiness and investment quality.

Non-goals: no source refresh, no provider calls, no live market data, no scan execution, no recommendation behavior, no ranking/scoring, and no advice semantics.

### ME-UNI05 - Import and normalize Professional Swing Universe seed list

Status: PLANNED AFTER ME-UNI04

Job family: ME-UNI - Ticker Universe

Goal: import the operator-provided professional swing universe seed list into the editable universe format defined by ME-UNI04.

Required behavior:

* ignore comments and section headers in the source list;
* normalize ticker symbols deterministically;
* detect and report duplicates;
* preserve or map categories when supported by the contract;
* mark all imported tickers with explicit active/source-policy defaults;
* avoid treating unsupported source coverage as investment quality.

Non-goals: no provider refresh, no source snapshot creation, no live data calls, no recommendation behavior, no ranking/scoring, and no advice semantics.

### ME-UNI06 - Implement editable universe loader and validation

Status: PLANNED AFTER ME-UNI05

Job family: ME-UNI - Ticker Universe

Goal: implement runtime loading and validation for editable universe configs so future runs can select active tickers from a named universe instead of relying on hardcoded or one-off ticker lists.

Required behavior:

* load the editable universe file;
* validate format version and required fields;
* select active tickers only when requested;
* fail closed on duplicate active symbols, invalid states, ambiguous source policy, or malformed config;
* provide deterministic test coverage;
* preserve local-only/non-actionable boundaries.

Non-goals: no provider refresh, no source snapshot creation, no live data calls, no recommendation behavior, no ranking/scoring, and no advice semantics.

### ME-SR05 - Classify source support for Professional Swing Universe

Status: PLANNED AFTER ME-UNI06

Job family: ME-SR - Source Refresh / Source Coverage

Goal: classify source support for all active tickers in the editable professional swing universe before attempting large-scale cached-source scans.

Required classification examples:

* `supported_cached`;
* `missing_snapshot`;
* `unsupported_sec_companyfacts`;
* `foreign_issuer_needs_mapping`;
* `ambiguous_identity`;
* `manual_review_only`;
* `excluded`.

Non-goals: no fabricated snapshots, no guessed CIKs, no live provider calls unless a later sprint explicitly authorizes a bounded refresh/backfill, no recommendation behavior, no ranking/scoring, and no advice semantics.

### ME-RUN20 - Execute clean supported-universe cached-source scan

Status: PLANNED AFTER ME-SR05

Job family: ME-RUN - Run / orchestration jobs

Goal: execute a local cached-source scan against the currently supported active subset of the editable professional swing universe and produce inspectable local artifacts.

This sprint must remain a local, provider-free, non-production dry-run unless explicitly re-scoped by a future approved sprint.

### ME-OUT01 - Define readable operator report from dry-run artifacts

Status: PLANNED AFTER ME-RUN20

Job family: ME-OUT - Operator Output / Reporting

Goal: define a readable, non-actionable operator report that summarizes dry-run artifact outcomes per ticker without requiring the operator to inspect raw JSON.

Non-goals: no BUY / SELL / HOLD, no ranking, no scores, no target prices, no allocation, and no execution advice.

### ME-CANDIDATE01 - Define non-actionable candidate classification contract

Status: PLANNED AFTER ME-OUT01

Job family: ME-CANDIDATE - Candidate Classification

Goal: define a controlled, non-actionable candidate classification layer, such as `watch`, `reject`, `needs_more_data`, or `potential_candidate`, without granting trading, allocation, target-price, or execution authority.

This sprint must explicitly preserve the boundary between information triage and investment decision authority.

## Insertion Reason

This sequence is inserted before broader scans, ranking, entry analysis, or buy planning because the operator's intended universe is an editable professional swing universe, not a fixed 12-to-13 ticker canonical test universe.

Without ME-UNI04 through ME-UNI06, ticker membership would remain brittle and large-universe scans would either require hardcoded lists or repeated manual edits.

Without ME-SR05, the system would not know which tickers are supported, unsupported, ambiguous, or missing local snapshots.

## Governance Boundary

This backlog update records planning only. It does not authorize provider calls, live data refresh, source snapshot backfill, portfolio/watchlist writes, production delivery, ranking, scoring, target prices, allocation, BUY / SELL / HOLD semantics, tradeability, or execution advice.
