# ME-PR01 — Backlog update

## Status

COMPLETED BY ME-PR01

## Sprint

ME-PR01 — Define Portfolio Review contract from Recommendation Review

## Job family

ME-PR — Portfolio Review jobs

## Purpose

This backlog update records that ME-PR01 completed the Portfolio Review contract definition after Setup Detection-aware Recommendation Review existed.

This file preserves the ME-PR01 backlog outcome without modifying runtime code, tests, providers, data, portfolio state, reporting, Telegram, Delivery, or Decision Engine behavior.

## Completed sprint outcome

ME-PR01 defined the first Portfolio Review contract from approved Recommendation Review output.

Implemented contract document:

* `docs/market_engine/portfolio_review/me_pr01_portfolio_review_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_pr01_portfolio_review_contract_audit.md`

Updated roadmap:

* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Approved input contract

Portfolio Review may consume only validated Recommendation Review output using:

```text
sec-companyfacts-recommendation-review-v1
```

The input may include Setup Detection-aware provenance preserved through ME-RR04.

## Approved portfolio-context input family

ME-PR01 defined the first portfolio-context input family:

```text
market-engine-portfolio-context-v1
```

Portfolio context must be explicitly supplied.

Portfolio Review must not infer holdings, exposure, concentration, or policy constraints from broker exports, reports, watchlists, generated outputs, old runtime data, or provider/source files unless a later sprint explicitly approves that source.

## Required future output contract

ME-PR01 defined the future Portfolio Review output contract:

```text
sec-companyfacts-portfolio-review-v1
```

Recommended future output path:

```text
data/market_engine/portfolio_reviews/<portfolio_review_run_id>/<ticker>/portfolio_review.json
```

ME-PR01 does not create or write this path.

## Contract boundaries defined

ME-PR01 defined:

* Portfolio Review job-family boundary;
* approved Recommendation Review input requirements;
* explicit portfolio-context requirements;
* position review semantics;
* exposure review semantics;
* concentration review semantics;
* portfolio-fit review semantics;
* allowed Portfolio Review categories;
* allowed Portfolio Review states;
* missing-data rules;
* stale-data rules;
* numeric-zero rules;
* Recommendation Review provenance preservation;
* Setup Detection-aware provenance preservation when present upstream;
* authority boundary between Portfolio Review and Decision Engine;
* ME-PR02 implementation requirements.

## Explicit non-scope preserved

ME-PR01 did not introduce:

* Python code;
* tests;
* runtime behavior;
* provider calls;
* broker calls;
* data writes;
* generated artifacts;
* portfolio mutation;
* watchlist mutation;
* Telegram;
* reporting;
* delivery behavior;
* Decision Engine behavior;
* BUY / SELL / HOLD action semantics;
* allocation execution;
* target weights;
* order generation;
* position sizing instructions;
* ranking;
* scoring;
* conviction;
* urgency;
* tradeability authority.

## Next approved sprint

### ME-PR02 — Implement Portfolio Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Portfolio Review

Status: RECOMMENDED NEXT

Goal: Implement Portfolio Review after the ME-PR01 contract definition.

Scope: Non-actionable Portfolio Review only.

ME-PR02 must consume only validated `sec-companyfacts-recommendation-review-v1` input and explicitly supplied approved portfolio context.

ME-PR02 must emit `sec-companyfacts-portfolio-review-v1`, preserve upstream Recommendation Review provenance, preserve Setup Detection-aware provenance when present, preserve missing/stale portfolio-context markers, preserve numeric-zero semantics, implement approved Portfolio Review categories and states, and remain upstream of Decision Engine handoff.

ME-PR02 must not introduce BUY / SELL / HOLD action semantics, allocation execution, target weights, order generation, position sizing instructions, ranking, scoring, conviction, urgency, tradeability authority, portfolio mutation, watchlist mutation, broker integration, Telegram, reporting, delivery, or Decision Engine behavior.

## Planned future sequence after ME-PR01

| Sequence | Sprint  | Job family              | Status           | Purpose |
| -------- | ------- | ----------------------- | ---------------- | ------- |
| 1 | ME-PR02 | Portfolio Review | Recommended next | Implement Portfolio Review |
| 2 | ME-DE01 | Decision Engine handoff | Planned future | Define Decision Engine handoff contract |
| 3 | ME-DE02 | Decision Engine handoff | Planned future | Implement controlled Decision Engine handoff |
| 4 | ME-DL01 | Delivery / Reporting | Planned future | Define Delivery / Reporting contract |
| 5 | ME-DL02 | Delivery / Reporting | Planned future | Implement controlled Delivery / Reporting output |
