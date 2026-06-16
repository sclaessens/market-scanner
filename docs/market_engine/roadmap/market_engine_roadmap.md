# Market Engine Roadmap

Owner role: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE ROADMAP AFTER ME-RM01

## Purpose

This roadmap preserves the Market Engine sprint sequence after ME-RR02 and records the insertion of Setup Detection as a required future layer before Portfolio Review.

ME-RM01 is documentation-only. It does not change runtime code, tests, providers, data files, portfolio files, Telegram/reporting files, or Decision Engine behavior.

## Completed Chain

Completed job-scoped chain:

| Sprint | Job family | Status |
|---|---|---|
| ME-SR01 | Source Refresh | Completed |
| ME-SC01 | Source Context | Completed |
| ME-SC02 | Source Context | Completed |
| ME-FO01 | Fundamental Observations | Completed |
| ME-FO02 | Fundamental Observations | Completed |
| ME-DO01 | Derived Observations | Completed |
| ME-AR01 | Analysis Review | Completed |
| ME-AR02 | Analysis Review | Completed |
| ME-RR01 | Recommendation Review | Completed |
| ME-RR02 | Recommendation Review | Completed |

ME-RR02 implemented the first non-actionable SEC CompanyFacts Recommendation Review layer with:

- input contract: `sec-companyfacts-analysis-review-v1`;
- output contract: `sec-companyfacts-recommendation-review-v1`;
- module: `src/market_engine/recommendation_review/`;
- tests: `tests/market_engine/recommendation_review/`;
- audit: `docs/market_engine/audits/me_rr02_recommendation_review_implementation_audit.md`.

## Architectural Chain

Current target architecture:

```text
Source Refresh / raw snapshots
→ Source Context
→ Fundamental Observations
→ Derived Observations
→ Setup Detection
→ Analysis Review
→ Recommendation Review
→ Portfolio Review
→ Decision Engine handoff / action authority
→ Delivery / reporting
```

Decision Engine remains the only action/allocation authority.

## Setup Detection Insertion

Analysis Review and Recommendation Review already exist before Setup Detection was formally added.

This does not invalidate completed ME-AR01, ME-AR02, ME-RR01, or ME-RR02 work.

ME-RM01 records Setup Detection as a missing future layer that must be inserted before Portfolio Review. Later Analysis Review and Recommendation Review sprints can extend existing completed layers to consume Setup Detection output.

Insertion reason:

Setup Detection is required so Market Engine can detect patterns/setups from Fundamental Observations and Derived Observations before downstream review layers attempt portfolio review or decision handoff. Without this layer, the project risks skipping a required interpretive layer and jumping too quickly from Recommendation Review to Portfolio Review.

## Recommended Next Sprint

### ME-SD01 — Define Setup Detection contract

Status: RECOMMENDED NEXT

Job family: Setup Detection

Goal: Define the contract for detecting patterns/setups from Fundamental Observations and Derived Observations.

ME-SD01 must be documentation-only unless explicitly re-scoped.

Required future input contracts:

- `sec-companyfacts-fundamental-observations-v1`;
- `sec-companyfacts-derived-cash-generation-observations-v1`.

Required future output contract:

- `sec-companyfacts-setup-detection-v1`.

## Planned Future Sprint Sequence

| Sequence | Sprint | Job family | Status | Purpose |
|---|---|---|---|---|
| 1 | ME-SD01 | Setup Detection | Recommended next | Define Setup Detection contract |
| 2 | ME-SD02 | Setup Detection | Planned future | Implement first Setup Detection layer |
| 3 | ME-AR03 | Analysis Review | Planned future | Extend Analysis Review contract for Setup Detection input |
| 4 | ME-AR04 | Analysis Review | Planned future | Implement Analysis Review consumption of Setup Detection |
| 5 | ME-RR03 | Recommendation Review | Planned future | Extend Recommendation Review contract for Setup Detection-aware Analysis Review |
| 6 | ME-RR04 | Recommendation Review | Planned future | Implement Setup Detection-aware Recommendation Review behavior |
| 7 | ME-PR01 | Portfolio Review | Planned future | Define Portfolio Review contract from Recommendation Review |
| 8 | ME-PR02 | Portfolio Review | Planned future | Implement Portfolio Review |
| 9 | ME-DE01 | Decision Engine handoff | Planned future | Define Decision Engine handoff contract |
| 10 | ME-DE02 | Decision Engine handoff | Planned future | Implement controlled Decision Engine handoff |
| 11 | ME-DL01 | Delivery / Reporting | Planned future | Define Delivery / Reporting contract |
| 12 | ME-DL02 | Delivery / Reporting | Planned future | Implement controlled Delivery / Reporting output |

## Planned Sprint Details

### ME-SD02 — Implement first Setup Detection layer

Implement the first non-actionable setup detection builder from approved observation inputs using local synthetic tests only. No live providers, portfolio mutation, Decision Engine behavior, or BUY / SELL / HOLD action semantics.

### ME-AR03 — Extend Analysis Review contract for Setup Detection input

Define how Analysis Review can consume Setup Detection output without recommendation authority.

### ME-AR04 — Implement Analysis Review consumption of Setup Detection

Implement Analysis Review support for Setup Detection input using local synthetic tests only.

### ME-RR03 — Extend Recommendation Review contract for Setup Detection-aware Analysis Review

Define how Recommendation Review consumes Setup Detection-aware Analysis Review while remaining non-actionable.

### ME-RR04 — Implement Setup Detection-aware Recommendation Review behavior

Implement Setup Detection-aware Recommendation Review behavior without action authority.

### ME-PR01 — Define Portfolio Review contract from Recommendation Review

Define Portfolio Review after Setup Detection-aware Recommendation Review exists. This sprint must remain documentation-only unless explicitly re-scoped and must not introduce execution authority.

### ME-PR02 — Implement Portfolio Review

Implement Portfolio Review after the contract is defined. It must not mutate portfolio state or execute actions.

### ME-DE01 — Define Decision Engine handoff contract

Define the boundary between Market Engine review output and actual decision/action authority.

### ME-DE02 — Implement controlled Decision Engine handoff

Implement controlled handoff according to ME-DE01 while preserving Decision Engine as the only action/allocation authority.

### ME-DL01 — Define Delivery / Reporting contract

Define how approved outputs may be delivered or reported.

### ME-DL02 — Implement controlled Delivery / Reporting output

Implement delivery/reporting only after contract and authority boundaries are defined.

## Possible Inserted Sprints

Possible inserted sprints are allowed only when a real problem, blocker, architectural gap, governance risk, test gap, data-quality issue, or newly discovered dependency requires insertion ahead of the planned sequence.

When such a sprint is inserted:

- the insertion reason must be documented in this roadmap;
- the insertion reason must be documented in `docs/market_engine/backlog/market_engine_backlog.md`;
- completed sprint outcomes must be preserved;
- the planned sequence must be updated rather than left ambiguous.

## Boundary Notes

Setup Detection, Analysis Review, Recommendation Review, Portfolio Review, Decision Engine handoff, and Delivery / Reporting must remain separate job families.

No future sprint may skip directly from Recommendation Review to Portfolio Review unless Setup Detection is explicitly deferred with documented governance approval.
