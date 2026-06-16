# Market Engine Roadmap

Owner role: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE ROADMAP AFTER ME-SD02

## Purpose

This roadmap preserves the Market Engine sprint sequence after ME-SD02 and records Setup Detection as an active Market Engine job family between Derived Observations and Analysis Review.

ME-SD02 implemented the first non-actionable Setup Detection runtime layer. It does not change providers, portfolio files, Telegram/reporting files, or Decision Engine behavior.

## Completed Chain

Completed job-scoped chain:

| Sprint  | Job family               | Status    |
| ------- | ------------------------ | --------- |
| ME-SR01 | Source Refresh           | Completed |
| ME-SC01 | Source Context           | Completed |
| ME-SC02 | Source Context           | Completed |
| ME-FO01 | Fundamental Observations | Completed |
| ME-FO02 | Fundamental Observations | Completed |
| ME-DO01 | Derived Observations     | Completed |
| ME-AR01 | Analysis Review          | Completed |
| ME-AR02 | Analysis Review          | Completed |
| ME-RR01 | Recommendation Review    | Completed |
| ME-RR02 | Recommendation Review    | Completed |
| ME-RM01 | Roadmap / Governance     | Completed |
| ME-SD01 | Setup Detection          | Completed |
| ME-SD02 | Setup Detection          | Completed |

ME-RR02 implemented the first non-actionable SEC CompanyFacts Recommendation Review layer with:

* input contract: `sec-companyfacts-analysis-review-v1`;
* output contract: `sec-companyfacts-recommendation-review-v1`;
* module: `src/market_engine/recommendation_review/`;
* tests: `tests/market_engine/recommendation_review/`;
* audit: `docs/market_engine/audits/me_rr02_recommendation_review_implementation_audit.md`.

ME-RM01 created the active Market Engine roadmap, inserted Setup Detection before Portfolio Review, and added the governance rule that future logical next sprints must be preserved in the backlog and roadmap when identified.

ME-SD01 defined the first Setup Detection contract with:

* input contracts:

  * `sec-companyfacts-fundamental-observations-v1`;
  * `sec-companyfacts-derived-cash-generation-observations-v1`;
* output contract: `sec-companyfacts-setup-detection-v1`;
* contract document: `docs/market_engine/setup_detection/me_sd01_setup_detection_contract.md`;
* audit: `docs/market_engine/audits/me_sd01_setup_detection_contract_audit.md`.

ME-SD02 implemented the first Setup Detection layer with:

* output contract: `sec-companyfacts-setup-detection-v1`;
* module: `src/market_engine/setup_detection/`;
* tests: `tests/market_engine/setup_detection/`;
* audit: `docs/market_engine/audits/me_sd02_setup_detection_implementation_audit.md`.

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

## Setup Detection Position

Analysis Review and Recommendation Review already existed before Setup Detection was formally added.

This does not invalidate completed ME-AR01, ME-AR02, ME-RR01, or ME-RR02 work.

ME-RM01 recorded Setup Detection as a missing future layer that must be inserted before Portfolio Review. ME-SD01 then defined the Setup Detection contract.

Later Analysis Review and Recommendation Review sprints can extend existing completed layers to consume Setup Detection output.

Insertion reason:

Setup Detection is required so Market Engine can detect patterns/setups from Fundamental Observations and Derived Observations before downstream review layers attempt portfolio review or decision handoff. Without this layer, the project risks skipping a required interpretive layer and jumping too quickly from Recommendation Review to Portfolio Review.

## Recommended Next Sprint

### ME-AR03 — Extend Analysis Review contract for Setup Detection input

Status: RECOMMENDED NEXT

Job family: Analysis Review

Goal: Define how Analysis Review can consume Setup Detection output without recommendation authority.

ME-AR03 must define how `sec-companyfacts-setup-detection-v1` becomes an approved Analysis Review input, how setup evidence is referenced, how missing-data states are preserved, and how Analysis Review remains non-recommendation and non-actionable.

## Planned Future Sprint Sequence

| Sequence | Sprint  | Job family              | Status           | Purpose                                                                         |
| -------- | ------- | ----------------------- | ---------------- | ------------------------------------------------------------------------------- |
| 1        | ME-AR03 | Analysis Review         | Recommended next | Extend Analysis Review contract for Setup Detection input                       |
| 2        | ME-AR04 | Analysis Review         | Planned future   | Implement Analysis Review consumption of Setup Detection                        |
| 3        | ME-RR03 | Recommendation Review   | Planned future   | Extend Recommendation Review contract for Setup Detection-aware Analysis Review |
| 4        | ME-RR04 | Recommendation Review   | Planned future   | Implement Setup Detection-aware Recommendation Review behavior                  |
| 5        | ME-PR01 | Portfolio Review        | Planned future   | Define Portfolio Review contract from Recommendation Review                     |
| 6        | ME-PR02 | Portfolio Review        | Planned future   | Implement Portfolio Review                                                      |
| 7        | ME-DE01 | Decision Engine handoff | Planned future   | Define Decision Engine handoff contract                                         |
| 8        | ME-DE02 | Decision Engine handoff | Planned future   | Implement controlled Decision Engine handoff                                    |
| 9        | ME-DL01 | Delivery / Reporting    | Planned future   | Define Delivery / Reporting contract                                            |
| 10       | ME-DL02 | Delivery / Reporting    | Planned future   | Implement controlled Delivery / Reporting output                                |

## Planned Sprint Details

### ME-AR03 — Extend Analysis Review contract for Setup Detection input

Define how Analysis Review can consume Setup Detection output without recommendation authority.

ME-AR03 must define how `sec-companyfacts-setup-detection-v1` becomes an approved Analysis Review input, how setup evidence is referenced, how missing-data states are preserved, and how Analysis Review remains non-recommendation and non-actionable.

### ME-AR04 — Implement Analysis Review consumption of Setup Detection

Implement Analysis Review support for Setup Detection input using local synthetic tests only.

ME-AR04 must not introduce Recommendation Review, Portfolio Review, Delivery, Telegram, reporting, Decision Engine behavior, or BUY / SELL / HOLD action semantics.

### ME-RR03 — Extend Recommendation Review contract for Setup Detection-aware Analysis Review

Define how Recommendation Review consumes Setup Detection-aware Analysis Review while remaining non-actionable.

ME-RR03 must preserve Recommendation Review as downstream of Analysis Review and must not introduce action authority.

### ME-RR04 — Implement Setup Detection-aware Recommendation Review behavior

Implement Setup Detection-aware Recommendation Review behavior without action authority.

ME-RR04 must not introduce portfolio mutation, delivery behavior, Telegram, reporting, Decision Engine behavior, allocation, position sizing, ranking, scoring, conviction, urgency, or tradeability authority.

### ME-PR01 — Define Portfolio Review contract from Recommendation Review

Define Portfolio Review after Setup Detection-aware Recommendation Review exists.

This sprint must remain documentation-only unless explicitly re-scoped and must not introduce execution authority.

### ME-PR02 — Implement Portfolio Review

Implement Portfolio Review after the contract is defined.

It must not mutate portfolio state, execute actions, call the Decision Engine, send Telegram, generate delivery output, or emit BUY / SELL / HOLD action semantics.

### ME-DE01 — Define Decision Engine handoff contract

Define the boundary between Market Engine review output and actual decision/action authority.

Decision Engine remains the only action/allocation authority.

### ME-DE02 — Implement controlled Decision Engine handoff

Implement controlled handoff according to ME-DE01 while preserving Decision Engine as the only action/allocation authority.

ME-DE02 must not bypass Portfolio Review, Recommendation Review, Analysis Review, Setup Detection, or authority boundaries.

### ME-DL01 — Define Delivery / Reporting contract

Define how approved outputs may be delivered or reported.

ME-DL01 must not introduce delivery behavior before upstream authority boundaries are defined.

### ME-DL02 — Implement controlled Delivery / Reporting output

Implement delivery/reporting only after contract and authority boundaries are defined.

ME-DL02 must not bypass Recommendation Review, Portfolio Review, or Decision Engine handoff authority boundaries.

## Possible Inserted Sprints

Possible inserted sprints are allowed only when a real problem, blocker, architectural gap, governance risk, test gap, data-quality issue, or newly discovered dependency requires insertion ahead of the planned sequence.

When such a sprint is inserted:

* the insertion reason must be documented in this roadmap;
* the insertion reason must be documented in `docs/market_engine/backlog/market_engine_backlog.md`;
* completed sprint outcomes must be preserved;
* the planned sequence must be updated rather than left ambiguous.

## Boundary Notes

Setup Detection, Analysis Review, Recommendation Review, Portfolio Review, Decision Engine handoff, and Delivery / Reporting must remain separate job families.

No future sprint may skip directly from Recommendation Review to Portfolio Review unless Setup Detection is explicitly deferred with documented governance approval.

No future sprint may skip from Setup Detection directly to Portfolio Review without first updating Analysis Review and Recommendation Review contracts, unless a documented governance-approved insertion or deferral reason is added to both roadmap and backlog.
