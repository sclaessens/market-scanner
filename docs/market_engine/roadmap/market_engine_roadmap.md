# Market Engine Roadmap

Owner role: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE ROADMAP AFTER ME-RR03

## Purpose

This roadmap preserves the Market Engine sprint sequence after ME-RR03 and records Setup Detection-aware Recommendation Review as the next active implementation step.

ME-SD02 implemented the first non-actionable Setup Detection runtime layer.

ME-AR03 extended the Analysis Review contract so ME-AR04 could consume Setup Detection output.

ME-AR04 implemented Analysis Review consumption of Setup Detection output while preserving the existing `sec-companyfacts-analysis-review-v1` contract.

ME-RR03 extended the Recommendation Review contract so ME-RR04 can consume Setup Detection-aware Analysis Review output while preserving Recommendation Review as downstream, non-actionable, provenance-preserving, missing-data-aware, and numeric-zero-safe.

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
| ME-AR03 | Analysis Review          | Completed |
| ME-AR04 | Analysis Review          | Completed |
| ME-RR03 | Recommendation Review    | Completed |

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

ME-AR03 extended the Analysis Review contract with:

* approved input contract: `sec-companyfacts-setup-detection-v1`;
* contract document: `docs/market_engine/analysis_review/me_ar03_setup_detection_input_contract.md`;
* audit: `docs/market_engine/audits/me_ar03_setup_detection_input_contract_audit.md`;
* implementation sprint: `ME-AR04 — Implement Analysis Review consumption of Setup Detection`.

ME-AR04 implemented Analysis Review consumption of Setup Detection with:

* output contract preserved: `sec-companyfacts-analysis-review-v1`;
* module: `src/market_engine/analysis_review/sec_companyfacts_analysis_review.py`;
* tests: `tests/market_engine/analysis_review/test_sec_companyfacts_analysis_review.py`;
* implementation documentation: `docs/market_engine/analysis_review/me_ar04_analysis_review_setup_detection_implementation.md`;
* audit: `docs/market_engine/audits/me_ar04_analysis_review_setup_detection_implementation_audit.md`.

ME-RR03 extended the Recommendation Review contract with:

* approved input contract preserved: `sec-companyfacts-analysis-review-v1`;
* contract document: `docs/market_engine/recommendation_review/me_rr03_setup_detection_aware_analysis_review_contract.md`;
* audit: `docs/market_engine/audits/me_rr03_setup_detection_aware_analysis_review_contract_audit.md`;
* implementation sprint: `ME-RR04 — Implement Setup Detection-aware Recommendation Review behavior`.

ME-RR03 defined that Recommendation Review may consume Setup Detection-aware Analysis Review only through validated `sec-companyfacts-analysis-review-v1`.

ME-RR03 preserved Recommendation Review as downstream of Analysis Review and prevented direct Setup Detection consumption, runtime behavior, provider calls, data writes, Portfolio Review behavior, Delivery behavior, Telegram behavior, reporting behavior, Decision Engine behavior, and BUY / SELL / HOLD action semantics.

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

ME-AR04 extended Analysis Review to consume Setup Detection output.

ME-RR03 extended Recommendation Review to consume Setup Detection-aware Analysis Review output.

Insertion reason:

Setup Detection is required so Market Engine can detect patterns/setups from Fundamental Observations and Derived Observations before downstream review layers attempt portfolio review or decision handoff. Without this layer, the project risks skipping a required interpretive layer and jumping too quickly from Recommendation Review to Portfolio Review.

## Recommended Next Sprint

### ME-RR04 — Implement Setup Detection-aware Recommendation Review behavior

Status: RECOMMENDED NEXT

Job family: Recommendation Review

Goal: Implement Setup Detection-aware Recommendation Review behavior.

Scope: Non-actionable Recommendation Review only; no action authority, portfolio mutation, delivery behavior, Telegram, reporting, or Decision Engine behavior.

ME-RR04 must implement Setup Detection-aware Recommendation Review according to the ME-RR03 contract.

ME-RR04 must preserve existing ME-RR02 behavior for non-setup-aware Analysis Review inputs while adding Setup Detection-aware routing, provenance preservation, missing-data preservation, numeric-zero safety, and forbidden-action boundary enforcement.

## Planned Future Sprint Sequence

| Sequence | Sprint  | Job family              | Status           | Purpose                                                        |
| -------- | ------- | ----------------------- | ---------------- | -------------------------------------------------------------- |
| 1        | ME-RR04 | Recommendation Review   | Recommended next | Implement Setup Detection-aware Recommendation Review behavior |
| 2        | ME-PR01 | Portfolio Review        | Planned future   | Define Portfolio Review contract from Recommendation Review    |
| 3        | ME-PR02 | Portfolio Review        | Planned future   | Implement Portfolio Review                                     |
| 4        | ME-DE01 | Decision Engine handoff | Planned future   | Define Decision Engine handoff contract                        |
| 5        | ME-DE02 | Decision Engine handoff | Planned future   | Implement controlled Decision Engine handoff                   |
| 6        | ME-DL01 | Delivery / Reporting    | Planned future   | Define Delivery / Reporting contract                           |
| 7        | ME-DL02 | Delivery / Reporting    | Planned future   | Implement controlled Delivery / Reporting output               |

## Planned Sprint Details

### ME-RR04 — Implement Setup Detection-aware Recommendation Review behavior

Implement Setup Detection-aware Recommendation Review behavior without action authority.

ME-RR04 must consume only validated `sec-companyfacts-analysis-review-v1`.

ME-RR04 must preserve existing ME-RR02 behavior when Setup Detection-aware Analysis Review items are absent.

ME-RR04 must preserve setup-aware provenance, missing setup observations, setup evidence, setup limitations, numeric-zero semantics, and non-actionable boundary markers.

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
