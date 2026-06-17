# Market Engine Roadmap

Owner role: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE ROADMAP AFTER ME-DE02

## Purpose

This roadmap preserves the Market Engine sprint sequence after ME-DE02 and records Delivery / Reporting contract definition as the next active planning step.

ME-SD02 implemented the first non-actionable Setup Detection runtime layer.

ME-AR03 extended the Analysis Review contract so ME-AR04 could consume Setup Detection output.

ME-AR04 implemented Analysis Review consumption of Setup Detection output while preserving the existing `sec-companyfacts-analysis-review-v1` contract.

ME-RR03 extended the Recommendation Review contract so ME-RR04 could consume Setup Detection-aware Analysis Review output while preserving Recommendation Review as downstream, non-actionable, provenance-preserving, missing-data-aware, and numeric-zero-safe.

ME-RR04 implemented Recommendation Review consumption of Setup Detection-aware Analysis Review output while preserving the existing `sec-companyfacts-recommendation-review-v1` contract.

ME-PR01 defined the Portfolio Review contract from Setup Detection-aware Recommendation Review output while preserving Portfolio Review as non-actionable, explicit-portfolio-context-dependent, missing-data-aware, stale-data-aware, numeric-zero-safe, provenance-preserving, and upstream of Decision Engine handoff.

ME-PR02 implemented Portfolio Review while preserving Portfolio Review as non-actionable, explicit-portfolio-context-dependent, missing-data-aware, stale-data-aware, numeric-zero-safe, provenance-preserving, and upstream of Decision Engine handoff.

ME-DE01 defined the Decision Engine handoff contract downstream of Portfolio Review while preserving the Decision Engine as the only future action and allocation authority.

ME-DE02 implemented controlled Decision Engine handoff-readiness payload construction while preserving Decision Engine as the only future action and allocation authority.

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
| ME-RR04 | Recommendation Review    | Completed |
| ME-PR01 | Portfolio Review         | Completed |
| ME-PR02 | Portfolio Review         | Completed |
| ME-DE01 | Decision Engine handoff  | Completed |
| ME-DE02 | Decision Engine handoff  | Completed |

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

ME-RR04 implemented Recommendation Review consumption of Setup Detection-aware Analysis Review with:

* output contract preserved: `sec-companyfacts-recommendation-review-v1`;
* module: `src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py`;
* tests: `tests/market_engine/recommendation_review/test_sec_companyfacts_recommendation_review.py`;
* implementation documentation: `docs/market_engine/recommendation_review/me_rr04_setup_detection_aware_recommendation_review_implementation.md`;
* audit: `docs/market_engine/audits/me_rr04_setup_detection_aware_recommendation_review_implementation_audit.md`.

ME-PR01 defined the Portfolio Review contract with:

* approved input contract: `sec-companyfacts-recommendation-review-v1`;
* approved portfolio context input family: `market-engine-portfolio-context-v1`;
* recommended output contract: `sec-companyfacts-portfolio-review-v1`;
* contract document: `docs/market_engine/portfolio_review/me_pr01_portfolio_review_contract.md`;
* audit: `docs/market_engine/audits/me_pr01_portfolio_review_contract_audit.md`;
* implementation sprint: `ME-PR02 — Implement Portfolio Review`.

ME-PR01 defined that Portfolio Review requires explicitly supplied portfolio context and must preserve Recommendation Review provenance, Setup Detection-aware provenance when present, missing portfolio-context data, stale portfolio-context data, and numeric-zero semantics.

ME-PR01 preserved Portfolio Review as a non-actionable review layer and prevented Python code, tests, runtime behavior, provider calls, broker calls, data writes, generated artifacts, portfolio mutation, watchlist mutation, Telegram, reporting, delivery, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation execution, target weights, order generation, position sizing instructions, ranking, scoring, conviction, urgency, and tradeability authority.

ME-PR02 implemented Portfolio Review with:

* output contract: `sec-companyfacts-portfolio-review-v1`;
* approved input contract: `sec-companyfacts-recommendation-review-v1`;
* approved portfolio context input family: `market-engine-portfolio-context-v1`;
* module: `src/market_engine/portfolio_review/sec_companyfacts_portfolio_review.py`;
* tests: `tests/market_engine/portfolio_review/test_sec_companyfacts_portfolio_review.py`;
* implementation documentation: `docs/market_engine/portfolio_review/me_pr02_portfolio_review_implementation.md`;
* audit: `docs/market_engine/audits/me_pr02_portfolio_review_implementation_audit.md`.

ME-PR02 preserved Recommendation Review provenance, Setup Detection-aware provenance when present, portfolio-context provenance, missing and stale portfolio-context markers, numeric-zero semantics, and non-actionable review boundaries.

ME-DE01 defined the Decision Engine handoff contract with:

* approved upstream input: `sec-companyfacts-portfolio-review-v1`;
* required portfolio-context family: `market-engine-portfolio-context-v1`;
* future handoff payload: `market-engine-decision-engine-handoff-v1`;
* contract document: `docs/market_engine/decision_engine/me_de01_decision_engine_handoff_contract.md`;
* audit: `docs/market_engine/audits/me_de01_decision_engine_handoff_contract_audit.md`;
* implementation sprint: `ME-DE02 — Implement controlled Decision Engine handoff`.

ME-DE01 defined Portfolio Review eligibility, blocked handoff states, fail-closed rules, numeric-zero preservation, provenance requirements, prohibited payload fields, and ME-DE02 implementation requirements.

ME-DE01 preserved Decision Engine as the only future action and allocation authority and did not introduce runtime behavior, tests, provider calls, broker calls, data writes, portfolio mutation, reporting, Telegram, delivery, BUY / SELL / HOLD action semantics, allocation, ranking, scoring, urgency, conviction, tradeability, order generation, or execution instructions.

ME-DE02 implemented controlled Decision Engine handoff with:

* output contract: `market-engine-decision-engine-handoff-v1`;
* approved input contract: `sec-companyfacts-portfolio-review-v1`;
* module: `src/market_engine/decision_engine_handoff/sec_companyfacts_handoff.py`;
* tests: `tests/market_engine/decision_engine_handoff/test_sec_companyfacts_handoff.py`;
* implementation documentation: `docs/market_engine/decision_engine/me_de02_decision_engine_handoff_implementation.md`;
* audit: `docs/market_engine/audits/me_de02_decision_engine_handoff_implementation_audit.md`.

ME-DE02 preserves Portfolio Review, portfolio-context, Recommendation Review, Analysis Review, Setup Detection-aware, missing-data, stale-data, and numeric-zero evidence.

ME-DE02 emits only handoff-readiness states and blocked reasons. It does not introduce Decision Engine decisions, trade instructions, allocation, target weights, order generation, position sizing, ranking, scoring, urgency, conviction, tradeability, execution, provider calls, broker calls, portfolio mutation, reporting, Telegram, delivery, or production data behavior.

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

ME-RR04 implemented Recommendation Review consumption of Setup Detection-aware Analysis Review output.

ME-PR01 defined Portfolio Review only after Setup Detection-aware Recommendation Review existed.

ME-PR02 implemented Portfolio Review only after the ME-PR01 contract was defined.

Insertion reason:

Setup Detection is required so Market Engine can detect patterns/setups from Fundamental Observations and Derived Observations before downstream review layers attempt portfolio review or decision handoff. Without this layer, the project risks skipping a required interpretive layer and jumping too quickly from Recommendation Review to Portfolio Review.

## Recommended Next Sprint

### ME-DL01 — Define Delivery / Reporting contract

Status: RECOMMENDED NEXT

Job family: Delivery / Reporting

Goal: Define how approved outputs may be delivered or reported after Decision Engine handoff boundaries exist.

Scope: Documentation-only contract sprint unless explicitly re-scoped.

ME-DL01 must define approved upstream input requirements, delivery eligibility, reporting eligibility, Telegram/reporting boundaries, user-facing output contract, audit and traceability requirements, fail-closed delivery rules, and ME-DL02 implementation requirements.

ME-DL01 must not introduce runtime behavior, provider calls, broker calls, portfolio mutation, Decision Engine behavior, trade instructions, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability authority.

## Planned Future Sprint Sequence

| Sequence | Sprint  | Job family              | Status           | Purpose                                                        |
| -------- | ------- | ----------------------- | ---------------- | -------------------------------------------------------------- |
| 1        | ME-DL01 | Delivery / Reporting    | Recommended next | Define Delivery / Reporting contract             |
| 2        | ME-DL02 | Delivery / Reporting    | Planned future   | Implement controlled Delivery / Reporting output |

## Planned Sprint Details

### ME-DL01 — Define Delivery / Reporting contract

Define how approved outputs may be delivered or reported.

ME-DL01 must not introduce delivery behavior beyond contract definition.

ME-DL01 must preserve Decision Engine handoff authority boundaries and must define fail-closed delivery/reporting rules before any delivery runtime exists.

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
