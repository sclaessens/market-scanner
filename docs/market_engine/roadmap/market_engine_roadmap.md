# Market Engine Roadmap

Owner role: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE ROADMAP AFTER ME-RUN06

## Purpose

This roadmap preserves the Market Engine sprint sequence after ME-RUN06.

ME-SD02 implemented the first non-actionable Setup Detection runtime layer.

ME-AR03 extended the Analysis Review contract so ME-AR04 could consume Setup Detection output.

ME-AR04 implemented Analysis Review consumption of Setup Detection output while preserving the existing `sec-companyfacts-analysis-review-v1` contract.

ME-RR03 extended the Recommendation Review contract so ME-RR04 could consume Setup Detection-aware Analysis Review output while preserving Recommendation Review as downstream, non-actionable, provenance-preserving, missing-data-aware, and numeric-zero-safe.

ME-RR04 implemented Recommendation Review consumption of Setup Detection-aware Analysis Review output while preserving the existing `sec-companyfacts-recommendation-review-v1` contract.

ME-PR01 defined the Portfolio Review contract from Setup Detection-aware Recommendation Review output while preserving Portfolio Review as non-actionable, explicit-portfolio-context-dependent, missing-data-aware, stale-data-aware, numeric-zero-safe, provenance-preserving, and upstream of Decision Engine handoff.

ME-PR02 implemented Portfolio Review while preserving Portfolio Review as non-actionable, explicit-portfolio-context-dependent, missing-data-aware, stale-data-aware, numeric-zero-safe, provenance-preserving, and upstream of Decision Engine handoff.

ME-DE01 defined the Decision Engine handoff contract downstream of Portfolio Review while preserving the Decision Engine as the only future action and allocation authority.

ME-DE02 implemented controlled Decision Engine handoff-readiness payload construction while preserving Decision Engine as the only future action and allocation authority.

ME-DL01 defined the Delivery / Reporting contract downstream of controlled Decision Engine handoff while preserving Delivery / Reporting as non-executing, non-broker-connected, provenance-preserving, blocked-state-preserving, and non-actionable.

ME-DL02 implemented the Delivery / Reporting contract as a non-actionable payload builder while preserving blocked upstream states, missing-data markers, stale-data markers, numeric-zero evidence, and upstream provenance.

ME-RUN05 implemented optional local dry-run artifact persistence while preserving stdout-only dry-run behavior by default and keeping persisted artifacts local, non-production, deterministic, and inspectable.

ME-RUN06 implemented a controlled local fixture/data input execution path while preserving embedded synthetic dry-run behavior, explicit in-memory payload compatibility, and all side-effect and authority boundaries.

## Completed Chain

Completed job-scoped chain:

| Sprint   | Job family               | Status    |
| -------- | ------------------------ | --------- |
| ME-SR01  | Source Refresh           | Completed |
| ME-SC01  | Source Context           | Completed |
| ME-SC02  | Source Context           | Completed |
| ME-FO01  | Fundamental Observations | Completed |
| ME-FO02  | Fundamental Observations | Completed |
| ME-DO01  | Derived Observations     | Completed |
| ME-AR01  | Analysis Review          | Completed |
| ME-AR02  | Analysis Review          | Completed |
| ME-RM01  | Roadmap / Governance     | Completed |
| ME-SD01  | Setup Detection          | Completed |
| ME-SD02  | Setup Detection          | Completed |
| ME-AR03  | Analysis Review          | Completed |
| ME-AR04  | Analysis Review          | Completed |
| ME-RR03  | Recommendation Review    | Completed |
| ME-RR04  | Recommendation Review    | Completed |
| ME-PR01  | Portfolio Review         | Completed |
| ME-PR02  | Portfolio Review         | Completed |
| ME-DE01  | Decision Engine handoff  | Completed |
| ME-DE02  | Decision Engine handoff  | Completed |
| ME-DL01  | Delivery / Reporting     | Completed |
| ME-DL02  | Delivery / Reporting     | Completed |
| ME-RUN05 | Run / orchestration      | Completed |
| ME-RUN06 | Run / orchestration      | Completed |

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

ME-DL01 defined Delivery / Reporting with:

* approved upstream input: `market-engine-decision-engine-handoff-v1`;
* future output contract: `market-engine-delivery-report-v1`;
* contract document: `docs/market_engine/delivery_reporting/me_dl01_delivery_reporting_contract.md`;
* audit: `docs/market_engine/audits/me_dl01_delivery_reporting_contract_audit.md`;
* implementation sprint: `ME-DL02 — Implement Delivery / Reporting contract`.

ME-DL01 defined delivery states, allowed reporting categories, forbidden reporting behavior, presentation rules, blocked/upstream handling, missing-data handling, stale-data handling, numeric-zero safety, provenance preservation, and ME-DL02 implementation requirements.

ME-DL01 did not introduce runtime behavior, tests, provider calls, data writes, report generation, Telegram delivery, email delivery, broker integration, portfolio mutation, watchlist mutation, scheduling, Decision Engine behavior, trade instructions, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability authority.

ME-DL02 implemented Delivery / Reporting with:

* output contract: `market-engine-delivery-report-v1`;
* approved input contract: `market-engine-decision-engine-handoff-v1`;
* module: `src/market_engine/delivery_reporting/sec_companyfacts_delivery_report.py`;
* tests: `tests/market_engine/delivery_reporting/test_sec_companyfacts_delivery_report.py`;
* implementation documentation: `docs/market_engine/delivery_reporting/me_dl02_delivery_reporting_implementation.md`;
* audit: `docs/market_engine/audits/me_dl02_delivery_reporting_implementation_audit.md`.

ME-DL02 emits non-actionable display sections only and preserves upstream blocked states, missing-data markers, stale-data markers, numeric-zero evidence, and provenance.

ME-DL02 did not introduce provider calls, live market data calls, Telegram delivery, email delivery, broker integration, generated reports, portfolio writes, watchlist writes, scheduling, UI behavior, Decision Engine behavior, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, trade instructions, allocation, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability, or execution advice.

ME-RUN05 implemented local dry-run artifact persistence with:

* input contract: `market-engine-end-to-end-dry-run-v1`;
* artifact contract: `market-engine-local-dry-run-artifact-v1`;
* manifest contract: `market-engine-local-dry-run-artifact-manifest-v1`;
* approved path category: `artifacts/market_engine/dry_runs/`;
* module: `src/market_engine/run/local_dry_run_artifacts.py`;
* command integration: `src/market_engine/run/end_to_end_dry_run_command.py`;
* tests: `tests/market_engine/run/test_local_dry_run_artifacts.py`;
* implementation documentation: `docs/market_engine/run/me_run05_local_dry_run_artifact_persistence_implementation.md`;
* audit: `docs/market_engine/audits/me_run05_local_dry_run_artifact_persistence_audit.md`.

ME-RUN05 preserves stdout-only dry-run behavior by default and requires explicit `--write-local-artifact` invocation before local artifact writing. Persisted artifacts are local non-production review evidence only.

ME-RUN05 did not introduce provider calls, live market data calls, Telegram delivery, email delivery, broker integration, generated production reports, portfolio writes, watchlist writes, scheduling, UI behavior, Decision Engine behavior, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, trade instructions, allocation, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability, or execution advice.

ME-RUN06 implemented local dry-run fixture/data input with:

* input fixture contract: `market-engine-local-dry-run-input-fixture-v1`;
* approved local command mode: `--input-mode local_snapshot_fixture` with `--stage-payloads-json`;
* runtime module: `src/market_engine/run/local_dry_run_inputs.py`;
* command integration: `src/market_engine/run/end_to_end_dry_run_command.py`;
* tests: `tests/market_engine/run/test_local_dry_run_inputs.py` and `tests/market_engine/run/test_end_to_end_dry_run_command.py`;
* implementation documentation: `docs/market_engine/run/me_run06_local_dry_run_fixture_data_input_implementation.md`;
* audit: `docs/market_engine/audits/me_run06_local_dry_run_fixture_data_input_audit.md`;
* backlog entry: `docs/market_engine/backlog/me_run06_local_dry_run_fixture_data_input_backlog_entry.md`.

ME-RUN06 preserves embedded synthetic dry-run behavior by default, preserves raw `explicit_in_memory_payload` compatibility, and requires a non-production wrapper for `local_snapshot_fixture` data input.

ME-RUN06 did not introduce provider calls, live market data calls, Telegram delivery, email delivery, broker integration, generated production reports, portfolio writes, watchlist writes, scheduling, UI behavior, Decision Engine behavior, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, trade instructions, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Architectural Chain
