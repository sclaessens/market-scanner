# ME02 Functional Flow Extraction

Owner role: Functional Analyst / Governance Auditor

Status: ME02 EXTRACTION COMPLETE

## Purpose

This document records the source areas inspected for ME02 and the functional lessons extracted for Market Engine.

The extraction is intentionally bounded. It is not an exhaustive inventory of every old file.

## Extraction Records

### Active Product And Functional Documentation

Reference source: active product vision and functional analysis

Repository path: `docs/active/project/product_vision.md`, `docs/active/project/functional_analysis.md`

Observed logic: The operator needs deterministic market review, preserved opportunities, source-data readiness separated from investment quality, portfolio visibility, one final decision authority, and reporting that explains without changing decisions.

Useful lesson: Keep the operator-centered flow: input -> approved sources -> discovery/classification -> context -> decision/review -> communication, with row preservation and visibility.

Known risk / failure mode: Old product language may imply continuation of the previous direction if copied without Market Engine boundaries.

Market Engine decision: keep.

Reason: The operator goals and separation principles are directly useful, but they must be expressed as Market Engine flow rather than old sprint continuation.

Implementation implication: ME03 and ME04 must preserve operator visibility, source readiness, and authority boundaries.

Testing implication: Future tests must cover row identity, explicit missingness, and no final-action semantics outside the authorized layer.

Extraction status: complete.

### Pipeline And Decision Engine Contracts

Reference source: active pipeline and Decision Engine contracts

Repository path: `docs/active/pipeline/pipeline_contract.md`, `docs/active/pipeline/decision_engine_contract.md`

Observed logic: Pipeline layers classify before allocation. Decision Engine is the only allocation, execution, arbitration, and final-action authority. Reporting communicates only.

Useful lesson: Market Engine early layers must produce descriptive evidence and review states only.

Known risk / failure mode: Scanner, source, or analysis layers could leak tradeability, conviction, urgency, allocation, or final-action language.

Market Engine decision: keep.

Reason: These are core governance boundaries.

Implementation implication: ME04 must define explicit forbidden fields and module boundaries; ME03 must keep scanner/fundamental context descriptive.

Testing implication: Tests must guard against BUY / SELL / HOLD, recommendation, allocation, urgency, conviction, tradeability, and hidden filtering outside the authorized boundary.

Extraction status: complete.

### Source Data And Provider Readiness

Reference source: source data strategy, data contracts, provider integration design, live provider smoke governance

Repository path: `docs/active/data/source_data_strategy.md`, `docs/active/data/data_contracts.md`, `docs/active/data/v2_provider_integration_design.md`, `docs/active/governance/live_provider_smoke_governance.md`

Observed logic: Raw source evidence, normalized input, generated output, reporting output, and local-only data are separate roles. Source readiness is not investment quality. Missing values are not zero. Live provider smokes require explicit approval and containment.

Useful lesson: Source intake must stop at source coverage, raw evidence, normalized data, and readiness states. It must not become analysis, recommendation, or production persistence.

Known risk / failure mode: A successful provider response could be mistaken for provider approval or stronger downstream decisions.

Market Engine decision: keep.

Reason: ME05 depends on this distinction for all-ticker source intake smoke.

Implementation implication: ME03 must define source states and missingness rules; ME04 must isolate provider access; ME05 must remain explicit and bounded.

Testing implication: Automated tests must use fake/synthetic provider responses and prove missing data remains missing.

Extraction status: complete.

### Canonical Runtime Architecture

Reference source: canonical runtime architecture

Repository path: `docs/active/architecture/v2_canonical_runtime_architecture.md`

Observed logic: The current canonical target separates application entrypoint, scanner/universe selection, provider/source access, fundamentals normalization, analysis, decision/review, message composition, report generation, and delivery.

Useful lesson: Market Engine should keep the functional sequence but define it as a clean product flow rather than a script-era migration path.

Known risk / failure mode: Runtime architecture records may be read as implementation authorization.

Market Engine decision: keep.

Reason: The ownership map is useful for ME04, but ME02 remains documentation-only.

Implementation implication: ME04 must translate functional stages into module ownership without creating temporary quick scripts.

Testing implication: Tests must prove side-effect-free boundaries before any provider, report, delivery, or portfolio behavior is connected.

Extraction status: complete.

### Scanner Boundary And Scanner Semantics Audits

Reference source: scanner runtime boundary and scanner semantics planning

Repository path: `docs/audits/runtime_boundary/v2_scanner_runtime_boundary_migration.md`, `docs/audits/legacy_runtime/bl128_canonical_scanner_provider_migration_path.md`, `docs/audits/legacy_runtime/bl129_canonical_scanner_semantics_contract_plan.md`, `src/market_scanner/scanner/scanner_contracts.py`, `tests/unit/test_v2_canonical_scanner.py`

Observed logic: The canonical scanner boundary is currently planning-only and side-effect-free. Old scanner material contains useful concepts such as universe selection, candidate construction, setup classification, liquidity state, trend, momentum, relative strength, trade-plan-shaped fields, ranking, and grading. It also contains yfinance/provider access and mixed scoring semantics.

Useful lesson: Keep scanner context as descriptive market/setup evidence. Reject the old mixed provider/scanner/scoring/runtime shape as the Market Engine foundation.

Known risk / failure mode: Scanner scoring, ranking, setup names, entry, stop, target, and rr fields may be interpreted as recommendations or hidden allocation priority.

Market Engine decision: keep / reject / defer.

Reason: Keep descriptive scanner lessons; reject implicit provider access and mixed runtime authority; defer exact scanner semantics to ME03.

Implementation implication: ME03 must decide which scanner concepts survive and how to prevent hidden ranking or recommendation leakage. ME04 must keep provider/source access outside scanner semantics.

Testing implication: Tests must prove scanner context has no provider calls, file writes, reports, Telegram, portfolio/watchlist mutation, Decision Engine invocation, or recommendation language.

Extraction status: complete.

### Fundamentals And Provider Contract Code

Reference source: fundamentals provider contracts and provider contract tests

Repository path: `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`, `tests/contract/test_v2_fundamentals_provider_contracts.py`

Observed logic: Provider contracts preserve raw evidence, normalized fundamentals, readiness records, growth evidence, provider categories, neutral source statuses, and forbidden authority fields.

Useful lesson: Fundamental source intake should preserve provenance and readiness while rejecting authority fields such as final action, allocation, urgency, conviction, tradeability, score, ranking, target price, threshold, recommendation, report message, and Telegram message.

Known risk / failure mode: Normalized fundamentals may be mistaken for investment conclusions if naming and boundaries are loose.

Market Engine decision: keep.

Reason: These contracts directly support the source intake and fundamental context parts of the functional flow.

Implementation implication: ME03 must define fundamental context separately from recommendation logic.

Testing implication: Tests must prove provider/fundamental records reject forbidden fields and do not import network or legacy script dependencies.

Extraction status: complete.

### Validation And Analysis Boundaries

Reference source: validation contracts and analysis boundary records

Repository path: `src/market_scanner/validation/validation_contracts.py`, `src/market_scanner/analysis/analysis_contracts.py`, `docs/audits/runtime_boundary/v2_analysis_runtime_boundary_migration.md`

Observed logic: Validation emits structure-classification issues without filtering or decisions. Analysis planning consumes governed evidence, preserves limitations, and forbids provider calls, data writes, reports, Telegram, portfolio/watchlist mutation, final outcomes, capital-action outputs, priority labels, numeric ranks, price projections, and execution-quality outputs.

Useful lesson: Source intake stops before analysis. Analysis begins only after governed evidence and normalized/context records exist.

Known risk / failure mode: Analysis can become a disguised decision layer if it emits ranks, action labels, price projections, or execution-quality outputs.

Market Engine decision: keep.

Reason: These boundaries define the transition from evidence preparation to review-oriented analysis.

Implementation implication: ME04 must define the ownership and field boundaries for analysis; ME07 must stay review-oriented.

Testing implication: Tests must prove analysis is side-effect-free and does not emit final action or capital-action semantics.

Extraction status: complete.

### Portfolio And Watchlist Boundaries

Reference source: portfolio source-of-truth documentation and contracts

Repository path: `docs/active/portfolio/portfolio_source_of_truth.md`, `src/market_scanner/portfolio/portfolio_source_contracts.py`

Observed logic: Manual portfolio source records are the only approved portfolio source-of-truth role. Generated portfolio review, reporting display input, and Telegram output are not source truth. Portfolio source records must not contain allocation, execution, urgency, conviction, tradeability, ranking, score, recommendation, reporting text, or Telegram text.

Useful lesson: Market Engine may use portfolio/watchlist concepts as read-only context when approved, but early layers must not mutate them or derive actions from them.

Known risk / failure mode: Portfolio display fields or watchlist state can be mistaken for source truth or decision authority.

Market Engine decision: keep / defer.

Reason: Keep the source-of-truth and mutation boundaries; defer exact Market Engine portfolio/watchlist input use to ME04 or later.

Implementation implication: ME04 must define read-only portfolio/watchlist input boundaries before implementation.

Testing implication: Tests must prove source, scanner, fundamental, and analysis layers do not mutate portfolio/watchlist data.

Extraction status: complete.

### Reporting, Telegram, And Delivery Boundaries

Reference source: reporting contract, Telegram UX baseline, reporting tests, delivery boundary audit

Repository path: `docs/active/reporting/reporting_contract.md`, `docs/active/reporting/reporting_telegram_ux.md`, `tests/contract/test_v2_reporting_synthetic_end_to_end.py`, `docs/audits/runtime_boundary/v2_delivery_runtime_boundary_migration.md`

Observed logic: Reporting communicates Decision Engine outputs. Telegram is compact, portfolio-first communication, not decision authority. Delivery is a transport boundary and must not send messages or read credentials unless explicitly authorized.

Useful lesson: Local operator review output may borrow communication lessons, but early Market Engine must not generate production reports or send Telegram.

Known risk / failure mode: Phrases like buy now, buy on pullback, and buy on breakout are useful display group names only after approved decision states exist; if used earlier, they leak recommendation semantics.

Market Engine decision: keep / defer.

Reason: Keep communication-only and side-effect boundaries; defer Telegram/reporting behavior to later authorized layers.

Implementation implication: ME08 may produce local operator review output, but Telegram/reporting delivery requires later approval.

Testing implication: Tests must prove local review output creates no Telegram artifacts, sends no messages, writes no production reports, and does not alter decision semantics.

Extraction status: complete.

### Old Backlog And Legacy Runtime Records

Reference source: old backlog and legacy runtime audits

Repository path: `docs/active/project/backlog.md`, `docs/audits/legacy_runtime/`, `docs/audits/reset_cleanup/`, `docs/audits/provider_smokes/`

Observed logic: Prior work captured many useful lessons about side effects, provider risk, archive-readiness, script-era coupling, missing-data handling, and test-family placement.

Useful lesson: Preserve old evidence and lessons, but do not continue the legacy cleanup line as the active Market Engine implementation path.

Known risk / failure mode: Treating old backlog completion records as Market Engine implementation approval would carry old assumptions forward.

Market Engine decision: keep / reject.

Reason: Keep evidence and lessons; reject old backlog as the active product path.

Implementation implication: ME03 and ME04 should cite old findings only where they materially improve Market Engine specifications.

Testing implication: Future tests should be written for Market Engine contracts and should not force old paths, generated artifacts, or compatibility wrappers.

Extraction status: complete.

## Summary Of Keep / Reject / Defer Decisions

Keep:

- operator visibility and preserved evidence;
- source readiness separate from investment quality;
- raw evidence, normalized input, generated output, report output, and local-only separation;
- row identity and explicit missingness;
- scanner and fundamental context as descriptive evidence;
- Decision Engine authority protection;
- reporting and Telegram as communication-only downstream layers;
- side-effect-free early boundaries;
- fake/synthetic provider responses for automated tests.

Reject:

- blind copying of old script-era code;
- implicit yfinance/provider access from scanner or tests;
- generated outputs as active source truth;
- missing values converted to zero;
- BUY / SELL / HOLD or recommendation leakage in early layers;
- reporting, Telegram, portfolio, watchlist, or Decision Engine behavior inside source/scanner/fundamental/analysis layers;
- old quick scripts as canonical runtime.

Defer:

- exact ticker universe source for ME05;
- exact scanner semantics to keep;
- exact fundamental metric set for the first analysis pass;
- exact local operator review format;
- optional downstream decision/reporting/notification integration.

