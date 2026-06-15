# ME04 Technical, Coding, and Testing Extraction

Owner role: Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: ME04 EXTRACTION COMPLETE

## Purpose

This document records the source areas and lessons used to create the Market Engine technical, coding, and testing architecture.

The extraction is intentionally bounded. It does not perform line-by-line code analysis and does not authorize implementation.

## Extraction Scope

In scope:

* Market Engine documentation created in ME01 through ME03;
* documentation-root cleanup decisions from ME04-PREP through ME04-PREP-C;
* legacy runtime, tests, and data inventory from ME04-PREP-D;
* old runtime, scripts, tests, data, reports, and root-level files as reference categories;
* implementation implications for ME05.

Out of scope:

* code changes;
* test changes;
* provider calls;
* runtime execution;
* data/report mutation;
* recommendation behavior;
* Decision Engine behavior changes.

## Extraction Records

### 1. ME02 Functional Flow

Reference source: Market Engine functional flow

Repository path: `docs/market_engine/analysis/functional_flow.md`

Observed logic:

ME02 defines the Market Engine flow from operator intent through source intake, source coverage, raw evidence, normalization, missing-data handling, scanner context, fundamental context, first analysis pass, flags, and local operator review.

Useful lesson:

Technical architecture must map directly to functional stages, not to old script-era folders.

Known risk / failure mode:

If architecture follows old folder names blindly, Market Engine will become another migration of the previous system.

Market Engine decision:

Keep the functional sequence and translate it into clean ownership boundaries.

Implementation implication:

Future modules should be organized around intent, universe, sources, evidence, normalization, scanner context, fundamental context, analysis, review, and shared primitives.

Testing implication:

Tests should align with ownership boundaries and prove side-effect-free transitions between stages.

Extraction status: complete.

### 2. ME03 Financial, Scanner, and Fundamental Logic

Reference source: ME03 logic specification

Repository path: `docs/market_engine/analysis/financial_scanner_fundamental_logic.md`

Observed logic:

ME03 defines financial, scanner, and fundamental outputs as descriptive and source-aware. It preserves source readiness, explicit missingness, ticker-level failure handling, scanner/fundamental context boundaries, and no recommendation leakage.

Useful lesson:

Technical architecture must isolate source intake, scanner context, fundamental context, and first analysis so that none can emit final action authority.

Known risk / failure mode:

Fields such as score, rank, target, threshold, setup, entry, or valuation could become hidden recommendation proxies.

Market Engine decision:

Keep descriptive contexts and explicitly forbid authority fields in early layers.

Implementation implication:

ME04 must define a forbidden field policy for source, scanner, fundamental, and analysis records.

Testing implication:

Future tests must include negative fixtures with forbidden authority fields.

Extraction status: complete.

### 3. ME01 Coding Standards

Reference source: Market Engine coding standards

Repository path: `docs/market_engine/architecture/coding_standards.md`

Observed logic:

Coding standards prohibit creating a new Python file for every new step, temporary quick scripts as canonical runtime, hidden provider calls, import-time production side effects, missing-to-zero conversion, and recommendation leakage in lower layers.

Useful lesson:

The technical architecture must define when new files are justified and where canonical runtime may live.

Known risk / failure mode:

Without file ownership rules, implementation will recreate script sprawl.

Market Engine decision:

Keep coding standards and translate them into file/module strategy.

Implementation implication:

New Python files require ownership-boundary justification.

Testing implication:

Future review should reject implementation PRs that add quick scripts as canonical runtime or create unnecessary module fragmentation.

Extraction status: complete.

### 4. ME01 Testing Strategy

Reference source: Market Engine testing strategy

Repository path: `docs/market_engine/testing/testing_strategy.md`

Observed logic:

Testing strategy requires fake/synthetic provider responses, no live provider calls in normal tests, preserving missingness, proving ticker failures do not stop batches, and guarding against side effects and recommendation leakage.

Useful lesson:

ME04 must define test-family ownership before ME05 and later implementation add tests.

Known risk / failure mode:

Old tests may be copied into new locations without translating their ownership or provider boundaries.

Market Engine decision:

Keep testing principles; do not reuse old tests automatically.

Implementation implication:

Future tests should be placed by Market Engine boundary, not by old script-era module path.

Testing implication:

The first ME05 implementation should use explicit manual smoke harness rules, not normal live-provider automated tests.

Extraction status: complete.

### 5. Documentation Archive Cleanup

Reference source: ME04-PREP, ME04-PREP-B, ME04-PREP-C

Repository paths:

* `docs/archive/market_scanner_reference/`
* `docs/market_engine/reference_extraction/legacy_reference_map.md`
* `docs/market_engine/reference_extraction/me04prep_remaining_legacy_documentation_inventory.md`
* `docs/market_engine/audits/me04prep_consolidate_remaining_legacy_docs_audit.md`

Observed logic:

`docs/market_engine/` is the only active Market Engine documentation root. Old v2, BL, reset, audit, and historical reference documentation is preserved under `docs/archive/market_scanner_reference/`.

Useful lesson:

ME04 should cite Market Engine docs as active authority and archived docs only as reference.

Known risk / failure mode:

Old archived documents could be mistaken for active architecture authority if paths are not interpreted correctly.

Market Engine decision:

Keep archive as reference only.

Implementation implication:

Future implementation must use Market Engine specs first. Archived documents require explicit extraction before becoming requirements.

Testing implication:

Tests should verify current Market Engine behavior, not old archived expectations unless translated.

Extraction status: complete.

### 6. Legacy Runtime, Tests, and Data Inventory

Reference source: ME04-PREP-D inventory

Repository path: `docs/market_engine/reference_extraction/me04prep_legacy_runtime_tests_data_inventory.md`

Observed logic:

The inventory classifies `src/market_scanner/` as `NEEDS_ME04_EXTRACTION`, `scripts/` as `REFERENCE_ONLY_NOW`, `tests/` as `NEEDS_QA_TRANSLATION`, source-like data as `NEEDS_DATA_OWNER_DECISION`, generated data/reports as `CANDIDATE_FOR_FUTURE_ARCHIVE`, and runtime-sensitive areas as `DO_NOT_TOUCH_YET`.

Useful lesson:

ME04 must define cutover policy before old code, tests, and data are archived or frozen.

Known risk / failure mode:

Moving runtime or tests before architecture could break repo expectations and hide important guardrails.

Market Engine decision:

Keep old runtime/tests/data in place for now, but classify them as legacy/reference until later cutover.

Implementation implication:

ME05 may not depend on old runtime as canonical implementation. It may inspect reference logic only through specs.

Testing implication:

QA must translate valuable legacy tests into Market Engine test families before old tests are retired.

Extraction status: complete.

### 7. Current Runtime Package

Reference source: legacy runtime inventory category for `src/market_scanner/`

Repository path: `src/market_scanner/`

Observed logic:

The old packaged runtime contains areas for analysis, context, decision, decisions, delivery, discovery, fundamentals, messaging, orchestration, portfolio, reporting, scanner, shared, timing, validation, and `app.py`.

Useful lesson:

The existing package reveals historical domain boundaries but must not automatically define Market Engine architecture.

Known risk / failure mode:

New development could continue inside `src/market_scanner/` and recreate the old system.

Market Engine decision:

Treat as reference needing extraction. Prefer future clean implementation under `src/market_engine/`.

Implementation implication:

Before writing code, ME05 must be bounded as source intake smoke and avoid broad runtime integration.

Testing implication:

Old package imports may remain in old tests until QA translation, but new Market Engine tests should target future Market Engine ownership.

Extraction status: complete.

### 8. Script-Era Runtime

Reference source: legacy runtime inventory category for `scripts/`

Repository path: `scripts/`

Observed logic:

The script tree contains old core, data source, diagnostics, fundamentals, ops, portfolio, reporting, Telegram, watchlist, and validation script material.

Useful lesson:

Historical behavior may be useful, but scripts must not be promoted as Market Engine canonical runtime.

Known risk / failure mode:

A fast ME05 smoke could be implemented as another quick script and become de facto runtime.

Market Engine decision:

Reject script-era code as implementation foundation. Keep as reference only.

Implementation implication:

ME05 smoke harness must be explicit, bounded, and non-canonical unless a later architecture promotes it.

Testing implication:

Automated tests should not call script-era runtime paths as Market Engine behavior.

Extraction status: complete.

### 9. Tests As Reference Guardrails

Reference source: legacy runtime inventory category for `tests/`

Repository path: `tests/`

Observed logic:

The old tests contain contract, unit, integration, core, diagnostics, fundamentals, reporting, portfolio, and operator visibility guardrails. They still reference old `market_scanner`, `scripts`, generated data, and reports paths.

Useful lesson:

Tests encode important guardrails, but their ownership must be translated before Market Engine implementation.

Known risk / failure mode:

Keeping old tests active as if they define Market Engine can force compatibility with old architecture.

Market Engine decision:

Needs QA translation.

Implementation implication:

ME04 defines future `tests/market_engine/` ownership conceptually. Actual test creation waits for implementation scope.

Testing implication:

Preserve guardrails around missing data, no live provider calls, side-effect exclusion, and recommendation leakage.

Extraction status: complete.

### 10. Data And Report Areas

Reference source: legacy runtime inventory data categories

Repository paths: `data/`, `reports/`, `tickers.txt`

Observed logic:

Source-like inputs, fixtures, raw/local material, portfolio/watchlist files, generated/processed/log outputs, and reports are mixed in the repository.

Useful lesson:

ME04 must define source truth versus generated output before ME05 writes or reads anything.

Known risk / failure mode:

Generated CSVs or old reports may become source truth by accident.

Market Engine decision:

Do not move or mutate data now. Require data-owner decisions before source-like inputs are used or archived.

Implementation implication:

ME05 must explicitly choose ticker universe and output location.

Testing implication:

Tests must prove generated output is not source truth and missing data remains explicit.

Extraction status: complete.

## Summary Of Keep / Reject / Defer Decisions

Keep:

* ME02 functional stages as architecture map.
* ME03 descriptive financial/scanner/fundamental boundaries.
* ME01 coding and testing guardrails.
* Old runtime/tests/data as reference until cutover.
* Provider/source access isolation.
* Missing data preservation.
* Per-ticker failure capture.
* Decision Engine authority protection.

Reject:

* blind migration of `src/market_scanner/`;
* script-era code as canonical Market Engine runtime;
* quick scripts as canonical smoke implementation;
* live provider calls in automated tests;
* generated outputs as source truth;
* hidden provider access;
* lower-layer reports, Telegram, portfolio/watchlist mutation, or Decision Engine behavior;
* BUY / SELL / HOLD, allocation, urgency, conviction, tradeability, ranking, or recommendation leakage.

Defer:

* actual `src/market_engine/` package creation;
* old runtime archival/freeze;
* old test translation implementation;
* data-owner decisions;
* generated data/report archive;
* provider/source family for ME05;
* exact ticker universe for ME05.

## Implementation Implications For ME05

ME05 should be constrained to an explicit all-ticker source intake smoke.

ME05 must not implement a full runtime, recommendation engine, reporting pipeline, Telegram sender, portfolio mutator, watchlist mutator, or Decision Engine integration.

ME05 must define:

* explicit ticker universe input;
* explicit source/provider target;
* explicit manual invocation path;
* explicit output as local/non-production smoke evidence;
* per-ticker status capture;
* missing-data preservation;
* provider error and unsupported ticker distinction where possible;
* no recommendation or downstream side effects.

## Extraction Status

ME04 extraction is complete enough to steer ME05 and future implementation planning.

Further line-by-line code extraction should occur only when a specific implementation sprint needs it, not as another broad documentation loop.
