# Market Engine Technical, Coding, and Testing Architecture

Owner role: Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: ME04 TECHNICAL SPECIFICATION

## Purpose

This document defines the technical, coding, and testing architecture for Market Engine.

ME04 translates the Market Engine functional flow, financial/scanner/fundamental logic, coding standards, testing strategy, and legacy runtime inventory into an implementation-ready architecture baseline.

This document does not implement runtime behavior. It authorizes the direction and boundaries for later implementation sprints, especially ME05.

## Scope

ME04 defines:

* active Market Engine technical ownership boundaries;
* the intended new package direction;
* how old `src/market_scanner/`, `scripts/`, tests, data, and reports are treated;
* provider/source access boundaries;
* raw evidence, normalization, scanner context, fundamental context, analysis, and local review boundaries;
* file/module strategy;
* coding standards;
* testing architecture;
* manual smoke harness policy;
* forbidden authority fields;
* archive and cutover prerequisites.

ME04 does not:

* create Python implementation;
* modify tests;
* run tests;
* execute provider calls;
* run yfinance, SEC, EDGAR, scanner, reporting, Telegram, portfolio, watchlist, or runtime commands;
* mutate data, portfolio, or watchlist files;
* generate reports;
* introduce BUY / SELL / HOLD, allocation, urgency, conviction, tradeability, ranking, or recommendation behavior.

## Baseline Inputs

ME04 is based on:

* ME01 documentation structure, coding standards, and testing strategy;
* ME02 functional flow;
* ME03 financial, scanner, and fundamental logic;
* ME04-PREP documentation archive consolidation;
* ME04-PREP-D legacy runtime, tests, and data inventory.

The active documentation root is `docs/market_engine/`.

Old market-scanner documentation is reference-only under `docs/archive/market_scanner_reference/`.

## Strategic Technical Decision

Market Engine is not a migration of the old `market_scanner` runtime.

The old runtime surfaces remain reference material:

* `src/market_scanner/` is classified as `NEEDS_ME04_EXTRACTION`.
* `scripts/` is classified as `REFERENCE_ONLY_NOW`.
* `tests/` is classified as `NEEDS_QA_TRANSLATION`.
* source-like data requires data-owner decision.
* generated data, logs, processed output, and reports are candidates for future archive.

The new implementation direction should be a clean Market Engine package, eventually under:

```text
src/market_engine/
```

ME04 does not create that package yet. It defines the module boundaries that later sprints should implement.

## Active Runtime Rule

Until a later sprint implements Market Engine code, the old runtime may remain physically present in the repository, but it is not the Market Engine implementation foundation.

Future work must not add new Market Engine behavior into `scripts/` as canonical runtime.

Future work must not treat `src/market_scanner/` as the automatic target package for new Market Engine features.

Useful old logic must first be restated in Market Engine specifications before implementation.

## Target Package Direction

The preferred future package is:

```text
src/market_engine/
```

Target conceptual structure:

```text
src/market_engine/
  intent/
  universe/
  sources/
  evidence/
  normalization/
  scanner_context/
  fundamental_context/
  analysis/
  review/
  shared/
```

These names are architectural ownership boundaries, not an instruction to immediately create all folders.

New Python files are allowed only when a new ownership boundary is explicitly justified by architecture or implementation scope.

## Module Ownership Map

| ME02 stage | Future technical owner | Purpose | Key boundary |
|---|---|---|---|
| Operator intent | `intent` | Capture run purpose and requested review scope. | No provider calls or side effects. |
| Ticker universe / watchlist selection | `universe` | Resolve ticker identities and selection reasons. | Watchlist/portfolio are read-only when approved. |
| Source intake request | `sources` | Build explicit source request plans. | No analysis or recommendation semantics. |
| Provider/source access | `sources` | Perform bounded approved source access. | Manual smoke only until promoted. |
| Source coverage validation | `sources` / `evidence` | Classify source availability and failures. | Source readiness is not investment quality. |
| Raw source result preservation | `evidence` | Preserve provenance and raw source evidence. | Raw evidence is not normalized input or generated output. |
| Normalized data view | `normalization` | Map approved evidence into program-ready records. | Missing values remain missing. |
| Missing-data handling | `normalization` / `shared` | Represent missing, partial, stale, invalid, unsupported, provider error, insufficient, and review-required states. | No missing-to-zero conversion. |
| Scanner context | `scanner_context` | Produce descriptive market/setup context. | No provider access, ranking, urgency, tradeability, or recommendations. |
| Fundamental context | `fundamental_context` | Produce descriptive company/financial context. | No final action, score, ranking, threshold authority, target-price action, or recommendation. |
| First analysis pass | `analysis` | Combine governed evidence into review-oriented analysis. | No Decision Engine authority. |
| Risk and quality flags | `analysis` / `review` | Surface evidence, source, and review limitations. | Flags do not allocate. |
| Local operator review output | `review` | Produce local human review output. | Communication only, no Telegram or production report unless later authorized. |

## Provider And Source Access Boundary

Provider access must be isolated in a dedicated source boundary.

Provider access must not occur from:

* imports;
* tests;
* scanner context;
* fundamental context;
* analysis;
* reporting;
* Telegram;
* portfolio/watchlist code;
* Decision Engine behavior;
* helper modules with hidden side effects.

Future provider access must be explicit, bounded, reviewable, and manually invoked until a later sprint authorizes a broader runtime path.

ME05 may build only a bounded all-ticker source intake smoke harness, and only according to this boundary.

## Data Ownership Model

Market Engine must keep these roles separate:

| Role | Meaning | Authority |
|---|---|---|
| Source-like input | Approved input, fixture, local evidence, raw provider capture, portfolio/watchlist source record. | Input evidence only. |
| Raw evidence | Preserved provider/source response with provenance. | Evidence, not program-ready interpretation. |
| Normalized data | Program-ready view derived from approved raw/source evidence. | Analysis input only. |
| Generated analysis output | Review-oriented analysis or limitation flags. | Generated output, not source truth. |
| Local review output | Operator communication. | Communication only. |
| Reporting/Telegram output | Deferred downstream communication. | Not active in early Market Engine layers. |

Generated output must not become source truth by default.

Missing values must remain explicit and must not be converted to zero.

## Data Directory Decisions

ME04 does not move data.

Current classification:

* `data/fixtures/`, `data/intake/`, `data/raw/`, `data/local/`, `tickers.txt`: `NEEDS_DATA_OWNER_DECISION`.
* `data/generated/`, `data/processed/`, `data/logs/`, `reports/`: `CANDIDATE_FOR_FUTURE_ARCHIVE`.
* `data/portfolio/`, `data/watchlist/`: do not mutate from lower layers.

ME05 should not treat generated data or reports as source truth.

ME05 may read only explicitly approved source/intake/universe inputs.

## Scanner Context Architecture

Scanner context is descriptive evidence.

Allowed future outputs:

* ticker identity;
* universe membership;
* selection reason;
* discovery reason;
* descriptive setup context;
* liquidity context;
* trend context;
* momentum context;
* relative strength context;
* scanner evidence state;
* scanner source references;
* scanner missing-data and review flags.

Forbidden outputs:

* BUY / SELL / HOLD;
* recommendation;
* allocation;
* urgency;
* conviction;
* tradeability;
* hidden ranking;
* entry/stop/target as trade instruction;
* Telegram/reporting text;
* portfolio/watchlist mutation;
* Decision Engine invocation.

Scanner context must consume governed data, not fetch providers implicitly.

## Fundamental Context Architecture

Fundamental context is descriptive source-backed company and financial evidence.

Allowed future outputs:

* source readiness;
* raw evidence reference;
* normalized field availability;
* period metadata;
* missing field names;
* data freshness;
* revenue, income, cash flow, capex, and free cash flow derivation status;
* growth, profitability, balance-sheet, cash-generation, and valuation context;
* review-required flags.

Forbidden outputs:

* final action;
* recommendation;
* allocation;
* urgency;
* conviction;
* tradeability;
* score/ranking as hidden recommendation;
* target-price action trigger;
* threshold authority;
* report message;
* Telegram message;
* hidden provider access from tests or analysis.

Derived metrics must be blocked when required components are missing.

## Analysis Boundary

Analysis begins only after source intake, coverage validation, raw evidence, normalization, missingness, scanner context, and fundamental context are available.

Analysis may produce:

* descriptive evidence summaries;
* limitation flags;
* missingness summaries;
* source-readiness summaries;
* review-required notes;
* local operator review inputs.

Analysis must not produce:

* BUY / SELL / HOLD;
* final action;
* allocation;
* execution instruction;
* position sizing;
* urgency;
* conviction;
* tradeability;
* hidden ranking;
* portfolio transaction advice.

Decision Engine authority remains deferred and protected.

## Review Output Boundary

Local operator review output is communication only.

It may show:

* source coverage;
* missingness;
* failures;
* evidence limitations;
* review-required states;
* analysis-ready observations.

It must not:

* send Telegram;
* generate production reports;
* mutate portfolio or watchlist;
* change decisions;
* emit final action authority.

## Forbidden Authority Field Policy

The following field families are forbidden in source, raw evidence, normalized data, scanner context, fundamental context, and first-analysis layers unless a later approved Decision Engine contract owns them:

* `buy`, `sell`, `hold`;
* `recommendation`;
* `final_action`;
* `allocation`;
* `position_size`;
* `execution_instruction`;
* `urgency`;
* `conviction`;
* `tradeability`;
* `rank` when used as hidden allocation priority;
* `score` when used as hidden recommendation;
* `target_price_action`;
* `telegram_message`;
* `report_message`;
* portfolio/watchlist mutation markers.

ME04 recommends that future test fixtures include negative cases for these forbidden fields.

## File And Module Strategy

Implementation sprints must follow these rules:

* Do not create a new Python file for every new step.
* Extend an existing Market Engine module when ownership is clear.
* Create new Python files only for explicit ownership boundaries.
* Do not create temporary quick scripts as canonical runtime.
* Do not place canonical Market Engine behavior in `scripts/`.
* Do not add hidden provider access to convenience helpers.
* Do not add import-time side effects.
* Do not write production data, reports, Telegram, portfolio, or watchlist output from lower layers.

ME05 may create a bounded manual smoke harness only if the harness is clearly non-canonical or explicitly promoted by architecture.

## Testing Architecture

Future tests should be organized by ownership boundary, not by old script-era shape.

Preferred future test families:

```text
tests/market_engine/
  contract/
  unit/
  integration_synthetic/
  smoke_manual/
```

These folders are conceptual until an implementation sprint authorizes test creation.

Test principles:

* Do not create new test files when an existing Market Engine test family is suitable.
* Create new test files only for new ownership boundaries.
* Automated tests must not execute live provider calls.
* Automated tests must use fake or synthetic provider responses.
* Tests must prove missing data remains missing.
* Tests must prove ticker-level failures do not stop batch processing.
* Tests must guard against BUY / SELL / HOLD and recommendation leakage.
* Tests must guard against Decision Engine, Telegram, reporting, portfolio, and watchlist side effects in lower layers.
* Tests must prove generated output does not become source truth.

Old tests are reference guardrails, not the Market Engine test architecture by default.

## Manual Smoke Harness Policy

ME05 may create an explicit all-ticker source intake smoke harness.

The harness must be:

* manual or explicitly invoked;
* bounded;
* auditable;
* non-canonical unless promoted later;
* isolated from automated tests;
* isolated from reports, Telegram, portfolio, watchlist, and Decision Engine behavior;
* limited to source coverage, raw evidence feasibility, normalization feasibility, missingness, and readiness states.

ME05 must not produce recommendation logic.

## Legacy Runtime Cutover Policy

ME04 does not archive code, tests, data, or reports.

Before a future cutover archive sprint can move old runtime/test/data areas, the following must exist:

* Market Engine module ownership specification, provided by this document;
* implementation target package decision;
* QA translation plan for old tests;
* data-owner decision for source-like inputs;
* generated-output archive decision;
* root tooling decision for `README.md`, `AGENTS.md`, `pyproject.toml`, and `requirements.txt`;
* explicit list of old runtime areas that are reference-only;
* confirmation that future implementation no longer depends on old import paths.

Recommended future classifications:

* `src/market_scanner/`: freeze/reference until Market Engine replacement exists.
* `scripts/`: archive/fail-close candidate after extraction readiness.
* `tests/`: translate useful guardrails, then archive old-path tests where safe.
* `data/generated`, `data/processed`, `data/logs`, `reports`: archive candidates after retention decision.
* source-like data: preserve until data-owner decision.

## ME05 Readiness Gate

ME05 may begin only when the following are true:

* source intake boundary is accepted;
* manual smoke harness policy is accepted;
* ticker universe input for smoke is chosen;
* first provider/source family is chosen;
* output location is explicitly local and non-production;
* missing-data preservation rules are explicit;
* per-ticker failure capture is required;
* no recommendation, reporting, Telegram, portfolio, watchlist, or Decision Engine behavior is introduced.

## Open Questions

* Should future Market Engine implementation package be `src/market_engine/` immediately, or should a temporary compatibility bridge exist?
* Which ticker universe should ME05 use first?
* Which provider/source family should ME05 smoke first?
* Should raw evidence be persisted in ME05, or should ME05 produce only coverage summaries?
* Which old test families should be translated first?
* When should `scripts/` be archived or fail-closed?
* When should generated data and reports be archived?
* How should root `README.md` and `AGENTS.md` be updated after Market Engine implementation starts?

## Final Architecture Statement

Market Engine must be built as a clean, bounded, side-effect-controlled system.

Old runtime, script-era code, tests, and data remain useful reference material, but they are not the implementation foundation.

ME05 may now proceed only as a bounded all-ticker source intake smoke, not as a full runtime rebuild, recommendation engine, reporting pipeline, Telegram sender, portfolio mutator, or Decision Engine integration.
