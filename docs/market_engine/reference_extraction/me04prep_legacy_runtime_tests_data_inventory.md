# ME04-PREP Legacy Runtime, Tests, and Data Inventory

Owner role: Technical Architect / Development Lead / Data Steward / QA Lead / Governance Auditor

Status: ME04-PREP-D INVENTORY

## Purpose

This inventory records the legacy runtime, script-era code, tests, data, reports, archive areas, and root-level repository files that remain before Market Engine cutover.

The purpose is to make the remaining non-documentation surface visible before ME04 defines the technical, coding, and testing architecture for Market Engine.

## Scope

In scope:

* Repository structure inspection.
* Legacy runtime, tests, data, reports, and root-level file classification.
* Archive-readiness guidance for future Market Engine cutover planning.
* Extraction needs for ME04.

Out of scope:

* Moving, deleting, or renaming files.
* Python code changes.
* Test changes.
* Data, CSV, report, portfolio, or watchlist changes.
* Provider calls, yfinance calls, SEC or EDGAR calls.
* Scanner, runtime, reporting, Telegram, portfolio, watchlist, or Decision Engine execution.
* Production writes.

## Strategic Decision

Market Engine is the new product direction. The old `market_scanner` package, `scripts/` tree, tests, and data outputs may contain useful reference knowledge, but they are not assumed to be the Market Engine implementation foundation.

Market Engine should later be built cleanly from Market Engine specifications. Legacy runtime material should be inspected, mined for lessons, and classified before any cutover or archive action is authorized.

## Active Market Engine Rule

`docs/market_engine/` is the active Market Engine documentation root.

Legacy runtime, script-era code, old tests, data outputs, generated reports, and root-level operational files remain readable reference material until a later sprint explicitly defines their future status.

## Inspection Commands Used

Safe inspection commands used:

```bash
git checkout main
git pull origin main
git status --short
git checkout -b me04-prep-inventory-legacy-runtime-tests-data
find . -maxdepth 2 -type d -path './.git' -prune -o -path './.venv' -prune -o -print | sort
find src scripts legacy archive tests -path './.git' -prune -o -path './.venv' -prune -o -type f \( -name '*.py' -o -name '*.toml' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) 2>/dev/null | sort
find data reports -type f 2>/dev/null | sort
find . -maxdepth 1 -type f | sort
find src -type f 2>/dev/null | wc -l
find scripts -type f 2>/dev/null | wc -l
find tests -type f 2>/dev/null | wc -l
find data -type f 2>/dev/null | wc -l
find reports -type f 2>/dev/null | wc -l
grep -R "market_scanner" pyproject.toml requirements.txt README.md AGENTS.md src scripts tests docs/market_engine 2>/dev/null | head -n 220
grep -R "scripts/" pyproject.toml requirements.txt README.md AGENTS.md src scripts tests docs/market_engine 2>/dev/null | head -n 220
grep -R "data/processed\|data/generated\|reports/" docs/market_engine src scripts tests 2>/dev/null | head -n 220
find src/market_scanner -maxdepth 2 -type d | sort
find scripts -maxdepth 2 -type d | sort
find tests -maxdepth 2 -type d | sort
find data -maxdepth 2 -type d | sort
find archive legacy -maxdepth 3 -type d 2>/dev/null | sort
```

No tests, providers, scanners, runtime entrypoints, reports, Telegram commands, portfolio commands, watchlist commands, or Decision Engine commands were run.

## Summary Of Findings

File counts observed:

| Area | Count |
| --- | ---: |
| `src/` | 110 files |
| `scripts/` | 57 files |
| `tests/` | 155 files |
| `data/` | 362 files |
| `reports/` | 25 files |

Major findings:

* `src/market_scanner/` remains the current packaged runtime surface with architecture, boundary, contract, provider, reporting, messaging, portfolio, scanner, and decision-related modules.
* `scripts/` remains script-era runtime material with scanner, Decision Engine, portfolio, watchlist, reporting, data source, diagnostics, and fundamentals code.
* `archive/legacy_runtime/` contains archived script-era runtime code, including core, data source, diagnostics, fundamentals, ops, portfolio, reporting, Telegram, and utility areas.
* Root-level `legacy/` contains Telegram and watchlist code.
* `tests/` includes contract, unit, integration, core, data source, diagnostics, fundamentals, ops, portfolio, reporting, fixture, and operator visibility tests.
* `tests/` still references `market_scanner`, `scripts/`, `archive/legacy_runtime/`, `data/processed`, `data/generated`, and `reports/` paths.
* `data/` includes fixtures, intake templates/pilots, local SEC/EDGAR review material, logs, raw inputs, processed outputs, portfolio files, watchlist files, normalized/generated placeholders, and scan logs.
* `reports/` includes generated or historical daily markdown reports and a Telegram message text file.
* Root-level `.env` exists but was not opened or printed.
* Root-level `README.md` still describes old v2 and script-era reference rules.
* Root-level `AGENTS.md` still contains governance rules that reference `scripts/core/decision_engine.py` as the only allocation authority for legacy checks.

## Category 1 - Current Runtime Code To Treat As Legacy Reference

Area: `src/market_scanner/`

Observed subareas:

* `analysis/`
* `context/`
* `decision/`
* `decisions/`
* `delivery/`
* `discovery/`
* `fundamentals/`
* `messaging/`
* `orchestration/`
* `portfolio/`
* `reporting/`
* `scanner/`
* `shared/`
* `timing/`
* `validation/`
* `app.py`

Classification: `NEEDS_ME04_EXTRACTION`

Functional and technical lessons likely present:

* Boundary modules and contract modules may contain useful ownership patterns.
* Existing canonical-owner strings may show old architecture intent, but they must not automatically define Market Engine architecture.
* Provider-related fundamentals modules may contain useful source-readiness, missing-data, and normalization lessons.
* Reporting, messaging, delivery, portfolio, and decision modules show side-effect and authority boundaries that ME04 should inspect before defining Market Engine boundaries.
* Import side-effect behavior should be inspected before any future cutover.

Market Engine decision:

* Keep as readable reference for ME04.
* Reject as automatic implementation foundation.
* Defer archive/freeze decision until Market Engine architecture and cutover policy are written.

Implementation implication:

* ME04 must decide which ownership boundaries, contracts, and side-effect controls are translated into Market Engine architecture.
* Market Engine modules should be specified from Market Engine documents, not copied blindly from `src/market_scanner/`.

Testing implication:

* Existing tests around `src/market_scanner/` may become reference guardrails or translation candidates.
* Future Market Engine tests should prove no lower-layer recommendation leakage, provider side effects, data output writes, reporting, Telegram, portfolio mutation, watchlist mutation, or Decision Engine behavior changes occur outside approved boundaries.

## Category 2 - Script-Era Code To Treat As Legacy Reference

Area: `scripts/`

Observed subareas and files:

* `scripts/core/`
* `scripts/data_sources/`
* `scripts/diagnostics/`
* `scripts/fundamentals/`
* `scripts/ops/`
* `scripts/portfolio/`
* `scripts/reporting/`
* `scripts/telegram/`
* `scripts/watchlist/`
* `scripts/validate_scans.py`

Classification: `REFERENCE_ONLY_NOW`

Functional and technical lessons likely present:

* Scanner, validation, context, stability, timing, fundamental, portfolio, reporting, Telegram, watchlist, and Decision Engine behavior may contain historical product lessons.
* `scripts/core/decision_engine.py` remains referenced by repository governance as legacy allocation authority.
* Several tests still reference script-era owner paths.

Market Engine decision:

* Keep as historical reference until extraction is complete.
* Reject as canonical Market Engine runtime.
* Reject quick scripts as canonical runtime.
* Defer archive/fail-closed planning until ME04 defines technical architecture and cutover requirements.

Implementation implication:

* Market Engine code must not be built by moving script-era code into a new location.
* Any useful logic must be restated as Market Engine specifications before implementation.

Testing implication:

* Tests that encode script-era behavior may still be useful as guardrails, but they need QA translation before becoming Market Engine tests.

## Category 3 - Tests To Treat As Reference Guardrails

Area: `tests/`

Observed test families:

* `tests/contract/`
* `tests/core/`
* `tests/data_sources/`
* `tests/diagnostics/`
* `tests/fixtures/`
* `tests/fundamentals/`
* `tests/integration/`
* `tests/ops/`
* `tests/portfolio/`
* `tests/reporting/`
* `tests/unit/`
* `tests/test_operator_visibility.py`

Classification: `NEEDS_QA_TRANSLATION`

Useful guardrails observed through safe search:

* Tests reference `market_scanner` package imports and canonical boundary modules.
* Tests reference script-era paths as legacy ownership evidence.
* Tests check that lower-level or synthetic paths do not create `data/generated`, `data/processed`, or `reports/daily/telegram_message.txt`.
* Tests include provider, persistence, reporting, Telegram, portfolio, source-readiness, and operator visibility concerns.

Market Engine decision:

* Keep tests as reference guardrails.
* Reject automatic reuse as the Market Engine test architecture.
* Defer exact preservation, translation, replacement, or archival decisions to ME04.

Implementation implication:

* ME04 must define Market Engine test-family ownership before any implementation sprint adds or moves tests.
* Test strategy should preserve the lessons that missing data remains missing, provider access is explicit, and lower layers do not create reports, Telegram output, generated data, portfolio mutations, watchlist mutations, or decision outputs.

Testing implication:

* Future Market Engine automated tests must use fake or synthetic provider responses.
* Future live-provider smoke checks must be explicit manual smoke harnesses, not normal automated tests.

## Category 4 - Data Inputs, Fixtures, And Source-Like Files

Areas:

* `data/fixtures/`
* `data/intake/`
* `data/local/sec_edgar/review/`
* `data/raw/`
* `data/portfolio/`
* `data/watchlist/`
* root-level `tickers.txt`

Classification: `NEEDS_DATA_OWNER_DECISION`

Observed material:

* V2 fixtures for portfolio transactions, source readiness, and universe candidates.
* Intake templates and pilot CSV files.
* Local SEC/EDGAR review JSON and CSV files.
* Raw fundamentals CSV files and raw backup files.
* Portfolio positions, metadata, summaries, transactions, review, and backups.
* Watchlist active, status, and transaction CSVs.
* Root-level ticker universe file.

Market Engine decision:

* Keep as source-like or fixture-like reference.
* Do not archive blindly.
* Defer data ownership, source truth, fixture promotion, and cutover decisions to ME04/ME05 planning.

Implementation implication:

* ME04 must specify which data areas are source inputs, fixtures, local review evidence, generated outputs, or operator-owned state.
* ME05 must not mutate portfolio/watchlist data and must not treat generated or historical output as source truth.

Testing implication:

* Fixture data may be useful for synthetic tests.
* Portfolio and watchlist files require mutation guards.
* SEC/EDGAR local review material must not authorize live SEC/EDGAR calls.

## Category 5 - Generated Data, Processed Output, Logs, And Reports

Areas:

* `data/generated/`
* `data/processed/`
* `data/logs/`
* `data/normalized/`
* `data/scans_log.csv`
* `reports/`

Classification: `CANDIDATE_FOR_FUTURE_ARCHIVE`

Observed material:

* Large ticker-level processed CSV outputs.
* Processed layer outputs such as context strength, entry quality, decisions, fundamental quality, regime, portfolio intelligence, scanner ranked data, stability, timing, validation, and reporting dashboard data.
* Logs for context, validation, Decision Engine, fundamentals, reporting, Telegram offsets, failed tickers, and scan validation.
* Daily report markdown files and `reports/daily/telegram_message.txt`.

Market Engine decision:

* Keep as historical/generated output until data retention and archive policy are defined.
* Reject as default Market Engine source truth.
* Defer archival until Data Steward and Governance Auditor define evidence retention needs.

Implementation implication:

* Market Engine must distinguish source input, raw evidence, normalized views, generated outputs, logs, and human review outputs.
* Generated output must not silently become source truth for ME05 or later analysis.

Testing implication:

* Future tests should guard that source intake and analysis do not create generated data, reports, Telegram messages, portfolio changes, watchlist changes, or Decision Engine outputs unless a sprint explicitly authorizes them.

## Category 6 - Root-Level Tooling And Repository Control Files

Observed files:

* `.env`
* `.gitignore`
* `AGENTS.md`
* `README.md`
* `pycharm_test.txt`
* `pyproject.toml`
* `requirements.txt`
* `tickers.txt`

Classification: `DO_NOT_TOUCH_YET`

Notes:

* `.env` exists and must not be printed, modified, or archived without explicit secret-handling policy.
* `.gitignore`, `pyproject.toml`, and `requirements.txt` are tooling/control files and should not be moved in a runtime archive sprint without explicit technical decision.
* `AGENTS.md` is active repository governance for local execution and must remain in place unless explicitly updated by governance.
* `README.md` is root onboarding documentation and currently references v2/script-era reset context.
* `tickers.txt` may be a universe/source-like input and needs Data Steward decision before any move.
* `pycharm_test.txt` needs manual decision because its repository role is unclear.

Market Engine decision:

* Keep in place for now.
* Defer root onboarding/tooling updates to an explicit governance or cutover sprint.

Implementation implication:

* ME04 should identify whether root documentation or tooling needs later Market Engine alignment.

Testing implication:

* Root-level dependency and configuration files should not be changed without clear test and runtime implications.

## Category 7 - Archive / Legacy Areas Needing Manual Decision

Areas:

* `archive/legacy_runtime/`
* `legacy/telegram/`
* `legacy/watchlist/`

Classification: `NEEDS_ME04_EXTRACTION`

Observed material:

* `archive/legacy_runtime/` contains archived script-era runtime code and related subareas.
* Root-level `legacy/telegram/` and `legacy/watchlist/` contain old Telegram and watchlist code.

Market Engine decision:

* Keep readable for extraction.
* Do not move or delete in ME04-PREP-D.
* Defer consolidation, freeze, or archive decisions until ME04 defines what has been extracted and what remains needed as evidence.

Implementation implication:

* ME04 should decide whether these areas are already sufficiently archived or need additional mapping into Market Engine reference records.

Testing implication:

* If tests depend on archived or legacy paths, QA must decide whether those dependencies remain reference-only or become explicit archive guardrails.

## Category 8 - Do Not Touch Before Explicit Cutover

Areas:

* `.github/`
* `.venv/`
* `.env`
* `config/`
* `src/`
* `scripts/`
* `tests/`
* `data/`
* `reports/`
* root-level tooling and control files

Classification: `DO_NOT_TOUCH_YET`

These areas may be read for inventory and extraction only. They must not be moved, deleted, renamed, rewritten, or repurposed before an explicit cutover sprint authorizes the action.

## Market Engine Extraction Needs Before Archiving Runtime

ME04 should extract the following before any runtime, test, data, or report archive sprint is authorized:

* Module ownership lessons from `src/market_scanner/`.
* Provider boundary lessons from fundamentals provider and smoke modules.
* Import side-effect lessons from package, boundary, adapter, reporting, Telegram, and app modules.
* Data/source separation lessons across fixtures, raw data, intake, local review evidence, normalized data, generated data, processed data, logs, reports, portfolio, and watchlist files.
* Missing-data handling and source-readiness rules.
* Test-family structure and test placement rules.
* Forbidden authority field policy for source, scanner, fundamental, analysis, reporting, portfolio, watchlist, and Decision Engine boundaries.
* Manual smoke harness standards and promotion rules.
* Generated-output and report-output risks.
* Portfolio/watchlist mutation boundaries.
* Root-level governance, onboarding, tooling, and dependency alignment needs.

## Archive-Readiness Classification

| Area | Classification | Rationale |
| --- | --- | --- |
| `src/market_scanner/` | `NEEDS_ME04_EXTRACTION` | Current packaged runtime contains architecture and boundary lessons but is not the Market Engine foundation. |
| `scripts/` | `REFERENCE_ONLY_NOW` | Script-era code may contain useful historical logic but must not become canonical Market Engine runtime. |
| `archive/legacy_runtime/` | `NEEDS_ME04_EXTRACTION` | Already archive-like, but still contains runtime evidence that may need mapping before final cutover. |
| `legacy/` | `NEEDS_ME04_EXTRACTION` | Old Telegram/watchlist code needs reference classification before any consolidation. |
| `tests/` | `NEEDS_QA_TRANSLATION` | Valuable behavioral guardrails remain tied to old package and script paths. |
| `data/fixtures/` | `NEEDS_DATA_OWNER_DECISION` | May contain reusable fixture material, but ownership must be defined. |
| `data/intake/` | `NEEDS_DATA_OWNER_DECISION` | Intake templates/pilots may inform ME05, but must not become source truth by default. |
| `data/local/` | `NEEDS_DATA_OWNER_DECISION` | Local SEC/EDGAR review material may be evidence, not authorization for live access. |
| `data/raw/` | `NEEDS_DATA_OWNER_DECISION` | Raw and backup fundamentals require source ownership decisions. |
| `data/portfolio/` | `DO_NOT_TOUCH_YET` | Operator/portfolio state must not be mutated or moved before explicit policy. |
| `data/watchlist/` | `DO_NOT_TOUCH_YET` | Watchlist state must not be mutated or moved before explicit policy. |
| `data/generated/`, `data/processed/`, `data/logs/`, `data/normalized/`, `data/scans_log.csv` | `CANDIDATE_FOR_FUTURE_ARCHIVE` | Likely generated or derived output, but retention and evidence policy must come first. |
| `reports/` | `CANDIDATE_FOR_FUTURE_ARCHIVE` | Generated or historical communication outputs; not active source truth. |
| root `.env` | `DO_NOT_TOUCH_YET` | Secret-bearing file must not be printed, modified, moved, or archived casually. |
| root `README.md`, `AGENTS.md`, `pyproject.toml`, `requirements.txt`, `.gitignore` | `DO_NOT_TOUCH_YET` | Repository control, onboarding, tooling, and governance files need explicit decision. |
| root `tickers.txt` | `NEEDS_DATA_OWNER_DECISION` | Potential universe/source input. |
| root `pycharm_test.txt` | `NEEDS_DATA_OWNER_DECISION` | Role unclear; needs manual decision. |
| `.github/`, `.venv/`, `config/` | `DO_NOT_TOUCH_YET` | Tooling, environment, and configuration areas are runtime-sensitive. |

## Recommended Follow-Up

1. Use this inventory in ME04 to define Market Engine technical, coding, and testing architecture.
2. In ME04, decide which runtime and test lessons become Market Engine specifications.
3. In ME04, define source input, fixture, local evidence, generated output, report output, portfolio state, and watchlist state ownership.
4. After ME04, decide whether a dedicated cutover sprint should freeze, archive, or isolate legacy runtime/test/data/report areas.
5. Keep ME05 limited to explicitly authorized source intake smoke behavior and do not treat historical generated outputs as source truth.

## Boundaries Confirmed

ME04-PREP-D confirms:

* No files were moved.
* No files were deleted.
* No files were renamed.
* No Python files were changed.
* No test files were changed.
* No data, CSV, or report files were changed.
* No provider calls were executed.
* No yfinance, SEC, or EDGAR calls were executed.
* No scanner, runtime, reporting, Telegram, portfolio, watchlist, or Decision Engine commands were run.
* No production writes were introduced.
* No BUY / SELL / HOLD, allocation, urgency, conviction, tradeability, or recommendation behavior was introduced.
