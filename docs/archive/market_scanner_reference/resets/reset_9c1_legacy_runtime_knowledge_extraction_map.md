# RESET-9C1 — Legacy Runtime Knowledge Extraction Map

## 1. Purpose

RESET-9C1 maps the legacy runtime knowledge that must be preserved, redesigned, rejected, or reapproved before the old runtime can be safely retired.

This is documentation-only. It does not authorize archive, delete, migration, refactor, or runtime execution.

## 2. Inputs

Primary input:

- `docs/resets/reset_9c_legacy_runtime_inventory_and_retirement_decision.md`

Related canonical documents:

- `docs/active/data_lifecycle.md`
- `docs/active/data_contracts.md`
- `docs/active/source_data_strategy.md`
- `docs/active/financial_analysis.md`
- `docs/active/pipeline_contract.md`
- `docs/active/decision_engine_contract.md`
- `docs/active/reporting_contract.md`
- `docs/active/testing_strategy.md`

## 3. Active Retirement Decision

RESET-9C remains active:

```text
DO_NOT_ARCHIVE_OR_DELETE_LEGACY_RUNTIME_YET
```

RESET-9C1 adds the required next condition:

```text
PROCEED_WITH_KNOWLEDGE_EXTRACTION_BEFORE_RUNTIME_RETIREMENT
```

No legacy runtime file may be archived or deleted until the relevant knowledge has either been carried forward into v2 doctrine/contracts, explicitly rejected, or marked as no longer in scope.

## 4. Extraction Doctrine

The required pattern is:

```text
legacy behavior -> extracted knowledge -> v2 doctrine/contract -> v2 implementation sprint
```

Legacy code must not be copied into v2 merely because it exists.

Mandatory guardrails:

- classification upstream;
- allocation downstream;
- Decision Engine is the only final-action authority;
- reporting communicates only;
- raw source data is not normalized input;
- normalized input is not generated output;
- generated output is not source-of-truth;
- reports are not inputs;
- missing values are not zero;
- source-data readiness is not investment quality.

## 5. Knowledge Map by Runtime Zone

### 5.1 Orchestration and Pipeline Execution

Legacy surface:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- manual workflow fallback through `.github/workflows/daily-market-scan.yml`

Knowledge to preserve:

- legacy end-to-end layer sequence;
- operator expectation of a single scan command;
- failure visibility expectations;
- dependencies between scanner, validation, context, fundamentals, timing, portfolio, decisions, reporting, and delivery.

Must be redesigned:

- orchestration belongs in `src/market_scanner/orchestration/`;
- side effects must be explicit;
- provider calls must not be hidden inside orchestration;
- generated outputs must not be automatically committed to `main`.

Must not be copied:

- direct legacy CSV write chains as architecture;
- reporting delivery embedded as decision authority;
- generated-output-as-input behavior.

V2 targets:

- `docs/active/pipeline_contract.md`
- `docs/active/data_lifecycle.md`
- `src/market_scanner/orchestration/`

Retirement blocker:

- no complete v2 orchestration replacement exists.

### 5.2 Discovery and Scanner Logic

Legacy surface:

- `scripts/core/scanner.py`
- `scripts/core/data_fetcher.py`
- `scripts/core/indicators.py`
- `scripts/core/regime.py`

Knowledge to preserve:

- ticker entry concepts;
- candidate discovery concepts;
- historical indicator concepts;
- market regime and sector context concepts;
- scanner row shape expected downstream.

Must be redesigned:

- discovery belongs under `src/market_scanner/discovery/`;
- provider access must be separated from deterministic classification;
- discovery output must be descriptive, auditable, and contract-defined.

Must not be copied:

- yfinance/provider access embedded in core scanner logic;
- hidden filtering before governance review;
- upstream tradeability or final-action semantics;
- generated scanner CSVs as v2 source-of-truth.

V2 targets:

- `docs/active/source_data_strategy.md`
- `docs/active/data_lifecycle.md`
- `src/market_scanner/discovery/`

Retirement blocker:

- v2 discovery remains scaffold-only.

### 5.3 Validation Layer

Legacy surface:

- `scripts/core/build_validation_layer.py`
- `scripts/core/validator.py`
- validation and validation-analysis helpers.

Knowledge to preserve:

- scanner-row validity concepts;
- completeness checks;
- failure categories;
- legacy tests that encode expected validation behavior.

Must be redesigned:

- validation belongs under `src/market_scanner/validation/`;
- validation must operate on normalized input or approved records;
- validation states must remain explicit and auditable.

Must not be copied:

- duplicate wrappers as permanent v2 architecture;
- validation outputs as final decision authority;
- silent filtering without visible reason.

V2 targets:

- `docs/active/data_contracts.md`
- `docs/active/testing_strategy.md`
- `src/market_scanner/validation/`

Retirement blocker:

- validation assumptions are not yet fully represented as v2 contracts and tests.

### 5.4 Context Layer

Legacy surface:

- `scripts/core/build_context_layer.py`
- `scripts/core/build_context_backfill.py`

Knowledge to preserve:

- context-strength concepts;
- sector or market-relative assumptions;
- historical backfill assumptions;
- downstream use of context.

Must be redesigned:

- context belongs under `src/market_scanner/context/`;
- context must remain classification-only;
- context output must preserve uncertainty and missing data.

Must not be copied:

- backfill outputs as current source-of-truth;
- context-derived tradeability;
- implicit filtering based on context alone.

V2 targets:

- `docs/active/functional_analysis.md`
- `docs/active/pipeline_contract.md`
- `src/market_scanner/context/`

Retirement blocker:

- no v2 context contract exists.

### 5.5 Fundamentals and Source Data

Legacy surface:

- fundamental layer wrappers;
- `scripts/fundamentals/*`;
- `scripts/data_sources/*`;
- data coverage diagnostics.

Knowledge to preserve:

- source-data readiness concepts;
- missing, partial, stale, and insufficient data handling;
- source metadata expectations;
- fundamentals history intake concepts;
- separation between data availability and investment quality.

Must be redesigned:

- raw provider or SEC capture must follow `docs/active/data_lifecycle.md`;
- normalized fundamentals input must be contract-defined;
- readiness must remain descriptive;
- financial analysis must not be reduced to source availability.

Must not be copied:

- live SEC/provider calls without approval;
- provider-specific CSV formats as v2 internal contracts;
- generated `fundamental_quality.csv` as source-of-truth;
- readiness states that imply buy/sell/hold or quality.

V2 targets:

- `docs/active/source_data_strategy.md`
- `docs/active/data_lifecycle.md`
- `docs/active/data_contracts.md`
- `docs/active/financial_analysis.md`
- `src/market_scanner/fundamentals/`

Retirement blocker:

- v2 readiness scaffold exists, but real raw-to-normalized source integration is not approved or implemented.

### 5.6 Timing Layer

Legacy surface:

- `scripts/core/build_timing_state_layer.py`
- `scripts/core/build_entry_quality_backfill.py`

Knowledge to preserve:

- timing-state concepts;
- entry-quality concepts;
- relationship between validation, fundamentals, and timing;
- historical backfill assumptions.

Must be redesigned:

- timing belongs under `src/market_scanner/timing/`;
- timing must remain classification-only until the Decision Engine uses it;
- timing must not become execution authority.

Must not be copied:

- provider fallbacks for missing historical data without approval;
- entry-quality backfill as active source-of-truth;
- timing-derived final actions.

V2 targets:

- `docs/active/pipeline_contract.md`
- `docs/active/decision_engine_contract.md`
- `src/market_scanner/timing/`

Retirement blocker:

- no v2 timing contract exists.

### 5.7 Portfolio Layer

Legacy surface:

- `scripts/core/build_portfolio_intelligence.py`
- `scripts/portfolio/*`
- `data/portfolio/*`

Knowledge to preserve:

- manual portfolio source assumptions;
- transaction-to-position concepts;
- portfolio review concepts;
- exposure and portfolio intelligence concepts.

Must be redesigned:

- manual portfolio source-of-truth must be explicitly identified;
- generated portfolio review outputs must be separated from source records;
- normalized portfolio input must be contract-defined;
- portfolio intelligence must remain classification/context until the Decision Engine acts.

Must not be copied:

- generated `portfolio_review.csv` as manual input;
- allocation decisions outside the Decision Engine;
- automatic overwrite of portfolio source records.

V2 targets:

- `docs/active/data_lifecycle.md`
- `docs/active/data_contracts.md`
- `src/market_scanner/portfolio/`

Retirement blocker:

- real portfolio source-of-truth boundary still requires explicit owner approval.

### 5.8 Decision Engine

Legacy surface:

- `scripts/core/decision_engine.py`
- legacy final-decision outputs.

Knowledge to preserve:

- downstream inputs historically considered by the old Decision Engine;
- rationale structure;
- fallback and review behavior;
- known anti-patterns around tradeability/allocation coupling.

Must be redesigned:

- final-action authority belongs only to `src/market_scanner/decisions/`;
- v2 Decision Engine must be deterministic, auditable, and contract-driven;
- upstream layers must never produce final actions.

Must not be copied:

- old final-action rules without governance review;
- hidden priority or filtering logic;
- allocation or tradeability semantics leaking upstream;
- generated final decisions as source data.

V2 targets:

- `docs/active/decision_engine_contract.md`
- `src/market_scanner/decisions/`

Retirement blocker:

- v2 Decision Engine is review-only scaffold.

### 5.9 Reporting and Communication

Legacy surface:

- `scripts/reporting/*`
- `reports/daily/*`

Knowledge to preserve:

- daily summary structure;
- user-facing report fields;
- Telegram message shape;
- warning and rationale communication patterns.

Must be redesigned:

- reporting belongs under `src/market_scanner/reporting/`;
- reporting must consume Decision Engine output;
- reporting must not create, alter, suppress, prioritize, filter, or override decisions.

Must not be copied:

- reporting-side filtering;
- Telegram delivery embedded as decision authority;
- report text as source-of-truth;
- old reports as normalized input.

V2 targets:

- `docs/active/reporting_contract.md`
- `src/market_scanner/reporting/`

Retirement blocker:

- v2 reporting scaffold is in-memory only.

### 5.10 Telegram and Operator Commands

Legacy surface:

- `scripts/telegram/process_telegram_commands.py`
- Telegram-related reporting scripts.

Knowledge to preserve:

- operator command concepts;
- Telegram UX expectations;
- manual fallback expectations;
- external delivery safety risks.

Must be redesigned:

- Telegram integration must be optional and downstream-only;
- inbound commands must not bypass v2 data contracts or Decision Engine authority;
- external API behavior must be isolated from deterministic core logic.

Must not be copied:

- Telegram scripts inside core pipeline authority;
- automatic delivery without explicit approval;
- command processing that mutates source data without contract control.

V2 targets:

- future delivery/operations contract;
- `docs/active/reporting_contract.md`.

Retirement blocker:

- no approved v2 Telegram/delivery design exists.

### 5.11 Watchlist Utilities

Legacy surface:

- `scripts/watchlist/*`
- `data/watchlist/*`

Knowledge to preserve:

- watchlist transaction/status concepts;
- candidate tracking expectations;
- links between watchlist, scanner candidates, and portfolio review.

Must be redesigned:

- watchlist data must have a lifecycle-stage classification;
- watchlist source records must not be confused with generated scanner outputs;
- watchlist classification must not create final decisions.

Must not be copied:

- old watchlist CSVs as v2 source-of-truth without approval;
- watchlist status as buy/sell/hold authority;
- utilities that mutate tracked data without contract boundaries.

V2 targets:

- `docs/active/data_lifecycle.md`
- `docs/active/data_contracts.md`

Retirement blocker:

- watchlist ownership and v2 scope are not yet defined.

### 5.12 Diagnostics and Operational Visibility

Legacy surface:

- diagnostics scripts;
- validation summaries;
- scan logs.

Knowledge to preserve:

- data coverage visibility;
- terminal/operator feedback expectations;
- scan completeness diagnostics;
- evidence needed to understand review-only or insufficient-data outcomes.

Must be redesigned:

- diagnostics should be explicit and side-effect-free where possible;
- diagnostics must remain separate from Decision Engine authority;
- operational visibility should be designed into v2 commands and reports.

Must not be copied:

- diagnostics that mutate outputs by default;
- diagnostics treated as decision rules;
- hidden terminal-only logic not captured in outputs or tests.

V2 targets:

- `docs/active/testing_strategy.md`
- `docs/active/reporting_contract.md`
- future operations/observability note.

Retirement blocker:

- v2 lacks explicit operational visibility specification.

## 6. Cross-Cutting Knowledge to Preserve

| Knowledge Area | Preserve | V2 Target | Risk if Lost |
|---|---|---|---|
| Layer sequence | old dependency order | pipeline contract | v2 may miss required inputs |
| Missing-data handling | insufficient/partial/stale concepts | data contracts/source strategy | false precision |
| Portfolio source distinction | manual source vs generated review | data lifecycle/portfolio contract | source overwrite risk |
| Source provenance | source status and metadata | normalized contracts | auditability loss |
| Report communication | user-facing summary and warnings | reporting contract | UX loss |
| Operator visibility | terminal/log feedback | future operations note | opaque debugging |
| Legacy tests | expected behavior | testing strategy | replacement regressions |

## 7. Knowledge That Must Not Be Preserved As-Is

The following patterns must not be carried into v2 unchanged:

- generated CSVs as source-of-truth;
- direct provider calls embedded in deterministic layers;
- automatic commits of generated outputs to `main`;
- upstream tradeability flags;
- hidden filtering before Decision Engine authority;
- report-side decision changes;
- Telegram delivery coupled to decision authority;
- compatibility wrappers as permanent v2 architecture;
- raw provider formats as normalized contracts;
- missing numeric values converted to zero;
- final decision rules copied without governance review.

## 8. Extraction Priority

| Priority | Area | Reason |
|---|---|---|
| P0 | Data lifecycle and source boundaries | prevents source/generated confusion |
| P0 | Portfolio source-of-truth | real portfolio data must not be overwritten |
| P0 | Decision Engine authority boundaries | prevents uncontrolled recommendations |
| P1 | Fundamentals/source-data readiness | needed before financial analysis |
| P1 | Orchestration and layer sequence | needed for v2 rebuild |
| P1 | Validation contracts | needed before production-like input |
| P2 | Reporting and Telegram UX | important after records stabilize |
| P2 | Watchlist utilities | product scope unresolved |
| P2 | Diagnostics/operator visibility | useful but non-authoritative |

## 9. Future Work Batches

Recommended sequence:

1. RESET-9C2 — Legacy Runtime Test Retirement Map.
2. RESET-9C3 — Legacy Data and Report Touchpoint Cleanup Plan.
3. RESET-10C — Portfolio Source-of-Truth Contract.
4. RESET-10D — Fundamentals Raw-to-Normalized Contract Plan.
5. RESET-10E — V2 Orchestration Contract and Dry-Run Command Design.
6. RESET-10F — Reporting Output Contract and Telegram Reapproval Decision.
7. RESET-9C5 — Approved Archive/Delete Execution Batch 1, only after replacements and approvals exist.

## 10. Stop Conditions

Do not archive, delete, or move legacy runtime if:

- v2 replacement is scaffold-only;
- manual fallback would be lost;
- source-data behavior has not been extracted;
- real portfolio source-of-truth is unclear;
- provider/network behavior is not isolated;
- tests still depend on the file;
- data/report touchpoints are unresolved;
- owner approval is missing.

## 11. Decision

Decision: PROCEED_WITH_KNOWLEDGE_EXTRACTION_BEFORE_RUNTIME_RETIREMENT

This preserves the RESET-9C conservative retirement decision and adds a knowledge extraction requirement before any archive/delete execution.

Legacy runtime remains untouched.

## 12. Recommended Next Action

Recommended next action:

```text
RESET-9C2 — Legacy Runtime Test Retirement Map
```

No runtime cleanup should occur before the test-retirement map exists.
