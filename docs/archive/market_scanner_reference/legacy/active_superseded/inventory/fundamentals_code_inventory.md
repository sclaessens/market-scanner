# Fundamentals Code Inventory

Status: ACTIVE INVENTORY

## 1. Purpose

This document is the Sprint C technical inventory for the Fundamentals Simplification Program. It records the current fundamentals-related code, tests, contracts, data artifacts, and runtime dependencies so a later Sprint D can define a developer-ready implementation specification.

This inventory does not authorize implementation, refactoring, source-data changes, generated-output updates, provider integration, pipeline changes, or runtime behavior changes.

## 2. Scope and Non-Scope

This is an inventory-only documentation artifact.

This inventory did not change code, tests, CSV files, source data, raw data, generated outputs, reports, workflow files, provider APIs, scraping behavior, pipeline behavior, or runtime behavior.

Non-scope:

- no Python edits;
- no test edits;
- no CSV edits;
- no generated-output edits;
- no source-data edits;
- no provider API calls;
- no scraping;
- no pipeline run;
- no runtime behavior changes;
- no Decision Engine authority changes.

## 3. Active Doctrine Baseline

The active target architecture for fundamentals is:

```text
raw fundamentals history
-> calculated metrics
-> quality classification
-> fundamental analysis classification
-> downstream portfolio and decision layers
```

Upstream fundamentals layers classify only. They may describe source coverage, calculation state, quality state, and business characteristics, but they may not determine allocation, ranking, scoring, tradeability, urgency, conviction, eligibility, buy/sell actions, or hidden filtering.

The Decision Engine remains the only allocation authority. Portfolio Intelligence and Reporting may consume descriptive outputs only. Reporting communicates decisions and does not create decision semantics.

## 4. Current Runtime Fundamentals Flow

The current observed runtime flow is:

```text
data/processed/scanner_ranked.csv
-> scripts/core/build_context_layer.py
-> data/processed/context_strength.csv
-> scripts/core/build_fundamental_layer.py
-> data/processed/fundamental_quality.csv
-> scripts/core/build_timing_state_layer.py
-> data/processed/timing_state_layer.csv
-> scripts/core/build_portfolio_intelligence.py
-> data/processed/portfolio_intelligence.csv
-> scripts/core/decision_engine.py
-> data/processed/final_decisions.csv
-> reporting and Telegram communication
```

Current input artifacts:

- `data/processed/context_strength.csv`
- optional `data/raw/fundamentals.csv`

Current output artifacts:

- `data/processed/fundamental_quality.csv`
- `data/logs/fundamental_layer_log.csv`

Current upstream dependency:

- the Fundamental Layer depends on Context Layer rows with `ticker` and `date`.

Current downstream consumers:

- Timing State consumes `fundamental_quality.csv`;
- Portfolio Intelligence consumes Timing State output and preserves upstream fundamentals columns;
- Decision Engine consumes Portfolio Intelligence output and requires `quality_state` among its inputs;
- reporting consumes final decisions only.

Current MVP differences from the target architecture:

- `data/raw/fundamentals.csv` is metric-like rather than raw multi-year financial statement history;
- `as_of_date` currently carries matching semantics that should be split in the target model;
- no separate `fundamental_metrics.csv` artifact exists;
- no separate `fundamental_analysis.csv` artifact exists;
- `build_fundamental_layer.py` currently combines source availability, provenance checks, metric sufficiency, freshness checks, quality state, and descriptive profile output in one layer;
- downstream layers expect the current `fundamental_quality.csv` shape, so migration will need compatibility planning.

## 5. Script Inventory

| File | Current role | Reads | Writes | Key functions/classes | Current status | Future classification | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/core/build_fundamental_layer.py` | Current Fundamental Layer builder | `data/processed/context_strength.csv`; optional `data/raw/fundamentals.csv` | `data/processed/fundamental_quality.csv`; `data/logs/fundamental_layer_log.csv` | `build_fundamental_layer`; `_build_from_raw_fundamentals`; `_normalize_raw_fundamentals`; `_classify_raw_match`; `_write_log` | Active MVP | COMPATIBILITY_WRAPPER_CANDIDATE | Central migration surface. It currently combines raw source matching, provenance validation, freshness checks, metric sufficiency, and quality/profile classification. |
| `scripts/core/build_context_layer.py` | Upstream Context Layer builder | `data/processed/scanner_ranked.csv`; optional `data/processed/sector_relative_strength.csv` | `data/processed/context_strength.csv`; context log | `build_context_layer`; `classify_context`; relative-strength helpers | Active upstream dependency | KEEP | Provides `ticker` and `date` rows consumed by the Fundamental Layer. No fundamentals implementation should move allocation semantics into this layer. |
| `scripts/core/build_timing_state_layer.py` | Downstream Timing State builder | `data/processed/fundamental_quality.csv`; optional `data/processed/entry_quality_metrics.csv` | `data/processed/timing_state_layer.csv`; timing log | `build_timing_state_layer`; timing classification helpers | Active downstream consumer | DOWNSTREAM_CONSUMER_ONLY | Preserves upstream fundamentals columns and appends timing classification. Future migration must preserve or explicitly bridge the input contract. |
| `scripts/core/build_portfolio_intelligence.py` | Downstream Portfolio Intelligence builder | `data/processed/timing_state_layer.csv`; `data/portfolio/portfolio_positions.csv`; optional `data/portfolio/portfolio_metadata.csv` | `data/processed/portfolio_intelligence.csv`; portfolio log | `build_portfolio_intelligence`; portfolio metadata validation helpers | Active downstream consumer | DOWNSTREAM_CONSUMER_ONLY | Preserves fundamentals and timing fields while adding descriptive portfolio metadata. It must not receive allocation authority. |
| `scripts/core/decision_engine.py` | Allocation authority and final decision builder | `data/processed/portfolio_intelligence.csv` | `data/processed/final_decisions.csv`; decision log | `build_decision_engine`; decision row helpers | Active allocation authority | KEEP | Only downstream consumer with allocation authority. Future fundamentals changes must not loosen or bypass this boundary. |
| `scripts/run_scan.py` | Main pipeline orchestration | Layer inputs across scanner, validation, context, fundamentals, timing, portfolio, decision, reporting | Layer outputs and reports through called builders | `main` | Active orchestrator | ORCHESTRATION_TOUCHPOINT | Currently calls one Fundamental Layer step. Future metrics, quality, and analysis builders would require an approved orchestration specification. |
| `scripts/run_full_pipeline.py` | Full-pipeline wrapper | Runtime arguments and `scripts/run_scan.py` | Pipeline outputs through delegated run | `main` | Active wrapper | ORCHESTRATION_TOUCHPOINT | Thin wrapper; future changes should follow the approved orchestration plan rather than direct ad hoc changes. |
| `scripts/data_sources/prefill_fundamentals.py` | Provider/operator-assisted prefill for current fundamentals MVP | Operator/provider export CSV | Optional `data/raw/fundamentals.csv` | `prepare_fundamentals_prefill`; `main` | Active support utility for current MVP | REPLACEMENT_CANDIDATE | Current output schema is metric-like and uses fields such as `source_last_updated`, `report_period`, and `free_cash_flow_positive`, which do not fully align with the current Fundamental Layer raw schema or the target raw-history model. |
| `scripts/data_sources/common.py` | Shared source-data prefill validation helpers | Input dataframes and configured paths | Validated dataframe outputs or governed writes through callers | `load_input`; `normalize_ticker_series`; `validate_forbidden_columns`; `write_output`; schema helpers | Active support utility | KEEP | Likely reusable for future governed raw-history intake if contracts are updated first. |
| `scripts/data_sources/prefill_portfolio_metadata.py` | Provider/operator-assisted prefill for portfolio metadata | Operator/provider export CSV | Optional `data/portfolio/portfolio_metadata.csv` | `prepare_portfolio_metadata_prefill`; `main` | Active portfolio metadata utility | KEEP | Related to source-data workflow but not directly part of fundamentals architecture. Keep separate from fundamentals migration. |
| `scripts/diagnostics/audit_data_coverage.py` | Read-only coverage diagnostics for source artifacts | scanner output, portfolio metadata, `data/raw/fundamentals.csv` | Diagnostic console output only | `audit_fundamentals`; `audit_portfolio_metadata`; `run_audit`; `main` | Active diagnostic utility | REFACTOR_CANDIDATE | Tied to current `fundamentals.csv` schema and date matching. Future diagnostics should be re-specified around raw history, metrics, quality, and analysis artifacts. |
| `scripts/reporting/build_reporting_layer.py` | Reporting communication layer | `data/processed/final_decisions.csv` and reporting inputs | reports | reporting builder functions | Active downstream communication | DOWNSTREAM_CONSUMER_ONLY | No direct fundamentals authority. It should continue to communicate final decisions only. |
| `scripts/reporting/build_telegram_summary.py` | Telegram communication layer | final decision/reporting artifacts | Telegram summary output | summary builder functions | Active downstream communication | DOWNSTREAM_CONSUMER_ONLY | No direct fundamentals authority. It should not infer urgency or allocation semantics. |

## 6. Test Inventory

| Test file | Current coverage | Related runtime file | Future relevance | Gaps | Notes |
|---|---|---|---|---|---|
| `tests/core/test_build_fundamental_layer.py` | Current Fundamental Layer schema, row preservation, missing raw artifact behavior, raw date matching, provenance validation, stale/future source freshness, partial/missing/invalid metric handling, duplicate raw rows, forbidden semantics | `scripts/core/build_fundamental_layer.py` | High | Needs future coverage for raw history, metrics artifact, analysis artifact, compatibility mode, and date-field split | Strong reuse for row preservation and no-allocation doctrine tests. |
| `tests/data_sources/test_prefill_fundamentals.py` | Current prefill schema, dry-run behavior, required columns, duplicates, malformed dates, partial/stale diagnostics, overwrite guard, forbidden semantic columns | `scripts/data_sources/prefill_fundamentals.py` | Medium | Needs replacement or expansion for raw financial statement history intake | Current tests protect the MVP but are not sufficient for `fundamentals_history.csv`. |
| `tests/diagnostics/test_audit_data_coverage.py` | Read-only coverage diagnostics for portfolio metadata and current fundamentals, date mismatch reporting, duplicate detection, target universe handling, no generated artifact writes, no reporting/Telegram dependency | `scripts/diagnostics/audit_data_coverage.py` | Medium | Needs future diagnostics for raw history, metrics completeness, quality, and analysis outputs | Useful as a pattern for inspection-only diagnostics. |
| `tests/core/test_build_context_layer.py` | Context output contract and upstream row structure | `scripts/core/build_context_layer.py` | High | No direct raw-history gaps | Protects the current upstream `ticker` and `date` contract. |
| `tests/core/test_build_timing_state_layer.py` | Timing consumes and preserves `fundamental_quality.csv`, handles auxiliary metrics, validates forbidden semantics | `scripts/core/build_timing_state_layer.py` | High | Needs compatibility tests if `fundamental_quality.csv` schema changes | Important downstream migration safety net. |
| `tests/core/test_build_portfolio_intelligence.py` | Portfolio Intelligence consumes Timing State, preserves upstream columns, appends portfolio metadata classifications, validates forbidden semantics | `scripts/core/build_portfolio_intelligence.py` | High | Needs tests if future fundamental analysis fields are passed through | Protects row preservation and downstream descriptive-only behavior. |
| `tests/core/test_decision_engine.py` | Decision Engine input requirements, allocation authority behavior, metadata/fundamental quality dependency as downstream inputs | `scripts/core/decision_engine.py` | High | Needs compatibility tests if upstream quality state names or required columns change | Must remain the only allocation authority. |
| `tests/reporting/test_build_reporting_layer.py` | Reporting from final decisions | `scripts/reporting/build_reporting_layer.py` | Medium | No direct fundamentals tests unless final decision schema changes | Reporting should remain communication-only. |
| `tests/reporting/test_build_telegram_summary.py` | Telegram summary from final decisions/reporting artifacts | `scripts/reporting/build_telegram_summary.py` | Medium | No direct fundamentals tests unless final decision schema changes | Should not become a fundamentals analysis layer. |
| `tests/data_sources/test_prefill_common.py` | Shared prefill validation helpers | `scripts/data_sources/common.py` | Medium | May need new helper cases for raw-history contracts | Reusable for future source-data intake validation. |
| `tests/data_sources/test_prefill_portfolio_metadata.py` | Portfolio metadata prefill workflow | `scripts/data_sources/prefill_portfolio_metadata.py` | Low for fundamentals; medium for source-data governance | No direct fundamentals gap | Keep separate from fundamentals raw-history migration. |
| `tests/test_operator_visibility.py` | Operator-facing pipeline visibility and output expectations | pipeline orchestration | Medium | May need future updates if new approved steps are added | Useful for orchestration visibility after Sprint D/E approval. |

## 7. Data Artifact Inventory

| Artifact | Current role | Tracked/ignored/generated | Current schema concern | Target architecture relationship | Notes |
|---|---|---|---|---|---|
| `data/raw/fundamentals.csv` | Current raw-like MVP source for Fundamental Layer | Source artifact; local raw-data handling should remain governed and not regenerated by this inventory | Metric-like fields by `ticker` and `as_of_date`; no multi-year statement history; `as_of_date` is overloaded; schema alignment differs between prefill/diagnostics and current layer expectations | Temporary MVP input, not target source of truth | Header observed includes `ticker`, `as_of_date`, provenance fields, currency, selected metrics, and notes. |
| `data/raw/fundamentals_history.csv` | Future raw fundamentals history source of truth | Future artifact; not observed as current runtime artifact | Not yet implemented | Target Raw Fundamentals History Layer | Should store raw reported statement values by ticker and fiscal period, without analysis or allocation semantics. |
| `data/processed/fundamental_quality.csv` | Current generated Fundamental Layer output | Generated processed artifact | Combines quality state, provenance, metric sufficiency, freshness, and descriptive profile columns | Future Quality Layer compatibility output | Downstream layers currently depend on this shape, especially `quality_state`. |
| `data/processed/fundamental_metrics.csv` | Future calculated metrics artifact | Future generated artifact | Not yet implemented | Target Fundamental Metrics Layer | Should contain deterministic calculations only and no ranking, scoring, or allocation authority. |
| `data/processed/fundamental_analysis.csv` | Future descriptive analysis artifact | Future generated artifact | Not yet implemented | Target Fundamental Analysis Layer | Should classify business characteristics descriptively only. |
| `data/processed/context_strength.csv` | Current upstream input to Fundamental Layer | Generated processed artifact | Provides `ticker` and `date` used for current raw matching | Upstream dependency remains | Future builders must preserve row alignment with context opportunities. |
| `data/processed/timing_state_layer.csv` | Current downstream consumer of `fundamental_quality.csv` | Generated processed artifact | Includes all current fundamental quality fields plus timing classifications | Downstream compatibility consumer | Migration must avoid breaking row preservation or required input columns. |
| `data/processed/portfolio_intelligence.csv` | Current downstream consumer of Timing State and portfolio metadata | Generated processed artifact | Preserves fundamentals and timing fields, then adds portfolio metadata fields | Downstream compatibility consumer | Decision Engine consumes this artifact. |
| `data/processed/final_decisions.csv` | Current Decision Engine output | Generated processed artifact | Requires selected upstream states including `quality_state` and portfolio metadata status | Allocation authority output | Must not be produced or changed by upstream fundamentals migration except through approved downstream compatibility. |

## 8. Contract Gap Inventory

- Current `data/raw/fundamentals.csv` is metric-like rather than raw-history based.
- Current `as_of_date` semantics are overloaded. The target model should split date semantics into fields such as `period_end_date`, `report_date`, `source_freshness_date`, and `extraction_date`.
- No separate `data/processed/fundamental_metrics.csv` layer exists yet.
- No separate `data/processed/fundamental_analysis.csv` layer exists yet.
- The current Fundamental Layer combines source presence, provenance checks, source freshness, metric sufficiency, metadata status, descriptive profiles, and quality classification.
- Current downstream layers expect the existing `data/processed/fundamental_quality.csv` shape.
- The current source-data utility and diagnostics paths are not fully aligned with the current Fundamental Layer raw schema. In particular, `prefill_fundamentals.py` and `audit_data_coverage.py` reference current-MVP fields such as `source_last_updated`, `report_period`, and `free_cash_flow_positive`, while the Fundamental Layer expects `source_reference`, `source_freshness_date`, and a broader metric list.
- Current orchestration calls one Fundamental Layer builder. The target architecture likely needs an approved builder sequence for raw history, metrics, quality, and analysis.
- Existing tests are strong for the current MVP, but future raw-history and metrics contracts need new fixtures and compatibility tests.

## 9. Migration Risk Inventory

| Risk | Description | Mitigation for future planning |
|---|---|---|
| Pipeline compatibility break | Replacing the current builder sequence could break `run_scan.py` and `run_full_pipeline.py`. | Define a compatibility strategy before implementation and add orchestration tests. |
| Decision Engine authority drift | New analysis states could be misused as allocation, ranking, or eligibility signals outside Decision Engine. | Keep all upstream outputs descriptive and preserve Decision Engine-only allocation doctrine in specs and tests. |
| Output schema breakage | Timing, Portfolio Intelligence, Decision Engine, and reporting expect current generated artifact shapes. | Maintain `fundamental_quality.csv` compatibility or introduce an explicit migration bridge. |
| Generated artifact drift | Running builders during migration could alter generated outputs unintentionally. | Keep future implementation validation explicit and separate from inventory or spec work. |
| Source-data inference | Raw history migration may tempt derived fields or precomputed ratios in raw data. | Define raw statement fields only and calculate metrics downstream. |
| Stale local ignored data | Local raw files can drift from documentation and tests. | Add explicit freshness, extraction, and fixture policies in Sprint D. |
| Date semantics ambiguity | `as_of_date` currently covers multiple meanings. | Split period, report, source freshness, and extraction dates in the new contract. |
| Downstream test churn | Existing tests are tied to the current `fundamental_quality.csv` behavior. | Reuse row-preservation and forbidden-semantics tests while adding compatibility and new-layer tests. |

## 10. Recommended Future Implementation Sequence

This is a future sequence only and does not authorize implementation.

1. Define raw history schema and fixtures.
2. Define metrics builder spec.
3. Define quality builder compatibility strategy.
4. Define analysis builder spec.
5. Define migration from current `fundamentals.csv`.
6. Define tests.
7. Implement only after approval.

## 11. Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Existing backlog coverage appears sufficient for the known fundamentals simplification, source-data strategy, raw-history architecture, and compatibility workstreams.

## 12. Validation

Commands run for this inventory:

- `git checkout main`
- `git pull origin main`
- `git status`
- `git checkout -b docs/sprint-c-fundamentals-code-inventory`
- `find scripts -type f -name "*.py" | sort`
- `find tests -type f -name "*.py" | sort`
- `rg -n "fundamental" scripts tests docs/active docs/sprints docs/archive | head -200`
- `rg -n "fundamentals" scripts tests docs/active docs/sprints docs/archive | head -200`
- `rg -n "source_data" scripts tests docs/active docs/sprints docs/archive | head -200`
- `rg -n "provider" scripts tests docs/active docs/sprints docs/archive | head -200`
- `rg -n "portfolio_metadata" scripts tests docs/active docs/sprints docs/archive | head -200`
- `head -1 data/raw/fundamentals.csv`
- `head -1 data/processed/fundamental_quality.csv`
- `head -1 data/processed/context_strength.csv`
- `head -1 data/processed/timing_state_layer.csv`
- `head -1 data/processed/portfolio_intelligence.csv`
- `head -1 data/processed/final_decisions.csv`

Validation status:

- only documentation files changed;
- no code files changed;
- no tests changed;
- no CSV files changed;
- no raw data changed;
- no generated files changed;
- no workflow files changed;
- no provider APIs called;
- no scraping performed;
- no pipeline run;
- no tests run.
