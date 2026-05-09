# Sprint 4 Architecture Validation — Timing State Layer

## 1. Executive Architecture Conclusion

Architecture validation status: PASS.

The proposed Sprint 4 Timing State Layer is technically coherent inside the certified architecture as a descriptive, classification-only, enrichment-only layer, provided future execution planning enforces the guardrails in this document.

The layer can technically exist after the certified Fundamental Quality Layer and before downstream portfolio, Decision Engine, and reporting consumption. Because the certified architecture names the watchlist position as timing-state tracking, the safest implementation interpretation is a governance-clean Timing/Watchlist classification stage that preserves the upstream row universe and does not reuse legacy watchlist inclusion, readiness, status sorting, or action-state behavior as-is.

Sprint 4 may proceed to execution planning. No implementation is authorized by this validation.

## 2. Documents Reviewed

Governance baseline:

- `AGENTS.md`
- `README.md`
- `docs/sprints/README.md`
- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/execution_roadmap_v2.md`
- `docs/sprints/sprint_3_closeout.md`

Sprint 4 governance and preparation:

- `docs/audits/sprint_4_governance_audit.md`
- `docs/sprints/sprint_4_timing_state_layer.md`
- `docs/sprints/sprint_4_governance_constraints.md`
- `docs/sprints/sprint_4_boundary_controls.md`
- `docs/sprints/sprint_4_execution_plan.md`

Architecture references:

- `docs/technical/Technical_Analysis_v3.md`
- `docs/functional/Functional_Analysis_v2.md`
- `docs/execution/execution_delivery_framework_v2.md`
- `docs/technical/decision_engine_design_v2.md`

## 3. Runtime Structure Inspected

Read-only runtime inspection covered:

- `scripts/`
- `scripts/core/`
- `scripts/watchlist/`
- `scripts/portfolio/`
- `scripts/reporting/`
- `tests/`
- `data/processed/`
- `data/logs/`

Key files inspected:

- `scripts/run_full_pipeline.py`
- `scripts/run_scan.py`
- `scripts/core/build_validation_layer.py`
- `scripts/core/build_context_layer.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/core/decision_engine.py`
- `scripts/watchlist/build_watchlist.py`
- `scripts/watchlist/evaluate_watchlist.py`
- `scripts/portfolio/build_portfolio.py`
- `scripts/reporting/build_telegram_summary.py`
- `tests/core/test_build_fundamental_layer.py`

No runtime files were modified.

## 4. Pipeline Position Assessment

Result: VALIDATED WITH GUARDRAILS.

Certified architecture:

scanner -> validation_layer -> context_layer -> fundamental_layer -> watchlist -> portfolio -> decision_engine -> reporting

Validated Sprint 4 placement:

scanner -> validation_layer -> context_layer -> fundamental_layer -> timing/watchlist classification -> portfolio -> decision_engine -> reporting

The Timing State Layer should be placed after `fundamental_quality.csv` so the full certified upstream classified opportunity universe is available before timing metadata is appended.

The layer must sit before portfolio, Decision Engine, and reporting. It must not be a downstream Decision Engine helper and must not read `final_decisions.csv`.

The phrase "before watchlist" is technically ambiguous because certified architecture identifies watchlist as the timing-state tracking layer. Architecture validation resolves this as follows:

- Sprint 4 may be implemented inside the watchlist/timing layer position.
- If a separate runtime file is created, it must sit after Fundamental and before any watchlist inclusion/action-state behavior.
- It must not alter `watchlist_active.csv` inclusion logic.
- It must not use Timing state to add, remove, prioritize, or sort watchlist membership.

## 5. Layer Responsibility Validation

Result: PASS.

The future Timing State Layer may:

- classify timing-condition observations
- classify extension and compression conditions
- classify pullback, breakout, consolidation, participation, and pattern observations
- append descriptive timing metadata
- emit distribution logs

It may not:

- allocate capital
- authorize execution
- produce BUY/SELL/REMOVE semantics
- produce tradeability semantics
- produce actionability semantics
- produce urgency semantics
- produce conviction semantics
- produce ranking semantics
- produce scoring semantics
- suppress opportunities
- filter opportunities
- reorder opportunities
- prioritize opportunities
- mutate upstream classifications
- reinterpret Validation, Context, or Fundamental outputs
- dilute Decision Engine authority

## 6. Enrichment-Only Feasibility Assessment

Result: FEASIBLE.

A future implementation can technically be written as a pure enrichment stage using the established Sprint 3 pattern:

- load an authoritative upstream row universe
- validate `ticker` and `date`
- fail fast on duplicate row keys
- preserve upstream order
- append only descriptive metadata columns
- write a processed output
- write an audit log
- avoid downstream interpretation

The current Fundamental Layer implementation demonstrates a governance-clean pattern for row preservation, missing-data behavior, duplicate-key checks, fixed schema, and log output. Sprint 4 should reuse that architectural pattern conceptually, without copying Fundamental semantics.

## 7. Data Contract Assessment

Result: ADDITIONAL DATA CONTRACT DETAIL REQUIRED DURING EXECUTION PLANNING.

Existing data contracts are sufficient to prove feasibility, but not sufficient for implementation authorization.

Available upstream row-universe candidates:

- `data/processed/fundamental_quality.csv`
- `data/processed/context_strength.csv`
- `data/processed/scanner_ranked.csv`

Validated contract direction:

- `fundamental_quality.csv` should be the preferred authoritative row universe because Sprint 4 follows Fundamental in the certified pipeline.
- `scanner_ranked.csv`, `entry_quality_metrics.csv`, and per-ticker OHLCV files may be considered read-only timing source data only after execution planning defines exact joins and missing-data behavior.
- Timing must never require `final_decisions.csv`, portfolio files, reporting files, or manual watchlist action files as authoritative upstream row universe.

Required future data contract documentation:

- authoritative input source
- allowed auxiliary timing source files
- primary row key, expected to be `ticker` + `date`
- duplicate-key behavior
- missing-key behavior
- missing timing-source behavior
- upstream-order preservation rule
- output artifact name
- output schema
- log schema

## 8. Schema Direction Assessment

Result: PASS.

The current schema candidates are governance-safe because they are descriptive and explicitly non-final:

- `timing_state`
- `timing_reason`
- `breakout_state`
- `pullback_state`
- `compression_state`
- `extension_state`
- `participation_state`
- `timing_environment`
- `timing_metadata_status`
- `timing_pattern_state`
- `trend_participation_state`
- `timing_structure_state`
- `source_data_status`

Execution planning must avoid schema names that imply:

- readiness
- quality
- rank
- score
- priority
- actionability
- approval
- rejection
- execution
- allocation
- conviction
- recommendation

Existing runtime fields such as `status`, `READY`, `FAILED`, and status-based sort order in `scripts/watchlist/evaluate_watchlist.py` are not governance-safe as a direct Sprint 4 schema pattern.

## 9. Distribution-Preservation Feasibility Assessment

Result: FEASIBLE AND MEASURABLE.

Future implementation can preserve:

- row count
- ticker universe
- ticker/date key set
- upstream ordering
- upstream distribution shape
- opportunity visibility

Required mechanics:

- output rows must be initialized from the authoritative upstream row universe
- auxiliary timing data must be left-joined or looked up without row loss
- missing auxiliary data must produce descriptive missing-source metadata
- no filtering by Timing state
- no sorting by Timing state
- no grouping into preferred subsets
- no deduplication except fail-fast rejection of upstream duplicate row keys

## 10. Cross-Layer Coupling Assessment

Result: PASS WITH IMPLEMENTATION GUARDRAILS.

Validation coupling risk:

- Timing must not become a second Validation Layer.
- Timing may use technical measurements as observations only.
- Timing may not change `structure_state`, `valid_setup`, or validation reasons.

Context coupling risk:

- Timing must not use leadership to alter timing labels.
- Timing must not synthesize leadership-confirmed timing or cross-sectional opportunity preference.

Fundamental coupling risk:

- Timing must not use quality metadata to alter timing labels.
- Timing must not create quality-adjusted timing or preferred setup semantics.

Portfolio coupling risk:

- Timing must not read portfolio state or open positions to classify timing.

Reporting coupling risk:

- Timing must not change report grouping, report action sections, or communication priority.

## 11. Decision Engine Boundary Assessment

Result: PASS WITH STRONG GUARDRAILS REQUIRED.

The current Decision Engine already interprets timing values as part of final allocation decisions. This is acceptable only because Decision Engine owns allocation authority.

Future Sprint 4 implementation must not move any of the following out of `scripts/core/decision_engine.py`:

- final action
- tradeability
- conviction
- allocation priority
- execution style
- urgency
- BUY logic
- SELL logic
- REMOVE logic
- portfolio-aware allocation

Timing output must be treated as raw descriptive input to the Decision Engine, not as a precomputed allocation recommendation.

## 12. Watchlist / Portfolio / Reporting Boundary Assessment

Result: PASS WITH LEGACY WATCHLIST CAUTION.

Watchlist boundary:

- Existing `scripts/watchlist/build_watchlist.py` manages manual watch/unwatch activity and active inclusion.
- Existing `scripts/watchlist/evaluate_watchlist.py` uses readiness-style states, thresholds, failure states, and status sorting.
- A future Sprint 4 implementation must not reuse that behavior as-is for a certified Timing State Layer.
- Timing must not add, remove, or reorder active watchlist membership.

Portfolio boundary:

- Existing portfolio builders derive position and risk-state artifacts from portfolio transactions and price data.
- Timing must not read or modify portfolio artifacts.
- Portfolio must remain downstream exposure/risk-state modelling only.

Reporting boundary:

- Existing reporting reads `final_decisions.csv` and communicates final actions.
- Timing must not alter reporting action sections, summary prioritization, or omission behavior.
- Reporting may communicate Timing metadata later only after Decision Engine-owned interpretation and reporting governance permit it.

## 13. Failure Mode Assessment

The following failure modes are technically real and must be controlled in execution planning:

| Failure Mode | Architecture Risk | Required Control |
|---|---|---|
| Timing metadata becomes execution readiness | Decision Engine leakage | Ban readiness/actionability fields and values |
| Timing state becomes watchlist inclusion logic | Hidden filtering | Do not modify `watchlist_active.csv` or watch/unwatch behavior |
| Extension state becomes rejection logic | Opportunity suppression | Preserve all rows and treat extension as descriptive |
| Compression state becomes priority logic | Hidden prioritization | Preserve upstream ordering and ban priority fields |
| Breakout state becomes BUY logic | Allocation leakage | Ban BUY/SELL semantics outside Decision Engine |
| Pullback state becomes actionable logic | Execution leakage | Ban actionability and execution-ready fields |
| Pattern state becomes ranking logic | Ranking authority leakage | Ban rank/order-by-pattern behavior |
| Timing environment becomes conviction logic | Conviction leakage | Ban conviction and quality-like fields |
| Metadata status becomes approval/rejection | Hidden gating | Use source/status availability language only |
| Timing layer becomes second Validation layer | Layer contamination | Do not mutate or reinterpret Validation output |
| Timing layer becomes mini Decision Engine | Allocation leakage | Do not emit final action, tradeability, priority, score, or conviction |
| Timing layer changes reporting behavior | Reporting leakage | Reporting remains final-decision communication only |
| Timing layer suppresses opportunities | Distribution collapse | Row-count and key-set equality checks |
| Timing layer reorders opportunities | Priority leakage | Preserve upstream order |
| Timing layer mutates upstream classifications | Contract corruption | Copy upstream keys only; append new fields only |

## 14. Required Implementation Guardrails

Future developer specification must require:

- new governance-clean Timing builder or explicitly purified implementation scope
- no direct reuse of legacy readiness/status-sorting behavior
- authoritative upstream row universe from the certified upstream layer
- auxiliary source data treated as optional metadata input
- one output row per upstream `ticker` + `date`
- fail-fast duplicate-key validation
- fail-fast missing key validation
- no row suppression under missing timing data
- no Timing-state sorting
- no watchlist inclusion changes
- no portfolio reads
- no Decision Engine reads
- no reporting changes
- fixed output schema approved before implementation
- forbidden-field and forbidden-value checks
- deterministic output with stable upstream-order preservation

## 15. Required Future Test Categories

Future tests are required, but not created during this validation.

Required categories:

- output schema order and field allowlist
- forbidden-field absence
- forbidden semantic-value absence
- row-count preservation
- ticker/date key-set preservation
- upstream-order preservation
- duplicate ticker/date fail-fast behavior
- missing ticker/date fail-fast behavior
- missing auxiliary timing-source preservation
- no filtering by extension, compression, breakout, pullback, pattern, or metadata status
- no sorting by Timing state
- no mutation of upstream fields
- deterministic repeated runs
- log schema and count consistency

## 16. Required Future Logging / Audit Requirements

Future implementation should write a Timing audit log.

Required log direction:

- generated timestamp
- input row count
- output row count
- unique ticker/date count
- duplicate ticker/date count
- missing auxiliary source count
- timing-state distribution
- extension-state distribution
- compression-state distribution
- pullback-state distribution
- breakout-state distribution
- metadata-source-status distribution

Logging must remain observational only and must not create priority, readiness, actionability, approval, rejection, conviction, allocation, or execution semantics.

## 17. Required Documentation Corrections

None.

No blocking architecture corrections are required.

No documentation corrections are required before Sprint 4 execution planning.

No backlog item is required from this validation because the required data-contract and implementation guardrails are immediate execution-planning inputs, not deferred work.

## 18. Recommended Next Step

Proceed to Sprint 4 execution planning.

Execution planning must define:

- exact upstream row-universe artifact
- exact auxiliary timing source artifacts
- exact output artifact name
- exact schema direction
- row-key contract
- missing-data behavior
- forbidden-field enforcement
- future validation commands
- implementation file boundaries
- explicit exclusion of legacy readiness/status-sorting behavior unless purified under Technical Lead specification

Execution planning must not authorize developer implementation.

## 19. Final Architecture Validation Decision

SPRINT 4 ARCHITECTURE VALIDATED — READY FOR EXECUTION PLANNING
