# Sprint 4 Developer Specification — Timing State Layer

## 1. Executive Specification Conclusion

Status: READY FOR SPRINT 4 DEVELOPMENT

This specification authorizes a future developer execution task only within the scope defined here.

This document does not implement runtime code, tests, schemas, generated artifacts, strategy logic, thresholds, filters, ranking logic, scoring logic, Decision Engine logic, watchlist behavior, portfolio behavior, or reporting behavior.

The future implementation must create a governance-clean Timing State Layer as a descriptive, classification-only, enrichment-only stage. It must not reuse legacy watchlist readiness/status-sorting semantics as-is.

## 2. Certified Baseline

Sprint 4 current state:

- Sprint 0 = CERTIFIED COMPLETE / CLOSED
- Sprint 1 = CERTIFIED COMPLETE / CLOSED
- Sprint 2 = CERTIFIED COMPLETE / CLOSED
- Sprint 3 = CERTIFIED COMPLETE / CLOSED
- Sprint 4 preparation = COMPLETE
- Sprint 4 governance audit = PASS
- Sprint 4 architecture validation = PASS
- Sprint 4 execution planning = COMPLETE
- Sprint 4 developer specification = COMPLETE after this document

Certified architecture:

scanner -> validation_layer -> context_layer -> fundamental_layer -> timing_state_layer -> watchlist -> portfolio -> decision_engine -> reporting

## 3. Governance Inheritance

Sprint 4 inherits the certified doctrine:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority
- upstream layers classify only
- reporting communicates only
- no upstream tradeability
- no hidden filtering
- no hidden allocation semantics outside Decision Engine
- no decision semantics outside Decision Engine
- no ranking authority outside Decision Engine
- no scoring authority outside Decision Engine
- distribution preservation is mandatory

The Timing State Layer must remain descriptive metadata only.

## 4. Architecture Validation Inheritance

This specification inherits `docs/audits/sprint_4_architecture_validation.md`.

Validated architecture requirements:

- Timing is feasible as pure enrichment
- Timing sits after Fundamental and before watchlist / portfolio / Decision Engine / reporting consumption
- preferred upstream row universe is `data/processed/fundamental_quality.csv`
- auxiliary timing sources may be read only as optional descriptive inputs
- Timing must not read Decision Engine, portfolio, reporting, or manual watchlist action files
- legacy watchlist readiness/status-sorting behavior must not be reused as-is

## 5. Execution Planning Inheritance

This specification inherits `docs/sprints/sprint_4_execution_planning.md`.

Execution planning requires:

- exact implementation file scope
- exact input/output/log contracts
- exact forbidden semantics
- exact distribution-preservation checks
- exact non-mutating enrichment checks
- exact future tests
- exact validation commands
- exact developer and Technical Lead acceptance criteria

## 6. Future Implementation Objective

Implement a standalone governance-clean Timing State Layer builder that:

- reads the certified upstream row universe
- preserves every upstream ticker/date row
- preserves upstream ordering
- appends descriptive timing metadata
- handles missing auxiliary timing data without row loss
- writes deterministic output
- writes deterministic audit logs
- exposes no allocation, execution, tradeability, urgency, conviction, ranking, scoring, priority, actionability, recommendation, or BUY/SELL semantics

## 7. In-Scope Future Implementation

Future developer execution may:

- create `scripts/core/build_timing_state_layer.py`
- create `tests/core/test_build_timing_state_layer.py`
- generate `data/processed/timing_state_layer.csv` during validation
- generate `data/logs/timing_state_layer_log.csv` during validation
- update `docs/sprints/sprint_4_developer_spec.md` only for implementation notes if required
- update `docs/sprints/sprint_status_tracker.md` only for approved Sprint 4 lifecycle status updates
- update `docs/sprints/project_backlog.md` only if a deferred non-blocking item is discovered

The initial Sprint 4 implementation must be standalone. It must not integrate the builder into `scripts/run_scan.py` unless a later Technical Lead instruction explicitly expands implementation scope.

## 8. Out-of-Scope Boundaries

Out of scope:

- `scripts/run_scan.py` orchestration changes
- `scripts/run_full_pipeline.py` orchestration changes
- Decision Engine changes
- portfolio changes
- reporting changes
- Telegram changes
- legacy watchlist behavior changes
- strategy optimization
- threshold tuning
- new filters
- ranking logic
- scoring logic
- allocation logic
- execution logic
- generated artifact commits unless explicitly authorized
- runtime output consumption by downstream layers

## 9. Future File Creation Scope

Future developer execution may create:

- `scripts/core/build_timing_state_layer.py`
- `tests/core/test_build_timing_state_layer.py`

Future validation may generate:

- `data/processed/timing_state_layer.csv`
- `data/logs/timing_state_layer_log.csv`

Generated files must remain untracked unless explicitly authorized.

## 10. Future File Update Scope

Future developer execution may update:

- `docs/sprints/sprint_4_developer_spec.md`, implementation notes only if needed
- `docs/sprints/sprint_status_tracker.md`, approved lifecycle status only
- `docs/sprints/project_backlog.md`, deferred item capture only if needed

No other existing source files may be modified without explicit Technical Lead expansion.

## 11. Files Forbidden From Modification

Do not modify:

- `scripts/core/decision_engine.py`
- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `scripts/core/build_validation_layer.py`
- `scripts/core/build_context_layer.py`
- `scripts/core/build_fundamental_layer.py`
- `scripts/watchlist/`
- `scripts/portfolio/`
- `scripts/reporting/`
- `scripts/telegram/`
- existing tests outside `tests/core/test_build_timing_state_layer.py`
- existing generated CSV/data files outside future Timing validation artifacts
- config threshold files
- strategy/scanner ranking logic

## 12. Pipeline Insertion Point

Target certified placement:

scanner -> validation_layer -> context_layer -> fundamental_layer -> timing_state_layer -> watchlist -> portfolio -> decision_engine -> reporting

Initial implementation must not wire this stage into the runtime pipeline. It must create a standalone builder whose output contract is ready for later pipeline integration after audit approval.

Future pipeline integration, if later authorized, must call Timing after `build_fundamental_layer()` and before any watchlist, portfolio, Decision Engine, or reporting consumption.

## 13. Input Contract

Authoritative input:

- `data/processed/fundamental_quality.csv`

Required input columns:

- `ticker`
- `date`

Protected optional upstream fields, if present:

- `quality_state`
- `quality_reason`
- quality profile metadata
- `quality_metadata_status`
- `source_data_status`

Allowed auxiliary descriptive inputs:

- `data/processed/scanner_ranked.csv`
- `data/processed/entry_quality_metrics.csv`
- per-ticker OHLCV files under `data/processed/`

Auxiliary data may provide descriptive timing observations only. It must not control row inclusion.

Forbidden inputs:

- `data/processed/final_decisions.csv`
- portfolio files
- reporting files
- Telegram files
- manual watchlist command/action files
- `data/watchlist/watchlist_active.csv` as row-universe authority
- `data/watchlist/watchlist_status.csv` as row-universe authority

## 14. Output Contract

Required output:

- `data/processed/timing_state_layer.csv`

Required output schema, in this exact order:

1. `ticker`
2. `date`
3. `timing_state`
4. `timing_reason`
5. `breakout_state`
6. `pullback_state`
7. `compression_state`
8. `extension_state`
9. `participation_state`
10. `timing_environment`
11. `timing_pattern_state`
12. `trend_participation_state`
13. `timing_structure_state`
14. `timing_metadata_status`
15. `source_data_status`
16. `source_timestamp`
17. `generated_at`

The output must contain one row per upstream `ticker` + `date` and must preserve upstream row order.

## 15. Logging Contract

Required log:

- `data/logs/timing_state_layer_log.csv`

Required log schema, in this exact order:

1. `generated_at`
2. `input_row_count`
3. `output_row_count`
4. `unique_ticker_date_count`
5. `duplicate_ticker_date_count`
6. `missing_auxiliary_source_count`
7. `timing_state_distribution`
8. `extension_state_distribution`
9. `compression_state_distribution`
10. `pullback_state_distribution`
11. `breakout_state_distribution`
12. `timing_metadata_status_distribution`
13. `source_data_status_distribution`

Log values must be observational only and must not imply priority, readiness, actionability, approval, rejection, conviction, allocation, execution, ranking, or scoring.

## 16. Allowed Metadata Semantics

Allowed metadata is descriptive only.

Allowed value direction:

- `OBSERVED`
- `NOT_OBSERVED`
- `UNKNOWN`
- `UNAVAILABLE`
- `INSUFFICIENT_DATA`
- `SOURCE_MISSING`
- `SOURCE_PARTIAL`
- `NEUTRAL`
- `EXTENDED`
- `COMPRESSED`
- `EXPANDING`
- `CONSOLIDATING`
- `PULLBACK_OBSERVED`
- `BREAKOUT_OBSERVED`
- `CONTINUATION_OBSERVED`
- `PARTICIPATING`
- `NOT_PARTICIPATING`
- `UNCLASSIFIED`

These values describe conditions only. They must not be mapped to actions inside the Timing Layer.

## 17. Forbidden Metadata Semantics

Forbidden columns and values include:

- `tradeable`
- `approved`
- `rejected`
- `high_conviction`
- `conviction`
- `conviction_score`
- `priority`
- `actionable`
- `execution_ready`
- `best_opportunity`
- `buy_candidate`
- `sell_candidate`
- `ranking_score`
- `timing_score`
- `final_score`
- `allocation_weight`
- `expected_return`
- `alpha_score`
- `opportunity_rank`
- `preferred_setup`
- `readiness_score`
- `readiness_status`
- `watchlist_priority`
- `timing_rank`
- `timing_grade`
- `timing_signal`
- `BUY`
- `SELL`
- `REMOVE`
- `URGENT`
- `READY`
- `FAILED`

Forbidden terms may appear in tests or documentation only as negative assertions.

## 18. Distribution-Preservation Requirements

Implementation must prove:

- output row count equals input row count
- output ticker/date key set equals input ticker/date key set
- upstream ordering is preserved exactly
- weak, neutral, incomplete, extended, missing-source, and unknown rows remain visible
- missing auxiliary data never suppresses rows
- no Timing state changes row inclusion
- no Timing state changes output order

## 19. Non-Mutating Enrichment Requirements

Implementation may append descriptive columns only.

It must not:

- mutate `ticker`
- mutate `date`
- mutate upstream quality fields
- overwrite upstream columns
- normalize away upstream classifications
- rewrite Validation, Context, or Fundamental classifications
- aggregate rows
- silently deduplicate rows
- collapse rows to one row per ticker across dates

## 20. Deterministic Behavior Requirements

Implementation must be deterministic:

- same input produces same output apart from `generated_at`
- output column order is fixed
- output row order follows upstream order
- distribution JSON strings, if used, must be stable and sorted
- missing data states are stable
- no randomization
- no clock-dependent classification except timestamp metadata
- no network access

## 21. Fail-Fast Requirements

Implementation must fail fast when:

- `data/processed/fundamental_quality.csv` is missing
- authoritative input is empty
- required columns are missing
- `ticker` is missing or blank
- `date` is missing or blank
- duplicate ticker/date rows exist
- output row count differs from input row count
- output ticker/date key set differs from input key set
- output ordering differs from upstream ordering
- forbidden output columns are present
- forbidden semantic values are present
- protected upstream fields are mutated

Missing auxiliary timing data must not fail the run. It must produce descriptive missing-source metadata.

## 22. Cross-Layer Integration Boundaries

Validation boundary:

- do not read `validation_layer.csv` as authoritative row universe
- do not mutate or reinterpret `valid_setup`
- do not invalidate structures

Context boundary:

- do not read `context_strength.csv` as authoritative row universe unless a future correction explicitly changes the upstream contract
- do not use leadership to alter Timing values
- do not create leadership-confirmed timing

Fundamental boundary:

- read `fundamental_quality.csv` as row universe
- preserve row identity and order
- do not use quality to upgrade, downgrade, prioritize, or suppress timing metadata

Watchlist boundary:

- do not read or write manual watchlist action files
- do not change watchlist membership
- do not reuse readiness/status sorting as-is

Portfolio boundary:

- do not read or write portfolio files
- do not create exposure or risk semantics

Decision Engine boundary:

- do not read or write `final_decisions.csv`
- do not emit final action, tradeability, conviction, allocation priority, urgency, or execution style

Reporting boundary:

- do not modify reporting files
- do not create communication priorities or recommendation language

## 23. Watchlist Legacy-Risk Controls

Do not reuse `scripts/watchlist/evaluate_watchlist.py` as-is.

Unsafe legacy patterns:

- `READY`
- `FAILED`
- readiness thresholds
- status-based sorting
- expiry as removal-like semantics
- active watchlist membership as row universe
- timing status as prioritization
- watch/unwatch action effects

If existing watchlist code is referenced during implementation, it may be used only as a source of anti-patterns and field-discovery context. It may not be imported, called, or copied into the Timing builder.

## 24. Decision Engine Boundary Controls

The Timing Layer must never implement:

- `_conviction_from_context`
- `_allocation_action`
- final action mapping
- tradeability mapping
- allocation priority mapping
- execution style mapping
- portfolio block logic
- BUY/SELL/REMOVE logic

Those responsibilities remain exclusively in `scripts/core/decision_engine.py`.

## 25. Reporting Boundary Controls

No reporting integration is authorized.

Do not modify:

- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/reporter.py`
- `scripts/reporting/send_telegram.py`
- report templates
- Telegram text output

Timing metadata must not be communicated as recommendation, urgency, priority, or action.

## 26. Future Implementation Sequence

Developer execution sequence:

1. Read this specification and Sprint 4 governance artifacts.
2. Inspect current file state and confirm no conflicting changes.
3. Create `scripts/core/build_timing_state_layer.py`.
4. Create `tests/core/test_build_timing_state_layer.py`.
5. Implement input contract validation.
6. Implement descriptive metadata defaults and optional auxiliary observation handling.
7. Implement output and log writing.
8. Implement forbidden-field and forbidden-value safeguards.
9. Run focused Timing tests.
10. Run broader test suite.
11. Run governance grep checks.
12. Report generated artifacts and git status.

## 27. Future Test Requirements

Required unit tests:

- output schema exactly matches specification
- log schema exactly matches specification
- missing authoritative input fails fast
- empty authoritative input fails fast
- missing `ticker` fails fast
- missing `date` fails fast
- duplicate ticker/date rows fail fast
- missing auxiliary source preserves rows
- output row count equals input row count
- output ticker/date key set equals input key set
- output order equals input order
- forbidden columns absent
- forbidden values absent
- deterministic repeated runs match apart from timestamp

Required integration tests:

- builder reads `fundamental_quality.csv`
- builder does not require Decision Engine output
- builder does not require portfolio files
- builder does not require watchlist files
- builder writes processed output and log

Required regression tests:

- `EXTENDED` does not suppress rows
- `SOURCE_MISSING` does not suppress rows
- `UNKNOWN` does not suppress rows
- no sorting by Timing state
- no status-order behavior from legacy watchlist code

Required governance tests:

- no allocation fields
- no tradeability fields
- no conviction fields
- no urgency fields
- no readiness fields
- no ranking fields
- no scoring fields
- no BUY/SELL/REMOVE semantic values

## 28. Future Validation Commands

Minimum commands after future implementation:

```bash
.venv/bin/python3 -m pytest tests/core/test_build_timing_state_layer.py
.venv/bin/python3 -m pytest tests/core
.venv/bin/python3 -m pytest
.venv/bin/python3 scripts/core/build_timing_state_layer.py
grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "tradeable" scripts/ | grep -v decision_engine.py
```

Additional Sprint 4 forbidden-term checks must scan the Timing builder and generated Timing output for:

- `execution_ready`
- `readiness`
- `priority`
- `conviction`
- `ranking`
- `score`
- `allocation`
- `approved`
- `rejected`
- `actionable`
- `best_opportunity`
- `preferred_setup`

## 29. Developer Acceptance Criteria

Developer implementation is complete only when:

- all in-scope files are created
- no out-of-scope files are modified
- Timing output schema matches this specification
- Timing log schema matches this specification
- distribution preservation is proven
- upstream order preservation is proven
- forbidden fields and values are absent
- legacy watchlist behavior is not reused as-is
- no Decision Engine, portfolio, reporting, or watchlist action coupling is introduced
- required tests pass
- required validation commands are reported
- generated artifact handling is documented

## 30. Technical Lead Audit Criteria

Technical Lead implementation audit must verify:

- file scope compliance
- schema compliance
- log compliance
- row-count preservation
- ticker/date preservation
- upstream-order preservation
- fail-fast behavior
- missing auxiliary source behavior
- forbidden-field absence
- forbidden-value absence
- no upstream mutation
- no legacy watchlist readiness/status-sorting reuse
- no Decision Engine authority leakage
- no portfolio coupling
- no reporting coupling
- tests and validation evidence
- generated artifact policy

## 31. Commit / Push Requirements

Future implementation commit may include:

- `scripts/core/build_timing_state_layer.py`
- `tests/core/test_build_timing_state_layer.py`
- approved Sprint 4 documentation updates

Future implementation commit must not include generated Timing CSV/log files unless explicitly authorized.

Before commit or push:

- inspect `git status --short`
- document generated files
- document tests and grep checks
- confirm no unrelated dirty files are included
- confirm no runtime files outside approved scope changed

## 32. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Timing metadata becomes actionability | Ban readiness/actionability schema and values |
| Timing metadata becomes priority | Preserve upstream order; ban priority/rank fields |
| Timing metadata becomes conviction | Ban conviction fields and values |
| Timing metadata becomes scoring | Ban score fields and composite scores |
| Legacy watchlist behavior leaks into Timing | Do not import, call, or copy legacy watchlist evaluator |
| Extension state suppresses rows | Test row preservation for extended values |
| Missing auxiliary data suppresses rows | Use descriptive source-data status |
| Timing mutates Fundamental output | Treat upstream as row universe only |
| Timing becomes mini Decision Engine | Ban final action, tradeability, allocation, urgency, conviction, and execution style |
| Reporting treats Timing as recommendation | No reporting integration in Sprint 4 implementation |

## 33. Recommended Next Step

Proceed to Sprint 4 developer implementation under this specification.

Implementation must remain limited to the approved files and must be followed by Technical Lead implementation audit before closeout.

## 34. Final Technical Analyst Recommendation

READY FOR SPRINT 4 DEVELOPMENT
