# Sprint 3 Developer Specification — Fundamental Quality Layer

## 1. Specification Status

Status: READY FOR DEVELOPER IMPLEMENTATION

This specification authorizes a future developer execution task only within the scope defined here.

This document does not implement runtime code, tests, or generated artifacts.

Sprint 3 current state:

- Sprint 0 = CERTIFIED COMPLETE / CLOSED
- Sprint 1 = CERTIFIED COMPLETE / CLOSED
- Sprint 2 = CERTIFIED COMPLETE / CLOSED
- Sprint 3 preparation = CERTIFIED
- Sprint 3 governance audit = COMPLETE
- Sprint 3 re-audit = COMPLETE
- Sprint 3 execution plan = COMPLETE
- Sprint 3 execution review = COMPLETE
- Sprint 3 developer specification = COMPLETE

## 2. Certified Governance Baseline

Sprint 3 inherits the certified doctrine:

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

The Fundamental Quality Layer must remain a pure classification and enrichment layer.

It may classify:

- profitability quality
- balance-sheet quality
- earnings quality
- capital efficiency
- cash-flow quality
- stability metrics
- quality-factor metadata
- Piotroski-style quality classification metadata
- Greenblatt-style quality metadata
- sector-relative quality metadata

It must not create or imply:

- allocation logic
- tradeability
- conviction
- urgency
- priority
- ranking authority
- scoring authority
- execution readiness
- BUY/SELL semantics
- hard gating
- hidden filtering
- opportunity suppression
- opportunity reordering
- portfolio logic
- Decision Engine leakage

## 3. Implementation Objective

Implement a governance-clean Fundamental Quality Layer that preserves the full upstream classified opportunity universe and enriches each row with descriptive quality metadata.

The implementation must be inspection-first:

1. Inspect current repository data sources and pipeline conventions.
2. Confirm whether a fundamentals data source already exists.
3. Preserve all upstream `ticker` + `date` rows regardless of fundamentals availability.
4. Emit governance-clean quality classification fields only.
5. Emit clear source-data and metadata status fields.
6. Emit a runtime audit log.
7. Add focused tests proving schema cleanliness, distribution preservation, missing-data behavior, duplicate handling, forbidden-field absence, and deterministic output.

If no approved fundamentals source exists, the first implementation must still preserve every upstream row and emit descriptive unavailable or insufficient-data metadata. Do not invent thresholds, strategy logic, scoring logic, ranking logic, or allocation semantics to compensate for missing source data.

## 4. In-Scope Files

Future developer execution may create or modify only:

- `scripts/core/build_fundamental_layer.py`
- `tests/core/test_build_fundamental_layer.py`
- `docs/sprints/sprint_3_developer_spec.md`, only for implementation notes if required
- `docs/sprints/sprint_status_tracker.md`, only for approved Sprint 3 phase status updates
- `docs/sprints/project_backlog.md`, only for backlog status updates or newly discovered deferred items

Future developer execution may generate during validation:

- `data/processed/fundamental_quality.csv`
- `data/logs/fundamental_layer_log.csv`

Generated files may not be committed unless the implementation task explicitly authorizes generated artifact updates.

## 5. Explicitly Out-of-Scope Files

Do not modify:

- scanner logic
- validation logic
- context logic
- watchlist logic
- portfolio logic
- Decision Engine logic
- reporting logic
- Telegram logic
- existing generated CSV/data files outside the approved future generated Fundamental artifacts
- trading thresholds
- strategy scoring
- allocation logic
- BUY/SELL/HOLD/TRIM/REMOVE behavior

Specifically out of scope:

- `scripts/core/scanner.py`
- `scripts/core/build_validation_layer.py`
- `scripts/core/build_context_layer.py`
- `scripts/core/build_context_backfill.py`
- `scripts/watchlist/`
- `scripts/portfolio/`
- `scripts/core/decision_engine.py`
- `scripts/reporting/`
- `scripts/telegram/`
- existing tests outside `tests/core/test_build_fundamental_layer.py`, unless a Technical Lead explicitly expands scope

## 6. Upstream Input Contract

Authoritative upstream input source for the first implementation:

- `data/processed/context_strength.csv`

Reason:

- Sprint 3 target placement is after Context and before Watchlist.
- `context_strength.csv` is the certified Context Layer output.
- Context output is classification-only and does not contain Decision Engine allocation semantics.
- This source preserves the current classified opportunity universe available immediately upstream of the Fundamental Layer.

Required input columns:

- `ticker`
- `date`

Allowed optional upstream columns for validation or pass-through inspection only:

- `rs_score`
- `rs_percentile`
- `rs_rank`
- `rs_vs_market`
- `rs_vs_sector`
- `context_strength`
- `context_reason`
- `leadership_state`

The Fundamental Layer must not require:

- tradeability
- allocation fields
- conviction fields
- final action fields
- Decision Engine output
- portfolio state
- watchlist timing state

The implementation must not exclude weak or neutral Context rows. Context weakness is leadership classification only, not a Fundamental Layer filter.

If `data/processed/context_strength.csv` is missing, the script must fail fast with a clear error unless a Technical Lead-approved alternative upstream classified source is explicitly provided through CLI arguments or documented configuration.

## 7. Row-Key and Duplicate Handling Contract

Primary row key:

- `ticker`
- `date`

Rules:

- Output must contain one row per upstream `ticker` + `date`.
- Output must preserve all upstream `ticker` + `date` pairs.
- Duplicate upstream `ticker` + `date` rows must fail fast with a clear error.
- Missing `ticker` must fail fast with a clear error.
- Missing `date` must fail fast with a clear error.
- The implementation may not silently deduplicate rows.
- The implementation may not aggregate rows.
- The implementation may not collapse rows to one row per ticker across dates.

Failure messages must name the contract violation and include enough context for operator diagnosis.

## 8. Output Data Contract

Required output file:

- `data/processed/fundamental_quality.csv`

Required output schema, in this exact order:

1. `ticker`
2. `date`
3. `quality_state`
4. `quality_reason`
5. `profitability_profile`
6. `balance_sheet_profile`
7. `earnings_quality_profile`
8. `capital_efficiency_profile`
9. `cashflow_profile`
10. `stability_profile`
11. `quality_metadata_status`
12. `source_data_status`
13. `source_timestamp`
14. `generated_at`

Output rules:

- Every upstream row must produce exactly one output row.
- `ticker` and `date` must match upstream values exactly.
- `generated_at` must be deterministic within a single run.
- If source data is unavailable, profile fields must use descriptive unknown or insufficient-data values.
- No output column may imply decisions, allocation, ranking, scoring authority, priority, conviction, urgency, actionability, execution readiness, or tradeability.

Do not include raw financial numeric fields in the first implementation unless explicitly approved by Technical Lead review. Raw numeric fundamentals can create implicit scoring or ranking pressure and must be separately governed before exposure.

## 9. Classification Semantics

Allowed `quality_state` values:

- `QUALITY_LEADING`
- `QUALITY_STRONG`
- `QUALITY_STABLE`
- `QUALITY_MIXED`
- `QUALITY_WEAK`
- `QUALITY_UNKNOWN`
- `INSUFFICIENT_DATA`

Allowed profile values for profile columns:

- `LEADING`
- `STRONG`
- `STABLE`
- `MIXED`
- `WEAK`
- `UNKNOWN`
- `INSUFFICIENT_DATA`
- `UNAVAILABLE`

Allowed `quality_metadata_status` values:

- `complete`
- `partial_data`
- `insufficient_data`
- `unavailable`
- `source_missing`
- `stale_data`

Allowed `source_data_status` values:

- `complete`
- `partial_data`
- `insufficient_data`
- `unavailable`
- `source_missing`
- `stale_data`

Allowed `quality_reason` values must be descriptive and non-decisional. Examples:

- `fundamental data unavailable`
- `insufficient fundamental data`
- `partial fundamental data`
- `source data stale`
- `quality metadata classified`

Forbidden classification values:

- `BUY`
- `SELL`
- `HOLD`
- `WAIT`
- `REVIEW`
- `APPROVED`
- `REJECTED`
- `TRADEABLE`
- `NOT_TRADEABLE`
- `HIGH_CONVICTION`
- `LOW_CONVICTION`
- `PRIORITY`
- `ACTIONABLE`
- `EXECUTION_READY`

Do not introduce new thresholds for assigning profile values. If no existing approved categorical source exists, use `QUALITY_UNKNOWN`, `INSUFFICIENT_DATA`, `UNKNOWN`, `UNAVAILABLE`, or other allowed missing-data statuses.

## 10. Missing / Partial / Stale Data Handling

Missing or partial fundamentals must never suppress rows.

Missing, partial, stale, or unavailable data may only affect:

- `quality_state`
- `quality_reason`
- profile fields
- `quality_metadata_status`
- `source_data_status`
- `source_timestamp`

Missing data must never produce:

- rejection
- exclusion
- tradeability failure
- priority downgrade
- conviction downgrade
- ranking penalty
- allocation impact
- execution readiness change

If source data is missing for a ticker/date:

- preserve the row
- set `quality_state` to `INSUFFICIENT_DATA` or `QUALITY_UNKNOWN`
- set profile fields to `INSUFFICIENT_DATA`, `UNKNOWN`, or `UNAVAILABLE`
- set `quality_metadata_status` and `source_data_status` to an allowed missing-data status
- write a descriptive `quality_reason`

## 11. Distribution Preservation Requirements

The implementation must:

- preserve full upstream row count
- preserve all upstream `ticker` + `date` pairs
- never suppress rows due to missing fundamentals
- never drop rows due to weak quality
- never drop rows due to partial or stale data
- never drop rows due to sector-relative quality missingness
- never narrow the opportunity universe
- never gatekeep opportunities

Required invariant:

```text
set(input[ticker, date]) == set(output[ticker, date])
len(input) == len(output)
```

Any violation must fail tests.

## 12. Deterministic Ordering Policy

Output ordering may be deterministic for reproducibility only.

Allowed deterministic ordering:

- preserve upstream input order
- or sort by `date` ascending, then `ticker` ascending
- or sort by `ticker` ascending, then `date` ascending

The chosen ordering must be documented in the implementation and tested.

Forbidden interpretation:

Output order must never imply:

- ranking
- priority
- conviction
- actionability
- opportunity quality order
- allocation preference
- execution readiness

Do not sort by:

- `quality_state`
- profile fields
- source-data status
- any future numeric fundamental metric

## 13. Forbidden Fields and Forbidden Semantics

The Fundamental Layer may not create fields containing or implying:

- `tradeable`
- `approved`
- `rejected`
- `high_conviction`
- `conviction`
- `conviction_score`
- `priority`
- `rank`
- `ranking`
- `score`
- `scoring`
- `actionable`
- `buy_candidate`
- `sell_candidate`
- `execution_ready`
- `best_opportunity`
- `allocation`
- `allocation_weight`
- `urgency`
- `final_action`
- `final_score`
- `decision`
- `signal_strength`
- `gate`
- `pass_fail`

Semantic variants are also forbidden. Examples:

- `quality_score`
- `quality_rank`
- `composite_score`
- `opportunity_rank`
- `capital_weight`
- `action_now`
- `execution_status`

The implementation may not contain logic that:

- determines tradeability
- determines conviction
- determines urgency
- determines allocation eligibility
- creates BUY/SELL/HOLD/TRIM/REMOVE behavior
- creates final actions
- filters rows based on quality
- filters rows based on missing fundamentals
- prioritizes opportunities
- ranks opportunities
- scores opportunities for action

## 14. Logging and Audit Trail Requirements

Required log file:

- `data/logs/fundamental_layer_log.csv`

Required log fields:

1. `generated_at`
2. `input_row_count`
3. `output_row_count`
4. `unique_ticker_date_count`
5. `duplicate_ticker_date_count`
6. `missing_fundamentals_count`
7. `partial_data_count`
8. `stale_data_count`
9. `quality_state_distribution`
10. `quality_metadata_status_distribution`
11. `source_data_status_distribution`

Logging rules:

- Logs are audit metadata only.
- Logs may not alter runtime output eligibility.
- Logs may not create ranking, priority, score authority, conviction, urgency, allocation, tradeability, actionability, or execution readiness.
- Distribution fields may be serialized as deterministic strings or JSON-like text, but must remain stable across identical inputs.

## 15. Testing Requirements

Create focused tests in:

- `tests/core/test_build_fundamental_layer.py`

Required tests:

- output schema contains required columns in the required order
- output preserves all input rows
- output preserves all upstream `ticker` + `date` pairs
- one row per `ticker` + `date` is enforced
- duplicate `ticker` + `date` rows fail fast
- missing `ticker` fails fast
- missing `date` fails fast
- missing fundamentals preserve rows
- partial fundamentals preserve rows
- stale fundamentals preserve rows, if stale metadata is represented
- sector-relative quality missingness preserves rows, if sector metadata is represented
- forbidden fields are absent from output
- forbidden semantic values are absent from output
- deterministic ordering is reproducible
- deterministic ordering does not depend on `quality_state`
- no BUY/SELL/HOLD/TRIM/REMOVE/actionability semantics are produced
- no tradeability fields are produced
- no conviction fields are produced
- no priority fields are produced
- no ranking or scoring fields are produced
- log output contains required audit metrics
- log row counts match output row counts
- source-data status distributions are deterministic

Tests may reference forbidden terms only as absence assertions.

## 16. Regression-Control Requirements

Implementation must preserve:

- existing Sprint 0 tests
- existing Sprint 1 tests
- existing Sprint 2 tests
- current Validation behavior
- current Context behavior
- current Decision Engine behavior
- current Reporting behavior
- current Telegram behavior
- existing generated CSV/data artifacts unless explicitly regenerated during validation

Do not commit generated CSV/data changes unless explicitly approved by the implementation task or Technical Lead audit.

No changes are allowed to:

- Decision Engine final action behavior
- allocation priority behavior
- conviction behavior
- tradeability behavior
- scanner distribution behavior
- validation schema
- context schema
- reporting presentation

## 17. Developer Execution Sequence

Required implementation sequence:

1. Read this developer specification.
2. Read `AGENTS.md`.
3. Read `docs/sprints/sprint_status_tracker.md`.
4. Read `docs/sprints/project_backlog.md`.
5. Inspect `data/processed/context_strength.csv`.
6. Inspect current `scripts/core/` conventions.
7. Inspect whether any approved fundamentals source already exists.
8. Confirm no runtime files outside the allowed scope need changes.
9. Create `scripts/core/build_fundamental_layer.py`.
10. Create `tests/core/test_build_fundamental_layer.py`.
11. Implement row-key validation before enrichment.
12. Implement distribution-preserving output generation.
13. Implement missing-data-safe descriptive metadata.
14. Implement audit log generation.
15. Add focused tests.
16. Run focused tests.
17. Run full test suite.
18. Run governance grep checks.
19. Run pipeline only if the implementation task explicitly requires it or generated artifacts are intentionally refreshed for validation.
20. Report files changed, tests run, grep interpretation, generated artifact status, and remaining risks.
21. Update `docs/sprints/sprint_status_tracker.md` only as instructed by the implementation task.
22. Update `docs/sprints/project_backlog.md` with any newly discovered deferred work; do not mark backlog items implemented unless implementation actually resolves them and Technical Lead audit confirms it.

## 18. Validation Commands

Focused tests:

```bash
.venv/bin/python3 -m pytest tests/core/test_build_fundamental_layer.py
```

Full tests:

```bash
.venv/bin/python3 -m pytest
```

Governance grep checks:

```bash
grep -R "tradeable" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "approved" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "rejected" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "high_conviction" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "conviction" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "priority" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "rank" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "ranking" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "score" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "scoring" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "actionable" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "buy_candidate" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "sell_candidate" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "execution_ready" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "best_opportunity" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "allocation" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "allocation_weight" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "urgency" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "final_action" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "final_score" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "decision" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "signal_strength" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "gate" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "pass_fail" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "BUY" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "SELL" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "HOLD" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "TRIM" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
grep -R "REMOVE" scripts/core/build_fundamental_layer.py tests/core/test_build_fundamental_layer.py
```

Expected grep interpretation:

- Active Fundamental source must not contain forbidden fields or active forbidden semantics.
- Tests may contain forbidden terms only as absence assertions.
- Generated caches must be ignored.

Pipeline command, only if required by implementation task:

```bash
PATH="/Users/stevenclaessens/market-scanner/.venv/bin:$PATH" python3 scripts/run_full_pipeline.py
```

Artifact inspection, if generated artifacts are produced:

```bash
head -n 1 data/processed/fundamental_quality.csv
tail -n 1 data/logs/fundamental_layer_log.csv
```

## 19. Acceptance Criteria

Sprint 3 developer implementation may pass only when:

- `scripts/core/build_fundamental_layer.py` exists and is limited to quality classification/enrichment
- `tests/core/test_build_fundamental_layer.py` exists and covers required governance behavior
- `data/processed/context_strength.csv` is the upstream input source unless Technical Lead explicitly approves an alternative
- output preserves full upstream row count
- output preserves every upstream `ticker` + `date`
- duplicate `ticker` + `date` rows fail fast
- missing `ticker` or `date` fails fast
- missing fundamentals do not suppress rows
- partial fundamentals do not suppress rows
- stale fundamentals do not suppress rows
- weak quality does not suppress rows
- deterministic ordering is reproducible and not quality-based
- output schema matches the required schema exactly
- forbidden fields are absent
- forbidden classification values are absent
- no tradeability semantics are introduced
- no allocation semantics are introduced
- no conviction semantics are introduced
- no urgency semantics are introduced
- no priority semantics are introduced
- no ranking or scoring authority is introduced
- no BUY/SELL/HOLD/TRIM/REMOVE behavior is introduced
- no Decision Engine, Validation, Context, Reporting, Telegram, Watchlist, Portfolio, or Scanner behavior changes are introduced
- focused tests pass
- full tests pass
- governance grep checks pass or are clearly interpreted as test absence assertions only
- generated artifact changes are either uncommitted validation outputs or explicitly approved
- Sprint status tracker is updated according to the implementation task
- project backlog is updated with any new deferred work

## 20. Sprint Status Tracker Update Requirement

Because this developer specification now exists, update `docs/sprints/sprint_status_tracker.md` to mark:

- Sprint 3 `DEVELOPER SPECIFICATION` = COMPLETE
- Sprint 3 current phase = IMPLEMENTATION
- Sprint 3 current next action = Start Sprint 3 developer implementation

Do not mark:

- IMPLEMENTATION complete
- IMPLEMENTATION AUDIT complete
- CLOSEOUT complete
- CLOSED

## 21. Project Backlog Update Requirement

Because BL-0001 through BL-0004 are incorporated into this developer specification, update `docs/sprints/project_backlog.md` to mark them as:

- Status: ACTIVE SPRINT
- Proposed next step: Implement through Sprint 3 developer execution under `docs/sprints/sprint_3_developer_spec.md`

Do not mark them as `IMPLEMENTED`.

Do not remove them from the backlog.

New deferred work discovered during implementation must be added as new backlog items.

## 22. Technical Lead Recommendation

READY FOR DEVELOPER IMPLEMENTATION
