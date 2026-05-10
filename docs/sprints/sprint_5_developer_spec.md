# Sprint 5 Developer Specification - Portfolio Intelligence Layer

## 1. Specification Status

Status: DEVELOPER SPECIFICATION COMPLETE - READY FOR IMPLEMENTATION.

This document is a developer specification only. It authorizes future implementation scope after sprint governance approval, but it does not perform implementation. It does not modify runtime code, tests, generated outputs, strategy rules, thresholds, allocation logic, Decision Engine logic, or reporting behavior.

Sprint 5 implementation must preserve the certified doctrine:

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

Certified architecture:

```text
scanner -> validation_layer -> context_layer -> fundamental_layer -> timing_state_layer -> watchlist -> portfolio -> decision_engine -> reporting
```

## 2. Documents Reviewed

This specification inherits and applies:

- `AGENTS.md`
- `README.md`
- `docs/sprints/README.md`
- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/execution_roadmap_v2.md`
- `docs/sprints/sprint_5_portfolio_intelligence.md`
- `docs/audits/sprint_5_governance_audit.md`
- `docs/sprints/sprint_4_closeout.md`
- `docs/technical/Technical_Analysis_v3.md`
- `docs/functional/Functional_Analysis_v2.md`
- `docs/execution/execution_delivery_framework_v2.md`
- `docs/technical/decision_engine_design_v2.md`

Where older roadmap or analysis language references portfolio pressure, portfolio heat, portfolio interaction, conviction influence, allocation priority, ranking, or tradeability, that language remains Decision Engine-owned or historical/contextual only. It must not be implemented as Sprint 5 Portfolio Intelligence authority.

## 3. Developer Objective

Implement, in a future development step, a standalone Portfolio Intelligence Layer that enriches the full upstream opportunity universe with neutral descriptive portfolio-awareness metadata.

The layer must remain:

- descriptive-only
- enrichment-only
- deterministic
- reproducible
- audit-traceable
- non-mutating
- semantically neutral
- distribution-preserving

The layer must not create allocation, execution, ranking, scoring, priority, conviction, tradeability, urgency, filtering, opportunity suppression, portfolio override, hidden risk-engine, hidden portfolio-manager, hidden allocation-engine, or Decision Engine semantics.

## 4. Target Files For Future Implementation

Future implementation may create:

- `scripts/core/build_portfolio_intelligence.py`
- `tests/core/test_build_portfolio_intelligence.py`
- `data/processed/portfolio_intelligence.csv`
- `data/logs/portfolio_intelligence_log.csv`

Future implementation may update documentation only as specified in section 21.

Future implementation must not modify:

- `scripts/core/decision_engine.py`
- existing certified upstream builders unless a later governance artifact explicitly authorizes it
- `scripts/reporting/`
- generated outputs other than `data/processed/portfolio_intelligence.csv` and `data/logs/portfolio_intelligence_log.csv`

## 5. Expected Input Files

Primary opportunity universe input:

- `data/processed/timing_state_layer.csv`

Descriptive portfolio source input:

- `data/portfolio/portfolio_positions.csv`

The Timing State Layer output is the authoritative Sprint 5 opportunity universe. Portfolio Intelligence must preserve it exactly and append metadata only.

No mandatory runtime dependency may be injected into Validation, Context, Fundamental, Timing State, Watchlist, Decision Engine, or Reporting layers.

## 6. Expected Output Files

Future implementation must write:

- `data/processed/portfolio_intelligence.csv`
- `data/logs/portfolio_intelligence_log.csv`

The processed output must contain all columns from `data/processed/timing_state_layer.csv` unchanged, in the same order, followed by Sprint 5 appended metadata columns.

The log output must contain one audit row per output row and must support deterministic verification of source provenance, classification rationale, row preservation, missing-source handling, and forbidden-semantics absence.

## 7. Required Output Schema

`data/processed/portfolio_intelligence.csv` must preserve every upstream input column unchanged, then append exactly these Sprint 5 columns:

- `in_portfolio`
- `portfolio_position_state`
- `exposure_state`
- `diversification_state`
- `concentration_state`
- `overlap_state`
- `sector_exposure_state`
- `position_context_state`
- `portfolio_environment`
- `portfolio_metadata_status`
- `portfolio_metadata_reason`
- `portfolio_source_provenance`
- `portfolio_classification_rationale`

The output must not add `ticker` or `date` duplicates if those columns already exist upstream. The output must preserve the upstream `ticker` and `date` columns exactly as received.

## 8. Required Log Schema

`data/logs/portfolio_intelligence_log.csv` must contain exactly these columns:

- `ticker`
- `date`
- `input_row_index`
- `output_row_index`
- `row_identity_preserved`
- `ticker_preserved`
- `date_preserved`
- `ordering_preserved`
- `upstream_columns_preserved`
- `upstream_values_preserved`
- `portfolio_source_status`
- `portfolio_source_provenance`
- `portfolio_classification_rationale`
- `portfolio_metadata_status`
- `portfolio_metadata_reason`
- `forbidden_semantics_absent`

The log must not contain action recommendations, allocation interpretation, tradeability interpretation, urgency interpretation, conviction interpretation, ranking interpretation, scoring interpretation, or preference language.

## 9. Allowed Values

Allowed `in_portfolio` values:

- `PRESENT`
- `ABSENT`
- `UNKNOWN`

Allowed `portfolio_position_state` values:

- `PRESENT`
- `ABSENT`
- `SOURCE_MISSING`
- `SOURCE_PARTIAL`

Allowed `exposure_state` values:

- `NONE`
- `LOW`
- `MODERATE`
- `HIGH`
- `SOURCE_MISSING`
- `SOURCE_PARTIAL`

Allowed `diversification_state` values:

- `NONE`
- `LIMITED`
- `BROAD`
- `SOURCE_MISSING`
- `SOURCE_PARTIAL`

Allowed `concentration_state` values:

- `NONE`
- `CONCENTRATED`
- `BALANCED`
- `DIVERSIFIED`
- `SOURCE_MISSING`
- `SOURCE_PARTIAL`

Allowed `overlap_state` values:

- `MATCHED`
- `UNMATCHED`
- `SOURCE_MISSING`
- `SOURCE_PARTIAL`

Allowed `sector_exposure_state` values:

- `NONE`
- `LOW`
- `MODERATE`
- `HIGH`
- `UNKNOWN_SECTOR`
- `SOURCE_MISSING`
- `SOURCE_PARTIAL`

Allowed `position_context_state` values:

- `PRESENT`
- `ABSENT`
- `SOURCE_MISSING`
- `SOURCE_PARTIAL`

Allowed `portfolio_environment` values:

- `EMPTY_PORTFOLIO`
- `POSITIONS_PRESENT`
- `SOURCE_MISSING`
- `SOURCE_PARTIAL`

Allowed `portfolio_metadata_status` values:

- `AVAILABLE`
- `PARTIAL`
- `MISSING`

These values are descriptive metadata only. They are not decision states, recommendation states, execution states, allocation states, approval states, rejection states, conviction states, suitability states, ranking states, or score states.

## 10. Deterministic Classification Rules

Implementation must use deterministic, non-adaptive classification rules only.

General rules:

- If the primary opportunity universe input is missing, fail fast.
- If required opportunity identity columns are missing, fail fast.
- If the portfolio source file is missing, preserve all opportunity rows and set portfolio metadata to `SOURCE_MISSING` / `MISSING`.
- If the portfolio source exists but lacks required descriptive columns, preserve all opportunity rows and set affected portfolio metadata to `SOURCE_PARTIAL` / `PARTIAL`.
- If portfolio source rows duplicate a ticker identity, aggregate descriptively by ticker using stable deterministic grouping. The grouping must not reorder the opportunity output.
- If source values are blank or null, classify the affected metadata as `SOURCE_PARTIAL` / `PARTIAL`.

Ticker participation rules:

- `in_portfolio = PRESENT` when the output ticker is present in the portfolio source ticker set.
- `in_portfolio = ABSENT` when the portfolio source is available and the output ticker is not present in the portfolio source ticker set.
- `in_portfolio = UNKNOWN` only when the portfolio source is missing or cannot provide ticker identity.

Position-state rules:

- `portfolio_position_state = PRESENT` when `in_portfolio = PRESENT`.
- `portfolio_position_state = ABSENT` when `in_portfolio = ABSENT`.
- `portfolio_position_state = SOURCE_MISSING` when the portfolio source is missing.
- `portfolio_position_state = SOURCE_PARTIAL` when the portfolio source exists but required descriptive identity fields are incomplete.

Overlap rules:

- `overlap_state = MATCHED` when `in_portfolio = PRESENT`.
- `overlap_state = UNMATCHED` when `in_portfolio = ABSENT`.
- `overlap_state = SOURCE_MISSING` when the portfolio source is missing.
- `overlap_state = SOURCE_PARTIAL` when portfolio source identity is incomplete.

Portfolio environment rules:

- `portfolio_environment = EMPTY_PORTFOLIO` when the portfolio source is available and contains no active descriptive position rows.
- `portfolio_environment = POSITIONS_PRESENT` when the portfolio source is available and contains at least one active descriptive position row.
- `portfolio_environment = SOURCE_MISSING` when the portfolio source is missing.
- `portfolio_environment = SOURCE_PARTIAL` when the portfolio source cannot be parsed into deterministic descriptive position rows.

Exposure, diversification, concentration, sector exposure, and position context rules:

- Future implementation may derive these values only from descriptive counts or source availability.
- No rule may use price targets, expected return, risk/reward, conviction, tradeability, ranking, score, priority, urgency, suitability, actionability, or allocation preference.
- Count-derived rules must be documented as descriptive distribution labels only.
- Count-derived rules must use fixed deterministic constants in the implementation and tests. These constants are governance controls for metadata binning only; they are not strategy thresholds and must not be interpreted as allocation logic.
- Any unknown sector or missing sector value must preserve the row and classify only the affected metadata as `UNKNOWN_SECTOR`, `SOURCE_MISSING`, or `SOURCE_PARTIAL`.

Reason and provenance rules:

- `portfolio_source_provenance` must identify the source file or source missing condition.
- `portfolio_classification_rationale` must explain the descriptive source condition used for the metadata.
- `portfolio_metadata_reason` must remain neutral and must not include recommendation, urgency, action, preference, ranking, scoring, conviction, or tradeability language.

## 11. Source Provenance Requirements

Every output row must have:

- explicit source provenance
- explicit classification rationale
- explicit metadata status
- explicit metadata reason

Every log row must allow an auditor to determine:

- which opportunity row was enriched
- whether row identity was preserved
- whether upstream columns were preserved
- whether upstream values were preserved
- which portfolio source condition was used
- whether missing or partial source handling was applied
- whether forbidden semantics were absent

## 12. Distribution-Preservation Checks

Future implementation must verify and test:

- output row count equals input row count
- output ticker universe equals input ticker universe
- output ticker ordering equals input ticker ordering
- output date ordering equals input date ordering where date exists
- input row index maps one-to-one to output row index
- all upstream columns exist in output
- all upstream columns preserve their original order
- all upstream values are unchanged
- no output row is removed
- no output row is added
- no output row is reordered
- no output row is ranked
- no output row is scored
- no output row is gatekept

Any failure of these checks must fail fast and must not write a misleading successful output.

## 13. Non-Mutation Checks

Future implementation must not mutate:

- `data/processed/timing_state_layer.csv`
- `data/processed/fundamental_quality.csv`
- `data/processed/context_strength.csv`
- `data/processed/validation_layer.csv`
- scanner outputs
- portfolio source inputs
- Decision Engine outputs
- reporting outputs

Tests must compare upstream input frames before and after enrichment and prove unchanged upstream columns and values.

## 14. Fail-Fast Data Contract Requirements

Implementation must fail fast for:

- missing primary opportunity input
- unreadable primary opportunity input
- missing `ticker` column in primary opportunity input
- duplicate output column names after enrichment
- attempted overwrite of an upstream column
- invalid appended schema
- forbidden appended column name
- forbidden metadata value
- row-count mismatch
- ticker-universe mismatch
- ordering mismatch
- upstream-value mutation
- nondeterministic repeated output

Implementation must not fail for:

- missing portfolio source file
- empty portfolio source file
- missing optional descriptive portfolio attributes
- unknown sector metadata

Those cases must preserve all opportunity rows and emit neutral missing/partial metadata.

## 15. Cross-Layer Boundary Controls

Validation boundary:

- May read preserved validation columns only as carried through the Timing State input.
- Must not change validation columns or validation meanings.
- Must not use validation state to gate portfolio metadata.

Context boundary:

- May read preserved context columns only as carried through the Timing State input.
- Must not convert leadership metadata into portfolio preference, allocation preference, ranking, or scoring.

Fundamental boundary:

- May read preserved fundamental columns only as carried through the Timing State input.
- Must not convert quality metadata into conviction, suitability, or desirability.

Timing State boundary:

- Must use `data/processed/timing_state_layer.csv` as the row-preserved input universe.
- Must not reinterpret timing state as readiness, urgency, or execution preference.

Watchlist boundary:

- Must not create watchlist readiness, status sorting, or execution readiness.
- Must not read or mutate watchlist runtime behavior.

Portfolio boundary:

- May read portfolio source data descriptively.
- Must not decide portfolio allocation, position sizing, eligibility, action, or recommendation.

Decision Engine boundary:

- Must not read, write, call, import, or alter `scripts/core/decision_engine.py`.
- Must not create, simulate, precompute, or shadow Decision Engine decisions.

Reporting boundary:

- Must not modify reporting code or reporting behavior.
- Must not create action narratives, recommendation narratives, urgency narratives, or allocation summaries.

## 16. Forbidden Semantics For Implementation

Implementation must not introduce:

- allocation authority
- execution authority
- BUY logic
- SELL logic
- TRIM logic
- REMOVE logic
- HOLD interpretation
- WAIT interpretation
- tradeability semantics
- urgency semantics
- conviction semantics
- scoring semantics
- ranking semantics
- priority semantics
- filtering
- opportunity suppression
- opportunity reordering
- portfolio override authority
- Decision Engine leakage
- hidden risk-engine authority
- hidden portfolio-manager authority
- hidden allocation-engine authority
- suitability semantics
- recommendation semantics
- actionability semantics
- execution readiness semantics
- preference semantics

## 17. Forbidden Column Names

Implementation must reject any appended column matching or equivalent to:

- `allocation_weight`
- `recommended_weight`
- `ideal_position_size`
- `high_conviction`
- `conviction_score`
- `portfolio_priority`
- `actionable`
- `execution_ready`
- `best_opportunity`
- `buy_candidate`
- `sell_candidate`
- `ranking_score`
- `portfolio_score`
- `final_score`
- `allocation_signal`
- `recommended_trade`
- `preferred_position`
- `preferred_opportunity`
- `execution_signal`
- `urgency`
- `priority`
- `recommendation`
- `suitability`
- `attractiveness`
- `optimal_weight`
- `target_weight`
- `rebalance_action`
- `portfolio_fit`
- `portfolio_capacity`
- `exposure_allowance`
- `tradeable`
- `tradeability`
- `conviction`
- `score`
- `rank`
- `weight`
- `signal`

Forbidden checks must be case-insensitive and must apply to output columns, log columns, generated values, and implementation-owned constants.

## 18. Test Requirements

Future implementation must include focused tests for:

- schema contract
- forbidden columns
- forbidden values and forbidden semantics
- row-count preservation
- ticker-universe preservation
- ordering preservation
- upstream-column preservation
- upstream-value non-mutation
- deterministic repeated output
- missing portfolio source behavior
- empty portfolio source behavior
- partial portfolio source behavior
- duplicate portfolio source ticker handling
- log schema
- source provenance
- classification rationale
- no Decision Engine import, write, call, or output mutation
- no BUY/SELL/actionable semantics
- no tradeability semantics
- no urgency semantics
- no conviction semantics
- no ranking/scoring/priority semantics
- no opportunity suppression
- no generated output outside approved files

Tests should be added in:

- `tests/core/test_build_portfolio_intelligence.py`

Tests must use controlled fixtures or temporary files and must not dirty tracked portfolio CSV files.

## 19. Validation Commands

Future implementation must run:

```bash
.venv/bin/python3 -m pytest tests/core/test_build_portfolio_intelligence.py
.venv/bin/python3 -m pytest tests/core
.venv/bin/python3 -m pytest
.venv/bin/python3 scripts/core/build_portfolio_intelligence.py
grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "tradeable" scripts/ | grep -v decision_engine.py
git diff --check
git status --short
```

If broad grep finds pre-existing legacy references outside Sprint 5 scope, the implementation handoff must list them explicitly and confirm Sprint 5 did not introduce them.

## 20. Future Implementation Sequence

The future developer must proceed in this order:

1. Re-read `AGENTS.md`, Sprint 5 preparation, Sprint 5 governance audit, and this developer specification.
2. Confirm no runtime code is modified outside the approved Sprint 5 files.
3. Add focused tests for schema, preservation, forbidden semantics, missing data, determinism, and logs.
4. Implement `scripts/core/build_portfolio_intelligence.py` as a standalone builder.
5. Use `data/processed/timing_state_layer.csv` as the preserved opportunity universe.
6. Read portfolio source data descriptively from `data/portfolio/portfolio_positions.csv`.
7. Append only approved Sprint 5 metadata columns.
8. Emit `data/processed/portfolio_intelligence.csv`.
9. Emit `data/logs/portfolio_intelligence_log.csv`.
10. Run focused tests and full validation commands.
11. Inspect generated output and log for forbidden semantics.
12. Confirm no Decision Engine, reporting, watchlist, or certified upstream code changed.
13. Prepare implementation evidence for audit.

## 21. Documentation Updates Required During Future Implementation

Future implementation must update:

- `docs/sprints/sprint_status_tracker.md` only as allowed by sprint lifecycle evidence

Future implementation may update:

- Sprint 5 implementation notes or implementation audit documents when created by authorized later phases

Future implementation must not rewrite this developer specification except to correct documented drift through governance.

## 22. Audit Handoff Requirements After Implementation

Implementation handoff must include:

- files created
- files modified
- generated artifacts
- exact validation commands run
- exact validation results
- row-count preservation evidence
- ticker-universe preservation evidence
- ordering preservation evidence
- upstream-column and upstream-value preservation evidence
- missing/partial source behavior evidence
- deterministic rerun evidence
- forbidden-column scan result
- forbidden-semantics scan result
- confirmation that Decision Engine was not changed
- confirmation that reporting was not changed
- confirmation that watchlist was not changed
- confirmation that certified upstream builders were not changed
- known residual risks, if any

## 23. Final Technical Lead Recommendation

READY FOR SPRINT 5 IMPLEMENTATION
