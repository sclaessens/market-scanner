# Sprint 4 Execution Planning — Timing State Layer

## 1. Executive Planning Conclusion

Status: READY FOR SPRINT 4 DEVELOPER SPECIFICATION

Sprint 4 execution planning is complete as a governance and delivery-planning artifact only.

This document prepares future implementation sequencing for a governance-clean Timing State Layer. It does not authorize runtime code changes, test changes, generated data changes, strategy changes, threshold changes, Decision Engine changes, watchlist behavior changes, portfolio behavior changes, reporting behavior changes, or developer implementation.

The future implementation must create or purify a Timing State Layer as a descriptive, classification-only, enrichment-only stage. It must not reuse legacy watchlist readiness/status-sorting behavior as-is.

## 2. Certified Baseline

Sprint 4 inherits the certified baseline:

- Sprint 0 = CERTIFIED COMPLETE / CLOSED
- Sprint 1 = CERTIFIED COMPLETE / CLOSED
- Sprint 2 = CERTIFIED COMPLETE / CLOSED
- Sprint 3 = CERTIFIED COMPLETE / CLOSED
- Sprint 4 preparation = COMPLETE
- Sprint 4 governance audit = PASS
- Sprint 4 architecture validation = PASS

Certified doctrine remains binding:

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

## 3. Architecture Validation Inheritance

Sprint 4 execution planning inherits `docs/audits/sprint_4_architecture_validation.md`.

Validated architecture direction:

scanner -> validation_layer -> context_layer -> fundamental_layer -> timing_state_layer -> watchlist -> portfolio -> decision_engine -> reporting

Architecture validation confirmed:

- Timing State Layer is technically feasible as pure enrichment
- preferred upstream row universe is `data/processed/fundamental_quality.csv`
- auxiliary timing source data may be read only after specification
- Timing must not read Decision Engine, portfolio, reporting, or manual watchlist action files as authoritative row universe
- legacy watchlist readiness/status-sorting behavior is not safe as-is
- distribution preservation must be measurable

## 4. Sprint 4 Delivery Objective

Prepare future implementation of a Timing State Layer that:

- appends descriptive timing metadata
- classifies timing-condition observations
- classifies extension and compression observations
- classifies pullback, breakout, consolidation, participation, and pattern observations
- emits deterministic audit metadata
- emits deterministic logs
- preserves the full upstream row universe

The layer must not create execution authority or allocation semantics.

## 5. In-Scope Implementation Preparation

Execution planning prepares:

- future implementation sequence
- developer handoff controls
- data contract guardrails
- schema guardrails
- logging and audit expectations
- future test categories
- deterministic behavior checks
- distribution-preservation checks
- cross-layer boundary checks
- Decision Engine boundary checks
- watchlist legacy-risk controls
- Technical Lead audit expectations
- closeout conditions

## 6. Out-of-Scope Boundaries

Out of scope for this execution plan:

- runtime implementation
- test implementation
- generated CSV/data changes
- strategy optimization
- threshold tuning
- filters
- ranking logic
- scoring logic
- allocation logic
- Decision Engine logic
- watchlist behavior changes
- portfolio behavior changes
- reporting behavior changes
- runtime schema creation
- developer execution
- Sprint 4 closeout

## 7. Future Implementation Sequence

Future Sprint 4 work must proceed in this sequence:

1. Execution planning
2. Developer specification
3. Developer implementation
4. Technical Lead implementation audit
5. Corrections, if required
6. Test / validation verification
7. Commit / push hygiene
8. Sprint closeout
9. Sprint certification

Developer implementation may not begin until a developer specification explicitly authorizes the implementation scope.

## 8. Data Contract Guardrails

Future developer specification must define:

- authoritative upstream row-universe artifact
- auxiliary timing source artifacts
- primary row key
- required input columns
- allowed optional columns
- duplicate-key handling
- missing-key handling
- missing auxiliary timing-data behavior
- output artifact name
- output schema
- log artifact name
- log schema

Planning direction:

- preferred upstream row universe: `data/processed/fundamental_quality.csv`
- expected primary row key: `ticker` + `date`
- allowed auxiliary sources may include `data/processed/scanner_ranked.csv`, `data/processed/entry_quality_metrics.csv`, and per-ticker OHLCV files under `data/processed/`
- auxiliary sources must not control row inclusion

Forbidden authoritative inputs:

- `data/processed/final_decisions.csv`
- portfolio files
- reporting files
- Telegram outputs
- manual watchlist action files
- `watchlist_active.csv` as row-universe authority

## 9. Schema Guardrails

Future schema must be descriptive only.

Governance-safe candidate direction may include:

- `ticker`
- `date`
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
- `source_timestamp`
- `generated_at`

Developer specification must freeze exact schema before implementation.

Schema must not include names implying:

- readiness
- quality
- priority
- rank
- score
- actionability
- approval
- rejection
- execution
- allocation
- conviction
- recommendation
- expected return
- expected alpha

## 10. Logging And Audit Requirements

Future implementation must emit a Timing audit log.

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

Logging must remain observational only. It must not create priority, readiness, actionability, approval, rejection, conviction, allocation, execution, or ranking semantics.

## 11. Future Test Strategy

Future tests are required but are not written during execution planning.

Required test categories:

- output schema order and allowlist
- forbidden-field absence
- forbidden semantic-value absence
- row-count preservation
- ticker/date key-set preservation
- upstream-order preservation
- duplicate ticker/date fail-fast behavior
- missing ticker/date fail-fast behavior
- missing auxiliary timing-source row preservation
- no filtering by extension, compression, breakout, pullback, pattern, or metadata status
- no sorting by Timing state
- no mutation of upstream fields
- deterministic repeated runs
- log schema and count consistency
- no coupling to Decision Engine, portfolio, reporting, or manual watchlist action files

## 12. Distribution-Preservation Controls

Future implementation must prove:

- input row count equals output row count
- input ticker/date key set equals output ticker/date key set
- upstream order is preserved exactly
- missing auxiliary timing data does not remove rows
- weak, neutral, incomplete, extended, stale, missing-source, and unknown rows remain visible
- output is not sorted by Timing state
- output is not grouped into preferred subsets

Any row-count or key-set mismatch must fail validation.

## 13. Non-Mutating Enrichment Controls

Timing implementation may append new descriptive columns only.

It must not:

- mutate upstream classifications
- overwrite upstream columns
- normalize away upstream signals
- rewrite `structure_state`
- rewrite `valid_setup`
- rewrite `context_strength`
- rewrite `leadership_state`
- rewrite `quality_state`
- rewrite quality profile metadata
- change upstream row identity

Future tests must compare protected upstream fields before and after enrichment where fields are passed through or referenced.

## 14. Cross-Layer Boundary Controls

Validation boundary:

- Timing may not become a second Validation Layer.
- Timing may not invalidate structures.
- Timing may not reinterpret `valid_setup`.

Context boundary:

- Timing may not use leadership to create preference or actionability.
- Timing may not combine leadership with timing into composite intelligence.

Fundamental boundary:

- Timing may not use quality to upgrade or downgrade timing state.
- Timing may not create quality-adjusted timing.

Portfolio boundary:

- Timing may not read portfolio state or open positions.
- Timing may not create exposure or risk semantics.

Reporting boundary:

- Timing may not alter communication grouping, action sections, or omission behavior.

## 15. Decision Engine Boundary Controls

The Decision Engine remains the only allocation authority.

Future Timing implementation must not produce:

- final action
- tradeability
- conviction
- allocation eligibility
- allocation priority
- execution style
- urgency
- BUY logic
- SELL logic
- REMOVE logic
- portfolio-aware allocation

Timing output must be treated as raw descriptive input that a future Decision Engine-owned sprint may interpret only under explicit Decision Engine governance.

## 16. Watchlist Legacy-Risk Controls

Legacy watchlist behavior must not be reused as-is.

Unsafe legacy patterns include:

- readiness-style states
- status-based sorting
- watchlist inclusion changes
- setup expiry as opportunity removal
- failure states as rejection
- thresholds that imply execution preference
- active watchlist membership as row-universe authority
- action-like watch/unwatch effects

Future developer specification must either:

- create a new governance-clean Timing builder, or
- explicitly define a narrow purification scope for existing watchlist code before any reuse

Required controls:

- no change to `watchlist_active.csv` membership
- no timing-state sorting
- no status ordering
- no row suppression under `STALE`, `FAILED`, `EXTENDED`, or missing data
- no readiness or execution-ready schema names
- no actionability language in output values

## 17. Reporting Boundary Controls

Reporting remains communication-only.

Future implementation may not:

- change `scripts/reporting/`
- change Telegram sections
- change report ordering
- change omission behavior
- create Timing-based report priorities
- communicate Timing metadata as recommendation

Reporting integration, if any, must be deferred until after Decision Engine-owned interpretation and reporting governance.

## 18. Forbidden Semantics Checklist

Future developer specification, implementation, tests, logs, and output schema must exclude:

- `tradeable`
- `approved`
- `rejected`
- `actionable`
- `urgent`
- `urgency`
- `priority`
- `high_conviction`
- `conviction`
- `execution_ready`
- `buy_candidate`
- `sell_candidate`
- `best_opportunity`
- `preferred_setup`
- `ranking_score`
- `timing_score`
- `final_score`
- `allocation_weight`
- `expected_return`
- `alpha_score`
- `opportunity_rank`
- `BUY`
- `SELL`
- `REMOVE`

Forbidden terms may appear only in governance documentation or negative tests as prohibited examples.

## 19. Fail-Fast Conditions

Future implementation must fail fast when:

- authoritative upstream input is missing
- authoritative upstream input is empty
- required row-key columns are missing
- `ticker` is missing or blank
- `date` is missing or blank
- upstream duplicate ticker/date rows exist
- output row count differs from input row count
- output ticker/date key set differs from input key set
- output ordering differs from upstream ordering
- forbidden output columns are present
- forbidden semantic values are present
- protected upstream fields are mutated

Missing auxiliary timing data must not fail the run unless the developer specification explicitly defines an auxiliary source as required. Default behavior must preserve rows with descriptive source-data metadata.

## 20. Developer Handoff Requirements

Developer specification must provide:

- exact files allowed to be created or modified
- exact files forbidden to modify
- exact input artifact contract
- exact auxiliary artifact contract
- exact output schema
- exact log schema
- exact function/script entrypoint
- exact validation commands
- exact tests to write
- exact forbidden-field checks
- exact generated-artifact policy
- exact documentation update rules

Developer handoff must explicitly state:

- no implementation outside the approved file scope
- no Decision Engine changes
- no portfolio changes
- no reporting changes
- no legacy watchlist reuse as-is
- no generated artifacts committed unless explicitly authorized

## 21. Technical Lead Audit Requirements

Technical Lead implementation audit must verify:

- implementation stayed within approved file scope
- output schema matches developer specification
- log schema matches developer specification
- row count is preserved
- ticker/date key set is preserved
- upstream order is preserved
- forbidden fields and values are absent
- upstream classifications are not mutated
- no Decision Engine, portfolio, reporting, or unsafe watchlist coupling was introduced
- tests pass
- validation commands pass
- generated artifact policy was followed
- git status is understood and documented

## 22. Commit / Validation Hygiene

Future implementation must report:

- files changed
- generated files created
- generated files intentionally left untracked
- tests run
- grep checks run
- pipeline command run or explicitly not run with reason
- dirty-file status after validation

Required future validation direction:

```bash
pytest
grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "tradeable" scripts/ | grep -v decision_engine.py
```

Additional Sprint 4-specific forbidden-term checks must be defined in the developer specification.

## 23. Sprint Closeout Requirements

Sprint 4 closeout may occur only after:

- developer implementation is complete
- Technical Lead implementation audit passes
- required corrections, if any, are complete
- validation evidence is documented
- generated artifact handling is documented
- no unauthorized runtime, test, data, strategy, threshold, Decision Engine, portfolio, reporting, or watchlist behavior changes remain
- sprint status tracker is updated
- closeout document certifies Sprint 4 completion

## 24. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Timing metadata becomes actionability | Ban readiness/actionability fields and values |
| Timing metadata becomes priority | Preserve upstream order and ban priority/rank semantics |
| Timing metadata becomes conviction | Ban conviction and quality-like interpretation |
| Timing metadata becomes scoring | Ban score fields and composite metrics |
| Legacy watchlist sorting leaks priority | Do not reuse status sorting as-is |
| Legacy watchlist readiness leaks execution semantics | Avoid readiness schema and values |
| Extension state suppresses opportunities | Preserve all rows regardless of extension |
| Missing timing data suppresses opportunities | Emit descriptive source-data metadata |
| Timing becomes second Validation Layer | Do not mutate or reinterpret Validation fields |
| Timing becomes mini Decision Engine | Ban final action, tradeability, allocation, conviction, urgency, and execution style |
| Reporting changes imply recommendation | Defer reporting changes |

## 25. Acceptance Criteria

Future developer implementation acceptance criteria:

- governance-clean Timing artifact created according to developer specification
- full upstream row universe preserved
- schema is descriptive only
- forbidden fields and values absent
- upstream fields not mutated
- deterministic output proven
- log emitted with required count and distribution fields
- tests pass
- validation checks pass
- no unauthorized layer coupling introduced

Future Technical Lead implementation audit acceptance criteria:

- implementation is scope-compliant
- governance doctrine is preserved
- distribution preservation is proven
- Decision Engine authority is intact
- legacy watchlist risk is controlled
- generated artifact handling is clean
- final recommendation supports Sprint 4 closeout or requires specific corrections

Future Sprint 4 closeout acceptance criteria:

- implementation audit passes
- no required corrections remain
- sprint tracker is updated
- closeout document certifies Sprint 4 complete
- Sprint 4 remains aligned with certified doctrine

## 26. Recommended Next Step

Proceed to Sprint 4 developer specification.

The developer specification must translate this execution plan into exact file scope, exact schemas, exact validation commands, exact test requirements, and exact implementation boundaries.

## 27. Final Scrum Master Recommendation

READY FOR SPRINT 4 DEVELOPER SPECIFICATION
