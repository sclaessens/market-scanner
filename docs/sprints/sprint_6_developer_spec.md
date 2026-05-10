# Sprint 6 Developer Specification - Decision Engine Core

## 1. Title and Status

Sprint 6 Developer Specification - Decision Engine Core.

Status: DEVELOPER SPECIFICATION COMPLETE / IMPLEMENTATION NEXT.

This specification authorizes future Sprint 6 implementation only after explicit implementation start. It does not implement runtime code, tests, generated CSVs, strategy optimization, portfolio optimization, or architecture redesign.

## 2. Reviewed Documents

Reviewed documents:

- `AGENTS.md`
- `README.md`
- `docs/sprints/README.md`
- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/execution_roadmap_v2.md`
- `docs/sprints/sprint_6_decision_engine_governance.md`
- `docs/audits/sprint_6_governance_audit.md`
- `docs/sprints/sprint_6_execution_plan.md`
- `docs/audits/sprint_6_execution_review.md`
- `docs/sprints/sprint_6_decision_engine_core.md`
- `docs/sprints/sprint_5_closeout.md`
- `docs/technical/Technical_Analysis_v3.md`
- `docs/functional/Functional_Analysis_v2.md`
- `docs/execution/execution_delivery_framework_v2.md`
- `docs/technical/decision_engine_design_v2.md`

Existing Sprint 6 developer specs reviewed:

- No prior Sprint 6 developer specification existed at specification start.

## 3. Executive Developer-Spec Conclusion

Sprint 6 may be implemented as a minimal deterministic Decision Engine Core.

Implementation must centralize Decision Engine authority inside `scripts/core/decision_engine.py`, preserve upstream artifacts without mutation, consume certified upstream evidence, emit deterministic and explainable downstream decisions, and provide audit evidence proving no authority leakage, no hidden filtering, no silent opportunity suppression, no adaptive behavior, and no hidden optimization.

`docs/sprints/sprint_6_decision_engine_core.md` remains historical architectural context and non-certified future-state ambition. It does not authorize probabilistic allocation, probabilistic smoothing, hidden scoring systems, advanced ranking engines, decision persistence, adaptive behavior, hidden optimization, opaque portfolio balancing, or allocation queues that suppress visibility.

## 4. Certified Governance and Execution-Review Inheritance

This developer specification inherits:

- Sprint 6 Governance Audit verdict: `SPRINT 6 PREPARATION CERTIFIED FOR EXECUTION PLANNING`
- Sprint 6 Execution Review verdict: `SPRINT 6 EXECUTION PLAN APPROVED FOR DEVELOPER SPECIFICATION`

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
- no hidden optimization
- no adaptive / ML authority
- no silent opportunity suppression
- no upstream mutation
- deterministic decisions only
- explainable decisions only
- audit-traceable decisions only

## 5. Implementation Objective

Implement a minimal deterministic Decision Engine Core that:

- consumes the certified Sprint 5 Portfolio Intelligence artifact as the authoritative evaluated universe
- emits one visible downstream decision row per input ticker/date row
- determines final action, allocation decision, and execution decision inside Decision Engine authority only
- emits deterministic arbitration and conflict-resolution rationale
- emits source provenance and decision contract metadata
- writes an audit log with preservation and authority-leakage evidence
- does not mutate or reinterpret upstream artifacts

## 6. Authorized Implementation Scope

Authorized scope:

- refactor `scripts/core/decision_engine.py` as the Sprint 6 Decision Engine Core
- create focused tests for Decision Engine governance behavior
- write `data/processed/final_decisions.csv` as the authoritative Decision Engine output
- write `data/logs/decision_engine_log.csv` as the Decision Engine audit log
- implement deterministic rule ordering and tie-breaking
- implement explicit final-action, allocation-decision, execution-decision, arbitration, rationale, and provenance fields
- implement fail-fast required input validation
- implement no-hidden-filtering and no-silent-suppression checks
- implement non-mutation checks in tests
- implement forbidden authority-leakage tests

## 7. Explicit Non-Scope

Out of scope:

- upstream builder changes
- upstream schema changes
- upstream generated artifact mutation
- reporting behavior changes
- watchlist behavior changes
- portfolio runtime behavior changes
- strategy optimization
- portfolio optimization
- threshold tuning outside explicitly documented deterministic rules
- machine learning
- adaptive behavior
- hidden optimization
- hidden persistence state
- decision persistence
- probabilistic smoothing
- advanced allocation queues
- opaque scoring systems
- opaque ranking engines
- runtime authority outside Decision Engine
- implementation of Sprint 7 stability or persistence scope

## 8. Authorized Files

Future implementation may modify or create only:

- `scripts/core/decision_engine.py`
- `tests/core/test_decision_engine.py`
- `data/processed/final_decisions.csv` as generated output from running the Decision Engine
- `data/logs/decision_engine_log.csv` as generated output from running the Decision Engine

Documentation may be updated after implementation only where required by implementation evidence:

- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/sprint_6_developer_spec.md` only for factual correction if implementation audit requires it

## 9. Forbidden Files

Future Sprint 6 implementation must not modify:

- scanner runtime files
- validation-layer runtime files
- context-layer runtime files
- fundamental-layer runtime files
- timing-state-layer runtime files
- portfolio-intelligence builder files
- watchlist runtime files
- portfolio runtime files
- reporting runtime files
- generated upstream CSV artifacts
- `data/processed/portfolio_intelligence.csv`
- `data/logs/portfolio_intelligence_log.csv`
- roadmap/governance doctrine except status evidence updates

## 10. Input Artifact Contract

Authoritative required input:

- `data/processed/portfolio_intelligence.csv`

Rationale:

- It is the certified Sprint 5 output.
- It preserves the upstream opportunity universe.
- It already carries certified validation, context, fundamental, timing-state, and portfolio-awareness metadata.
- It avoids uncertified `portfolio_state.csv` references from older Sprint 6 context.

Required input columns:

- `ticker`
- `date`
- `quality_state`
- `timing_state`
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

Optional input columns may be consumed if present, read-only, and preserved:

- `quality_reason`
- `quality_metadata_status`
- `source_data_status`
- `timing_reason`
- `extension_state`
- `participation_state`
- `timing_environment`
- `timing_metadata_status`
- `portfolio_source_provenance`
- `portfolio_classification_rationale`

Current watchlist artifacts:

- `data/watchlist/watchlist_status.csv` is not a required Sprint 6 input.
- If future implementation reads it, it may use only safe descriptive fields explicitly listed in implementation comments and tests.
- It must not consume legacy `action_now`, `urgency`, `entry_bias`, `why_now`, or similar execution/urgency fields.

Current portfolio artifacts:

- `data/portfolio/portfolio_review.csv` is not a required Sprint 6 input.
- If future implementation reads it, it may use only descriptive risk/exposure fields and must not create portfolio override authority.

Forbidden input:

- `portfolio_state.csv` must not be used unless a future governance artifact certifies it as safe.

## 11. Output Artifact Decision

Authoritative Sprint 6 output artifact:

- `data/processed/final_decisions.csv`

Decision:

- The older `decision_output.csv` concept is retired as conceptual legacy wording.
- `docs/technical/decision_engine_design_v2.md` identifies `data/processed/final_decisions.csv` as the current runtime Decision Engine output.
- Sprint 6 implementation must update that existing authoritative output rather than create `decision_output.csv`.

## 12. Output Schema Contract

Required output columns, in deterministic order:

- `ticker`
- `date`
- `final_action`
- `allocation_decision`
- `execution_decision`
- `portfolio_decision_state`
- `opportunity_decision_state`
- `arbitration_state`
- `allocation_rationale`
- `execution_rationale`
- `arbitration_reason`
- `conflict_resolution_reason`
- `source_provenance`
- `decision_contract_version`
- `input_row_hash`

Optional pass-through evidence columns may be included after required columns:

- `quality_state`
- `timing_state`
- `in_portfolio`
- `portfolio_position_state`
- `exposure_state`
- `diversification_state`
- `concentration_state`
- `overlap_state`
- `portfolio_metadata_status`

Forbidden output columns:

- `decision_output`
- `conviction_score`
- `ranking_score`
- `portfolio_score`
- `final_score`
- `recommended_trade`
- `recommended_weight`
- `optimal_weight`
- `target_weight`
- `allocation_queue`
- `execution_urgency`
- `urgency`
- `actionable`
- `execution_ready`

## 13. Log Schema Contract

Required log artifact:

- `data/logs/decision_engine_log.csv`

Required log columns:

- `run_id`
- `generated_at`
- `input_artifact`
- `output_artifact`
- `input_row_count`
- `output_row_count`
- `row_count_preserved`
- `ticker_date_universe_preserved`
- `input_order_preserved`
- `upstream_artifacts_mutated`
- `decision_contract_version`
- `forbidden_authority_leakage_detected`
- `hidden_filtering_detected`
- `silent_suppression_detected`
- `rationale_completeness_status`
- `source_provenance_status`
- `classification_rationale`

Log rows may be per run or per input row, but tests must prove the log contains enough evidence to audit preservation, provenance, rationale completeness, and authority leakage.

## 14. Row Identity and Universe Preservation Contract

The Decision Engine must preserve:

- input row count
- ticker/date universe
- upstream row visibility
- upstream informational richness
- deterministic ordering unless explicitly documented as operational-only output ordering

No input opportunity may disappear silently.

If multiple rows share a ticker/date key, implementation must fail fast unless the developer can prove and test deterministic duplicate handling without suppression.

## 15. Deterministic Decision Contract

Implementation must be deterministic under identical inputs.

Required controls:

- explicit rule order
- explicit tie-breaking
- no randomness
- no live external state without captured provenance
- no hidden persistence state
- no adaptive behavior
- no implicit filesystem ordering
- stable column ordering
- stable row ordering
- reproducible output content

Decision contract version:

- use a constant such as `SPRINT_6_DECISION_ENGINE_CORE_V1`
- emit it in output and log artifacts

## 16. Final-Action Contract

Allowed `final_action` values:

- `BUY`
- `SELL`
- `HOLD`
- `TRIM`
- `WAIT`
- `REMOVE`
- `REVIEW`
- `PREPARE`
- `NO_ACTION`

These values may exist only in Decision Engine-owned output, logs, tests, and governance documentation.

Every `final_action` must have:

- `allocation_rationale`
- `execution_rationale`
- `arbitration_reason`
- `conflict_resolution_reason`
- source provenance

No `final_action` may be generated by upstream layers.

## 17. Allocation Decision Contract

Allowed `allocation_decision` values:

- `ALLOCATE`
- `DO_NOT_ALLOCATE`
- `MAINTAIN`
- `REDUCE`
- `EXIT`
- `REVIEW_REQUIRED`
- `NO_ALLOCATION_ACTION`

Allocation decisions are Decision Engine-only authority.

No numeric allocation sizing, recommended weight, target weight, optimal weight, or capital amount may be implemented in Sprint 6.

## 18. Execution Decision Contract

Allowed `execution_decision` values:

- `EXECUTE`
- `DO_NOT_EXECUTE`
- `MONITOR`
- `REVIEW_REQUIRED`
- `NO_EXECUTION_ACTION`

Execution decision values must not create urgency semantics. Do not implement `execution_aggressiveness`, `urgent`, `immediate`, `high_priority`, or similar values in Sprint 6.

## 19. Arbitration and Conflict-Resolution Contract

Allowed `arbitration_state` values:

- `NO_CONFLICT`
- `PORTFOLIO_POSITION_CONFLICT`
- `MISSING_METADATA`
- `TIMING_CONFLICT`
- `QUALITY_CONFLICT`
- `REVIEW_REQUIRED`

Arbitration is deterministic and visibility-preserving.

If arbitration results in no allocation or no execution, the row must remain in `final_decisions.csv` with rationale.

## 20. Rationale and Provenance Contract

Every output row must include:

- source artifact provenance
- input row identity evidence
- allocation rationale
- execution rationale
- arbitration reason
- conflict-resolution reason
- decision contract version

Rationales must describe Decision Engine reasoning only. They must not rewrite upstream classifications or imply upstream authority.

## 21. No-Hidden-Filtering and No-Silent-Suppression Contract

Implementation must prove:

- output row count equals input row count
- ticker/date universe equals input ticker/date universe
- every no-allocation outcome remains visible
- every no-execution outcome remains visible
- missing optional metadata does not remove rows
- conflicts produce rationale, not suppression

## 22. Upstream Non-Mutation Contract

Implementation must not write to:

- `data/processed/portfolio_intelligence.csv`
- `data/logs/portfolio_intelligence_log.csv`
- validation artifacts
- context artifacts
- fundamental artifacts
- timing artifacts
- watchlist artifacts
- portfolio artifacts
- reporting artifacts

Tests must snapshot relevant upstream input content before and after Decision Engine execution.

## 23. Authority-Leakage Prevention Contract

Implementation must prove forbidden authority does not leak outside `scripts/core/decision_engine.py`.

Forbidden outside Decision Engine runtime:

- BUY/SELL/HOLD/TRIM/REMOVE final-action generation
- tradeability
- conviction
- allocation priority
- execution readiness
- urgency
- scoring authority
- ranking authority
- recommendation authority

Existing legacy references must be classified during validation as either expected historical references, test assertions, or leakage.

## 24. Concept Inclusion/Exclusion Table

| Concept | Sprint 6 Decision | Implementation Instruction |
|---|---|---|
| Probabilistic evaluation | Omit probabilistic mechanics; use deterministic rules. | Do not implement randomness or probability calculations. |
| Conviction scoring | Omit. | Do not add `conviction_score` or numeric conviction. |
| Opportunity ranking | Narrow to deterministic row-preserving arbitration only. | Do not rank or suppress rows. |
| Allocation priority | Omit numeric priority; use `allocation_decision`. | Do not add `allocation_priority`. |
| Tradeability scoring | Omit scoring; use allocation/execution decision states. | Do not add `tradeability` or tradeability score columns. |
| Portfolio balancing | Narrow to conflict-resolution evidence. | Use Portfolio Intelligence metadata only as evidence. |
| Execution aggressiveness | Omit. | Use `execution_decision`; do not add urgency/aggressiveness. |
| Decision persistence | Defer to Sprint 7. | Do not implement state persistence. |
| Probabilistic smoothing | Defer to Sprint 7. | Do not implement smoothing. |
| Escalation tracking | Omit. | Do not create persistent or urgency-like escalation fields. |
| Allocation queues | Omit. | Do not implement queues. |
| Decision distributions | Allow only in logs as descriptive output counts. | Do not use distributions as optimization targets. |

## 25. Required Implementation Steps

Future developer must:

1. Read this specification and referenced governance artifacts.
2. Inspect current `scripts/core/decision_engine.py`.
3. Replace legacy input contract with `data/processed/portfolio_intelligence.csv` as required input.
4. Preserve `data/processed/final_decisions.csv` as authoritative output.
5. Add `data/logs/decision_engine_log.csv` output.
6. Implement fail-fast required input validation.
7. Implement deterministic row-preserving decision rules.
8. Emit required output schema in deterministic order.
9. Emit required log schema.
10. Add focused tests in `tests/core/test_decision_engine.py`.
11. Run all required validation commands.
12. Document implementation evidence for implementation audit.

## 26. Required Tests

Create or update:

- `tests/core/test_decision_engine.py`

Required tests:

- output schema contract test
- required input schema contract test
- one decision per ticker/date row test
- deterministic output under identical inputs test
- deterministic tie-breaking test
- no upstream artifact mutation test
- no hidden filtering test
- no silent opportunity suppression test
- ticker/date visibility preservation test
- source provenance completeness test
- decision rationale completeness test
- forbidden semantics outside Decision Engine test
- no Reporting authority leakage test
- no Watchlist authority leakage test
- no Portfolio authority leakage test
- no Portfolio Intelligence authority leakage test
- missing input fail-fast test
- missing optional metadata behavior test
- log schema contract test
- forbidden output column test
- deferred concept absence test for persistence and smoothing

## 27. Required Validation Commands

Minimum required validation:

```bash
pytest tests/core/test_decision_engine.py
pytest tests/core
pytest
git diff --check
git status --short
```

If running the Decision Engine directly is part of implementation validation:

```bash
python scripts/core/decision_engine.py
```

## 28. Required Grep and Semantic Checks

Required governance scans:

```bash
grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "tradeable" scripts/ | grep -v decision_engine.py
grep -R "conviction" scripts/ | grep -v decision_engine.py
grep -R "allocation_priority" scripts/ | grep -v decision_engine.py
```

Additional Sprint 6 semantic checks must inspect Sprint 6 runtime files and generated artifacts for forbidden terms:

- `conviction_score`
- `ranking_score`
- `portfolio_score`
- `final_score`
- `recommended`
- `preferred`
- `suitability`
- `attractiveness`
- `optimal`
- `urgency`
- `execution_ready`
- `actionable`
- `allocation_queue`
- `persistence`
- `smoothing`

Any finding outside Decision Engine-owned implementation, tests, or governance documentation must be classified before implementation audit.

## 29. Required Implementation Audit Evidence

Implementation handoff must include:

- files changed
- generated artifacts
- output row count
- input row count
- ticker/date universe preservation evidence
- upstream non-mutation evidence
- deterministic output evidence
- rationale completeness evidence
- source provenance evidence
- authority-leakage scan results
- grep/semantic check results
- test results
- known limitations
- backlog impact assessment readiness

## 30. Risks and Controls

Risk: legacy Decision Engine continues to consume uncertified watchlist urgency/action fields.

Control: required input is `portfolio_intelligence.csv`; watchlist artifacts are not required and forbidden fields may not be consumed.

Risk: output loses legacy columns expected downstream.

Control: Sprint 6 may change Decision Engine output schema only under this spec; Reporting changes are out of scope and must be handled in later governance if needed.

Risk: deterministic rules become hidden scoring.

Control: omit numeric conviction, ranking, tradeability, and allocation-priority scores.

Risk: no-action states become hidden filtering.

Control: preserve row visibility and require rationale for every row.

Risk: portfolio evidence becomes portfolio override authority.

Control: Portfolio Intelligence metadata may inform conflict rationale only; Decision Engine retains authority.

## 31. Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Rationale: deferred persistence and smoothing remain covered by Sprint 7 roadmap scope. Other exclusions are Sprint 6 implementation constraints rather than new deferred backlog items.

## 32. Final Technical Lead Recommendation

READY FOR SPRINT 6 IMPLEMENTATION
