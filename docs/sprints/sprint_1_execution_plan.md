# Sprint 1 Execution Plan — Structure Classification Alignment & Governance Refinement

## 1. Sprint Status

Status: CERTIFIED COMPLETE

Closeout note: Sprint 1 is formally certified complete. See `docs/sprints/sprint_1_closeout.md`.

Sprint 1 Institutional Re-Alignment Audit is complete.

Sprint 1 Documentation Correction is complete.

Final Governance Re-Review result: READY FOR DEVELOPER SPEC.

This document is the Scrum Master execution package. It is not an implementation prompt and does not authorize architecture redesign, trading-logic changes, or allocation changes.

## 2. Governance Baseline

Sprint 1 inherits the certified Sprint 0 doctrine:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority
- upstream layers classify only
- reporting communicates only
- no upstream tradeability
- no hidden filtering
- no hidden allocation semantics outside Decision Engine

Validation Layer responsibility:

- classify structure
- preserve opportunity distribution
- emit governance-clean structure metadata
- avoid all allocation, action, conviction, urgency, and execution-readiness semantics

## 3. Sprint Objective

Sprint 1 aligns and stabilizes the Validation Layer contract around:

- `structure_state`
- `structure_reason`

Sprint 1 also clarifies deprecated compatibility aliases:

- `valid_setup`
- `validation_reason`

The objective is governance alignment, terminology correction, schema stabilization, compatibility cleanup, test coverage, and CI/governance enforcement.

## 4. Scrum Master Execution Summary

Sprint 1 must be executed as a constrained governance refinement sprint.

The work should start with schema and test inspection, then move through narrowly scoped validation-layer contract changes only if required. The team must avoid broad rewrites. The sprint succeeds when the Validation Layer contract is explicit, tests enforce classification-first behavior, and CI/governance checks prevent reintroduction of upstream tradeability or filtering-first semantics.

No sprint work may change Decision Engine allocation behavior.

## 5. Sprint Workstreams

### Workstream A — Validation Contract Stabilization

Scope:

- confirm `structure_state` is the primary runtime contract
- confirm `structure_reason` is the primary reason contract
- confirm `valid_setup` is a deprecated compatibility alias only
- confirm `validation_reason` is a deprecated compatibility alias only

Expected outcome:

- Validation contract is clear, deterministic, and classification-first
- new code prefers `structure_state` and `structure_reason`
- compatibility aliases do not imply tradeability or allocation eligibility

### Workstream B — Forbidden Field Enforcement

Forbidden Validation Layer outputs:

- `tradeable_setup`
- `context_tradeable`
- `tradeability`
- `conviction`
- `allocation_priority`
- `final_action`
- `urgency`
- `actionable`
- `BUY`
- `SELL`
- `HOLD`
- `TRIM`
- `REMOVE`

Expected outcome:

- forbidden fields are absent from Validation Layer runtime output
- tests and CI checks enforce absence
- any forbidden terms in tests appear only when asserting absence

### Workstream C — Descriptive Metadata Governance

Descriptive-only metadata:

- `entry_quality`
- extension
- volume quality
- breakout quality
- RR quality

Expected outcome:

- descriptive metadata does not change `structure_state`
- descriptive metadata does not block `valid_setup`
- descriptive metadata does not eliminate opportunities
- descriptive metadata does not become a hidden filter

### Workstream D — Schema And Output Validation

Files and outputs:

- `data/processed/validation_layer.csv`
- `data/processed/entry_quality_metrics.csv`, if relevant

Expected outcome:

- `validation_layer.csv` exposes structure-classification fields only
- `entry_quality_metrics.csv`, if used, remains descriptive metadata only
- generated historical artifacts with stale fields are treated as artifact hygiene risks unless separately scoped

### Workstream E — Distribution Observability

Monitor:

- structure-state distribution
- missing-data distribution
- compatibility drift
- migration drift

Expected outcome:

- distribution changes are logged or observed
- distributions are not optimized through hidden filtering
- distribution monitoring does not create new thresholds or strategy logic

### Workstream F — Test And CI Enforcement

Coverage areas:

- pytest validation tests
- schema assertions
- forbidden-field grep checks
- deterministic output validation

Expected outcome:

- tests prove classification-first validation behavior
- CI/governance checks prevent forbidden-field regression
- full repository tests pass before sprint completion

## 6. Execution Sequence

1. Review Sprint 1 governance baseline and scope boundaries.
2. Inspect current Validation Layer runtime schema and tests.
3. Inspect generated validation output schema.
4. Identify whether implementation changes are required or whether current runtime already satisfies the corrected contract.
5. If changes are required, limit them to Validation Layer schema/contract code, tests, CI checks, or documentation.
6. Add or update tests for compatibility aliases and forbidden fields.
7. Run required tests and governance checks.
8. Perform Technical Lead governance review.
9. Perform Functional Analyst review.
10. Record final sprint outcome and any unresolved artifact hygiene risks.

## 7. Developer Handoff Requirements

The Technical Lead Developer Spec must include:

- exact files allowed to change
- exact files out of scope
- expected Validation Layer schema
- compatibility alias rules
- forbidden field list
- required tests
- required grep checks
- validation commands
- documentation update expectations

The handoff must not contain:

- new trading rules
- threshold optimization
- scanner/context/watchlist/portfolio changes
- Decision Engine allocation changes
- reporting changes
- architecture redesign

## 8. Required Files To Inspect

Governance and sprint documents:

- `AGENTS.md`
- `README.md`
- `docs/sprints/sprint_0_governance_status.md`
- `docs/sprints/sprint_1_structure_classification.md`
- `docs/audits/sprint_0_final_governance_audit.md`
- `docs/audits/documentation_alignment_audit_post_sprint_0.md`
- `docs/technical/Technical_Analysis_v3.md`
- `docs/functional/Functional_Analysis_v2.md`
- `docs/execution/execution_delivery_framework_v2.md`
- `docs/technical/decision_engine_design_v2.md`

Runtime and test files:

- `scripts/core/build_validation_layer.py`
- `scripts/core/build_entry_quality_backfill.py`, inspection only unless separately scoped
- `tests/core/test_build_validation_layer.py`
- `tests/core/test_entry_quality.py`
- `tests/core/test_build_entry_quality_backfill.py`

Generated schema inspection:

- `data/processed/validation_layer.csv`
- `data/processed/entry_quality_metrics.csv`

## 9. Required Files That May Be Changed

Sprint 1 may change only:

- `docs/sprints/sprint_1_structure_classification.md`
- `docs/sprints/sprint_1_execution_plan.md`
- `scripts/core/build_validation_layer.py`, only if required for Validation Layer schema/contract alignment
- validation-related tests under `tests/core/`
- CI/governance check configuration, if present and required

Any change outside this list requires Technical Lead approval before implementation.

## 10. Files Explicitly Out Of Scope

Do not change:

- scanner logic
- context logic
- watchlist logic
- portfolio logic
- Decision Engine allocation logic
- reporting behavior
- trading thresholds
- strategy scoring
- BUY/SELL/HOLD/TRIM/REMOVE behavior

Explicitly out-of-scope paths:

- `scripts/core/scanner.py`
- `scripts/core/build_context_layer.py`
- `scripts/core/build_context_backfill.py`
- `scripts/watchlist/`
- `scripts/portfolio/`
- `scripts/core/decision_engine.py`
- `scripts/reporting/`
- `scripts/telegram/`

Generated artifacts with stale columns are artifact hygiene risks and are not active Sprint 1 runtime scope unless explicitly added as a separate cleanup subtask.

## 11. Required Tests

Required test commands:

```bash
.venv/bin/python3 -m pytest
```

Required validation focus:

- validation schema exactness
- `structure_state` and `structure_reason` primary contract
- `valid_setup` compatibility alias remains structure-only
- `validation_reason` compatibility alias remains structure-only
- RR quality does not gate structure state
- extension does not gate structure state
- volume quality does not gate structure state
- entry quality remains descriptive metadata
- forbidden validation fields are absent

Pipeline validation, if implementation changes runtime behavior:

```bash
PATH="/Users/stevenclaessens/market-scanner/.venv/bin:$PATH" python3 scripts/run_full_pipeline.py
```

## 12. Required CI/Governance Checks

Run these checks against Validation Layer source and tests:

```bash
grep -R "tradeable_setup" scripts/core/build_validation_layer.py tests/core/test_build_validation_layer.py
grep -R "context_tradeable" scripts/core/build_validation_layer.py tests/core/test_build_validation_layer.py
grep -R "allocation_priority" scripts/core/build_validation_layer.py tests/core/test_build_validation_layer.py
grep -R "final_action" scripts/core/build_validation_layer.py tests/core/test_build_validation_layer.py
grep -R "urgency" scripts/core/build_validation_layer.py tests/core/test_build_validation_layer.py
grep -R "BUY" scripts/core/build_validation_layer.py tests/core/test_build_validation_layer.py
grep -R "SELL" scripts/core/build_validation_layer.py tests/core/test_build_validation_layer.py
grep -R "REMOVE" scripts/core/build_validation_layer.py tests/core/test_build_validation_layer.py
```

Expected interpretation:

- source should not emit forbidden fields
- tests may reference forbidden terms only to assert absence
- no grep hit should authorize upstream tradeability, allocation, conviction, urgency, actionability, or action logic

## 13. Acceptance Criteria

Sprint 1 is acceptable only when:

- `structure_state` is the authoritative validation contract
- `structure_reason` is the authoritative validation reason field
- `valid_setup` is compatibility-only
- `validation_reason` is compatibility-only
- Validation emits no tradeability fields
- Validation emits no allocation fields
- Validation emits no conviction fields
- Validation emits no action fields
- descriptive metadata does not gate structure classification
- entry quality and extension are not gates
- distribution monitoring is observability only
- CI/governance checks pass
- Technical Lead governance review passes
- Functional Analyst review passes

## 14. Definition Of Done

Sprint 1 is done only when:

- all accepted work remains within approved scope
- no runtime architecture redesign occurred
- no Decision Engine allocation behavior changed
- no scanner/context/watchlist/portfolio/reporting behavior changed
- tests pass
- governance grep checks are interpreted and documented
- generated output schemas are inspected if runtime code changed
- Technical Lead review confirms governance compliance
- Functional Analyst review confirms functional clarity
- Sprint status is documented

## 15. Risks And Controls

| Risk | Severity | Control |
|---|---|---|
| `valid_setup` is treated as tradeability | HIGH | Keep it deprecated and structure-only |
| `validation_reason` becomes primary contract again | MEDIUM | Prefer `structure_reason` in new code/tests |
| entry quality or extension becomes a hidden gate | HIGH | Test metadata-only behavior |
| distribution monitoring becomes filtering | HIGH | Treat distribution metrics as observability only |
| stale generated artifacts mislead developers | MEDIUM | Document as artifact hygiene risk unless separately scoped |
| developer broadens scope into Decision Engine or scanner | HIGH | Enforce explicit out-of-scope list |
| reporting or portfolio terms are mistaken for validation scope | MEDIUM | Restrict Sprint 1 to Validation Layer contract |

## 16. Review Checkpoints

Checkpoint 1 — Pre-Spec Readiness:

- Scrum Master confirms scope, boundaries, and workstreams
- Technical Lead confirms no additional documentation correction is required

Checkpoint 2 — Developer Spec Review:

- Technical Lead converts this plan into a scoped developer spec
- Scrum Master verifies spec does not add implementation beyond Sprint 1 boundaries

Checkpoint 3 — Implementation Review:

- Technical Lead reviews code/test changes if implementation is required
- Functional Analyst reviews language and behavior for classification-first consistency

Checkpoint 4 — Sprint Close:

- tests and governance checks pass
- any unresolved generated-artifact risks are documented
- sprint status is updated

## 17. Final Scrum Master Recommendation

READY FOR TECHNICAL LEAD DEVELOPER SPEC

The Sprint 1 execution package is ready for Technical Lead conversion into a developer specification. No blocker-level contradictions were found in active governance documents. Low-severity historical terminology remains constrained by Sprint 0 certification and the corrected Sprint 1 document.
