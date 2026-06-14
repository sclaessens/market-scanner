# Sprint 2 Execution Plan — Cross-Sectional Leadership Layer

## 1. Sprint Status

Status: CERTIFIED COMPLETE

Sprint 2 is certified complete. See `docs/sprints/sprint_2_closeout.md`.

Execution completed with test hardening and artifact-hygiene documentation only. Runtime Context code was not changed.

## 2. Governance Inheritance

Sprint 2 inherits the certified Sprint 0 and Sprint 1 doctrine:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority
- upstream layers classify only
- reporting communicates only
- no upstream tradeability
- no hidden filtering
- no hidden allocation semantics outside Decision Engine

Sprint 1 certified the Validation Layer contract. Sprint 2 must preserve that governance baseline while hardening the Context Layer as leadership and relative-strength classification only.

## 3. Sprint Objective

Sprint 2 stabilizes the Cross-Sectional Leadership Layer so Context remains a governance-clean classification layer.

The sprint focuses on:

- context schema and contract clarity
- forbidden-field enforcement
- historical artifact hygiene
- distribution preservation
- sector-relative enrichment governance
- test and CI hardening

Sprint 2 is not a strategy redesign sprint and must not change allocation behavior.

## 4. Scrum Master Execution Summary

The Sprint 2 governance audit found the active Context runtime governance-clean:

- no active tradeability leakage
- no active allocation leakage
- no hidden filtering found
- current runtime schema is governance-clean
- weak, neutral, strong, and leading context states are classification-only
- sector-relative data is enrichment-only and nullable

The main Sprint 2 work is governance stabilization, not runtime reconstruction. Developers must inspect first, prove gaps before changing runtime code, and prefer test/schema enforcement over implementation changes where the runtime already satisfies doctrine.

## 5. Sprint Workstreams

### Workstream A — Context Contract Stabilization

Confirm and enforce the Context Layer contract:

- `context_strength`
- `context_reason`
- `leadership_state`
- `rs_percentile`
- `rs_rank`

These fields are classification outputs only. They must not imply tradeability, actionability, capital allocation, execution readiness, or final actions.

### Workstream B — Forbidden Field Enforcement

The Context Layer must not emit or derive these forbidden Context fields:

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

Tests and governance checks should enforce absence from active Context runtime outputs. Test references are acceptable only when used as explicit forbidden-field assertions.

### Workstream C — Historical Artifact Hygiene

Handle stale historical artifacts without broad runtime redesign:

- `context_strength_historical.csv`
- stale `context_tradeable` fields
- regeneration or quarantine policy

Historical artifacts that still contain legacy fields must be treated as schema hygiene risks. The Technical Lead should decide whether Sprint 2 developer execution regenerates them, quarantines them, or documents them as non-runtime legacy artifacts.

### Workstream D — Distribution Preservation

Context must preserve opportunity distribution:

- no row suppression
- weak context remains classification-only
- strong/leading context remains classification-only
- no hidden filtering

No Context state may destroy opportunities or act as an allocation gate.

### Workstream E — Sector-Relative Enrichment Governance

Sector-relative data must remain enrichment only:

- sector-relative data remains enrichment only
- missing sector data remains non-blocking
- nullable `rs_vs_sector` handling

Missing sector data must not prevent Context output generation and must not alter allocation semantics.

### Workstream F — Test & CI Enforcement

Harden governance coverage with:

- forbidden-field assertions
- deterministic output validation
- schema assertions
- observability checks

CI/governance enforcement should focus on active Context code and tests, excluding generated cache files such as `__pycache__`.

## 6. Execution Sequence

1. Review Sprint 0, Sprint 1, and Sprint 2 governance documentation.
2. Inspect active Context runtime implementation before making changes.
3. Inspect Context tests and current Context output schemas.
4. Confirm whether runtime code already satisfies Sprint 2 governance.
5. If runtime is clean, avoid runtime edits and harden tests/docs only.
6. If a concrete governance gap is proven, make the smallest possible patch in approved Context scope.
7. Add or strengthen forbidden-field and schema tests.
8. Validate distribution preservation and non-blocking sector-relative behavior.
9. Address historical artifact hygiene according to Technical Lead direction.
10. Run required tests and governance checks.
11. Submit for Technical Lead governance review.

## 7. Developer Handoff Requirements

The developer handoff must be a readiness package, not a broad implementation prompt.

The Technical Lead developer spec must instruct developers to:

- inspect first
- change runtime only if a proven governance gap exists
- preserve deterministic outputs
- preserve opportunity distribution
- avoid Context Layer reconstruction
- avoid new strategy logic, filters, allocation semantics, or threshold optimization
- keep weak/strong/leading semantics classification-only

## 8. Required Files To Inspect

- `AGENTS.md`
- `README.md`
- `docs/sprints/sprint_0_governance_status.md`
- `docs/sprints/sprint_1_closeout.md`
- `docs/sprints/sprint_2_cross_sectional_leadership.md`
- `docs/sprints/execution_roadmap_v2.md`
- `docs/audits/sprint_0_final_governance_audit.md`
- `docs/audits/documentation_alignment_audit_post_sprint_0.md`
- `docs/technical/Technical_Analysis_v3.md`
- `docs/functional/Functional_Analysis_v2.md`
- `docs/execution/execution_delivery_framework_v2.md`
- `docs/technical/decision_engine_design_v2.md`
- `scripts/core/build_context_layer.py`
- `scripts/core/build_context_backfill.py`
- `tests/core/test_build_context_layer.py`
- `tests/core/test_build_context_backfill.py`
- `data/processed/context_strength.csv`
- `data/processed/context_strength_historical.csv`

## 9. Required Files That May Be Changed

Sprint 2 may change only the following files if required:

- `scripts/core/build_context_layer.py`
- `scripts/core/build_context_backfill.py`
- context-related tests under `tests/core/`
- Context schema/governance CI configuration, if present
- `docs/sprints/sprint_2_cross_sectional_leadership.md`
- `docs/sprints/sprint_2_execution_plan.md`
- future Sprint 2 Technical Lead/developer documentation
- generated Context artifacts only if the approved Sprint 2 developer spec explicitly requires regeneration or quarantine

Runtime changes are allowed only after a concrete governance gap is proven.

## 10. Files Explicitly Out Of Scope

Sprint 2 must not modify:

- scanner logic
- validation logic
- watchlist logic
- portfolio logic
- Decision Engine allocation logic
- reporting behavior
- Telegram behavior
- strategy scoring
- trading thresholds
- BUY/SELL/HOLD/TRIM/REMOVE behavior
- allocation logic
- scanner, validation, watchlist, portfolio, reporting, or Decision Engine schemas unless explicitly required by a narrow compatibility finding

## 11. Required Tests

Sprint 2 developer execution must include:

- focused Context Layer tests
- focused Context backfill tests, if historical artifact handling is in scope
- full test suite after changes
- schema assertions for active Context output
- forbidden-field absence assertions
- deterministic output validation
- row-preservation assertions
- sector-missing-data non-blocking assertions

Recommended commands:

```bash
.venv/bin/python3 -m pytest tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
.venv/bin/python3 -m pytest
```

If runtime code or generated Context artifacts are changed, run the full pipeline according to repository norms.

## 12. Required Governance/Grep Checks

Run and interpret:

```bash
grep -R "context_tradeable" scripts/core/build_context_layer.py tests/core/
grep -R "tradeability" scripts/core/build_context_layer.py tests/core/
grep -R "allocation_priority" scripts/core/build_context_layer.py tests/core/
grep -R "final_action" scripts/core/build_context_layer.py tests/core/
grep -R "conviction" scripts/core/build_context_layer.py tests/core/
grep -R "urgency" scripts/core/build_context_layer.py tests/core/
grep -R "BUY" scripts/core/build_context_layer.py tests/core/
grep -R "SELL" scripts/core/build_context_layer.py tests/core/
grep -R "REMOVE" scripts/core/build_context_layer.py tests/core/
```

Expected interpretation:

- Active Context runtime must not contain forbidden semantics.
- Tests may contain forbidden terms only as explicit absence assertions.
- Generated cache directories such as `__pycache__` are not governance evidence.
- Documentation references are acceptable only when they clearly describe governance boundaries or historical risks.

## 13. Historical Artifact Handling

The active Context runtime schema is governance-clean. The known artifact risk is stale historical Context output that may still contain legacy fields such as:

- `context_tradeable`
- `context_tradeable_reason`

Sprint 2 must decide one approved handling path:

- regenerate historical Context artifacts using governance-clean backfill code
- quarantine stale artifacts as pre-governance legacy output
- document the artifact as non-runtime legacy data and schedule cleanup separately

This decision must not rewrite Context strategy logic or introduce new classification thresholds.

## 14. Distribution-Preservation Enforcement

Sprint 2 must preserve Context row distribution:

- all eligible upstream rows should remain represented in Context output
- weak context must not be dropped
- missing sector data must not suppress rows
- neutral/weak leadership must not imply rejection
- strong/leading leadership must not imply tradeability
- tests should assert row counts where fixtures make that meaningful

Distribution observability may be added as logging or test coverage, but not as a filter.

## 15. Sector-Relative Data Handling Policy

Sector-relative values may enrich Context classification and observability.

Sector-relative handling rules:

- `rs_vs_sector` may be nullable
- missing sector data is non-blocking
- sector data must not become an allocation gate
- sector data must not determine final action
- sector data must not create tradeability, urgency, or conviction semantics
- sector enrichment must preserve deterministic output behavior

## 16. Acceptance Criteria

Sprint 2 passes only when:

- Context remains classification-only
- no active forbidden Context fields exist
- weak/strong/leading states remain non-allocative
- sector data remains enrichment-only
- no hidden filtering is introduced
- no row suppression is introduced
- tests pass
- governance checks pass
- Technical Lead review passes

## 17. Definition Of Done

Sprint 2 is done only when:

- Context runtime and tests are inspected
- schema expectations are documented and tested
- forbidden-field checks are in place
- historical artifact handling is resolved or explicitly documented as a non-blocking risk
- distribution preservation is tested or otherwise verified
- sector-relative enrichment remains non-blocking
- no out-of-scope files are modified
- full test suite passes
- governance grep checks are clean or clearly interpreted
- Technical Lead signs off
- Sprint 2 status documentation is updated

## 18. Risks And Controls

| Risk | Severity | Control |
|---|---|---|
| Stale historical CSV contains `context_tradeable` fields | MEDIUM | Regenerate, quarantine, or document as legacy artifact before certification |
| Tests only partially enforce forbidden fields | MEDIUM | Add full forbidden-field assertions for active Context outputs |
| Strong/leading language could be mistaken for tradeability | MEDIUM | Require tests/docs to state these are leadership classifications only |
| Sector-relative enrichment becomes a blocking dependency | HIGH | Assert missing sector data remains non-blocking |
| Developer over-reconstructs Context Layer | HIGH | Inspect-first and minimal-change rules in Technical Lead spec |
| Grep checks include `__pycache__` noise | LOW | Exclude generated cache files from governance interpretation |

## 19. Review Checkpoints

1. Scrum Master execution planning review
2. Technical Lead developer specification review
3. Developer inspection-first execution checkpoint
4. Focused Context test review
5. Historical artifact handling review
6. Full governance grep review
7. Final Technical Lead certification review

## 20. Final Scrum Master Recommendation

READY FOR TECHNICAL LEAD DEVELOPER SPEC
