# Sprint 1 — Structure Classification Alignment & Governance Refinement

Trading System — Institutional Decision Engine

## 1. Sprint Status

Status: READY FOR FINAL GOVERNANCE REVIEW

Sprint 1 inherits the certified Sprint 0 architecture:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority
- upstream layers classify only
- reporting communicates only
- no upstream tradeability
- no hidden filtering
- no hidden allocation semantics outside Decision Engine

Authoritative references:

- `AGENTS.md`
- `README.md`
- `docs/sprints/sprint_0_governance_status.md`
- `docs/audits/sprint_0_final_governance_audit.md`
- `docs/audits/documentation_alignment_audit_post_sprint_0.md`
- `docs/technical/Technical_Analysis_v3.md`
- `docs/functional/Functional_Analysis_v2.md`
- `docs/execution/execution_delivery_framework_v2.md`

## 2. Executive Summary

Sprint 1 is not a full Validation Layer rebuild.

Sprint 1 is a governance refinement sprint for the Validation Layer. Its purpose is to stabilize structure-classification terminology, schemas, compatibility aliases, tests, and CI enforcement so future work cannot reintroduce filtering-first validation semantics.

Validation may classify structural coherence only.

Validation may not determine:

- tradeability
- allocation eligibility
- conviction
- urgency
- actionability
- execution readiness
- BUY/SELL/HOLD/TRIM/REMOVE behaviour

## 3. Sprint Objective

Sprint 1 aligns the Validation Layer around the governance-clean runtime contract:

- `structure_state`
- `structure_reason`

Sprint 1 must preserve deterministic output, opportunity distribution, and separation of concerns.

Sprint 1 must not add strategy logic, optimize thresholds, create filters, or alter downstream allocation behavior.

## 4. Validation Contract

### Primary Contract

The authoritative validation output contract is:

- `structure_state`
- `structure_reason`

These fields describe structural classification only.

### Deprecated Compatibility Aliases

The following fields may remain temporarily for compatibility only:

- `valid_setup`
- `validation_reason`

Compatibility rules:

- `valid_setup` means structure coherence only.
- `valid_setup` must never mean tradeability, allocation eligibility, actionability, or capital worthiness.
- `validation_reason` must mirror or map to `structure_reason`.
- New code should prefer `structure_state` and `structure_reason`.

## 5. Structure States

Allowed structure-state vocabulary:

- `COHERENT`
- `BROKEN`
- `INCOMPLETE`
- `MISSING_DATA`
- `DEGRADED_STRUCTURE` only if needed and explicitly defined as descriptive metadata

Forbidden binary vocabulary:

- `VALID`
- `INVALID`

Rationale:

Validation classifies structure. It does not make capital-allocation or execution decisions.

## 6. Descriptive Metadata Policy

The following are descriptive metadata only:

- entry quality
- extension
- volume quality
- breakout quality
- RR quality

These fields or concepts may not:

- change `structure_state`
- block the `valid_setup` compatibility alias
- eliminate opportunities
- act as hidden filters
- imply tradeability
- imply allocation readiness
- imply execution readiness

If emitted, descriptive metadata must remain separate from the primary validation contract.

## 7. Structure Reason Policy

Sprint 1 must use classification-first language.

Preferred terms:

- structure classification
- `structure_reason`
- `structure_failure_metadata`
- structure-state distribution
- broken or incomplete structures

Avoid or replace:

- technical invalidation
- invalidation metadata
- invalidation distribution
- invalid structures

Reason labels must describe structure, data quality, or classification state. They must not describe rejection, tradeability, actionability, urgency, allocation, or execution.

## 8. Sprint Scope

In scope:

- align Sprint 1 documentation with certified Sprint 0 governance
- review and stabilize `validation_layer.csv` schema expectations
- confirm `structure_state` and `structure_reason` are the primary contract
- mark `valid_setup` and `validation_reason` as deprecated compatibility aliases
- ensure tests validate classification-first behavior
- ensure CI/governance checks detect forbidden validation fields
- monitor structure-state distribution
- monitor missing-data distribution
- monitor compatibility drift
- monitor migration drift

Out of scope:

- rebuilding the full Validation Layer without a specific governance defect
- changing scanner behavior
- changing context behavior
- changing watchlist behavior
- changing portfolio behavior
- changing Decision Engine allocation behavior
- changing reporting behavior
- adding strategy logic
- adding filters
- optimizing thresholds
- creating BUY/SELL/HOLD/TRIM/REMOVE behavior
- adding allocation, conviction, urgency, actionability, or execution readiness semantics

## 9. Required Inputs

Documentation:

- `AGENTS.md`
- `README.md`
- `docs/technical/Technical_Analysis_v3.md`
- `docs/functional/Functional_Analysis_v2.md`
- `docs/execution/execution_delivery_framework_v2.md`
- `docs/audits/sprint_0_final_governance_audit.md`
- `docs/audits/documentation_alignment_audit_post_sprint_0.md`
- `docs/sprints/sprint_0_governance_status.md`

Runtime data for schema inspection:

- `data/processed/scanner_ranked.csv`
- `data/processed/validation_layer.csv`
- `data/processed/entry_quality_metrics.csv`

## 10. Required Outputs

### Runtime Schema Expectation

`validation_layer.csv` must expose structure-classification fields only.

Required primary fields:

- `ticker`
- `date`
- `structure_state`
- `structure_reason`
- `setup_type`

Deprecated compatibility aliases, if still present:

- `valid_setup`
- `validation_reason`

Forbidden validation output fields:

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

## 11. Generated Artifact Hygiene

Legacy generated CSV columns such as:

- `context_tradeable`
- `entry_quality_flag`
- `urgency`
- `entry_plan`
- `action_now`

are schema/artifact hygiene risks only.

They are not active Sprint 1 runtime scope unless explicitly added as a separate cleanup subtask. Sprint 1 may document these risks but must not expand scope implicitly.

## 12. Distribution Preservation

Sprint 1 must monitor:

- structure-state distribution
- missing-data distribution
- compatibility drift
- migration drift

Sprint 1 may not optimize these distributions through hidden filtering.

Distribution changes are observations. They are not permission to add filters, thresholds, or allocation shortcuts.

## 13. Governance Rules

Hard rules:

- Validation = structure classification only.
- `structure_state` is the authoritative validation contract.
- `structure_reason` is the authoritative classification reason.
- `valid_setup` is compatibility-only.
- `validation_reason` is compatibility-only.
- Entry quality and extension may not act as gates.
- Validation may not emit allocation fields.
- Validation may not emit action fields.
- Validation may not interpret context, watchlist, portfolio, fundamentals, or reporting outputs.

## 14. CI / Governance Enforcement

Required checks for Validation Layer source and tests:

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

Expected result:

- no active validation logic emits forbidden fields
- tests only reference forbidden terms when asserting absence
- compatibility aliases are explicitly documented as deprecated and structure-only

## 15. Acceptance Criteria

Sprint 1 passes only when:

- `structure_state` is the authoritative validation contract
- `structure_reason` is the authoritative validation reason field
- `valid_setup` is compatibility-only
- `validation_reason` is compatibility-only
- Validation emits no tradeability fields
- Validation emits no allocation fields
- Validation emits no conviction fields
- Validation emits no action fields
- entry quality is descriptive metadata only
- extension is descriptive metadata only
- RR quality is descriptive metadata only
- volume quality is descriptive metadata only
- no metadata field acts as a hidden filter
- structure-state distribution is monitored, not optimized through filtering
- missing-data distribution is monitored, not optimized through filtering
- CI checks for forbidden fields pass
- Technical Lead governance review passes
- Functional Analyst review passes

## 16. Definition of Done

Sprint 1 is complete only when:

- documentation is aligned with certified Sprint 0 doctrine
- runtime schema expectations are governance-clean
- tests enforce classification-first validation behavior
- compatibility aliases are explicitly marked and tested as non-allocation fields
- no hidden filtering is introduced
- no new strategy logic is introduced
- no threshold optimization is introduced
- no downstream allocation behavior is changed
- final governance review confirms readiness for developer execution or confirms no implementation work is required

## 17. Risks And Controls

| Risk | Control |
|---|---|
| `valid_setup` is treated as tradeability | Mark as deprecated compatibility alias for structure coherence only |
| `VALID` / `INVALID` vocabulary reintroduces binary gating | Use `COHERENT`, `BROKEN`, `INCOMPLETE`, `MISSING_DATA` |
| entry quality becomes a hidden gate | Keep entry quality separate and descriptive |
| extension becomes a hidden gate | Test extension as metadata only |
| distribution monitoring becomes optimization | Treat distributions as observability only |
| legacy generated artifacts mislead developers | Document as artifact hygiene risk outside Sprint 1 runtime scope |

## 18. Final Sprint Doctrine

Classify structure.

Preserve opportunity distribution.

Stabilize schema language.

Keep compatibility aliases temporary and explicit.

Do not allocate upstream.

Do not filter through hidden assumptions.
