# Sprint 2 Technical Lead Developer Specification — Cross-Sectional Leadership Layer

## 1. Sprint Status

Status: READY FOR DEVELOPER EXECUTION

Sprint 2 has completed Scrum execution planning and is ready for a scoped, inspection-first developer execution pass.

This specification does not authorize broad runtime reconstruction, strategy redesign, allocation changes, or threshold optimization.

## 2. Governance Inheritance

Sprint 2 inherits the certified Sprint 0 doctrine:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority
- upstream layers classify only
- reporting communicates only
- no upstream tradeability
- no hidden filtering
- no hidden allocation semantics outside Decision Engine

Sprint 2 also inherits Sprint 1 certification:

- Validation Layer contract is already stabilized
- `structure_state` and `structure_reason` are authoritative
- `valid_setup` and `validation_reason` are compatibility-only
- upstream layers may not reinterpret validation or context output as allocation eligibility

Context Layer responsibility is leadership and relative-strength classification only.

## 3. Technical Objective

Sprint 2 must harden the Cross-Sectional Leadership Layer governance contract without changing allocation behavior.

The objective is to confirm and enforce that Context:

- classifies leadership and relative strength
- preserves opportunity distribution
- treats sector-relative data as enrichment only
- emits no tradeability, allocation, conviction, urgency, actionability, execution-readiness, or final-action fields
- keeps weak, neutral, strong, and leading states classification-only

The current governance audit found active Context runtime clean. The expected developer work is test hardening, schema enforcement, and historical artifact handling unless inspection proves a runtime defect.

## 4. Technical Lead Implementation Philosophy

The developer must:

1. FIRST inspect whether Context runtime already satisfies Sprint 2 governance.
2. ONLY modify runtime code if governance gaps are proven.
3. Prefer test hardening and artifact hygiene over runtime rewrites.
4. Avoid Context Layer reconstruction.
5. Preserve deterministic outputs.
6. Preserve opportunity distribution.
7. Preserve weak/strong/leading context as classification-only.
8. Preserve sector-relative data as enrichment-only.
9. Preserve Decision Engine as sole allocation authority.

No implementation step may add strategy logic, filters, allocation semantics, tradeability semantics, execution semantics, or threshold optimization.

## 5. Runtime Inspection Requirements

Before editing any file, inspect:

- `scripts/core/build_context_layer.py`
- `scripts/core/build_context_backfill.py`
- `tests/core/test_build_context_layer.py`
- `tests/core/test_build_context_backfill.py`
- `data/processed/context_strength.csv`
- `data/processed/context_strength_historical.csv`

Inspection must determine:

- whether active runtime output already matches the required Context contract
- whether tests fully enforce forbidden-field absence
- whether tests enforce row preservation
- whether tests enforce weak/strong/leading classification-only semantics
- whether tests enforce sector-relative missingness as non-blocking
- whether the historical Context artifact contains stale legacy schema fields

Runtime changes are not permitted unless this inspection proves an active governance gap.

## 6. Allowed Implementation Scope

Sprint 2 developer execution may address:

- Context schema exactness
- forbidden-field test enforcement
- row-preservation tests
- weak/neutral/strong/leading classification-only tests
- sector-relative non-blocking tests
- deterministic output tests
- historical Context artifact handling
- Context governance documentation updates
- CI/governance checks if directly required

The expected path is minimal and test-first. If active runtime is clean, strengthen tests and document artifact handling without changing runtime code.

## 7. Explicitly Forbidden Implementation Scope

Do NOT:

- rewrite the Context Layer
- redesign architecture
- add trading logic
- add allocation logic
- add tradeability semantics
- add execution-readiness semantics
- add hidden filters
- optimize strategy thresholds
- change Decision Engine behavior
- change reporting behavior
- change scanner, validation, watchlist, portfolio, or Telegram behavior
- change BUY/SELL/HOLD/TRIM/REMOVE behavior
- reinterpret weak, neutral, strong, or leading leadership states as capital readiness

## 8. Exact Files Allowed To Change

Only these files may be changed, and only if required:

- `scripts/core/build_context_layer.py`
- `scripts/core/build_context_backfill.py`
- `tests/core/test_build_context_layer.py`
- `tests/core/test_build_context_backfill.py`
- `docs/sprints/sprint_2_cross_sectional_leadership.md`
- `docs/sprints/sprint_2_execution_plan.md`
- `docs/sprints/sprint_2_developer_spec.md`
- CI/governance configuration files, if directly required
- `data/processed/context_strength_historical.csv`, only if regeneration or quarantine is explicitly chosen and justified

Generated Context artifact changes must not be blindly committed. They require a documented artifact-handling decision.

## 9. Exact Files Forbidden To Change

Do NOT change:

- `scripts/core/scanner.py`
- `scripts/core/build_validation_layer.py`
- `scripts/watchlist/`
- `scripts/portfolio/`
- `scripts/core/decision_engine.py`
- `scripts/reporting/`
- `scripts/telegram/`
- scanner logic
- validation logic
- watchlist logic
- portfolio logic
- Decision Engine logic
- reporting logic
- Telegram logic
- strategy scoring
- trading thresholds
- allocation logic
- BUY/SELL/HOLD/TRIM/REMOVE behavior
- context output semantics beyond governance alignment

If an unexpected compatibility issue appears outside the allowed files, stop and request Technical Lead review.

## 10. Required Context Runtime Contract

The active Context Layer runtime contract is:

- `ticker`
- `date`
- `rs_score`
- `rs_percentile`
- `rs_rank`
- `rs_vs_market`
- `rs_vs_sector`
- `context_strength`
- `context_reason`
- `leadership_state`

Context Layer may classify:

- leadership
- relative strength
- cross-sectional strength
- sector-relative leadership
- market-relative leadership

Context Layer may NOT determine:

- tradeability
- allocation eligibility
- conviction
- urgency
- actionability
- execution readiness
- final action
- BUY/SELL/HOLD/TRIM/REMOVE behavior

## 11. Required Schema Expectations

`data/processed/context_strength.csv` must remain governance-clean and must not contain forbidden Context fields.

Expected active Context schema:

| Field | Governance Meaning |
|---|---|
| `ticker` | identifier |
| `date` | observation date |
| `rs_score` | relative-strength score metadata |
| `rs_percentile` | cross-sectional distribution percentile |
| `rs_rank` | cross-sectional rank |
| `rs_vs_market` | market-relative strength metadata |
| `rs_vs_sector` | sector-relative strength metadata, nullable |
| `context_strength` | leadership classification |
| `context_reason` | classification reason |
| `leadership_state` | leadership-state classification |

No schema change is required unless inspection proves the active runtime has drifted from this contract.

## 12. Historical Artifact Handling Rules

`data/processed/context_strength_historical.csv` currently may contain stale legacy fields:

- `context_tradeable`
- `context_tradeable_reason`

The developer must choose the least invasive governance-safe option and document it:

1. Regenerate the historical artifact using governance-clean backfill code.
2. Quarantine or ignore it as a pre-governance legacy artifact.
3. Document it as a non-runtime legacy risk and schedule cleanup separately.

Sprint 2 developer execution decision:

- selected option: 3
- rationale: active Context runtime and backfill source are governance-clean; the stale historical CSV is a generated legacy artifact, not active runtime logic
- action: do not regenerate or commit generated historical data in this sprint; carry the artifact as a non-runtime legacy risk for Technical Lead review and later cleanup scheduling

The chosen option must not:

- rewrite Context strategy logic
- introduce thresholds
- change allocation behavior
- hide runtime governance leakage
- blindly commit generated data

If regeneration is chosen, run the required validation commands and inspect the regenerated header before including the artifact.

## 13. Distribution Preservation Rules

Context may not:

- drop rows because leadership is weak
- suppress rows because sector data is missing
- filter rows because market-relative strength is poor
- cap output to strong/leading names
- create hidden allocation filters

Context must preserve the input opportunity distribution available to it, subject only to existing structural input validation such as required columns and duplicate key checks.

Tests should assert row preservation where fixtures make the expected row count deterministic.

## 14. Sector-Relative Data Rules

Sector-relative data rules:

- `rs_vs_sector` may be nullable
- missing sector data is non-blocking
- sector-relative data is enrichment only
- sector data may not become allocation logic
- sector data may not become tradeability logic
- sector data may not become urgency logic
- sector data may not become conviction logic
- missing sector data must not suppress Context rows

Existing sector-relative behavior should be preserved unless a governance gap is proven.

## 15. Forbidden Field Rules

Forbidden Context output fields:

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

Mandatory semantic rules:

- WEAK context = weak leadership classification only
- NEUTRAL context = neutral leadership classification only
- STRONG context = strong leadership classification only
- LEADING context = leading leadership classification only
- none of these imply tradeability, actionability, allocation priority, urgency, or execution readiness

Tests may reference forbidden terms only as absence assertions.

## 16. Required Implementation Sequence

1. Read Sprint 2 planning documents and governance docs.
2. Inspect Context runtime and backfill source.
3. Inspect Context runtime and historical output headers.
4. Inspect Context tests.
5. Decide whether runtime changes are required.
6. If runtime is governance-clean, do not modify runtime source.
7. Add or strengthen tests for schema exactness, forbidden-field absence, row preservation, sector missingness, deterministic output, and classification-only semantics.
8. Decide and document historical artifact handling.
9. If artifact regeneration is chosen, regenerate only the approved artifact and inspect the schema.
10. Run focused Context tests.
11. Run full test suite.
12. Run governance grep checks.
13. Run full pipeline only if runtime source or generated Context artifacts changed.
14. Report changes, validation results, artifact status, and residual risks.

## 17. Required Tests

Required Sprint 2 test coverage:

- context schema exactness
- forbidden-field absence
- row preservation
- weak context classification-only
- neutral context classification-only
- strong/leading context classification-only
- sector missingness non-blocking
- deterministic output
- `context_strength_historical` artifact handling, if in scope

Required commands:

```bash
.venv/bin/python3 -m pytest tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
.venv/bin/python3 -m pytest
```

## 18. Required Governance/Grep Checks

Use active Context source and tests, avoiding `__pycache__` noise:

```bash
grep -R "context_tradeable" scripts/core/build_context_layer.py scripts/core/build_context_backfill.py tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
grep -R "tradeability" scripts/core/build_context_layer.py scripts/core/build_context_backfill.py tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
grep -R "allocation_priority" scripts/core/build_context_layer.py scripts/core/build_context_backfill.py tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
grep -R "final_action" scripts/core/build_context_layer.py scripts/core/build_context_backfill.py tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
grep -R "conviction" scripts/core/build_context_layer.py scripts/core/build_context_backfill.py tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
grep -R "urgency" scripts/core/build_context_layer.py scripts/core/build_context_backfill.py tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
grep -R "BUY" scripts/core/build_context_layer.py scripts/core/build_context_backfill.py tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
grep -R "SELL" scripts/core/build_context_layer.py scripts/core/build_context_backfill.py tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
grep -R "REMOVE" scripts/core/build_context_layer.py scripts/core/build_context_backfill.py tests/core/test_build_context_layer.py tests/core/test_build_context_backfill.py
```

Expected interpretation:

- active Context source must not emit forbidden fields
- tests may reference forbidden terms only as absence assertions
- generated cache files must be ignored
- historical artifact references must be classified as legacy/artifact hygiene, not active runtime leakage

## 19. Runtime Validation Procedure

If runtime source is unchanged:

- run focused Context tests
- run full test suite
- run grep checks
- inspect current Context output headers
- document that full pipeline was not required because runtime source did not change

If runtime source or generated Context artifacts changed:

- run focused Context tests
- run full test suite
- run grep checks
- run the full pipeline
- inspect `data/processed/context_strength.csv` header
- inspect `data/processed/context_strength_historical.csv` header if regenerated
- document generated artifact changes explicitly

Full pipeline command:

```bash
PATH="/Users/stevenclaessens/market-scanner/.venv/bin:$PATH" python3 scripts/run_full_pipeline.py
```

## 20. Acceptance Criteria

Sprint 2 implementation passes only when:

- Context remains classification-only
- active Context output matches the required runtime contract
- active Context output contains no forbidden Context fields
- weak context remains non-allocative
- neutral context remains non-allocative
- strong/leading context remains non-allocative
- sector-relative data remains enrichment-only
- missing sector data remains non-blocking
- no hidden filtering is introduced
- no row suppression is introduced
- deterministic output is preserved
- historical artifact handling is documented or resolved
- tests pass
- governance checks pass or are clearly interpreted as absence assertions
- Technical Lead review passes

## 21. Definition Of Done

Sprint 2 is done only when:

- Context runtime was inspected before edits
- runtime code was modified only if a proven governance gap existed
- Context tests enforce schema exactness
- Context tests enforce forbidden-field absence
- Context tests enforce row preservation
- Context tests enforce sector missingness as non-blocking
- historical artifact handling is completed or documented as a non-blocking residual risk
- no forbidden out-of-scope files were modified
- full test suite passes
- required grep checks are clean or interpreted correctly
- pipeline result is documented if required
- generated artifact status is documented
- Technical Lead implementation audit approves certification

## 22. Risks And Controls

| Risk | Severity | Control |
|---|---|---|
| Historical artifact still contains `context_tradeable` fields | MEDIUM | Choose and document regeneration, quarantine, or scheduled cleanup |
| Developer rewrites Context despite clean runtime | HIGH | Inspect-first rule; runtime edits require proven governance gap |
| Tests reference forbidden terms ambiguously | MEDIUM | Permit forbidden terms only in explicit absence assertions |
| Weak context becomes rejection semantics | HIGH | Add classification-only tests |
| Strong/leading context becomes tradeability semantics | HIGH | Add classification-only tests |
| Missing sector data suppresses rows | HIGH | Add non-blocking sector-missingness tests |
| Row count changes hide opportunity distribution | HIGH | Add row-preservation tests |
| Generated artifact changes are blindly committed | MEDIUM | Require explicit artifact decision and header inspection |

## 23. Technical Lead Implementation Notes

The current active Context runtime appears governance-clean based on the Sprint 2 audit. The highest-value developer work is expected to be:

- strengthening `tests/core/test_build_context_layer.py`
- strengthening `tests/core/test_build_context_backfill.py`
- documenting or resolving stale `context_strength_historical.csv` schema risk
- preserving runtime source unless inspection proves drift

Do not convert this sprint into a Context strategy improvement sprint. Cross-sectional leadership semantics may be tested and clarified, but not optimized.

## 24. Final Technical Lead Recommendation

READY FOR DEVELOPER EXECUTION
