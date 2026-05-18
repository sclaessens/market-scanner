# Sprint 6 Execution Plan - Decision Engine Core

## 1. Title and Sprint Status

Sprint 6 Execution Plan - Decision Engine Core.

Status: EXECUTION PLANNING COMPLETE / EXECUTION REVIEW NEXT.

This is a Scrum execution-planning artifact only. It does not authorize implementation, developer specification, runtime refactoring, strategy optimization, portfolio optimization, tests, generated output changes, or architecture redesign.

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
- `docs/sprints/sprint_6_decision_engine_core.md`
- `docs/sprints/sprint_5_closeout.md`
- `docs/technical/Technical_Analysis_v3.md`
- `docs/functional/Functional_Analysis_v2.md`
- `docs/execution/execution_delivery_framework_v2.md`
- `docs/technical/decision_engine_design_v2.md`

Existing Sprint 6 execution plans reviewed:

- No prior Sprint 6 execution plan existed at planning start.

## 3. Executive Execution-Planning Conclusion

Sprint 6 is ready for Technical Lead execution review.

The execution plan narrows Sprint 6 to a minimal deterministic Decision Engine Core. It preserves the certified doctrine that all allocation, execution, arbitration, prioritization, final-action, and capital-coordination authority belongs only inside the Decision Engine.

The plan does not treat `docs/sprints/sprint_6_decision_engine_core.md` as implementation scope. That document remains historical architectural context and non-certified future-state ambition. All older probabilistic, conviction, ranking, scoring, execution-aggressiveness, queueing, persistence, and smoothing concepts are narrowed, deferred, or placed behind explicit developer-specification controls.

## 4. Certified Audit Inheritance

Sprint 6 execution planning inherits the Sprint 6 Governance Audit verdict:

```text
SPRINT 6 PREPARATION CERTIFIED FOR EXECUTION PLANNING
```

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

## 5. Sprint 6 Execution Objective

Sprint 6 execution objective:

Define the future implementation path for a governance-safe Decision Engine Core that consumes certified upstream classifications and emits deterministic, explainable, audit-traceable downstream decisions without contaminating upstream layers or downstream reporting.

The future Decision Engine Core may own:

- final action determination
- allocation decision determination
- execution decision determination
- opportunity arbitration
- conflict resolution
- portfolio coordination
- decision rationale emission
- authority-leakage prevention

## 6. Governance-Safe Scope

Sprint 6 developer specification may be prepared for:

- reviewing existing `scripts/core/decision_engine.py`
- defining an authoritative Decision Engine input contract
- reconciling current output naming with `data/processed/final_decisions.csv`
- defining deterministic final-action decision logic
- defining deterministic allocation and execution decision states
- defining deterministic arbitration and conflict-resolution rules
- defining rationale and provenance logging
- defining no-hidden-filtering and no-silent-suppression checks
- defining authority-leakage checks outside Decision Engine
- defining focused tests for deterministic, explainable, Decision Engine-only authority

Any future implementation must remain inside Decision Engine-owned files or files explicitly certified by the developer specification.

## 7. Explicit Non-Scope

Out of scope for Sprint 6 execution planning and future Sprint 6 developer specification unless separately certified:

- upstream runtime changes
- upstream schema changes
- upstream generated artifact mutation
- reporting logic changes
- watchlist authority changes
- portfolio runtime authority changes
- strategy optimization
- portfolio optimization
- machine learning or adaptive systems
- hidden optimization
- hidden persistence state
- probabilistic smoothing
- decision persistence
- undocumented scoring
- undocumented ranking
- allocation queues that suppress visibility
- execution urgency leakage
- changes to certified Sprint 0 through Sprint 5 doctrine

## 8. Execution Sequencing

Sprint 6 must proceed in this order:

1. Execution plan creation.
2. Technical Lead execution review.
3. Developer specification.
4. Implementation.
5. Implementation audit.
6. Closeout.

No developer specification may be created until execution review certifies this plan. No implementation may begin until a certified developer specification exists.

## 9. Phase 1: Artifact and Schema Reconciliation

Goal: resolve artifact naming and input contracts before any runtime work.

Required future developer-specification decisions:

- confirm the authoritative Decision Engine output artifact
- reconcile older `decision_output.csv` language with current `data/processed/final_decisions.csv`
- confirm whether Sprint 6 updates the existing output or writes a new certified output
- define the complete output schema before implementation
- define required input artifacts and row identity
- reconcile older `portfolio_state.csv` references with certified Sprint 5 `data/processed/portfolio_intelligence.csv` or a separately certified Portfolio artifact
- identify whether watchlist and portfolio runtime artifacts are required inputs
- define fail-fast behavior for missing required inputs
- define non-mutating behavior for all upstream inputs

No implementation may begin until these naming and artifact decisions are explicit in the developer specification.

## 10. Phase 2: Minimal Deterministic Decision Engine Core

Goal: establish a minimal deterministic Decision Engine Core before advanced allocation ambition.

Allowed future scope:

- one deterministic Decision Engine evaluation per preserved opportunity row
- one deterministic `final_action` per evaluated ticker/date row
- deterministic `allocation_decision` state
- deterministic `execution_decision` state
- deterministic fallback or no-action state with rationale
- explicit rule order
- explicit tie-breaking
- explicit source provenance
- explicit decision rationale

Forbidden in Phase 2:

- adaptive evaluation
- hidden optimization
- probabilistic randomness
- opaque scoring
- silent row removal
- upstream mutation
- downstream reporting reinterpretation

## 11. Phase 3: Arbitration and Conflict-Resolution Logic

Goal: define Decision Engine-owned conflict resolution without hidden filtering.

Allowed future scope:

- opportunity arbitration inside Decision Engine only
- portfolio conflict resolution inside Decision Engine only
- deterministic precedence rules
- deterministic tie-breaking rules
- explicit `arbitration_reason`
- explicit `conflict_resolution_reason`
- visible no-allocation or no-execution decisions

Required control:

- if an opportunity is not allocated or not executed, it must still remain visible in the Decision Engine output with a rationale.

## 12. Phase 4: Explainability and Rationale Outputs

Goal: make every downstream decision auditable.

Future developer specification must define rationale outputs such as:

- `allocation_rationale`
- `execution_rationale`
- `arbitration_reason`
- `conflict_resolution_reason`
- source provenance fields
- decision rule version or decision contract version

Rationales must describe the Decision Engine's downstream decision. They must not rewrite upstream classifications or imply that upstream layers made allocation, execution, conviction, tradeability, scoring, or ranking decisions.

## 13. Phase 5: Leakage and No-Hidden-Filtering Controls

Goal: prove that authority stays inside the Decision Engine and opportunities remain visible.

Future developer specification must require:

- forbidden semantic scans outside Decision Engine-owned files
- upstream artifact non-mutation checks
- output row visibility checks
- ticker/date universe checks
- deterministic ordering checks
- no silent suppression checks
- no Reporting reinterpretation checks
- no Watchlist allocation-authority checks
- no Portfolio allocation-authority checks
- no Portfolio Intelligence preference-authority checks

## 14. Phase 6: Validation and Audit Evidence

Goal: prepare the implementation audit evidence before implementation begins.

Future implementation must produce evidence for:

- schema compliance
- deterministic output behavior
- rationale completeness
- source provenance completeness
- one decision per evaluated ticker/date
- no upstream mutation
- no hidden filtering
- no silent opportunity suppression
- no authority leakage outside Decision Engine
- no adaptive or ML behavior
- no hidden persistence state
- no undocumented ranking/scoring behavior

## 15. Concept Narrowing Table

| Concept | Execution-Plan Decision | Status | Required Control | Developer-Spec Implication |
|---|---|---|---|---|
| Probabilistic evaluation | Narrow to deterministic rule-based decision interpretation. | Narrowed | No randomness, no adaptive optimization, explicit rule order, reproducible outputs. | Developer spec may use descriptive probability language only if translated into deterministic, auditable rules. |
| Conviction scoring | Narrow to Decision Engine-only deterministic conviction state or documented score only if explicitly justified. | Narrowed | Schema-defined, formula-defined, rationale-backed, leakage-tested. | Developer spec must decide whether conviction is included; if included, it must be deterministic and Decision Engine-only. |
| Opportunity ranking | Narrow to deterministic downstream arbitration order only. | Narrowed | Preserve evaluated opportunity visibility; no upstream reordering; explicit tie-breaks. | Developer spec must avoid "best opportunity" semantics outside Decision Engine output. |
| Allocation priority | Allow only as downstream Decision Engine-owned allocation state if schema-defined. | Allowed with controls | No upstream leakage, no Reporting reinterpretation, explicit rationale. | Developer spec may include allocation priority only inside Decision Engine output/logs. |
| Tradeability scoring | Narrow to deterministic Decision Engine-only tradeability/allocation-readiness state if included. | Narrowed | No upstream tradeability; no hidden filtering; no opaque score. | Developer spec must choose safe naming and rationale fields. |
| Portfolio balancing | Narrow to downstream portfolio conflict-resolution evidence. | Narrowed | No portfolio override outside Decision Engine; no source mutation. | Developer spec may consume Portfolio Intelligence but must not modify it. |
| Execution aggressiveness | Narrow to deterministic execution decision style only if necessary. | Narrowed | Must not become urgency leakage; must include execution rationale. | Developer spec must define exact allowed values or defer. |
| Decision persistence | Defer to Sprint 7 by default. | Deferred | No hidden state in Sprint 6. | Developer spec must exclude unless separately certified. |
| Probabilistic smoothing | Defer to Sprint 7 by default. | Deferred | No smoothing state or adaptive behavior in Sprint 6. | Developer spec must exclude unless separately certified. |
| Escalation tracking | Narrow to non-persistent rationale only or defer. | Narrowed / deferred | Must not create hidden urgency or persistence. | Developer spec must either omit or define deterministic non-persistent fields. |
| Allocation queues | Narrow to visible deterministic sequencing only, or omit. | Narrowed | Must not suppress opportunities; must preserve output visibility. | Developer spec must avoid queue mechanics unless fully auditable. |
| Decision distributions | Allow as audit/observability of emitted outputs only. | Allowed with controls | Must not become scoring validation or optimization target. | Developer spec may define log distribution summaries. |

## 16. Deferred Concepts Table

| Concept | Execution-Plan Decision | Rationale | Future Handling |
|---|---|---|---|
| Decision persistence | Deferred to Sprint 7 by default. | Existing roadmap places persistence and stability in Sprint 7; Sprint 6 should avoid hidden state. | Include in Sprint 7 preparation or backlog only if later governance requires detail. |
| Probabilistic smoothing | Deferred to Sprint 7 by default. | Smoothing implies persistence/stability behavior and can hide adaptive state. | Address under Sprint 7 Stability & Persistence governance. |
| Hidden state | Forbidden in Sprint 6. | Hidden state would break deterministic replay and auditability. | Any future state must be explicit, versioned, and audited. |
| Adaptive behavior | Forbidden in Sprint 6. | Adaptive behavior could create hidden optimization or ML authority. | Requires separate architecture review if ever proposed. |
| Advanced allocation queues | Deferred unless developer spec proves visibility-preserving deterministic sequencing is required. | Queueing can become silent prioritization or suppression. | Revisit only with explicit audit controls. |
| Unresolved portfolio balancing | Narrowed to Decision Engine conflict-resolution evidence; advanced balancing deferred. | Balancing can become hidden portfolio management. | Developer spec may use certified portfolio metadata only as evidence, not as override authority. |

No new backlog item is required for these deferrals because persistence and smoothing are already represented by Sprint 7 roadmap scope, while other items remain governed Sprint 6 narrowing decisions.

## 17. Developer Specification Readiness Gates

Developer specification may begin only after Technical Lead execution review confirms:

- execution plan scope is narrow enough
- artifact naming is ready to be resolved in the developer spec
- input artifact reconciliation is ready to be resolved in the developer spec
- persistence and smoothing are excluded unless separately certified
- no upstream layer changes are required
- no runtime authority leakage is permitted
- deterministic decision controls are explicit
- explainability and auditability controls are explicit
- future tests and semantic checks are sufficiently defined

## 18. Required Future Tests and Validation Checks

Future developer specification must require tests for:

- output schema contract
- required input schema contract
- one decision per ticker/date row
- deterministic output under identical inputs
- deterministic tie-breaking
- no upstream artifact mutation
- no hidden filtering
- no silent opportunity suppression
- ticker/date visibility preservation
- source provenance completeness
- decision rationale completeness
- forbidden semantics outside Decision Engine
- no Reporting authority leakage
- no Watchlist authority leakage
- no Portfolio authority leakage
- no Portfolio Intelligence authority leakage
- missing input fail-fast behavior
- missing optional metadata behavior

## 19. Required Future Grep and Semantic Checks

Future implementation validation must include targeted scans proving forbidden Decision Engine authority does not leak outside Decision Engine-owned files.

Minimum future scan direction:

```bash
grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "tradeable" scripts/ | grep -v decision_engine.py
grep -R "conviction" scripts/ | grep -v decision_engine.py
grep -R "allocation_priority" scripts/ | grep -v decision_engine.py
```

Future developer specification may refine these commands to avoid false positives, but must preserve the audit intent.

Semantic checks must also cover:

- `urgency`
- `execution_ready`
- `actionable`
- `ranking_score`
- `portfolio_score`
- `final_score`
- `recommended`
- `preferred`
- `suitability`
- `attractiveness`
- `optimal`

Allowed occurrences outside Decision Engine must be limited to governance docs, tests asserting absence, or explicit historical-context references.

## 20. Required Future Output and Log Schema Expectations

Future output schema must be defined before implementation.

Candidate Decision Engine output concepts:

- `ticker`
- `date`
- `final_action`
- `allocation_decision`
- `execution_decision`
- `portfolio_decision_state`
- `opportunity_decision_state`
- `allocation_rationale`
- `execution_rationale`
- `arbitration_reason`
- `conflict_resolution_reason`
- source provenance fields
- decision contract version

Future log schema should include:

- run identifier
- run timestamp
- input artifact paths
- input row count
- output row count
- evaluated row identity
- decision rule version
- source provenance
- decision rationale
- conflict-resolution rationale
- no-hidden-filtering evidence
- no-upstream-mutation evidence
- authority-leakage scan evidence

Final schema authority belongs to the future developer specification.

## 21. Determinism Controls

Future developer specification must require:

- stable input ordering
- stable joins
- explicit row-key contract
- explicit tie-breaking
- explicit rule precedence
- no random authority-bearing behavior
- no live external state without provenance
- no hidden persistence state
- no adaptive behavior
- reproducible outputs under identical inputs

## 22. Explainability Controls

Future developer specification must require:

- every `final_action` has rationale
- every allocation decision has rationale
- every execution decision has rationale
- every conflict-resolution outcome has rationale
- every no-allocation or no-execution outcome remains visible with rationale
- rationales reference source evidence without rewriting upstream classifications

## 23. Auditability Controls

Future developer specification must require:

- source provenance for every decision row
- documented schema contracts
- documented decision contract version
- deterministic validation commands
- semantic scan evidence
- preservation evidence
- non-mutation evidence
- rationale completeness evidence
- implementation audit handoff notes

## 24. Cross-Layer Boundary Controls

Validation, Context, Fundamental, Timing State, and Portfolio Intelligence remain upstream evidence sources only.

Watchlist and Portfolio must not become allocation authorities.

Reporting must communicate Decision Engine outputs only and must not reinterpret, reprioritize, or override them.

The Decision Engine must not write to upstream artifacts or require upstream builders to run through hidden coupling.

## 25. Risks and Controls

Risk: older Sprint 6 ambition re-enters as automatic scope.

Control: developer specification must cite this execution plan and the governance audit, and must treat `sprint_6_decision_engine_core.md` as context only.

Risk: scoring/ranking becomes opaque prioritization.

Control: require deterministic formulas or decision tables, explicit tie-breaking, and rationale logs.

Risk: no-action outcomes become silent filtering.

Control: require output visibility and rationale for every evaluated opportunity.

Risk: portfolio balancing becomes hidden portfolio management.

Control: constrain balancing to Decision Engine-owned conflict resolution and forbid portfolio source mutation.

Risk: execution aggressiveness becomes urgency leakage.

Control: require exact value definitions, deterministic rationale, and reporting boundary checks.

Risk: artifact naming ambiguity causes implementation drift.

Control: developer specification must reconcile `decision_output.csv` vs `data/processed/final_decisions.csv` and `portfolio_state.csv` vs certified Portfolio Intelligence artifacts before implementation.

## 26. Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Rationale: persistence and smoothing are already covered by Sprint 7 roadmap scope. Other narrowed concepts remain part of Sprint 6 execution review and developer-specification control rather than deferred backlog items.

## 27. Scrum Master Execution Recommendation

READY FOR SPRINT 6 EXECUTION REVIEW
