# Sprint 6 Decision Engine Governance Preparation

## 1. Sprint Status

Sprint 6 status: PREPARATION COMPLETE / GOVERNANCE AUDIT NEXT.

Sprint 6 theme: Decision Engine Core.

Sprint 6 is the first sprint where formal allocation authority may exist. This preparation document does not authorize implementation, runtime changes, strategy optimization, portfolio optimization, test creation, generated-output changes, or architecture redesign.

## 2. Preparation Scope

This document prepares Sprint 6 for governance audit only.

Reviewed documents:

- `AGENTS.md`
- `README.md`
- `docs/sprints/README.md`
- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/execution_roadmap_v2.md`
- `docs/sprints/sprint_5_closeout.md`
- `docs/technical/Technical_Analysis_v3.md`
- `docs/functional/Functional_Analysis_v2.md`
- `docs/execution/execution_delivery_framework_v2.md`
- `docs/technical/decision_engine_design_v2.md`
- `docs/sprints/sprint_6_decision_engine_core.md`

`docs/sprints/sprint_6_decision_engine_core.md` is treated as historical architectural context and non-certified future-state ambition. It is not implementation scope until narrowed, audited, and certified.

## 3. Certified Doctrine Inheritance

Sprint 6 inherits Sprint 0 through Sprint 5 doctrine:

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
- enrichment layers remain standalone and non-mutating
- mandatory backlog reconciliation is required during audits and closeout

Sprint 6 also inherits the certified Sprint 5 architecture:

```text
scanner -> validation_layer -> context_layer -> fundamental_layer -> timing_state_layer -> portfolio_intelligence_layer -> watchlist -> portfolio -> decision_engine -> reporting
```

The certified upstream maturity entering Sprint 6 is:

- Validation Layer: certified
- Context Layer: certified
- Fundamental Layer: certified
- Timing State Layer: certified
- Portfolio Intelligence Layer: certified

## 4. Executive Preparation Conclusion

Sprint 6 may proceed to governance audit as an audit-first Decision Engine preparation sprint.

The governance-safe Sprint 6 objective is to define the Decision Engine Core as the only downstream authority for allocation, execution, prioritization, opportunity arbitration, portfolio conflict resolution, execution governance, capital coordination, and allocation coordination.

This preparation explicitly does not certify older Sprint 6 ambitions such as probabilistic allocation, conviction systems, ranking systems, allocation scoring systems, persistence systems, or execution aggressiveness concepts as implementation-ready. Those concepts require governance review, architectural narrowing, deterministic explainability controls, allocation-authority safeguards, and audit certification before any developer specification may be created.

## 5. Decision Engine Authority Doctrine

The Decision Engine is the only layer that may produce downstream decision authority.

Governance-safe Decision Engine authorities may include:

- final action determination
- allocation decision determination
- execution decision determination
- opportunity arbitration
- capital allocation coordination
- portfolio conflict resolution
- exposure governance
- position coordination
- allocation sequencing
- portfolio capacity coordination
- decision rationale emission

The Decision Engine may consume upstream classifications and metadata, but it must not mutate, rewrite, reinterpret, or back-propagate changes into upstream artifacts.

## 6. Downstream-Only Allocation Doctrine

Allocation authority is downstream-only.

Sprint 6 must centralize allocation interpretation inside `scripts/core/decision_engine.py` or a certified Decision Engine-owned module explicitly approved during developer specification.

No upstream layer may:

- allocate capital
- recommend allocations
- recommend execution
- rank opportunities
- score opportunities
- prioritize opportunities
- suppress opportunities
- create urgency semantics
- create conviction semantics
- create execution readiness
- create portfolio preference semantics
- create capital eligibility semantics

Any future allocation output must be emitted only by the Decision Engine and must be fully audit-traceable.

## 7. Execution Governance Doctrine

Execution authority may exist only inside the Decision Engine.

Potential downstream execution concepts may include:

- `execution_decision`
- `execution_rationale`
- `final_action`
- `allocation_decision`

Execution concepts must not leak into scanner, validation, context, fundamental, timing state, portfolio intelligence, watchlist, portfolio, or reporting layers.

Execution governance must be deterministic and explainable. Any execution decision must be reproducible from the same certified inputs and must include an explicit rationale.

## 8. Arbitration and Conflict-Resolution Doctrine

Sprint 6 may define downstream arbitration and conflict resolution, but only as Decision Engine-owned behavior.

Potential governance-safe downstream concepts include:

- `portfolio_decision_state`
- `opportunity_decision_state`
- `arbitration_reason`
- `conflict_resolution_reason`
- `allocation_rationale`
- `execution_rationale`

Arbitration must not suppress upstream opportunities silently. If the Decision Engine decides no allocation or no execution, the output must preserve the opportunity record and explain the downstream decision.

Conflict resolution must be:

- deterministic
- documented
- reproducible
- audit-traceable
- explainable
- free from hidden adaptive optimization

## 9. What the Decision Engine Is Not Allowed To Do

The Decision Engine must not:

- mutate upstream artifacts
- rewrite upstream classifications
- modify scanner, validation, context, fundamental, timing, or portfolio intelligence outputs
- create hidden adaptive optimization
- introduce hidden machine learning behavior
- use non-deterministic allocation behavior
- create hidden scoring systems without governance
- use undocumented prioritization
- emit opaque execution logic
- silently filter opportunities out of visibility
- create hidden state persistence
- leak runtime authority into reporting, watchlist, portfolio, or upstream layers
- introduce hidden strategy drift
- produce non-auditable allocation decisions
- convert reporting into decision interpretation
- convert watchlist or portfolio into allocation authorities

Sprint 6 must not use Decision Engine authority as permission to redesign upstream architecture.

## 10. Certified Upstream Contamination Controls

Sprint 6 must preserve upstream certified layers as:

- descriptive
- classification-only
- enrichment-only where applicable
- non-mutating
- distribution-preserving
- allocation-free
- execution-free

Required controls:

- no writes to upstream generated artifacts
- no imports from upstream builders that create runtime coupling
- no upstream schema changes unless separately certified
- no upstream BUY/SELL/actionable/tradeability/conviction/urgency fields
- no upstream ranking, scoring, or priority fields
- no upstream filtering or row suppression
- no upstream opportunity reordering caused by Decision Engine execution

The Decision Engine may read certified upstream outputs as input evidence only.

## 11. Cross-Layer Boundary Controls

Validation boundary:

- Validation supplies structure classification only.
- Decision Engine must not ask Validation to determine allocation eligibility.

Context boundary:

- Context supplies leadership classification only.
- Decision Engine must not push tradeability or allocation preference into Context.

Fundamental boundary:

- Fundamental supplies quality metadata only.
- Decision Engine must not mutate quality classifications or convert Fundamental into a scoring authority.

Timing State boundary:

- Timing State supplies descriptive timing-state metadata only.
- Decision Engine must not rewrite timing classifications or turn timing outputs into upstream execution readiness.

Portfolio Intelligence boundary:

- Portfolio Intelligence supplies neutral portfolio-awareness metadata only.
- Decision Engine may consume that metadata as downstream evidence.
- Portfolio Intelligence must not emit allocation preference, portfolio priority, execution readiness, suitability, attractiveness, ranking, scoring, or recommendation semantics.

Watchlist boundary:

- Watchlist may track state, but must not determine allocation, execution, tradeability, conviction, urgency, ranking, or scoring.
- Sprint 6 must not move Decision Engine authority into Watchlist state tracking.

Portfolio boundary:

- Portfolio may model exposure/risk-state, but must not determine BUY/SELL, suppress opportunities, or override Decision Engine decisions.
- Sprint 6 conflict resolution must remain Decision Engine-owned.

Reporting boundary:

- Reporting communicates certified outputs only.
- Reporting must not reinterpret urgency, prioritize allocation, inject decision logic, or override Decision Engine output.

## 12. Auditability Requirements

Every Sprint 6 decision concept must be:

- deterministic
- reproducible
- audit-traceable
- explainable
- stable under identical inputs
- supported by explicit source provenance
- supported by explicit decision rationale
- supported by a documented schema contract

Any future developer specification must define:

- required inputs
- required output schemas
- required log schemas
- source provenance fields
- rationale fields
- forbidden upstream mutation checks
- deterministic output checks
- no hidden filtering checks
- no non-Decision Engine authority leakage checks

## 13. Deterministic Decision Requirements

Sprint 6 decisions must be deterministic under identical inputs.

Future implementation must not rely on:

- random seeds for authority-bearing outputs
- live external state without captured provenance
- adaptive runtime behavior
- hidden persistence state
- order-unstable joins
- implicit filesystem ordering
- opaque model outputs
- non-versioned thresholds

If thresholds, precedence rules, or decision matrices are proposed later, they must be explicit, versioned in documentation, tested, and certified by governance audit before implementation.

## 14. Explainability Requirements

Every final downstream decision must be explainable.

Potential rationale fields may include:

- `allocation_rationale`
- `execution_rationale`
- `arbitration_reason`
- `conflict_resolution_reason`

Rationales must describe why the Decision Engine reached a downstream decision. They must not rewrite upstream classifications or imply that upstream layers made allocation decisions.

Explainability must support audit review of:

- input evidence used
- conflict-resolution path
- final downstream decision
- absence of hidden filtering
- absence of upstream mutation
- absence of authority leakage

## 15. Portfolio Coordination Boundaries

The Decision Engine may coordinate portfolio state downstream.

Portfolio coordination may include:

- exposure governance
- position coordination
- portfolio capacity coordination
- portfolio conflict resolution
- allocation sequencing

Portfolio coordination must not:

- mutate portfolio source files outside approved Decision Engine outputs
- convert Portfolio Intelligence into a preference engine
- convert Portfolio runtime behavior into an allocation authority
- suppress opportunities from Decision Engine visibility
- create hidden portfolio overrides
- emit undocumented sizing or allocation guidance

Any future sizing or capital-allocation concept requires explicit developer specification and governance audit certification.

## 16. Potential Output Concepts

Potential downstream Decision Engine output concepts may include:

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

These concepts are candidate planning vocabulary only. Final schema authority belongs to a future certified developer specification.

Older references to `decision_output.csv` are conceptual. `docs/technical/decision_engine_design_v2.md` states that the current runtime Decision Engine output is `data/processed/final_decisions.csv`; any Sprint 6 developer specification must resolve the output filename and schema contract before implementation.

## 17. Forbidden Semantics Outside Decision Engine

The following remain forbidden outside the Decision Engine:

- `tradeable_setup`
- `context_tradeable`
- `conviction`
- `BUY NOW`
- `SET LIMIT BUY`
- `SET STOP BUY`
- `urgency`
- `execution gating`
- hidden filtering
- allocation gating
- allocation recommendation
- execution recommendation
- portfolio priority
- actionability
- execution readiness
- suitability
- attractiveness
- preferred opportunity
- ranking score
- portfolio score
- final score

Sprint 6 must include checks proving these semantics do not leak outside Decision Engine-owned outputs.

## 18. Existing Sprint 6 Doctrine Conflicts

The existing `docs/sprints/sprint_6_decision_engine_core.md` contains governance-safe directional doctrine and uncertified implementation ambitions.

Governance-safe directional concepts:

- Decision Engine is the exclusive allocation authority.
- Upstream layers must not determine tradeability.
- Upstream layers must not determine conviction.
- Final actions must originate from the Decision Engine.
- Decision Engine must not rewrite upstream classifications.
- Outputs must be deterministic and reproducible.

Concepts requiring governance review before implementation:

- probabilistic evaluation
- conviction scoring
- opportunity ranking
- allocation priority
- tradeability scoring
- portfolio balancing
- execution aggressiveness
- decision persistence
- probabilistic smoothing
- escalation tracking
- allocation queues
- decision distributions

Potential conflicts with certified Sprint 0 through Sprint 5 doctrine:

- Broad ranking/scoring language could become opaque prioritization unless narrowed to Decision Engine-owned, deterministic, audited rules.
- Probabilistic allocation language could create hidden optimization unless deterministic and explainable.
- Persistence and smoothing belong more naturally to Sprint 7 unless explicitly narrowed and certified for Sprint 6.
- Execution aggressiveness language could become urgency semantics unless emitted only by Decision Engine with rationale and governance controls.
- Portfolio balancing language could become hidden portfolio override authority unless constrained to downstream Decision Engine conflict resolution.
- Required input references to `portfolio_state.csv` conflict with Sprint 5 certified outputs unless updated to use the certified Portfolio Intelligence artifact or a future certified Portfolio artifact.
- Older `decision_output.csv` references must be reconciled with `data/processed/final_decisions.csv` before implementation.

## 19. Concepts Requiring Controls Before Developer Specification

The following concepts may not proceed directly to implementation:

Probabilistic evaluation:

- requires deterministic rule definition
- requires explainability
- requires no hidden adaptive optimization
- requires audit certification

Conviction systems:

- require Decision Engine-only scope
- require clear schema naming
- require deterministic rationale
- require leakage checks outside Decision Engine

Ranking or allocation priority:

- requires proof that ranking exists only downstream
- requires documented tie-breaking
- requires no upstream reordering
- requires no silent suppression

Scoring systems:

- require explicit formula or rule contract
- require governance certification
- require tests for no opaque authority
- require stable output behavior

Execution aggressiveness:

- requires distinction from urgency leakage
- requires deterministic rationale
- requires Reporting boundary controls

Persistence or smoothing:

- requires decision whether deferred to Sprint 7
- requires hidden-state controls
- requires deterministic replay requirements

Portfolio balancing:

- requires Portfolio Intelligence interaction boundary
- requires no portfolio override authority outside Decision Engine
- requires no upstream mutation

## 20. In Scope for Sprint 6 Preparation

Sprint 6 preparation scope:

- define Decision Engine authority boundaries
- define downstream-only allocation doctrine
- define execution governance doctrine
- define arbitration and conflict-resolution doctrine
- define auditability requirements
- define deterministic decision requirements
- define explainability requirements
- identify conflicts in pre-certified Sprint 6 ambition
- identify concepts requiring governance controls
- prepare for Sprint 6 governance audit

## 21. Out of Scope for Sprint 6 Preparation

Out of scope:

- runtime implementation
- test implementation
- generated CSV updates
- strategy optimization
- portfolio optimization
- upstream architecture redesign
- Decision Engine refactoring
- threshold tuning
- machine learning or adaptive systems
- hidden optimization
- developer specification
- execution plan creation
- sprint closeout

## 22. Risks and Controls

Risk: hidden upstream contamination.

Control: require audit checks that no upstream file, schema, or runtime behavior gains allocation, execution, tradeability, conviction, urgency, ranking, scoring, priority, or recommendation semantics.

Risk: opaque scoring or ranking.

Control: require deterministic formulas or decision tables, tie-breaking rules, rationale logs, and audit certification before implementation.

Risk: probabilistic language becomes non-deterministic behavior.

Control: require deterministic implementation, reproducible outputs, no random/adaptive authority, and stable replay under identical inputs.

Risk: portfolio coordination becomes hidden portfolio management.

Control: constrain portfolio coordination to Decision Engine-owned downstream conflict resolution and rationale emission.

Risk: persistence concepts create hidden state.

Control: require explicit governance review and consider deferral to Sprint 7 unless Sprint 6 audit narrows the scope.

Risk: reporting receives urgency or recommendation authority.

Control: reporting may communicate Decision Engine outputs only and must not reinterpret them.

Risk: opportunity arbitration becomes silent filtering.

Control: preserve output visibility for the evaluated universe and require explicit rationale for downstream no-allocation or no-execution decisions.

## 23. Future Governance Audit Acceptance Criteria

Sprint 6 may proceed beyond preparation only if governance audit confirms:

- Sprint 0 through Sprint 5 doctrine is inherited.
- Decision Engine is the only allocation authority.
- Execution authority remains Decision Engine-owned.
- Upstream layers remain classification/enrichment-only.
- No upstream contamination is introduced.
- Portfolio Intelligence remains descriptive-only and neutral.
- Watchlist, Portfolio, and Reporting boundaries remain intact.
- Existing Sprint 6 ambition has been narrowed into governance-safe scope.
- Probabilistic/ranking/scoring/conviction concepts have deterministic explainability controls or are deferred.
- Persistence and smoothing concepts are either excluded from Sprint 6 or explicitly certified with hidden-state controls.
- Output schema direction is reconciled with current runtime output naming.
- Required input artifacts are reconciled with certified Sprint 5 outputs.
- No hidden filtering, opportunity suppression, or upstream mutation is authorized.
- Mandatory backlog reconciliation will be included in the governance audit.

## 24. Backlog Reconciliation Preparation

No new deferred backlog item is added during Sprint 6 preparation.

The concepts requiring governance narrowing are part of Sprint 6 audit scope rather than deferred work. If the Sprint 6 governance audit decides to defer persistence, smoothing, scoring, ranking, or portfolio-balancing ambitions, those decisions must be reconciled into `docs/sprints/project_backlog.md` during the audit under Mandatory Backlog Reconciliation.

## 25. Scrum Master Recommendation

READY FOR SPRINT 6 GOVERNANCE AUDIT
