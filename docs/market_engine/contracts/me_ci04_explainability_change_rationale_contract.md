# ME-CI04 - Explainability / Change-Rationale Contract v1

## Purpose

ME-CI04 defines the ChatGPT-readable Explainability / Change-Rationale Context:
the controlled contract for explaining current states, evidence deltas,
blockers, uncertainty, freshness, portfolio-rationale changes, and
state transitions without inventing causality.

Approved identity:

```text
contract_name: chatgpt_explainability_change_rationale_context
contract_version: v1
schema_version: chatgpt-explainability-change-rationale-context-v1
artifact_type: market-engine-chatgpt-explainability-change-rationale-context
```

ME-CI04 structures which reasons, changes, and explanatory relationships are
proven by upstream Market Engine artifacts. It does not create a temporal diff
engine, causal model, materiality model, recommendation engine, Decision Engine
reasoning layer, or ChatGPT prompt.

## Architectural position

ME-CI04 sits inside the ChatGPT advisory architecture:

```text
ME-CI01 Structured Decision Output
-> ME-CI02 ChatGPT Advisory Context
-> ME-CI03 Portfolio Intelligence Context
-> ME-CI04 Explainability / Change-Rationale Context
-> later typed schema / validator
-> later deterministic assembler
-> later prompt contract
-> later controlled advisory dry run
```

ChatGPT may explain only proven reason and change relationships. It must not
reconstruct missing history, infer root cause, rank factor importance, determine
materiality, or turn temporal association into causality.

## Source-of-truth matrix

| Explanation family | Upstream state producer | Canonical artifact family | Explicit explanation source? | ME-CI04 handling | Forbidden without evidence |
| --- | --- | --- | --- | --- | --- |
| Recommendation rationale | Recommendation Review | `sec-companyfacts-recommendation-review-v1` | Partly: review state, category, item message, supporting/blocking categories, missing data | Include direct reasons and references | Actionability, stronger recommendation, causal root cause |
| Analysis interpretation | Analysis Review | `sec-companyfacts-analysis-review-v1` | Partly: review items, states, missing observations, source/derived refs | Include descriptive supporting/opposing context | Investment judgement or recommendation |
| Portfolio fit rationale | Portfolio Review | `sec-companyfacts-portfolio-review-v1` | Yes for review states/items and data limitations | Include portfolio-fit reason and limitation states | Standalone recommendation change or sizing |
| Position-management rationale | Governor | `market-engine-governor-buy-zone-position-management-explanation-v1` | Yes when reason codes and explanation states are present | Include Governor reason codes and limitations | Order, allocation, quantity, or action authority |
| Buy-zone rationale | Governor | `market-engine-governor-buy-zone-position-management-explanation-v1` | Yes when approved price/setup context exists | Include bounded explanation and evidence refs | Price target, stop order, entry order, urgency |
| Blocker rationale | Review/governance artifacts | Analysis Review, Recommendation Review, Portfolio Review, handoff, Governor, ME-CI02 | Yes when blocker is explicit | Include blocker state and source | Treat blocker as negative evidence by default |
| Readiness rationale | Readiness contracts | ME-SA09 readiness, Decision Engine handoff, ME-CI02 eligibility | Yes for readiness states and blocked reasons | Include readiness blockers and limits | Convert readiness to actionability |
| Change since prior run | Versioned comparable artifacts | Current and reference artifacts | Only when comparable artifacts exist | Include state/evidence deltas with comparison caveats | Infer change from missing prior context |
| Evidence added/removed/refreshed | Source/evidence lineage | Source Context, observations, review refs, manifests | Usually delta only unless upstream reason says so | Include temporal association or explicit reason level | Claim evidence caused conclusion change |
| Freshness degradation | Freshness metadata | ME-CI02/CI03 freshness, source timestamps, stale markers | Yes for stale markers; not necessarily cause | Include freshness-driven scope limits | Claim whole analysis refreshed from one fresh family |
| Confidence change | Upstream confidence semantics | Structured Decision Output, Governor scores when present | Only when explicit | Include direct confidence deltas | Improve confidence because more data exists |
| Uncertainty change | Uncertainty semantics | ME-CI02 uncertainty, blockers, missingness | Partly | Include uncertainty delta and unresolved limits | Claim all uncertainty resolved from one blocker |
| Portfolio-context change | Portfolio artifacts | ME-CI03-compatible context, Portfolio Review | Partly | Include holdings/fit/context availability delta | Allocation rationale or standalone attractiveness |
| Decision-output change | Structured Decision Output | `structured-decision-output-v1` | Yes for current fields; change only if prior comparable artifact exists | Include state transition and reason codes | Decision Engine reasoning if absent |
| Presentation rationale | Dispatch Station | `market-engine-dispatch-station-governor-report-v1` | Presentation only | Reference as non-authoritative display context | Treat presentation summary as source of truth |

Absence of an explanation source is not permission to infer an explanation.

## Identity

Every artifact must contain:

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | yes | Must equal `chatgpt-explainability-change-rationale-context-v1`. |
| `artifact_type` | yes | Must equal `market-engine-chatgpt-explainability-change-rationale-context`. |
| `generated_at` | yes | Context generation time. Not evidence freshness. |
| `run_id` | yes | Explainability context run identity. |
| `instrument` | yes | Current instrument identity. |
| `current_run_identity` | yes | Current advisory or source run identity. |
| `reference_run_identity` | yes | Reference, prior, baseline, or explicit null with reason. |
| `comparison_window` | yes | Current/reference window and scope. |
| `source_artifact_refs` | yes | Artifacts used for current explanation and comparison. |
| `current_advisory_context_ref` | yes | Current ME-CI02 context reference when available. |
| `prior_advisory_context_ref` | yes | Reference ME-CI02 context reference, or null with missingness reason. |

The following identities are distinct and must not be collapsed:

```text
current run
reference run
previous chronological run
previous comparable run
baseline run
```

## Explanation availability

Allowed availability states:

| State | Meaning |
| --- | --- |
| `available` | Required reason evidence is present and traceable for the declared scope. |
| `partial` | Some rationale is proven, but material sources, deltas, or attribution are incomplete. |
| `unavailable` | No sufficient explanation context exists for the requested scope. |
| `blocked` | Invalid, conflicting, unsupported, stale, or unprovenanced context prevents explanation. |
| `not_comparable` | Change rationale cannot be supported because runs or artifacts are not comparable. |

Hard boundaries:

```text
no prior comparable run != no change
missing evidence delta != unchanged evidence
same recommendation state != same underlying evidence
changed recommendation state != one proven causal reason
```

## Current-state rationale

Current-state rationale must distinguish:

```text
supporting_evidence
opposing_evidence
constraint
blocker
uncertainty
context_limitation
governance_restriction
freshness_caveat
portfolio_context_contribution
governor_contribution
```

These are not synonyms. A blocker is not automatically bearish evidence.
Uncertainty is not automatically negative evidence. A governance restriction is
not a recommendation.

Each rationale entry must include:

```text
family
state
summary_code
source_ref
attribution_level
provenance
limitations
```

## Change classification

Allowed v1 classifications:

```text
state_changed
state_unchanged
evidence_changed
evidence_unchanged
mixed_change
blocked_change
not_comparable
unknown
```

The following materiality states are reserved and not claimable unless an
upstream contract explicitly defines materiality semantics:

```text
materially_strengthened
materially_weakened
materially_unchanged
```

ME-CI04 does not invent materiality thresholds. If upstream materiality is
absent, ChatGPT may describe observable deltas but must not claim material
change.

## State transition

A transition entry must include:

```text
previous_state
current_state
transition_type
transition_run_refs
transition_timestamps
source_artifact_refs
readiness_impact
eligibility_impact
portfolio_impact
blockers_added
blockers_resolved
blockers_persisted
blockers_unknown
```

Transitions may use only canonical upstream states such as ME-CI02
`eligible`, `descriptive_only`, `blocked`; Portfolio Review
`portfolio_context_supported`, `portfolio_context_partial`, and
`blocked_by_missing_portfolio_context`; Decision Engine handoff readiness
states; Governor explanation states; and Structured Decision Output states.

ME-CI04 must not introduce a new BUY / SELL / HOLD taxonomy.

## Evidence delta

Allowed evidence delta states:

```text
added
removed
refreshed
became_stale
superseded
contradicted
unchanged
availability_unknown
not_comparable
```

Each entry must include:

```text
evidence_family
identifier
previous_state
current_state
change_type
provenance
timestamps
materiality
affected_downstream_states
attribution_level
```

New evidence present is not evidence that it caused a conclusion change.

## Reason attribution levels

Allowed attribution levels:

| Level | Meaning | ChatGPT wording boundary |
| --- | --- | --- |
| `explicit_upstream_reason` | An upstream artifact explicitly states that a factor is a reason for a state. | May say the upstream context identifies X as a reason. |
| `supported_contributing_factor` | Upstream semantics support X as a contributing factor, but not sole cause. | Must use contributing-factor wording. |
| `associated_change_only` | X changed in the comparison window, but causality is not proven. | Must disclose association without causal proof. |
| `unknown` | Insufficient information. | Must say rationale is unknown or unavailable. |
| `prohibited_inference` | The proposed explanation would reconstruct unsupported causality. | Must not provide the explanation. |

## Blocker delta

Allowed blocker delta states:

```text
blocker_added
blocker_resolved
blocker_persisted
blocker_severity_changed
blocker_evidence_changed
blocker_status_unknown
```

Blockers must remain tied to canonical upstream semantics and source refs.
ChatGPT may explain a blocker as the reason for limited eligibility only when
the upstream context supports that relationship.

## Uncertainty delta

Allowed uncertainty delta states:

```text
uncertainty_reduced
uncertainty_increased
uncertainty_unchanged
uncertainty_reshaped
uncertainty_unresolved
comparison_unavailable
```

More evidence is not automatically less uncertainty. Fresh evidence is not
automatically higher confidence. A resolved blocker does not mean all
uncertainty is resolved.

## Freshness-driven rationale

Freshness rationale is per family:

```text
evidence_refreshed
evidence_aging
evidence_stale
mixed_freshness
comparison_distorted_by_freshness_mismatch
```

If fundamentals are unchanged, price/setup context is refreshed, and portfolio
snapshot is stale, the valid explanation scope must say exactly that. It must
not summarize the full analysis as refreshed.

## Portfolio rationale

ME-CI04 consumes ME-CI03-compatible portfolio semantics for:

```text
holding_state_change
position_weight_change
portfolio_fit_change
concentration_warning_change
position_management_change
portfolio_context_availability_change
```

Hard boundaries:

```text
portfolio fit changed != standalone recommendation changed
position weight changed != Governor position-management recommendation changed
cash context changed != new sizing authority
```

Portfolio rationale may explain proven portfolio context. It must not
reconstruct allocation rationale.

## Unchanged conclusion rationale

Same state does not mean nothing changed.

Allowed unchanged-conclusion patterns:

| Pattern | Required proof |
| --- | --- |
| `conclusion_unchanged_evidence_changed` | Same current/prior state plus evidence deltas. |
| `conclusion_unchanged_evidence_unchanged` | Same state plus comparable evidence showing no relevant delta. |
| `conclusion_unchanged_mixed_changes` | Mixed evidence deltas with explicit attribution limits. |
| `conclusion_unchanged_blocker_persisted` | Same state plus persistent blocker linked to eligibility limitation. |
| `unchanged_not_explainable` | Same state but no comparable evidence or attribution. |

ChatGPT must not compress unchanged state into "nothing changed" unless the
evidence-delta contract supports that exact claim.

## Contradiction handling

Conflict cases include:

* current artifacts disagree;
* prior artifact conflicts with current lineage;
* Dispatch Station summary conflicts with Structured Decision Output;
* Governor explanation conflicts with review state;
* portfolio rationale conflicts with standalone rationale;
* freshness timestamps make comparison invalid;
* recommendation state changed but no canonical transition source exists.

Allowed results:

```text
explanation_blocked
change_rationale_unavailable
descriptive_delta_only
mandatory_contradiction_disclosure
context_invalid
```

## Explainability permission matrix

| Permission | Allowed only when |
| --- | --- |
| Explain current state | Current state, source, provenance, and reason evidence are present. |
| Name explicit reasons | Attribution level is `explicit_upstream_reason`. |
| Summarize contributing factors | Attribution level is `supported_contributing_factor`. |
| Describe evidence deltas | Comparable artifacts and lineage exist. |
| Describe state transitions | Previous and current states are comparable. |
| Explain unchanged conclusion | Same state plus explicit unchanged-rationale pattern. |
| Explain portfolio contribution | ME-CI03-compatible context proves the portfolio relation. |
| Explain freshness caveats | Freshness is per-family and traceable. |
| Mention associated changes | Must include causal caveat. |

Forbidden:

* root cause determination;
* factor-importance ranking;
* exclusive cause claims;
* materiality determination;
* missing historical state reconstruction;
* recommendation cause from price movement;
* blocker importance invention;
* confidence-score adjustment;
* hidden motive or intent attribution;
* unchanged state as "nothing changed";
* correlation as causation;
* counterfactuals such as "without blocker X this would be BUY" unless
  explicitly upstream-supported.

## Use-case matrix

| User question | Required context | Safe answer class |
| --- | --- | --- |
| "Why is ASML still HOLD?" | Current state, comparable prior state, supporting/opposing factors, blockers, uncertainty, unchanged-rationale source refs | Explain unchanged state and proven rationale; avoid "nothing changed" unless proven. |
| "Why is AMD more attractive than the previous analysis?" | Comparable prior/current run, state or evidence delta, attribution level, freshness compatibility | Answer only if "more attractive" is upstream-proven; otherwise describe observable deltas. |
| "What changed since last week?" | Reference run, comparison window, evidence delta, state delta, blocker delta, uncertainty delta | Describe deltas and caveats; do not infer causes. |
| "Why can you not give a buy recommendation yet?" | Eligibility state, blockers, missingness, readiness constraints, advisory permission boundary | Explain blockers and non-actionability from upstream states. |
| "Why is the company fundamentally good but still not a buy?" | Fundamental/analysis context, setup/valuation context, Recommendation Review, Portfolio Review, governance, actionability state | Separate evidence support from recommendation and actionability. |
| "What must change for a stronger conclusion?" | Upstream-declared resolution conditions or blockers | Explain declared blockers/resolution conditions only. |

## Temporal comparison rules

Runs are comparable only when:

* instrument identity matches;
* schema versions are compatible;
* source families are comparable;
* methodology is compatible;
* provenance is valid;
* timestamps are valid;
* lineage is unambiguous;
* baseline or reference identity is explicit.

Non-comparable cases:

| Case | Result |
| --- | --- |
| v1 versus unsupported v2 schema | `not_comparable` or `context_invalid`. |
| Different source coverage | `descriptive_delta_only` or `not_comparable`. |
| Partial prior run | Deltas limited to available families. |
| Stale prior run | Freshness caveat or `not_comparable`. |
| Missing Governor output in one run | Governor rationale unavailable for comparison. |
| Missing Portfolio Review in one run | Portfolio rationale unavailable for comparison. |
| Different portfolio snapshot | Portfolio deltas require caveat or block. |
| Different instrument identity mapping | `context_invalid`. |

## Fail-closed matrix

| Condition | Result |
| --- | --- |
| Missing current run identity | `context_invalid` |
| Missing reference run identity for change question | `change_rationale_unavailable` |
| Missing provenance | `explanation_blocked` |
| Incompatible schemas | `context_invalid` or `not_comparable` |
| Non-comparable runs | `change_rationale_unavailable` |
| Conflicting state sources | `explanation_blocked` |
| Missing prior state | `descriptive_delta_only` or `change_rationale_unavailable` |
| Missing evidence lineage | `explanation_blocked` for affected claim |
| Unknown freshness | `eligible_with_mandatory_caveat` or blocked for critical family |
| Stale critical evidence | `explanation_blocked` or `descriptive_delta_only` |
| Unsupported causal attribution | `descriptive_delta_only` |
| Unsupported materiality claim | `descriptive_delta_only` |
| Blocker conflict | `explanation_blocked` |
| Uncertainty conflict | `explanation_blocked` or caveated explanation |
| Malformed Governor explanation | Governor rationale unavailable |
| Malformed Portfolio Review context | Portfolio rationale unavailable |
| Dispatch Station / Structured Decision Output mismatch | `explanation_blocked` |

Allowed top-level validation states:

```text
context_invalid
explanation_blocked
change_rationale_unavailable
descriptive_delta_only
eligible_with_mandatory_caveat
fully_explainable
```

## Required validation object

```text
validation.contract_valid
validation.validation_state
validation.errors
validation.warnings
validation.blocked_reasons
validation.unsupported_claims
validation.semantic_boundary_version
```

## Implementation decision

ME-CI04 is docs-only.

Rationale:

* Existing roadmap places ME-CI04 before typed schema, validator, deterministic
  assembler, prompt contract, and advisory dry run.
* No temporal diff runtime exists.
* No canonical materiality or causal attribution engine exists.
* Runtime comparison now would prematurely couple historical artifacts and
  advisory output.

## Next sprint

Recommended next sprint:

```text
ME-CI05 - Produce daily ChatGPT-ready advisory artifact
```
