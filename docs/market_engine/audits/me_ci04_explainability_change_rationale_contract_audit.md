# ME-CI04 - Explainability / Change-Rationale Contract Audit

## Objective

ME-CI04 defines `chatgpt_explainability_change_rationale_context` v1: the
controlled contract for explaining current states and change rationale within
the ChatGPT advisory architecture.

The sprint enables future advisory answers about why a state exists, what
changed, what did not change, which blockers remain, which evidence moved, and
which rationale is unavailable, without allowing ChatGPT to invent causal
chains, root causes, materiality, or historical state.

## Inspected upstream contracts

Inspected sources included:

* ME-RM06 ChatGPT advisory roadmap reposition;
* ME-CI01 Structured Decision Output contract;
* ME-CI02 ChatGPT Advisory Context contract and examples;
* ME-CI03 Portfolio Intelligence Context contract and examples;
* Analysis Review contracts and runtime semantics;
* Recommendation Review contracts and runtime semantics;
* Portfolio Review contracts and runtime semantics;
* Decision Engine handoff contracts and runtime semantics;
* Governor investment evaluation, scoring, recommendation, buy-zone, and
  position-management explanation contracts;
* Dispatch Station Governor report output contract;
* Delivery / Reporting contracts;
* ME-SA09 readiness semantics;
* run and audit documents containing historical run/blocker evidence.

## Architecture position

ME-CI04 sits under ME-CI01, ME-CI02, and ME-CI03. It is an advisory-context
contract, not a Market Engine stage, temporal diff engine, causal model,
Recommendation Review replacement, Portfolio Review replacement, Governor
replacement, or Decision Engine reasoning layer.

## Source-of-truth matrix

The source-of-truth matrix is defined in:

```text
docs/market_engine/contracts/me_ci04_explainability_change_rationale_contract.md
```

Key findings:

* Recommendation Review, Analysis Review, Portfolio Review, Governor, handoff,
  readiness, and advisory context artifacts already expose states, messages,
  blockers, reason codes, missingness, freshness, and provenance.
* Dispatch Station is presentation context only and cannot become a source of
  new decision truth.
* No canonical materiality model exists.
* No canonical causal attribution engine exists.
* No temporal comparison runtime exists.

## Contract identity

Approved identity:

```text
contract_name: chatgpt_explainability_change_rationale_context
contract_version: v1
schema_version: chatgpt-explainability-change-rationale-context-v1
artifact_type: market-engine-chatgpt-explainability-change-rationale-context
```

## Explanation availability

Allowed states:

```text
available
partial
unavailable
blocked
not_comparable
```

Hard boundaries:

```text
no prior comparable run != no change
missing evidence delta != unchanged evidence
same recommendation state != same underlying evidence
changed recommendation state != one proven causal reason
```

## Current-state rationale

Current-state rationale separates:

* supporting evidence;
* opposing evidence;
* constraints;
* blockers;
* uncertainty;
* context limitations;
* governance restrictions;
* freshness caveats;
* portfolio context contribution;
* Governor contribution.

A blocker is not automatically negative evidence. Uncertainty is not
automatically bearish evidence. A governance restriction is not a
recommendation.

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

Materiality states are reserved until a future upstream contract defines
materiality semantics. ME-CI04 does not invent thresholds.

## State transition semantics

State transitions must preserve previous state, current state, transition type,
run references, timestamps, source refs, readiness impact, eligibility impact,
portfolio impact, and blocker deltas.

Transitions may use only canonical upstream states. ME-CI04 does not introduce a
new BUY / SELL / HOLD taxonomy.

## Evidence delta semantics

Allowed evidence deltas:

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

New evidence is not by itself proof that the evidence caused a conclusion
change.

## Reason attribution levels

Allowed levels:

```text
explicit_upstream_reason
supported_contributing_factor
associated_change_only
unknown
prohibited_inference
```

This is the core ME-CI04 boundary. ChatGPT may explain attribution level, but it
must not upgrade association to causality.

## Blocker delta

Allowed blocker deltas:

```text
blocker_added
blocker_resolved
blocker_persisted
blocker_severity_changed
blocker_evidence_changed
blocker_status_unknown
```

Blockers remain tied to canonical upstream semantics. Blocker importance is not
invented by ChatGPT.

## Uncertainty delta

Allowed uncertainty deltas:

```text
uncertainty_reduced
uncertainty_increased
uncertainty_unchanged
uncertainty_reshaped
uncertainty_unresolved
comparison_unavailable
```

More data does not automatically mean less uncertainty. Fresh evidence does not
automatically mean higher confidence.

## Freshness-driven rationale

Freshness rationale is per family. A refreshed setup context and stale portfolio
snapshot must not be summarized as the full analysis being refreshed.

## Portfolio rationale

ME-CI04 consumes ME-CI03-compatible portfolio semantics and preserves these
separations:

```text
portfolio fit changed != standalone recommendation changed
position weight changed != Governor position-management recommendation changed
cash context changed != new sizing authority
```

## Unchanged conclusion semantics

Same state does not mean nothing changed.

The contract supports unchanged-state explanations with evidence changed,
evidence unchanged, mixed changes, persistent blockers, or unavailable
unchanged rationale.

## Contradiction handling

Conflicts between Structured Decision Output, Governor, Portfolio Review,
Dispatch Station, freshness, or lineage may produce:

```text
explanation_blocked
change_rationale_unavailable
descriptive_delta_only
mandatory_contradiction_disclosure
context_invalid
```

## Permission matrix

Allowed with proof:

* explain current state;
* name explicit reasons;
* summarize contributing factors;
* explain blockers;
* describe evidence deltas;
* describe state transitions;
* explain unchanged conclusion;
* explain portfolio context contribution;
* explain freshness caveats.

Allowed only with attribution caveat:

* associated changes without causal proof.

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
* unsupported counterfactuals.

## Use-case matrix

The contract covers:

* why a state is still unchanged;
* why a candidate appears stronger only when upstream evidence supports it;
* what changed since a reference run;
* why stronger recommendation language is blocked;
* why fundamental support does not equal actionability;
* what must change for a stronger conclusion when upstream blockers declare
  resolution conditions.

## Temporal comparison rules

Comparable runs require same instrument identity, compatible contract versions,
comparable source families, compatible methodology, valid provenance, valid
timestamps, no lineage ambiguity, and explicit baseline identity.

Non-comparable runs produce `not_comparable`, `descriptive_delta_only`, or
`change_rationale_unavailable`.

## Fail-closed matrix

The contract distinguishes:

```text
context_invalid
explanation_blocked
change_rationale_unavailable
descriptive_delta_only
eligible_with_mandatory_caveat
fully_explainable
```

Blocking cases include missing current run identity, missing provenance,
incompatible schemas, conflicting state sources, missing evidence lineage,
unknown or stale critical freshness, unsupported causality, unsupported
materiality, blocker conflicts, malformed Governor explanation, malformed
Portfolio Review context, and Dispatch Station / Structured Decision Output
mismatch.

## Examples

Synthetic examples were added for:

* current-state explanation;
* recommendation strengthened with attribution boundary;
* conclusion unchanged despite mixed evidence changes;
* change rationale unavailable;
* portfolio fit changed while standalone recommendation remained unchanged.

They are shape examples only and make no production market claim.

## Implementation decision

ME-CI04 remains docs-only.

Rationale:

* The roadmap positions ME-CI04 before typed schema, validator, deterministic
  assembler, prompt contract, and controlled advisory dry run.
* No temporal diff runtime exists.
* No canonical materiality model exists.
* No causal attribution engine exists.
* Runtime comparison now would create premature coupling to historical
  artifacts and incomplete advisory assembly.

## Validation

Validation results:

```text
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_explainability_change_rationale_context_v1_current_state.json
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_explainability_change_rationale_context_v1_recommendation_strengthened.json
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_explainability_change_rationale_context_v1_unchanged_mixed_delta.json
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_explainability_change_rationale_context_v1_change_unavailable.json
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_explainability_change_rationale_context_v1_portfolio_fit_changed.json
PASS - git diff --check
PASS - git status --short
PASS - git diff --stat
REVIEWED - rg -n "OpenAI|ChatGPT|Telegram|causal|cause|root cause|materiality|portfolio write|watchlist write|broker|allocation|position size|target weight|yfinance|SEC|EDGAR" src/market_engine scripts tests/market_engine docs/market_engine
PASS - PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q (1394 passed)
REVIEWED - grep -R "BUY" scripts/ | grep -v decision_engine.py
REVIEWED - grep -R "SELL" scripts/ | grep -v decision_engine.py
REVIEWED - grep -R "tradeable" scripts/ | grep -v decision_engine.py
```

The governance scan and mandatory greps returned expected existing boundary,
test, legacy portfolio, and pycache hits. ME-CI04 introduced documentation and
JSON examples only; it added no runtime behavior.

## Residual gaps

* No typed schema or validator exists for ME-CI04.
* No deterministic temporal comparison engine exists.
* No deterministic advisory context assembler exists.
* No canonical materiality model exists.
* No causal attribution engine exists.
* No ChatGPT prompt/runtime/API integration exists.

## Next sprint recommendation

Recommended next sprint:

```text
ME-CI05 - Produce daily ChatGPT-ready advisory artifact
```
