# ME-SA09 - Multi-Source Analysis-Context Readiness Contract

Sprint ID: ME-SA09
Status: COMPLETED DOCS-ONLY CONTRACT
Job family: ME-SA / Analysis Review
Date: 2026-06-29
Runtime impact: none
Test impact: none

## Purpose

ME-SA09 defines how Market Engine describes the readiness of a validated
multi-source analysis context.

The contract separates five concepts that must not be collapsed:

```text
descriptive_only
partial_analysis
recommendation_eligible
actionable_review
decision_ready
```

Readiness describes evidence completeness and downstream contract eligibility.
It does not rank candidates, determine tradeability, create conviction, select
an action, allocate capital, or authorize execution.

## Authority and Compatibility

This contract is subordinate to the repository doctrine:

```text
classification upstream
allocation downstream
Decision Engine = ONLY allocation authority
```

ME-SA09 does not change the current non-actionable Recommendation Review
contract, Portfolio Review contract, Decision Engine handoff contract, or
Delivery / Reporting contract.

In particular:

* `company_profile_only_context_non_actionable` remains mandatory for valid
  company-profile-only Recommendation Review input;
* current Recommendation Review output remains non-actionable under ME-RR01
  through ME-RR04;
* `ready_for_decision_engine_review` remains a structural handoff-readiness
  state and is not an action decision;
* Delivery / Reporting remains communication only;
* no ME-SA09 readiness level authorizes BUY, SELL, HOLD, allocation, position
  sizing, order creation, broker activity, or portfolio mutation.

## Source and Evidence Families

### Descriptive context

Includes company identity, business description, sector, industry, country,
currency, exchange, website, and descriptive source-availability metadata.

The current `company_profile` family belongs here. Descriptive context may
improve identification and interpretation, but it is not fundamental,
valuation, setup, price, market, recommendation, portfolio, or decision
evidence.

### Fundamental and financial evidence

Includes validated financial observations and derived financial observations,
such as source-grounded income, cash-flow, balance-sheet, growth, quality, or
cash-generation evidence supported by an approved contract.

Presence alone does not provide setup, price, market, portfolio, or action
evidence.

### Valuation context

Includes validated valuation observations required by an explicitly approved
strategy or review contract.

Valuation is conditional: where no approved contract requires valuation, its
absence does not automatically block `recommendation_eligible`. Where a future
contract does require it, missing or stale valuation context must block the
corresponding transition.

Descriptive fields must never be transformed into valuation evidence.

### Setup, price, and market context

Includes approved, source-grounded setup observations and any required price,
liquidity, trend, volatility, or market-context evidence.

Setup-like labels without their required price or market inputs are incomplete.
Setup context must not infer financial quality.

### Portfolio context

Includes explicit position state, exposure, concentration, policy constraints,
portfolio snapshot identity, freshness, and provenance under the approved
Portfolio Review contracts.

Portfolio context is required for downstream portfolio fit and handoff review.
It must not create or upgrade upstream analytical evidence.

### Provenance, manifest, and staleness context

Includes contract versions, source identity, snapshot and manifest identity,
timestamps, lineage, validation state, ticker alignment, missing-data markers,
and staleness state.

This family is a gate across every readiness level. Evidence that is malformed,
unsupported, stale beyond its approved threshold, unprovenanced, or
identity-misaligned cannot satisfy a higher readiness requirement.

### Delivery, reporting, and handoff context

Includes Portfolio Review provenance, Decision Engine handoff readiness,
blocked reasons, delivery safety state, and reporting lineage.

This context is downstream control evidence only. It must not backfill missing
fundamental, valuation, setup, price, or market evidence and must not create a
decision.

## Readiness Levels

### `descriptive_only`

Meaning:

* validated context can identify or describe the company or its source
  availability;
* no sufficient non-descriptive analytical evidence exists;
* company-profile-only context always belongs here.

Implications:

* Analysis Review may preserve descriptive context and limitations;
* Recommendation Review must be blocked or non-actionable;
* Portfolio Review must not treat the context as recommendation evidence;
* Decision Engine handoff must not be ready;
* Delivery / Reporting may communicate the blocked state only through an
  approved downstream path.

Required ME-SA08 reason for company-profile-only input:

```text
company_profile_only_context_non_actionable
```

### `partial_analysis`

Meaning:

* at least one approved non-descriptive analytical family is present;
* required complementary evidence is absent, incomplete, stale, conflicted, or
  unprovenanced.

Examples:

* fundamentals without required setup, price, or market context;
* setup-like evidence without sufficient fundamental or financial evidence;
* otherwise sufficient evidence with unresolved missing-data or staleness
  gates;
* required valuation context missing under a future strategy contract.

Implications:

* Analysis Review may produce evidence notes, gap notes, and missing-data
  notes;
* Recommendation Review may preserve a blocked or insufficient-evidence state;
* no final action fields may be produced;
* Portfolio Review and Decision Engine handoff must not be made ready by the
  incomplete analysis;
* reporting must preserve the limitation.

### `recommendation_eligible`

Meaning:

* sufficient validated, provenance-backed, non-stale analytical families exist
  for Recommendation Review to evaluate the case;
* both fundamental or financial evidence and setup, price, or market evidence
  are required unless an explicit future strategy contract defines another
  requirement set;
* any strategy-required valuation context is present;
* no blocking identity, provenance, missing-data, or staleness issue remains.

Eligibility is permission to evaluate under the approved Recommendation Review
contract. It is not a favorable review, a recommendation, an action, a rank, or
a Decision Engine handoff.

Under the current ME-RR contracts, Recommendation Review remains a
non-actionable human-review layer even when input is
`recommendation_eligible`.

### `actionable_review`

Meaning:

* reserved readiness level for a future explicitly approved Recommendation
  Review contract that defines what an actionable review means;
* all `recommendation_eligible` requirements and future contract-specific
  evidence requirements are satisfied;
* evidence is valid, non-stale, provenance-backed, and free of blocking
  limitations.

Current availability:

```text
not currently claimable
```

ME-RR01 through ME-RR04 prohibit actionable Recommendation Review output.
ME-SA09 does not override those contracts. No current runtime or artifact may
claim `actionable_review`.

Even if a future contract activates this level, descriptive context alone must
never supply action, entry, exit, stop, target, conviction, sizing, or trade
guidance. Recommendation Review must still not allocate capital, generate an
order, call a broker, or mutate portfolio state.

### `decision_ready`

Meaning:

* reserved highest multi-source readiness level;
* a future approved `actionable_review` prerequisite is satisfied;
* Portfolio Review and required portfolio context are valid, complete,
  non-stale, aligned, and not blocked;
* handoff provenance, missing-data gates, and downstream safety checks are
  satisfied;
* the context may be considered for an approved Decision Engine boundary.

Current availability:

```text
not currently claimable by the ME-SA09 readiness classifier
```

The existing ME-DE state `ready_for_decision_engine_review` is structurally
ready for Decision Engine evaluation and remains distinct from ME-SA09
`decision_ready`. Neither state is an action, allocation, execution, broker, or
delivery authorization.

Only the Decision Engine may determine an action or allocation under its own
approved contract.

## Readiness Matrix

Legend:

* D = descriptive context
* F = fundamental or financial evidence
* V = strategy-required valuation context
* S = setup, price, or market context
* P = portfolio context
* G = valid provenance, manifest, identity, and freshness gates
* H = approved downstream handoff and safety context

| Evidence combination | Readiness | Required outcome |
| --- | --- | --- |
| D + G only | `descriptive_only` | Preserve description; block Recommendation Review with `company_profile_only_context_non_actionable` when D is company-profile-only. |
| D without valid G | No readiness asserted | Fail closed with provenance, identity, contract, or staleness reason. |
| F + G without required S | `partial_analysis` | Preserve financial evidence; block higher readiness with `missing_setup_or_price_context`. |
| S + G without sufficient F | `partial_analysis` | Preserve setup or market evidence; block higher readiness with `missing_fundamental_evidence`. |
| F + S with incomplete or stale G | `partial_analysis` at most | Block higher readiness with `stale_or_unprovenanced_analysis_context`. |
| D + F + S + G, with V when required | `recommendation_eligible` | Recommendation Review may evaluate under its approved contract; no positive or actionable result is implied. |
| D + F + S + G without required V | `partial_analysis` | Block the strategy-specific transition and preserve missing valuation context. |
| Recommendation-eligible evidence plus a future approved actionable-review contract | `actionable_review` | Reserved and currently unavailable; must remain blocked under current ME-RR contracts. |
| Future actionable review + valid P + G + H | `decision_ready` | Reserved ME-SA09 level; only an approved Decision Engine may subsequently determine action or allocation. |
| Any combination with malformed, unsupported, identity-mismatched, or materially unprovenanced required input | No higher readiness asserted | Fail closed and preserve deterministic blocked reasons. |

Additional descriptive context never upgrades a row by itself. Readiness is
determined by the required non-descriptive evidence and control gates.

## Deterministic Blocked Reasons

ME-SA09 reserves or reuses these reason names for future classification:

| Reason | Use |
| --- | --- |
| `company_profile_only_context_non_actionable` | Valid context contains only descriptive company-profile evidence. |
| `insufficient_analysis_context` | Available evidence does not meet the next readiness level and no more specific reason fully explains the gap. |
| `missing_fundamental_evidence` | Required fundamental or financial evidence is absent or insufficient. |
| `missing_setup_or_price_context` | Required setup, price, or market evidence is absent or insufficient. |
| `stale_or_unprovenanced_analysis_context` | Required evidence is stale, lacks approved lineage, or fails manifest/provenance requirements. |

Existing downstream contracts retain their more specific reasons, including
`blocked_missing_portfolio_context`, `blocked_stale_portfolio_context`,
`blocked_incomplete_provenance`, `blocked_ticker_mismatch`, and
`blocked_insufficient_evidence`.

Reason precedence for a future deterministic classifier should be:

1. invalid or unsupported contract and identity failures;
2. provenance and staleness failures;
3. company-profile-only boundary;
4. missing required fundamental evidence;
5. missing required setup, price, or market context;
6. strategy-specific missing evidence;
7. generic insufficient analysis context.

Multiple reasons may be preserved when the output contract supports an ordered
reason list.

## Explicit Non-Actionable Cases

The following cases must not become actionable or Decision Engine-ready:

* company-profile-only context;
* descriptive context plus provenance metadata only;
* fundamentals without required setup, price, or market context;
* setup-like or price context without sufficient fundamental evidence;
* missing strategy-required valuation context;
* stale analytical evidence;
* unprovenanced or manifest-invalid evidence;
* ticker or entity identity mismatch;
* partial, malformed, unsupported, or conflicted required input;
* Recommendation Review without valid Portfolio Review and portfolio context;
* structurally ready handoff treated as if it were a Decision Engine decision;
* Delivery / Reporting output used to backfill or reinterpret upstream
  evidence.

## Prohibited Inferences

Market Engine layers must not infer:

* fundamentals, quality, growth, cash generation, or valuation from company
  identity, description, sector, industry, country, exchange, or website;
* setup, price level, liquidity, volatility, trend, entry, exit, stop, or target
  from descriptive or fundamental context alone;
* financial quality from setup, price, or market context alone;
* missing values as numeric zero;
* current evidence from stale evidence;
* provenance from filenames, directory names, or unsupported metadata;
* recommendation eligibility from source-family count alone;
* a favorable review from recommendation eligibility;
* actionability from a human-review state;
* portfolio fit from Recommendation Review;
* action, allocation, conviction, urgency, tradeability, rank, or position size
  outside Decision Engine authority;
* a Decision Engine decision from handoff readiness;
* delivery permission from analytical or handoff readiness.

## Downstream Implications

| Layer | Contract implication |
| --- | --- |
| Analysis Review | May classify evidence readiness and preserve descriptive context, analytical evidence, gaps, provenance, and staleness. Must not create recommendation or action authority. |
| Recommendation Review | May consume only an approved eligible Analysis Review contract. Must preserve profile-only and partial-analysis blockers. Current contracts remain non-actionable. |
| Portfolio Review | May assess explicit portfolio context only after approved Recommendation Review input. Must not repair or upgrade insufficient upstream evidence. |
| Decision Engine handoff | May be structurally ready only under ME-DE requirements. Handoff readiness is not a decision and must preserve all blockers and provenance. |
| Delivery / Reporting | May communicate approved upstream states only. Must preserve blocked and non-actionable states and must not create interpretation, urgency, prioritization, or action logic. |

## Transition Invariants

Readiness transitions must be monotone with respect to validated evidence, but
may move downward when evidence becomes stale, invalid, unsupported, or
unprovenanced.

The following invariants apply:

* D alone cannot move above `descriptive_only`;
* D added to another context cannot compensate for missing F, V, S, P, G, or H;
* `partial_analysis` cannot skip directly to `decision_ready`;
* `recommendation_eligible` does not imply `actionable_review`;
* `actionable_review` does not imply `decision_ready`;
* handoff readiness does not imply a Decision Engine outcome;
* delivery readiness does not imply analytical, recommendation, or decision
  readiness;
* a blocked upstream state must remain blocked downstream.

## Non-Goals

ME-SA09 does not:

* implement a readiness enum, classifier, schema, builder, validator, or
  persistence path;
* change runtime code or tests;
* change cached-source validation;
* add multi-source discovery or orchestration;
* select providers or make network calls;
* define strategy thresholds;
* define valuation formulas;
* add recommendation logic;
* activate `actionable_review` or `decision_ready`;
* change Portfolio Review, Decision Engine, handoff, Delivery / Reporting,
  Telegram, broker, portfolio, or watchlist behavior;
* authorize generated data or production writes.

## Future ME-SA10 Implementation Notes

A future ME-SA10 may implement only the non-authoritative readiness
classification defined here.

Before implementation, ME-SA10 should define:

* a versioned readiness output contract;
* typed source-family inventory and applicability rules;
* deterministic reason precedence;
* explicit strategy requirement input rather than hidden defaults;
* freshness and provenance inputs sourced from approved upstream contracts;
* ticker and entity alignment checks;
* fail-closed handling for unknown source families and contract versions;
* tests for every readiness matrix row and downgrade path;
* proof that profile context cannot upgrade analytical readiness;
* proof that classification adds no BUY, SELL, HOLD, conviction, urgency,
  ranking, tradeability, allocation, sizing, execution, or delivery authority.

ME-SA10 must treat `actionable_review` and `decision_ready` as unavailable unless
separate, explicit governance and contract work first authorizes those levels.

## Acceptance

ME-SA09 is complete when this contract, its audit, and its backlog and roadmap
records exist as documentation-only changes.
