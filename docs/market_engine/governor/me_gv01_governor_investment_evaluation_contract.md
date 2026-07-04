# ME-GV01 - The Governor Investment Evaluation Contract

Owner roles: Product Owner / Technical Architect / Financial Analyst / Data Steward / QA Lead / Governance Auditor

Job family: ME-GV - The Governor

Status: COMPLETED DOCS-ONLY CONTRACT

Contract version: `market-engine-governor-investment-evaluation-v1`

## Purpose

ME-GV01 defines the first contract for **The Governor**: a future governed investment-evaluation layer that consumes approved Market Engine evidence and produces an inspectable, fail-closed evaluation payload.

The Governor sits after Refinery and Analyzer evidence:

```text
Boiler
  -> Refinery
  -> Analyzer
  -> The Governor
  -> Dispatch Station
```

ME-GV01 is contract-only. It does not implement runtime behavior, scoring, recommendation-state logic, buy-zone logic, delivery behavior, portfolio mutation, watchlist mutation, broker behavior, or Decision Engine authority.

## Position in the roadmap

ME-GV01 starts after ME-RUN29 because ME-RUN29 produced the first reproducible expanded generic coverage classification evidence from staging-validation fixture evidence.

The active sequence is:

```text
ME-SA12 - Generic supported-universe cached-source coverage contract
  -> ME-SA13 - Generic coverage classifier
  -> ME-SA14 - Staging-validation adapter
  -> ME-RUN29 - Expanded generic coverage classification evidence
  -> ME-GV01 - Define The Governor investment evaluation contract
  -> ME-GV02 - Define Governor factor taxonomy and evidence requirements
```

ME-GV01 defines the envelope, states, required evidence categories, failure behavior, and authority boundaries. ME-GV02 must define the factor taxonomy and minimum evidence requirements before any scoring implementation begins.

## Contract identity

The future Governor output contract is:

```text
market-engine-governor-investment-evaluation-v1
```

The contract is versioned because later Governor sprints may add factor scoring, recommendation-state mapping, buy-zone explanation, and Dispatch Station output contracts. Those later sprints must either preserve this contract or explicitly version a successor.

## Approved input families

The Governor may consume only explicit, approved, versioned Market Engine evidence. It must not infer investment quality from raw or unapproved data.

Approved future input families for v1:

| Input family | Required role | Boundary |
| --- | --- | --- |
| Refinery coverage/readiness evidence | Proves source support, validity, provenance, freshness, consumability, completeness, and readiness blockers | Data trust only |
| Source Context evidence | Provides normalized source context accepted by Analyzer contracts | Evidence only |
| Fundamental Observations | Provides approved financial/fundamental observations where available | Evidence only |
| Derived Observations | Provides approved derived analytical observations | Evidence only |
| Setup Detection evidence | Provides approved technical/setup context where available | Evidence only |
| Analysis Review evidence | Provides analyzer-level evidence packaging and limitations | Non-actionable evidence review |
| Recommendation Review boundary evidence | Indicates whether upstream evidence may support later recommendation-state mapping | Boundary evidence only |
| Portfolio Context evidence | Future optional input for portfolio-fit analysis, only after ME-PR03 or equivalent approval | No mutation authority |
| Delivery/Reporting context | Future downstream output context only; not an input for investment quality | No delivery authority |

The Governor must not consume:

* raw provider payloads directly;
* unvalidated staged packages;
* missing-data defaults treated as real values;
* ticker-specific allowlists or shortcuts;
* live market data unless a later explicit source contract approves it;
* portfolio data unless an approved portfolio-context contract exists;
* manual operator opinions as evidence unless a later explicit human-review contract defines them.

## Required top-level output shape

The future output payload should use the following top-level structure:

```text
{
  "contract_version": "market-engine-governor-investment-evaluation-v1",
  "evaluation_id": "...",
  "ticker": "...",
  "market": "...",
  "company_name": "...",
  "input_references": {...},
  "evidence_readiness": {...},
  "evaluation_state": "...",
  "factor_evaluations": [...],
  "overall_evaluation": {...},
  "recommendation_state": {...},
  "buy_zone_explanation": {...},
  "position_management_explanation": {...},
  "risk_and_limitations": [...],
  "missing_evidence": [...],
  "blocked_reasons": [...],
  "authority_boundary": {...},
  "provenance": {...}
}
```

ME-GV01 defines this shape as a contract target only. No code writes this payload in this sprint.

## Evaluation states

The Governor v1 output must use explicit states. It must fail closed when evidence is missing, stale, unprovenanced, non-consumable, malformed, or insufficient.

Allowed contract states:

| State | Meaning | Authority |
| --- | --- | --- |
| `not_started` | Governor evaluation was not attempted | None |
| `blocked` | Required evidence is absent, invalid, stale, unprovenanced, non-consumable, or insufficient | None |
| `descriptive_only` | Only descriptive evidence exists; no investment evaluation may be produced | None |
| `partial_evaluation` | Some approved evidence exists, but not enough for full factor evaluation or recommendation-state mapping | Non-actionable explanation only |
| `evaluation_ready` | Required evidence is sufficient for a future Governor evaluation scaffold | Contract readiness only |
| `evaluation_completed_non_actionable` | A future scaffold may produce an inspectable evaluation while still withholding action semantics | Non-actionable explanation only |

Reserved future states:

```text
actionable_review
recommendation_state_ready
decision_ready
de_ready
```

The reserved future states must remain unreachable until later explicit governance approves them. ME-GV01 does not approve them.

## Evidence readiness requirements

The Governor must evaluate evidence sufficiency before it evaluates investment quality.

Required readiness gates:

1. **Contract identity**: all input evidence must declare an approved contract version.
2. **Source validity**: source packages must pass Refinery validation.
3. **Provenance**: source and derived evidence must include traceable provenance.
4. **Freshness**: stale evidence must be explicit and must downgrade or block evaluation.
5. **Consumability**: non-consumable or malformed evidence must fail closed.
6. **Completeness**: missing required evidence must remain explicit.
7. **Analyzer integrity**: upstream Analyzer stages must preserve limitations and blockers.
8. **Recommendation boundary**: Recommendation Review must not be bypassed for recommendation-state mapping.
9. **Portfolio boundary**: portfolio-fit evaluation must be disabled unless approved portfolio-context evidence exists.
10. **Authority boundary**: Decision Engine readiness remains false unless explicitly approved later.

## Factor evaluations

ME-GV01 reserves factor evaluation output but does not define the taxonomy. ME-GV02 must define taxonomy, scale, evidence requirements, and downgrade behavior.

Reserved factor families:

```text
fundamentals
growth
valuation
trend
momentum
risk
technical_setup
portfolio_fit
data_confidence
```

Until ME-GV02 defines the taxonomy, every factor must be treated as `not_defined` or `blocked_not_defined` by any future scaffold.

## Overall evaluation

The future `overall_evaluation` section may eventually summarize factor evidence, but ME-GV01 does not authorize scoring.

Forbidden in ME-GV01:

* numeric overall score;
* weighted score;
* ranking;
* urgency;
* conviction;
* tradeability;
* target price;
* target weight;
* position size;
* order-ready output.

The contract may reserve fields for future versions, but they must remain null, absent, or explicitly blocked until later sprints authorize them.

## Recommendation-state boundary

ME-GV01 does not authorize BUY / SELL / HOLD semantics.

The future `recommendation_state` section must be fail-closed by default:

```text
{
  "state": "blocked_not_authorized",
  "reason": "governor_recommendation_state_mapping_not_implemented_or_not_authorized",
  "actionable": false,
  "decision_engine_ready": false
}
```

ME-GV05 is the first planned sprint that may define recommendation-state mapping, and only under an approved boundary.

## Buy-zone and position-management boundary

ME-GV01 does not authorize buy-zone or position-management guidance.

The future sections must be blocked by default:

```text
buy_zone_explanation.state = "blocked_not_authorized"
position_management_explanation.state = "blocked_not_authorized"
```

ME-GV06 is the first planned sprint that may define buy-zone and position-management explanation contracts, and only after price, setup, market, valuation, and portfolio evidence requirements are explicit.

## Portfolio-fit boundary

Portfolio fit may not be evaluated from absent, ad hoc, or unapproved portfolio evidence.

Until an approved portfolio-context contract exists, the future `portfolio_fit` factor must be:

```text
blocked_missing_approved_portfolio_context
```

ME-PR03 or an equivalent later sprint remains the likely portfolio-context dependency.

## Fail-closed behavior

The Governor must never manufacture an investment evaluation from incomplete evidence.

Mandatory fail-closed cases:

* unsupported source family;
* invalid manifest;
* missing payload;
* missing provenance;
* stale source evidence;
* non-consumable snapshot;
* incomplete fundamentals;
* missing setup/price/market evidence;
* blocked Analysis Review;
* blocked Recommendation Review;
* missing approved portfolio context for portfolio-fit evaluation;
* missing factor taxonomy;
* unknown contract version;
* malformed input payload;
* unexpected state value.

Fail-closed output must include:

* explicit `evaluation_state`;
* deterministic `blocked_reasons`;
* missing evidence list;
* upstream references where available;
* `actionable=false`;
* `decision_engine_ready=false`.

## Authority boundary

ME-GV01 does not introduce or authorize:

* provider calls;
* live market data calls;
* source acquisition;
* snapshot import;
* staging validator changes;
* generic coverage classifier changes;
* Analyzer semantic changes;
* Recommendation Review semantic changes;
* Portfolio Review semantic changes;
* Dispatch Station behavior;
* Telegram/email sending;
* production reports;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* broker behavior;
* BUY / SELL / HOLD action semantics;
* scoring;
* ranking;
* urgency;
* conviction;
* tradeability;
* target prices;
* target weights;
* allocation;
* position sizing;
* order generation;
* execution instructions;
* Decision Engine decisions.

## Acceptance criteria for future implementation

A later ME-GV03 scaffold may implement this contract only if it proves:

* deterministic output;
* fail-closed behavior for every missing or invalid evidence gate;
* no provider, broker, delivery, portfolio/write, watchlist/write, scheduler, or UI side effects;
* no scoring until ME-GV04;
* no recommendation-state mapping until ME-GV05;
* no buy-zone or position-management explanation until ME-GV06;
* reserved actionable / Decision Engine-ready states remain unreachable unless explicitly approved.

## Next sprint

```text
ME-GV02 - Define Governor factor taxonomy and evidence requirements
```

ME-GV02 must define the factor families, factor states, evidence requirements, downgrade behavior, and factor-level fail-closed rules before any Governor runtime scaffold is implemented.
