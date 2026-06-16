# ME-PR01 — Portfolio Review contract from Recommendation Review

## Status

COMPLETED BY ME-PR01

## Sprint

ME-PR01 — Define Portfolio Review contract from Recommendation Review

## Job family

ME-PR — Portfolio Review jobs

## Purpose

ME-PR01 defines the first Portfolio Review contract after Setup Detection-aware Recommendation Review exists.

Portfolio Review consumes approved Recommendation Review output and applies explicit portfolio context so downstream layers can later decide whether the reviewed candidate fits the current portfolio.

Portfolio Review is a review layer only. It does not buy, sell, hold, rebalance, resize, execute, alert, deliver, report, mutate portfolio state, or call the Decision Engine.

## Governing upstream chain

Approved architectural chain:

```text
Source Refresh / raw snapshots
→ Source Context
→ Fundamental Observations
→ Derived Observations
→ Setup Detection
→ Analysis Review
→ Recommendation Review
→ Portfolio Review
→ Decision Engine handoff / action authority
→ Delivery / reporting
```

Portfolio Review is downstream of Recommendation Review.

Portfolio Review must not bypass Recommendation Review, Analysis Review, Setup Detection, Derived Observations, Fundamental Observations, Source Context, or Source Refresh.

Portfolio Review must not reinterpret raw provider payloads or source snapshots directly.

## Approved input contract

The approved Recommendation Review input contract for the first Portfolio Review layer is:

```text
sec-companyfacts-recommendation-review-v1
```

This includes Recommendation Review output that preserves Setup Detection-aware Analysis Review provenance as implemented by ME-RR04.

Portfolio Review may consume a Recommendation Review artifact only when the artifact declares the approved contract version.

A missing, malformed, unknown, unsupported, or incompatible Recommendation Review contract must fail closed or emit controlled invalid-input output in a future ME-PR02 implementation.

## Required Recommendation Review input requirements

The Recommendation Review input must preserve, when available:

* recommendation review format version;
* recommendation review run identifier;
* ticker;
* upstream Analysis Review references;
* Setup Detection-aware Analysis Review references;
* setup categories;
* setup states;
* setup evidence;
* setup limitations;
* missing setup observations;
* source observation references;
* derived observation references;
* Recommendation Review state;
* Recommendation Review category;
* missing-data markers;
* numeric-zero evidence and references;
* non-actionable boundary metadata.

Portfolio Review must preserve Recommendation Review provenance and must not flatten setup-aware or source-aware evidence in a way that loses traceability.

## Approved portfolio context input

Portfolio Review requires explicit portfolio context separate from Recommendation Review.

The first Portfolio Review contract defines the portfolio-context input family as:

```text
market-engine-portfolio-context-v1
```

This context is not implemented by ME-PR01.

A future ME-PR02 implementation may accept a structured portfolio context object or persisted portfolio-context artifact only after validating that it is explicitly supplied by an approved portfolio-context source.

Portfolio Review must not silently infer portfolio holdings from broker exports, reports, watchlists, old runtime data, or generated output folders unless a later sprint explicitly approves that input path.

## Required portfolio context fields

The first portfolio context must provide enough information to review fit without creating action authority.

Required minimum context fields or equivalent structured fields:

* portfolio context format version;
* portfolio context run identifier or snapshot identifier;
* portfolio snapshot timestamp;
* portfolio base currency;
* ticker being reviewed;
* current position state for the ticker;
* current quantity, if held;
* current market value or equivalent explicit exposure value, if held;
* current portfolio total value or denominator used for exposure calculations;
* current ticker exposure percentage, if calculable;
* sector, industry, geography, asset-class, or other exposure buckets when explicitly available;
* configured concentration thresholds, if explicitly available;
* configured portfolio policy constraints, if explicitly available;
* missing portfolio-context fields;
* stale portfolio-context fields;
* context provenance.

Portfolio Review must not substitute missing portfolio context with zero.

Numeric zero remains valid when explicitly provided and source-grounded, for example `current_quantity = 0` or `current_ticker_exposure_pct = 0` for a non-held ticker.

## Position-state semantics

Approved position states:

| State | Meaning |
| --- | --- |
| `not_held` | The ticker is not currently held, and the zero exposure is explicit. |
| `held` | The ticker is currently held and position details are available. |
| `partially_known` | The ticker may be held or exposure is known only partially. |
| `unknown` | The portfolio context cannot determine whether the ticker is held. |
| `stale` | Position data exists but is older than the approved freshness threshold. |
| `invalid` | Position data is malformed or internally inconsistent. |

Position states are review context only.

They must not imply buy, sell, hold, reduce, increase, rebalance, trim, add, or execute.

## Portfolio Review output contract

Recommended output contract:

```text
sec-companyfacts-portfolio-review-v1
```

Recommended future persistence path:

```text
data/market_engine/portfolio_reviews/<portfolio_review_run_id>/<ticker>/portfolio_review.json
```

ME-PR01 does not create or write this path.

A future ME-PR02 implementation must use temporary directories in tests and must refuse overwrite by default when persistence is implemented.

## Required output structure

A future `sec-companyfacts-portfolio-review-v1` artifact should preserve:

* output contract version;
* portfolio review run identifier;
* ticker;
* created timestamp, when available;
* input Recommendation Review contract version;
* input Recommendation Review run identifier or artifact reference;
* input portfolio context contract version;
* input portfolio context run identifier or snapshot reference;
* portfolio review items;
* position state;
* exposure review result;
* concentration review result;
* portfolio fit review result;
* missing portfolio-context fields;
* stale portfolio-context fields;
* upstream Recommendation Review provenance;
* Setup Detection-aware provenance when present upstream;
* source and derived references when preserved upstream;
* non-actionable boundary metadata.

## Approved Portfolio Review categories

Initial approved categories:

| Category | Purpose |
| --- | --- |
| `position_context_review` | Reviews whether the ticker is currently held, not held, unknown, stale, or invalid in the supplied portfolio context. |
| `exposure_context_review` | Reviews explicit ticker exposure and whether exposure data is present, missing, stale, or invalid. |
| `concentration_context_review` | Reviews whether explicit exposure appears within configured concentration context when thresholds are supplied. |
| `portfolio_fit_context_review` | Reviews whether the Recommendation Review candidate can be assessed against portfolio policy context. |
| `portfolio_data_limitation_review` | Preserves missing, stale, partial, invalid, or unsupported portfolio context. |
| `downstream_handoff_readiness_review` | Reviews whether output is structurally ready for a later Decision Engine handoff contract without granting action authority. |
| `input_contract_invalid` | Represents invalid or unsupported Recommendation Review or portfolio context input. |

Categories are review categories only.

They must not become ranking categories, score categories, conviction categories, urgency categories, tradeability categories, or action categories.

## Approved Portfolio Review states

Initial approved states:

| State | Meaning |
| --- | --- |
| `portfolio_review_required` | Recommendation Review and portfolio context are structurally present and require portfolio-context review. |
| `portfolio_context_supported` | Required portfolio context is present enough for non-actionable portfolio review. |
| `portfolio_context_partial` | Portfolio context is present but incomplete. |
| `portfolio_context_missing` | Required portfolio context is absent. |
| `portfolio_context_stale` | Required portfolio context exists but is stale. |
| `portfolio_context_invalid` | Required portfolio context is malformed or internally inconsistent. |
| `position_already_held` | The ticker is explicitly held. |
| `position_not_held` | The ticker is explicitly not held. |
| `position_unknown` | The position state cannot be determined from approved context. |
| `exposure_known` | Current ticker exposure is explicitly available. |
| `exposure_missing` | Current ticker exposure is not available. |
| `concentration_within_context` | Exposure is within explicitly supplied concentration context. |
| `concentration_requires_review` | Exposure or threshold context requires human review. |
| `blocked_by_missing_portfolio_context` | Portfolio Review is blocked by missing portfolio context. |
| `blocked_by_invalid_input` | Portfolio Review is blocked by invalid input contract or malformed input. |
| `ready_for_decision_engine_handoff_review` | Output may be structurally reviewed by a later Decision Engine handoff contract, but no action authority is granted here. |
| `not_applicable` | Portfolio Review is not applicable to the supplied input. |

States are portfolio-review states only.

They must not imply BUY, SELL, HOLD, rebalance, add, trim, reduce, increase, execute, order, target weight, target price, ranking, score, conviction, urgency, or tradeability.

## Allowed review semantics

Allowed language:

* “Portfolio context is available for review.”
* “The ticker is already held according to approved portfolio context.”
* “The ticker is not held according to approved portfolio context.”
* “Portfolio context is partial and requires human review.”
* “Portfolio Review is blocked by missing portfolio context.”
* “Exposure data is present but remains non-actionable.”
* “Concentration context requires downstream review.”
* “Decision Engine authority is outside Portfolio Review.”

Allowed semantics:

* preserve portfolio context;
* preserve Recommendation Review provenance;
* preserve setup-aware provenance from Recommendation Review;
* identify missing or stale portfolio context;
* identify whether a position is held, not held, unknown, stale, or invalid;
* identify whether exposure data is known, missing, stale, or invalid;
* identify whether configured concentration context requires human review;
* prepare structurally valid non-actionable output for a later Decision Engine handoff contract.

## Forbidden semantics

Portfolio Review must not emit or imply:

* BUY;
* SELL;
* HOLD as a trading instruction;
* add;
* trim;
* reduce;
* increase;
* rebalance;
* open position;
* close position;
* enter position;
* exit position;
* execute;
* order generation;
* allocation execution;
* target weight;
* target price;
* position sizing instruction;
* conviction;
* urgency;
* score;
* ranking;
* rating;
* tradeability;
* delivery eligibility;
* Telegram eligibility;
* reporting eligibility;
* Decision Engine action.

Portfolio Review may preserve the word `held` only as a position-state descriptor such as `position_already_held` or `position_not_held`.

It must not use HOLD as an action recommendation.

## Missing-data rules

Missing Recommendation Review input must fail closed or produce invalid-input output.

Missing portfolio context must block Portfolio Review or produce explicit `blocked_by_missing_portfolio_context` output.

Missing position quantity, market value, total value, exposure percentage, sector, geography, threshold, or policy data must remain explicit.

Missing portfolio data must not be converted into zero.

Unknown position state must remain `position_unknown` or equivalent controlled output.

Partial portfolio data must remain partial and must not be promoted into complete context.

## Stale-data rules

Portfolio context must include a timestamp or equivalent freshness marker.

If freshness cannot be assessed, Portfolio Review must mark the context as partial, stale, or blocked according to the future implementation contract.

A future ME-PR02 implementation must define an explicit freshness threshold or require it as supplied context.

Stale portfolio context must not be used as if it were current.

Stale exposure must not generate allocation or execution guidance.

## Numeric-zero rules

Numeric zero remains present when explicitly supplied and source-grounded.

Examples:

* `current_quantity = 0` may support `position_not_held` when the portfolio context explicitly provides that value.
* `current_ticker_exposure_pct = 0` may support `exposure_known` for a non-held ticker when explicitly supplied.
* zero exposure must not be treated as missing.

Missing numeric values must not be coerced to zero.

## Authority boundary with Decision Engine

Portfolio Review is not the Decision Engine.

Portfolio Review may prepare non-actionable portfolio-context review output.

Only a later Decision Engine handoff contract may define how Portfolio Review output becomes eligible for Decision Engine evaluation.

Only the Decision Engine may later own action/allocation authority if explicitly implemented by approved future sprint scope.

ME-PR01 grants no authority to execute, allocate, rebalance, deliver, notify, or report.

## Relationship to Recommendation Review

Portfolio Review consumes Recommendation Review output as upstream evidence.

Portfolio Review must not change Recommendation Review states or categories.

Portfolio Review must not convert Recommendation Review human-review routing into action guidance.

Portfolio Review must preserve Recommendation Review missing-data markers, numeric-zero semantics, and setup-aware provenance.

## Relationship to Delivery / Reporting

Portfolio Review does not create delivery output.

Portfolio Review does not create reports.

Portfolio Review does not send Telegram messages.

Portfolio Review does not define user-facing delivery eligibility.

Delivery / Reporting remains a later job family and must not be introduced by ME-PR01 or ME-PR02 unless explicitly re-scoped.

## Relationship to portfolio mutation

Portfolio Review does not mutate portfolio state.

Portfolio Review does not write broker data.

Portfolio Review does not edit positions, cash, watchlists, targets, allocations, or holdings.

Portfolio Review does not reconcile broker statements.

A future portfolio-context source may be defined separately, but ME-PR01 does not authorize it.

## ME-PR02 implementation requirements

ME-PR02 may implement Portfolio Review only after this contract.

ME-PR02 must:

* consume only validated `sec-companyfacts-recommendation-review-v1` input;
* consume only explicitly supplied approved portfolio context;
* emit `sec-companyfacts-portfolio-review-v1` output;
* preserve Recommendation Review provenance;
* preserve Setup Detection-aware provenance when present upstream;
* preserve missing-data markers;
* preserve stale-data markers;
* preserve numeric-zero semantics;
* implement approved Portfolio Review categories;
* implement approved Portfolio Review states;
* fail closed for unsupported Recommendation Review contracts;
* fail closed or emit controlled limitation output for unsupported portfolio context contracts;
* add local synthetic tests only;
* avoid live provider calls;
* avoid broker calls;
* avoid production data writes;
* avoid portfolio mutation;
* avoid watchlist mutation;
* avoid Decision Engine calls;
* avoid Telegram, reporting, and delivery behavior;
* avoid legacy `scripts` or old `market_scanner` imports.

ME-PR02 must test:

* valid Recommendation Review plus held-position portfolio context;
* valid Recommendation Review plus non-held-position portfolio context;
* valid Recommendation Review plus partial portfolio context;
* valid Recommendation Review plus missing portfolio context;
* valid Recommendation Review plus stale portfolio context;
* valid Recommendation Review plus invalid portfolio context;
* unsupported Recommendation Review contract fails closed;
* unsupported portfolio context contract fails closed or emits controlled limitation output;
* numeric zero remains present and is not treated as missing;
* upstream Recommendation Review references are preserved;
* setup-aware provenance is preserved when present;
* missing and stale portfolio-context fields remain explicit;
* forbidden action-authority terms are not emitted as guidance;
* no legacy `scripts` or old `market_scanner` imports are introduced.

ME-PR02 must not introduce BUY / SELL / HOLD action semantics, allocation execution, target weights, order generation, position sizing instructions, portfolio mutation, watchlist mutation, broker integration, Telegram, reporting, delivery behavior, or Decision Engine behavior.

## Explicit non-scope

ME-PR01 does not introduce:

* Python code;
* tests;
* runtime behavior;
* provider calls;
* broker calls;
* data writes;
* generated artifacts;
* portfolio mutation;
* watchlist mutation;
* Decision Engine behavior;
* Delivery / Reporting behavior;
* Telegram behavior;
* BUY / SELL / HOLD action semantics;
* allocation execution;
* order generation;
* target weights;
* position sizing instructions;
* ranking;
* scoring;
* conviction;
* urgency;
* tradeability authority.

## Next sprint

Recommended next sprint:

```text
ME-PR02 — Implement Portfolio Review
```
