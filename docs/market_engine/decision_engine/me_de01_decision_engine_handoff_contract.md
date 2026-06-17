# ME-DE01 - Decision Engine handoff contract

## Status

COMPLETED BY ME-DE01

## Sprint

ME-DE01 - Define Decision Engine handoff contract

## Job family

ME-DE - Decision Engine handoff jobs

## Purpose

ME-DE01 defines the formal handoff contract between Market Engine Portfolio Review and a future Decision Engine layer.

The contract allows Market Engine to prepare a controlled, auditable handoff payload from approved Portfolio Review output. It does not allow Market Engine to make action, allocation, ranking, scoring, order, execution, delivery, or broker-facing decisions.

Decision Engine remains the only future action and allocation authority.

## Architectural position

Approved chain:

```text
Source Refresh / raw snapshots
-> Source Context
-> Fundamental Observations
-> Derived Observations
-> Setup Detection
-> Analysis Review
-> Recommendation Review
-> Portfolio Review
-> Decision Engine handoff / action authority
-> Delivery / reporting
```

Decision Engine handoff is a boundary layer between Portfolio Review and future Decision Engine authority.

The handoff must not bypass Portfolio Review. Direct handoff from Recommendation Review, Analysis Review, Setup Detection, Fundamental Observations, Derived Observations, Source Context, Source Refresh, raw provider payloads, reports, watchlists, or portfolio files is not approved by this contract.

## Job-family boundary

Market Engine may prepare a controlled Decision Engine handoff payload only from approved Portfolio Review output.

Market Engine may pass review states, portfolio-context summaries, upstream provenance, limitations, missing-data markers, stale-data markers, and non-actionable review categories.

Market Engine must not decide:

* action outcome;
* allocation;
* target weight;
* rebalance instruction;
* order generation;
* execution instruction;
* broker instruction;
* ranking;
* score;
* urgency;
* conviction;
* tradeability.

## Approved upstream input

The only approved upstream input for the first Decision Engine handoff contract is:

```text
sec-companyfacts-portfolio-review-v1
```

The Portfolio Review input must be produced by ME-PR02-compatible behavior.

The Portfolio Review input must include explicit `market-engine-portfolio-context-v1` context or a controlled missing/blocked portfolio-context state.

The Portfolio Review input must preserve:

* Portfolio Review contract identity and version;
* Portfolio Review run identifier;
* ticker identity;
* CIK and provider identity where available;
* Portfolio Review state and category;
* Portfolio Review items;
* portfolio-context state summary;
* portfolio-context provenance;
* Recommendation Review provenance;
* Setup Detection-aware provenance when present upstream;
* source and derived observation lineage when present upstream;
* missing portfolio-context markers;
* stale portfolio-context markers;
* numeric-zero values when explicitly supplied.

## Portfolio Review handoff eligibility

Portfolio Review output is eligible for Decision Engine handoff preparation only when all of the following are true:

* the Portfolio Review artifact is present;
* the Portfolio Review format is `sec-companyfacts-portfolio-review-v1`;
* the ticker identity is present and verified against portfolio context where available;
* the Portfolio Review is not invalid, missing, stale, or blocked;
* the portfolio context contract is `market-engine-portfolio-context-v1`;
* portfolio context is not missing or stale;
* Recommendation Review provenance is present;
* upstream provenance required for traceability is present;
* the Portfolio Review includes a `downstream_handoff_readiness_review` item with state `ready_for_decision_engine_handoff_review`;
* no explicit data limitation blocks downstream interpretation.

The handoff contract may still create a blocked handoff payload for auditability when eligibility is not met, but that payload must not be marked ready.

## Blocking conditions

Decision Engine handoff must block when any of the following conditions applies:

* Portfolio Review is missing.
* Portfolio Review contract or version is invalid.
* Portfolio Review is stale when staleness metadata requires blocking.
* Portfolio Review state or category is invalid or unsupported.
* Portfolio Review does not include a downstream handoff readiness item.
* Portfolio Review is not ready for Decision Engine handoff review.
* Portfolio context is missing.
* Portfolio context contract or version is invalid.
* Portfolio context is stale.
* Portfolio context is malformed or internally inconsistent.
* Upstream Recommendation Review is missing.
* Upstream Recommendation Review is not approved or not reviewable.
* Required provenance is missing or incomplete.
* Ticker identity cannot be verified.
* Ticker identity conflicts across Portfolio Review and portfolio context.
* Data limitations prevent downstream interpretation.
* Unsupported future contract versions are supplied.

Blocking must be explicit and auditable. Blocking must not be converted into a default action, default allocation, default ranking, or default recommendation.

## Handoff payload contract

Future ME-DE02 implementation should produce:

```text
market-engine-decision-engine-handoff-v1
```

The payload should include:

* handoff format identity and version;
* ticker;
* CIK, when available;
* provider name, when available;
* handoff run identifier;
* handoff creation timestamp, when available;
* Portfolio Review reference;
* Portfolio Review format version;
* Portfolio Review run identifier;
* Portfolio Review state summary;
* Portfolio Review category summary;
* Portfolio Review item references;
* portfolio-context format version;
* portfolio-context run identifier or snapshot identifier;
* portfolio-context state summary;
* portfolio-context missing-data markers;
* portfolio-context stale-data markers;
* upstream Recommendation Review reference;
* upstream Analysis Review reference when available through provenance;
* Setup Detection reference and state when available through provenance;
* source context and observation lineage references when available through provenance;
* missing-data markers;
* stale-data markers;
* numeric-zero preservation notes or evidence references;
* handoff readiness state;
* non-action/allocation boundary statement;
* audit and provenance block.

The handoff payload is a readiness and provenance artifact only.

## Approved handoff readiness states

Approved ME-DE01 handoff readiness states:

| State | Meaning |
| --- | --- |
| `ready_for_decision_engine_review` | Portfolio Review is structurally ready for a future Decision Engine to evaluate. This is not an action decision. |
| `blocked_missing_portfolio_review` | No Portfolio Review artifact was supplied. |
| `blocked_invalid_portfolio_review_contract` | Portfolio Review contract or version is missing, malformed, unsupported, or incompatible. |
| `blocked_unapproved_portfolio_review` | Portfolio Review is not in an approved or reviewable state for handoff. |
| `blocked_missing_portfolio_context` | Required portfolio context is absent. |
| `blocked_stale_portfolio_context` | Portfolio context is stale according to supplied metadata. |
| `blocked_incomplete_provenance` | Required Portfolio Review, Recommendation Review, portfolio-context, or upstream lineage provenance is incomplete. |
| `blocked_ticker_mismatch` | Ticker identity cannot be verified or conflicts across inputs. |
| `blocked_insufficient_evidence` | Data limitations or missing evidence prevent downstream interpretation. |
| `not_applicable` | Handoff does not apply to the supplied input. |

These are handoff-readiness states only. They are not Decision Engine decisions.

## Prohibited payload fields

The handoff payload may not contain fields or values that provide:

* BUY instruction;
* SELL instruction;
* HOLD instruction;
* recommendation action;
* allocation percentage;
* target weight;
* rebalance instruction;
* order quantity;
* order type;
* entry price;
* exit price;
* stop loss;
* take profit;
* urgency;
* conviction;
* tradeability;
* ranking;
* score;
* execution instruction;
* broker instruction;
* Telegram instruction;
* reporting instruction;
* delivery instruction.

If a future implementation uses machine-readable boundary metadata, it must use non-actionable names such as `handoff_readiness_state`, `boundary_notes`, `blocked_reasons`, `provenance`, and `limitations`.

## Market Engine request boundary

Market Engine may ask a future Decision Engine to evaluate an approved handoff payload.

Market Engine may pass:

* Portfolio Review states;
* portfolio-context summaries;
* Recommendation Review provenance;
* Setup Detection-aware provenance when present;
* upstream source and observation lineage;
* non-actionable review categories;
* limitations;
* missing-data markers;
* stale-data markers;
* numeric-zero evidence.

Market Engine may not pre-fill, imply, suggest, or default the action outcome.

## Decision Engine authority

Only a future approved Decision Engine contract and runtime may decide:

* whether an actionable decision exists;
* BUY, SELL, HOLD, or equivalent action semantics;
* allocation;
* target weight;
* position sizing;
* order generation;
* urgency, conviction, or tradeability if those concepts are explicitly approved;
* execution or broker-facing instructions;
* delivery or reporting of actionable output if later approved by the Delivery / Reporting contract.

No upstream Market Engine layer may assume or simulate these decisions.

## Fail-closed rules

ME-DE02 must fail closed or emit a blocked handoff payload when:

* Portfolio Review is missing;
* Portfolio Review contract or version is invalid;
* Portfolio Review is stale and staleness metadata requires blocking;
* Portfolio Review state is invalid or unsupported;
* Portfolio Review lacks handoff-readiness evidence;
* portfolio context is missing;
* portfolio context is stale;
* portfolio context contract or version is invalid;
* ticker identity is missing or mismatched;
* Recommendation Review provenance is missing;
* required upstream provenance is missing;
* numeric fields required for context interpretation are missing;
* required structures are empty or malformed;
* unsupported future contract versions are supplied.

Fail-closed output must preserve the reason for blocking.

## Numeric-zero safety

Valid numeric zero values must not be treated as missing.

Examples of valid zero values include:

* `0`;
* `0.0`;
* zero position quantity;
* zero market value;
* zero cash;
* zero exposure;
* zero weight.

Missingness must be represented explicitly with `None`, missing markers, empty required structures, or documented missing-data states.

Handoff logic must not infer missingness from falsy numeric values.

## Audit and traceability requirements

Every handoff payload must preserve:

* Portfolio Review provenance;
* portfolio-context provenance;
* Recommendation Review provenance;
* Setup Detection-aware provenance when present upstream;
* Analysis Review references when present upstream;
* source context and observation lineage when present upstream;
* missing-data markers;
* stale-data markers;
* blocked-state reasons;
* boundary metadata confirming no action/allocation authority.

The payload must be explainable without calling providers, reading old reports, reading generated data folders, mutating portfolio files, or invoking the Decision Engine.

## Future persistence expectations

ME-DE01 does not implement persistence.

If ME-DE02 implements persistence, the approved candidate path is:

```text
data/market_engine/decision_engine_handoffs/<run_id>/<ticker>/decision_engine_handoff.json
```

Persistence must:

* write JSON only;
* refuse overwrite by default;
* use temporary directories in tests;
* avoid old data, CSV, portfolio, watchlist, report, Telegram, and delivery paths;
* preserve the full handoff payload and provenance;
* not treat persisted handoff payloads as action instructions.

## ME-DE02 implementation requirements

ME-DE02 should implement controlled handoff behavior from:

```text
sec-companyfacts-portfolio-review-v1
```

to:

```text
market-engine-decision-engine-handoff-v1
```

ME-DE02 must:

* validate Portfolio Review contract and version;
* validate ticker identity;
* validate portfolio-context version and state;
* validate Portfolio Review handoff-readiness evidence;
* validate Recommendation Review provenance;
* preserve Setup Detection-aware provenance when present;
* preserve missing-data markers;
* preserve stale-data markers;
* preserve numeric-zero semantics;
* produce only a handoff-readiness payload;
* avoid Decision Engine decisions;
* avoid action, allocation, ranking, scoring, execution, delivery, Telegram, and reporting behavior;
* include local synthetic tests only;
* avoid live provider calls;
* optionally persist JSON under the approved dedicated handoff path with overwrite protection.

## Non-scope

ME-DE01 does not implement:

* Python runtime code;
* tests;
* provider calls;
* broker calls;
* data writes;
* portfolio or watchlist mutation;
* Decision Engine runtime behavior;
* Delivery / Reporting behavior;
* Telegram behavior;
* action recommendations;
* allocation logic;
* order generation;
* execution instructions.

## Next sprint

Recommended next sprint:

```text
ME-DE02 - Implement controlled Decision Engine handoff
```
