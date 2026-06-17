# ME-DL01 - Delivery / Reporting contract

## Status

COMPLETED BY ME-DL01

## Sprint

ME-DL01 - Define Delivery / Reporting contract

## Job family

ME-DL - Delivery / Reporting jobs

## Purpose

ME-DL01 defines the canonical Delivery / Reporting contract downstream of controlled Decision Engine handoff.

Delivery / Reporting is the final user-facing presentation layer. It may summarize, explain, and surface upstream review output in a safe, provenance-preserving way.

Delivery / Reporting must not create new investment decisions, execute or prepare trades, route orders, connect to brokers, send notifications, mutate portfolio/watchlist state, or become a hidden Decision Engine.

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
-> Delivery / Reporting
```

Decision Engine remains the only action and allocation authority.

Delivery / Reporting may consume only an approved Decision Engine handoff payload. It must not bypass Decision Engine handoff by reading Portfolio Review, Recommendation Review, Analysis Review, Setup Detection, observations, source contexts, raw provider payloads, legacy reports, watchlists, or old runtime output directly.

## Legacy reporting and Telegram reference status

Legacy reporting, Telegram, scanner-output, and watchlist-output files remain reference material only.

Examples inspected during ME-DL01 included archived and legacy reporting/Telegram areas such as:

* `archive/legacy_runtime/scripts/reporting/`
* `archive/legacy_runtime/scripts/telegram/`
* `docs/archive/market_scanner_reference/active/reporting/`
* `docs/archive/market_scanner_reference/archive/execution/`
* `docs/archive/market_scanner_reference/resets/`
* `src/market_scanner/delivery/`
* `src/market_scanner/reporting/`

These areas are not canonical Market Engine Delivery / Reporting behavior.

ME-DL02 must not revive legacy Telegram or reporting behavior as active Market Engine behavior unless a future sprint explicitly approves a safe channel contract and implementation.

## Approved input

The only approved primary input for ME-DL01 is:

```text
market-engine-decision-engine-handoff-v1
```

The input must be produced by ME-DE02-compatible behavior.

The input must preserve:

* handoff format identity and version;
* handoff run identifier;
* handoff readiness state;
* ticker and entity identifiers where available;
* Portfolio Review reference;
* portfolio-context reference;
* Recommendation Review reference;
* Analysis Review reference when available;
* Setup Detection reference when available;
* source context / observation lineage when available;
* missing-data markers;
* stale-data markers;
* blocked reasons;
* numeric-zero evidence;
* authority boundary text confirming no action/allocation authority.

## Input handling

Delivery / Reporting must fail closed or emit a blocked report payload when the upstream handoff input is:

* missing;
* malformed;
* unsupported;
* from an unsupported contract version;
* stale;
* blocked upstream;
* missing required provenance;
* missing required ticker/entity identity;
* missing required blocked reasons for a blocked state;
* missing required handoff readiness state;
* inconsistent with the expected non-actionable authority boundary.

Blocked upstream state must remain blocked downstream.

Delivery / Reporting must not convert blocked, incomplete, stale, unsupported, or malformed input into user-facing action language.

## Required lineage

When present upstream, the report payload must preserve lineage from:

* SEC CompanyFacts or source-facts layer;
* Source Context;
* Fundamental Observations;
* Derived Observations;
* Setup Detection;
* Analysis Review;
* Recommendation Review;
* Portfolio Review;
* Decision Engine handoff.

Delivery / Reporting must not invent lineage.

If lineage is unavailable, the report must explicitly state that lineage is missing or unavailable.

## Output contract

Future ME-DL02 implementation should produce:

```text
market-engine-delivery-report-v1
```

The payload should include:

* delivery/report format identity and version;
* report identifier;
* generated timestamp;
* source handoff format/version;
* source handoff run identifier;
* ticker and entity identifiers where available;
* delivery state;
* display sections;
* blocked or unavailable reasons;
* upstream provenance summary;
* missing-data summary;
* stale-data summary;
* numeric-zero-safe evidence references;
* allowed user-facing language category;
* forbidden-language guardrail confirmation;
* audit metadata;
* non-execution boundary statement.

The output is a presentation artifact only.

## Delivery states

Approved delivery states:

| State | Meaning |
| --- | --- |
| `ready_for_user_review` | The approved handoff is ready to be presented for human review. This is not an action instruction. |
| `blocked_upstream` | The upstream handoff is blocked and must remain blocked in the report. |
| `insufficient_data` | Required evidence or provenance is incomplete. |
| `stale_data` | Upstream handoff or lineage contains stale-data markers. |
| `unsupported_input` | Input contract/version is unsupported. |
| `contract_violation` | Input violates the Delivery / Reporting contract or includes prohibited action semantics. |

These states are presentation states only.

## Allowed reporting categories

Delivery / Reporting may emit only non-actionable reporting categories:

* factual summary;
* evidence summary;
* upstream review summary;
* portfolio-context summary;
* risk / limitation summary;
* missing-data summary;
* stale-data summary;
* requires-human-review note.

Allowed categories must preserve upstream uncertainty and must not add new investment conclusions.

## Forbidden reporting behavior

Delivery / Reporting must not emit or imply:

* buy instruction;
* sell instruction;
* hold instruction;
* position sizing;
* allocation advice;
* target price;
* urgency label;
* conviction label or score;
* ranking;
* best-pick language;
* broker-ready instruction;
* order instruction;
* execution instruction;
* notification text that implies immediate action;
* generated trade ticket;
* portfolio mutation;
* watchlist mutation.

Forbidden concepts may appear in contract guardrails, tests, and audit documents only as prohibited behavior.

## Presentation rules

Reports must phrase upstream states carefully.

### Blocked states

Blocked upstream handoff states must be reported as blocked.

The report may say:

```text
The upstream handoff is blocked and cannot be presented as actionable output.
```

The report must not soften a blocked state into a suggestion.

### Incomplete data

Incomplete data must remain explicit.

The report may summarize which required evidence is unavailable.

The report must not fill missing evidence from old reports, generated files, watchlists, provider calls, or assumptions.

### Stale data

Stale data must be displayed as stale.

The report must not describe stale evidence as current.

### Numeric zero

Valid numeric zero values must be preserved.

Examples include zero quantity, zero market value, zero exposure, zero cash, or zero weight when explicitly supplied upstream.

Zero must not be treated as missing.

### Missing facts

Missing facts must be shown as missing or unavailable.

Missing source facts must not be converted into zero, estimated values, inferred values, or prior-period substitutes.

### Upstream disagreement

Conflicts, limitations, or disagreement in upstream review lineage must remain visible.

The report may describe conflict as a limitation requiring human review.

The report must not collapse conflict into a definitive instruction.

### Unavailable portfolio context

Unavailable portfolio context must remain unavailable.

The report may state that portfolio context is missing, partial, stale, or blocked according to upstream handoff evidence.

### Unsupported ticker or entity input

Unsupported entity input must be reported as unsupported.

The report must not substitute a nearby ticker or inferred entity.

### Provider or provenance gaps

Provider gaps, missing provenance, or source lineage gaps must be preserved as limitations.

The report must not hide gaps behind clean summary language.

## Safety and governance

Delivery / Reporting is:

* non-executing;
* non-broker-connected;
* not authorized to send notifications unless a future sprint explicitly implements a safe channel;
* not authorized to schedule delivery;
* not authorized to prompt the user toward action;
* not authorized to transform review evidence into advice.

Delivery / Reporting must preserve:

* upstream blocked states;
* upstream missing-data markers;
* upstream stale-data markers;
* upstream provenance;
* numeric-zero semantics;
* non-action/allocation boundaries.

## Future implementation requirements

Recommended next sprint:

```text
ME-DL02 - Implement Delivery / Reporting contract
```

ME-DL02 must:

* consume only approved `market-engine-decision-engine-handoff-v1` payloads;
* emit `market-engine-delivery-report-v1`;
* preserve blocked upstream state as blocked;
* preserve missing-data markers;
* preserve stale-data markers;
* preserve numeric-zero semantics;
* preserve upstream provenance;
* emit only allowed non-actionable reporting categories;
* include local synthetic tests only;
* test ready, blocked, insufficient-data, stale-data, unsupported-input, malformed-input, and contract-violation cases;
* test that forbidden language is rejected or not emitted;
* avoid provider calls;
* avoid live market data calls;
* avoid Telegram/email/broker writes;
* avoid portfolio/watchlist writes;
* avoid ranking, conviction, urgency, target-price, buy/sell/hold, allocation, or execution semantics.

## Explicit non-goals

ME-DL01 does not implement:

* Python runtime code;
* tests;
* Telegram delivery;
* email delivery;
* Streamlit UI;
* broker integration;
* portfolio writes;
* watchlist writes;
* scheduler, cron, or automation;
* live provider fetches;
* new financial analysis logic;
* Decision Engine logic;
* Recommendation Review logic;
* Portfolio Review logic;
* user-facing report generation;
* notification delivery;
* report persistence.

## Acceptance criteria for ME-DL02

ME-DL02 is complete only when:

* the builder/parser accepts only `market-engine-decision-engine-handoff-v1`;
* blocked upstream state remains blocked;
* output contract is `market-engine-delivery-report-v1`;
* required provenance is preserved;
* missing/stale markers are preserved;
* numeric zero is preserved;
* forbidden language is not emitted;
* tests are synthetic and provider-free;
* no Telegram, email, broker, portfolio, watchlist, reporting delivery, or execution side effects are introduced.

## Next sprint

```text
ME-DL02 - Implement Delivery / Reporting contract
```
