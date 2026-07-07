# ME-CI02 - ChatGPT Advisory Context Contract Audit

## Purpose

ME-CI02 defines the formal ChatGPT Advisory Context Contract: the controlled,
evidence-backed context envelope that Market Engine may offer to an external
ChatGPT Advisory Layer.

The sprint establishes one architectural boundary:

```text
ChatGPT may advise only from controlled, traceable, readiness-aware context.
```

## Inspected upstream contracts

ME-CI02 was composed from existing contracts and planning artifacts, including:

* ME-RM06 ChatGPT Advisory delivery reposition;
* ME-CI01 Structured Decision Output contract;
* ME-DS01 Dispatch Station Governor report output contract;
* ME-GV06 Governor buy-zone and position-management explanation contract;
* ME-DE01 / ME-DE02 Decision Engine handoff contracts;
* Recommendation Review contracts and audits;
* Portfolio Review contracts and audits;
* ME-SA12 readiness and coverage semantics;
* Delivery / Reporting contracts.

## Architectural boundary

ChatGPT is not a Market Engine stage, source acquisition layer, analysis engine,
Setup Detection engine, Recommendation Review replacement, Portfolio Review
replacement, Decision Engine replacement, Governor authority, or production
action override path.

ChatGPT is an advisory interpretation layer above controlled Market Engine
artifacts. It may explain upstream evidence, governance, and decision semantics.
It must not expand, replace, or override them.

## Context contract

Approved identity:

```text
contract_name: chatgpt_advisory_context
contract_version: v1
schema_version: chatgpt-advisory-context-v1
artifact_type: market-engine-chatgpt-advisory-context
```

## Field families

The contract defines required field families for:

* contract identity;
* advisory eligibility;
* Structured Decision Output consumption;
* Governor context;
* Dispatch Station context;
* provenance context;
* freshness context;
* uncertainty context;
* portfolio context boundary;
* recommendation boundary;
* human advisory guidance;
* prohibited inputs;
* prohibited inferences;
* fail-closed state;
* validation.

## Provenance semantics

The context must show which evidence families are present, missing, referenced,
or lineage-linked. It may not include raw provider payloads unless a later
contract explicitly approves them.

ChatGPT must not use unreferenced values as source-of-truth facts.

## Freshness semantics

Freshness must be represented per evidence family. A global timestamp is not
enough when source, Governor, portfolio, price/setup, or Dispatch Station
artifacts have different freshness states.

Unknown freshness is not fresh. Stale critical evidence must downgrade advisory
eligibility to descriptive-only or blocked according to the contract matrix.

## Uncertainty semantics

The context must preserve confidence, uncertainty, missing evidence,
contradictory evidence, unresolved blockers, assumptions, and limitations.

Machine-readable reason codes are preferred. Free text may explain but must not
be the only representation of a material blocker or limitation.

## Eligibility and readiness semantics

ME-CI02 reuses existing readiness concepts and does not create a parallel
investment taxonomy.

Allowed advisory eligibility states:

```text
eligible
descriptive_only
blocked
```

`eligible` means ChatGPT may interpret approved context within the declared
scope. It does not mean order-ready, broker-ready, allocation-ready, or
Decision Engine override-ready.

## Governor integration boundary

Governor context may be explained by ChatGPT, including readiness, blockers,
warnings, policy constraints, non-actionability reasons, buy-zone explanation,
and position-management explanation.

ChatGPT must not ignore or override Governor governance signals. Fixed-false
authority fields remain fixed-false.

## Dispatch Station integration boundary

Dispatch Station output may be used as:

* artifact reference only;
* selected summary context;
* not included.

It is presentation context, not a second source of decision truth. Conflicts
between Dispatch Station output and Structured Decision Output must block or be
explicitly marked non-consumable.

## ME-CI01 Structured Decision Output consumption

Structured Decision Output v1 is the canonical upstream decision-facing input.

ME-CI02 may embed compact fields or reference the full artifact, but it may not
change the meaning of ME-CI01 fields. `semantic_override_allowed` is always
false.

## Portfolio context boundary

Portfolio context may be included only when upstream approved contracts prove
it. Missing portfolio context remains missing.

ChatGPT must not invent holdings, exposure, cash, current weight, target weight,
max weight, position size, allocation, or portfolio fit.

Without portfolio context, ChatGPT may explain non-portfolio evidence and
blockers, but it must not answer portfolio-sizing or add/reduce questions as if
holdings were known.

## Prohibited inputs

ME-CI02 prohibits:

* raw unvalidated web snippets;
* unprovenanced analyst opinions;
* stale snapshots without stale markers;
* current price without timestamp or evidence reference;
* portfolio holdings without run/source identity;
* unsupported target price;
* unsupported stop-loss;
* unsupported probability;
* broker secrets, credentials, or production action handles;
* prompt-only instructions that bypass structured fields.

## Prohibited inferences

ChatGPT must not infer:

* missing facts;
* current freshness from old timestamps;
* actionability from descriptive-only output;
* readiness by ignoring blockers;
* portfolio state from ticker ownership assumptions;
* price, target, stop-loss, or position size from incomplete context;
* certainty from incomplete evidence;
* recommendation from company profile alone;
* fundamentals by extrapolating missing values;
* allocation or order instructions from candidate states;
* Decision Engine approval from handoff readiness.

## Fail-closed matrix

ME-CI02 defines fail-closed behavior for missing schema versions, missing ticker
identity, missing run identity, missing provenance, unknown freshness, stale
critical evidence, missing Structured Decision Output, malformed Governor
output, conflicting readiness states, blocked Recommendation Review, missing
portfolio context, and incomplete Dispatch Station report.

Not every missing field has the same result. The contract distinguishes:

* context invalid;
* advisory blocked;
* descriptive-only allowed;
* advisory allowed with mandatory caveat.

## Examples

Synthetic examples were added for:

* advisory eligible;
* descriptive only;
* blocked.

They are contract-shape examples only and make no production market claim.

## Implementation decision

ME-CI02 remains docs-only.

Rationale:

* ME-RM06 and ME-CI01 position CI02 as a contract definition before runtime
  context assembly.
* No ChatGPT consumer, prompt runner, or advisory context runtime exists yet.
* A schema/validator implementation belongs in a later explicitly scoped sprint.

## Tests and validation

Validation results:

```text
PASS - git diff --check
PASS - git status --short
PASS - git diff --stat
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_advisory_context_v1_eligible.json
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_advisory_context_v1_descriptive_only.json
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_advisory_context_v1_blocked.json
REVIEWED - rg -n "OpenAI|ChatGPT|Telegram|portfolio write|watchlist write|provider|yfinance|SEC|EDGAR" src/market_engine scripts tests/market_engine docs/market_engine
PASS - .venv/bin/python -m pytest (1394 passed)
REVIEWED - grep -R "BUY" scripts/ | grep -v decision_engine.py
REVIEWED - grep -R "SELL" scripts/ | grep -v decision_engine.py
REVIEWED - grep -R "tradeable" scripts/ | grep -v decision_engine.py
```

Runtime tests are not required because ME-CI02 changes no runtime code.

The governance boundary scan produced expected existing documentation, test, and
source-acquisition references plus the new ME-CI02 docs. No runtime, provider,
ChatGPT API, Telegram, portfolio/write, watchlist/write, or Decision Engine
behavior was introduced.

The script grep hits were pre-existing legacy portfolio and `__pycache__`
matches. ME-CI02 changed documentation and JSON examples only.

## Residual gaps

* No runtime advisory-context assembler exists yet.
* No typed schema/validator exists yet.
* No ChatGPT prompt contract exists yet.
* No OpenAI API, LLM runtime, notification, dashboard, or delivery integration
  is approved.

## Recommended next sprint boundary

Recommended next sprint:

```text
ME-CI03 - ChatGPT-readable Portfolio Intelligence context
```

ME-CI03 should remain contract-first and must preserve the ME-CI02 rule that
missing portfolio context cannot be invented by ChatGPT.
