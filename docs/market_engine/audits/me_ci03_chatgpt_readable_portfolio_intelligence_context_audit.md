# ME-CI03 - ChatGPT-readable Portfolio Intelligence Context Audit

## Objective

ME-CI03 defines the formal `chatgpt_portfolio_intelligence_context` v1 contract:
a controlled portfolio subcontext that can be embedded in, or referenced beside,
`chatgpt-advisory-context-v1`.

The sprint answers how proven portfolio information may be selected,
normalized, and bounded for later ChatGPT advisory interpretation without
creating portfolio analysis, allocation, sizing, rebalancing, broker, prompt, or
runtime assembly behavior.

## Inspected upstream architecture

Inspected sources included:

* ME-CI01 Structured Decision Output contract;
* ME-CI02 ChatGPT Advisory Context contract, audit, and examples;
* ME-PR01 Portfolio Review contract;
* ME-PR02 Portfolio Review implementation documentation;
* `src/market_engine/portfolio_review/sec_companyfacts_portfolio_review.py`;
* `tests/market_engine/portfolio_review/test_sec_companyfacts_portfolio_review.py`;
* ME-DE01 Decision Engine handoff contract;
* ME-DE02 Decision Engine handoff implementation documentation and runtime;
* ME-GV06 Governor buy-zone and position-management explanation contract;
* ME-DS01 Dispatch Station Governor report output contract;
* ME-RUN18 local portfolio-context documentation;
* ME-RUN19 portfolio-context-aware dry-run documentation;
* ME-RUN24 non-production portfolio-context fixture documentation;
* `src/market_engine/run/local_portfolio_context_fixture.py`;
* `data/market_engine/portfolio_contexts/local_portfolio_context.json`;
* central roadmap and backlog ChatGPT advisory planning sections.

## Source-of-truth matrix

The canonical source-of-truth matrix is included in:

```text
docs/market_engine/contracts/me_ci03_chatgpt_readable_portfolio_intelligence_context_contract.md
```

Key conclusions:

* `market-engine-portfolio-context-v1` is the approved current portfolio context
  input family.
* `sec-companyfacts-portfolio-review-v1` is the canonical Portfolio Review
  output family.
* `market-engine-decision-engine-handoff-v1` is a readiness/provenance handoff,
  not a Decision Engine decision.
* ME-GV06 may supply Governor position-management explanation, but not sizing
  or execution authority.
* Legacy portfolio CSVs are not automatically canonical Market Engine advisory
  inputs for ME-CI03.

## Architecture position

ME-CI03 sits below the ChatGPT Advisory Layer and inside the ME-CI02 advisory
boundary.

It selects and structures proven portfolio context. It does not read broker
data, compute portfolio state, infer holdings, calculate exposure, produce cash
availability, optimize allocation, size positions, or rebalance.

## Relation to ME-CI02

ME-CI03 supports three modes relative to `chatgpt-advisory-context-v1`:

* `embedded_subcontext`;
* `referenced_companion_artifact`;
* `absent`.

Embedded summaries and referenced companion artifacts must not conflict. If they
do, the portfolio advisory context fails closed.

## Contract identity

Approved identity:

```text
contract_name: chatgpt_portfolio_intelligence_context
contract_version: v1
schema_version: chatgpt-portfolio-intelligence-context-v1
artifact_type: market-engine-chatgpt-portfolio-intelligence-context
```

## Holdings semantics

Holdings may be represented only when approved upstream context proves them.
`held`, `not_held`, `partially_known`, `unknown`, `stale`, and `invalid` retain
their Portfolio Review semantics.

Missing holdings are not converted to zero. A ticker absent from partial context
does not prove that the user owns no position.

## Position semantics

The contract separates:

* holding facts;
* position intelligence;
* Portfolio Review judgement;
* Governor position-management interpretation;
* later ChatGPT explanation.

These layers must not be collapsed into BUY, SELL, sizing, or allocation
guidance.

## Exposure semantics

Current v1 supports explicit single-name exposure and explicitly supplied
exposure buckets only. ME-CI03 does not calculate sector, theme, geography,
currency, factor, or asset-class exposure when those families are not already
canonical upstream.

## Concentration semantics

Concentration may be represented as measured concentration, classified
concentration, warning, or advisory explanation. The contract preserves
Portfolio Review concentration states such as `concentration_within_context` and
`concentration_requires_review`.

ChatGPT must not independently determine whether adding a candidate creates
excessive concentration.

## Cash semantics

No canonical ME-CI03 cash source exists in the current Market Engine advisory
chain. Cash is therefore represented as unknown, not provided, or unavailable
unless a future approved source supplies amount, currency, timestamp,
provenance, and deployability semantics.

Cash balance is not deployable cash. Deployable cash is not position size.

## Allocation semantics

Current exposure may be explained when proven. Policy constraints may be
explained when supplied. Target allocation, recommended allocation, target
weight, and position sizing remain unavailable unless a future approved Decision
Engine or sizing contract supplies them.

## Constraint semantics

Constraints may be included only when upstream context provides them as policy
constraints, concentration thresholds, Governor limitations, or review blockers.
ME-CI03 does not invent thresholds.

## Portfolio-fit semantics

ME-CI03 consumes Portfolio Review states but does not create a new investment
taxonomy. Portfolio fit is not recommendation. Portfolio fit is not position
sizing. Position sizing is not allocation authority.

## Recommendation-to-position relationship

The contract safely expresses whether a recommendation relates to a proven held
instrument, proven non-held instrument, partial held state, unknown relation, or
blocked relation conflict. It does not remake the recommendation.

## Missingness semantics

The contract defines explicit missingness states:

```text
known_absent
unknown
not_applicable
not_provided
stale
conflicting
blocked
partial
unsupported
```

Unknown remains unknown. Not provided remains not provided. Conflicting context
blocks portfolio-specific interpretation.

## Provenance

Every material portfolio claim requires lineage to portfolio snapshot, holdings,
valuation, cash, exposure, concentration, Portfolio Review, Governor context,
position-management explanation, or handoff artifact.

Free-floating portfolio facts are invalid.

## Freshness

Freshness is per family, not one global timestamp:

* holdings freshness;
* valuation freshness;
* cash freshness;
* Portfolio Review freshness;
* Governor interpretation freshness;
* exposure freshness;
* concentration freshness.

Fresh context generation does not refresh stale portfolio evidence.

## Advisory permission boundary

ChatGPT may explain proven portfolio context and conditionally interpret
recommendation-to-position relationship when upstream context supports it.

ChatGPT may not determine target allocation, target weight, max weight unless
upstream-proven, order size, shares to buy, cash to deploy, rebalance
instruction, tax strategy, broker action, synthetic stop, synthetic target
price, unsupported sell urgency, or unsupported add urgency.

## Prohibited inference matrix

ME-CI03 prohibits inference of:

* holdings from absent rows;
* zero cash from unavailable cash;
* not-held state from partial context;
* deployable cash from cash balance;
* target weight from current weight;
* max weight from concentration thresholds unless explicitly defined;
* position size from cash or recommendation;
* rebalancing action from concentration warning;
* sell candidates from current holdings;
* tax strategy from unrealized result;
* portfolio fit from standalone recommendation;
* Decision Engine approval from handoff readiness.

## Fail-closed matrix

The contract distinguishes:

* `context_invalid`;
* `portfolio_advisory_blocked`;
* `portfolio_specific_advice_unavailable`;
* `descriptive_only_allowed`;
* `eligible_with_mandatory_caveat`.

Blocking cases include missing or unsupported schema, missing identity,
conflicting snapshots, missing holdings provenance, stale holdings for
ownership-sensitive use cases, missing Portfolio Review for portfolio-fit
answers, blocked Portfolio Review, malformed Governor position context,
unsupported allocation fields, and unsupported sizing fields.

## Use-case matrix

The contract covers questions about adding to an owned position, checking
whether a candidate fits beside existing holdings, exact sizing, selling to make
room, and investing cash. Exact sizing, selling to make room, and investing
cash generally require downstream authority beyond ME-CI03.

## Examples

Synthetic examples were added:

* complete portfolio intelligence context;
* partial context;
* blocked context;
* held instrument with sizing unavailable.

They are shape examples only and make no production market claim.

## Implementation decision

ME-CI03 remains docs-only.

Rationale:

* ME-CI03 is contract-first in the ME-RM06 / ME-CI chain.
* Existing runtime already defines Portfolio Review and handoff contracts.
* No ChatGPT advisory context assembler exists yet.
* A typed schema, validator, deterministic assembler, and prompt contract should
  be separate explicit sprints.
* Runtime work now would risk premature coupling to non-production fixtures or
  legacy portfolio CSVs.

## Validation

Validation results:

```text
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_portfolio_intelligence_context_v1_complete.json
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_portfolio_intelligence_context_v1_partial.json
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_portfolio_intelligence_context_v1_blocked.json
PASS - .venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_portfolio_intelligence_context_v1_held_sizing_unavailable.json
PASS - git diff --check
PASS - git status --short
PASS - git diff --stat
REVIEWED - rg -n "OpenAI|ChatGPT|Telegram|portfolio write|watchlist write|broker|allocation|position size|target weight|max weight|yfinance|SEC|EDGAR" src/market_engine scripts tests/market_engine docs/market_engine
PASS - PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q (1394 passed)
REVIEWED - grep -R "BUY" scripts/ | grep -v decision_engine.py
REVIEWED - grep -R "SELL" scripts/ | grep -v decision_engine.py
REVIEWED - grep -R "tradeable" scripts/ | grep -v decision_engine.py
```

The governance scan and mandatory greps returned expected existing boundary,
test, legacy portfolio, and pycache hits. ME-CI03 introduced documentation and
JSON examples only; it added no runtime behavior.

## Residual gaps

* No typed JSON schema or validator exists for ME-CI03.
* No deterministic ChatGPT advisory context assembler exists.
* No runtime Portfolio Intelligence artifact writer exists.
* No OpenAI API, prompt contract, notification adapter, dashboard, Telegram, or
  delivery integration exists.
* Cash remains unavailable in the current approved ME-CI03 source set.
* Multi-holding sector, currency, thematic, factor, and correlation exposure
  require future explicit Portfolio Intelligence contracts.

## Recommended next sprint

Recommended next sprint:

```text
ME-CI04 - Define explainability/change-rationale contract
```

Recommended later Portfolio Intelligence sprint:

```text
ME-PI01 - Define Portfolio Intelligence exposure contract
```
