# ME-CI03 - ChatGPT-readable Portfolio Intelligence Context Backlog Entry

## Status

COMPLETED DOCS-ONLY CONTRACT

## Goal

Define `chatgpt-portfolio-intelligence-context-v1`, the controlled,
ChatGPT-readable portfolio subcontext for proven portfolio state, Portfolio
Review output, Governor position-management interpretation, provenance,
freshness, missingness, and advisory permission boundaries.

## Scope

* contract documentation;
* source-of-truth matrix;
* ME-CI02 inclusion/reference/absent modes;
* holdings, position, exposure, concentration, cash, allocation, constraint,
  portfolio-fit, and recommendation-to-position semantics;
* missingness, provenance, and freshness semantics;
* advisory permission boundary;
* prohibited inputs and prohibited inferences;
* fail-closed matrix;
* synthetic JSON examples;
* audit and roadmap updates.

## Non-goals

* no runtime assembler;
* no typed schema or validator;
* no ChatGPT API call;
* no prompt template;
* no provider, yfinance, SEC, or EDGAR change;
* no broker integration;
* no portfolio or watchlist writes;
* no Telegram, notification, dashboard, or delivery behavior;
* no allocation engine;
* no position sizing engine;
* no rebalancing engine;
* no Portfolio Review, Governor, Dispatch Station, or Decision Engine redesign.

## Contract identity

```text
contract_name: chatgpt_portfolio_intelligence_context
contract_version: v1
schema_version: chatgpt-portfolio-intelligence-context-v1
artifact_type: market-engine-chatgpt-portfolio-intelligence-context
```

## Dependencies

* ME-RM06 - ChatGPT advisory delivery roadmap reposition.
* ME-CI01 - Structured Decision Output contract.
* ME-CI02 - ChatGPT Advisory Context Contract.
* ME-PR01 / ME-PR02 - Portfolio Review contract and implementation.
* ME-DE01 / ME-DE02 - Decision Engine handoff contract and implementation.
* ME-GV06 - Governor buy-zone and position-management explanation contract.
* ME-DS01 - Dispatch Station Governor report output contract.
* ME-RUN18 / ME-RUN19 / ME-RUN24 - non-production portfolio-context run evidence.

## Acceptance criteria

* Contract identity is explicit and versioned.
* Source-of-truth matrix identifies canonical portfolio sources and conflict
  behavior.
* Relationship to `chatgpt-advisory-context-v1` is explicit.
* Availability states distinguish available, partial, unavailable, and blocked.
* Holdings and position semantics preserve unknown versus zero.
* Exposure and concentration semantics do not create a new exposure engine.
* Cash remains unavailable unless explicitly proven by an approved source.
* Allocation, target weight, position sizing, and rebalancing remain forbidden.
* Portfolio fit remains separate from recommendation and sizing.
* Missingness, provenance, and freshness are machine-readable.
* Fail-closed behavior is documented.
* Examples exist for complete, partial, blocked, and held-with-sizing-unavailable contexts.
* No runtime code is changed.

## Follow-ups

* ME-CI04 - Define explainability/change-rationale contract.
* ME-CI05 - Produce daily ChatGPT-ready advisory artifact.
* ME-PI01 - Define Portfolio Intelligence exposure contract.
* Future typed schema / validator for ME-CI context families.
* Future deterministic advisory context assembler.
