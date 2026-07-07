# ME-CI04 - Explainability / Change-Rationale Contract Backlog Entry

## Status

COMPLETED DOCS-ONLY CONTRACT

## Goal

Define `chatgpt-explainability-change-rationale-context-v1`, the controlled
contract for current-state explanation, evidence deltas, state transitions,
reason attribution, blocker deltas, uncertainty deltas, freshness rationale,
portfolio rationale, unchanged conclusion rationale, contradiction handling, and
temporal comparison boundaries.

## Scope

* contract documentation;
* source-of-truth matrix;
* relation to ME-CI01, ME-CI02, and ME-CI03;
* explanation availability states;
* current-state rationale model;
* change classification;
* state transition semantics;
* evidence delta semantics;
* reason attribution levels;
* blocker and uncertainty deltas;
* freshness-driven rationale;
* portfolio rationale boundary;
* unchanged conclusion semantics;
* contradiction handling;
* temporal comparison rules;
* fail-closed matrix;
* JSON examples;
* audit and roadmap updates.

## Non-goals

* no runtime explainability engine;
* no temporal diff engine;
* no artifact comparison engine;
* no causal attribution engine;
* no materiality engine;
* no typed schema or validator;
* no ChatGPT API integration;
* no prompt template;
* no provider, yfinance, SEC, or EDGAR change;
* no broker integration;
* no portfolio or watchlist writes;
* no allocation, position sizing, or rebalancing engine;
* no Recommendation Review, Portfolio Review, Governor, Dispatch Station, or
  Decision Engine redesign.

## Contract identity

```text
contract_name: chatgpt_explainability_change_rationale_context
contract_version: v1
schema_version: chatgpt-explainability-change-rationale-context-v1
artifact_type: market-engine-chatgpt-explainability-change-rationale-context
```

## Acceptance criteria

* Contract identity is explicit and versioned.
* Canonical explanation sources are documented.
* Availability states distinguish available, partial, unavailable, blocked, and
  not comparable.
* Reason attribution levels prevent correlation-as-causation.
* Temporal comparison rules distinguish current, reference, prior comparable,
  previous chronological, and baseline runs.
* Unchanged conclusion handling does not imply "nothing changed."
* Portfolio rationale remains separate from standalone recommendation rationale.
* Fail-closed behavior is documented.
* JSON examples validate.
* No runtime code is changed.

## Dependencies

* ME-RM06 - ChatGPT advisory delivery roadmap reposition.
* ME-CI01 - Structured Decision Output contract.
* ME-CI02 - ChatGPT Advisory Context Contract.
* ME-CI03 - ChatGPT-readable Portfolio Intelligence Context.
* Recommendation Review contracts.
* Portfolio Review contracts.
* Decision Engine handoff contracts.
* Governor explanation contracts.
* Dispatch Station Governor report output contract.

## Follow-ups

* ME-CI05 - Produce daily ChatGPT-ready advisory artifact.
* future typed schema / validator.
* future deterministic advisory context assembler.
* future prompt contract.
* future controlled advisory dry run.
