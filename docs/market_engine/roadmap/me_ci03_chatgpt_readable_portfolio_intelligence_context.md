# ME-CI03 - ChatGPT-readable Portfolio Intelligence Context Roadmap Entry

## Status

COMPLETED DOCS-ONLY CONTRACT

## Roadmap position

ME-CI03 follows ME-CI02:

```text
ME-RM06 - ChatGPT advisory delivery roadmap reposition
  -> ME-CI01 - Structured Decision Output contract
  -> ME-CI02 - ChatGPT Advisory Context Contract
  -> ME-CI03 - ChatGPT-readable Portfolio Intelligence Context
  -> ME-CI04 - Explainability/change-rationale contract
  -> ME-CI05 - Daily ChatGPT-ready advisory artifact
  -> ME-PI01 - Portfolio Intelligence exposure contract
  -> typed schema / validator
  -> deterministic assembler
  -> prompt contract
  -> controlled advisory dry run
```

## Contract identity

```text
contract_name: chatgpt_portfolio_intelligence_context
contract_version: v1
schema_version: chatgpt-portfolio-intelligence-context-v1
artifact_type: market-engine-chatgpt-portfolio-intelligence-context
```

## Summary

ME-CI03 defines how proven portfolio state and review intelligence may be made
ChatGPT-readable inside the ME-CI02 advisory boundary.

It composes:

* `market-engine-portfolio-context-v1`;
* `sec-companyfacts-portfolio-review-v1`;
* `market-engine-decision-engine-handoff-v1`;
* Governor position-management explanation context when present;
* provenance, freshness, missingness, uncertainty, and fail-closed rules.

ME-CI03 remains docs-only. It does not implement a runtime assembler, typed
schema, prompt, ChatGPT API integration, portfolio source connector, allocation
engine, sizing engine, rebalancing behavior, broker integration, notification
adapter, or Decision Engine change.

## Gate for ME-CI04

ME-CI04 may define explainability/change-rationale context only by preserving
the ME-CI01, ME-CI02, and ME-CI03 source-of-truth boundaries.

ME-CI04 must not allow ChatGPT to infer missing portfolio state, target weights,
position sizes, allocation, rebalancing, or Decision Engine approval from
portfolio intelligence context.

## Next

```text
ME-CI04 - Define explainability/change-rationale contract
```
