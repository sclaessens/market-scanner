# ME-CI04 - Explainability / Change-Rationale Contract Roadmap Entry

## Status

COMPLETED DOCS-ONLY CONTRACT

## Roadmap position

ME-CI04 follows ME-CI03:

```text
ME-RM06 - ChatGPT advisory delivery roadmap reposition
  -> ME-CI01 - Structured Decision Output contract
  -> ME-CI02 - ChatGPT Advisory Context Contract
  -> ME-CI03 - ChatGPT-readable Portfolio Intelligence Context
  -> ME-CI04 - Explainability / Change-Rationale Contract
  -> ME-CI05 - Daily ChatGPT-ready advisory artifact
  -> typed schema / validator
  -> deterministic context assembler
  -> prompt contract
  -> controlled advisory dry run
```

## Contract identity

```text
contract_name: chatgpt_explainability_change_rationale_context
contract_version: v1
schema_version: chatgpt-explainability-change-rationale-context-v1
artifact_type: market-engine-chatgpt-explainability-change-rationale-context
```

## Summary

ME-CI04 defines the advisory explainability and change-rationale boundary for
current state explanation, evidence deltas, state transitions, blocker deltas,
uncertainty deltas, freshness rationale, portfolio rationale, unchanged
conclusion rationale, contradiction handling, and temporal comparison.

The contract introduces reason attribution levels:

```text
explicit_upstream_reason
supported_contributing_factor
associated_change_only
unknown
prohibited_inference
```

This prevents ChatGPT from turning correlation into causation or missing
history into a change narrative.

ME-CI04 remains docs-only. It does not implement a runtime explainability
engine, temporal diff engine, causal model, materiality model, typed schema,
validator, deterministic assembler, prompt, ChatGPT API integration, portfolio
write, watchlist write, allocation engine, sizing engine, rebalancing engine,
or Decision Engine semantic change.

## Gate for ME-CI05

ME-CI05 may produce a daily ChatGPT-ready advisory artifact only by preserving
ME-CI01, ME-CI02, ME-CI03, and ME-CI04 source-of-truth boundaries.

ME-CI05 must not allow daily advisory artifacts to invent change rationale,
materiality, causal root cause, hidden prior state, portfolio allocation,
position sizing, or Decision Engine approval.

## Next

```text
ME-CI05 - Produce daily ChatGPT-ready advisory artifact
```
