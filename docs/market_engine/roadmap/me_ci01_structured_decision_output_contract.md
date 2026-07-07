# ME-CI01 - Structured Decision Output Contract Roadmap Entry

Sprint ID: ME-CI01

Status: COMPLETED DOCS-FIRST CONTRACT

Job family: ME-CI / ChatGPT Advisory Integration

Date: 2026-07-07

## Roadmap position

ME-CI01 is the first implementation of the ME-RM06 delivery reposition:

```text
Market Engine
-> Structured Artifacts / Structured Decision Output
-> ChatGPT Advisory Layer
-> user
```

It precedes:

```text
ME-CI02 - ChatGPT Advisory Context Contract
  -> ME-CI03 - ChatGPT-readable Portfolio Intelligence context
  -> ME-CI04 - explainability/change-rationale contract
  -> ME-CI05 - daily ChatGPT-ready advisory artifact
  -> ME-PI01 - Portfolio Intelligence exposure contract
  -> ME-PS01 - Position Sizing decision contract
  -> ME-NL01 - channel-neutral Notification Layer contract
```

## Roadmap decision

Structured Decision Output v1 is the stable machine-readable contract family for
future consumer use.

Approved identifiers:

```text
schema_version: structured-decision-output-v1
artifact_type: market-engine-structured-decision-output
```

The contract is not a runtime implementation and does not create ChatGPT,
Notification Layer, dashboard, provider, portfolio/watchlist, order, broker,
allocation, or execution behavior.

## Gate for ME-CI02

ME-CI02 may define ChatGPT Advisory Context only by consuming structured
artifact semantics and preserving Structured Decision Output as source of truth.

ME-CI02 must not allow ChatGPT to invent missing scores, price levels, portfolio
context, actionability, allocation, or Decision Engine output.

## Next sprint

```text
ME-CI02 - Define ChatGPT Advisory Context Contract
```
