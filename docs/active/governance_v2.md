# Governance v2

Status: ACTIVE

This file is a concise active pointer. `AGENTS.md` remains the highest local
repository instruction. Detailed governance is preserved under
`docs/market_engine/governance/`.

## Core doctrine

```text
classification upstream
allocation downstream
Decision Engine = only allocation authority
```

No upstream layer may create hidden filtering, tradeability, conviction,
urgency, allocation, sizing, or execution authority.

## Portfolio authority

- Only user-confirmed ledger events may change transaction truth.
- Analysis, recommendations, screenshots, watchlists, and inferred broker
  activity may not create transactions.
- Positions must be rebuildable from the authoritative transaction ledger.
- Corrections are append-only events.
- Missing portfolio data is unavailable, not zero.
- Real portfolio data must remain outside the public repository.

## Price-data authority

- Canonical market-data publication remains subject to ME-SR23/24 evidence and
  publication gates.
- Advisory price data is a separate descriptive source.
- Advisory prices may not satisfy, bypass, or impersonate canonical evidence.
- Price enrichment may not change quantity, average cost, realized profit/loss,
  or transaction history.

## Source and consumer separation

Source Refresh acquires and validates data. Portfolio Review interprets proven
portfolio context. Recommendation Review and Portfolio Review remain separate.
The Decision Engine alone owns allocation decisions. Delivery and ChatGPT
communicate approved outputs and do not execute them.

## Change process

- Runtime implementation is performed by Codex/local execution after review.
- Documentation-only roadmap work may be performed directly through GitHub.
- New work uses job-scoped sprint IDs.
- Active sequence changes require an explicit roadmap rationale.
- Historical evidence is preserved; historical `Next` labels do not override
  the current active baseline pointer.
- Runtime, data, workflow, or canary execution requires explicit sprint scope.

## Current roadmap authority

The current active order is defined by ME-RM08 and
`docs/market_engine/roadmap/ACTIVE_BASELINE_DIRECTION.md`.
