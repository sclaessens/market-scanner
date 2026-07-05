# ME-DS01 - Dispatch Station Governor Report Output Contract Backlog Entry

Sprint ID: ME-DS01

Status: COMPLETED DOCS-ONLY CONTRACT

Job family: ME-DS / Dispatch Station

Date: 2026-07-05

## Goal

Define the versioned Dispatch Station output contract that converts approved Governor evaluation output into deterministic operator-readable report structures without creating new investment semantics or delivery authority.

## Result

ME-DS01 defines:

```text
market-engine-dispatch-station-governor-report-v1
```

The contract establishes:

* approved Governor-only upstream input;
* fail-closed report states;
* canonical report identity and section structure;
* factor, recommendation, buy-zone, and position-management rendering rules;
* mandatory blocker, risk, limitation, missing-evidence, provenance, and authority representation;
* canonical JSON, operator Markdown, and channel-neutral compact-preview profiles;
* deterministic rendering and cross-format equivalence rules;
* explicit ME-DS02 implementation acceptance criteria.

## Authority boundary

ME-DS01 authorizes no runtime implementation and no delivery behavior.

It does not authorize scoring, ranking, recommendation reinterpretation, invented price levels, targets, stops, urgency, conviction, tradeability, allocation, position sizing, order generation, provider/network calls, live-price retrieval, Telegram/email sending, production publishing, portfolio/watchlist mutation, broker behavior, scheduler behavior, UI behavior, or Decision Engine decisions.

## Next backlog item

```text
ME-DS02 - Implement local non-production Governor report artifact
```
