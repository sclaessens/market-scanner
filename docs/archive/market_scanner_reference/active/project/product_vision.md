# Product Vision

Status: ACTIVE
Reset stage: RESET-1

## Vision

Build a clean v2 market-scanner platform that turns market, portfolio, and source-data evidence into auditable decision-support outputs without weakening the separation between classification, allocation, and communication.

The product should feel operationally reliable, explainable, and easy to maintain. It should avoid historical ballast and should not require future work to interpret many overlapping sprint-era documents before making safe changes.

## User Value

The platform should help the operator:

- see which market opportunities exist;
- understand why each opportunity is structurally valid, weak, incomplete, or blocked by source-data readiness;
- distinguish source-data readiness from investment quality;
- understand current portfolio exposure;
- receive final decisions from one authoritative downstream layer;
- review reporting output without wondering whether reporting changed the decision;
- learn from historical decisions and predictions under a research-only framework.

## Product Principles

1. Preserve evidence before interpreting it.
2. Classify before allocating.
3. Allocate only in the Decision Engine.
4. Communicate without changing source decisions.
5. Keep generated outputs separate from source inputs.
6. Keep source-data review separate from investment analysis until contracts are approved.
7. Prefer simple canonical documentation over layered historical governance.
8. Write v2 code from scratch based on v2 contracts.

## Strategic Product Direction

The rebuild should first create a small, deterministic core pipeline. Fundamentals, SEC, Telegram, prediction tracking, and historical learning should return only when the v2 skeleton and contracts are stable.

## Non-Goals

v2 is not a trading bot, broker integration, live execution engine, or discretionary override mechanism. It must not automate buying or selling. It must not produce hidden ranking or urgency outside the Decision Engine.

## Current Strategic Priority

The immediate priority is not more features. The immediate priority is canonical documentation, repository structure, contracts, and fixtures that allow a clean v2 implementation to begin safely.
