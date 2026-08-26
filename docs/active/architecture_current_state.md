# Architecture Current State

Status: ACTIVE

Effective date: 2026-08-13

Verified base: `9169420427d33864851d36f2b183e35b8bd0c089`

## Product model

The Market Engine produces reproducible analysis and candidate artifacts.
The user makes every investment decision. A private manual transaction ledger
records only user-confirmed portfolio events. Derived positions and advisory
prices enrich later analysis without becoming execution authority.

```text
market/source evidence
  -> observations and analysis
  -> recommendation review
  -> portfolio review
  -> Decision Engine
  -> structured advisory output
  -> user decision
```

A separate user-confirmed transaction path records what actually happened:

```text
user confirmation
  -> private append-only transaction ledger
  -> derived positions
  -> portfolio context
```

## Current capabilities

- canonical universe of 952 instruments;
- broad technical screening and candidate ranking with explicit readiness
  blockers;
- checksum-bound targeted fundamental derivation candidates for the
  authoritative technical top-25 funnel;
- scheduled canonical refresh infrastructure with fail-closed publication;
- advisory-only full-universe price evidence, freshness validation, and a
  validated analysis consumer;
- advisory-only bounded daily OHLC history, current full-universe technical
  screening, exact SR25 close reconciliation, and a conditional RUN33 input
  handoff;
- private manual transaction ledger, deterministic derived positions, and
  transaction-derived Portfolio Review context;
- structured ChatGPT advisory contracts and local grounding validation.

## Current gaps

- ME-DATA11 review remediation produced three persisted, replayable pending
  approval candidates but no operator-approved authoritative import;
- automatic canonical `market-data` publication remains blocked;
- fundamental evidence coverage remains partial;
- later model invocation, position sizing, and notification work is deferred.

## Active implementation sequence

```text
ME-SR25 (COMPLETED) -> ME-DATA11 (READY FOR HUMAN APPROVAL; NOT NEXT)
  -> ME-SR26 implementation (COMPLETED)
  -> ME-SR26 operational canary (FIRST RUN BLOCKED BY RUNNER INTERRUPTION)
  -> runner interruption diagnostic (COMPLETED)
  -> second controlled canary authorization (READY / NEXT)
  -> successful controlled ME-SR26 canary (REQUIRED)
  -> approval / DATA07 / DATA06 / RUN31 checkpoint
  -> ME-RUN33 (CONDITIONAL) -> ME-CI12 (PLANNED)
```

See:

- `docs/market_engine/roadmap/ACTIVE_BASELINE_DIRECTION.md`;
- `docs/market_engine/roadmap/me_rm08_portfolio_ledger_and_advisory_price_roadmap_realign.md`.

## Authority boundaries

- Transaction ledger: source of truth for confirmed portfolio transactions.
- Derived positions: rebuildable projection only.
- Advisory price artifact: immutable acquisition-time freshness evidence;
  current-price use requires a separate effective freshness calculation at
  each trusted load or consumption time.
- Advisory OHLC history: separate non-canonical technical evidence. Its loader
  replays checksums, universe/policy identity, bar semantics, and effective
  freshness. Only that private validated context can enter current screening.
- RUN33 handoff: input-only conditional contract. Pending DATA11 approval
  forces every candidate ineligible and grants no downstream execution.
- Portfolio Review: non-actionable position/exposure interpretation.
- Fundamental derivation: deterministic evidence generation only; a pending
  candidate grants no DATA07, DATA06, or RUN31 authority.
- Decision Engine: only allocation authority.
- ChatGPT: explanation and user interaction, not calculation or execution
  authority.
- Broker/order execution: absent.

## Privacy boundary

The repository is public. Real transactions, account identifiers, position
snapshots, and other personal portfolio data must not be committed. Only code,
contracts, documentation, and synthetic fixtures belong in the repository.
