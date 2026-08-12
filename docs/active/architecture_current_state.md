# Architecture Current State

Status: ACTIVE

Effective date: 2026-08-12

Verified base: `8e71af5935db5e4bc0cd5261035497115df0573d`

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
- partial governed fundamental evidence coverage;
- scheduled canonical refresh infrastructure with fail-closed publication;
- Portfolio Review contract and in-memory implementation;
- structured ChatGPT advisory contracts and local grounding validation.

## Current gaps

- no authoritative manual transaction-ledger runtime;
- no stable transaction-derived portfolio-context assembler;
- no operational advisory-only daily price artifact;
- automatic canonical `market-data` publication remains blocked;
- fundamental evidence coverage remains partial;
- later model invocation, position sizing, and notification work is deferred.

## Active implementation sequence

```text
ME-PR03 -> ME-SR25 -> ME-PI01 -> ME-CI12 -> ME-PS01 -> ME-NL01/02/03
```

See:

- `docs/market_engine/roadmap/ACTIVE_BASELINE_DIRECTION.md`;
- `docs/market_engine/roadmap/me_rm08_portfolio_ledger_and_advisory_price_roadmap_realign.md`.

## Authority boundaries

- Transaction ledger: source of truth for confirmed portfolio transactions.
- Derived positions: rebuildable projection only.
- Advisory price artifact: descriptive market enrichment only.
- Portfolio Review: non-actionable position/exposure interpretation.
- Decision Engine: only allocation authority.
- ChatGPT: explanation and user interaction, not calculation or execution
  authority.
- Broker/order execution: absent.

## Privacy boundary

The repository is public. Real transactions, account identifiers, position
snapshots, and other personal portfolio data must not be committed. Only code,
contracts, documentation, and synthetic fixtures belong in the repository.
