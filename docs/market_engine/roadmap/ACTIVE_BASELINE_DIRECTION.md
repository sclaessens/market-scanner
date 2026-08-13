# Active Baseline Direction

Status: ACTIVE PRODUCT ROADMAP POINTER

Effective date: 2026-08-12

Roadmap decision: ME-RM08

Base state verified at `main` commit
`4753c94c0ab572d619fbb7b82496ba2864797a9f`, the merge commit for PR #477.

## Active product objective

The Market Engine should help the user make better self-directed investment
decisions:

1. analyse the supported ticker universe and surface review candidates;
2. let the user decide whether to buy, sell, or do nothing;
3. record only user-confirmed portfolio transactions;
4. rebuild current positions deterministically from those transactions;
5. add proven position context to candidate analysis;
6. refresh non-canonical advisory price data daily so analysis can use recent
   market context.

The system does not place orders, connect to a broker, infer unreported
transactions, or move allocation authority outside the Decision Engine.

## Current verified state

| Area | Current state |
|---|---|
| Canonical universe | 952 instruments |
| Broad technical analysis | Operational with explicit blockers |
| Fundamental coverage | Partial; ME-DATA11 remains planned |
| Scheduled canonical price refresh | Workflow exists, but automatic canonical publication is blocked |
| Latest canonical evidence result | ME-SR24 completed with blockers because no approved production price-evidence provider route exists |
| Canonical `market-data` publication | Not operational for daily automatic updates |
| Portfolio Review | ME-PR01 contract and ME-PR02 in-memory implementation completed |
| Portfolio source contracts | Private confirmed-event ledger and deterministic position projection implemented by ME-PR03 |
| Legacy portfolio scripts | Reference-only; not canonical runtime |
| ChatGPT advisory contracts | CI01-CI10 completed; later model-invocation work is not the active priority |
| Broker/order execution | Not implemented and not planned in this sequence |

## Active sequence

```text
ME-RM08 - Realign roadmap around portfolio authority and advisory price freshness
  -> ME-PR03 - Manual portfolio transaction ledger and transaction-derived
               portfolio-aware candidate context (COMPLETED)
  -> ME-SR25 - Implement advisory price evidence artifacts (IMPLEMENTED)
  -> ME-DATA11 - Target fundamental derivation at the highest-ranked technical
                 candidate funnel
  -> ME-RUN33 - First useful end-to-end candidate analysis release
  -> ME-CI12 - ChatGPT consumption without new analysis logic
```

ME-PR03 is complete. ME-SR25 is implemented on its review branch and must not
be silently replaced by canonical-publication work. A post-merge operational
validation run remains separately approval-gated.

## What ME-PR03 completed

The user needs portfolio-aware analysis based on actual holdings. Existing
ME-PR02 accepted explicitly supplied `market-engine-portfolio-context-v1`, but
the repository had no approved runtime that turned user-confirmed transactions
into that context.

ME-PR03 closes that gap with one append-only transaction ledger and a
deterministic derived-position projection. A position snapshot is derived
state, not a second source of truth.

The live ledger contains personal financial data. Because this repository is
public, live transactions and position snapshots must remain outside Git and
must be ignored by default. Only schemas, code, documentation, and synthetic
fixtures belong in the repository.

## ME-PR03 authority model

```text
user states a transaction
  -> normalized transaction preview
  -> explicit user confirmation
  -> append-only private ledger event
  -> deterministic position rebuild
  -> validated portfolio-context adapter
  -> non-actionable portfolio-aware candidate review
  -> Decision Engine remains the only allocation authority
```

No purchase or sale may be inferred from an analysis result. Corrections and
cancellations are new ledger events; historical records are not silently
rewritten.

## Why ME-SR25 is separate from ME-SR24

ME-SR24 concerns canonical market-data publication evidence. That line remains
blocked by the absence of an approved production evidence route.

ME-SR25 serves a different purpose: best-effort, clearly labelled, advisory
market context for analysis. It may reuse the existing supported acquisition
path without claiming canonical or publication evidence. Its output must carry
source, observation date, retrieval time, freshness state, and per-ticker
failures. Stale or missing data remains explicit.

ME-SR25 must not:

- change the `market-data` branch;
- satisfy or bypass ME-SR24 publication gates;
- create mutation receipts for canonical publication;
- become a transaction or position source of truth;
- place orders or trigger automatic portfolio changes;
- require a new commercial data-provider contract as a roadmap prerequisite.

## Deferred but preserved work

The following work remains valid but is not the active next step:

- ME-CI11D and the provider-invocation troubleshooting line;
- canonical publication activation after a future approved source policy;
- remaining EA, TMHC, historical-addition, and precision-rewrite remediation;
- ME-PI01 portfolio exposure and concentration intelligence;
- further portfolio expansion, position sizing, notification-channel
  implementation, broker integration, and cloud portfolio storage.

Historical documents may contain a local `Next` label that was correct when
that sprint closed. Those labels are historical evidence only. This file and
ME-RM08 define the current active order.

## Governing documents

- `AGENTS.md`
- `docs/active/governance_v2.md`
- `docs/active/architecture_current_state.md`
- `docs/market_engine/governance/me_gov01_job_scoped_sprint_naming_convention.md`
- `docs/market_engine/roadmap/me_rm08_portfolio_ledger_and_advisory_price_roadmap_realign.md`
- `docs/market_engine/backlog/me_pr03_manual_portfolio_transaction_ledger.md`
- `docs/market_engine/backlog/me_sr25_advisory_price_refresh.md`

## Next implementation

```text
ME-DATA11 - Target fundamental derivation at the highest-ranked technical
candidate funnel
```

No canary, production provider activation, canonical publication, portfolio
data write, broker connection, or order execution was performed by ME-SR25.
Operational validation remains a separately approved post-merge action.
