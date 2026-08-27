# Active Baseline Direction

Status: ACTIVE PRODUCT ROADMAP POINTER

Effective date: 2026-08-13

Roadmap decision: ME-RM08

Base state verified at `main` commit
`9169420427d33864851d36f2b183e35b8bd0c089`, the merge commit for PR #478.

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
| Fundamental coverage | ME-DATA11 ready for human approval: 3 persisted pending candidates, 0 approved imports |
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
  -> ME-SR25 - Implement advisory price evidence artifacts (COMPLETED / MERGED)
  -> ME-DATA11 - Target fundamental derivation at the highest-ranked technical
                 candidate funnel (READY FOR HUMAN APPROVAL; NOT NEXT)
  -> ME-SR26 implementation - advisory OHLC history, current technical screening,
                              and conditional RUN33 handoff (COMPLETED)
  -> ME-SR26 operational canary (FIRST RUN BLOCKED BY RUNNER INTERRUPTION)
  -> Runner interruption diagnostic (COMPLETED)
  -> Second controlled canary (BLOCKED BY HOSTED-RUNNER COMMUNICATION LOSS)
  -> ME-SR27 runner/workflow hardening review (COMPLETED)
  -> ME-SR28 bounded acquisition observability and diagnostic retention (NEXT)
  -> Reviewed ME-SR28 merge
  -> Third controlled ME-SR26 canary (REQUIRED; NOT YET AUTHORIZED)
  -> Human approval / DATA07 / DATA06 / RUN31 checkpoint
  -> ME-RUN33 - First useful end-to-end candidate analysis release (CONDITIONAL)
  -> ME-CI12 - ChatGPT consumption without new analysis logic (PLANNED)
```

ME-PR03 and ME-SR25 are complete. PR #478 merged ME-SR25 at
`9169420427d33864851d36f2b183e35b8bd0c089`. ME-DATA11 then acquired official
SEC evidence for ten ranked issuers. Review remediation persisted replayable
pending approval candidates for ASH, BIO, and CI. Separate checksum-bound human
approval remains the blocker to authoritative DATA07, DATA06, and RUN31
changes. ME-SR26 supplies current technical prerequisites without simulating
that approval; ME-RUN33 is not immediately executable.

ME-SR27 proved that the current 952-symbol yfinance call is serial at ticker
level under `threads=False`, the 15-second timeout is request-scoped rather
than an aggregate deadline, and the monolithic history step has no durable
progress boundary before final artifact upload. ME-SR28 is therefore the next
implementation sprint. It must add only the reviewed bounded chunk,
static execution/upload/receipt/gate workflow boundaries, parent-enforced
worker-process deadline, heartbeat/resource/version telemetry, diagnostic-only
checkpoint, and fail-closed final assembly contract. Local completion is not
persistence; only a successfully completed upload-action boundary is durable.
A third canary is not ready or authorized until that implementation is reviewed
and merged.

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
ME-RUN33 - First useful end-to-end candidate analysis release (CONDITIONAL)
```

No canonical publication, portfolio data write, broker connection, order,
notification, or ME-RUN33 implementation was performed by ME-DATA11.
