# ME-RM08 — Portfolio ledger and advisory price roadmap realignment

Sprint ID: ME-RM08

Status: COMPLETED DOCS-ONLY ROADMAP REALIGNMENT

Job family: ME-RM / Roadmap Governance

Date: 2026-08-12

Base commit: `8e71af5935db5e4bc0cd5261035497115df0573d`

## Decision

The active Market Engine roadmap is realigned around the user's actual product
workflow:

```text
Market Engine proposes review candidates
  -> user decides
  -> user-confirmed transactions become portfolio truth
  -> positions are rebuilt from those transactions
  -> candidate analysis receives proven position context
  -> advisory prices refresh daily for recent market context
```

The system is advisory and user-directed. It does not connect to a broker,
place orders, infer transactions, or mutate the portfolio from an analysis.

## Current-state correction

The previous roadmap pointer was stale in several ways:

- its title still said `AFTER ME-RUN26`;
- its active chain stopped at ME-SR19 and ME-DATA11;
- it did not identify ME-RUN32 or ME-SR20 through ME-SR24 as completed or
  blocker-closeout work;
- it continued to present ME-CI11D as the active next advisory sprint;
- it deferred ME-PR03 even though ME-PR02 explicitly identified the approved
  portfolio-context source/persistence boundary as the next portfolio gap;
- it did not distinguish blocked canonical publication from the simpler
  advisory price freshness needed by the user;
- `AGENTS.md` referenced a missing `docs/active/` source-of-truth layer.

ME-RM08 corrects those inconsistencies without rewriting historical evidence.

## Active order

```text
ME-RM08 (docs-only realignment)
  -> ME-PR03 (next)
  -> ME-SR25
  -> ME-PI01
  -> ME-CI12
  -> ME-PS01
  -> ME-NL01/02/03
```

## Insertion rationale

ME-PR03 is inserted ahead of ME-DATA11 and ME-CI11D because the user explicitly
needs portfolio-aware analysis based on actual holdings, and existing Portfolio
Review has no authoritative runtime source for those holdings.

ME-SR25 follows ME-PR03 because recent price data is required for useful
advisory analysis and later unrealized profit/loss and exposure enrichment.
It is separated from ME-SR24 because canonical publication evidence is not
required for non-executable advisory context.

## Story boundaries

### ME-PR03

ME-PR03 owns private manual transaction persistence, deterministic position
rebuild, and non-actionable portfolio-context integration. It does not own
price acquisition, broker integration, orders, allocation, or sizing.

### ME-SR25

ME-SR25 owns scheduled advisory price acquisition, validation, freshness, and
artifact handoff. It does not own canonical publication, portfolio mutation,
recommendation, allocation, or execution.

### ME-PI01

ME-PI01 remains the later exposure/concentration story. It may consume
transaction-derived positions and separately sourced advisory prices only after
both upstream contracts are stable.

## Canonical publication line

ME-SR24 remains correctly closed as `COMPLETED WITH BLOCKERS`. The absence of
an approved production evidence route still blocks automatic canonical
`market-data` publication. ME-RM08 does not weaken that boundary.

Canonical publication work is parked until the source-policy prerequisite is
explicitly satisfied. No canary is authorized by this docs-only change.

## Deferred work

The following remains planned, but is not current next work:

- ME-DATA11 fundamental derivation pilot;
- ME-CI11D provider-environment propagation;
- ME-CI12 until portfolio and advisory-price inputs are stable;
- canonical publication provider selection and activation;
- remaining canonical mutation and lifecycle blockers;
- notification adapters.

## Documentation authority

`docs/active/architecture_current_state.md` is the concise current-state
pointer. Detailed backlog and roadmap history remains under
`docs/market_engine/`. Historical `Next` statements describe the state when
those documents were written and do not override ME-RM08.

## Non-goals

- no Python code;
- no tests;
- no workflow change;
- no provider call;
- no portfolio data write;
- no market-data write;
- no canary;
- no broker or order behavior;
- no recommendation or Decision Engine semantic change.

## Acceptance criteria

- current main and PR #475 merge are recorded;
- active roadmap sequence is unambiguous;
- ME-PR03 is fully specified as the next implementation story;
- ME-SR25 is present directly after ME-PR03;
- canonical and advisory price paths are explicitly separated;
- real portfolio data is forbidden from the public repository;
- ME-DATA11 and ME-CI11D remain preserved but deferred;
- active documentation exists at the paths declared by `AGENTS.md`;
- backlog and roadmap point to the same active sequence;
- no runtime or data change is made.
