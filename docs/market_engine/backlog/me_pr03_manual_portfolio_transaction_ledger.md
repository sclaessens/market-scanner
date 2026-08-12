# ME-PR03 — Manual portfolio transaction ledger and portfolio-aware candidate context

Sprint ID: ME-PR03

Status: PLANNED / NEXT IMPLEMENTATION STORY

Job family: ME-PR / Portfolio Review

## Product outcome

The user can report a purchase or sale, have the normalized transaction shown
for confirmation, persist the confirmed event in a private append-only ledger,
rebuild current positions deterministically, and let candidate analysis use
the resulting proven position context.

Example user input:

```text
I bought 3 AMD shares on 2026-08-12 at USD 174.20 per share with USD 5.00 fees
through Bolero.
```

The implementation must normalize this into a preview and require explicit
confirmation before persistence. It must never infer a transaction from an
analysis, recommendation, screenshot, watchlist, or broker assumption.

## Existing foundation to reuse

ME-PR03 must build on, reconcile, and version the existing portfolio surfaces:

- `src/market_scanner/portfolio/portfolio_contracts.py`;
- `src/market_scanner/portfolio/portfolio_source_contracts.py`;
- `market-engine-portfolio-context-v1`;
- `sec-companyfacts-portfolio-review-v1`;
- ME-PR01 and ME-PR02;
- existing portfolio-source contract tests;
- legacy `scripts/portfolio/` code as reference only.

The legacy scripts are not canonical runtime and must not be copied blindly.

## Source-of-truth decision

The confirmed transaction ledger is the only authoritative source for
transaction-derived holdings after activation.

```text
authoritative transaction ledger
  -> deterministic position projection
  -> portfolio context
  -> Portfolio Review / candidate enrichment
```

Derived position files, reports, portfolio intelligence, market values, and
ChatGPT explanations are rebuildable outputs. They must not become independent
portfolio sources of truth.

Existing support for manual position snapshots may be retained only as an
explicit migration/bootstrap input. It must not silently coexist as equal
authority after the ledger is activated.

## Required transaction contract

Define a versioned contract such as:

```text
manual-portfolio-transaction-ledger-v1
```

Each event must include:

- ledger/schema version;
- stable transaction ID;
- portfolio/account ID;
- authoritative instrument ID;
- canonical ticker snapshot;
- transaction type: `BUY` or `SELL`;
- trade date and optional execution timestamp;
- quantity;
- unit price;
- trade currency;
- fees and fee currency, with explicit zero versus unavailable semantics;
- broker/account label when supplied;
- source type `manual_user_input`;
- recorded timestamp;
- optional user note;
- optional external reference;
- correction/reversal reference when applicable.

Use exact decimal arithmetic. Floating-point rounding must not determine
quantities, cost basis, fees, or profit/loss.

## Append-only correction model

Confirmed events are immutable.

A correction or cancellation creates a new event that references the original
event. The system must preserve:

- original event;
- correcting or reversing event;
- reason;
- recorded timestamp;
- resulting deterministic projection.

Silent row edits, destructive overwrite, deletion of history, and replacement
of the ledger by a derived positions file are forbidden.

## Validation behavior

The transaction boundary must reject or block with specific issue codes:

- missing or duplicate transaction ID;
- unknown instrument or ambiguous ticker;
- unsupported transaction type;
- zero or negative quantity;
- negative price;
- currency mismatch or unsupported currency;
- fee currency ambiguity;
- invalid date or future date;
- sale exceeding the available position;
- duplicate replay of the same confirmed transaction;
- correction pointing to an unknown or already fully reversed event;
- unordered events that cannot be deterministically reconciled;
- unsupported short position;
- malformed numeric value;
- unavailable required value.

Missing values must remain unavailable and must not be converted to zero.

## Derived position projection

Define a versioned rebuildable output such as:

```text
market-engine-derived-positions-v1
```

For each portfolio/account and instrument, derive at minimum:

- quantity;
- open/closed status;
- weighted average cost in the transaction currency;
- cumulative fees;
- remaining cost basis;
- realized profit/loss under one explicitly documented cost-basis method;
- first and most recent transaction dates;
- last confirmed transaction ID;
- transaction count;
- projection timestamp;
- ledger digest and transaction references;
- calculation status and blockers.

The first version must use one deterministic cost-basis method and document it.
It must not mix FIFO, LIFO, and moving-average outcomes. Currency conversion,
tax reporting, dividends, stock splits, mergers, transfers, short selling, and
options remain unavailable unless separately implemented later.

## Portfolio-aware candidate context

ME-PR03 must adapt the derived position projection into a versioned portfolio
context accepted by the existing Portfolio Review boundary.

Candidate enrichment may state only proven context such as:

- position held, not held, closed, partial, stale, invalid, or unknown;
- current quantity;
- weighted average cost;
- native transaction currency;
- realized profit/loss when calculable;
- last transaction date;
- transaction-count reference;
- whether required portfolio context is missing.

It may compare a candidate with existing holdings through the approved
Portfolio Review contract. It must not create or change candidate ranking,
recommendation, conviction, urgency, tradeability, allocation, target weight,
position size, BUY, SELL, HOLD, add, trim, or execution semantics.

Current market price, market value, unrealized profit/loss, exposure percentage,
cash, sector concentration, and portfolio total remain unavailable unless
supplied by a separately approved source. ME-SR25 and ME-PI01 are the planned
follow-ups for market-price enrichment and broader exposure intelligence.

## Private persistence boundary

The repository is public. Real portfolio transactions and position projections
must not be committed.

The implementation must:

- use a user-controlled private persistence location;
- keep live ledger and derived position paths ignored by Git;
- refuse to write live financial data under committed fixture or artifact
  paths;
- redact transaction details from logs unless explicitly required for a local
  user-facing confirmation;
- commit only code, schemas, documentation, and synthetic fixtures;
- provide export and backup instructions without exposing live data.

The exact private persistence adapter may be local or another explicitly
approved private store, but the contract and deterministic projection must
remain storage-independent.

## Allowed implementation scope

- portfolio transaction contracts and validators;
- private append-only ledger adapter;
- explicit confirmation command or input boundary;
- deterministic position rebuild;
- one-way migration from legacy local transaction data;
- Portfolio Review context adapter;
- portfolio-aware non-actionable candidate enrichment;
- focused schemas, synthetic fixtures, tests, documentation, and audit.

## Forbidden scope

- broker connection or broker export scraping;
- order placement;
- automatic BUY or SELL;
- automatic transaction creation from analysis;
- automatic portfolio mutation from a recommendation;
- canonical market-data publication;
- live price acquisition;
- cash inference;
- tax calculation;
- currency conversion without an approved FX source;
- allocation or position sizing authority;
- hidden filtering or ranking;
- storing real transactions in GitHub;
- changing Decision Engine authority.

## Allowed files or directories

The implementation sprint may modify only the portfolio-related runtime,
contracts, schemas, tests, documentation, and narrowly required command entry
points identified during repository inspection. Expected areas include:

- `src/market_scanner/portfolio/`;
- `src/market_engine/portfolio_review/`;
- `scripts/portfolio/` only for a controlled migration or compatibility shim;
- `tests/portfolio/`;
- `tests/market_engine/portfolio_review/`;
- portfolio schemas/configuration;
- ME-PR03 documentation and audit.

Any cross-job change must be justified explicitly or split into a follow-up.

## Test impact

Required tests include:

- valid BUY;
- multiple BUY events and weighted average cost;
- partial SELL and full SELL;
- oversell rejection;
- fees and explicit zero fees;
- duplicate transaction rejection;
- deterministic replay and idempotent rebuild;
- correction and full reversal;
- same ticker in two accounts;
- authoritative instrument identity and ticker alias validation;
- decimal precision;
- invalid/future dates;
- missing versus numeric zero;
- unavailable currency conversion remains unavailable;
- live data path is Git-ignored;
- synthetic fixtures contain no real portfolio data;
- derived positions reproduce exactly from the ledger;
- derived position files cannot override ledger truth;
- portfolio context adapter preserves provenance;
- held/not-held/unknown states;
- candidate analysis remains non-actionable;
- no broker, provider, notification, or canonical data side effects.

## Acceptance criteria

ME-PR03 is complete only when:

1. confirmed manual transactions are persisted append-only in private storage;
2. every position is reproducible from the ledger alone;
3. corrections preserve full history;
4. oversells, ambiguous identities, duplicates, and malformed transactions fail
   closed;
5. live portfolio data cannot be committed accidentally;
6. transaction-derived portfolio context is accepted by Portfolio Review;
7. candidate output visibly distinguishes held, not held, and unknown;
8. missing current prices do not block position-state context and are not
   invented;
9. no recommendation, allocation, sizing, broker, or execution authority is
   added;
10. focused and full relevant test suites pass;
11. documentation and an audit are complete;
12. implementation is delivered through a draft PR for review.

## Dependency and follow-up order

Dependencies:

- ME-PR01;
- ME-PR02;
- existing portfolio source contracts;
- ME-CI03 portfolio-context semantics.

Follow-ups:

```text
ME-PR03
  -> ME-SR25 - scheduled advisory price refresh
  -> ME-PI01 - exposure and concentration intelligence
```

ME-SR25 must not be pulled into ME-PR03. Price enrichment is descriptive and is
not part of transaction truth.
