# ME-PR03 Manual Portfolio Transaction Ledger Implementation

Status: COMPLETED

## Product outcome

ME-PR03 implements a private, storage-independent, append-only transaction
ledger for explicit user-reported and user-confirmed transactions. The ledger
is the sole authoritative source for transaction-derived holdings. Position
projections and portfolio-aware candidate context are deterministic derived
outputs and cannot overwrite ledger truth.

```text
authoritative instrument registry
  -> normalized manual transaction preview
  -> confirmation of the exact preview digest
  -> immutable append-only private ledger event
  -> deterministic moving-weighted-average projection
  -> market-engine-portfolio-context-v1 adapter
  -> non-actionable candidate position context
```

The implementation does not infer transactions. The words `BUY` and `SELL`
describe historical transaction types supplied by the user; they are not
recommendations, decisions, orders, or execution instructions.

## Versioned contracts

| Contract | Purpose |
|---|---|
| `manual-portfolio-transaction-preview-v1` | Canonical preview and confirmation digest |
| `manual-portfolio-transaction-ledger-header-v1` | Private JSONL ledger header |
| `manual-portfolio-transaction-ledger-v1` | Immutable transaction, correction, or reversal event |
| `market-engine-derived-positions-v1` | Rebuildable position projection |
| `market-engine-portfolio-aware-candidate-context-v1` | Non-actionable candidate enrichment |
| `market-engine-portfolio-context-v1` | Existing Portfolio Review input boundary |

Machine-readable event and projection schemas are stored under
`config/market_engine/portfolio/`.

Every ledger event binds a stable transaction ID, portfolio and account,
authoritative instrument ID, canonical ticker snapshot, historical transaction
type, trade date, optional execution timestamp, exact-decimal quantity and unit
price, native currency, explicit fee availability, optional broker/account
label, fixed `manual_user_input` source type, recorded timestamp, optional note
and external reference, and correction or reversal references.

Required decimals reject Python floats. Decimal strings are serialized without
binary floating-point conversion. Missing required values remain errors.
Unavailable fees are represented as unavailable with null amount and currency;
an explicit zero fee is represented as available amount `0` plus its currency.

## Instrument identity

The command loads the authoritative canonical Market Engine instrument
registry. Callers may identify an instrument by authoritative instrument ID,
canonical ticker, or a unique explicitly mapped alias. An unknown instrument,
ambiguous alias, unsupported legacy alias, or disagreement between instrument
ID and ticker fails closed. The normalized event derives its instrument ID,
canonical ticker, currency, and exchange identity from that registry.

## Confirmation boundary

Normalization has no persistence side effect. It returns the canonical event,
its SHA-256 preview digest, and the same digest as confirmation token. The
confirm operation recomputes the digest and requires exact equality among the
event, preview digest, preview token, and supplied confirmation token. It then
revalidates the canonical event against the current authoritative registry.

Any preview change invalidates the prior confirmation. No analysis,
recommendation, candidate artifact, screenshot, watchlist, or broker assumption
can invoke append without the exact confirmed preview.

The narrow command boundary is:

```text
market-engine-portfolio-ledger preview --input <private-input.json>
market-engine-portfolio-ledger confirm --preview <exact-preview.json> \
  --confirmation-token <sha256> --ledger <private-ledger.jsonl>
market-engine-portfolio-ledger rebuild --ledger <private-ledger.jsonl>
```

The command prints previews and projections. It does not persist derived
positions automatically.

## Append-only persistence and corrections

The ledger is newline-delimited canonical JSON. Its header fixes the portfolio
and schema. Initial creation uses exclusive create and owner-only permissions.
Existing appends take an exclusive file lock, read and validate the complete
ledger, validate the candidate event and resulting projection, append exactly
one line, flush, `fsync`, and preserve mode `0600`.

Transaction IDs are unique. Duplicate confirmation or replay fails closed and
does not create a second economic effect. A correction is a new fully specified
economic event that references and replaces one still-active economic event. A
reversal is a new non-economic event that fully voids one still-active economic
event. Targets remain in history. Unknown, cross-identity, previously replaced,
and previously reversed targets are rejected.

No edit, replacement, deletion, truncation, or derived-file import operation is
provided.

## Private storage and backups

External user-controlled paths are supported. Repository-local live ledgers are
allowed only below `data/portfolio/private/`, which is Git-ignored. Runtime path
validation refuses all other repository-local locations and refuses an already
tracked path. Responses redact the parent path and transaction details.

Only synthetic test inputs belong in Git. Real ledger or projection files must
not be copied into fixtures, artifacts, documentation, or tracked data paths.

For backup or export, stop writers, copy the complete JSONL ledger to another
user-controlled encrypted/private location, preserve file permissions, and
verify a deterministic rebuild and ledger digest from the backup. A derived
positions export is convenience output only and is not a sufficient backup.

## Deterministic position projection

V1 uses one cost-basis method: `moving_weighted_average_v1`, calculated with
Python `Decimal` at a deterministic precision of 50 significant digits.

For an available same-currency fee:

* a purchase adds `quantity * unit_price + fee` to cost basis;
* weighted average cost is remaining cost basis divided by open quantity;
* a sale releases `weighted_average_cost * sold_quantity` from cost basis;
* realized profit/loss is sale proceeds minus the sale fee minus released cost
  basis;
* a partial sale preserves the moving average of the remaining units;
* a full sale sets remaining quantity, cost basis, and weighted average cost to
  exact zero.

Cumulative fees include purchase and sale fees. If a required fee is
unavailable, quantity remains projectable but dependent fees, cost basis, or
realized profit/loss remain unavailable with explicit blockers. Fee or trade
currency disagreement is not converted. No FX source is present.

Active economic events are ordered deterministically by trade date, execution
timestamp, recorded timestamp, and transaction ID. A same-session purchase and
sale without execution timestamps is rejected as non-deterministic. Oversells
and short positions are rejected before append.

The projection includes quantity, open/closed state, weighted average cost,
cumulative fees, remaining cost basis, realized profit/loss, first and last
transaction dates, last confirmed transaction ID, active transaction count,
projection timestamp, ledger digest, transaction references, calculation
status, and blockers. Rebuilding the same ledger produces identical output and
never reads a derived position file.

## Portfolio Review and candidate context

The adapter selects a portfolio/account/instrument projection and constructs
the existing `market-engine-portfolio-context-v1` input. It preserves ledger
digest, transaction references, account and instrument identity, calculation
status, blockers, average cost, native currency, realized result, transaction
count, and last transaction date.

States remain descriptive: `held`, `not_held`, `closed`, `partially_known`, or
`unknown`. The existing `stale` and `invalid` Portfolio Review states remain
available for controlled consumers. A closed position is reviewed as explicit
non-held context, while partial calculation evidence remains partial.

Current price, market value, unrealized profit/loss, exposure, cash,
concentration, and portfolio total remain explicitly unavailable. Their absence
does not prevent a proven held/not-held/closed position state. Candidate context
does not rank, filter, recommend, allocate, size, or execute.

## Stable issue codes

The implementation emits stable issue codes for missing or duplicate IDs,
unknown or ambiguous instruments, identity mismatch, unapproved aliases,
unsupported transaction/event types, invalid quantities or prices, malformed
decimals, unsupported or mismatched currencies, fee ambiguity, invalid or
future dates, invalid timestamps, oversells, duplicate replay, invalid
correction/reversal targets, non-deterministic ordering, short positions,
missing values, corrupt/incompatible ledgers, confirmation failures, unsafe
storage paths, tracked private paths, and portfolio mismatch.

Any error occurs before append or leaves the existing file byte-identical.

## Explicit non-goals

ME-PR03 adds no broker connection, scraping, order placement, automatic
transaction creation, provider or network request, live price acquisition,
canonical publication, cash inference, FX acquisition, taxes, dividends,
corporate-action processing, transfers, short selling, options, exposure or
concentration intelligence, allocation, position sizing, notification,
Telegram, or Decision Engine change. ME-SR25 remains the next story.
