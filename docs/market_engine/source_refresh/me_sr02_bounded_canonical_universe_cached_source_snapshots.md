# ME-SR02 - Bounded canonical-universe SEC CompanyFacts cached source snapshots

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh

Status: COMPLETED BY ME-SR02

## Purpose

ME-SR02 produces a bounded, version-controlled SEC CompanyFacts cached-source snapshot bundle for the canonical ticker universe path consumed by ME-RUN16.

The sprint exists because ME-RUN16 selected 13 active `cached_source_only` tickers from the canonical universe but every ticker failed closed with `blocked_missing_cached_source` because the repository had no cached SEC CompanyFacts snapshots under the approved source snapshot root.

## Approved source snapshot root

```text
data/market_engine/source_snapshots/sec_companyfacts
```

## Snapshot run id

```text
me-sr02-canonical-universe-20260619T000000Z
```

## Snapshot envelope contract

```text
sec-companyfacts-raw-v1
```

Every raw snapshot is stored as a SEC CompanyFacts snapshot envelope with:

```text
metadata
raw_payload
```

The metadata fields follow the existing `SecCompanyFactsRawSnapshot` loader contract:

```text
ticker
cik
source_name
fetched_at
snapshot_id
payload_format_version
```

## Bounded universe result

Canonical active `cached_source_only` universe from ME-UNI03 / ME-RUN16:

```text
NVDA
AMD
ASML
META
MSFT
VRT
CLS
CRDO
IREN
COST
HO
AVGO
TSM
```

ME-SR02 produced valid cached-source raw snapshot envelopes for the 12 SEC CompanyFacts-supported canonical rows below:

```text
NVDA
AMD
ASML
META
MSFT
VRT
CLS
CRDO
IREN
COST
AVGO
TSM
```

ME-SR02 did not produce a raw snapshot for:

```text
HO
```

Reason: `HO` is the Euronext ticker for Thales and is retained in the canonical universe, but it is not represented by a supported SEC CompanyFacts CIK in this bounded snapshot bundle. The ticker remains explicit in `provider_errors.csv` and must continue to fail closed in RUN until a future approved non-SEC or manual-source contract exists.

`SMCI` remains excluded because the canonical universe marks it as `manual_review_only`.

## Snapshot file layout

```text
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/NVDA_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/AMD_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/ASML_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/META_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/MSFT_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/VRT_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/CLS_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/CRDO_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/IREN_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/COST_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/AVGO_companyfacts.json
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/TSM_companyfacts.json
```

Manifest and explicit unsupported-provider marker:

```text
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/ticker_manifest.csv
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/provider_errors.csv
data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/snapshot_metadata.json
```

## Data boundary

The committed raw payloads are bounded SEC CompanyFacts-shaped cache evidence for local Market Engine contract execution. They are not portfolio data, recommendations, trade plans, watchlist mutations, delivery payloads, production reports, broker instructions, or financial advice.

The payloads include the approved tags needed by the existing source-context mapper:

```text
Revenues
NetIncomeLoss
NetCashProvidedByUsedInOperatingActivities
PaymentsToAcquirePropertyPlantAndEquipment
```

Each tag contains one FY USD fact so the existing mapper can preserve source availability, missingness, numeric values, and provenance through downstream dry-run contracts.

## Expected next sprint

Recommended next sprint:

```text
ME-RUN17 - Execute canonical-universe cached-source batch dry-run after ME-SR02 snapshots
```

Expected RUN17 behavior:

* 12 tickers should have one valid cached SEC CompanyFacts snapshot candidate;
* `HO` should remain blocked as missing or unsupported until a future approved source contract exists;
* no provider fallback should occur;
* generated local RUN artifacts should remain uncommitted by default;
* no Telegram, broker, portfolio, watchlist, scheduler, UI, production-report, allocation, ranking, scoring, target-price, urgency, conviction, tradeability, or execution authority should be introduced.

## Boundaries preserved

ME-SR02 did not introduce runtime Python code, provider refresh automation, live SEC/EDGAR calls from the application, yfinance calls, live market data calls, Telegram delivery, email delivery, broker behavior, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.
