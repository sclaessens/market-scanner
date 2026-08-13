# ME-SR25 Advisory Price Evidence Implementation

## Purpose

ME-SR25 provides recent, explicitly non-canonical price context to downstream
ticker analysis. It does not create canonical market-data evidence and carries
no recommendation, ranking, allocation, execution, portfolio, notification, or
publication authority.

## Artifact boundary

Each run writes two stable JSON documents:

- `advisory_price_observations.json` contains one deterministically ordered
  record for every instrument in the authoritative universe;
- `advisory_price_manifest.json` contains run metadata, source and policy
  identity, reconciled counts, retention, and SHA-256 bindings.

The manifest contract is `market-engine-advisory-price-manifest-v1`; the
observation container is `market-engine-advisory-price-observations-v1`; and
each observation is `market-engine-advisory-price-observation-v1`. The shared
artifact version is `me-sr25-advisory-price-evidence-v1`.

The builder binds the raw universe file digest, normalized universe checksum,
freshness-policy digest, observation-file digest, and a canonical manifest
digest. The loader also revalidates run identity, retrieval time, universe
schema, retention, full-universe membership, totals, currency, and calculated
freshness. Rehashing a semantically changed document therefore does not make it
trusted analysis input.

## Acquisition and identity

The production command reuses the existing yfinance batch and single-symbol
daily-history adapters. It does not add a provider or provider contract. The
source is identified as `yahoo-finance-yfinance`, the adapter as
`existing_yfinance_daily_history_adapter`, and the observation semantic as
`provider_reported_daily_close`; values are never described as live prices.

Only authoritative instrument IDs, canonical tickers, source mappings, and
currencies are accepted. Unmapped identities fail closed without a provider
request. Batch gaps use at most two single-symbol attempts. A provider or
validation failure affects only the corresponding instrument and remains an
explicit `missing` or `invalid` record.

## Validation and freshness

The same semantic validation functions protect creation, loading, and consumer
access. Valid prices are finite positive canonical decimal strings. Floats,
booleans, zero, negative values, exponent notation, `NaN`, and infinity are
outside the validated domain. Timestamps must be timezone-aware and cannot
claim observations after retrieval or retrieval in the trusted future.

The policy in `config/market_engine/advisory_price_freshness_policy.json`
classifies a successful official-close observation as fresh only when its
completed-session lag is zero. The existing completed-session resolver finds
the latest completed market session. The deliberately small age calculation
counts weekdays between the observed session and expected session. It does not
introduce a second exchange-calendar framework; market-specific holiday
precision remains limited by this documented method.

Every canonical instrument is present as `fresh`, `stale`, `missing`, or
`invalid`. The manifest reconciles attempted and per-state totals. Duplicate,
missing, and extra identities fail closed.

## Consumer contract

`consume_advisory_price_context` accepts an artifact directory, not a
caller-supplied price object. It reloads and validates both documents against
the current authoritative universe and policy, then requires the exact
instrument-ID/ticker pair. Only a validated `fresh`/`succeeded` observation
exposes `current_price` and `currency`. Other states retain their provenance,
observation semantic, timestamps, freshness, and diagnostics with no current
price.

## Schedule and retention

`.github/workflows/advisory-price-evidence.yml` is scheduled for 05:30 UTC and
also supports reviewed manual dispatch. It checks out source read-only, builds
under the runner temporary directory, and uploads a 14-day GitHub Actions
artifact. It has no publication job and no write permission. ME-SR25 did not
execute this workflow or contact the production provider; operational
validation requires separate post-merge approval.

## Explicit non-goals

ME-SR25 does not write or advance `market-data`, persist raw provider payloads,
read or mutate portfolio data, create canonical receipts, rank candidates,
change the Decision Engine, place orders, notify users, or publish results.
It implements neither ME-DATA11 nor the candidate release.
