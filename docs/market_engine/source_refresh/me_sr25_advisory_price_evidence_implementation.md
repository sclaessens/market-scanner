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

All internally generated clock values pass through one canonical UTC helper
and are serialized with a trailing `Z`. Build, load, consume, and their CLI
paths use that same helper when no timestamp is supplied. Explicit timestamps
remain subject to the strict timezone-aware canonical-UTC and trusted-future
checks; naive input is not normalized into acceptance.

Freshness has two deliberately separate layers:

- artifact freshness is the immutable acquisition-time classification stored
  in each observation and reconciled by manifest totals. The loader recomputes
  it against `generated_at` to detect manipulation;
- effective freshness is a non-persisted load/consume view recomputed against
  `trusted_now` with the same completed-session resolver and freshness helper.

Retention does not grant freshness. When a newer market session has completed,
a previously fresh close becomes effectively stale even though its stored
artifact status remains fresh. A weekend or pre-close interval retains the
previous close as fresh when the authoritative resolver reports no newer
completed session. Unavailable effective-session resolution fails closed as
`invalid` and never exposes a current price.

## Consumer contract

`consume_advisory_price_context` accepts an artifact directory, not a
caller-supplied price object. It reloads and validates both documents against
the current authoritative universe and policy, then requires the exact
instrument-ID/ticker pair. Consumer contract v2 reports
`artifact_freshness_status`, `artifact_observation_age_completed_sessions`,
`effective_freshness_status`,
`effective_observation_age_completed_sessions`, and canonical `evaluated_at`.
`price_context_status` is always the effective status. Only a validated
effectively `fresh`/`succeeded` observation exposes `current_price`; other
states retain provenance, acquisition-time evidence, and diagnostics with no
current price.

## Schedule and retention

`.github/workflows/advisory-price-evidence.yml` is scheduled for 05:30 UTC and
also supports reviewed manual dispatch. It checks out source read-only, builds
under the runner temporary directory, and uploads a 14-day GitHub Actions
artifact. It has no publication job and no write permission. ME-SR25 did not
execute this workflow during implementation. After merge, five scheduled runs
completed successfully: `31777995934`, `31868074950`, `31930153267`,
`32000009638`, and `32104872490`. The first four each classified 946 records
fresh, one stale, and five invalid. Run `32104872490` classified only three
fresh, 944 stale, and five invalid because almost the whole provider snapshot
was one completed session behind at the 05:30 UTC acquisition window. Workflow
success therefore proves execution and artifact delivery, not analytic
freshness.

## Explicit non-goals

ME-SR25 does not write or advance `market-data`, persist raw provider payloads,
read or mutate portfolio data, create canonical receipts, rank candidates,
change the Decision Engine, place orders, notify users, or publish results.
It implements neither ME-DATA11 nor the candidate release.
