# ME-SR23 Corporate-Action Lifecycle Cutoff Remediation Audit

Status: `completed_with_blockers`

## Third Review Findings

The third review of draft PR #474 identified three final merge blockers:

1. Seven EA gap-fill bars had been transcribed from Investing.com without a
   repository approval proving acquisition, raw storage, and canonical
   publication rights.
2. The trusted publisher validated a checksum-bound hand-authored evidence
   record, but could not replay a raw provider response and independently
   reconcile every OHLCV value with the staged canonical CSV row.
3. Legacy and canonical lifecycle observation aliases were coalesced without
   rejecting contradictory values.

These findings invalidate the earlier claim that the seven Investing.com bars
were governed canonical source evidence. Previous favorable EA canary results
remain historical diagnostics only and are not source-provenance or
merge-readiness evidence.

## Source-Governance Decision

The repository review covered the provider and source-approval contracts,
credential and acquisition boundaries, artifact retention patterns,
provenance fields, and existing market-data adapters. No repository contract
approves Investing.com, Alpha Vantage, Yahoo Finance, or another fallback
provider for all three required uses: daily-OHLCV acquisition, immutable raw
response storage, and canonical gap-fill publication.

The production policy is therefore explicit and empty:

`config/market_engine/source_policies/market_price_sources.json`

The policy distinguishes provider reachability from acquisition, raw-storage,
and canonical-publication approval. A fallback must have a non-empty approval
ID, daily-OHLCV scope, explicit exchange scope, retention and redistribution
classifications, and all three approval booleans. Unknown, partially approved,
wrong-exchange, or missing-approval providers fail closed.

The former `verified_daily_ohlcv_evidence.json` registry and
`verified_price_observations.py` injection path were deleted. No transcribed EA
OHLCV value remains in production code or configuration. In production, the
normal primary request still covers the complete bounded interval. Missing
expected sessions are recorded as fallback-required sessions, but remain
blocked because the source policy currently contains no approved fallback.

Consequently, EA cannot yet satisfy the eight-session interval from the
July 23 baseline through the August 4 lifecycle cutoff. This is the intended
fail-closed outcome, not a reason to recreate hand-authored evidence.

## Replayable Observation Receipt Contract

The v7 freshness manifest introduces a source-agnostic observation-receipt
ledger. A policy-approved fallback response is usable only through this chain:

`raw artifact -> deterministic parser -> receipt -> canonical row`

Raw JSON payloads are stored beneath a provider-bound
`evidence/market_price/<provider>/<sha256>.json` locator. Locators cannot be
absolute, traverse directories, select another provider directory, or disagree
with the payload checksum. Credential-like fields are rejected before storage
and are not included in diagnostics.

The `canonical-json-daily-ohlcv` parser version `v1` produces deterministic
decimal strings and integer volume. Every receipt binds:

- receipt and canonical-row schema versions;
- instrument, ticker, exchange, currency, and session;
- provider and approval IDs;
- UTC retrieval time and bounded request window;
- response status and content type;
- raw artifact locator and SHA-256;
- retention and redistribution classifications;
- parser name and version;
- normalized open, high, low, close, adjusted close, and volume;
- canonical row SHA-256 and receipt SHA-256.

The canonical row digest covers instrument, session, complete OHLCV, volume,
and currency. The observation receipt root is a SHA-256 digest over the sorted
set of canonical row digests, making ordering irrelevant while rejecting
missing, extra, or duplicate leaves.

## Trusted Publisher Reconciliation

The trusted publisher independently:

1. compares the staged dataset with the trusted `market-data` baseline;
2. rejects changes to existing historical rows;
3. derives every newly added session from the CSV diff;
4. reconciles added sessions with the primary acquisition journal and the
   fallback-required session set;
5. verifies the exact raw-artifact fileset;
6. reloads the source policy and approval;
7. reloads and hashes each raw artifact;
8. reruns the named parser version;
9. rebuilds every receipt and receipt checksum;
10. recalculates the receipt root;
11. enforces one receipt per fallback-required session; and
12. compares all replayed OHLCV fields and volume with the staged CSV row.

The publisher rejects absent or extra artifacts, absent or duplicate receipts,
empty evidence for required fallback sessions, wrong ticker, instrument,
exchange, session, approval, provider, request window, parser, raw checksum,
row value, volume, adjusted close, root, or lifecycle cutoff. Primary and
fallback observations cannot silently overwrite one another. Publication
remains atomic and gated by trusted main. The workflow now carries the governed
raw-evidence directory into `market-data` and stages additions and deletions
explicitly.

## Alias Boundary

Lifecycle parsing now uses one generic `resolve_semantic_alias` boundary for:

- `last_trading_session` and `delisting_end_date`;
- `canonical_ohlcv_last_observed_session` and
  `price_observation_end_session`; and
- `terminal_session_daily_ohlcv_status` and
  `final_session_observation_status`.

Dates are compared as semantic session dates, including equivalent midnight
ISO timestamps. Observation statuses use an explicit legacy-to-canonical map.
Legacy-only, canonical-only, and semantically equal dual inputs are accepted
and immediately projected to one canonical internal representation. Invalid,
unknown, or conflicting values fail closed with ticker, field names, raw and
normalized values, and contract version. Canonical v5 output does not emit the
legacy observation aliases.

TMHC remains separately governed: July 24 is its formal last trading session,
the provider-specific daily-OHLCV absence status remains temporary, a later
valid July 24 bar remains admissible, and any post-cutoff bar remains rejected.

## Regression Evidence

The regression set covers approved complete primary acquisition, exact
approved fallback supplementation with raw replay, unapproved and partially
approved providers, approval and exchange mismatch, missing or mutated raw
artifacts, parser mismatch, request-window and post-cutoff evidence, primary
and fallback conflict, secret rejection, deterministic numeric serialization,
receipt/root ordering, and trusted-publisher CSV and fileset mutations.

Alias coverage includes legacy-only, canonical-only, semantically equal,
conflicting, equivalent date notation, invalid date, mapped legacy status,
unknown status, and singleton/batch behavior.

### Local validation

| Command | Result | Duration |
|---|---:|---:|
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_observation_receipts.py tests/market_engine/data/test_me_sr18_lifecycle_aware_freshness.py tests/market_engine/data/test_scheduled_canonical_price_refresh.py tests/market_engine/data/test_scheduled_canonical_price_refresh_workflow.py -q --tb=short` | 194 passed, 0 failed, 0 skipped | 3.11 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/run -q --tb=short` | 197 passed, 0 failed, 0 skipped | 2.51 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q --tb=short` | 1488 passed, 1 failed, 0 skipped | 7.37 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q --tb=short` | 2155 passed, 1 failed, 0 skipped | 8.53 s |
| Lifecycle v5 and market-price source-policy v1 loaders | passed | <0.1 s |
| `git diff --check` | passed | <0.1 s |

Both broad suites have only the known historical artifact failure:
`test_compact_checksums_match_committed_files_and_local_full_runs` cannot find
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`.
The exact test reproduces identically on clean base SHA
`a0409a49e8f8f3ef9dce352c22b039ce4387faab` (one failure in 0.03 seconds).
There are no new failures.

The lifecycle registry checksum is
`664416d7d81830b159a395ca4f00de5689b0180037ae5386cee75bfc1132a4db`.
The empty production source-policy checksum is
`55cd77a75c9e45c699d02e075b7c3ddf33bfb96d6ea5147ba9b7ad612c908d61`.

The mandatory repository greps still find pre-existing BUY and SELL command
parsing and transaction logging under `scripts/portfolio`; the current diff
does not touch those files. The `tradeable` grep is empty. No Decision Engine,
allocation, or reporting semantics changed in this remediation.

## Final Non-Publishing Canary

Exactly one post-remediation workflow was dispatched with `publish=false`:

| Evidence | Result |
|---|---|
| Workflow | [31323447253](https://github.com/sclaessens/market-scanner/actions/runs/31323447253) |
| Branch / canary head | `me-sr23-corporate-action-lifecycle-cutoff-remediation` / `43a5b646253fca1833d2882e41252cfaaa145203` |
| Run identity | `me-sr23-canonical-price-refresh-20260809T161908Z` |
| Input / job duration | `publish=false` / 5 minutes 48 seconds |
| Trusted source main | `a0409a49e8f8f3ef9dce352c22b039ce4387faab` |
| Universe | 952 total; 946 active; 6 retained inactive; 0 pending |
| Status | 942 updated; 4 already current; 5 not expected; 1 failed; 0 stale or unsupported |
| Coverage | 942 sufficient; 4 limited; 5 retained inactive; 1 not applicable; 0 insufficient unexplained |
| Source policy | v1; checksum `55cd77a75c9e45c699d02e075b7c3ddf33bfb96d6ea5147ba9b7ad612c908d61`; 0 approved fallbacks |
| Freshness artifact | `canonical-price-freshness-me-sr23-canonical-price-refresh-20260809T161908Z`; GitHub digest `8486953b76eece07306348187fa6e10f5d812f75f8ac255ab7017399a6fd130e` |
| Extracted report SHA-256 | `772a5ababbe6d06ed35d67008abdcf8941d54a525caa257c5338a6b8892a7cae` |
| Manifest | v7; checksum `99331b0cb70bba7d7a747e167c9e329b2cdcf25b6ef34707adab66f11f3634ce` |
| Changed files | 946 declared and 946 unique; EA not included |
| Receipts / roots | 0 / 0, because no fallback is approved or used |
| Publication | required `false`; set valid `false`; publication artifact skipped |
| Publish job | skipped |
| `market-data` before / after | `95c88276763b1762cbbfbccc402ec8535268127b` / unchanged |

EA is the sole failed instrument. Its trusted baseline remains 389 rows through
July 23 with checksum
`758b5bd8ed67403eebc2ba1673e500ea8cc219ad708f4b0653ca0a180fb867a0`.
The primary provider returned only August 4. The required interval contains
July 24, 27, 28, 29, 30, and 31 and August 3 and 4. The seven internal sessions
were classified as fallback-required, but the empty production policy yielded
no receipts, raw artifacts, or receipt root. The pipeline preserved the
baseline unchanged: `previous_last_observation=2026-07-23`,
`resulting_last_observation=2026-07-23`, and `rows_added=0`. It did not reuse
the former Investing.com values.

TMHC retained `last_trading_session=2026-07-24`, canonical observation end
July 23, and the precise temporary
`no_valid_daily_ohlcv_bar_from_provider_as_of` status. No legacy observation
aliases were emitted. Local regressions prove that a later complete July 24
bar remains admissible and a July 25 bar remains quarantined.

Artifact review found that the canary journal classified TMHC's already
explained July 24 no-bar session as fallback-required even though it added no
canonical row. The final implementation now excludes explicitly explained
no-observation sessions from gap-fill receipt requirements and requires every
remaining missing session to be explained before accepting the provider
result. A new local regression is green. Per the one-canary constraint, this
post-canary correction was not dispatched again and is not represented by a
second workflow run.

## Remaining Blocker and Rollback

The current production policy contains no approved canonical daily-OHLCV
fallback. EA therefore remains incomplete and the final status cannot be
`READY FOR RE-REVIEW`. The one non-publishing canary demonstrated this
fail-closed condition without modifying `market-data`.

Rollback is a normal revert of the reviewed ME-SR23 commits. Do not rewrite
branch history, manually edit `market-data`, restore the deleted hand-authored
EA registry, or weaken the receipt and alias gates.
