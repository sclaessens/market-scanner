# ME-SR17 Scheduled Canonical Price Refresh and Freshness Publication Audit

Status: `implementation_complete_pending_post_merge_canary`

## Executive Summary

ME-SR17 implements a daily GitHub Actions price-refresh and publication path
for the 952-instrument active canonical Market Engine universe. It reuses the
ME-DATA05 yfinance provider normalization, incremental merge, CSV validator,
and atomic per-ticker replacement. ME-SR17 adds exchange-aware completed-session
selection, bounded multi-symbol requests and retries, isolated staging,
historical-prefix integrity checks, a checksum-bound freshness manifest, a
data-only `market-data` branch, and a fail-closed analysis consumption boundary.

The implementation is proven locally with offline providers and deterministic
fixtures. It has not published production price data from this feature branch.
A real scheduled run and `market-data` canary remain post-merge requirements.

## Original Structural Failure

The previous ChatGPT daily refresh expectation had no executable data boundary.
The only active workflow, `.github/workflows/daily-market-scan.yml`, was
manual-only and executed a side-effect-free canonical application dry-run. It
had no schedule, no price-provider call, no exchange calendar, no persisted
dataset, no freshness manifest, and no validated data handoff to analysis.

ChatGPT automation is not the data executor. It does not own a durable GitHub
runner workspace, repository write token, provider retry lifecycle, or atomic
publication transaction. A later ChatGPT automation may monitor the latest
workflow and manifest, but GitHub Actions plus Market Engine runtime code must
execute and publish the data refresh.

## Reused Authoritative Components

ME-SR17 reuses:

- `incremental_market_data_refresh.refresh_one_instrument` for ME-DATA05
  incremental acquisition, normalization, merge, validation, and atomic file
  replacement;
- `download_yfinance_batch` and the existing yfinance symbol normalization;
- `local_market_data_universe.validate_price_history_csv` as the authoritative
  canonical OHLC history validator;
- the committed ME-DATA04 canonical universe snapshot at
  `artifacts/market_engine/data_runs/me-data04-complete-dataset-20260713T133000Z-coverage-after/universe_snapshot.json`;
- the existing `data/processed/<source_symbol>.csv` layout;
- ME-RUN30 full canonical-universe analysis as the downstream consumer after
  publication validation.

The authoritative snapshot contains 952 active instruments. All 952 have a
mapped existing yfinance provider identity. No new universe resolver, price
schema, provider, or ticker membership was introduced.

## Runtime and Status Contract

The new runtime is:

```text
src/market_engine/data/scheduled_canonical_price_refresh.py
```

Canonical freshness statuses are:

```text
updated
already_current
stale
failed
insufficient
unsupported
```

Stable reason codes include:

```text
VALIDATED_UPDATE_PERSISTED
NO_NEW_SESSION_EXPECTED
ALREADY_CURRENT
EXPECTED_SESSION_NOT_AVAILABLE
LOCAL_HISTORY_MISSING_AND_PROVIDER_EMPTY
VALID_HISTORY_INSUFFICIENT_ROWS
UNSUPPORTED_EXCHANGE
PROVIDER_MAPPING_MISSING
PROVIDER_TIMEOUT
PROVIDER_RATE_LIMITED
PROVIDER_ERROR
PROVIDER_PAYLOAD_SCHEMA_INVALID
PROVIDER_PAYLOAD_MALFORMED
PROVIDER_PAYLOAD_NOT_CHRONOLOGICAL
PROVIDER_DUPLICATE_TIMESTAMP
PROVIDER_FUTURE_DATED_BAR
PROVIDER_OHLC_INVALID
PRICE_VALIDATION_FAILED
PRICE_MERGE_FAILED
HISTORY_TRUNCATION_BLOCKED
HISTORY_DUPLICATE_DATE_BLOCKED
HISTORICAL_VALUE_REWRITE_BLOCKED
```

A run is `completed` only when every expected mapped ticker is current and
valid. It is `degraded` when valid retained histories coexist with stale,
failed, insufficient, or unsupported tickers. It is `failed` when the full
publication set cannot be validated. Valid ticker updates may be published
during a degraded run, but the workflow remains visibly failed and automated
analysis refuses the degraded manifest.

## Exchange Calendar Semantics

The runtime derives the last completed session from exchange, country,
timezone, local close time, weekends, and exchange holidays. It supports:

- US equities: NYSE/Nasdaq aliases, New Year, Martin Luther King Day,
  Presidents Day, Good Friday, Memorial Day, Juneteenth, Independence Day,
  Labor Day, Thanksgiving, and Christmas observance;
- continental Europe: Amsterdam, Brussels, Paris, and Frankfurt timezones,
  Good Friday, Easter Monday, Labour Day, Christmas, and Boxing Day;
- United Kingdom: London timezone, Good Friday, Easter Monday, early/spring/
  summer bank holidays, Christmas, and Boxing Day observance.

An unknown exchange with no supported country mapping is `unsupported`. The
ME-DATA04 snapshot records the existing canonical US histories with US country
metadata, so its legacy `UNKNOWN` exchange value resolves conservatively to
the US calendar without changing instrument identity or universe membership.

## Provider, Batching, Retry, and Timeout Policy

The default provider remains Yahoo Finance through yfinance. Requests are
grouped into bounded batches of 25 provider symbols. Batches are sequential;
there is no unbounded parallelism. Only the missing range is requested. Each
batch has at most three attempts, exponential delays of one and two seconds,
and a 15-second provider timeout.

Timeouts, rate limits, provider errors, empty responses, malformed schemas,
duplicate timestamps, unordered timestamps, future bars, non-finite values,
and invalid OHLC relationships receive controlled classifications. Logs do not
contain provider response bodies, cookies, credentials, or tokens.

## Staging and Atomic Publication

Refresh always runs against an isolated staging tree. If `market-data` exists,
its current `data/processed` tree is copied into staging first. A ticker failure
therefore leaves the previously published bytes unchanged. ME-DATA05 writes a
candidate CSV to a temporary sibling, validates it, and atomically replaces the
staged file. ME-SR17 then verifies that every pre-existing historical date and
value is unchanged. A truncation or rewrite restores the previous bytes and
blocks that ticker.

The publication job independently revalidates the complete bundle using code
checked out from trusted `main`. It materializes `market-data`, copies only
`data/processed/*.csv` and the latest manifest, revalidates the exact tree,
creates a normal Git commit, and pushes without force. `main` receives no
automatic price-data commit.

When `market-data` does not exist, the publication job creates an orphan branch
and removes the temporary main worktree contents before seeding only validated
price CSVs and the manifest. If bootstrap acquisition cannot produce a complete
valid publication set, no branch commit or authoritative manifest is created.

## Manifest Contract

Schema:

```text
market-engine-me-sr17-canonical-price-freshness-manifest-v1
```

The latest authoritative path on `market-data` is:

```text
manifests/canonical_price_freshness_latest.json
```

The manifest contains run ID, UTC generation time, source main SHA, workflow
run ID, data branch, universe version/checksum/size, provider configuration,
expected completed sessions per exchange, status counts, run status,
publication decision, a separate fundamental-evidence boundary, and sorted
per-ticker identity, market, provider, previous/resulting/expected dates, rows
added, validation/freshness status, reason code, relative path, and file
checksum. `manifest_checksum` is recomputed over canonical JSON bytes excluding
that field.

Publication validation recalculates canonical manifest bytes and every file
checksum, requires the exact authoritative universe binding and ordered ticker
set, rejects loose unbound CSVs and executable content, and can bind the source
SHA to the trusted main checkout during publication.

## Consumption Boundary

The updated daily analysis workflow checks out executable code from `main` and
materializes `market-data` into a separate path. It never imports or executes
code from the data branch. `consume-analysis` validates the manifest, universe,
file set, checksums, current exchange sessions, and non-degraded status before
passing the validated `data/processed` root to ME-RUN30. Missing, malformed,
tampered, stale, degraded, or universe-inconsistent publications stop before
analysis.

Local operator commands retain their explicit local price-root convention.
The only automated analysis workflow uses the published validation boundary;
there is no workflow fallback to arbitrary loose CSVs.

## Fundamental Evidence Boundary

ME-SR17 does not generate, reuse, or change DATA09 or DATA10 approval decisions.
The freshness report records fundamental evidence as `not_evaluated` with
`NO_RELIABLE_AUTOMATED_FUNDAMENTAL_FRESHNESS_CONTRACT`. This prevents a price
refresh from being presented as a fundamental refresh. A future reliable
filing-freshness check may report `approval_required`, but operator approval
must remain human and checksum-bound.

## Workflow Security Boundary

`Canonical Price Refresh` runs daily at `05:30 UTC`, including weekends, and
supports manual dispatch. Concurrency uses one repository-scoped group with
`cancel-in-progress: false`. The refresh job has `contents: read`; only the
separate publication job has `contents: write`. Publication requires:

- a valid publication set;
- at least one changed validated price file;
- a trusted `refs/heads/main` run;
- either a scheduled event or an explicit manual `publish: true` input.

A feature-branch dispatch can validate and upload freshness evidence but cannot
publish, even when the input is selected. The workflow uses no force push,
history rewrite, main data commit, secret logging, approval generation, or code
execution from `market-data`.

Repository Actions must retain permission for GitHub Actions to create and
push ordinary commits to `market-data`. If branch protection is enabled, its
rules must explicitly allow `github-actions[bot]` ordinary non-force pushes or
use a reviewed repository ruleset exception. No bypass is implemented here.

## Offline Test Matrix

The deterministic tests cover:

- one and multiple missing sessions, missing history initialization, current
  history, weekend, insufficient history, US and European holidays, timezone
  close boundaries, and unknown exchanges;
- missing provider mapping, bounded batching, timeout retry, rate limit,
  provider error, empty response, malformed schema, duplicate/unordered/future
  timestamps, and invalid OHLC;
- truncation, historical rewrite, per-ticker byte preservation, partial
  success, atomic writes, canonical ordering, and idempotent content;
- manifest and file checksums, manifest tampering, source-SHA binding, universe
  mismatch, loose files, missing/malformed manifest, stale consumption, and
  analysis non-invocation;
- workflow triggers, concurrency, permissions, feature-branch publication
  blocking, data-only bootstrap, no-force push, evidence upload, and validated
  analysis ordering;
- no fundamental operator approval generation and the unchanged ME-DATA05
  regression suite.

Local validation results:

- ME-SR17 runtime tests: 39 passed;
- ME-SR17 workflow tests: 6 passed;
- existing ME-DATA05 tests: 13 passed;
- complete Market Engine data suite: 315 passed;
- complete Market Engine suite: 1,340 passed;
- complete repository suite: 2,007 passed;
- Ruby standard-library YAML parsing: both workflow files valid;
- `git diff --check`: passed.

`actionlint` was not installed locally and was not added as an uncontrolled
project dependency. Workflow behavior is additionally covered by deterministic
static contract tests.

## Governance Boundary

ME-SR17 changes data acquisition, freshness classification, publication, and
consumption validation only. It does not change recommendation labels, ranking
semantics, conviction, urgency, tradeability, portfolio/allocation behavior,
Telegram delivery, broker/order execution, ticker membership, fundamental
approval authority, or Decision Engine authority.

## Local Proof Versus Live Proof

Locally proven:

- deterministic calendar and refresh behavior;
- controlled offline provider outcomes;
- per-ticker and global fail-closed preservation;
- canonical manifest construction and validation;
- atomic staging and publication-tree preparation;
- publication and consumption workflow structure;
- feature-branch publication denial;
- downstream analysis non-invocation after validation failure.

Not proven until after merge:

- GitHub schedule execution from default-branch `main`;
- real yfinance availability, latency, and response coverage for all 952
  instruments;
- first creation or update of the remote `market-data` branch;
- repository branch/ruleset permission compatibility;
- downloadable GitHub freshness artifact;
- successful consumption of the real published dataset by the daily analysis
  workflow;
- absence of secrets in real Actions logs.

## Post-Merge Canary

After merge, an operator must:

1. open **Actions → Canonical Price Refresh → Run workflow**;
2. select branch `main`;
3. set `publish` to `true`;
4. start the workflow and confirm refresh evidence is downloadable;
5. verify the refresh job used the 952-instrument universe and inspect all
   `updated`, `already_current`, `stale`, `failed`, `insufficient`, and
   `unsupported` counts;
6. confirm `market-data` was created or advanced by a normal commit without a
   force push and contains only `data/processed/*.csv` plus
   `manifests/canonical_price_freshness_latest.json`;
7. download or inspect the manifest and verify its checksum, universe binding,
   file checksums, source main SHA, and workflow run ID;
8. confirm the Daily Market Scan workflow materializes and accepts the
   publication only when the manifest is current and non-degraded;
9. inspect logs for bounded provider calls and verify no secret values appear;
10. rerun the workflow with no new completed session and verify that it uploads
    freshness evidence without creating an empty `market-data` commit.

A future ChatGPT monitor may read the latest workflow result and freshness
manifest, summarize status counts, and warn when the workflow did not run. It
must not execute the provider refresh or create approvals.
