# ME-SR23 Corporate-Action Lifecycle Cutoff Remediation Audit

Status: `ready_for_re_review`

## Second Review Findings

The second review of draft PR #474 identified three merge blockers:

1. EA's retained-inactive refresh accepted a terminal August 4 bar while seven
   expected XNYS sessions between the July 23 baseline and that bar were absent.
2. Retained-inactive result construction replaced EA's real pre-merge
   `previous_last_observation` with the post-merge date.
3. TMHC's observation contract described the absence of a complete provider
   bar too broadly, lacked field-specific provenance, and could act as a static
   cutoff against a later valid July 24 daily OHLCV bar.

These findings supersede the previous canary's favorable EA claims. Run
`31278593816` proved neither a complete EA backfill nor correct pre-merge
observation metadata and is not merge-readiness evidence.

## Root Cause and Evidence

The EA acquisition request already covered July 24 through August 4. Provider
parsing and retained-inactive processing did not discard rows: the supported
Yahoo Finance response itself contained only the August 4 terminal row. The
old validator checked only that the resulting terminal date matched the formal
lifecycle boundary; it did not reconcile every exchange session inside the
bounded interval. The old retained-result wrapper then derived observation
metadata from the already merged dataframe.

The corrected EA interval contains these expected XNYS sessions:

- 2026-07-24
- 2026-07-27
- 2026-07-28
- 2026-07-29
- 2026-07-30
- 2026-07-31
- 2026-08-03
- 2026-08-04

Seven complete daily OHLCV observations are held in the checksum-bound
`market-engine-verified-daily-ohlcv-evidence-v1` registry. Each row binds the
instrument, session, complete OHLCV values, HTTPS source, source identity,
retrieval timestamp, validation status, and record checksum. The supported
provider supplies the August 4 observation. This evidence is merged only
through the normal validation and staging path; canonical CSV files are never
edited manually.

For TMHC, the SEC 8-K and NYSE Form 25 establish the lifecycle boundary: the
formal last trading session is July 24 and suspension followed the close. They
do not prove the absence of trading or of a price. A provider request for the
exact July 24 window returned an empty response on
`2026-08-09T14:34:08.615413Z`; SHA-256
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
binds that empty response. This supports only the temporary, provider-specific
claim that no complete valid daily canonical OHLCV bar was returned as of that
retrieval.

## Corrected Contracts

Lifecycle registry v5 separates formal lifecycle truth from canonical data
availability. `last_trading_session` remains the only lifecycle cutoff.
`canonical_ohlcv_last_observed_session`,
`terminal_session_daily_ohlcv_status`, `observation_status_as_of`, and
`observation_evidence` describe current daily-OHLCV availability without
changing that cutoff. The no-bar status requires provider identity, exact UTC
retrieval time, as-of date, request window, response outcome, relevant session,
daily-OHLCV validation status, locator, and response checksum.

A later complete TMHC July 24 bar is inside the lifecycle boundary, is fully
revalidated, and replaces the temporary no-bar status. A July 25 bar remains
post-cutoff and is quarantined before merge. An empty response or a loose price
cannot be canonicalized as a daily OHLCV observation.

Freshness manifest and validation contracts advance to v6. For bounded
refreshes with an existing observation, the exchange calendar determines all
required sessions from the next session through the expected boundary. The
contract is:

`expected sessions = observed valid sessions + explicitly proven exceptions`

Missing sessions trigger bounded provider re-fetch and full revalidation.
Internal or unexplained terminal gaps fail closed. The only current terminal
exception is the precise checksum-bound TMHC provider observation; provider
absence alone is not a generic exception.

Pre-merge checksum, last observation, and row count are captured before any
mutation. Result checksum, last observation, row count, rows added, and the
changed file are captured after the validated merge. Lifecycle/freshness
classification cannot recompute those values. Manifest validation reconciles
all fields against the independent baseline and rejects forged pre-state,
inconsistent row counts, checksums, dates, evidence, or session coverage.

## Local Validation

| Command | Result | Duration | Notes |
|---|---:|---:|---|
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_me_sr18_lifecycle_aware_freshness.py tests/market_engine/data/test_scheduled_canonical_price_refresh.py tests/market_engine/data/test_scheduled_canonical_price_refresh_workflow.py -q` | 150 passed | 1.71 s | Expected sessions, bounded recovery, active recent-listing bounds, retained metadata, TMHC semantics, provider, singleton, quarantine, manifest, and workflow regressions |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/run -q` | 197 passed | 2.47 s | No failures or skips |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q` | 1444 passed, 1 failed | 6.48 s | Only the known missing historical artifact |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q` | 2111 passed, 1 failed | 7.37 s | Same known failure only |
| Lifecycle v5 and verified-observation v1 schema loaders | passed | <0.1 s | Lifecycle checksum `664416d7d81830b159a395ca4f00de5689b0180037ae5386cee75bfc1132a4db`; observation checksum `a6935b5cbe328b03bcd4b34142e606fdc0210a0f5f03f1c1f39ba69c695ef71c` |
| `git diff --check` | passed | <0.1 s | No whitespace errors |

The sole suite failure is
`test_compact_checksums_match_committed_files_and_local_full_runs`: historical
artifact
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`
is absent. The exact test reproduces identically on clean base SHA
`a0409a49e8f8f3ef9dce352c22b039ce4387faab` (one failure in 0.02 seconds).
There are no new failures.

## First Remediation Canary and Material Correction

Full-universe `publish=false` run
[`31319120331`](https://github.com/sclaessens/market-scanner/actions/runs/31319120331)
executed commit `be8430828541077fd283baa0771434e13531c77c`. It failed closed
after 4 minutes 13 seconds with four active recent-listing completeness
failures (`FDXF`, `HONA`, `Q`, and `SOLS`). No publication bundle was uploaded,
the publish job was skipped, and `market-data` remained at
`95c88276763b1762cbbfbccc402ec8535268127b`.

The run nevertheless proved the three review targets in isolated staging:

- EA reconciled all eight expected sessions, added eight rows, and reported
  `2026-07-23` -> `2026-08-04` with row counts 389 -> 397.
- NSA remained an unchanged July 21 retained history.
- TMHC retained the formal July 24 cutoff, the canonical July 23 endpoint, and
  the precise checksum-bound terminal daily-OHLCV exception.

The new failure was a material implementation defect: guarded provider
completeness calculated active recent-listing required sessions from the broad
provider request start instead of from the session following the existing
canonical endpoint. The final manifest calculation already used the correct
bounded interval. The correction makes both paths start at
`previous_last_observation + 1 exchange session` and adds an active
recent-listing regression. This is not an ordinary retry; the first run is
insufficient canary evidence because its full-universe status was degraded.
The contract permits one second canary only after this material correction and
its full local validation.

## Corrected Remediation Canary

The corrected full-universe `publish=false` canary completed successfully:

| Evidence | Result |
|---|---|
| Workflow | [`31319846978`](https://github.com/sclaessens/market-scanner/actions/runs/31319846978) |
| Branch / head | `me-sr23-corporate-action-lifecycle-cutoff-remediation` / `5d8966586c3e0f85d730355539895d364aa7f030` |
| Run identity | `me-sr23-canonical-price-refresh-20260809T145833Z` |
| Input / duration | `publish=false` / 5 minutes 29 seconds |
| Source main | `a0409a49e8f8f3ef9dce352c22b039ce4387faab` |
| Status counts | 942 updated; 4 already current; 6 not expected; 0 stale, failed, or unsupported |
| Coverage | 942 sufficient; 4 limited history; 6 retained inactive; 0 insufficient unexplained |
| Changed files | 947 declared, 947 unique, independently reconciled |
| Publication | required `true`; set valid `true`; empty commit `false` |
| Freshness artifact | `canonical-price-freshness-me-sr23-canonical-price-refresh-20260809T145833Z` (SHA-256 `6b70f5920b056405d37f75338188d4f527bd66fb8c4fe1fe78b79aa11f0737d8`) |
| Publication artifact | `canonical-price-publication-me-sr23-canonical-price-refresh-20260809T145833Z` (SHA-256 `23d07786f81bdc8fcb64240a5b16b3f14c61bccf70ad7f9fb2c5e9fe8fad6ad0`) |
| Manifest | schema v6; checksum `466a49a8dd648e31170881c21fdc0dd7d917a8fff453efe1d1b73c0781aefdc8` |
| Independent validation | validated against detached `market-data` baseline; zero issues, reason codes, or stale tickers |
| Publish job | skipped |
| `market-data` before / after | `95c88276763b1762cbbfbccc402ec8535268127b` / unchanged |

The publication artifact contains 952 governed canonical CSVs plus the
manifest. EA contains exactly the eight expected XNYS sessions from July 24
through August 4, has no post-cutoff row, reports 389 -> 397 rows, and preserves
`previous_last_observation=2026-07-23` and
`resulting_last_observation=2026-08-04`. Its previous and resulting checksums
are `758b5bd8ed67403eebc2ba1673e500ea8cc219ad708f4b0653ca0a180fb867a0`
and `624be997bf229447fb65bb6f2094d3442387e8a1128b539160a6c36e469bc4b6`.

NSA remained a 387-row no-op through July 21 with an unchanged checksum. TMHC
remained a 389-row no-op through July 23 while its formal lifecycle cutoff
remained July 24. The manifest explicitly reconciles July 24 as the one
provider- and timestamp-specific explained daily-OHLCV exception; no July 25
or later row entered canonical output.

## Remaining Risks and Rollback

- The verified EA observations depend on a governed external historical-data
  page and its retrieval evidence; later source corrections require a new
  checksum-bound review, not an in-place canonical edit.
- TMHC's temporary status is provider- and time-specific. Later complete July
  24 OHLCV must replace it; it must never be interpreted as proof of no trade.
- Provider corrections can rewrite history. Session completeness, staged
  checksum reconciliation, atomic publication, and the trusted-main publisher
  gate remain fail closed.
- Rollback is a normal revert of the reviewed ME-SR23 commits. Do not rewrite
  branch history or edit `market-data` manually.
