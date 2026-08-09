# ME-SR23 Corporate-Action Lifecycle Cutoff Remediation Audit

Status: `canary_pending`

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
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_me_sr18_lifecycle_aware_freshness.py tests/market_engine/data/test_scheduled_canonical_price_refresh.py tests/market_engine/data/test_scheduled_canonical_price_refresh_workflow.py -q` | 149 passed | 1.79 s | Expected sessions, bounded recovery, retained metadata, TMHC semantics, provider, singleton, quarantine, manifest, and workflow regressions |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/run -q` | 197 passed | 2.51 s | No failures or skips |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q` | 1443 passed, 1 failed | 5.84 s | Only the known missing historical artifact |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q` | 2110 passed, 1 failed | 6.96 s | Same known failure only |
| Lifecycle v5 and verified-observation v1 schema loaders | passed | <0.1 s | Lifecycle checksum `664416d7d81830b159a395ca4f00de5689b0180037ae5386cee75bfc1132a4db`; observation checksum `a6935b5cbe328b03bcd4b34142e606fdc0210a0f5f03f1c1f39ba69c695ef71c` |
| `git diff --check` | passed | <0.1 s | No whitespace errors |

The sole suite failure is
`test_compact_checksums_match_committed_files_and_local_full_runs`: historical
artifact
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`
is absent. The exact test reproduces identically on clean base SHA
`a0409a49e8f8f3ef9dce352c22b039ce4387faab` (one failure in 0.02 seconds).
There are no new failures.

## Remediation Canary

Pending. Exactly one full-universe `publish=false` canary will be dispatched
after the reviewed implementation and this pre-canary evidence are committed
and pushed. This section will be replaced with the run, artifact, independent
baseline, EA/NSA/TMHC, changed-fileset, publish-skip, and unchanged
`market-data` evidence from that run.

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
