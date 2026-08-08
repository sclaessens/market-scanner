# ME-SR23 Corporate-Action Lifecycle Cutoff Remediation Audit

Status: `review_remediation_locally_validated_canary_pending`

## Incident and Review Findings

Scheduled Canonical Price Refresh run `31243056845` failed closed on August 8,
2026. Atomic publication prevented all staged changes from reaching
`market-data`. Draft PR #474 subsequently received four material review
findings:

1. TMHC lifecycle truth had been lowered to the final available provider bar.
2. `changed_price_file_count` counted freshness classifications rather than
   actual changed canonical files.
3. Conflicting lifecycle alias fields could be silently normalized.
4. Singleton provider revalidation did not receive the batch route's lifecycle
   context.

This remediation addresses all four findings without changing Decision Engine
semantics, weakening atomic publication, or editing `market-data` directly.

## Primary Evidence and Date Semantics

| Ticker | Formal last trading session | Last valid price observation | Transaction closing | Suspension / inactive boundary | Evidence |
|---|---:|---:|---:|---|---|
| EA | 2026-08-04 | 2026-08-04 | 2026-08-04 | before open 2026-08-05 | [EA completion announcement](https://www.ea.com/news/ea-announces-completion-of-acquisition), [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/712515/000114036126031157/ef20079099_8k.htm) |
| NSA | 2026-07-21 | 2026-07-21 | 2026-07-22 | before open 2026-07-22 | [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/1618563/000110465926085888/tm2620871d8_8k.htm) |
| TMHC | 2026-07-24 | 2026-07-23 | 2026-07-24 | after close 2026-07-24; inactive 2026-07-25 | [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/1562476/000119312526316037/d148123d8k.htm), [NYSE Form 25](https://www.sec.gov/Archives/edgar/data/1562476/000087666126000640/xslF25X02/primary_doc.xml), [independent trade history](https://stockanalysis.com/stocks/tmhc/history/) |

The SEC 8-K states that the transaction completed on July 24 and that NYSE was
asked to suspend trading after the close on July 24. NYSE's official Form 25
was signed on July 24. The independent history exposes July 24 as the last
trade-price date but has no valid July 24 OHLC/volume row; the available
canonical/provider history ends on July 23. This evidence does not establish a
normal July 24 OHLC session or a proven zero-volume bar. The generic contract
therefore records `last_trading_session=2026-07-24` as lifecycle truth and
`price_observation_end_session=2026-07-23` with
`final_session_observation_status=no_valid_price_observation`. The lifecycle
date is not derived from provider availability.

## Contract and Implementation Remediation

Lifecycle registry v4 separates the formal lifecycle boundary from the final
valid price observation. It adds machine-validated
`price_observation_end_session`, `final_session_observation_status`, and
`trading_suspension_effective_timing`. Closing, suspension, inactive, formal
trading, and observation dates are validated as a coherent chronology while
allowing before-open, effective-time, and after-close events.

Both v2 and v3 inputs remain compatibility inputs. If
`delisting_end_date` and `last_trading_session` are both present, they must be
equal before normalization. A conflict fails closed with ticker, both field
names, and both values in `InstrumentLifecycleError`.

Freshness manifest and validation contracts advance to v5. Every ticker entry
binds both its previous and persisted file checksum. The changed-file list and
count are derived from checksum differences, independently of freshness
status, so an inactive bounded backfill is counted even though its final status
is `not_expected`. Validation reconciles the declared count, sorted unique
list, ticker checksum bindings, and staged canonical files. The trusted-main
publisher additionally materializes the current `market-data` tree and checks
the bundle against that independent baseline before installation; a forged or
inconsistent fileset fails closed.

The singleton retry route now receives the same lifecycle context as the batch
route and calls the same provider-frame validator. Post-cutoff rows receive the
same `quarantined_not_persisted` disposition and diagnostics in both routes,
are excluded before canonical merge, do not contribute to freshness, and
cannot reactivate an inactive instrument.

## Validation Evidence

| Command | Result | Duration | Notes |
|---|---:|---:|---|
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_me_sr18_lifecycle_aware_freshness.py tests/market_engine/data/test_scheduled_canonical_price_refresh.py tests/market_engine/data/test_scheduled_canonical_price_refresh_workflow.py -q` | 138 passed | 1.52 s | Lifecycle, provider/singleton, quarantine, manifest, publisher, and workflow regressions |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/run -q` | 197 passed | 2.87 s | No failures |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q` | 1432 passed, 1 failed | 5.76 s | Only the known missing local artifact |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q` | 2099 passed, 1 failed | 6.90 s | Same known failure only |
| Lifecycle v4 load/apply plus TMHC and publisher-baseline assertions | passed | 0.23 s | Registry checksum `1962d5cbf63df5c005c775bad05ad6bcc428d6b64e1ea4af3be5457a750cc43a` |
| `git diff --check` | passed | <0.1 s | No whitespace errors |

The only suite failure is
`test_compact_checksums_match_committed_files_and_local_full_runs`, because
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`
is absent. Running that exact test on clean base SHA
`a0409a49e8f8f3ef9dce352c22b039ce4387faab` reproduces the identical failure
in 0.02 seconds. No new test failure exists.

The required governance searches still report only the pre-existing explicit
trade-command handling in `scripts/portfolio/parse_trade_commands.py` and
`scripts/portfolio/portfolio_manager.py`; this remediation does not modify any
file under `scripts/`.

## Previous Canary Correction

Run `31276951551` reported 942 `updated` instruments, but EA also received a
bounded inactive backfill. The earlier audit incorrectly equated that status
count with changed price files. The actual prior changed price-file count was
943. That canary is invalidated as merge evidence because its v4 manifest did
not independently reconcile the changed fileset.

## Remediation Canary

Pending. Exactly one full-universe `publish=false` run will be dispatched only
after the locally validated changes are committed and pushed. This section will
be replaced with run, artifact, ticker, fileset, publish-job, and unchanged
`market-data` evidence from that run.

## Remaining Risks and Recovery

- The evidence supports a formal TMHC lifecycle through July 24 but does not
  supply a valid July 24 OHLC/volume observation. The explicit no-observation
  contract preserves that distinction without fabricating market data.
- Provider corrections can change historical bars. The checksum-bound staged
  fileset and independent publisher baseline must remain fail-closed.
- Rollback is a normal revert of the reviewed ME-SR23 commit. No manual edit or
  branch-history rewrite of `market-data` is permitted.
