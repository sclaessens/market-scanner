# ME-RUN32 Post-PR472 Non-Publishing Canonical Price Refresh Audit

## Status

COMPLETED WITH BLOCKERS on 2026-07-28.

ME-RUN32 executed exactly one operator-dispatched `Canonical Price Refresh`
from merged `main` with `publish=false`. It did not authorize publication, a
canary, a retry, or runtime, test, workflow, configuration, or market-data
changes.

## Execution identity

| Field | Evidence |
|---|---|
| GitHub Actions run | `30356787529` |
| URL | `https://github.com/sclaessens/market-scanner/actions/runs/30356787529` |
| Refresh job | `90266736361` |
| Skipped publish job | `90267554594` |
| Event and ref | `workflow_dispatch` on `main` |
| Source SHA | `a0409a49e8f8f3ef9dce352c22b039ce4387faab` |
| PR472 runtime head | `2fc21c75e9cb1d4ac1e1c24a8be69b5e8f519b24` |
| Input | `publish=false` |
| Started / finished | `2026-07-28T11:55:58Z` / `2026-07-28T11:59:46Z` |
| Conclusion | `failure` |

The source SHA is PR472's merge commit and contains its runtime head. Refresh
and compact-result steps succeeded. The job then failed at `Mark degraded or
failed refresh visibly`, the intended fail-visible boundary for the degraded
manifest.

## Artifact and checksum evidence

The run uploaded exactly one artifact,
`canonical-price-freshness-me-sr18-canonical-price-refresh-20260728T115623Z`.
Its GitHub artifact ID is `8687214545`. No publication bundle was created.

| Object | SHA-256 |
|---|---|
| GitHub artifact | `33d6350fc1cd1ad556572e7494c02e2d2d2cd2766a289ae3d5cefda632833479` |
| Downloaded manifest | `9ae7972af25adcf8e94e61bbf700dad50a49485bfb2f58092c443e1f0837eb0e` |
| Downloaded run log | `04e6c7d31ac53506f41dd6fbf68812d94d59c252e5373fa035c2144428f941e3` |
| Manifest content | `5679fb0c84c3f1b364b011748bc9a7bafc7a1a086be368b2acec4aa200c63d75` |

## Publication boundary proof

The log records `INPUT_PUBLISH: false`; the compact result records
`"trusted_publish": false`; and the `publish-market-data` job was skipped. The
remote branch was checked immediately before and after the run:

```text
market-data before: 95c88276763b1762cbbfbccc402ec8535268127b
market-data after:  95c88276763b1762cbbfbccc402ec8535268127b
```

The run therefore created no publication commit and changed no remote market
data. Its 931 staged files existed only in the ephemeral runner checkout.

## Canonical result

| Measure | Count |
|---|---:|
| Canonical instruments checked | 952 |
| Active / inactive retained | 948 / 4 |
| Updated | 931 |
| Already current | 4 |
| Not expected | 3 |
| Stale | 13 |
| Failed | 1 |
| Unsupported | 0 |
| Sufficient / limited analytical history | 944 / 4 |
| Unexplained insufficient history | 0 |

Expected completed sessions were `2026-07-27` for `US` and `2026-07-24` for
`NYSE`. Reason totals were 931 `VALIDATED_UPDATE_PERSISTED`, 13
`EXPECTED_SESSION_NOT_AVAILABLE`, 4 `ALREADY_CURRENT`, 3
`INACTIVE_AFTER_COMPLETED_CORPORATE_ACTION`, and 1
`RETAINED_HISTORY_ENDS_BEFORE_EXPECTED_SESSION`.

All 948 active retrievals terminated as `PROVIDER_BATCH_COMPLETE`. There were
no timeout, rate-limit, transport, invalid-OHLC, split-retry, or
singleton-revalidation classifications. PR472's singleton path remained
available but was not triggered by this provider response.

## Original 70-blocker reconciliation

All 70 historical blockers were decided: 68 automatically recovered, 2
remained blocked, and 0 were undecided.

The 44 former `PROVIDER_OHLC_INVALID` symbols were:

```text
ABT ACM ADC ALL ATR AVNT BBY BJ BRO CLS CSL CUBE DAR DCI DOCN DTE EFX EGP ELS
EPR GDDY GPC H HLT L LII LVS MS NVR ORI PAG PATH PKG RPM RVTY SSD TDG UDR UGI
VICI WBS WLK WSM WST
```

Each took the ordinary batch route, received 2 bars, passed validation, staged
2 rows, reached `2026-07-27`, finished `updated` with
`VALIDATED_UPDATE_PERSISTED`, and was automatically recovered. No invalid OHLC
or singleton fallback occurred.

The 24 former transient batch/session symbols were:

```text
MZTI NBIX NCLH NDAQ NDSN NEE NEM NET NEU NFG NFLX NI NJR NKE NLY NNN NOC NOV
NOVT NOW NRG NSC NTAP NTNX
```

Each took the ordinary batch route, received 4 bars, passed validation, staged
2 rows, reached `2026-07-27`, finished `updated` with
`VALIDATED_UPDATE_PERSISTED`, and was automatically recovered.

The two remaining historical blockers are:

| Ticker | Historical route | Terminal diagnostic | Bars | Validation | Staged | Last session | Freshness | Decision |
|---|---|---|---:|---|---:|---|---|---|
| NSA | session unavailable | `PROVIDER_BATCH_COMPLETE` | 1 | valid | 0 | `2026-07-21` | stale | blocked |
| TMHC | lifecycle boundary | provider-excluded | 0 | blocked | 0 | `2026-07-23` | failed | blocked |

NSA's complete batch contained no stageable session through expected
`2026-07-27`. TMHC was correctly excluded after its evidence-bound lifecycle
transition, but retained history ends before expected session `2026-07-24`.
Every per-ticker decision above is traceable to the checksum-identified
manifest.

## New blockers outside the original 70

Twelve symbols were newly stale:

```text
AN BHP CBT EWBC IDCC IT JBHT MIDD MTD POST SMG WM
```

Each returned `PROVIDER_BATCH_COMPLETE`, passed validation, and staged one row
through `2026-07-24`, while its expected US session was `2026-07-27`. These are
observed provider/session freshness blockers, not invalid-OHLC regressions.

## Decision

PR472's remediation is operationally compatible with the canonical refresh:
68 historical failures recovered through validated staging, invalid OHLC did
not recur, and publication safety held exactly. A green publication canary is
not authorized because NSA, TMHC, and 12 newly observed session-freshness
blockers remain fail-closed. The next action is evidence review of those 14
current blockers, not a rerun, threshold change, hidden filter, or broad
ranking layer.
