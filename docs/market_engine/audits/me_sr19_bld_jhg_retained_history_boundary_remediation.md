# ME-SR19 BLD/JHG Retained-History Boundary Remediation Audit

Status: `implementation_complete_pending_review`

## Scope and Trigger

ME-SR19 is a separate, evidence-first remediation for the two retained-history
boundary failures exposed by the first post-merge ME-SR18 production canary:

```text
Canonical Price Refresh run: 30149436386
source main SHA:             d7cc5b709d40bec4acafb576e5368297c4122098
source market-data SHA:      525ef93fcc6612726ab65d0b996d8cf5fc56e5db
run_status:                  failed
publication_set_valid:       false
publication_required:        false
publish-market-data:         skipped
```

Daily Market Scan received only a skipped workflow envelope. No analysis ran.
The canary also reported 44 `PROVIDER_OHLC_INVALID` and 26
`EXPECTED_SESSION_NOT_AVAILABLE` results. Those 70 active-instrument blockers
are explicitly outside ME-SR19 and remain unresolved.

## Authoritative Contract and Data Ownership

The lifecycle registry defines `status_effective_date` as the first inactive
day and `delisting_end_date` as the exact final retained-history date. For an
inactive record, the effective date must follow the final trading date. Both
the producer and trusted consumer recompute the CSV boundary and fail closed
when the actual end date differs from the governed end date.

Inactive instruments are retained but never provider-refreshed. Their
`provider_identity` remains null. The `market-data` branch owns the canonical
price CSVs; executable code, configuration, tests, and governance documents
remain on `main`. Historical price corrections therefore use a reviewable
feature branch and draft pull request targeting `market-data`. No direct push
to either protected branch is part of this remediation.

## Before-Fix Reproduction

The current `market-data` source was copied into an isolated temporary
worktree. An offline full refresh used an injected empty provider, so no
network or real provider request occurred.

| Ticker | Lifecycle | Expected end | CSV range | Data rows | SHA-256 | Boundary | Validation | Provider |
|---|---|---:|---|---:|---|---|---|---|
| BLD | inactive from 2026-07-01 | 2026-06-30 | 2026-06-30 through 2026-07-02 | 3 | `f5503fa97cc9bfe651728dd27419249a1f329c68bf619f9126c6d1a36fb0f0cb` | `extends_after` | blocked | none |
| JHG | inactive from 2026-07-01 | 2026-06-30 | 2026-06-30 through 2026-07-02 | 3 | `05f370cf792ea44a360cf61501b564054167fa28c72d038f16844a6caa7be5fa` | `extends_after` | blocked | none |

Both ticker rows reported:

```text
freshness_status=failed
reason_code=RETAINED_HISTORY_EXTENDS_AFTER_DELISTING
retained_history_boundary_reason_code=RETAINED_HISTORY_EXTENDS_AFTER_DELISTING
publication_set_valid=false
publication_required=false
```

For both files, the July 1 and July 2 rows repeated the prior adjusted close in
every OHLC field and recorded zero volume. The June 30 observations had
non-zero volume and non-flat intraday ranges.

## Primary Evidence

Evidence was retrieved on 2026-07-27.

| Ticker | Source | Filing/publication date | Effective fact established |
|---|---|---:|---|
| BLD | [TopBuild Form 8-K, Items 2.01, 3.01, and 3.03](https://www.sec.gov/Archives/edgar/data/1633931/000110465926079876/tm2618991d10_8k.htm) | 2026-07-01 | QXO completed the acquisition on July 1; NYSE filed Form 25; BLD trading was suspended before the July 1 NYSE open; former shares became merger-consideration rights. |
| BLD | [QXO acquisition-completion release, SEC Exhibit 99.1](https://www.sec.gov/Archives/edgar/data/1236275/000110465926079864/tm2618991d7_ex99-1.htm) | 2026-07-01 | The acquisition completed and TopBuild shares stopped trading before the July 1 market open. |
| JHG | [Janus Henderson Form 8-K, Items 2.01, 3.01, and 3.03](https://www.sec.gov/Archives/edgar/data/1274173/000110465926079401/tm2619303d2_8k.htm) | 2026-06-30 | The take-private completed June 30; outstanding shares converted to cash rights; the company requested an NYSE halt before the July 1 open, listing withdrawal, and Form 25 filing. |
| JHG | [Janus Henderson completion release, SEC Exhibit 99.1](https://www.sec.gov/Archives/edgar/data/1274173/000110465926079401/tm2619303d2_ex99-1.htm) | 2026-06-30 | The transaction completed, the shares converted to cash rights, and the ordinary shares were delisted from NYSE. |

July 1 and July 2 were ordinary US-equities exchange sessions under the
repository calendar. That general calendar fact does not make the two
security-specific rows valid: the official filings establish a halt before
the July 1 open and no continuing regular-way share rights. The zero-volume,
flat carry-forward rows are supporting data-shape evidence, not the
corporate-action authority.

## Ticker Decisions

### BLD — Decision A

June 30, 2026 was the last valid regular-way trading day. The July 1 closing
date is distinct from a trading date because the official source states that
trading was suspended before that day's open. The existing lifecycle record
is correct:

```text
delisting_end_date=2026-06-30
status_effective_date=2026-07-01
```

Only the zero-volume July 1 and July 2 carry-forward rows are removed.

### JHG — Decision A

The merger completed on June 30, each public share became a cash right, and
the issuer requested a halt before the July 1 open. June 30, 2026 was
therefore the last valid regular-way trading day. The existing lifecycle
record is correct:

```text
delisting_end_date=2026-06-30
status_effective_date=2026-07-01
```

Only the zero-volume July 1 and July 2 carry-forward rows are removed.

No lifecycle configuration, production runtime, provider mapping, or other
ticker data changes are justified.

## Implemented Correction

The data correction is isolated on:

```text
branch: me-sr19-bld-jhg-retained-history-boundary-remediation-data
base:   market-data@525ef93fcc6612726ab65d0b996d8cf5fc56e5db
```

It changes only:

```text
data/processed/BLD.csv
data/processed/JHG.csv
```

The June 30 rows, headers, column order, encoding, newline convention, and all
pre-boundary values remain byte-for-byte represented as before. No
replacement row is synthesized and no missing session is filled.

The main-target branch changes only regression coverage and governance
documentation. It does not change lifecycle configuration or runtime code.

## After-Fix Validation

| Ticker | Lifecycle | Expected end | CSV range | Data rows | SHA-256 | Boundary | Validation | Provider |
|---|---|---:|---|---:|---|---|---|---|
| BLD | inactive from 2026-07-01 | 2026-06-30 | 2026-06-30 through 2026-06-30 | 1 | `c9ed7cb9f30af004d33cad6e11e0d7e796fec1c45b58c4fb7b0024767c08de97` | `aligned` | valid | none |
| JHG | inactive from 2026-07-01 | 2026-06-30 | 2026-06-30 through 2026-06-30 | 1 | `2e6567a53a8a6280a173f968a45072688cc5a3fe2c154096acefe548e341671e` | `aligned` | valid | none |

The same isolated offline refresh reported for both tickers:

```text
freshness_status=not_expected
reason_code=INACTIVE_AFTER_COMPLETED_CORPORATE_ACTION
history_coverage_status=retained_inactive
retained_history_boundary_status=aligned
retained_history_boundary_reason_code=RETAINED_HISTORY_ENDS_ON_EXPECTED_SESSION
provider_identity=null
rows_added=0
validation_status=valid
```

The focused BLD/JHG condition is corrected. The isolated universe run remains
`degraded` because the injected empty provider intentionally cannot refresh
unrelated active instruments. It nevertheless proves
`publication_set_valid=true`, `publication_required=false`, and
`approval_generated=false`. It does not prove full-universe publication
readiness.

## Regression Coverage

The ME-SR18 lifecycle suite now explicitly covers the corrected BLD/JHG shape:

- an exact June 30 boundary is aligned and valid;
- the retained June 30 bytes reach staging unchanged;
- inactive instruments do not call the provider;
- a July 1 or July 2 tail remains blocked by the existing generic contract;
- no ticker-specific runtime exception exists;
- degraded and failed runs retain their existing non-publication behavior.

The repository-required bundled workspace Python was used with the local
pure-Python site packages because the local virtual environment does not
provide the complete working runtime dependency set. Final results:

```text
test_me_sr18_lifecycle_aware_freshness.py:       57 passed
test_scheduled_canonical_price_refresh.py:       43 passed
test_scheduled_canonical_price_refresh_workflow.py: 9 passed
tests/market_engine/data:                       379 passed
tests/market_engine:                           1,404 passed
complete repository:                           2,071 passed
workflow YAML parse:                            2 passed
actionlint:                                     not installed; not added
git diff --check:                               passed on both branches
```

The mandatory governance greps found only pre-existing portfolio transaction
input and accounting language for BUY/SELL outside Decision Engine. They found
no new hit in the ME-SR19 diff and no non-Decision-Engine `tradeable` source
logic. ME-SR19 changes neither allocation nor recommendation authority.

## Remaining Risk and Canary Preconditions

The data-only correction intentionally does not hand-edit a publication
manifest. After both draft pull requests are reviewed and the data correction
is merged through GitHub, a trusted `main` Canonical Price Refresh must create
and validate the next manifest v3 against the corrected CSV checksums. Only a
fully successful refresh may publish and allow the existing workflow-success
gate to start Daily Market Scan.

Before that canary can be green, the separate 44
`PROVIDER_OHLC_INVALID` and 26 `EXPECTED_SESSION_NOT_AVAILABLE` results still
require their own evidence and remediation. ME-SR19 neither changes nor
bypasses them. No workflow dispatch or Daily Market Scan is authorized by
this audit.

The data change is recoverable by reverting its dedicated commit, or by
restoring the four removed source rows in a new reviewed `market-data` pull
request. Direct branch mutation and force-push are not part of the recovery
procedure.
