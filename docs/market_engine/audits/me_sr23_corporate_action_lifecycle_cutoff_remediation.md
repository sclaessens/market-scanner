# ME-SR23 Corporate-Action Lifecycle Cutoff Remediation Audit

Status: `implementation_complete_pending_non_publishing_canary`

## Incident and Impact

Scheduled Canonical Price Refresh run `31243056845` on August 8, 2026 failed
closed. Its freshness artifact was
`canonical-price-freshness-me-sr18-canonical-price-refresh-20260808T060333Z`.
The run staged 942 valid price-file updates, found four already-current
instruments and three retained-inactive instruments, but classified EA and NSA
as stale and TMHC as an invalid retained history. Atomic publication correctly
prevented all staged changes from reaching `market-data`; the publish job was
skipped and the branch remained at `95c88276763b1762cbbfbccc402ec8535268127b`.

EA's provider rows dated August 5 and August 6 existed only in the isolated
staging result. The pre-run canonical EA file ended on July 23, and the failed
run did not partially publish it.

## Primary Evidence and Date Semantics

| Ticker | Last trading session | Transaction closing | Suspension / inactive boundary | Evidence |
|---|---:|---:|---:|---|
| EA | 2026-08-04 | 2026-08-04 | before open 2026-08-05 | [EA completion announcement](https://www.ea.com/news/ea-announces-completion-of-acquisition), [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/712515/000114036126031157/ef20079099_8k.htm) |
| NSA | 2026-07-21 | 2026-07-22 | before open 2026-07-22 | [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/1618563/000110465926085888/tm2620871d8_8k.htm) |
| TMHC | 2026-07-23 | 2026-07-24 | 2026-07-24 | [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/1562476/000119312526316037/d148123d8k.htm) |

The TMHC filing describes an intended suspension following the close on July
24, while the approved provider and independent market-history observations
end on July 23. ME-SR23 records July 24 as the transaction closing and
suspension/inactive boundary, and July 23 as the observed last trading
session. This explicitly preserves the evidence discrepancy instead of
silently treating the closing date as a price-session requirement.

## Root Cause

Lifecycle registry v2 overloaded `delisting_end_date` as both an explicit
final price boundary and, for TMHC, a transaction closing date. The freshness
and retained-history validators then required equality to that overloaded
date. EA and NSA had no governed lifecycle records, so they remained active;
EA provider bars after its real trading cutoff passed ordinary active-ticker
validation because the expected session was August 7.

## Contract and Implementation Decision

Lifecycle registry v3 adds separately validated
`last_trading_session`, `transaction_closing_date`,
`trading_suspension_effective_date`, and `inactive_effective_date`. The legacy
`delisting_end_date` remains a compatibility alias for an explicitly governed
last-trading boundary; it is never interpreted as a closing date. Freshness,
maximum accepted provider date, retained-history validation, manifest
bindings, and consumer reconciliation use `last_trading_session`.

An inactive instrument whose canonical history ends before its governed
cutoff receives one normal bounded backfill through that cutoff. Provider rows
after the cutoff are removed before merge, never extend freshness, and produce
diagnostics containing ticker, row date, cutoff, lifecycle event, provenance
checksum, provider, retry and `quarantined_not_persisted` disposition. Trusted
lifecycle provenance permits the run to continue with the bounded valid rows;
missing or contradictory lifecycle metadata still fails closed. Already
aligned inactive histories never call the provider.

Manifest and trusted validation contracts advance to v4. Atomic publication,
exact-fileset validation, source-SHA binding and the trusted-main publisher
gate are unchanged.

## Validation Evidence

- Targeted lifecycle/refresh/workflow tests: 124 passed.
- `tests/market_engine/run`: 197 passed.
- `tests/market_engine`: 1,418 passed, one pre-existing missing local artifact
  failure.
- Full repository suite: 2,085 passed, the same one pre-existing failure.
- The unchanged base SHA `a0409a49e8f8f3ef9dce352c22b039ce4387faab`
  reproduces that failure in
  `test_compact_checksums_match_committed_files_and_local_full_runs` because
  `artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`
  is not committed in a clean worktree.

## Canary and Publication

The non-publishing canary result will be appended after the branch is pushed.
The `market-data` SHA must remain unchanged and the publish job must be skipped.

Production publication cannot be trusted from a feature branch: the workflow
requires `refs/heads/main`. After review and merge, an operator may run exactly
one `publish=true` dispatch only if the recorded canary is fully publishable.

## Remaining Risks and Recovery

- TMHC's issuer filing language and observed final price date disagree; both
  dates remain explicit and provenance-bound.
- A provider can emit post-cutoff rows again; they are now quarantined, but the
  diagnostic must remain visible in every affected manifest.
- Rollback is a normal revert of the ME-SR23 main commit. No manual edit to
  `market-data` is permitted. If a published bundle is later shown invalid,
  restore the last known-good data commit through the normal reviewed,
  validated publisher path rather than rewriting branch history.
