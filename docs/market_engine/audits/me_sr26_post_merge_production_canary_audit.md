# ME-SR26 Post-Merge Production Canary Audit

## Status

BLOCKED on 2026-08-26.

Exactly one reviewed manual `workflow_dispatch` of `Advisory OHLC History and
Current Technical Screening` was executed from merged `main`. GitHub Actions
canceled the bounded history-build step before it emitted a manifest. The
quality gate, technical screening, and `if: always()` artifact upload were
skipped, leaving no artifact that could independently establish provider,
freshness, replay, screening, or volume semantics. No retry was performed.

## Pre-canary safety

| Assertion | Result |
|---|---|
| Clean canary worktree | PASS |
| Local canary checkout equals `origin/main` | PASS |
| Merge commit present | PASS — `19f207b35947040db5ee466c54a140160909c7ce` |
| Merged PR head is an ancestor | PASS — `193dc121fc6b13147ece2086426e564931c0db91` |
| Workflow path | `.github/workflows/advisory-ohlc-history.yml` |
| Workflow permission | PASS — `contents: read` |
| History build, semantic gate, screening, upload | PRESENT |
| Evidence preservation | `if: always()` and `retention-days: 14` present |
| Publication, `market-data`, downstream, portfolio, order, or broker action | ABSENT |

## Execution identity

| Field | Evidence |
|---|---|
| Workflow | `Advisory OHLC History and Current Technical Screening` |
| Event and ref | `workflow_dispatch` on `main` |
| Dispatch count | `1` |
| GitHub Actions run | `32951786805` |
| URL | `https://github.com/sclaessens/market-scanner/actions/runs/32951786805` |
| Job | `98124588792` |
| Executed source SHA | `19f207b35947040db5ee466c54a140160909c7ce` |
| Started | `2026-08-26T09:12:46Z` |
| Completed | `2026-08-26T09:16:13Z` |
| Workflow conclusion | `failure` |
| Terminal step | `Build bounded advisory history` — `cancelled` |
| Terminal log | `The operation was canceled.` |

The checkout log independently records the executed source SHA. Source checkout
therefore matches the triggering workflow SHA and contains the merged PR head.
The history builder produced no terminal JSON result before cancellation, so a
history-manifest `source_main_sha` comparison was impossible.

## Artifact evidence

The GitHub Actions artifact API returned `total_count: 0`. The expected
`advisory-ohlc-history-screening-32951786805` artifact does not exist. The
artifact upload step was skipped despite its configured `if: always()` guard
because GitHub canceled the operation before later workflow steps ran.
Consequently there are no downloaded manifest checksums to record and no
evidence was modified.

## History authority result

| Measure | Result |
|---|---|
| Attempted | NOT EVIDENCED |
| Fresh | NOT EVIDENCED |
| Stale | NOT EVIDENCED |
| Insufficient history | NOT EVIDENCED |
| Missing | NOT EVIDENCED |
| Invalid | NOT EVIDENCED |
| Blocked identity | NOT EVIDENCED |
| Blocked adjustment policy | NOT EVIDENCED |
| Producer fresh coverage | NOT EVALUATED |
| Effective fresh coverage | NOT EVALUATED |
| Analytic authority status | NOT ESTABLISHED |
| Non-fresh instrument list | UNAVAILABLE |

No provider-lag numerator, denominator, ratio, threshold result, provider
failure affected count, or global-provider-failure result was emitted. These
values must not be inferred from an incomplete run.

## Authority and integrity assertions

| Assertion | Result |
|---|---|
| Workflow checkout SHA binding | PASS |
| History-manifest source SHA binding | NOT EVALUATED — no manifest |
| Canonical 952-identity runtime reconciliation | NOT EVALUATED |
| Canonical history-policy runtime binding | NOT EVALUATED |
| Checksum-file enumeration and verification | NOT EVALUATED |
| History semantic replay | NOT EXECUTED |
| Load-time effective freshness | NOT EVALUATED |
| Screening policy binding | NOT EXECUTED |
| Technical replay | NOT EXECUTED |
| Nullable missing-volume semantics | NOT EVALUATED |

The merged workflow source points only to repository-owned authority constants,
but this canary's objective required actual runtime evidence. Source inspection
cannot substitute for the missing artifact.

## Technical screening

The semantic quality gate and technical screening steps were skipped. No
screening status, authority status, instrument count, screened count, candidate
count, ranking gap, setup distribution, outcome distribution, blocker
distribution, or top-candidate table exists. RUN30 remained audit-only because
no current ranking was produced.

## Repository safety

| Assertion | Before | After | Result |
|---|---|---|---|
| `main` | `19f207b35947040db5ee466c54a140160909c7ce` | `19f207b35947040db5ee466c54a140160909c7ce` | UNCHANGED |
| `market-data` | `95c88276763b1762cbbfbccc402ec8535268127b` | `95c88276763b1762cbbfbccc402ec8535268127b` | UNCHANGED |

No publication, generated-artifact commit, downstream production execution,
DATA11 approval, DATA07, DATA06, RUN31, RUN33, Decision Engine execution,
portfolio access, order operation, broker operation, force-push, or workflow
retry occurred. The sole new branch and draft PR contain audit documentation
only.

## Verdict

BLOCKED.

The workflow failed during real bounded history acquisition, no evidence
artifact was uploaded, and the required runtime authority and integrity
assertions could not be independently evaluated. ME-SR26 remains implemented
but not operationally validated. RUN33 remains a separate conditional future
step and was not executed or authorized.

## Next action recommendation

Open a separate reviewed remediation task to determine why GitHub canceled the
history-build operation and why the `if: always()` evidence upload did not run.
Do not dispatch another canary until that defect is understood and an explicit
new authorization is granted.
