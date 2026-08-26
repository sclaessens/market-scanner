# ME-SR26 Second Controlled Production Canary Audit

## Status

`BLOCKED` on 2026-08-26.

Primary blocker: `HOSTED_RUNNER_COMMUNICATION_LOSS_BEFORE_USABLE_EVIDENCE`.

The second controlled production canary was a new manual dispatch, not a
rerun of interrupted run `32951786805`. It survived for approximately 50
minutes but the GitHub-hosted runner lost communication with the server while
bounded history acquisition was still active. No history manifest, quality
gate, technical screening, or artifact upload completed. There was no retry.

The first canary's exact `The runner has received a shutdown signal` message
did not recur. This run instead received GitHub's hosted-runner communication
loss annotation. It is therefore not classified under the prompt's exact
`REPEATED_RUNNER_INTERRUPTION` signature rule. It nevertheless establishes a
second runner-level failure before usable evidence and requires a separate
runner/workflow hardening review before any later canary is considered.

## Pre-dispatch baseline and safety

| Field | Evidence |
|---|---|
| Baseline `main` | `7b48c89398f26cecf9dda0bf8f806191642a6c66` |
| PR #482 merge ancestry | PASS — the merge commit is the baseline `main` commit |
| `market-data` before | `95c88276763b1762cbbfbccc402ec8535268127b` |
| Worktree | clean and detached at current `origin/main` before dispatch |
| Workflow permissions | `contents: read` |
| Concurrency | repository-scoped group; `cancel-in-progress: false` |
| Job timeout | 90 minutes |
| Advisory-only boundary | PASS — no publication, Git write, canonical mutation, DATA11 approval, DATA07, DATA06, RUN31, RUN33, portfolio operation, or broker/order operation |
| Roadmap sequencing | PASS — a successful controlled canary remained required before the downstream checkpoint and RUN33 |

Exactly one new `workflow_dispatch` was authorized and performed. GitHub's
rerun functionality was not used. No second dispatch, local live-provider
experiment, manual ticker retry, or downstream execution occurred.

## Execution record

| Field | Evidence |
|---|---|
| Workflow | `Advisory OHLC History and Current Technical Screening` |
| Run / job | `32966745030` / `98170802856` |
| URL | `https://github.com/sclaessens/market-scanner/actions/runs/32966745030` |
| Event / attempt | `workflow_dispatch` / `1` |
| Actor / triggering actor | `sclaessens` / `sclaessens` |
| Branch / executed SHA | `main` / `7b48c89398f26cecf9dda0bf8f806191642a6c66` |
| Created / started | `2026-08-26T12:05:52Z` / `2026-08-26T12:05:52Z` |
| Job started / completed | `2026-08-26T12:05:55Z` / `2026-08-26T12:55:56Z` |
| Run completed | `2026-08-26T12:55:57Z` |
| Run / job conclusion | `failure` / `failure` |
| Dispatch count | `1` |
| Workflow retries | `0` |

Setup, checkout, Python setup, and dependency installation completed
successfully. `Build bounded advisory history` started at
`2026-08-26T12:06:11Z` and never received a completion timestamp. The semantic
history quality gate, current screening, artifact upload, and action post-steps
remained pending.

## Runner evidence

| Field | Evidence |
|---|---|
| Runner name / ID | `GitHub Actions 1000000178` / `1000000178` |
| Runner labels / group | `ubuntu-latest` / `GitHub Actions` |
| Runner image | NOT EXPOSED — the job log archive is unavailable |
| Runner version | NOT EXPOSED — the job log archive is unavailable |
| Runner region | NOT EXPOSED — the job log archive is unavailable |
| Exact shutdown-signal text observed | NO |
| Competing run observed | NO |
| History completion reached | NO |
| Exact repeated-runner signature | NO |

The check-run annotation is:

```text
The hosted runner lost communication with the server. Anything in your
workflow that terminates the runner process, starves it for CPU/Memory, or
blocks its network access can cause this error.
```

GitHub exposes no more specific cause through the run, job, or check APIs.
`gh run view --log` reports `log not found` for job `98170802856`; consequently
the image, runner version, region, installed package versions, provider output,
and process-level terminal text cannot be recovered. The annotation does not
establish whether infrastructure loss, resource starvation, process
termination, or blocked runner network access was causal.

No repository or workflow run overlapped this canary. Earlier runs
`32951786805` and `32956572481` had completed before this run began.

## Artifact inventory

The Actions artifact API returned:

```text
total_count: 0
```

Expected artifact name:

```text
advisory-ohlc-history-screening-32966745030
```

Actual artifact name: NOT PRODUCED.

Because the runner disappeared before the `if: always()` upload step, no
history or screening files exist to retrieve. Artifact manifest and checksum
index SHA-256 values are therefore `NOT AVAILABLE`, not failed checksums.

## History authority and provider health

No manifest or index was emitted. Counts and ratios must not be inferred as
zero:

| Metric | Result |
|---|---|
| attempted | NOT EVALUATED |
| fresh | NOT EVALUATED |
| stale | NOT EVALUATED |
| insufficient_history | NOT EVALUATED |
| missing | NOT EVALUATED |
| invalid | NOT EVALUATED |
| blocked_identity | NOT EVALUATED |
| blocked_adjustment_policy | NOT EVALUATED |
| producer fresh coverage | NOT EVALUATED |
| effective fresh coverage | NOT EVALUATED |
| analytic authority status | NOT ESTABLISHED |
| otherwise-valid lag denominator | NOT EVALUATED |
| exactly one session late | NOT EVALUATED |
| lag ratio / `0.80` threshold | NOT EVALUATED |
| widespread lag detected | NOT EVALUATED |
| global provider failure / affected count | NOT EVALUATED |

There is no complete non-fresh instrument list because no `history_index.json`
was produced. A partial provider process cannot serve as retained authority.

## Integrity and authority assertions

| Assertion | Result | Basis |
|---|---|---|
| Source HEAD binding | NOT EVALUATED | no history manifest |
| Canonical 952 universe | NOT EVALUATED | no history index |
| Canonical history policy | NOT EVALUATED | no manifest/policy binding |
| History artifact checksums | NOT EVALUATED | no artifact |
| History semantic replay | NOT EVALUATED | quality gate never ran; no artifact |
| Load-time freshness | NOT EVALUATED | no loadable artifact |
| Nullable volume semantics | NOT EVALUATED | no retained series evidence |
| Screening history binding | NOT EVALUATED | screening never ran |
| Screening policy binding | NOT EVALUATED | screening never ran |
| Technical semantic replay | NOT EVALUATED | screening never ran |
| Cross-artifact authority binding | NOT EVALUATED | neither complete artifact exists |

The workflow checkout SHA is known, but source provenance requires the history
manifest's `source_main_sha` to bind to that SHA. Without the manifest, the
source HEAD assertion cannot be upgraded to PASS.

## Technical screening and ranking

The screening step remained pending. Its run status, analytic authority,
instrument count, screened count, candidate count, ranking count, ranking gap,
setup-state distribution, outcome-label distribution, blocker distribution,
and top candidates are all `NOT EVALUATED`. No prior RUN30 ranking was treated
as current authority.

## Dependency observation

The dependency-install step succeeded, but the missing job log prevents
recovery of actual installed versions for Python, pandas, numpy, yfinance, and
curl_cffi. Material drift from the first canary is `NOT EVALUATED`; version
drift is not classified as causal without evidence.

## Repository safety

| Assertion | Result |
|---|---|
| `main` before / after | unchanged: `7b48c89398f26cecf9dda0bf8f806191642a6c66` |
| `market-data` before / after | unchanged: `95c88276763b1762cbbfbccc402ec8535268127b` |
| Commit created by workflow | NO |
| Branch created by workflow | NO EVIDENCE; workflow had read-only contents authority |
| PR created by workflow | NO |
| Generated artifact committed | NO |
| Publication | NO |
| Canonical mutation | NO |
| DATA11 approval / DATA07 / DATA06 / RUN31 / RUN33 | NO |
| Portfolio access | NO |
| Broker/order action | NO |

No Git commit appeared during the workflow window. The only new branch and PR
after the outcome are the separately created documentation/audit changes that
contain this record.

## Documentation validation

```text
git diff --check
PASS

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/sclaessens/Documents/market-scanner/.venv/bin/python -m pytest -q
2571 passed, 1 failed
```

The single failure is
`test_compact_checksums_match_committed_files_and_local_full_runs`; this clean
worktree does not contain the uncommitted local DATA06 artifact referenced by
the compact evidence index. The failure is unrelated to these documentation
changes. No runtime, test, workflow, artifact, CSV, or report file changed.

The mandatory `BUY` and `SELL` scans return only existing manual portfolio
transaction parsing/ledger strings in `scripts/portfolio`; the `tradeable`
scan returns no match. This branch adds none of those terms to `scripts/`.

## Verdict

```text
BLOCKED
```

The workflow failed before usable evidence, the history authority could not be
validated, and screening did not execute. The exact first-run shutdown-signal
signature did not recur, so the special exact-signature classification is not
used. The different hosted-runner communication-loss failure still makes a
third canary unsafe without a separate hardening review.

## Next action

Perform one documentation-and-diagnostics-only runner/workflow hardening review
that explains the hosted-runner communication loss and defines retained
checkpoint/telemetry requirements before any proposal for another canary. This
audit does not authorize a third canary, runtime remediation, or downstream
execution.
