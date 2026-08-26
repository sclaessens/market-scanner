# ME-SR26 Runner Interruption Diagnostic

## Status

DIAGNOSTIC COMPLETE on 2026-08-26.

Primary classification: `TRANSIENT_GITHUB_RUNNER_INTERRUPTION_LIKELY`.

Recanary decision: `RECANARY_SAFE_WITHOUT_CODE_CHANGES`.

This decision does not execute, schedule, or automatically retry a canary. A
second controlled canary requires separate explicit authorization and is a new
validation run. ME-SR26 remains implemented but not operationally validated
until such a run succeeds.

## Baseline

| Field | Evidence |
|---|---|
| Current `main` | `08ba20c7b4c2f3c8659b105d41ae33febb117c6a` |
| PR481 merge ancestry | PASS — current `main` is the PR481 merge commit |
| `market-data` | `95c88276763b1762cbbfbccc402ec8535268127b` |
| Diagnostic worktree | clean before documentation changes |

The active documentation correctly placed runner diagnosis before any new
canary, downstream checkpoint, or RUN33 action. No baseline rewrite was made
before the diagnostic result was known.

## Run and job forensics

| Field | Evidence |
|---|---|
| Run / job | `32951786805` / `98124588792` |
| Workflow | `Advisory OHLC History and Current Technical Screening` |
| Event / attempt | `workflow_dispatch` / `1` |
| Head branch / SHA | `main` / `19f207b35947040db5ee466c54a140160909c7ce` |
| Actor / triggering actor | `sclaessens` / `sclaessens` |
| Run created / started / completed | `2026-08-26T09:12:46Z` / `2026-08-26T09:12:46Z` / `2026-08-26T09:16:13Z` |
| Job started / completed | `2026-08-26T09:12:50Z` / `2026-08-26T09:16:12Z` |
| History step started / interrupted | `2026-08-26T09:13:06Z` / `2026-08-26T09:16:10Z` |
| Active history duration | approximately 3 minutes 4 seconds |
| Run / job conclusion | `failure` / `failure` |
| History-step conclusion | `cancelled` |

The runner was version `2.336.0` on Ubuntu 24.04.4, image
`20260816.277.1`, in Azure region `eastus`. The job label was
`ubuntu-latest`; runner name was `GitHub Actions 1000000176`.

The setup, checkout, Python setup, and dependency installation steps succeeded.
The history step was canceled. The quality gate, screening, upload, and action
post-steps were skipped. The diagnostic message was:

```text
The runner has received a shutdown signal. This can happen when the runner
service is stopped, or a manually started runner is canceled.
```

It was followed by `The operation was canceled.` No Python traceback,
ME-SR26 exception, provider error, policy error, universe error, semantic
failure, explicit application timeout, out-of-memory message, exit code 137,
or disk-exhaustion message preceded the runner shutdown.

The Actions run and job APIs expose the triggering actors but no canceling
actor or cancellation-cause field. They therefore do not prove explicit user
or API cancellation. Repository metadata cannot resolve whether the runner
service stopped because of transient hosted infrastructure or another
external control-plane action.

## Competing runs and concurrency

The repository and workflow run APIs were queried for
`2026-08-26T09:10:00Z` through `2026-08-26T09:20:00Z`. Run `32951786805` was
the only repository workflow run and the only run of this workflow in that
window. No later scheduled or manually dispatched run overlapped it.

The executed workflow used:

```yaml
concurrency:
  group: advisory-ohlc-history-${{ github.repository }}
  cancel-in-progress: false
```

No job-level concurrency block exists. GitHub documents that
`cancel-in-progress: true` is required to cancel an already running member of
a concurrency group. No competing group member existed in any event.

Result: `CONCURRENCY_CANCELLATION_NOT_EVIDENCED`.

## Executed workflow analysis

The workflow was inspected from executed SHA
`19f207b35947040db5ee466c54a140160909c7ce`, not inferred from current main.

| Contract | Executed value |
|---|---|
| Runner | `ubuntu-latest` |
| Permissions | `contents: read` |
| Job timeout | `90` minutes |
| Workflow-level timeout | none separate from the job timeout |
| Checkout | `actions/checkout@v4` |
| Python | `actions/setup-python@v5`, Python 3.11 |
| Dependencies | `pip install -r requirements.txt` |
| History command | production module `build --run-id ...` |
| Quality gate | production module `quality-gate --artifact-root ...` |
| Screening | production current-technical-screening module |
| Upload | `actions/upload-artifact@v4`, `if: always()` |
| Retention | 14 days |
| Environment protection | none configured |

There is no shell trap, `timeout` command, background process, matrix,
step-level timeout, signal operation, or conditional cancellation rule. The
interruption at 3 minutes 4 seconds is far below the 90-minute timeout.

Result: `REPOSITORY_CONTROLLED_3_MINUTE_TIMEOUT_NOT_EVIDENCED`.

## Static history acquisition analysis

The production call graph is:

```text
build_advisory_ohlc_history
  -> _build_advisory_ohlc_history_impl
    -> _acquire_with_existing_adapter
      -> download_yfinance_batch
        -> yfinance.download
      -> up to 25 _download_yfinance_history fallbacks
        -> yfinance.download
```

The adapter constructs one sorted, de-duplicated batch of uniquely mapped
provider symbols for the canonical universe. Ambiguous identities are blocked
without retrieval. It then makes one batch call and at most 25 sequential
singleton fallbacks for empty/missing batch frames. The theoretical provider
call bound is therefore 26 calls: one batch plus 25 fallbacks.

Both batch and singleton adapters specify `threads=False`, `progress=False`,
`auto_adjust=False`, and a 15-second provider timeout. The repository uses no
thread pool, process pool, multiprocessing, recursion, backoff, or retry wave in
this path. Fallback exceptions are isolated into missing-provider evidence.

The only subprocess is a bounded 10-second `git rev-parse --verify HEAD` used
before acquisition for source provenance. It does not explain a shutdown three
minutes into provider execution. Relevant `SystemExit` uses are ordinary CLI
entry-point exits after command completion. No relevant `signal`, `SIGTERM`,
`SIGKILL`, `os.kill`, `os._exit`, `KeyboardInterrupt`, or resource-limit logic
exists in the production path. No obvious unbounded loop was found.

The 15-second `yfinance.download` timeout is passed to the provider library.
Static inspection cannot prove how every dependency-internal network operation
uses that value, but repository code contains no three-minute global timeout or
mechanism that terminates the runner process.

## Resource profile

The retention maximum is:

```text
952 instruments x 420 sessions = 399,840 rows
```

Six 64-bit OHLCV/date-scale values per row are about 19 MB of dense numeric
cells before indexes and object overhead. A wide batch frame, normalized
per-symbol frames, Python bar dictionaries, replay structures, and JSON
serialization can plausibly raise peak process memory into the hundreds of
megabytes; a conservative order-of-magnitude allowance remains below 1–2 GB.
Serialized series evidence should be on the order of tens to low hundreds of
megabytes, not gigabytes.

GitHub documents 4 CPUs, 16 GB RAM, and 14 GB SSD for public-repository
standard Linux `ubuntu-latest` runners. Ordinary ME-SR26 workload size is far
below those limits. Provider/library bugs or abnormal allocations cannot be
mathematically ruled out without telemetry, but the run emitted no OOM or disk
evidence.

Result: `MEMORY_EXHAUSTION_NOT_SUPPORTED_BY_STATIC_WORKLOAD_SIZE`.

## Dependency and version review

The interrupted run installed:

| Component | Version |
|---|---|
| Python | `3.11.16` |
| pandas | `3.0.5` |
| numpy | `2.4.6` |
| yfinance | `1.6.0` |
| curl_cffi | `0.16.2` |

`requirements.txt` leaves pandas, numpy, yfinance, requests, and lxml
unconstrained. Jsonschema alone is range constrained (`>=4.23,<5`). There is no
lock or constraints file. The current local deterministic environment contains
Python 3.13.13, pandas 3.0.3, numpy 2.4.5, yfinance 1.3.0, and curl_cffi 0.15.0.
The repository does not preserve the exact environment used for the last
ME-SR26 validation, so that comparison cannot establish causal version drift.

Unconstrained installation is a reproducibility risk. However, the canary log
contains no dependency exception, Python traceback, provider exception, or
process exit attributable to these versions. The runner-level shutdown message
does not establish a dependency defect. Dependency drift is therefore a risk,
not the root-cause classification.

## GitHub runner interpretation

GitHub's official workflow-cancellation reference says that server cancellation
sends a cancellation message to the runner and the runner then signals the
step process. The official runner source separately emits this exact message
from its runner-shutdown token for `ShutdownReason.UserCancelled`; its comment
states that this path covers runner shutdown by control signal, including a
stopped runner service or manually canceled runner. It does not identify which
external actor or infrastructure mechanism supplied the shutdown signal.

References:

- `https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation`
- `https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idtimeout-minutes`
- `https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency`
- `https://docs.github.com/en/actions/reference/runners/github-hosted-runners`
- `https://github.com/actions/runner/blob/32973239a4d4020aa93d027df5ff4104421074d5/src/Runner.Worker/JobRunner.cs#L125-L143`

The public run/job/check APIs expose no root-cause or canceling-actor field for
this run. Thus explicit manual/API cancellation is unresolved, while
repository concurrency and timeout explanations are not supported.

## Artifact upload assessment

The runner terminated before the later upload step could execute. This does
not demonstrate a defect in `if: always()`: that condition can preserve local
evidence after an ordinary command-level failure only while a runner remains
available. No repository-local upload design can run after the hosted VM is
gone.

Incremental checkpointing or runner-independent storage would introduce new
complexity and authority considerations. A single non-reproduced runner
interruption does not justify that redesign.

Classification: `NOT REQUIRED` before a separately authorized recanary.

## Deterministic validation

No live provider or production workflow was invoked.

```text
PYTHONPATH=src /Users/sclaessens/Documents/market-scanner/.venv/bin/python \
  -m pytest -q \
  tests/market_engine/source_refresh/test_advisory_ohlc_history.py \
  tests/market_engine/data/test_incremental_market_data_refresh.py

54 passed in 1.21s

PYTHONPATH=src /Users/sclaessens/Documents/market-scanner/.venv/bin/python \
  -m compileall -q src/market_engine

PASS

git diff --check

PASS
```

These suites cover canonical public wrappers, private provider seams, mocked
batch acquisition and bounded fallback, fail-closed provider errors, artifact
round trips, workflow safety assumptions, canonical authority bindings, and
incremental provider error preservation.

## Cause assessment

| Cause | Assessment | Basis |
|---|---|---|
| Explicit/manual cancellation | UNRESOLVED | API exposes no canceling actor or cause |
| Concurrency cancellation | NOT SUPPORTED | no competing run; `cancel-in-progress: false` |
| Workflow timeout | NOT SUPPORTED | 3m04s versus configured 90 minutes |
| Application exception | NOT SUPPORTED | no traceback or ME-SR26 exception |
| Provider failure | NOT SUPPORTED | no provider error or completed provider result |
| Resource exhaustion | NOT SUPPORTED | bounded size, 16 GB runner, no OOM/disk evidence |
| Dependency/runtime issue | UNRESOLVED | unconstrained versions are risk; no causal evidence |
| Transient hosted-runner interruption | SUPPORTED | positive runner-shutdown signal; repository causes not evidenced |

## Conclusion and authorization decision

The strongest justified primary classification is:

```text
TRANSIENT_GITHUB_RUNNER_INTERRUPTION_LIKELY
```

This is supported by positive runner-shutdown evidence, absence of competing
runs and repository cancellation logic, a timeout mismatch, reasonable static
resource bounds, healthy deterministic tests, and no application/provider
failure evidence. The exact external shutdown cause remains unavailable, so
the classification is likely rather than proven.

The recanary decision is:

```text
RECANARY_SAFE_WITHOUT_CODE_CHANGES
```

No concrete repository-controlled defect or meaningful reproducible runtime
risk was identified. Dependency pinning and checkpoint redesign are not
required before recanary on this evidence. This decision authorizes nothing by
itself.

## Repository safety and next action

No workflow dispatch, rerun, live provider call, publication, `market-data`
mutation, downstream production run, portfolio access, broker/order operation,
runtime change, workflow change, or automatic merge occurred.

The next controlled action is human review and explicit authorization of one
new ME-SR26 production canary without code changes. Do not execute it as an
automatic retry of run `32951786805`.
