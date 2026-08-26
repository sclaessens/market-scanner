# ME-SR27 Advisory History Runner / Workflow Hardening Review

## Status and decision

Story ID: `ME-SR27`

Status: `REVIEW COMPLETE` on 2026-08-26.

Remediation gate:

```text
MINIMAL_HARDENING_REQUIRED_BEFORE_THIRD_CANARY
```

ME-SR27 is a diagnostic, architecture, observability, evidence-retention, and
remediation-design sprint. It makes no runtime or workflow change and does not
authorize a third production canary. The next implementation story is:

```text
ME-SR28 — Bounded Advisory History Acquisition Observability and Diagnostic Retention
```

The minimum justified design preserves the existing history artifact and all
analytic gates. It adds deterministic acquisition chunks within one sequential
job, structured heartbeat and lightweight resource telemetry, runtime version
capture before acquisition, atomic diagnostic-only checkpoints, and a unique
immutable Actions artifact upload after every completed chunk. Final history
assembly remains impossible until all 952 identities reconcile exactly.

## Baseline

| Field | Evidence |
|---|---|
| Current `main` | `aaa6e9833011f943353fd2b39a8fd510312dfcbd` |
| PR #483 merge ancestry | PASS — current `main` is the PR #483 merge commit |
| Current `market-data` | `95c88276763b1762cbbfbccc402ec8535268127b` |
| Review branch | `codex/me-sr27-advisory-history-runner-hardening-review` |
| Worktree before review | clean |
| Workflow | `.github/workflows/advisory-ohlc-history.yml` |
| History policy | `config/market_engine/advisory_ohlc_history_policy.json` |
| Production entry point | `python -m market_engine.source_refresh.advisory_ohlc_history build` |
| Public builder | `build_advisory_ohlc_history` |
| Private construction seam | `_build_advisory_ohlc_history_impl` |
| Production provider seam | `_acquire_with_existing_adapter` |
| Existing batch adapter | `download_yfinance_batch` |
| Existing singleton adapter | `_download_yfinance_history` |

The requirements file declares `pandas`, `numpy`, `yfinance`, `requests`, and
`lxml` without version constraints. Only `jsonschema>=4.23,<5` is bounded.
There is no constraints or lock file. The clean local environment used for
static inspection contains Python 3.13.13, pandas 3.0.3, numpy 2.4.5,
yfinance 1.3.0, and curl_cffi 0.15.0. Canary 1 installed Python 3.11.16,
pandas 3.0.5, numpy 2.4.6, yfinance 1.6.0, and curl_cffi 0.16.2. The second
canary's exact installed versions are unavailable because its job log did not
survive.

The governed policy remains unchanged:

| Policy field | Value |
|---|---:|
| Indicator warm-up | 200 sessions |
| Safety margin | 52 sessions |
| Minimum history | 252 sessions |
| Maximum history | 420 sessions |
| Request window | 700 calendar days |
| Widespread one-session lag threshold | 0.80 |
| Minimum fresh screening coverage | 0.99 |
| Maximum individual fallbacks | 25 |
| Retention | 14 days |
| Price basis | `unadjusted_ohlc` |
| Adjustment policy | `provider_reported_unadjusted_with_adj_close_excluded` |

The active documentation had not drifted. It placed runner/workflow hardening
before a successful canary, the human approval/DATA07/DATA06/RUN31 checkpoint,
and conditional RUN33.

## Two-canary evidence

| Field | Canary 1 | Canary 2 |
|---|---|---|
| Run | `32951786805` | `32966745030` |
| Executed SHA | `19f207b35947040db5ee466c54a140160909c7ce` | `7b48c89398f26cecf9dda0bf8f806191642a6c66` |
| History step | `09:13:06Z`–`09:16:10Z` | `12:06:11Z`–job loss at `12:55:56Z` |
| History duration | 3m04s | approximately 49m45s |
| Runner failure | shutdown signal / operation canceled | hosted runner lost communication with server |
| Retained job log | YES | NO — `log not found` |
| Retained artifact | NO | NO |
| Application failure evidenced | NO | NO |
| Provider failure evidenced | NO | NO |
| Quality gate reached | NO | NO |
| Screening reached | NO | NO |

Canary 1 retained setup and dependency output plus the runner-level shutdown
diagnostic. Canary 2 retained only run/job/check metadata. Its job API still
showed history acquisition `in_progress`, all later steps pending, and zero
artifacts. The exact terminal signatures differ, so this review does not invent
one common root cause. The shared operational fact is two runner-level losses
during the same long, currently opaque history step before durable progress
evidence existed.

Evidence links:

- `https://github.com/sclaessens/market-scanner/actions/runs/32951786805`
- `https://github.com/sclaessens/market-scanner/actions/runs/32966745030`

## Complete production execution path

```text
.github/workflows/advisory-ohlc-history.yml
  -> checkout / Python 3.11 / unconstrained dependency install
  -> advisory_ohlc_history.run_command(build)
  -> build_advisory_ohlc_history
  -> canonical universe and policy validation
  -> immutable Git HEAD resolution
  -> _build_advisory_ohlc_history_impl
  -> _acquire_with_existing_adapter
  -> 952 mapped identities -> 952 unique provider symbols
  -> download_yfinance_batch
  -> yf.download(... threads=False, progress=False, timeout=15)
  -> synchronous yfinance _download_one per provider symbol
  -> cookie/crumb acquisition + Yahoo chart request per symbol
  -> normalize combined DataFrame into one frame per symbol
  -> up to 25 singleton yf.download fallbacks
  -> per-frame bar normalization
  -> per-instrument classification and series construction
  -> global semantic replay / lag / coverage / provider-failure calculation
  -> observation and manifest digest generation
  -> final history files and checksum index
  -> public production quality-gate load and semantic replay
  -> current technical screening over validated history
  -> final upload-artifact step
```

## Execution-phase observability map

| Phase | Entry / exit | Work volume | Network | CPU / memory | Filesystem | Current logging | Durable evidence | Observability |
|---|---|---:|---|---|---|---|---|---|
| Checkout/setup/install | workflow steps | one repository and environment | dependency install | low/moderate | checkout and environment | step logs | server log only if runner communicates | OBSERVABLE |
| Authority bootstrap | builder entry to provider call | 952 identities; two governed files; one Git lookup | none | low | reads universe/policy; no output | none | none | OPAQUE |
| Primary yfinance acquisition | `download_yfinance_batch` entry/exit | 952 synchronous ticker histories | high | growing shared DataFrames | yfinance cache only; no governed output | progress disabled; errors summarized only after completion | none | OPAQUE |
| Singleton fallback acquisition | batch return through fallback loop | at most 25 empty histories | high | per-frame normalization | no governed output | none | none | OPAQUE |
| Frame normalization | batch return and `_frame_bars` | up to 952 frames and 420 retained bars each | none | DataFrame slicing, iteration, Decimal/string objects | none | none | none | OPAQUE |
| Classification / series materialization | provider return through classification loop | exactly 952 rows; at most 399,840 retained bars | none | Python dictionaries/lists and session resolution | none | none | none | OPAQUE |
| Semantic replay and digests | classification exit through payload construction | 952 rows plus every retained bar | none | replay, canonical JSON, SHA-256, duplicate in-memory serialization | none | none | none | OPAQUE |
| Final history write | destination creation through checksum index | manifest, index, eligibility, up to 952 series | none | serialization | final directory created only near end | only final CLI JSON after success | local workspace until upload | PARTIALLY_OBSERVABLE |
| Quality gate | separate workflow step | exact artifact enumeration and full semantic replay | none | file reads, hashes, replay | reads history artifact | final status only | local history plus later upload if runner survives | PARTIALLY_OBSERVABLE |
| Technical screening | separate workflow step | 952 classifications and ranking | none | indicator calculation, ranking, serialization | writes screening artifact | final status only | local screening plus later upload if runner survives | PARTIALLY_OBSERVABLE |
| Evidence upload | final workflow step | history and screening trees | Actions service | compression | reads artifact roots | action progress | immutable GitHub artifact after completion | OBSERVABLE |

The tens-of-minutes blind spot begins immediately before the production
provider call. The builder does not create the destination directory until all
provider work, normalization, classification, semantic replay, and digest
construction finish. Any interruption before line-of-authority finalization
therefore loses all governed progress evidence.

## Provider wall-clock analysis

### Actual current call shape

Static canonical reconciliation produced:

```text
canonical identities:       952
mapped identities:          952
unique provider symbols:    952
ambiguous provider symbols: 0
repository yf.download calls: 1 primary + at most 25 singleton fallbacks
```

The repository-level count of at most 26 `yf.download` invocations does not
mean at most 26 HTTP requests. In yfinance 1.6.0, `download(...,
threads=False)` iterates the ticker list and calls `_download_one` synchronously
for every ticker. `_download_one` calls `Ticker.history`, which makes a Yahoo
chart request for that ticker. The first request may also require cookie and
crumb operations. An HTTP error triggers a second request after switching
cookie strategy. Configured transient retries default to zero, so an unbounded
retry wave is not present, but one invocation is still a multi-request path.

The repository's scalar `timeout=15` reaches curl_cffi as a request timeout.
curl_cffi 0.16.2 maps a scalar non-stream timeout to libcurl
`TIMEOUT_MS`; it does not create a deadline around the 952-ticker loop.
yfinance cookie/crumb helpers also have separate 30-second defaults in paths
where `_make_request` calls `_get_cookie_and_crumb()` without forwarding the
chart timeout.

Even ignoring cookie/crumb and error-response retries, 977 possible ticker
history operations multiplied by 15 seconds represent 14,655 seconds
(4h04m15s) of request-level allowance. This is not a prediction of normal
runtime; it proves that 15 seconds cannot bound the aggregate provider phase.
The workflow's outer 90-minute job timeout is the only current whole-job wall
cap, and it does not preserve a later upload when the runner disappears.

```text
PROVIDER_WALL_CLOCK_BOUND: NOT_GLOBALLY_BOUNDED
```

This classification applies to the application/provider phase. Individual
curl_cffi requests are bounded by their own timeout settings, and the complete
Actions job is externally capped at 90 minutes. Repository code has no
provider-phase deadline, chunk deadline, last-progress deadline, or aggregate
elapsed-time check.

Primary source references:

- `https://github.com/ranaroussi/yfinance/blob/1.6.0/yfinance/multi.py`
- `https://github.com/ranaroussi/yfinance/blob/1.6.0/yfinance/data.py`
- `https://github.com/ranaroussi/yfinance/blob/1.6.0/yfinance/scrapers/history.py`
- `https://github.com/ranaroussi/yfinance/blob/1.6.0/yfinance/config.py`
- `https://github.com/lexiforest/curl_cffi/blob/v0.16.2/curl_cffi/requests/utils.py`

## Approximately 50-minute silent-window hypothesis matrix

| Hypothesis | Assessment | Evidence |
|---|---|---|
| A. Serial yfinance ticker behavior | SUPPORTED | yfinance 1.6.0 synchronously loops all 952 tickers when `threads=False` |
| B. Timeout is per HTTP operation, not total acquisition | SUPPORTED | repository passes 15 to history; curl_cffi applies it to each request; no aggregate deadline exists |
| C. Dependency retry/backoff wave | NOT SUPPORTED | yfinance retries default to zero; one cookie-strategy response retry exists, but no configured exponential retry wave |
| D. Provider rate limiting | PLAUSIBLE | yfinance explicitly detects HTTP 429; no provider log survived to prove it occurred |
| E. Large DataFrame normalization | PLAUSIBLE | one combined frame is normalized for 952 symbols after serial retrieval; no phase timing exists |
| F. Python object/materialization pressure | PLAUSIBLE | up to 399,840 bars are converted into nested dictionaries and exact decimal strings; no RSS telemetry exists |
| G. JSON serialization/checksum generation | PLAUSIBLE | complete index/series structures are canonicalized for observations and per-file checksums before writing; no marker distinguishes this phase |
| H. Disk pressure | UNRESOLVED | no disk telemetry survived; governed final writes occur late, so disk is less explanatory for the early provider window but cannot be excluded |
| I. CPU starvation | UNRESOLVED | GitHub's annotation lists starvation as possible; no CPU/load/RSS telemetry exists |
| J. Runner network starvation or loss | SUPPORTED AS TERMINAL CONDITION | GitHub recorded server communication loss; it does not establish when degradation began or whether provider traffic contributed |
| K. Hosted-runner infrastructure failure | PLAUSIBLE | two runner-level failures occurred, but GitHub exposes no causal infrastructure field for canary 2 |

The supported mechanisms explain how a long silent step is possible. They do
not prove which condition caused the final runner communication loss.

## Instrumentation gap review

| Metric / signal | Classification | Reason |
|---|---|---|
| Phase start/end with monotonic elapsed time | REQUIRED_BEFORE_NEXT_CANARY | separates provider, normalization, replay, serialization, and write time |
| Canonical identity and provider-symbol totals before network | REQUIRED_BEFORE_NEXT_CANARY | proves expected work volume before the opaque phase |
| Structured heartbeat at least every 60 seconds | REQUIRED_BEFORE_NEXT_CANARY | distinguishes a live blocked request from total process silence while runner communication exists |
| Completed chunk/identity counters and fallback budget use | REQUIRED_BEFORE_NEXT_CANARY | proves forward progress and preserves the global max-25 policy |
| Last-progress timestamp and current chunk/phase | REQUIRED_BEFORE_NEXT_CANARY | localizes the last successful boundary |
| Python and selected package versions before acquisition | REQUIRED_BEFORE_NEXT_CANARY | prevents a second unrecoverable dependency record |
| Runner OS label plus source/universe/policy digests before acquisition | REQUIRED_BEFORE_NEXT_CANARY | binds operational diagnostics to the intended run authority without granting analytic authority |
| Process RSS, system available memory, load average, and disk free | REQUIRED_BEFORE_NEXT_CANARY | directly tests starvation/disk hypotheses named by GitHub with standard Linux/stdlib facilities |
| Completed/empty/error frame counts at each chunk exit | REQUIRED_BEFORE_NEXT_CANARY | distinguishes provider response health from later calculation work |
| Output file count and bytes at final-write boundaries | USEFUL_BUT_OPTIONAL | helps size analysis but is not needed to retain acquisition progress |
| Per-symbol latency histogram and sanitized provider error classes | USEFUL_BUT_OPTIONAL | useful for later tuning; should not produce 952 log lines |
| Raw HTTP bodies, cookies, crumbs, full request URLs, or headers | NOT_JUSTIFIED | secret/privacy and log-volume risk; not required for causal phase localization |
| Full process/environment dumps | NOT_JUSTIFIED | excessive, potentially sensitive, and lower value than selected metrics |

## Minimal heartbeat and progress model

The next implementation must emit one structured English log record:

- at every phase start and exit;
- at every chunk start and exit;
- every 60 seconds while a provider chunk remains active; and
- immediately before and after final assembly, quality-gate replay, and
  screening.

Required operational fields are:

```text
event
run_id
source_main_sha
phase
chunk_index
chunk_count
completed_identities
total_identities
fallbacks_used
fallbacks_max
elapsed_seconds
seconds_since_progress
process_rss_bytes
system_memory_available_bytes
disk_free_bytes
load_average_1m
timestamp_utc
```

Operational timestamps and resource values are explicitly nondeterministic
telemetry and must never enter history observations, ranking, freshness,
checksums, or analytic decisions. A daemon heartbeat may read shared progress
state but may not mutate provider results. Log cadence is bounded to avoid
spam; no ticker-by-ticker success output is required.

Heartbeat logging alone is insufficient. Canary 2 proves a hosted runner can
lose its entire retrievable job log. Durable progress therefore requires an
upload completed before the final artifact step.

## Chunking options comparison

| Option | Benefits | Costs / risks | Decision |
|---|---|---|---|
| A. One 952-symbol batch plus telemetry | smallest provider change; unchanged request shape | still one opaque failure domain; no durable checkpoint until the end; heartbeat can vanish with VM | REJECTED AS SUFFICIENT; heartbeat retained as a component |
| B. Fixed canonical chunks in one sequential job | bounded progress units; lower per-call materialization; deterministic order; enables upload boundaries; one runner/session and no concurrency | more provider invocations and cookie/rate-limit exposure; chunk contract and global fallback accounting required | SELECTED |
| C. Matrix or one job per chunk | completed jobs retain artifacts after another runner fails; natural isolation | many runners/sessions, concurrency and rate-limit risk, cross-job authority envelope, artifact transfer/merge complexity, higher cost | P2_OPTIONAL only if the selected same-job design remains inadequate |
| D. Provider adapter replacement or parallel redesign | could improve throughput | new provider/request semantics, compatibility risk, threshold effects, speculative root-cause response | REJECTED |

The selected design uses fixed slices of the canonical instrument ordering,
with an implementation target of at most 64 identities per primary chunk.
Chunk size is an operational constant, not policy authority. Chunks execute
sequentially within the existing job. Each chunk has an explicit wall-clock
cap and writes through a temporary path followed by atomic rename. The
implementation must fail closed when a chunk exceeds its cap; it must not
silently skip or retry the chunk.

Primary batch chunks do not consume the fallback budget. After every primary
chunk completes, missing frames are reconciled in the original canonical
order and at most 25 singleton fallbacks execute globally, not per chunk. This
preserves the governed fallback ceiling and deterministic selection order.

Chunking changes operational request grouping, so semantic equivalence must be
proved with deterministic/mock tests. It does not change price basis, bar
normalization, classification, provider-lag detection, fresh coverage, or
final replay.

## Checkpoint and partial-evidence authority contract

Partial evidence is never authoritative.

Every completed acquisition chunk writes an atomic diagnostic checkpoint under
a distinct non-final root, for example:

```text
artifacts/market_engine/advisory_ohlc_history_diagnostics/<run_id>/
  preflight/runtime_environment.json
  checkpoints/chunk-000/checkpoint.json
  checkpoints/chunk-001/checkpoint.json
```

Minimum checkpoint content:

```text
schema_version: market-engine-advisory-ohlc-history-diagnostic-checkpoint-v1
complete: false
run_id
source_main_sha
canonical_universe_sha256
canonical_universe_identity_digest
history_policy_sha256
phase
chunk_index
chunk_count
chunk_identity_digest
chunk_attempted_identity_count
chunk_completed_identity_count
cumulative_attempted_identity_count
cumulative_completed_identity_count
successful_frame_count
empty_frame_count
provider_error_count
fallbacks_used
generated_at
authority_boundary: diagnostic_only_partial_never_analytic
```

The checkpoint contains no `analytic_authority_status`, no fresh-coverage
claim, no history manifest schema, and no final artifact digest. It uses a
different root, schema, and artifact name from production history. Existing
history loaders require `manifest.json`, exact final schemas, exact file
enumeration, checksums, universe/policy bindings, and full 952 reconciliation;
screening accepts only that validated context. RUN33 accepts only validated
screening/history bindings. Checkpoint roots therefore fail closed at every
existing production consumer.

Each completed checkpoint is uploaded immediately with a unique immutable
artifact name such as:

```text
advisory-ohlc-history-diagnostic-<run_id>-chunk-000
```

`actions/upload-artifact@v4` artifacts are immutable and immediately available
after upload. Unique names are mandatory; v4 does not allow multiple jobs or
steps to mutate one artifact. Retention remains 14 days.

Local chunk data required for final assembly may remain in a separate
non-authoritative staging root. It is never exposed as a valid history
artifact. Final assembly must require:

- the exact governed run/source/universe/policy envelope;
- every expected chunk exactly once;
- every canonical identity exactly once;
- no unexpected or duplicate identity;
- a cumulative count of exactly 952;
- the single global fallback budget of at most 25;
- successful semantic replay over the assembled data; and
- the existing final manifest, observation digest, checksum index, and loader
  validation unchanged.

Any missing, duplicate, mismatched, corrupt, incomplete, or stale checkpoint
prevents final artifact construction. This design does not add resume. Every
canary starts from an empty run-specific staging root.

Official artifact behavior reference:

- `https://github.com/actions/upload-artifact/blob/v4/README.md`
- `https://github.com/actions/upload-artifact/blob/v4/docs/MIGRATION.md`

## Evidence survivability matrix

`YES*` means only evidence successfully emitted or uploaded before the failure
boundary survives.

| Failure class | Console log | Workspace file | Later `if: always()` upload | Earlier per-chunk Actions artifact | Step summary | External storage |
|---|---|---|---|---|---|---|
| Ordinary provider exception | YES | YES | YES if runner remains healthy | YES* | YES | YES* |
| Ordinary Python exception | YES | YES | YES if runner remains healthy | YES* | YES | YES* |
| Quality-gate failure | YES | YES; final history already exists | YES | YES* | YES | YES* |
| Process SIGTERM while runner remains alive | partial YES | prior atomic files YES | usually YES if runner converts termination to step failure; not guaranteed for runner-level termination | YES* | partial / runner-dependent | YES* |
| Runner service shutdown | server-received prefix only | inaccessible after runner loss | NO | YES* | not reliable | YES* |
| Hosted-runner communication loss / VM disappearance | not reliable; canary 2 retained none | NO | NO | YES* | NO | YES* |
| GitHub Actions job timeout | partial log likely | no durable guarantee after job termination | NO guarantee; later step may not start | YES* | not reliable | YES* |

A local checkpoint plus one final `if: always()` upload cannot solve runner
service shutdown, VM disappearance, or whole-job timeout. Only an upload that
already completed—or external storage already written—survives those classes.

## Storage and checkpoint option review

| Option | Assessment |
|---|---|
| Periodic local checkpoint only | useful for ordinary process failure; insufficient for runner/VM loss |
| Separate upload after each completed deterministic chunk | selected P0; GitHub-native, immutable, bounded retention, low authority risk with distinct schema/root |
| One workflow job per chunk | deferred; stronger runner isolation but materially higher orchestration, session, rate-limit, and merge complexity |
| GitHub cache | rejected; cache is mutable/evictable build acceleration, not retained audit evidence |
| Distinct artifacts from multiple completed jobs | P2 fallback if same-job chunk persistence remains insufficient |
| External durable storage | rejected for current evidence; adds credentials, security, retention, cost, and authority surface |

## Dependency reproducibility decision

Dependency drift is not proven causal, but inability to recover the second
canary's versions is an observability defect.

```text
DEPENDENCY_HARDENING: REQUIRED_BEFORE_NEXT_CANARY
```

P0 is deliberately limited to capturing Python, OS, pandas, numpy, yfinance,
curl_cffi, requests, lxml, and a full sorted installed-package inventory before
any provider call, binding it to run ID and source SHA, and immediately
uploading that diagnostic preflight artifact. Version capture must also appear
in the console log. Environment variables and secrets must not be captured.

A constraints or lock file is P1, not root-cause remediation. It requires a
separate compatibility test over Python 3.11 and the ME-SR26 deterministic
suites before adoption. Casual pinning in this review would be speculative.

## Resource telemetry decision

```text
RESOURCE_TELEMETRY_REQUIRED: YES
```

The next implementation must use standard Linux/stdlib facilities rather than
adding psutil solely for telemetry:

- `/proc/self/status` for process RSS;
- `/proc/meminfo` for system `MemAvailable`;
- `os.getloadavg()` for one-minute load;
- `shutil.disk_usage()` for free bytes;
- `time.monotonic()` for elapsed and last-progress duration; and
- UTC only for operational event timestamps.

Metrics are emitted at preflight, phase boundaries, chunk boundaries, and the
60-second heartbeat. If an optional metric is unavailable, telemetry records
`null` plus a reason; it does not change market evidence or analytic results.
Secrets, cookie/crumb material, portfolio data, and raw HTTP content are
forbidden.

## Minimum recommended remediation

### P0_REQUIRED_BEFORE_NEXT_CANARY

1. Split the 952 canonical ordering into fixed sequential chunks of at most 64
   identities within the existing job; do not introduce parallel jobs.
2. Preserve one global maximum-25 singleton fallback budget and original
   deterministic fallback selection order across all chunks.
3. Add an explicit bounded wall-clock cap per acquisition chunk and fail closed
   without retry when exceeded; retain the 90-minute job cap.
4. Emit structured phase/chunk events and a 60-second heartbeat with progress,
   elapsed time, RSS, available memory, load, and disk free.
5. Capture and immediately upload a diagnostic runtime/preflight artifact
   before provider acquisition.
6. Atomically write a distinct `complete: false` diagnostic checkpoint after
   every completed chunk and immediately upload it under a unique immutable
   14-day Actions artifact name.
7. Assemble the existing final history artifact only after exact 952 identity,
   source, universe, policy, chunk, fallback-budget, and checksum reconciliation;
   keep the existing quality gate and screening unchanged.
8. Add fail-closed tests proving partial diagnostics cannot be loaded as
   history, cannot satisfy screening, and cannot reach RUN33.

### P1_RECOMMENDED

- add a reviewed constraints file after compatibility validation;
- retain sanitized provider error-class and chunk latency summaries;
- record final file counts/bytes and compression duration;
- document an operator interpretation guide for heartbeat/checkpoint states.

### P2_OPTIONAL

- move chunks to distinct sequential matrix jobs only if same-job checkpoints
  still fail to retain enough evidence;
- retain raw non-authoritative chunk data for offline diagnosis if metadata-only
  checkpoints prove insufficient;
- add external durable telemetry only if GitHub-native artifacts fail again.

### REJECTED

- a third canary with no code change;
- heartbeat-only hardening;
- parallel provider acquisition without a separate rate-limit/semantic review;
- per-chunk fallback budgets;
- resumable acquisition;
- provider replacement or broad adapter redesign;
- GitHub cache as audit evidence;
- raw HTTP/cookie/crumb logging;
- external storage in the minimum design;
- any relaxed universe, freshness, lag, history, checksum, or authority gate.

This is minimum hardening rather than major workflow redesign: one runner and
one sequential provider path remain; provider selection and final artifacts do
not change; no resume, external service, matrix fan-out, or new analytic
semantics is introduced.

## Authority invariants

ME-SR28 must preserve without exception:

- repository-owned canonical 952 universe;
- repository-owned history and screening policies;
- repository-owned provider selection, UTC authority, and Git HEAD provenance;
- exact source SHA, universe, and policy binding;
- unadjusted OHLC and excluded Adj Close;
- nullable volume with no zero imputation;
- minimum 252 and maximum 420 sessions;
- one global maximum of 25 singleton fallbacks;
- 0.80 widespread one-session-lag threshold;
- 0.99 effective fresh-coverage threshold;
- exact identity reconciliation and deterministic ordering;
- semantic replay and checksum integrity;
- fail-closed global provider failure and lag behavior;
- no analytic authority for partial evidence;
- no canonical publication or `market-data` mutation;
- no DATA11 approval, DATA07, DATA06, RUN31, RUN33, Decision Engine, portfolio,
  broker, order, or trade authority.

## Required implementation tests

ME-SR28 must add or preserve deterministic tests for:

1. fixed chunk planning covers all 952 identities exactly once in canonical
   instrument order with no duplicate, omission, or caller-selected authority;
2. chunk size is bounded and chunk IDs/digests are deterministic;
3. provider mocks receive the expected chunk symbols without live network;
4. primary chunks do not consume fallback budget and the whole run performs no
   more than 25 deterministic singleton fallbacks;
5. heartbeat cadence and phase transitions under a fake monotonic clock;
6. resource telemetry values and graceful `null` handling without altering
   provider/history outputs;
7. dependency capture occurs before the provider seam is invoked;
8. checkpoint schema, exact fields, `complete: false`, atomic rename, source,
   universe, policy, phase, count, and chunk binding;
9. partial checkpoint roots are rejected by the public history loader, current
   screening loader, and RUN33 handoff path;
10. missing, duplicate, extra, corrupt, cross-run, cross-SHA, cross-universe,
    cross-policy, or incomplete chunks block final assembly;
11. process failure or synthetic chunk timeout leaves no final history artifact;
12. complete chunk assembly is semantically equivalent to the existing
    monolithic mocked provider result, including status counts, lag detection,
    coverage, observations digest, nullable volume, and checksums;
13. final history loader replay, current technical screening replay, and
    history/screening cross-binding remain unchanged;
14. workflow structure retains `contents: read`, `cancel-in-progress: false`,
    the 90-minute job cap, 14-day retention, unique preflight/chunk artifacts,
    sequential execution, and no publication/downstream steps; and
15. ordinary provider/Python/quality-gate failures preserve already completed
    diagnostic uploads without allowing them to satisfy analytic gates.

Required existing suites include:

```text
tests/market_engine/source_refresh/test_advisory_ohlc_history.py
tests/market_engine/data/test_incremental_market_data_refresh.py
tests/market_engine/run/test_current_technical_screening_handoff.py
tests/market_engine/data/test_scheduled_canonical_price_refresh_workflow.py
```

The implementation review must also run `python -m compileall -q
src/market_engine`, `git diff --check`, and the repository governance scans.

## Explicit third-canary gate

A third canary is not ready and is not authorized by ME-SR27.

It may be proposed only after ME-SR28 is implemented, deterministically tested,
reviewed, merged, and independently verified to preserve every authority
invariant above. That later authorization must still permit exactly one new
controlled run and no retry. A successful canary remains required before the
human approval/DATA07/DATA06/RUN31 checkpoint and conditional RUN33.

## Review validation

The documentation-only review completed the required offline checks:

- the four targeted existing test modules passed: `82 passed`;
- `python -m compileall -q src/market_engine` passed;
- `git diff --check` passed;
- the mandatory `BUY` and `SELL` scans returned only the existing explicit
  transaction parsing and portfolio transaction-recording paths; and
- the mandatory `tradeable` scan returned no match outside the Decision
  Engine.

No validation command contacted a live market-data provider.

## Safety record

ME-SR27 performed zero workflow dispatches, zero reruns, zero live provider
calls, zero 952-history retrievals, zero runtime changes, zero workflow
changes, zero publication, zero `market-data` mutation, zero downstream
production runs, zero portfolio access, zero broker/order actions, and zero
automatic merges. Static installed-package and official upstream source
inspection plus deterministic local commands were read-only.

## Exact next sprint

`ME-SR28 — Bounded Advisory History Acquisition Observability and Diagnostic
Retention`: implement and test the P0 sequential chunk, heartbeat, resource and
dependency telemetry, diagnostic checkpoint, immediate per-chunk upload, and
fail-closed final assembly contract. Do not execute a third canary in ME-SR28.
