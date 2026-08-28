# ME-SR28 Bounded Advisory History Observability Implementation Handoff

Story ID: ME-SR28

Status: IMPLEMENTATION COMPLETE — REVIEW PENDING

ME-SR28 implements the P0 contract selected by ME-SR27. It does not execute or
authorize a third ME-SR26 canary.

## Implemented runtime boundary

Production advisory history acquisition now uses repository-owned operational
configuration at `config/market_engine/advisory_ohlc_history_runtime.json`:

- 64 identities per primary chunk;
- exactly 15 canonical primary chunks for 952 identities;
- 600 seconds per provider stage;
- five seconds of termination grace; and
- a 60-second heartbeat interval.

Callers cannot override these values, chunk membership, provider authority,
source SHA, universe, history policy, or production time authority. The retired
public monolithic build path fails closed. The private deterministic builder
remains only as an analytic-equivalence test seam and final assembly primitive.

Every provider stage runs in a new-session child process. The parent owns the
monotonic deadline, sends `SIGTERM` to the process group, waits no more than the
configured grace, escalates to `SIGKILL`, reaps the worker, removes only its
temporary staging, records `provider_chunk_timeout`, and permits no retry,
fallback, later primary chunk, final history, or screening after timeout.

The production workflow is manual-validation-only while ME-SR26 operational
validation remains incomplete. Its sole trigger is `workflow_dispatch`; it has
no schedule or other automatic production trigger. Scheduled acquisition may
only return through a separately reviewed governance decision after successful
operational validation.

## Durable diagnostic boundary

The single workflow job statically declares preflight and 15 primary
execute/upload/receipt/gate groups, followed by one required global fallback
group. A stage is durable only after `actions/upload-artifact@v4` returns a
non-empty artifact ID and its documented raw lowercase 64-character SHA-256
hexadecimal `artifact-digest` output, without a `sha256:` prefix. The subsequent
receipt binds those outputs to the run, source, universe, policies, runtime
config, stage, checkpoint digest, and unique artifact name. Local completion
and atomic rename alone are not durability claims.

## Corrected operational execution record

Exactly two controlled canaries occurred. Three production workflow executions
reached history acquisition during this validation period because the
then-active schedule also started one uncontrolled production run. That
scheduled run is not, and must not be retroactively classified as, a controlled
canary.

| Run | Event | Controlled canary | Failure mode |
|---|---|---:|---|
| `32951786805` | `workflow_dispatch` | yes | runner shutdown signal |
| `32956572481` | `schedule` | no | runner shutdown signal |
| `32966745030` | `workflow_dispatch` | yes | hosted-runner communication loss |

Scheduled run `32956572481` executed SHA
`19f207b35947040db5ee466c54a140160909c7ce`, entered live history acquisition,
and later received `The runner has received a shutdown signal.` This
uncontrolled execution strengthens the hardening rationale but does not change
the ME-SR28 P0 architecture or satisfy any canary gate.

`continue-on-error` is limited to preserving the immediately following
diagnostic upload. Each gate validates the true GitHub step `outcome`, the
repository checkpoint execution status, and the persistence receipt. It never
uses a rewritten step conclusion as proof of successful execution.

## Authority boundary

Diagnostic checkpoints use
`market-engine-advisory-ohlc-history-diagnostic-checkpoint-v1`, always contain
`complete: false`, and remain under a root distinct from final history. Their
authority is `diagnostic_only_partial_never_analytic`. Persistence receipts are
diagnostic persistence evidence only. History, quality-gate, current technical
screening, and RUN33 paths reject diagnostic roots.

Final history assembly requires the exact expected stage set, successful stage
statuses and gates, every persistence receipt, exact run/source/universe/policy/
runtime bindings, exact canonical chunk membership, exact 952 primary identity
reconciliation, and at most 25 globally selected singleton fallbacks. The
fallback stage exists even when its selected count is zero. Any mismatch leaves
no final manifest.

Before provider-stage reconciliation, assembly revalidates the full preflight
authority envelope through the same canonical stage helpers used for provider
stages. This requires the exact checkpoint schema and authority bindings, a
successful execution status, the exact passed gate, the receipt schema and
diagnostic-only authority, its checkpoint SHA and artifact-name bindings, a
non-empty canonical artifact ID, and the raw 64-hex upload-artifact digest.
Missing or corrupt preflight evidence blocks assembly before any final manifest
is created.

The operational runtime config is not analytic history policy and is not added
to the final analytic manifest. Chunked and monolithic mocked 952-identity
fixtures produce byte-identical final analytic artifacts, including observation
and checksum semantics, plus byte-identical current technical screening output.

## Offline validation record

- targeted ME-SR26/ME-SR28/provider-seam/screening/workflow suites: 138 passed;
- new ME-SR28 runtime suite: 28 passed;
- complete `source_refresh` suite: 182 passed;
- complete `run` suite: 216 passed;
- complete Market Engine suite: 1,932 passed with one documented baseline
  failure caused by the absent historical local DATA06 compact-checksum
  manifest;
- complete repository suite: 2,599 passed with the same single documented
  baseline failure;
- Python compilation, changed JSON/schema validation, workflow YAML parsing,
  diff validation, and repository governance scans passed.

No validation used a live provider or dispatched a workflow.

## Review gate

Required next step: review and merge ME-SR28. A separate later authorization
must decide whether exactly one third controlled ME-SR26 canary may run. No
canary, publication, `market-data` mutation, DATA11 approval, DATA07, DATA06,
RUN31, RUN33, Decision Engine production path, portfolio path, or broker/order
path is part of this implementation sprint.
