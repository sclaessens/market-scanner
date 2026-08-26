# ME-SR26 — Advisory OHLC History and RUN33 Handoff

## Authority closure status

The implementation now enforces the intended boundary: public callers select evidence artifact paths, while the repository selects the canonical 952-identity universe, history policy, unchanged RUN30 screening policy, and SR25 price policy. Production also owns provider selection, UTC time, and source provenance derived from repository `HEAD`. Custom authority inputs are confined to private deterministic test helpers. Merge readiness remains subject to the complete validation and review record.

Sprint ID: ME-SR26
Status: IMPLEMENTED — OPERATIONAL VALIDATION BLOCKED; ME-SR28 NEXT

The single authorized post-merge canary on 2026-08-26 was interrupted because
the GitHub-hosted runner received a shutdown signal while bounded history
acquisition was still executing and before a manifest was emitted. No
application, provider, policy, universe, or workflow-timeout failure was
evidenced, and the external termination cause remains unresolved.
No artifact, quality-gate result, or screening evidence was produced, and no
retry occurred. Operational validation remains open. RUN33 remains a separate
conditional future controlled step and was not executed.

The read-only runner diagnostic found no competing run, repository cancellation
path, workflow timeout, ordinary resource-exhaustion evidence, application
exception, or provider failure. Deterministic history/provider tests remain
healthy. It classified the event as
`TRANSIENT_GITHUB_RUNNER_INTERRUPTION_LIKELY` and found a second controlled
canary `RECANARY_SAFE_WITHOUT_CODE_CHANGES`. This does not itself dispatch or
authorize the run. A separately authorized second controlled canary was
dispatched as run `32966745030`. After approximately 50 minutes in bounded
history acquisition, GitHub reported that the hosted runner lost communication
with the server. No history manifest, quality-gate result, screening, log
archive, or uploaded artifact survived. The exact first-canary shutdown-signal
text did not recur, but this second runner-level failure keeps operational
validation blocked and requires runner/workflow hardening review before any
later canary proposal.

ME-SR27 completed that review. It proved the current primary `yf.download`
invocation iterates 952 unique provider symbols synchronously under
`threads=False`; the 15-second value is request-scoped and does not bound the
whole phase. Heartbeat alone cannot survive hosted-runner disappearance.
ME-SR28 must implement statically sequenced deterministic chunk execution,
official upload, persistence-receipt, and gate steps; a parent-enforced
repository-configured worker-process deadline; resource and dependency
telemetry; diagnostic-only partial checkpoints; and exact fail-closed final
assembly before a third canary can be proposed. Local chunk completion is not
durable evidence until its upload action succeeds.

## Outcome

ME-SR26 adds a non-canonical, advisory-only route for bounded full-universe
daily OHLC history and reruns the unchanged RUN30 technical setup and ranking
semantics only over validated current history. It emits a checksum-bound,
conditional RUN33 input handoff; it does not execute RUN33.

## Acceptance contract

- Exactly 952 canonical identities must reconcile.
- Each series is one currency, source mapping, unadjusted OHLC basis, and
  adjustment policy, with exact positive decimal strings and explicit volume.
- Minimum history is 252 sessions: 200 indicator warm-up sessions plus a
  52-session safety margin. At most 420 sessions are retained.
- Missing, invalid, insufficient, stale, identity-blocked, and
  adjustment-policy-blocked histories remain distinct.
- At least 80% of otherwise-valid series exactly one session late blocks the
  run as widespread provider-session lag. Individual fallbacks are capped at
  25; no retry wave exists.
- A runtime semantic replay gate requires at least 0.99 fresh coverage. It
  permits isolated blockers while keeping them ineligible and blocks global
  provider lag or failure.
- Production time is internal UTC authority; producer status is immutable and
  effective freshness is recalculated at load.
- The production history provider is internally selected. Public history,
  screening, and RUN33 APIs accept artifact paths but no provider or
  freshness-time override; deterministic injection is private and test-only.
- Missing volume remains explicitly missing rather than becoming zero.
- SR25 price and the latest history close must match identity, session,
  currency, source semantics, and exact decimal value.
- DATA11 approval remains pending for ASH, BIO, and CI, so the RUN33 handoff is
  conditional and all candidates remain ineligible.
- A synthetic test proves the separate positive route through canonical
  approval validation, DATA07/DATA06/RUN31 receipts, execution proof,
  downstream after-authority, and the private RUN33 handoff loader.

## Required next action

Implement and review ME-SR28's bounded acquisition observability and diagnostic
retention contract. Do not authorize or dispatch a third canary during that
implementation. Do not publish, advance `market-data`, or execute RUN33.
