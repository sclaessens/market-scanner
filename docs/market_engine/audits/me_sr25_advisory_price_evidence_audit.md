# ME-SR25 Advisory Price Evidence Audit

## Scope and provenance

- Base branch: `origin/main`
- Base SHA: `4753c94c0ab572d619fbb7b82496ba2864797a9f`
- Implementation branch: `codex/me-sr25-advisory-price-evidence`
- Preserved `market-data` SHA: `95c88276763b1762cbbfbccc402ec8535268127b`
- Authoritative universe: 952 instruments from the existing ME-DATA04
  coverage-after snapshot
- Provider path: existing yfinance daily-history adapters only

The implementation worktree was created separately because the original main
worktree contained pre-existing untracked artifacts. Those files were not
removed, overwritten, staged, or read as sprint inputs.

## Acceptance evidence

ME-SR25 adds a separate observations document, checksum-bound manifest,
configurable freshness policy, JSON Schemas, fail-closed loader, minimal
consumer, and read-only scheduled artifact workflow. Synthetic tests cover the
complete 952-instrument universe while provider behavior is injected; no real
provider acquisition is performed.

The validation boundary covers exact decimal prices, currencies, canonical
identity, source and observation semantics, timezone-aware temporal ordering,
trusted-future rejection, full-universe reconciliation, deterministic output,
partial failure preservation, bounded fallback, manifest and observation
integrity, recomputed freshness, schema versions, JSON Schema enforcement, and
forged caller-context rejection.

The consumer exposes current price context only for validated effectively
fresh evidence. Stale, missing, and invalid states remain explicit and cannot
masquerade as a current price.

## P1 review remediation

The two P1 review blockers were remediated on the same branch and draft pull
request:

1. one canonical clock helper now generates trailing-`Z` UTC defaults for the
   builder, loader, consumer, CLI, and exact workflow command path. Default
   execution no longer produces the rejected `+00:00` representation;
2. artifact freshness remains immutable acquisition-time evidence, while the
   loader creates a separate effective freshness view at every trusted load.
   Consumer contract v2 uses that effective status to gate `current_price` and
   retains the original status and session age for audit.

Regression coverage exercises default API and CLI clocks, the exact workflow
build arguments without acquisition or network access, canonical timestamp
roundtrip, fresh-to-stale transition after a completed session, weekend
continuity, missing and invalid preservation, unavailable session resolution,
trusted time before generation, immutable files and manifest totals, and the
existing rehashed-forgery rejection. Artifact retention is explicitly not a
freshness decision.

Review-remediation validation produced these results:

- timestamp and consumption-freshness regression selection: 10 passed;
- complete ME-SR25 suite: 50 passed;
- relevant data, ticker-universe, and source-refresh suite: one baseline-only
  failure and 706 passed;
- recommendation, handoff, advisory, and advice suite: 228 passed;
- complete Market Engine suite: one baseline-only failure and 1,676 passed;
- complete repository suite: one baseline-only failure and 2,343 passed;
- changed-file Python compilation: passed;
- policy and JSON Schema parsing: passed;
- representative valid/invalid Draft 2020-12 payload validation: passed;
- `git diff --check`: passed;
- governance and changed-boundary side-effect searches: no new violation.

The sole broad failure was reproduced again at exact base SHA
`4753c94c0ab572d619fbb7b82496ba2864797a9f`: the same
`test_compact_checksums_match_committed_files_and_local_full_runs` test fails
at the same `path.is_file()` assertion for the absent historical DATA06
manifest. No other broad failure occurred.

## Regression evidence

Final validation produced these results:

- focused ME-SR25 suite: 42 passed;
- relevant data, ticker-universe, and source-refresh suite: one baseline-only
  failure and 698 passed;
- recommendation, handoff, advisory, and advice suite: 228 passed;
- complete Market Engine suite: one baseline-only failure and 1,668 passed;
- complete repository suite: one baseline-only failure and 2,335 passed;
- changed-file Python compilation: passed;
- all changed JSON policy and schema documents parsed successfully;
- representative valid and invalid payloads were exercised against both Draft
  2020-12 JSON Schemas by the focused suite;
- `git diff --check`: passed;
- changed-boundary searches found no market-data, portfolio, artifact, broker,
  notification, publication, or credential side effect.

The sole broad failure is
`test_compact_checksums_match_committed_files_and_local_full_runs`. It asserts
that
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`
exists. The identical targeted test fails with the identical `path.is_file()`
assertion in a clean detached worktree at base SHA
`4753c94c0ab572d619fbb7b82496ba2864797a9f`. ME-SR25 does not create, consume,
or change that historical fixture.

The mandatory repository governance searches still report the pre-existing
legacy BUY/SELL command handling under `scripts/portfolio/`; the ME-SR25 diff
does not touch those files. No `tradeable` occurrence was found outside the
Decision Engine. A scoped ME-SR25 runtime/workflow search produced only the
ordinary term `order` in deterministic record ordering, not trade execution.

## Authority and side-effect review

- `market-data` is unchanged and no canonical publication path is invoked.
- No real portfolio data is read, created, migrated, or changed.
- No Decision Engine, allocation, sizing, recommendation, or ranking logic is
  introduced.
- No broker, order, notification, publication, or credential path is present.
- No workflow run, canary, provider acquisition, merge, or publication was
  executed during this sprint.

## Product route

The active route is now ME-SR25, ME-DATA11, ME-RUN33, and ME-CI12. ME-RUN33 is
the first unreserved run identifier found in authoritative documentation and is
reserved for an accelerated 5-15 candidate release, or explicit no-candidate
evidence. ME-PI01 and later portfolio, sizing, notification, broker, cloud, and
canonical-publication work remain deferred until after that release.
