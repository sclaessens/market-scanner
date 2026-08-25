# ME-SR26 Advisory OHLC History Audit

## Final authority-boundary remediation

The public history, screening, and RUN33 APIs no longer expose canonical universe or policy paths. RUN33 internally binds SR25 validation to `DEFAULT_PRICE_POLICY_PATH`. The public history builder no longer accepts `source_main_sha`; production provenance is the fail-closed Git `HEAD` of the executing repository. Provider and UTC freshness authority remain internal. Alternate universe, policy, SHA, provider, and time inputs remain available only through private deterministic test seams.

## Evidence reviewed

- Base: `51025546bfeadd52c095382525bbbac19e98e415`.
- Canonical `market-data` before implementation:
  `95c88276763b1762cbbfbccc402ec8535268127b`.
- RUN30 cutoff: 2026-07-10.
- RUN31 prestate: 6 complete, 39 partial, 907 missing; 6 advice-input-ready;
  zero full-advice-ready.
- DATA11 candidates ASH, BIO, and CI: pending.
- SR25 runs `31777995934` through `32000009638`: 946 fresh, one stale, five
  invalid. Run `32104872490`: three fresh, 944 stale, five invalid.

## Controls

The implementation keeps `market-data` read-only, preserves the approved
technical formulas and tie-breaks, rejects non-exact financial values, isolates
per-instrument failures, detects a widespread one-session lag, and binds every
authority-bearing stage. Fixtures reconcile 952 instruments with 949 fresh,
one insufficient, one missing, and one invalid history as analytically usable
under the 0.99 policy. Rebound tests replace bars and re-sign the index,
eligibility, manifest, observation digest, artifact digest, and checksum index;
semantic replay still rejects the false freshness claims. Effective load-time
freshness is separately recalculated under the internal UTC clock.

API-surface tests verify that public history build cannot accept a provider or
time override and that public history load, screening, and RUN33 build/load
cannot accept any clock, `trusted_now`, `now`, `as_of`, evaluation-time, or
acquisition-time alias. Deterministic provider/time injection is confined to
private helpers. A public stale-history test and a public stale RUN33
history/price test prove that backdating is unreachable.

The pending handoff test proves zero DATA07, DATA06, RUN31, and RUN33 calls.
A separate full synthetic positive test uses canonical approval validation,
`execute_approved_candidate`, validated DATA07/DATA06/RUN31 results and
receipts, the private execution proof, `load_downstream_after_authority`, all
952 identities, and the private RUN33 handoff loader. It reaches
`ready_for_run33` with eligible records. Forged mappings and fully rebound
handoff JSON remain rejected. Missing volume stays nullable.

Validation produced 174 passing focused history, screening/handoff, and DATA11
authority tests. The complete repository suite produced 2,568 passes. The sole
repository-suite failure is
`test_compact_checksums_match_committed_files_and_local_full_runs`, which
expects the absent historical file
`me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`. The identical
test, assertion, missing path, and cause reproduce at base
`51025546bfeadd52c095382525bbbac19e98e415`; it is not an ME-SR26 regression.
Python compilation, strict JSON parsing, Draft 2020-12 schema validation,
checksum replay, and `git diff --check` pass.

## Deferred operational evidence

One post-merge 09:30 UTC canary remains necessary. It must be reviewed as
advisory evidence only. No workflow dispatch, provider canary, canonical
publication, merge, or RUN33 execution is part of this sprint.
