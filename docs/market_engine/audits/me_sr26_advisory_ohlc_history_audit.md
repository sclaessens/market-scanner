# ME-SR26 Advisory OHLC History Audit

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
one insufficient, one missing, and one invalid history. The handoff test proves
zero DATA07, DATA06, RUN31, and RUN33 calls while DATA11 approval is pending.

Validation produced 41 passing focused ME-SR26 tests and 2,551 passing tests
in the complete repository suite. The sole repository-suite failure is
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
