# ME-DATA09 Source-Approved Bounded Operator Fundamental Metric Pilot

## Review-fix outcome

ME-DATA09 completed its corrected immutable run on 2026-07-19. The approval
runtime is generic: ticker identity is data in `approved_tickers`, the generic
scope is `bounded_operator_fundamental_metric_pilot`, and the package may contain
at most one unique ticker. No runtime branch names AAPL, Apple, or a market.

Source documents are bound through relative logical paths and an
operator-supplied `--source-document-root`. The root itself is used only at
runtime and is not published. Absolute paths, empty paths, traversal, symlink
escape, missing files, and checksum mismatches fail before package parsing.

## Source evidence and approval

- Approved ticker set: `AAPL`.
- Reporting period: FY2026 Q2, 2025-12-28 through 2026-03-28.
- Apple earnings release URL:
  `https://www.apple.com/newsroom/2026/04/apple-reports-second-quarter-results/`.
- Apple document SHA-256:
  `b421dd69c74aab85ac6d003884ec0017cfa2ca4bc4c4a89576e7eb6e0baba954`.
- SEC Form 10-Q URL:
  `https://www.sec.gov/Archives/edgar/data/320193/000032019326000013/aapl-20260328.htm`.
- SEC document SHA-256:
  `a61f508a797f02384801dadc55b18feef248e28c83ec07281e960dd7d0f4620d`.
- Permitted local use: `approved_for_bounded_me_data09_aapl_pilot_only`.
- Full source documents remain local and uncommitted.

Direct metrics are `revenue_growth_yoy` at 17.0 percent (normalized to 0.17)
and `eps_growth_yoy` at 22.0 percent (normalized to 0.22). `gross_margin`,
`operating_margin`, and `debt_to_equity` remain explicitly missing. No metric
was derived, estimated, interpolated, rounded, or represented as zero.

## Immutable artifact binding

- Package ID: `me-data09-aapl-fy2026-q2-20260719T155116Z`.
- DATA08 input SHA-256:
  `b6353a66cb537a31720117bd780eaaaf51f0174590d48b8033ef330dd7a5fe39`.
- Accepted package SHA-256:
  `9507a07605168841163460ef4e7417e7c51a4d881bbea9e15585b02aeff8f41e`.
- DATA08 validation report SHA-256:
  `366eab1a2311094eecf48e06a29512396b63800c5d1f664203a65a64340b1b64`.
- DATA09 decision SHA-256:
  `5fd17e54c948a54a448abb3849b51ce29c87c65c458d9b28de3fa73daf10b684`.
- DATA08 result: `accepted`, zero errors, structurally valid for explicit
  source-approval review.
- DATA09 approval validation: `approved`, with no issues or reason codes.

## Corrected full run

- DATA07: `me-data09-aapl-bounded-operator-pilot-20260719T155116Z`.
- DATA06: `me-data06-after-me-data09-aapl-20260719T155116Z`.
- RUN31: `me-run31-after-me-data09-aapl-20260719T155116Z`.
- Selected 12; imported 1; normalized 1; success 1; blocked 11; failed 0;
  pending 0; not selected 940.
- Reconciliation: passed for all 952 canonical instruments.
- Raw snapshot: validated, one record, checksum equal to the accepted package
  checksum `9507a07605168841163460ef4e7417e7c51a4d881bbea9e15585b02aeff8f41e`.
- DATA07 provider calls: 0. DATA07 network calls: 0.

## Coverage attribution

The current ME-DATA09 sprint comparison is measured from the validated DATA06
baseline and the corrected downstream result:

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| Fundamental complete | 6 | 6 | 0 |
| Fundamental partial | 39 | 39 | 0 |
| Fundamental missing | 907 | 907 | 0 |
| Invalid/stale/conflicting | 0 | 0 | 0 |
| Canonical advice-input-ready | 6 | 6 | 0 |
| Full-advice-ready | 0 | 0 | 0 |
| Unable-to-advise | 946 | 946 | 0 |

AAPL remained partial and not advice-input-ready. Its selected evidence now
contains the two directly reported growth metrics, while three metrics remain
missing. This is a traceable evidence change without an aggregate status or
readiness transition.

DATA06 also retains an internal comparison against its older historical origin
baseline. That comparison contains 24 earlier improvement tickers and the
historical 4/17/931 state. It is emitted only as
`historical_origin_comparison` with `attributable_to_current_sprint: false` and
is not a current ME-DATA09 improvement claim. The compact current-sprint delta
does not copy those historical tickers.

## Compact publication boundary

The complete DATA07, DATA06, RUN31, and raw snapshot outputs remain locally
available at their repository-relative run paths but are removed from Git
tracking. The old 20260719T150409Z full run and operator input are also retained
locally and ignored through exact path rules.

Committed evidence is limited to:

- `operator_input/market_engine/me-data09/aapl/20260719T155116Z/`: input,
  accepted package, DATA08 report, and portable approval decision.
- `artifacts/market_engine/run_evidence/me-data09-aapl-bounded-operator-pilot-20260719T155116Z/`:
  compact manifest, source checksums, approval validation, DATA08 report,
  DATA07 summary, approved-ticker evidence, current coverage delta, downstream
  identities, checksums, and report.

The compact manifest records `full_run_committed: false` and
`compact_evidence_committed: true`. It contains no full 952-ticker list, source
document, absolute workstation path, credential, cookie, or secret.

## Next sprint definition

ME-DATA10 is **Implement a generic governed primary-source fundamental metric
derivation engine and execute a bounded pilot**. It will define one bounded
ticker pilot with a versioned formula contract, numerator and denominator
lineage, identical periods and units, deterministic calculations, denominator
safety checks, freshness and checksum binding, and visible
direct-versus-derived classification for gross margin, operating margin, and/or
debt-to-equity.

ME-DATA10 is not implemented in this PR. It does not authorize a broad batch,
estimates, interpolation, hidden formulas, recommendations, ranking, allocation,
portfolio or delivery writes, broker actions, or Decision Engine authority.
