# ME-DATA09 Source-Approved Bounded Operator Fundamental Metric Pilot

## Outcome

ME-DATA09 completed on 2026-07-19. The earlier preflight correctly stopped
because no concrete operator evidence existed. The resumed sprint used the
operator's explicit bounded authorization, an official Apple earnings release,
and the associated SEC filing. ME-DATA08 accepted the compact input, the
checksum-bound DATA09 approval passed, ME-DATA07 imported and normalized AAPL,
and DATA06/RUN31 measured the downstream result.

No provider or network call occurred during DATA07 processing. No downloaded
source document is stored in the repository.

## Evidence and authorization

- Scope: `bounded_aapl_operator_fundamental_metric_pilot`.
- Permitted local use: `approved_for_bounded_me_data09_aapl_pilot_only`.
- Publication boundary: official URLs, document identities, checksums, compact
  extracted facts, and audit metadata may be committed; complete downloaded
  documents remain local and uncommitted.
- Apple source: *Apple reports second quarter results*, published 2026-04-30,
  `https://www.apple.com/newsroom/2026/04/apple-reports-second-quarter-results/`.
- Apple document SHA-256:
  `b421dd69c74aab85ac6d003884ec0017cfa2ca4bc4c4a89576e7eb6e0baba954`.
- SEC cross-check: Apple Inc. Form 10-Q, CIK `0000320193`, accession
  `0000320193-26-000013`, filed 2026-05-01, report period 2026-03-28,
  `https://www.sec.gov/Archives/edgar/data/320193/000032019326000013/aapl-20260328.htm`.
- SEC document SHA-256:
  `a61f508a797f02384801dadc55b18feef248e28c83ec07281e960dd7d0f4620d`.

The Apple source was approved as the direct primary metric source. The SEC
filing independently cross-checked company identity, fiscal context, and the
quarter from 2025-12-28 through 2026-03-28. The canonical mapping is AAPL,
`equity:aapl`, Apple Inc. All review dimensions passed: authenticity, primary
source, identity, mapping, lineage, reporting period, fiscal context, unit and
scale, freshness, permitted local use, and publication boundary.

## Direct metrics

The package contains only directly published percentages for FY2026 Q2:

| Metric | Raw value | DATA07 normalized value |
|---|---:|---:|
| `revenue_growth_yoy` | 17.0 percent | 0.17 ratio |
| `eps_growth_yoy` | 22.0 percent | 0.22 ratio |

`gross_margin`, `operating_margin`, and `debt_to_equity` are explicitly
missing. No value was derived, estimated, rounded, or represented as zero.

## DATA08 and DATA09 binding

- Package ID: `me-data09-aapl-fy2026-q2-20260719T150409Z`.
- DATA08 input SHA-256:
  `fc5b27ad944ebb5f26c7a7270d3d9b1a420e3c6c5da5933300d176c8e25d1f8d`.
- Accepted package SHA-256:
  `264c574fa0c366262637ecefefeed5f08e40a6aa6991f4dc41707debce90c5ca`.
- DATA08 report SHA-256:
  `bc0ffce9fda4d2b1da101ecfc1ef23b0bcfb864ab6662f6ed144f2205611c5f8`.
- DATA08 result: `accepted`, zero errors,
  `structurally_valid_for_explicit_source_approval_review`.
- DATA09 decision: `approved` by the recorded Operator, Data Steward, and
  Governance Auditor roles.
- DATA09 validation: `approved`, with no reason codes or issues.

The runtime treats `approved_operator_supplied_import` only as route
eligibility. The explicit DATA09 artifact is required before parsing; missing,
malformed, blocked, rejected, unknown, mismatched, or incomplete approval stops
before parser, snapshot, and downstream work.

## DATA07 pilot

- Run ID: `me-data09-aapl-bounded-operator-pilot-20260719T150409Z`.
- Deterministically selected: AAPL, ALL, AMD, A, AAL, ABBV, ABNB, ASML, BF.B,
  ADC, ADM, ADP.
- Selected 12; imported 1; normalized 1; success 1; blocked 11; failed 0;
  pending 0; not selected 940.
- AAPL result: valid partial evidence with two metrics and three explicit
  missing metrics.
- Reconciliation: passed for all 952 canonical instruments.
- Raw snapshot: validated, one AAPL record, checksum
  `264c574fa0c366262637ecefefeed5f08e40a6aa6991f4dc41707debce90c5ca`.
- Provider calls: 0. Network calls: 0. Requests attempted: 0.

## Downstream measurement

- DATA06: `me-data06-after-me-data09-aapl-20260719T150409Z`, completed.
- RUN31: `me-run31-after-me-data09-aapl-20260719T150409Z`, completed.

| Measure | Validated pre-pilot baseline | Post-pilot |
|---|---:|---:|
| Fundamental complete | 6 | 6 |
| Fundamental partial | 39 | 39 |
| Fundamental missing | 907 | 907 |
| Invalid/stale/conflicting | 0 | 0 |
| Canonical advice-input-ready | 6 | 6 |
| Full-advice-ready | 0 | 0 |
| Unable-to-advise | 946 | 946 |

AAPL remained partial and not canonical advice-input-ready, but its selected
evidence changed to the newer primary-source package with the two validated
growth metrics. There was no AAPL readiness transition, no aggregate coverage
change relative to the validated DATA06 baseline, and no regression. The three
missing AAPL metrics remain the blocker. Full-advice readiness, recommendation,
ranking, tradeability, allocation, and execution authority are unchanged.

## Artifact map

- Input, package, decision:
  `operator_input/market_engine/me-data09/aapl/20260719T150409Z/`.
- DATA08 report:
  `artifacts/market_engine/operator_fundamental_metric_packages/me-data09-aapl-20260719T150409Z/`.
- DATA07 manifest, approval validation, summaries, normalized AAPL evidence,
  per-ticker status, blockers, and coverage:
  `artifacts/market_engine/fundamental_metric_sourcing_runs/me-data09-aapl-bounded-operator-pilot-20260719T150409Z/`.
- Raw snapshot:
  `data/market_engine/source_snapshots/fundamental_metrics/me-data09-aapl-bounded-operator-pilot-20260719T150409Z/`.
- DATA06 and compact RUN31 evidence:
  `artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T150409Z/`
  and `artifacts/market_engine/run_evidence/me-run31-after-me-data09-aapl-20260719T150409Z/`.
- The committed compact RUN31 package records the full-run identity and
  checksums; the large transient full output was not published.

The next bounded continuation is ME-DATA10: select and source-review the next
ticker from the existing deterministic pilot. It must retain explicit approval,
one-ticker evidence scope, direct facts only, and measured downstream outcomes.
