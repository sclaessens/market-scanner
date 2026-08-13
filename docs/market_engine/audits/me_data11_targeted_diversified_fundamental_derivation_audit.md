# ME-DATA11 Targeted Diversified Fundamental Derivation Audit

Status: COMPLETED WITH BLOCKERS

Date: 2026-08-13

Base: `9169420427d33864851d36f2b183e35b8bd0c089`

## Authoritative inputs

- technical run: `me-run30-full-canonical-universe-analysis-ranking-20260714T143209Z`;
- technical cutoff: `2026-07-10`;
- ranking SHA-256: `15b933748a1b12f5957798ebd6636a609e72f97eb8fb19af6d0259ad201f5530`;
- ranking manifest SHA-256: `1361c3d16aad412025d0dbd9612dfacfca0eeeff762fbd31bd73f0ad04a2ebae`;
- canonical universe SHA-256: `c0e2c466af51ee3f34202148b5275f46248c893ce4b3851352d0fbded87148be`;
- formula catalog SHA-256: `7ae5ca07ad9b45eaed54fccd9e77de254616da6f598d42cd44ff78dcd80673d1`.

The top 25 contained 24 US-GAAP equities and DIA. It contained no IFRS issuer.
The rank-first ten-equity cohort was ASB, ASH, ATR, AXP, BIO, BKH, BMRN, BMY,
CHRW, and CI.

## Pilot results

Ten official SEC CompanyFacts snapshots were acquired. Seven issuers safely
produced pending candidates: ASH (gross and operating margin), ATR (operating
margin), BIO (gross and operating margin), BKH (operating margin), BMRN
(operating margin), CHRW (operating margin), and CI (operating margin). ASB,
AXP, and BMY were blocked because their newest aligned revenue period did not
contain an approved gross-profit or operating-income numerator.

Counts: 10 attempted, 7 pending-success, 3 blocked, 0 failed, 9 derived metric
candidates, 0 direct approved metrics, and 0 approved imports. No authoritative
fundamental or advice-readiness transition occurred. Candidate-only evidence
would make seven records partial after approval, but that is not claimed as an
authoritative delta.

## Blockers

- all nine derived metrics require separate checksum-bound operator approval;
- ASB, AXP, and BMY lack required latest-period numerator facts for the
  catalogued margin formulas;
- the funnel provides no IFRS issuer, so cross-framework runtime evidence is
  not available in this bounded pilot;
- direct approved revenue-growth and EPS-growth evidence is absent;
- ME-RUN33 cannot consume pending evidence as authoritative input.

## Safety result

The route reused DATA10 and introduced no ticker-specific runtime branch, new
formula, recommendation, allocation, broker, order, portfolio, notification,
or publication behavior. `market-data` remained at
`95c88276763b1762cbbfbccc402ec8535268127b`.

## Validation

- focused ME-DATA11 suite: 46 passed;
- DATA06–10, RUN30/31, recommendation, handoff, and advisory selection: 523 passed;
- full Market Engine suite: 1,722 passed, 1 known baseline failure;
- full repository suite: 2,389 passed, 1 known baseline failure;
- Python compilation, strict JSON parsing, checksum reconciliation, and
  `git diff --check`: passed.

The only failure is
`test_compact_checksums_match_committed_files_and_local_full_runs`, at the
assertion that the historical
`me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json` exists. The
identical test and assertion fail on the exact clean base
`9169420427d33864851d36f2b183e35b8bd0c089`; it is therefore classified as the
documented baseline failure and does not mask a ME-DATA11 regression.
