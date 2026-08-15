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

The top 25 contained 24 equities and DIA. Accounting framework is unknown for
the equities before source inspection and is not inferred from asset type.
The rank-first ten-equity cohort was ASB, ASH, ATR, AXP, BIO, BKH, BMRN, BMY,
CHRW, and CI.

## Pilot results

Ten existing official SEC CompanyFacts snapshots were inspected and all ten
contain a validated US-GAAP namespace. Three issuers safely produced pending
candidates: ASH and BIO (gross and operating margin) and CI (operating margin).
The other seven instruments fail closed on missing, non-applicable, unaligned,
or conflicting latest-period inputs.

Counts: 10 attempted, 3 pending-success, 7 blocked, 0 failed, 5 derived metric
candidates, 0 direct approved metrics, and 0 approved imports. No authoritative
fundamental or advice-readiness transition occurred. Candidate-only evidence
would make three records partial after approval, but that is not claimed as an
authoritative delta.

## Blockers

- all five derived metrics require separate checksum-bound operator approval;
- ASB, AXP, and BMY lack required latest-period numerator facts for the
  catalogued margin formulas;
- no IFRS namespace was found in the ten inspected sources; no claim is made
  about uninspected top-25 frameworks;
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

## Review remediation

The remediation adds two versioned authority contracts. The RUN30 contract
binds the ranking (`15b933...5530`), manifest (`1361c3...8ae`), universe index
(`175023...5059`), and canonical universe (`c0e2c4...48be`). The downstream
contract binds the exact DATA06 manifest, summary, per-ticker status, and RUN31
compact evidence used as the before state.

The order-independent duration policy reduced the defensible pending set to
ASH, BIO, and CI. Those three have persisted replayable approval bundles; all
remain pending. Other candidates fail closed where aligned facts are missing,
formulas are inapplicable, or semantic duplicates conflict. Candidate-only
status is never authoritative `partial`.

The measured authoritative before state is 6 complete, 39 partial, 907
missing, 0 invalid/stale/conflicting, 6 advice-input-ready, 0 full-advice-ready,
and 946 unable-to-advise. No downstream runner executed, so the authoritative
after state is identical. Zero outside-cohort regressions is proven by the
unchanged checksum-bound authoritative artifacts.

Status after remediation: **READY FOR HUMAN APPROVAL**. A human must review
each ASH, BIO, and CI bundle and issue an explicit checksum-bound decision.
Only a valid approval may activate DATA07; DATA06 and RUN31 follow only after
that import. ME-RUN33 is conditional on the completed approval/import
checkpoint. No provider acquisition, approval, downstream run, recommendation,
allocation, execution, portfolio mutation, or publication occurred.

Final review remediation makes that checkpoint result-bound. Approval
validation returns an immutable binding to the exact governed package,
decision, identity, metric set, and calculation checksums. Authority-carrying
caller kwargs are rejected. DATA06 can start only from a validated DATA07
output and receipt, and RUN31 only from the corresponding validated DATA06
pair. Any blocked, failed, exceptional, malformed, or checksum-inconsistent
stage stops the chain. A downstream after-state is accepted only through the
952-row checksum-bound producer/consumer authority loader; an ordinary mapping
grants no authority. Production acquisition time comes from the internal UTC
clock, while historical replay is manifest-bound and read-only.

Remediation validation results:

- focused ME-DATA11 and blocker regression suites: 128 passed;
- full Market Engine suite: 1,804 passed and the one baseline failure above;
- full repository suite: 2,471 passed and the same one baseline failure;
- Python compilation, strict JSON parsing, JSON Schema validation for every
  new artifact variant, checksum/replay validation, governance greps, and
  `git diff --check`: passed.
