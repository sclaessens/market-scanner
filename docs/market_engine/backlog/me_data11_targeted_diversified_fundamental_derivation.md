# ME-DATA11 — Targeted Diversified Fundamental Derivation

Sprint ID: ME-DATA11

Status: READY FOR HUMAN APPROVAL

Job family: ME-DATA / Fundamental Evidence

## Purpose

Prepare the highest-ranked technical equity candidates for later comparison
without creating a fundamental investment ranking or moving allocation
authority outside the Decision Engine.

## Implemented scope

- validate and checksum-bind the authoritative ME-RUN30 machine-readable
  ranking, manifest, canonical universe, cutoff date, and top 25;
- select a deterministic rank-first cohort of 8–12 supported equities;
- resolve issuer identity with the official SEC ticker index and acquire
  bounded official CompanyFacts snapshots;
- classify framework from inspected source namespaces and extract canonical
  US-GAAP duration facts without ticker-specific branches;
- reuse the DATA10 formula catalog and derivation engine;
- keep every derivation pending until a separate checksum-bound operator
  decision is supplied;
- emit a complete top-25 comparison matrix and honest downstream readiness
  delta.

## Pilot outcome

Run `me-data11-targeted-diversified-fundamental-derivation-20260813T151200Z`
selected ASB, ASH, ATR, AXP, BIO, BKH, BMRN, BMY, CHRW, and CI. Framework
remains unknown for the uninspected top-25; all ten inspected cohort sources
validated as US-GAAP, and no IFRS namespace was found in that inspected cohort.

The original run acquired ten official source snapshots. Review remediation
replayed those existing local snapshots without provider acquisition and
applied the order-independent duration-selection and conflict rules. ASH, BIO,
and CI now have persisted, replayable pending approval bundles. Remaining
issuers fail closed on missing, non-applicable, unaligned, or conflicting facts.

No approved direct package or operator derivation decision existed for this
cohort. Therefore DATA07, DATA06, and RUN31 were not executed. Authoritative
fundamental and advice-readiness counts remain unchanged.

## Boundaries

ME-DATA11 added no advice label, cross-ticker score, valuation formula,
position sizing, portfolio mutation, notification, publication, broker or
order behavior. It did not change `market-data` and did not implement
ME-RUN33.

## Approval/import checkpoint

The next action is not ME-RUN33. A human must review the source evidence,
mapping, fact, formula, derivation, validation, and governed-package files for
ASH, BIO, and CI and provide explicit checksum-bound decisions. Valid approved
bundles may then pass through DATA07, followed by DATA06 and RUN31. ME-RUN33
remains conditional on that successful, result- and receipt-bound checkpoint;
pending artifacts or caller-supplied after-state mappings grant no authority.
The supported production route begins at the approved decision path and uses a
checksum-verified immutable approval-bundle snapshot; no caller-created
validation mapping or binding can start the stage chain.
