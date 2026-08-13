# ME-DATA11 — Targeted Diversified Fundamental Derivation

Sprint ID: ME-DATA11

Status: COMPLETED WITH BLOCKERS

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
- extract canonical US-GAAP duration facts without ticker-specific branches;
- reuse the DATA10 formula catalog and derivation engine;
- keep every derivation pending until a separate checksum-bound operator
  decision is supplied;
- emit a complete top-25 comparison matrix and honest downstream readiness
  delta.

## Pilot outcome

Run `me-data11-targeted-diversified-fundamental-derivation-20260813T151200Z`
selected ASB, ASH, ATR, AXP, BIO, BKH, BMRN, BMY, CHRW, and CI. The top-25
contained no IFRS issuer, so framework diversity was not forced.

All ten official source snapshots were acquired. Seven issuers produced nine
pending derived metric candidates. ASB, AXP, and BMY remained blocked because
the latest aligned period lacked the numerator facts required by the approved
margin formulas. Missing facts were not treated as zero and older, fuller
periods were not substituted for fresher evidence.

No approved direct package or operator derivation decision existed for this
cohort. Therefore DATA07, DATA06, and RUN31 were not executed. Authoritative
fundamental and advice-readiness counts remain unchanged.

## Boundaries

ME-DATA11 added no advice label, cross-ticker score, valuation formula,
position sizing, portfolio mutation, notification, publication, broker or
order behavior. It did not change `market-data` and did not implement
ME-RUN33.

## Next

ME-RUN33 remains the next sprint. Its usable authoritative fundamental input
depends on human review and checksum-bound approval of acceptable ME-DATA11
candidates; pending artifacts alone grant no consumption authority.
