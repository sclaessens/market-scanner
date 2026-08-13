# ME-DATA11 Targeted Diversified Fundamental Derivation Implementation

## Runtime

`targeted_diversified_fundamental_derivation.py` orchestrates one bounded
evidence run. Before cohort selection, `data11_governance.py` validates the
tracked RUN30 authority contract. That contract checksum-binds the ranking,
manifest, canonical universe, and 952-row universe analysis index. Every
top-25 identity, symbol, source symbol, asset type, eligibility field, and
price-history trace is reconciled to that index. Caller-selected files cannot
become authority by having their checksum recorded after loading.

The route resolves CIK identity through the official SEC ticker index and
stores complete CompanyFacts responses as local, uncommitted raw snapshots.
Compact committed evidence retains source paths, checksums, publication dates,
periods, issuer identity, raw tags, canonical mappings, and extraction status.

## Fact and formula behavior

Generic US-GAAP tag priority data maps revenue, gross profit, and operating
income into DATA10 canonical concepts. Duration facts are classified as
discrete quarter, year-to-date, or annual. Selection first fixes the latest
reporting identity and then prefers a discrete quarter for Q1-Q3 or an annual
duration for Q4/FY. The policy includes start, end, fiscal identity, accession,
filing date, and duration class; it is independent of source-array order.
Conflicting semantic duplicates fail closed. Numerators must match the chosen
denominator exactly, and an older complete period never replaces a newer
incomplete period.

The existing DATA10 engine calculates only catalogued gross and operating
margin candidates. Missing facts remain blocked. Debt-to-equity is not offered
unless its complete non-overlapping debt and equity fact set exists. Derived
values remain clearly classified as derived.

## Approval and downstream boundary

Each successful derivation has a persisted approval bundle containing minimal
source evidence, mapping review, fact package, formula catalog, derived output,
validation output, governed DATA07 candidate, and a checksum-bound approval
candidate. The source and fact extracts, mappings, formulas, calculations, and
governed package are semantically replayed during validation. Approval remains
pending until all required human review fields and the mapping decision are
explicitly approved. Without that decision, DATA07, DATA06, and RUN31 receive
zero calls.

The downstream prestate is loaded through a separate tracked authority
contract over the exact DATA06 and RUN31 artifacts. Invalid or unavailable
prestate is `unknown_not_measured`. With no approved import, the authoritative
after-state equals the validated prestate and candidate-only status is reported
separately as `candidate_partial_pending_approval`.

All timestamps use canonical UTC with trailing `Z`. Source publication,
acquisition, generation, and trusted evaluation time are ordered fail closed.
Immutable acquisition-time freshness is retained separately from effective
freshness recalculated at consumption time.

## Evidence

The compact evidence directory and per-ticker approval bundles are:

```text
artifacts/market_engine/run_evidence/
  me-data11-targeted-diversified-fundamental-derivation-20260813T151200Z/
```

Full SEC snapshots remain local under the declared source snapshot root and
are not part of the Git commit. The tracked approval bundles contain the
minimal checksum-bound observations required for clean-checkout replay.

The human review checkpoint must inspect, for each persisted ticker,
`source_evidence.json`, `mapping_review.json`, `fact_package.json`,
`formula_catalog.json`, `derived_package.json`, `derivation_validation.json`,
and `governed_package_candidate.json`; it must then create an explicit approved
decision derived from `approval_candidate.json`. ME-RUN33 remains conditional
on successful approval, DATA07 import, DATA06 refresh, and RUN31 evidence.
