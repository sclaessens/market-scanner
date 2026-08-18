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

Framework is unknown in the top-25 funnel until source inspection. A non-empty
`us-gaap` namespace establishes US-GAAP for the inspected issuer; IFRS-only,
ambiguous, or unsupported namespaces fail closed. Generic US-GAAP tag priority data maps revenue, gross profit, and operating
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
zero calls. Successful validation captures the exact decision and approval
bundle bytes and binds their checksums, package ID, ticker, instrument,
approved metrics, and calculation checksums into a private execution context.
Callers may supply only explicitly allowlisted operational settings. Every
stage requires a versioned completed result, receipt, exact input binding, and
matching artifact checksums; a blocked, failed, exceptional, or malformed
stage prevents every later invocation.

The downstream prestate is loaded through a separate tracked authority
contract over the exact DATA06 and RUN31 artifacts. Invalid or unavailable
prestate is `unknown_not_measured`. With no approved import, the authoritative
after-state equals the validated prestate and candidate-only status is reported
separately as `candidate_partial_pending_approval`.

When downstream execution is claimed, a mapping is never accepted directly.
The after-state loader requires the validated stage-chain proof plus bound
DATA06 and RUN31 artifacts, exact run and universe lineage, and complete
per-ticker reconciliation across all 952 canonical instruments.

All timestamps use canonical UTC with trailing `Z`. Production generation and
acquisition use one internal UTC clock; neither is exposed as a normal CLI
authority override. Source publication, acquisition, generation, and trusted
evaluation time are ordered fail closed.
Immutable acquisition-time freshness is retained separately from effective
freshness recalculated at consumption time. Historical replay is explicitly
read-only, checksum-bound to its manifest, and grants no provider, approval, or
downstream mutation authority.

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

The only supported ME-DATA11 production execution route is
`execute_approved_candidate(decision_path, ...)`. That entrypoint loads and
validates the decision and every bound approval artifact in one call, captures
their exact bytes, and gives DATA07 only a temporary checksum-verified snapshot
of those bytes. DATA07 input, output, and receipt bind both the decision SHA-256
and governed-package SHA-256. The temporary snapshot is removed after the
bounded stage chain returns.

Callers cannot submit a prebuilt approval-validation mapping or construct an
execution binding through a public factory. Caller-controlled operational
values are limited to a validated run ID, a new non-escaping output root under
`artifacts/market_engine`, and an optional logging level. Canonical universe,
price history, baselines, raw snapshot root, and temporal inputs are derived
internally from tracked authorities. This boundary is enforced by the public
API and immutable dataflow; a Python object token is not treated as a security
boundary.
