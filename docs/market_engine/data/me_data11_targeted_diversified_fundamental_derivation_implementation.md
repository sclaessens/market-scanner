# ME-DATA11 Targeted Diversified Fundamental Derivation Implementation

## Runtime

`targeted_diversified_fundamental_derivation.py` orchestrates one bounded
evidence run. It validates the ME-RUN30 ranking and manifest, retains exactly
the top 25 eligible machine-readable candidates, and selects the highest-
ranked supported equities without a ticker allowlist.

The route resolves CIK identity through the official SEC ticker index and
stores complete CompanyFacts responses as local, uncommitted raw snapshots.
Compact committed evidence retains source paths, checksums, publication dates,
periods, issuer identity, raw tags, canonical mappings, and extraction status.

## Fact and formula behavior

Generic US-GAAP tag priority data maps revenue, gross profit, and operating
income into DATA10 canonical concepts. The newest revenue period is selected
first; numerator facts must match its start, end, fiscal identity, and SEC
accession. An older complete period never replaces a newer incomplete period.

The existing DATA10 engine calculates only catalogued gross and operating
margin candidates. Missing facts remain blocked. Debt-to-equity is not offered
unless its complete non-overlapping debt and equity fact set exists. Derived
values remain clearly classified as derived.

## Approval and downstream boundary

The acquisition and canonical mapping references are review candidates, not
approvals. DATA10 output is marked `pending_no_authority`. The orchestrator
does not call DATA07 when no authoritative checksum-bound decision is present;
consequently DATA06 and RUN31 also remain unexecuted. The downstream artifact
records identical authoritative before and after states and separately reports
candidate-only potential.

## Evidence

The compact nine-file evidence directory is:

```text
artifacts/market_engine/run_evidence/
  me-data11-targeted-diversified-fundamental-derivation-20260813T151200Z/
```

Full SEC snapshots remain local under the declared source snapshot root and
are not part of the Git commit.
