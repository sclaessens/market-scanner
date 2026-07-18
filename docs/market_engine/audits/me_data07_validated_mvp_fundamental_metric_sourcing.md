# ME-DATA07 Validated MVP Fundamental Metric Sourcing Audit

Status: IMPLEMENTED / OPERATOR IMPORT OPERATIONAL / ACQUISITION BLOCKED BY MISSING OPERATOR EVIDENCE

## 1. Executive Summary

ME-DATA07 implemented a deterministic, local-first sourcing boundary for the
remaining canonical-universe fundamental evidence gaps. Repository inspection
did not find a governance-approved provider adapter that covers the complete
MVP metric contract. The only safe route is therefore a validated,
operator-supplied primary-source package.

The actual pilot run failed closed because the explicit operator input did not
exist. It performed zero provider and network calls, persisted no raw snapshot,
ran no downstream ME-DATA06 or ME-RUN31 flow, and claimed no coverage change.
This is the governance-proof blocked outcome defined for the sprint.

## 2. ME-DATA06 Baseline

The validated ME-DATA06 baseline is:

```text
ME-DATA06: me-data06-fundamental-evidence-coverage-review-fix-20260718T113254Z
ME-RUN31:  me-run31-after-me-data06-review-fix-20260718T113254Z
canonical universe: 952
complete: 6
partial: 39
missing: 907
invalid/stale/conflicting: 0
canonical advice-input-ready: 6
full-advice-ready: 0
unable-to-advise: 946
```

## 3. Scope And Non-Goals

Implemented scope:

- full baseline gap analysis;
- deterministic sourcing tiers;
- source approval evidence;
- explicit canonical-to-source symbol mapping;
- operator-package schema, validation, normalization, and snapshot boundary;
- controlled pilot, expanded, and full batch selection;
- explicit downstream ME-DATA06 gate after successful validation;
- compact run, blocker, validation, and coverage artifacts.

Non-goals remained unchanged: no synthetic data, investment recommendation,
allocation, sizing, conviction, broker or order action, portfolio or watchlist
mutation, Telegram delivery, model invocation, Decision Engine change, or
technical ranking change.

## 4. Current MVP Metric Contract

The implementation imports `MVP_METRIC_FIELDS` and
`FRESHNESS_MAX_AGE_DAYS` from the merged ME-DATA06 runtime. The current contract
requires:

```text
revenue_growth_yoy
eps_growth_yoy
gross_margin
operating_margin
debt_to_equity
freshness maximum: 120 days
```

Complete means that all five metrics are numeric, current, same-period,
non-conflicting, provenance-backed, and normalized as ratios. Growth values
must be source-reported comparable year-over-year values. Debt-to-equity must
be a source-reported total-debt-to-total-equity ratio. The flow performs no
derivation, annualization, averaging, imputation, or null-to-zero conversion.

## 5. Gap Analysis

The run reconciled 952 unique canonical tickers:

```text
complete: 6
partial: 39
missing: 907
```

Missing metric counts:

```text
revenue_growth_yoy: 933
eps_growth_yoy: 939
gross_margin: 938
operating_margin: 935
debt_to_equity: 944
```

The detailed artifact records one row per ticker. Compact aggregations cover
missing combinations, region, asset type, selected source family, mapping
status, sourcing eligibility, technical candidate status, readiness blocker,
and sourcing tier.

## 6. Sourcing Priority

Deterministic tier counts were:

```text
tier_1: 322
tier_2: 22
tier_3: 593
not selected/eligible: 15
```

Tier 1 contains mapped technical candidates with fundamental context as the
remaining evidence-readiness gap. Tier 2 contains remaining partial contexts.
Tier 3 contains remaining mapped canonical equities. The pilot selects the
first 12 tickers by tier and stable ticker order. This is a data-sourcing
priority only and carries no investment quality or allocation meaning.

## 7. Source Routes Inspected

The source gate inspected:

- existing local ME-DATA06 evidence: operational but already consumed;
- SEC CompanyFacts: bounded smoke adapter, not an approved broad provider and
  incomplete for the five-ratio contract;
- automated cached-source acquisition: company-profile evidence only;
- operator-supplied local import: approved when the package validates;
- third-party provider acquisition: blocked because provider identity,
  licensing, approval, full-contract mapping, and configuration are absent;
- credentialed acquisition: blocked because credentials cannot authorize an
  otherwise unapproved provider route;
- unsupported region/exchange mappings: fail closed.

## 8. Source Approval Decision

The selected mode was `operator_import`, with approval status
`approved_operator_supplied_import`. Authentication is not required for a
local package. The run still blocked because the explicit path
`operator_input/market_engine/me-data07/fundamental_metrics.json` did not exist.

The request, retry, timeout, and budget limits are all zero. No secret name or
value is persisted in artifacts.

## 9. Import Architecture

The implemented flow is:

```text
operator package
  -> schema and identity validation
  -> canonical symbol mapping validation
  -> metric/date/period/unit/provenance validation
  -> immutable run-scoped raw snapshot
  -> normalized metric evidence
  -> explicit ME-DATA06 input
  -> real ME-RUN31 rerun through ME-DATA06
```

The raw snapshot and downstream steps are reachable only after the entire
selected import validates. A failed or missing package writes blocker evidence
but no raw snapshot.

## 10. Ticker Mapping

The actual canonical mapping inventory reported:

```text
mapped: 939
mapped_with_explicit_alias: 4
unsupported_asset_type: 9
ambiguous: 0
missing_provider_symbol: 0
rejected_duplicate_listing: 0
```

Aliases come from the canonical universe override contract. ETFs are not
silently assigned company fundamentals. Ambiguous, duplicate-listing,
unsupported-exchange, and missing-symbol states fail closed.

## 11. Snapshot And Provenance Contract

A successful import writes a run-scoped immutable copy under
`data/market_engine/source_snapshots/fundamental_metrics/<run_id>/` with a raw
SHA-256 checksum, timestamp, request identity, parser version, ticker list,
record count, and provenance. Each normalized record retains instrument and
provider identity, provider symbol, source and reporting dates, reporting
period, acquisition timestamp, source reference, parser version, source
package checksum, normalized record checksum, snapshot ID, and per-metric
lineage.

The actual blocked pilot produced no snapshot because no source package was
available.

## 12. Metric Normalization

Supported input units are explicit `ratio` and `percent`. Percent values are
divided by 100; ratios use identity transformation. Negative numeric values
remain valid. Missing and null values remain missing. Invalid numeric types,
unsupported units, future dates, stale dates, incomplete provenance, identity
mismatches, mapping failures, same-record period mismatches, and conflicting
duplicate periods fail validation. Identical duplicate periods are
deterministically deduplicated and counted.

## 13. Pilot Result

Actual run:

```text
run_id: me-data07-validated-mvp-fundamental-metric-sourcing-20260718T122028Z
batch: pilot
selected: 12
provider calls: 0
imports attempted: 1
imported: 0
normalized: 0
complete: 0
partial: 0
blocked: 12
reason: operator_import_package_missing
```

Pilot tickers were `AAPL`, `ALL`, `AMD`, `A`, `AAL`, `ABBV`, `ABNB`, `ASML`,
`BF.B`, `ADC`, `ADM`, and `ADP`. The set deliberately includes partial and
missing contexts, multiple sectors, technical candidates, and canonical alias
cases. A non-US primary-listing claim was not made because no concrete source
package existed to establish region support.

## 14. Expanded And Full Batch Result

Expanded and full acquisition were not executed. The pilot did not obtain or
validate evidence, so the broader-batch gate remained closed. This prevented
unapproved acquisition and an unsupported coverage claim.

## 15. Validation Results

Targeted tests cover gap ordering, source decisions, secrets, exact and alias
mappings, ADR/listing metadata, share classes, ambiguous and missing mappings,
ETFs, duplicate listings, operator import, batch scoping, nulls, negative
values, units, periods, dates, freshness, identity, provenance, checksums,
duplicates, conflicts, snapshots, overwrite protection, downstream invocation,
and safety guardrails.

The actual artifact validation reconciled 11 JSON artifacts, 952 unique sorted
tickers, unique symbol mappings, baseline counts, zero requests, zero imported
records, no snapshot, no downstream run, and identical before/after coverage.

Executed test results:

```text
ME-DATA07 targeted: 30 passed
data tests: 105 passed
ME-RUN31 targeted: 37 passed
tests/market_engine: 1130 passed
full pytest: 1797 passed
```

## 16. Before And After ME-DATA06

No ME-DATA06 rerun was allowed because sourcing did not produce validated
evidence:

```text
complete: 6 -> 6
partial: 39 -> 39
missing: 907 -> 907
invalid/stale/conflicting: 0 -> 0
```

The after values are an explicit unchanged baseline, not a measured downstream
rerun or a coverage-improvement claim.

## 17. Before And After ME-RUN31

No ME-RUN31 rerun was allowed because ME-DATA06 was not executed:

```text
canonical advice-input-ready: 6 -> 6
full-advice-ready: 0 -> 0
unable-to-advise: 946 -> 946
```

## 18. Newly Ready Tickers

Not computable from a new downstream run because no evidence was imported.
No newly ready ticker is claimed.

## 19. Regressions

Not computable from a new downstream run. No empty regression list is
presented as measured evidence. Baseline coverage was not mutated.

## 20. Remaining Blockers

The immediate external blocker is the missing governance-approved operator
package with primary-source lineage. Baseline evidence remains partial for 39
tickers and missing for 907. Nine ETFs are intentionally excluded from the
company-fundamental sourcing contract.

## 21. Side-Effect Safety

The manifest records zero provider and network calls, zero model invocation,
no broker/order execution, no allocation, no portfolio/watchlist mutation, no
Telegram delivery, no Decision Engine change, and no recommendation-rule
change. No secret value is read into or written by the import route.

## 22. Operator Command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m market_engine.data.validated_fundamental_metric_sourcing \
  --run-id me-data07-validated-mvp-fundamental-metric-sourcing-20260718T122028Z \
  --source-mode operator_import \
  --batch-tier pilot \
  --baseline-data06-run artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-fundamental-evidence-coverage-review-fix-20260718T113254Z \
  --baseline-run-evidence artifacts/market_engine/run_evidence/me-run31-after-me-data06-review-fix-20260718T113254Z \
  --operator-import-path operator_input/market_engine/me-data07/fundamental_metrics.json
```

The command exits non-zero for this blocked result. A validated package may be
processed again with a new run ID. Downstream execution additionally requires
`--execute-downstream` and explicit downstream run IDs.

## 23. Artifact Map

```text
artifacts/market_engine/fundamental_metric_sourcing_runs/
  me-data07-validated-mvp-fundamental-metric-sourcing-20260718T122028Z/
```

The root contains the manifest, source decision, gap analysis, sourcing plan,
symbol mapping, batch summary, single per-ticker detail artifact, validation
summary, normalized evidence artifact, before/after artifact, blocker report,
and Markdown report.

## 24. Known Limitations

- No approved full-contract live provider exists in the repository.
- No operator package was available for the actual pilot.
- Expanded/full sourcing and downstream coverage measurement remain gated.
- Region and listing coverage can only be established against a concrete
  approved source package or provider contract.
- Source licensing remains an operator responsibility until a documented
  provider route is approved.

## 25. Recommended Next Sprint

```text
ME-DATA08 - Prepare and validate a governance-approved operator fundamental metric package
```

The next sprint should provide a permitted primary-source package for the
ME-DATA07 pilot, including source references, dates, periods, provider symbols,
units, parser identity, and all available MVP metrics. It should then execute
the existing ME-DATA07 import and downstream gates. It must not introduce a new
provider architecture without an explicit source, licensing, credential,
mapping, and persistence approval contract.
