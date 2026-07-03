# ME-SA14 - Staging Validation to Generic Coverage Adapter Audit

Sprint ID: ME-SA14
Status: COMPLETED BY ME-SA14
Job family: ME-SA / Source Acquisition and Source Coverage
Date: 2026-07-03
Architecture layer: Refinery

## Purpose

ME-SA14 closes the implemented generic coverage chain:

```text
ME-SA12 generic coverage contract
  -> ME-SA13 deterministic generic coverage classifier
  -> ME-SA14 staging-validation evidence adapter
```

The adapter converts existing cached-source staging-validation entries into
`CachedSourceCoverageInput` values accepted by:

```text
classify_cached_source_coverage(...)
classify_cached_source_coverage_batch(...)
```

ME-SA14 does not execute the expanded classification run. ME-RUN29 remains the
next logical sprint.

## Implementation

The new Refinery adapter is:

```text
src/market_engine/source_support/staging_validation_coverage_adapter.py
```

Public APIs:

```python
adapt_staging_validation_to_cached_source_coverage_input(...)
adapt_staging_validation_batch_to_cached_source_coverage_inputs(...)
```

The adapter is pure and deterministic. It consumes an in-memory validation
entry, report, or entry sequence. It performs no filesystem read, clock read,
provider call, network call, acquisition, persistence, or delivery action.

## Mapping

The mapping is intentionally narrow:

| Staging evidence | Generic coverage input |
| --- | --- |
| `company_profile` | `company_profile` |
| `sec_companyfacts` | `fundamental_facts` |
| unknown or unapproved family | unsupported, or an explicit fail-closed blocker when no family can be identified safely |
| payload or staged directory present | available evidence |
| missing payload and directory evidence | missing snapshot |
| missing, malformed, unknown, inconsistent, or integrity-invalid manifest | invalid manifest |
| required identity or retrieval lineage missing | unprovenanced |
| `fresh` | accepted freshness |
| stale, unknown, or missing freshness | stale |
| staging accepted, validation passed, no validation errors, and usable | consumable |
| otherwise | not consumable |
| accepted `company_profile` | complete for descriptive coverage only |
| `sec_companyfacts` staging evidence | partial until analytical completeness is proved elsewhere |

No source-family alias upgrades capability. In particular,
`sec_companyfacts` staging acceptance proves package validation, not complete
fundamental analysis evidence.

## Validator Output Compatibility

The existing staging validator now exposes already-read manifest metadata in
each result entry:

* market/exchange;
* source retrieval and publication timestamps;
* validation errors and warnings;
* cached-source usability;
* blocked reason.

These are additive audit fields. Manifest validation, staging acceptance, and
rejection semantics are unchanged.

`CachedSourceCoverageInput` and its classification output now preserve the
optional market value as data. Batch uniqueness uses the `(ticker, market)`
instrument key. Neither ticker nor market selects classification behavior.

## Fail-Closed Behavior

The adapter rejects malformed entry shapes, missing ticker identity,
unsupported validation states, malformed issue collections, unsupported
report versions, and empty batches with
`StagingValidationCoverageAdapterError`.

Missing manifests, missing provenance, stale evidence, validation failures,
unsupported families, non-consumable snapshots, and incomplete evidence map
to the corresponding generic coverage gate without fallback or inference.
`source_family_hint` is available only when the caller already has a governed
generic family identity; it does not make an unknown raw family supported.

## Test Coverage

Focused tests cover:

* accepted company-profile staging evidence and descriptive-only output;
* rejected validation and explicit consumability blocking;
* missing manifest;
* missing provenance;
* stale evidence;
* unsupported source family;
* partial SEC CompanyFacts evidence;
* deterministic batch ordering;
* ticker-independent behavior;
* malformed input and unsupported report versions;
* unreachable actionable and Decision Engine-ready states.

## Governance Boundary

ME-SA14 remains Refinery-only. It adds no Governor or Dispatch Station
behavior and no Recommendation Review execution.

It performs no provider or live-source access, SEC/EDGAR call, yfinance use,
network access, acquisition, expanded run, production write, portfolio write,
watchlist write, delivery, scheduler, UI, broker, order, allocation, ranking,
scoring, urgency, conviction, tradeability, target-price, target-weight, or
position-sizing behavior.

Reserved `actionable`, `actionable_review`, `decision_ready`, and `de_ready`
states remain unreachable. Decision Engine authority is unchanged.

## Validation

```text
78 passed - tests/market_engine/source_support
63 passed - tests/market_engine/source_refresh
600 passed - tests/market_engine
1267 passed - full pytest
```

The source-support and source-refresh paths both exist, so no requested test
path substitution was required.

Governance grep interpretation:

* no ticker literal was added to adapter runtime;
* apparent ticker hits in the scoped adapter grep are substrings in constant
  names such as `SNAPSHOT`, not instrument branches;
* new authority-term hits are contract enums, negative guardrail assertions,
  tests proving false/unreachable states, and audit boundary text;
* repository-wide hits include pre-existing historical documentation, tests,
  fixtures, and legacy runtime outside the ME-SA14 diff;
* pre-existing `scripts/portfolio` BUY/SELL hits remain visible under the
  repository-mandated legacy grep; ME-SA14 does not modify `scripts/`;
* `/dev/tty` is unavailable in the managed execution environment, so the
  requested `tee /dev/tty` commands reported an environment error after `rg`
  emitted their results. Scoped fallback greps were run directly.

## Next Sprint

```text
ME-RUN29 - Run expanded generic coverage classification from staging-validation evidence
```

ME-RUN29 may consume the new adapter but must remain local, evidence-only,
non-actionable, mutation-free, and delivery-free.
