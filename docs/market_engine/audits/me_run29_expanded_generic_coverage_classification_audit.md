# ME-RUN29 - Expanded Generic Coverage Classification Audit

Sprint ID: ME-RUN29
Status: COMPLETED BY ME-RUN29
Job family: ME-RUN / Run and orchestration
Date: 2026-07-03
Architecture layer: Refinery / RUN evidence
Branch: `me-run29-expanded-generic-coverage-classification`

## Purpose

ME-RUN29 executes the completed generic coverage chain:

```text
ME-SA12 generic coverage contract
  -> ME-SA13 generic coverage classifier
  -> ME-SA14 staging-validation adapter
  -> ME-RUN29 expanded evidence classification
```

The run produces inspectable JSON and Markdown evidence for the next planning
sprint without starting Governor work.

## Input Evidence

No committed real staging artifact root is available. Existing local
ME-RUN27/28 staging artifacts are gitignored and are not a reproducible
repository input. They also predate the additive ME-SA14 staging evidence
fields.

ME-RUN29 therefore uses:

```text
tests/fixtures/market_engine/run/me_run29_staging_validation_evidence.json
```

The fixture states explicitly:

```text
ME-RUN29 uses deterministic committed staging-validation fixtures as evidence simulation because no committed real staging artifact root is available.
```

It contains seven generic data rows covering:

* accepted `company_profile`;
* accepted but analytically incomplete `sec_companyfacts`;
* stale `sec_companyfacts`;
* unprovenanced `sec_companyfacts`;
* rejected/non-consumable `sec_companyfacts`;
* missing manifest evidence;
* a staging-accepted but generically unsupported source family.

Ticker and market values are fixture data only. No ticker value appears in
runner control flow.

## Runner

The deterministic local runner is:

```text
scripts/market_engine/me_run29_expanded_generic_coverage_classification.py
```

It:

1. reads one explicit local staging-validation JSON input;
2. validates the staging report contract version;
3. orders entries by data fields;
4. invokes the ME-SA14 adapter for every entry;
5. invokes the ME-SA13 batch classifier;
6. aggregates coverage, readiness, and blocker counts;
7. asserts that reserved authority states remain zero;
8. writes deterministic JSON and Markdown output.

The timestamp, run ID, input path, and artifact root are caller-supplied. The
runner performs no clock read.

## Run Identity

```text
run_id: me-run29-expanded-generic-coverage-classification-20260703T000000Z
classification_timestamp: 2026-07-03T00:00:00Z
input: tests/fixtures/market_engine/run/me_run29_staging_validation_evidence.json
artifact_root: artifacts/market_engine/me-run29-expanded-generic-coverage-classification-20260703T000000Z
target_capability: recommendation_review
```

Generated local artifacts:

```text
coverage_classification_summary.json
coverage_classification_report.md
```

The artifact root is ignored deliberately. The committed fixture, runner,
tests, and audit make the result reproducible.

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/market_engine/me_run29_expanded_generic_coverage_classification.py \
  --input-evidence tests/fixtures/market_engine/run/me_run29_staging_validation_evidence.json \
  --run-id me-run29-expanded-generic-coverage-classification-20260703T000000Z \
  --classification-timestamp 2026-07-03T00:00:00Z \
  --artifact-root artifacts/market_engine/me-run29-expanded-generic-coverage-classification-20260703T000000Z
```

## Classification Summary

```text
tickers_total: 7
generic_source_families_total: 2
staging_entries_total: 7

readiness:
  blocked: 5
  descriptive_only: 1
  unavailable: 1

aggregate coverage:
  descriptive_only: 1
  invalid_manifest: 1
  missing_snapshot: 2
  not_consumable: 1
  stale: 1
  unprovenanced: 1

recommendation_eligible_count: 0
actionable_count: 0
decision_ready_count: 0
de_ready_count: 0
```

The accepted company-profile row remains `descriptive_only`. The accepted SEC
CompanyFacts row has a `partial` fundamental-family result and remains
non-actionable because staging acceptance does not prove analytical
completeness or the other Recommendation Review requirements.

## Dominant Blockers

The dominant aggregate blockers are:

| Blocker | Count | Interpretation |
| --- | ---: | --- |
| `missing_cached_source_snapshot` | 22 | Required Recommendation Review families are absent across the evidence rows. |
| `missing_setup_or_price_context` | 14 | No fixture row claims price/setup evidence. |
| `missing_fundamental_evidence` | 7 | No row proves complete fundamental evidence. |
| `recommendation_review_blocked` | 7 | The Recommendation Review input contract remains absent. |

Specific source-gate blockers are each visible once:

```text
invalid_manifest
missing_provenance
source_family_incomplete
source_not_consumable
stale_source
unsupported_source_family
company_profile_only_context_non_actionable
```

These blockers classify evidence sufficiency. They are not investment
opinions, scores, rankings, or recommendations.

## Reserved-State Confirmation

The machine-readable artifact records:

```text
actionable: 0
actionable_review: 0
decision_ready: 0
de_ready: 0
```

No recommendation-state upgrade occurs. The result is suitable as evidence
input for ME-GV01 only; it does not define or execute a Governor contract.

## Governance Boundary

ME-RUN29 remains entirely in Refinery / RUN evidence.

It adds no source acquisition, expanded acquisition coverage, provider call,
SEC/EDGAR live call, yfinance use, network call, snapshot import, staging
validator semantic change, ME-SA13 classifier semantic change, ME-SA14 adapter
semantic change, Governor code, Dispatch Station code, delivery, production
write, portfolio/watchlist mutation, scheduler/UI/broker behavior, scoring,
ranking, BUY/SELL/HOLD semantics, allocation, target price, target weight,
position sizing, order generation, execution advice, or Decision Engine
authority.

## Validation

```text
5 passed - focused ME-RUN29 tests
78 passed - tests/market_engine/source_support
63 passed - tests/market_engine/source_refresh
119 passed - tests/market_engine/run
605 passed - tests/market_engine
1272 passed - full pytest
```

All requested test paths exist, so no path substitution was required.

Governance grep interpretation:

* the ME-RUN29 runner contains no ticker literal from the requested
  ticker-specific audit set;
* synthetic ticker strings exist only in committed fixture/test data and are
  verified not to appear in runner source;
* new authority-term runtime hits are fixed-false side-effect fields,
  reserved-state counts, boundary text, and zero-state assertions;
* new documentation hits are negative governance statements;
* repository-wide results include pre-existing historical docs, fixtures,
  legacy scripts, and contract guardrails outside the ME-RUN29 diff;
* the mandatory legacy `scripts/` BUY/SELL grep still reports pre-existing
  portfolio and bytecode hits; ME-RUN29 does not modify those paths;
* `/dev/tty` is unavailable in the managed execution environment, so both
  requested `tee /dev/tty` commands emitted their `rg` results and then
  reported an environment error. Direct scoped fallback greps were clean
  except for the expected guardrail/reserved-state references above.

## Next Sprint

```text
ME-GV01 - Define The Governor investment evaluation contract
```

ME-GV01 is a contract sprint. ME-RUN29 does not pre-implement it.
