# ME-DATA06 Fundamental Evidence Coverage Expansion Audit

Status: IMPLEMENTED WITH MEASURED LOCAL COVERAGE IMPROVEMENT

Draft PR: pending

## Executive Summary

ME-DATA06 implemented a local-only fundamental evidence coverage command and
executed it across the full 952-instrument canonical universe. The run
inventoried local fundamental evidence sources, normalized consumeable rows
into the existing ME-RUN31 `fundamental_quality.csv` contract, and reran the
real ME-RUN31 full-advice-readiness flow against that normalized artifact.

The sprint improved fundamental coverage but did not claim completeness:

```text
fundamental_complete: 4 -> 6
fundamental_partial: 17 -> 39
fundamental_missing: 931 -> 907
canonical_advice_input_ready: 4 -> 6
full_advice_ready: 0 -> 0
unable_to_advise: 948 -> 946
```

Newly advice-input-ready tickers:

```text
ENPH
FTNT
```

## ME-RUN31 Trigger

ME-RUN31 proved that broad technical screening and the canonical advice handoff
work, but the final reproducible run
`me-run31-broad-non-price-evidence-full-advice-readiness-20260715T154103Z`
showed 931 missing and 17 partial fundamental contexts. ME-DATA06 therefore
focused on local fundamental evidence inventory, normalization, validation, and
downstream consumption.

## Scope And Non-Goals

Scope:

- local evidence inventory;
- local evidence normalization into the existing ME-RUN31 fundamental quality
  CSV contract;
- provenance, source-date, freshness, validation, duplicate, and conflict
  handling;
- full canonical-universe coverage reporting;
- real downstream ME-RUN31 rerun using the normalized artifact.

Non-goals:

- no live provider calls;
- no OpenAI or other model calls;
- no synthetic fundamental data;
- no missing numeric values converted to zero;
- no recommendation-rule changes;
- no allocation, broker, portfolio, watchlist, Telegram, scheduler, or
  Decision Engine authority changes.

## Local Evidence Inspection

The implemented inventory found five local evidence source families:

```text
existing_processed_fundamental_quality: 291 tickers found, 290 canonical matches, consumed as baseline
manual_mvp_fundamentals: 36 tickers found, 36 canonical matches, consumed
sec_companyfacts_source_context: 12 tickers found, 12 canonical matches, consumed as partial source context only
intake_placeholder_fundamentals: 10 tickers found, 10 canonical matches, rejected
company_profile: 9 records found, 3 canonical matches, rejected for fundamental quality
```

Rejected sources were not runtime fundamental-quality evidence. Intake rows
were source-required placeholders. Company-profile snapshots were descriptive
identity/profile context only and were not promoted to fundamental quality.

## Normalization

ME-DATA06 writes a normalized CSV under the run artifact:

```text
artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-fundamental-evidence-coverage-expansion-20260715T163629Z/normalized_fundamental_quality.csv
```

The normalized CSV uses the existing ME-RUN31 columns and is passed explicitly
to ME-RUN31. `data/processed/fundamental_quality.csv` is not overwritten.

Source priority:

```text
manual_mvp_fundamentals: 100
existing_processed_fundamental_quality: 80
sec_companyfacts_source_context: 60
```

Manual MVP rows become complete only when all required MVP fields are present,
numeric, current, and non-conflicting. SEC CompanyFacts source context is
preserved as partial evidence because it does not satisfy the current MVP
quality metric contract.

## Provenance And Freshness

The run used:

```text
as_of_date: 2026-07-10
freshness_max_age_days: 120
```

Each selected source records source family, source name, source path, source
reference, source date, selected status, missing metrics, and blockers.
Invalid or future source dates fail closed. Stale sources are classified
separately.

## Coverage Classification

ME-DATA06 uses these deterministic statuses:

```text
complete
partial
missing
stale
invalid
conflicting
unsupported
```

Complete means all existing MVP fundamental quality metrics are present,
numeric, current, and non-conflicting. Partial means at least one approved
local fundamental evidence family exists but the MVP quality contract remains
incomplete.

Family coverage is reported for:

```text
company_identity_profile
revenue_growth
profitability
balance_sheet_strength
cash_flow
valuation
overall_canonical_fundamental_context
```

Valuation remains unsupported because no locally valid valuation evidence was
found.

## Implementation Overview

Implemented command:

```text
src/market_engine/data/fundamental_evidence_coverage.py
```

Targeted tests:

```text
tests/market_engine/data/test_fundamental_evidence_coverage.py
```

The command writes:

```text
manifest.json
evidence_source_inventory.json
fundamental_coverage_summary.json
per_ticker_fundamental_status.json
missing_fundamental_evidence.json
partial_fundamental_evidence.json
invalid_or_stale_evidence.json
before_after_comparison.json
coverage_report.md
normalized_fundamental_quality.csv
```

## Operator Commands

ME-DATA06 run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m market_engine.data.fundamental_evidence_coverage \
  --run-id me-data06-fundamental-evidence-coverage-expansion-20260715T163629Z \
  --as-of-date 2026-07-10 \
  --technical-screening-artifact artifacts/market_engine/universe_analysis_runs/me-run30-full-canonical-universe-analysis-ranking-20260714T143209Z \
  --run31-run-id me-run31-after-me-data06-fundamental-evidence-coverage-20260715T163629Z
```

## Before And After

Before baseline:

```text
source_run_id: me-run31-broad-non-price-evidence-full-advice-readiness-20260715T154103Z
fundamental_complete: 4
fundamental_partial: 17
fundamental_missing: 931
invalid_stale_conflicting: 0
canonical_advice_input_ready: 4
full_advice_ready: 0
unable_to_advise: 948
```

After ME-DATA06:

```text
fundamental_complete: 6
fundamental_partial: 39
fundamental_missing: 907
invalid_stale_conflicting: 0
canonical_advice_input_ready: 6
full_advice_ready: 0
unable_to_advise: 946
```

Transitions:

```text
missing_to_complete: ENPH, FTNT
missing_to_partial: AMAT, ANET, ASML, AVGO, COST, DELL, EOG, EQIX, EW, EXPD, FDX, HAL, HLT, HPE, META, MSFT, NVDA, TSM
partial_to_complete: none
newly_advice_input_ready: ENPH, FTNT
```

## Downstream ME-RUN31 Rerun

Downstream run:

```text
run_id: me-run31-after-me-data06-fundamental-evidence-coverage-20260715T163629Z
full_artifact: artifacts/market_engine/full_advice_readiness_runs/me-run31-after-me-data06-fundamental-evidence-coverage-20260715T163629Z/
compact_evidence: artifacts/market_engine/run_evidence/me-run31-after-me-data06-fundamental-evidence-coverage-20260715T163629Z/
```

Downstream metrics:

```text
attempted_instruments: 952
technical_analysed: 946
technical_ranking_eligible: 330
canonical_advice_input_ready: 6
advice_engine_completed: 952
unable_to_advise: 946
full_advice_ready: 0
failed: 0
```

Advice distribution changed to:

```text
avoid_for_now: 2
wait_for_price: 4
unable_to_advise: 946
```

No full-advice-ready candidate exists yet.

## Validation

Artifact validation confirmed:

```text
json_files: 8
normalized_csv_rows: 952
unique_tickers: 952
complete: 6
partial: 39
missing: 907
```

Targeted tests:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_fundamental_evidence_coverage.py -q
15 passed
```

Additional suite results are recorded in the PR after final validation.

## Remaining Blockers

Dominant remaining blocker:

```text
missing_fundamental_context: 907
```

Nine SEC CompanyFacts tickers are source-context partials only because the
current MVP quality contract requires growth, margin, and balance-sheet fields
that the existing SEC source-context observations do not fully provide.

## Side-Effect Safety

The manifest records no network access, provider calls, model calls,
broker/order execution, allocation, portfolio/watchlist mutation, Telegram
delivery, scheduler behavior, Decision Engine changes, or recommendation-rule
changes.

## Artifact Map

ME-DATA06 artifact:

```text
artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-fundamental-evidence-coverage-expansion-20260715T163629Z/
```

Downstream ME-RUN31 compact evidence:

```text
artifacts/market_engine/run_evidence/me-run31-after-me-data06-fundamental-evidence-coverage-20260715T163629Z/
```

The downstream full artifact tree remains local and uncommitted.

## Conclusion

ME-DATA06 proves that the local evidence expansion flow works and improves the
canonical advice-input-ready count from 4 to 6. The improvement is real but
limited by the amount and shape of currently available local fundamental
evidence.

## Recommended Next Sprint

Recommended next sprint:

```text
ME-DATA07 - Expand validated MVP fundamental metric sourcing for the remaining canonical-universe blockers
```

Reason: ME-DATA06 consumed all currently valid local evidence it could safely
normalize. The next bottleneck is not adapter mechanics but the lack of
approved, current, normalized MVP fundamental metrics for 907 canonical
instruments.
