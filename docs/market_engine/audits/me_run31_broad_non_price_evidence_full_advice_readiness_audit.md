# ME-RUN31 Broad Non-Price Evidence Full Advice Readiness Audit

Status: IMPLEMENTED, REVIEW-HARDENED, AND VALIDATED

Draft PR: `#461`

## Review Findings Addressed

The PR review found that the initial ME-RUN31 implementation had these material gaps:

- freshness used a hardcoded historical date;
- technical input defaulted to one historical ME-RUN30 artifact;
- fundamental CSV duplicate rows were silently overwritten;
- market context selection depended on the final CSV row;
- missing source dates could be treated as not stale;
- advice completion could be confused with advice readiness;
- the full run artifact was too large to commit and had no compact committed evidence package.

## Runtime Fixes

ME-RUN31 now requires an explicit technical screening artifact path. The CLI requires:

```text
--technical-screening-artifact <path>
```

Freshness is runtime-driven. The reference date resolves from:

```text
1. explicit --freshness-reference-date
2. technical input manifest cutoff/as-of date
3. fail-closed structural error
```

The new full run used the explicit reference date:

```text
freshness_reference_date: 2026-07-10
resolution_source: explicit_cli
fundamental_stale_after_days: 120
market_stale_after_days: 120
portfolio_stale_after_days: 45
```

Technical input validation now checks path existence, schema, manifest
existence, manifest/index run-id consistency, selected instrument coverage,
unknown instrument IDs, and universe compatibility. The ME-RUN30 artifact does
not persist a universe version, so compatibility for this review run is
documented as `inferred_from_exact_instrument_ids`; explicit universe-version
mismatches still fail closed.

## Selection Policies

Fundamental evidence uses:

```text
selection_policy: latest_valid_source_date
date priority: source_last_updated, source_timestamp, date, generated_at
```

Identical duplicate rows are deduplicated. Same-date conflicting rows become
invalid with `duplicate_fundamental_rows_conflict`. Missing, invalid, or future
fundamental source dates fail closed.

Market context uses:

```text
selection_policy: latest_valid_market_context_date
```

Rows are date-validated, order-independent, and duplicate same-date conflicts
become invalid with `duplicate_market_context_date_conflict`. Missing, invalid,
or future market dates fail closed.

Applicable portfolio context now requires a valid snapshot date. Missing,
invalid, future, or stale dates fail closed for applicable positions.
Not-applicable portfolio context remains non-blocking.

## New Run

Run artifact:

```text
artifacts/market_engine/full_advice_readiness_runs/me-run31-broad-non-price-evidence-full-advice-readiness-20260715T112146Z/
```

Compact evidence package:

```text
artifacts/market_engine/run_evidence/me-run31-broad-non-price-evidence-full-advice-readiness-20260715T112146Z/
```

The previous run `me-run31-broad-non-price-evidence-full-advice-readiness-20260715T095117Z` is superseded by this review-hardened run.

## Compact Evidence

The compact package is committed instead of the full per-ticker artifact tree.
It contains:

```text
manifest.json
run_evidence_index.json
evidence_coverage_summary.json
advice_readiness_report.json
technical_to_advice_transition.json
blocker_report.json
throughput_report.json
full_advice_ranking.json
top_level_checksums.json
```

No per-ticker `dry_run.json` files are included in the compact package.

Full artifact metrics:

```text
file_count: 971
total_size_bytes: 25901389
full_tree_digest: 36aa76ec3a7c1e902f3a9582cb1a41160a373dea86a4493e8513a8a007be70d4
compact_size: 284K
```

## Coverage Result

```text
canonical_instruments: 952
attempted_instruments: 952
technical_analysed: 946
technical_ranking_eligible: 330
blocked: 0
failed: 0
```

Fundamental evidence:

```text
available: 4
partial: 17
missing: 931
stale: 0
invalid: 0
```

Market context:

```text
available: 952
stale: 0
invalid: 0
```

Portfolio context:

```text
available: 1
not_applicable: 951
stale: 0
invalid: 0
```

## Advice Semantics

ME-RUN31 now reports advice engine completion separately from advice readiness:

```text
advice_generation_attempted: 952
advice_engine_completed: 952
canonical_advice_input_ready: 4
non_unable_advice_outputs: 4
unable_to_advise: 948
full_advice_ready: 0
```

Canonical deterministic advice output:

```text
wait_for_price: 4
unable_to_advise: 948
buy_candidate: 0
watchlist: 0
avoid_for_now: 0
hold_existing: 0
take_loss_review: 0
```

The four canonical advice-input-ready tickers remain partial, not full-advice-ready:

```text
GM: wait_for_price, partial, no_clear_setup
PLD: wait_for_price, partial, no_clear_setup
TT: wait_for_price, partial, price_or_risk_not_preferred
WELL: wait_for_price, partial, price_or_risk_not_preferred
```

## Blockers

Top blockers:

```text
evidence_readiness: 948
missing_fundamental_context: 931
no_clear_setup: 407
price_or_risk_not_preferred: 257
weak_or_high_risk_setup: 177
partial_fundamental_context: 17
technical_context_not_available: 6
insufficient_history: 4
insufficient_forward_data: 2
```

The 17 partial fundamental rows are selected deterministically. Examples
include `AAPL`, `ALL`, `AMD`, `BKR`, and `BMY`. Missing fundamental examples
include `A`, `AA`, `AAL`, `AAON`, and `ABBV`.

The only applicable local portfolio context remains `COST`, which is blocked
by missing fundamental context and weak or high-risk setup evidence.

## Full Artifact Compaction Decision

The full artifact still persists per-ticker dry-run files because the existing
deterministic advice engine consumes filesystem artifact paths. Removing or
bundling those files would require a broader advice-engine contract change.
ME-RUN31 therefore keeps full local auditability and commits only the compact
evidence package.

## Governance

The manifest records:

```text
openai_api_invocation_performed: false
model_invocation_performed: false
live_provider_call_performed: false
yfinance_download_performed: false
broker_order_execution_performed: false
allocation_performed: false
portfolio_watchlist_mutation_performed: false
telegram_delivery_performed: false
scheduler_or_worker_started: false
decision_engine_authority_changed: false
parallel_advice_rules_added: false
```

## Recommended Next Sprint

Recommended next sprint:

```text
ME-DATA06 - Expand canonical fundamental evidence coverage from local approved evidence sources
```

Reason: ME-RUN31 now proves the hardened adapter, freshness policy, technical
input validation, evidence selection policies, canonical advice handoff, and
compact committed evidence path. Full-advice readiness remains blocked by
fundamental evidence coverage: 931 missing and 17 partial fundamental contexts.
