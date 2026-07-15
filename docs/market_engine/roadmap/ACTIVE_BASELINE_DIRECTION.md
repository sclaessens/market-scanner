# Active Baseline Direction

Status: ACTIVE RESULTS-FIRST BASELINE POINTER

## Active product objective

The active Market Engine baseline is now:

```text
Process a broad ticker universe, such as 500 tickers,
produce deterministic advice labels,
then evaluate which advice rules work and which do not.
```

The baseline is not allowed to drift into additional standalone preparation layers before advice output unless a concrete blocker makes the advice path technically impossible.

## Canonical roadmap lock

```text
docs/market_engine/roadmap/me_rm07_results_first_advice_roadmap_lock.md
```

## Canonical backlog lock

```text
docs/market_engine/backlog/me_rm07_results_first_advice_backlog.md
```

## Supporting no-API guardrail

```text
docs/market_engine/governance/me_gh01_github_first_no_api_baseline_guardrail.md
```

The no-API guardrail remains active: the baseline must not require paid OpenAI API usage or provider invocation.

## Current baseline sequence

```text
ME-GH02 - Batch artifact discovery and ticker status index
  -> ME-ADV01 - Minimal deterministic advice engine v1 (completed)
  -> ME-ADV02 - 500-ticker advice batch output (completed)
  -> ME-DATA01 - Close highest-impact advice data coverage gaps (completed)
  -> ME-EVAL01 - Advice outcome tracking and feedback loop (completed)
  -> ME-EVAL02 - Scheduled/future outcome refresh using local snapshots (completed)
  -> ME-DATA02 - Import missing and forward local price snapshots for unresolved outcomes (implementation complete / coverage partial)
  -> ME-BOOT03 - Bootstrap authoritative universe and local price-history coverage (implementation complete / coverage partial)
  -> ME-DATA04 - Build complete canonical local market dataset (operational dataset partial)
  -> ME-DATA05 - Incremental market data refresh and forward evaluation (completed / incremental_refresh_operational)
  -> ME-RUN30 - Full canonical-universe analysis and candidate ranking (completed / completed_with_blockers)
  -> ME-RUN31 - Add broader non-price evidence to canonical-universe ranking (completed / completed_with_blockers)
  -> ME-DATA06 - Expand canonical fundamental evidence coverage from local approved evidence sources
```

## Superseded baseline pointers

Any older roadmap text that points to provider invocation, API-key propagation, paid OpenAI API advisory generation, ME-CI11D, ME-CI12 provider batch output, or a standalone ME-GH03 ranking/review queue as the next baseline step is superseded for baseline planning.

Review-priority logic may be implemented inside ME-ADV01 if it directly supports advice label production, but it must not become a standalone detour before advice output.

## No-deviation rule

Every future baseline sprint must directly support at least one of these outcomes:

```text
1. process more tickers;
2. make more tickers advice-ready;
3. produce advice labels;
4. evaluate advice quality.
```

If a sprint does not directly support one of those outcomes, it is not part of the baseline roadmap.

## Current completed baseline implementation

ME-ADV01 produced deterministic advice labels from the ME-GH02 ticker status
index and linked dry-run artifacts. The sample run produced concrete
`watchlist` labels for the 12 ME-GH02 sample tickers and did not require
OpenAI API, provider invocation, source acquisition, live data refresh, broker
orders, portfolio/watchlist mutation, Telegram, or delivery side effects.

## Current completed ME-ADV02 result

ME-ADV02 produced deterministic advice batch output under:

```text
artifacts/market_engine/advice_batches/me-adv02-advice-batch-20260711T130000Z/
```

The sample batch consumed the widest available ME-GH02 status index:

```text
artifacts/market_engine/batch_status/me-gh02-sample-status-index-20260711T120000Z/ticker_status_index.json
```

It reported coverage against a 500-ticker target:

```text
tickers in status index: 12
tickers with advice labels: 12
tickers missing artifact/status: 488
coverage percentage: 2.40%
```

Advice distribution:

```text
buy_candidate: 0
wait_for_price: 0
watchlist: 12
avoid_for_now: 0
hold_existing: 0
take_loss_review: 0
unable_to_advise: 0
```

Top missing inputs for buy-candidate diversity:

```text
portfolio_context: 12
setup_price_market_context: 12
```

ME-ADV02 therefore produced visible batch advice output, but it also proved
that outcome tracking was not yet useful because only `watchlist` labels were
produced.

## Current completed ME-DATA01 result

ME-DATA01 added deterministic setup/price/market context extraction from
existing local artifacts and local price-history CSVs. It did not perform live
source acquisition, provider invocation, broker/order execution,
portfolio/watchlist mutation, Telegram, or delivery side effects.

The ME-DATA01 sample batch reused:

```text
artifacts/market_engine/batch_status/me-gh02-sample-status-index-20260711T120000Z/ticker_status_index.json
```

and produced:

```text
run_id: me-data01-setup-price-market-context-20260711T140000Z
tickers with advice labels: 12
coverage percentage: 2.40%
buy_candidate: 4
wait_for_price: 2
watchlist: 5
avoid_for_now: 1
hold_existing: 0
take_loss_review: 0
unable_to_advise: 0
```

Setup/price/market context distribution:

```text
partial: 8
missing: 4
available: 0
invalid: 0
```

ME-DATA01 therefore broke the watchlist-only output and made outcome tracking
useful.

## Current completed ME-EVAL01 result

ME-EVAL01 added deterministic advice outcome tracking under:

```text
src/market_engine/evaluation/
```

It consumes an advice batch `advice_index.json`, reads existing local
`data/processed` price-history CSVs only, computes 5/21/63 trading-day horizon
returns where enough local forward data exists, and writes an evaluation run
under:

```text
artifacts/market_engine/evaluation_runs/<run_id>/
```

The sample run used:

```text
input advice_index: artifacts/market_engine/advice_batches/me-data01-setup-price-market-context-20260711T140000Z/advice_index.json
run_id: me-eval01-advice-outcomes-20260712T120000Z
price_data_root: data/processed
```

and produced:

```text
tickers_total: 12
resolved_outcomes: 0
unresolved_outcomes: 12
resolved_by_horizon:
  1w: 0
  1m: 0
  3m: 0
unresolved_reasons:
  insufficient_forward_data: 8
  missing_price_history: 4
label_counts:
  buy_candidate: 4
  wait_for_price: 2
  watchlist: 5
  avoid_for_now: 1
```

ME-EVAL01 therefore created the feedback loop, but current local data cannot
yet resolve sample outcomes. The dominant blocker is insufficient local forward
history after the recent advice date. Four tickers also lack local
price-history CSVs.

## Current completed ME-EVAL02 result

ME-EVAL02 added a manual local refresh flow for existing unresolved advice
outcomes. It does not implement a scheduler. It can be rerun later when newer
local price-history snapshots are available.

The sample refresh run used:

```text
input evaluation artifact: artifacts/market_engine/evaluation_runs/me-eval01-advice-outcomes-20260712T120000Z/advice_outcome_index.json
run_id: me-eval02-refresh-local-snapshots-20260712T130000Z
price_history_root: data/processed
```

and produced:

```text
selected_outcomes: 12
resolved: 0
still_unresolved: 12
insufficient_forward_data: 8
missing_price_history: 4
missing_price_history_tickers: CLS, CRDO, IREN, VRT
other_blockers: 0
```

ME-EVAL02 proves unresolved advice outcomes can be deterministically retried
against local snapshots. Current local data still cannot resolve any sample
outcome because the local price histories stop before the advice date and four
tickers have no local price-history CSV.

## Current completed ME-DATA02 result

ME-DATA02 added a canonical local market-data universe configuration and a
deterministic local coverage/import command:

```text
config/market_engine/universes/canonical_universe.json
src/market_engine/data/local_market_data_universe.py
src/market_engine/data/supported_universe_price_history_command.py
```

The full report-only data run used:

```text
run_id: me-data02-full-coverage-report-only-20260712T142000Z
price_history_root: data/processed
```

and produced:

```text
total_canonical_instruments: 308
unique_equities: 299
etf_count: 9
context_count: 3
valid: 0
imported: 0
refreshed: 0
missing: 12
insufficient: 293
invalid: 1
unsupported: 2
completion_status: completed_with_blockers
```

ME-DATA02 did not claim full S&P 500, Nasdaq-100, S&P MidCap 400, or STOXX
Europe coverage because no reproducible local membership source was present in
the repository. Those layers are explicitly blocked in the universe
configuration.

The post-ME-DATA02 ME-EVAL02 refresh check remained:

```text
resolved: 0
insufficient_forward_data: 8
missing_price_history: 4
missing_price_history_tickers: CLS, CRDO, IREN, VRT
```

## Next baseline sprint

```text
ME-DATA06 - Expand canonical fundamental evidence coverage from local approved evidence sources
```

Not:

```text
ME-GH03 - Deterministic ranking and review queue
ME-DATA06 - Another data-only infrastructure sprint without an operational blocker
ME-RUN32 - Another broad ranking layer before evidence coverage improves
```

Remaining data gaps are now the operational blocker to full-advice readiness:

```text
missing_fundamental_context: 931
partial_fundamental_context: 17
canonical_advice_input_ready: 4 of 952
full_advice_ready: 0 of 952
```

## Current completed ME-DATA04 result

ME-DATA04 built the complete canonical local market dataset under:

```text
data/processed
```

The completed run established:

```text
total_canonical_instruments: 952
valid_histories: 946
insufficient_histories: 6
missing_histories: 0
invalid_histories: 0
unsupported_mappings: 0
completion_status: completed_with_blockers
```

ME-DATA04 proved the canonical universe, local price-history storage,
validation, coverage reporting, and ME-EVAL02 local-history consumption.

## Current completed ME-DATA05 result

ME-DATA05 added the incremental refresh command:

```text
src/market_engine/data/incremental_market_data_refresh.py
```

The sample operational run used:

```text
run_id: me-data05-incremental-refresh-20260713T140000Z
cutoff_date: 2026-07-10
overlap_calendar_days: 7
provider: Yahoo Finance via yfinance
price_history_root: data/processed
```

and produced:

```text
histories_checked: 952
already_current: 946
incrementally_updated: 0
new_snapshot_created: 0
full_rebuild_completed: 0
stale_after_update: 2
insufficient_history: 4
download_failed: 0
merge_failed: 0
validation_failed: 0
rows_downloaded: 12
rows_added: 0
files_rewritten: 0
files_unchanged: 952
```

The second same-cutoff idempotency run
`me-data05-idempotency-refresh-20260713T141000Z` reproduced those metrics with
`files_rewritten: 0`. Coverage stayed at 946 valid histories and 6
insufficient histories. ME-EVAL02 executed before and after refresh and
remained:

```text
selected_outcomes: 12
resolved: 0
still_unresolved: 12
newly_resolved: 0
blocker_counts:
  insufficient_forward_data: 12
```

ME-DATA05 therefore answers the operational question: the local market dataset
can now be refreshed safely and efficiently without full re-downloads for
already current histories, and coverage plus evaluation are refreshed
automatically. The absence of newly resolved outcomes is a real-data timing
constraint, not a refresh failure.

PR review follow-up clarified two operational boundaries:

```text
refresh_universe_requested: supported as an explicit request flag
refresh_universe_performed: always false unless a supported implementation actually runs
refresh_universe_status: not_requested for normal runs
```

If `--refresh-universe` is requested today, ME-DATA05 fails closed with a clear
error instead of silently pretending that canonical membership changed. The
persisted artifact structure is also compact: `per_ticker_status.json` is the
only complete per-ticker record list, `refresh_summary.json` contains aggregate
metrics only, and duplicate `already_current.json` artifacts are no longer
written.

## Current completed ME-RUN30 result

ME-RUN30 added the broad canonical-universe analysis command:

```text
src/market_engine/run/full_canonical_universe_analysis.py
```

The corrected replacement run used:

```text
run_id: me-run30-full-canonical-universe-analysis-ranking-20260714T143209Z
cutoff_date: 2026-07-10
price_history_root: data/processed
ranking_scope: technical_setup_screening
```

and produced:

```text
total_canonical_instruments: 952
attempted_instruments: 952
eligible_analyzed: 946
blocked_insufficient_history: 4
blocked_stale_history: 2
failed: 0
ranked_candidates: 330
technical_setup_candidate: 105
technical_wait_for_entry: 257
technical_watch: 407
technical_risk_exclusion: 177
unable_to_analyse: 6
```

The PR review fix replaced the original canonical-looking advice labels with
explicit technical screening labels because full canonical advice requires
dry-run artifacts and complete fundamental context that are not yet available
across the broad universe. The determinism rerun matched final statuses,
technical labels, candidate scores, blockers, and candidate ranking order.
Ranking is traceable to canonical setup/price/market context payloads derived
from local price histories and includes a missing-evidence penalty for absent
fundamental, portfolio, and market context. All ranked candidates remain
`full_advice_ready: false`.

## Current completed ME-RUN31 result

ME-RUN31 added the broad non-price evidence advice-readiness command:

```text
src/market_engine/run/broad_non_price_evidence_advice_readiness.py
```

The full run used:

```text
run_id: me-run31-broad-non-price-evidence-full-advice-readiness-20260715T095117Z
technical_screening_artifact: artifacts/market_engine/universe_analysis_runs/me-run30-full-canonical-universe-analysis-ranking-20260714T143209Z/
price_history_root: data/processed
fundamental_evidence_path: data/processed/fundamental_quality.csv
market_context_path: data/processed/market_regime.csv
```

and produced:

```text
attempted_instruments: 952
technical_analysed: 946
technical_ranking_eligible: 330
canonical_advice_input_ready: 4
advice_attempted: 952
advice_completed: 952
failed: 0
wait_for_price: 4
unable_to_advise: 948
full_advice_ready: 0
full_advice_ranking_eligible: 0
missing_fundamental_context: 931
partial_fundamental_context: 17
```

ME-RUN31 proves the broad evidence adapter and deterministic advice handoff
work. It also proves that another ranking layer is not the next bottleneck:
full-advice readiness is blocked primarily by missing local fundamental
evidence coverage.

## Completed ME-ADV01 result

ME-ADV01 produces deterministic advice output with labels:

```text
buy_candidate
wait_for_price
watchlist
avoid_for_now
hold_existing
take_loss_review
unable_to_advise
```

The first target is visible advice output, not another abstract readiness layer.
