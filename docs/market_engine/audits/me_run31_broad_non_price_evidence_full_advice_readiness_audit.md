# ME-RUN31 Broad Non-Price Evidence Full Advice Readiness Audit

Status: IMPLEMENTED AND VALIDATED

Run artifact:

```text
artifacts/market_engine/full_advice_readiness_runs/me-run31-broad-non-price-evidence-full-advice-readiness-20260715T095117Z/
```

## Scope

ME-RUN31 connects available local non-price evidence to the broad canonical-universe analysis result from ME-RUN30 and hands the resulting per-ticker input index to the existing deterministic advice engine.

The sprint preserves separation between technical setup screening and canonical deterministic advice. It does not copy canonical advice rules into the broad run module, does not create allocation authority, and does not perform broker, order, portfolio, watchlist, Telegram, provider, scheduler, or production side effects.

## Inputs

Technical screening source:

```text
artifacts/market_engine/universe_analysis_runs/me-run30-full-canonical-universe-analysis-ranking-20260714T143209Z/
```

Local evidence inputs:

```text
config/market_engine/universes/canonical_universe.json
data/processed
data/processed/fundamental_quality.csv
data/processed/market_regime.csv
data/market_engine/portfolio_contexts/local_portfolio_context.json
```

## Produced Artifacts

The run writes:

```text
manifest.json
evidence_coverage_index.json
evidence_coverage_summary.json
evidence_coverage_summary.md
canonical_advice_input_index.json
canonical_advice_output_index.json
advice_readiness_report.json
advice_readiness_report.md
technical_to_advice_transition.json
technical_ranking.json
full_advice_ranking.json
full_advice_ranking.md
top_full_advice_candidates.md
unable_to_advise.json
unable_to_advise.md
blocker_report.json
source_lineage.json
throughput_report.json
```

The implementation also writes `canonical_advice_inputs/` as the traceable dry-run payload source consumed by the deterministic advice engine.

## Coverage Result

```text
canonical_instruments: 952
attempted_instruments: 952
technical_analysed: 946
technical_ranking_eligible: 330
canonical_advice_input_ready: 4
advice_attempted: 952
advice_completed: 952
failed: 0
full_advice_ready: 0
full_advice_ranking_eligible: 0
```

Fundamental evidence:

```text
available: 4
partial: 17
missing: 931
stale: 0
invalid: 0
blocked: 0
```

Market context:

```text
available: 952
partial: 0
missing: 0
stale: 0
invalid: 0
blocked: 0
```

Portfolio context:

```text
available: 1
not_applicable: 951
missing: 0
stale: 0
invalid: 0
blocked: 0
```

## Advice Result

Canonical deterministic advice output:

```text
buy_candidate: 0
wait_for_price: 4
watchlist: 0
avoid_for_now: 0
hold_existing: 0
take_loss_review: 0
unable_to_advise: 948
partial_advice: 4
```

The four available-fundamental cases are `GM`, `PLD`, `TT`, and `WELL`. They reached canonical deterministic advice but remained partial:

```text
GM: wait_for_price, partial, blocker no_clear_setup
PLD: wait_for_price, partial, blocker no_clear_setup
TT: wait_for_price, partial, blocker price_or_risk_not_preferred
WELL: wait_for_price, partial, blocker price_or_risk_not_preferred
```

No instrument reached full-advice-ready status because the canonical advice engine did not return a ready actionable review output for any broad-universe ticker.

## Ranking Result

Technical ranking remains separate from full advice ranking.

Top technical candidates after non-price evidence attachment:

```text
1 ASB 75 unable_to_advise missing_fundamental_context
2 ASH 75 unable_to_advise missing_fundamental_context
3 ATR 75 unable_to_advise missing_fundamental_context
4 AXP 75 unable_to_advise missing_fundamental_context
5 BIO 75 unable_to_advise missing_fundamental_context
```

Full advice ranking:

```text
eligible_candidates: 0
```

This is the correct fail-closed result: missing or partial non-price evidence does not count as positive evidence.

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

There were no market-context blockers, no runtime failures, and no non-US instruments in the current canonical universe snapshot.

The only applicable local portfolio context was `COST`, which remained `unable_to_advise` because fundamental context was missing and the technical setup was weak or high risk.

ETF examples were processed and remained blocked by missing fundamental context, including `DIA`, `IWM`, `QQQ`, `SMH`, and `SOXX`.

## Throughput

```text
attempted: 952
total_runtime_seconds: 0.337035
tickers_per_second: 2824.634208
mean_ticker_runtime_seconds: 0.00007028
median_ticker_runtime_seconds: 0.00006604
p95_ticker_runtime_seconds: 0.00008317
```

## Determinism Check

A fixed subset run over `ASB`, `COST`, `GM`, `NVDA`, `PLD`, `TT`, and `WELL`
was executed twice. Stable fields matched across both runs: instrument id,
symbol, technical screening label, technical candidate score, canonical advice
label, advice readiness, full-advice-ready flag, missing evidence, blockers,
and ordering.

## Governance

The run manifest records:

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

Reason: ME-RUN31 proves the broad advice-readiness adapter and canonical advice handoff work, but full-advice readiness is blocked primarily by `missing_fundamental_context: 931` and `partial_fundamental_context: 17`. The next highest-value sprint should expand local fundamental evidence coverage without changing advice semantics or allocation authority.
