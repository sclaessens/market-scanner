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
  -> ME-DATA01 - Close highest-impact advice data coverage gaps
  -> ME-EVAL01 - Advice outcome tracking and feedback loop
  -> ME-APP01 - App/report view for advice candidates
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
that outcome tracking is not yet useful because only `watchlist` labels were
produced.

## Next baseline sprint

```text
ME-DATA01 - Close highest-impact advice data coverage gaps
```

Not:

```text
ME-GH03 - Deterministic ranking and review queue
```

ME-DATA01 must be narrowly targeted at the gaps proven by ME-ADV02:
setup/price/market context, portfolio context, and status/advice coverage
beyond the current 12-ticker sample. It must not become generic infrastructure
or a provider-refresh detour.

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
