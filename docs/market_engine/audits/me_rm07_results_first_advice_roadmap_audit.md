# ME-RM07 - Results-First Advice Roadmap Audit

Status: COMPLETED ROADMAP LOCK AUDIT

## Objective

Respond to board-level product feedback that the Market Engine roadmap has drifted too often into infrastructure and preparation without producing visible investment advice results.

## Decision

The baseline roadmap is locked to a results-first sequence:

```text
ME-GH02 - Batch artifact discovery and ticker status index
  -> ME-ADV01 - Minimal deterministic advice engine v1
  -> ME-ADV02 - 500-ticker advice batch output
  -> ME-EVAL01 - Advice outcome tracking and feedback loop
  -> ME-APP01 - App/report view for advice candidates
```

## Superseded baseline detour

A standalone ME-GH03 ranking/review-queue sprint is not the next baseline step.

Review-priority logic may be implemented inside ME-ADV01 only when it directly supports advice label production.

## Rationale

The project already has enough artifact/status infrastructure to begin producing conservative deterministic advice labels.

The next product risk is not insufficient governance. The next product risk is failing to produce visible advice output quickly enough to evaluate whether the system works.

## Required advice output

ME-ADV01 must produce a deterministic advice label per ticker using available status/artifact evidence.

Allowed labels:

```text
buy_candidate
wait_for_price
watchlist
avoid_for_now
hold_existing
take_loss_review
unable_to_advise
```

## Guardrail retained

This roadmap lock does not authorize unsafe action side effects.

Still forbidden:

```text
broker/order execution
portfolio mutation
watchlist mutation
Telegram/delivery side effects
paid OpenAI API dependency
provider invocation as baseline requirement
invented evidence
missing data counted as positive evidence
```

## No-deviation rule

Every future baseline sprint must directly contribute to at least one of these outcomes:

```text
1. process more tickers;
2. make more tickers advice-ready;
3. produce advice labels;
4. evaluate advice quality.
```

## Files changed

```text
docs/market_engine/roadmap/me_rm07_results_first_advice_roadmap_lock.md
docs/market_engine/backlog/me_rm07_results_first_advice_backlog.md
docs/market_engine/audits/me_rm07_results_first_advice_roadmap_audit.md
docs/market_engine/roadmap/ACTIVE_BASELINE_DIRECTION.md
```

## Validation

Documentation-only roadmap lock.

No runtime code changes.

No provider/API code added.

No source acquisition added.

No broker/order/portfolio/watchlist/delivery side effects added.

## Outcome

`results_first_advice_roadmap_locked`

## Next sprint

```text
ME-ADV01 - Minimal deterministic advice engine v1
```
