# ME-RM07 - Results-First Advice Backlog Entry

Status: ACTIVE BASELINE BACKLOG LOCK

## Objective

Lock the Market Engine backlog to a results-first investment advice path.

The project goal is no longer merely to prove safe artifact handling. The project must quickly produce deterministic advice labels across a broad ticker universe and then evaluate those labels.

## Active baseline sequence

```text
ME-GH02 - Batch artifact discovery and ticker status index
  -> ME-ADV01 - Minimal deterministic advice engine v1
  -> ME-ADV02 - 500-ticker advice batch output
  -> ME-EVAL01 - Advice outcome tracking and feedback loop
  -> ME-APP01 - App/report view for advice candidates
```

## Backlog rule

Every proposed baseline sprint must directly support at least one of these four outcomes:

```text
1. process more tickers;
2. make more tickers advice-ready;
3. produce advice labels;
4. evaluate advice quality.
```

A sprint that does not support one of these outcomes must be rejected, unless it documents a concrete blocker that prevents the active advice path.

## ME-ADV01 backlog definition

### ME-ADV01 - Minimal deterministic advice engine v1

Owner roles: Product Owner / Financial Analyst / Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: NEXT BASELINE SPRINT AFTER ME-GH02

Goal: convert `ticker_status_index.json` plus referenced Market Engine artifacts into a first deterministic advice proposal per ticker.

Required advice labels:

```text
buy_candidate
wait_for_price
watchlist
avoid_for_now
hold_existing
take_loss_review
unable_to_advise
```

Required outputs:

```text
artifacts/market_engine/advice_runs/<run_id>/
  manifest.json
  advice_index.json
  advice_index.md
  unable_to_advise.json
  advice_summary.json
```

Required ticker fields:

```text
ticker
advice
confidence
primary_reasons
missing_for_better_advice
next_action
source_status
artifact_path
```

Hard constraints:

```text
- no paid OpenAI API dependency;
- no provider invocation;
- no broker/order execution;
- no portfolio/watchlist mutation;
- no Telegram/delivery side effects;
- no invented evidence;
- missing data must not count as positive evidence.
```

## ME-ADV02 backlog definition

### ME-ADV02 - 500-ticker advice batch output

Status: NEXT AFTER ME-ADV01

Goal: run the deterministic advice engine over the broad target ticker universe and produce an advice batch output.

Expected outputs:

```text
advice_index.json
advice_index.md
top_buy_candidates.md
watchlist.md
unable_to_advise.md
missing_data_report.md
```

This sprint must prioritize producing visible advice output, even if many tickers remain `unable_to_advise`.

## ME-EVAL01 backlog definition

### ME-EVAL01 - Advice outcome tracking and feedback loop

Status: NEXT AFTER ME-ADV02

Goal: compare advice labels against later market outcomes and create an evidence loop for improving deterministic advice rules.

Questions to answer:

```text
Did buy_candidate outperform watchlist after 1 week / 1 month / 3 months?
Which blockers predicted poor outcomes?
Which rules are too strict?
Which rules are too loose?
Which missing data fields are most damaging?
```

## ME-APP01 backlog definition

### ME-APP01 - App/report view for advice candidates

Status: NEXT AFTER ME-EVAL01 OR EARLIER ONLY IF ADVICE OUTPUT EXISTS

Goal: expose advice candidates, waitlist, avoid, and unable-to-advise groups in a user-facing app/report view.

The app view must not precede advice output.

## Deferred unless directly blocking advice output

The following are deferred and must not be inserted before ME-ADV01 unless proven necessary:

```text
standalone ranking/review queue
additional contract-only work
additional governance-only work
provider invocation fixes
model/prompt hardening
notification/delivery adapters
UI polish without advice output
```
