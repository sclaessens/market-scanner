# ME-RM07 - Results-First Advice Roadmap Lock

Status: ACTIVE CANONICAL ROADMAP LOCK

Owner roles: Product Owner / Board-facing Project Manager / Technical Architect / Governance Auditor

Date: 2026-07-11

## Board feedback

The Market Engine project has spent too long producing infrastructure, contracts, guardrails, readiness layers, and safe blocked states without delivering enough visible investment output.

The active product objective is now explicit:

```text
Process a broad ticker universe, such as 500 tickers,
produce deterministic advice labels,
then evaluate which advice rules work and which do not.
```

The roadmap must no longer drift into additional standalone preparation, governance, provider, delivery, prompt, or reporting layers unless they are the shortest path to advice output.

## Product goal

The baseline product goal is:

```text
500 tickers
  -> data / artifacts
  -> ticker status
  -> deterministic advice labels
  -> advice batch output
  -> outcome evaluation
  -> app/report view
```

The first visible result must be an advice index, not another advisory scaffold.

## Canonical roadmap sequence

The active roadmap sequence is now:

```text
ME-GH02 - Batch artifact discovery and ticker status index
  -> ME-ADV01 - Minimal deterministic advice engine v1
  -> ME-ADV02 - 500-ticker advice batch output
  -> ME-EVAL01 - Advice outcome tracking and feedback loop
  -> ME-APP01 - App/report view for advice candidates
```

This supersedes any older baseline pointer that inserts a standalone ME-GH03 ranking/review-queue sprint before advice output.

## No-deviation rule

Every future baseline sprint must directly contribute to at least one of these outcomes:

```text
1. process more tickers;
2. make more tickers advice-ready;
3. produce advice labels;
4. evaluate advice quality.
```

If a proposed sprint does not directly contribute to one of those four outcomes, it is not part of the baseline roadmap.

Exceptions are allowed only when a concrete blocker makes ME-ADV01, ME-ADV02, ME-EVAL01, or ME-APP01 technically impossible. Any exception must document:

```text
- the blocker;
- why it prevents advice output;
- why the proposed sprint is the shortest path back to advice output;
- which canonical roadmap step resumes after the exception.
```

## Advice labels required from ME-ADV01

ME-ADV01 must introduce a minimal deterministic advice engine that produces one advice label per ticker.

Allowed v1 labels:

```text
buy_candidate
wait_for_price
watchlist
avoid_for_now
hold_existing
take_loss_review
unable_to_advise
```

Human-facing labels:

```text
Kopen mogelijk
Wachten op betere prijs
Opvolgen
Voorlopig vermijden
Bestaande positie houden
Verlies nemen bekijken
Geen advies mogelijk
```

## Minimal v1 rule posture

ME-ADV01 must be simple, deterministic, and evidence-bound. It should not wait for perfect data.

The engine may use conservative rules such as:

```text
invalid artifact -> unable_to_advise
stale data -> watchlist or unable_to_advise
missing fundamental data -> unable_to_advise
missing setup/price context -> watchlist, not buy_candidate
valid partial analysis with unresolved blockers -> watchlist or avoid_for_now
sufficient positive setup and fundamentals -> buy_candidate
existing position with high loss/risk -> take_loss_review
existing position with acceptable state -> hold_existing
```

These rules may be imperfect. That is acceptable because ME-EVAL01 exists to measure and improve them.

## ME-ADV01 required outputs

ME-ADV01 must produce:

```text
artifacts/market_engine/advice_runs/<run_id>/
  manifest.json
  advice_index.json
  advice_index.md
  unable_to_advise.json
  advice_summary.json
```

Each ticker row must include at minimum:

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

## Baseline guardrails retained

The results-first roadmap does not authorize unsafe side effects.

Still forbidden in baseline runtime:

```text
broker/order execution
portfolio mutation
watchlist mutation
Telegram/delivery side effects
paid OpenAI API dependency
provider invocation as a baseline requirement
invented evidence
missing data treated as positive evidence
```

The roadmap does authorize deterministic advice labels, because advice output is now the explicit product goal.

## Provider/ChatGPT positioning

ChatGPT remains the interactive interpretation layer over generated artifacts.

The baseline must not require OpenAI API usage. Provider-generated reports remain optional/deferred and must not block deterministic advice output.

## Explicitly de-scoped as standalone baseline sprints

The following are not allowed as standalone baseline sprints before ME-ADV01 unless a concrete blocker proves they are necessary:

```text
extra governance-only work
extra contract-only work
extra provider/API invocation work
extra prompt/model hardening
extra reporting polish
extra delivery/notification work
extra UI before advice output
standalone ranking/review queue that does not produce advice labels
```

Review-priority logic may be implemented inside ME-ADV01 if it directly supports advice label production, but it must not become a separate detour.

## Current next sprint

After ME-GH02 is merged, the next sprint is:

```text
ME-ADV01 - Minimal deterministic advice engine v1
```

Not:

```text
ME-GH03 - Deterministic ranking and review queue
```

## Definition of roadmap adherence

A sprint adheres to this roadmap only when it answers yes to this question:

```text
Does this sprint get us closer to advice labels for a broad ticker universe and later evaluation of those labels?
```

If the answer is no, the sprint must be rejected or rewritten.
