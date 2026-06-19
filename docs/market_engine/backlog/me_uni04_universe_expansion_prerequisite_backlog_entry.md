# ME-UNI04 prerequisite - Expand canonical ticker universe before execution

Owner roles: Product Owner / Operator / Data Steward / Financial Analyst / Technical Architect / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: REQUIRED PREREQUISITE BEFORE ME-UNI04

## Purpose

Before ME-UNI04 is executed, the canonical ticker universe must be deliberately expanded beyond the initial ME-UNI03 starter universe.

ME-UNI03 intentionally created a small first canonical universe so the loader, CSV contract, validation rules and downstream cached-source RUN integration could be activated safely. That small universe is not the intended long-term Market Engine coverage set.

## Required backlog instruction

ME-UNI04 must not start as a narrow update that only edits one or two symbols unless explicitly re-scoped by the Product Owner.

Immediately before ME-UNI04 execution, the operator and product owner must review and expand the canonical ticker universe into a broader, structured universe.

The expanded universe should target a controlled but materially broader first operating set, for example approximately 50-100 tickers, unless a smaller or larger number is explicitly justified in the sprint scope.

## Required expansion categories

The ME-UNI04 preparation pass should consider at least these buckets:

* current portfolio holdings and ETF references;
* user-requested watchlist names;
* AI and semiconductors;
* datacenter, power and infrastructure names;
* cybersecurity;
* defence and aerospace;
* quality compounders;
* European opportunities;
* smaller or less obvious growth candidates;
* high-risk or governance-sensitive names that should remain `manual_review_only` or `blocked` when appropriate.

## Required governance rules

The expansion must preserve the ME-UNI01/ME-UNI02/ME-UNI03 boundaries:

* no provider refresh;
* no live market data calls;
* no SEC/EDGAR or yfinance calls unless a later sprint explicitly authorizes them;
* no Telegram delivery;
* no portfolio writes;
* no watchlist writes;
* no broker behavior;
* no Decision Engine action semantics;
* no BUY / SELL / HOLD labels;
* no allocation advice;
* no target prices;
* no position sizing;
* no ranking, scoring, urgency, conviction, tradeability or execution advice.

## Expected ME-UNI04 input

ME-UNI04 should start from:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

and expand that file in a controlled, reviewable way.

The sprint should document:

* which buckets were added;
* which tickers were kept active;
* which tickers were set to `manual_review_only`;
* which tickers were blocked, if any;
* why the final universe size is appropriate for the next RUN or Telegram-preview sequence.

## Relationship to ME-RUN16

ME-RUN16 may still proceed first if the goal is to prove that the initial ME-UNI03 universe can be consumed safely by cached-source batch execution.

This backlog entry specifically prevents ME-UNI04 from being treated as a tiny cosmetic CSV edit. ME-UNI04 should be the deliberate expansion sprint for the canonical operating universe.
