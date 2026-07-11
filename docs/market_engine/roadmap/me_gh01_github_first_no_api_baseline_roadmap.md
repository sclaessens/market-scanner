# ME-GH01 - GitHub-first No-API Baseline Roadmap Lock

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE ROADMAP OVERRIDE

## Purpose

This roadmap lock corrects the active Market Engine direction after the ME-CI11 provider-invocation sequence exposed an architectural mismatch with the user's intended product path.

The intended path has always been:

```text
GitHub retrieves or prepares data
  -> GitHub/local runtime runs deterministic analysis or prepares analysis-ready artifacts
  -> ChatGPT inspects and interprets those artifacts interactively
  -> the app/user uses the resulting status, rankings, blockers and review queues
```

The intended path is not:

```text
paid OpenAI API call per ticker
  -> AI-generated advisory report per ticker
  -> provider quota/billing/model debugging as a prerequisite for baseline analysis
```

## Active roadmap rule

This document overrides any older active-roadmap wording that treats ME-CI11D, provider invocation, or paid API-generated reports as the next baseline step.

The ME-CI provider advisory line is now optional/deferred for baseline planning.

The active baseline roadmap is GitHub-first and no-paid-API by default.

## Current baseline direction

```text
ME-GH01 - Lock GitHub-first no-API baseline and redirect roadmap
  -> ME-GH02 - Batch artifact discovery and ticker status index (completed)
  -> ME-GH03 - Deterministic ranking and review queue
  -> ME-GH04 - ChatGPT-readable batch analysis package
  -> ME-GH05 - GitHub Actions manual/scheduled batch run
  -> ME-GH06 - Scale test toward 100/500 tickers
```

## ME-GH01 - Lock GitHub-first no-API baseline and redirect roadmap

Status: CURRENT DOCS/GOVERNANCE SPRINT

Goal: prevent future sprint drift by documenting that the baseline system must produce GitHub/local deterministic batch artifacts and must not depend on paid OpenAI API provider calls.

Scope:

* Add a project guardrail for GitHub-first / no-paid-API baseline.
* Mark provider advisory output as optional/deferred.
* Set ME-GH02 as the next implementation sprint.
* Define batch-readiness targets for 50/100/500 tickers.
* Preserve CI11 artifacts as optional provider-output evidence, not as the baseline roadmap.

Non-scope:

* No runtime changes.
* No provider calls.
* No OpenAI API integration work.
* No model prompt/schema work.
* No source acquisition implementation.
* No broker, portfolio, watchlist or notification side effects.

## ME-GH02 - Batch artifact discovery and ticker status index

Status: COMPLETED BASELINE IMPLEMENTATION SPRINT

Goal: create a deterministic batch index over existing Market Engine artifacts so the system can summarize ticker readiness, blockers, freshness and available analysis context without requiring paid model calls.

Implemented output:

```text
artifacts/market_engine/batch_status/<run_id>/
  manifest.json
  ticker_status_index.json
  ticker_status_index.md
  discovery_summary.json
  failures.json
```

Required per-ticker fields:

* ticker
* source artifact path
* artifact type / schema version
* data freshness state
* readiness state
* blocked reasons
* missing data summary
* analysis families present
* setup/fundamental/derived observation availability
* recommendation-review boundary state
* portfolio context availability where already present
* status: `review_ready`, `descriptive_only`, `blocked`, `stale`, `invalid_artifact`, or `missing_artifact`

Non-scope:

* No AI/provider calls.
* No generated advisory reports.
* No ticker-specific special cases.
* No scoring semantics that create BUY/SELL/HOLD or allocation authority.

Sample evidence:

```text
artifacts/market_engine/batch_status/me-gh02-sample-status-index-20260711T120000Z/
```

The sample run indexed 12 valid ME-RUN28 dry-run artifacts. All 12 tickers were preserved as `blocked` with `partial_analysis` readiness because the source artifacts preserve upstream blocked portfolio-review state. Provider advisory output remains optional/deferred and is not part of the baseline.

## ME-GH03 - Deterministic ranking and review queue

Status: NEXT BASELINE IMPLEMENTATION SPRINT

Goal: produce deterministic non-actionable review queues from batch status data.

The output may rank review priority, not investment actionability. Ranking must be based on transparent deterministic criteria such as data completeness, freshness, analysis availability, setup context presence and blocker severity.

Required guardrail:

```text
review priority != recommendation
review queue != buy list
ranking != allocation authority
```

## ME-GH04 - ChatGPT-readable batch analysis package

Status: PLANNED AFTER ME-GH03

Goal: produce compact markdown and JSON artifacts that ChatGPT can inspect interactively.

Expected questions supported:

* Which tickers are ready for review?
* Which tickers are blocked and why?
* Which have stale data?
* Which have the richest available analysis context?
* Which 20 tickers should be reviewed first?
* Which tickers are relevant to the user's portfolio context if approved context exists?

ChatGPT remains an interpretation layer. It must not invent missing data, override blockers, or create recommendation/allocation authority.

## ME-GH05 - GitHub Actions manual/scheduled batch run

Status: PLANNED AFTER ME-GH04

Goal: allow GitHub to run the deterministic batch pipeline manually or on a schedule and persist artifacts for review.

Required constraints:

* No OpenAI API secret required.
* No provider advisory model calls.
* No broker or portfolio mutation.
* No Telegram or notification side effects.
* Output artifacts are retained in a predictable path or uploaded as workflow artifacts.

## ME-GH06 - Scale test toward 100/500 tickers

Status: PLANNED AFTER ME-GH05

Goal: validate runtime, artifact size, failure summaries and review queues at larger ticker counts.

Success criteria:

* Works on a materially larger universe than the current sample set.
* Produces a machine-readable index.
* Produces a human-readable summary.
* Separates runtime failures from source/data blockers.
* Provides enough structure for ChatGPT to review the batch result interactively.

## Deferred optional provider line

The following line is explicitly deferred for baseline planning:

```text
ME-CI11D or later - paid provider advisory output continuation
```

It may only resume when the user explicitly approves paid API usage. It must never block ME-GH02 through ME-GH06.

## Roadmap adherence rule

Future ChatGPT/Codex sprint prompts must cite this roadmap lock when selecting next work.

Any proposed sprint that does not advance the GitHub-first 500-ticker baseline must explain why it is allowed before it can interrupt the sequence.
