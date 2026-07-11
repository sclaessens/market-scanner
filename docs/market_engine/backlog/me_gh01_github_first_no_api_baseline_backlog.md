# ME-GH01 - GitHub-first No-API Baseline Backlog Entry

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE / CURRENT

## Goal

Lock the Market Engine baseline direction to GitHub-first, deterministic, no-paid-OpenAI-API batch analysis so the project does not drift back into provider-invocation work after a few sprints.

## Problem statement

The ME-CI11 provider advisory line reached the point where a real OpenAI API invocation required paid API billing/quota and produced a 429 provider failure in local testing. The user clarified that the intended architecture has always been GitHub/local data acquisition and analysis preparation, with ChatGPT used interactively over artifacts, not paid OpenAI API calls as a baseline dependency.

The backlog and roadmap must therefore prevent future drift toward:

* paid provider invocation as the next baseline step;
* one model call per ticker;
* provider debugging as a blocker for 500-ticker analysis;
* ChatGPT API integration as a mandatory baseline engine component.

## Scope

* Add a GitHub-first / no-paid-API baseline guardrail.
* Add a roadmap lock that supersedes conflicting CI11-provider-next wording.
* Preserve CI11 as optional/deferred provider-output evidence.
* Define ME-GH02 as the next baseline implementation sprint.
* Define the ME-GH02 through ME-GH06 direction toward 500-ticker analysis.

## Non-scope

* No runtime code changes.
* No provider calls.
* No API-key work.
* No OpenAI API integration work.
* No prompt/schema/model work.
* No broker, portfolio, watchlist, delivery or notification side effects.
* No recommendation/allocation semantics.

## Backlog guardrail

The baseline Market Engine backlog must follow this rule:

```text
Baseline Market Engine must not depend on paid OpenAI API calls.
```

The baseline backlog sequence is now:

```text
ME-GH01 - Lock GitHub-first no-API baseline and redirect roadmap
  -> ME-GH02 - Batch artifact discovery and ticker status index
  -> ME-GH03 - Deterministic ranking and review queue
  -> ME-GH04 - ChatGPT-readable batch analysis package
  -> ME-GH05 - GitHub Actions manual/scheduled batch run
  -> ME-GH06 - Scale test toward 100/500 tickers
```

## Acceptance criteria

* A governance guardrail document exists.
* A roadmap lock document exists.
* This backlog entry exists.
* CI11 provider-advisory continuation is explicitly optional/deferred for baseline work.
* ME-GH02 is clearly the next implementation sprint.
* Future sprint prompts can cite this entry to prevent roadmap drift.

## Next sprint

```text
ME-GH02 - Batch artifact discovery and ticker status index
```

ME-GH02 must discover existing Market Engine artifacts and produce a deterministic batch status index without provider calls, without ChatGPT API calls, and without ticker-specific special cases.
