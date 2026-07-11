# ME-GH01 - GitHub-first No-API Baseline Guardrail

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: ACTIVE PROJECT GUARDRAIL

## Purpose

This guardrail locks the Market Engine baseline direction after the ME-CI11 provider-invocation attempts showed that OpenAI API usage is a paid optional path and not the user's intended primary architecture.

The primary Market Engine route is now GitHub-first and artifact-first:

```text
GitHub/local data acquisition
  -> deterministic Market Engine analysis
  -> batch artifacts, readiness states, rankings and review queues
  -> ChatGPT as human-in-the-loop interpretation layer over those artifacts
  -> user review and follow-up questions
```

The baseline route must not require paid OpenAI API calls, ChatGPT API calls, provider-generated advisory reports, or one model call per ticker.

## Hard project rule

Baseline Market Engine must not depend on OpenAI API calls.

The approved baseline architecture is:

* GitHub Actions, local Python jobs, or explicit operator commands acquire and prepare market/source data.
* GitHub/local runtime produces deterministic artifacts, readiness states, status indexes, rankings, review queues, and machine-readable summaries.
* ChatGPT is used interactively by the user to inspect, compare, interpret, and challenge those generated artifacts.
* Provider-based AI advisory generation is optional, deferred, and never required for the 500-ticker baseline flow.

## Prohibited default drift

Future sprints must not make any of the following the primary path unless the user explicitly re-approves paid provider usage:

* OpenAI API invocation fixes.
* API-key propagation fixes.
* Per-ticker model-generated advisory reports.
* Prompt tuning to make provider reports better.
* Model/schema hardening that is only useful for paid provider output.
* Batch runners that require model calls for every ticker.
* ChatGPT API integration as a mandatory engine dependency.

The project may still keep the ME-CI provider-advisory implementation as an optional/deferred path, but it must not block GitHub-first batch analysis work.

## Required sprint test

Before proposing or accepting any future sprint, the sprint must be checked against this question:

```text
Does this bring the project closer to GitHub-first 500-ticker analysis without mandatory paid OpenAI API usage?
```

If the answer is no, the sprint is not the next baseline sprint.

Exceptions require explicit user approval and must be recorded in the backlog and roadmap.

## Approved baseline output target

The next baseline product target is not a generated AI advisory report. The next baseline product target is a reproducible batch package such as:

```text
artifacts/market_engine/batch_runs/<run_id>/
  manifest.json
  ticker_status_index.json
  ticker_status_index.md
  rankings.json
  rankings.md
  failures.json
  data_freshness_summary.json
  tickers/
    <TICKER>/
      dry_run.json
      readiness.json
      analysis_summary.json
```

The first useful 500-ticker package should answer:

* Which tickers are ready for review?
* Which tickers are blocked by stale or missing data?
* Which tickers have usable deterministic setup/fundamental/derived observations?
* Which tickers should enter a human review queue?
* Which artifacts should ChatGPT inspect interactively?

## CI11 status

ME-CI11 through ME-CI11C remain useful implementation and blocker evidence for optional provider advisory output.

They are now explicitly deferred for the baseline route because:

* OpenAI API usage is paid separately from ChatGPT usage.
* The user does not want extra API costs for the baseline system.
* The intended architecture is GitHub/local artifact generation plus ChatGPT interpretation, not paid provider generation per ticker.

Provider advisory output may be revisited later only for selected top candidates, and only after explicit user approval.

## Next approved direction

The next approved baseline direction is:

```text
ME-GH01 - Lock GitHub-first no-API baseline and redirect roadmap
  -> ME-GH02 - Batch artifact discovery and ticker status index
  -> ME-GH03 - Deterministic ranking and review queue
  -> ME-GH04 - ChatGPT-readable batch analysis package
  -> ME-GH05 - GitHub Actions manual/scheduled batch run
  -> ME-GH06 - Scale test toward 100/500 tickers
```

## Enforcement

This guardrail is a project-level rule. It should be cited in future sprint prompts when scope could drift back toward paid provider output.

If backlog or roadmap entries conflict with this guardrail, the guardrail takes precedence until the backlog and roadmap are corrected.
