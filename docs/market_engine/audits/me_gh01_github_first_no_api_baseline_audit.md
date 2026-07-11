# ME-GH01 - GitHub-first No-API Baseline Audit

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: COMPLETED DOCS/GOVERNANCE AUDIT

## Objective

Record the explicit product-direction correction from provider-based advisory output toward GitHub-first deterministic batch analysis without mandatory paid OpenAI API usage.

## Trigger

During ME-CI11 through ME-CI11C, the project attempted to move from prepared Market Engine artifacts into real provider-generated advisory output. The user clarified that the intended architecture is not a paid OpenAI API call per ticker. The intended architecture is GitHub/local data acquisition, deterministic artifact preparation and analysis, and ChatGPT as an interactive human-in-the-loop interpretation layer over those artifacts.

## Findings

### Current central backlog state before ME-GH01

The central backlog still described the ME-CI advisory chain as continuing from ME-CI11 to ME-CI11B and ME-CI11C, with ME-CI11C positioned as the next advisory sprint once credentials were available.

This created drift risk because future sprint planning could continue provider-invocation work even though the user does not want extra OpenAI API costs for the baseline system.

### Current central roadmap state before ME-GH01

The central roadmap still contained a repositioned sequence where ME-CI11C was followed by ME-CI12 batch grounded advisory runner for ticker universe.

That sequence implied provider-generated advisory output remained the route to batch processing. This conflicts with the clarified baseline direction.

## Corrective action

ME-GH01 adds three docs-only lock documents:

* `docs/market_engine/governance/me_gh01_github_first_no_api_baseline_guardrail.md`
* `docs/market_engine/roadmap/me_gh01_github_first_no_api_baseline_roadmap.md`
* `docs/market_engine/backlog/me_gh01_github_first_no_api_baseline_backlog.md`

Together they establish:

```text
GitHub-first deterministic batch analysis is the baseline path.
Paid OpenAI API provider advisory output is optional/deferred.
ME-GH02 is the next implementation sprint.
```

## New baseline guardrail

```text
Baseline Market Engine must not depend on paid OpenAI API calls.
```

Future sprints must be tested against:

```text
Does this bring the project closer to GitHub-first 500-ticker analysis without mandatory paid OpenAI API usage?
```

If the answer is no, the sprint is not the next baseline sprint unless the user explicitly approves an exception.

## New baseline roadmap

```text
ME-GH01 - Lock GitHub-first no-API baseline and redirect roadmap
  -> ME-GH02 - Batch artifact discovery and ticker status index
  -> ME-GH03 - Deterministic ranking and review queue
  -> ME-GH04 - ChatGPT-readable batch analysis package
  -> ME-GH05 - GitHub Actions manual/scheduled batch run
  -> ME-GH06 - Scale test toward 100/500 tickers
```

## CI11 disposition

ME-CI11 through ME-CI11C are not deleted and are not treated as failed work. They remain useful optional-provider evidence and implementation scaffolding.

Their baseline status is now:

```text
optional / deferred
not required for baseline 500-ticker analysis
not allowed to block ME-GH02 through ME-GH06
```

## Scope control

This sprint is docs/governance only.

No runtime code changed.
No provider call was executed.
No OpenAI API key work was added.
No source acquisition runtime was changed.
No Decision Engine, portfolio, watchlist, broker, delivery or notification side effect was added.

## Remaining follow-up

The central long-form `market_engine_backlog.md` and `market_engine_roadmap.md` still contain older ME-CI11C/ME-CI12 sequence text in historical sections. The new ME-GH01 guardrail and roadmap lock take precedence immediately.

A later docs-cleanup sprint may inline this lock into the long-form central files if desired, but implementation work must not wait for that cleanup.

## Recommended next sprint

```text
ME-GH02 - Batch artifact discovery and ticker status index
```

ME-GH02 should build or specify the deterministic index over existing artifacts without provider calls and without paid API usage.
