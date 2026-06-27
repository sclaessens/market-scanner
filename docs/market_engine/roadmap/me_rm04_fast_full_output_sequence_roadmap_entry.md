# ME-RM04 — Fast full-output sprint sequence roadmap entry

Owner roles: Product Owner / Scrum Master / Technical Architect / Data Steward / QA Lead / Governance Auditor

Job family: ME-RM — Roadmap / Governance

Status: ACTIVE PLANNING UPDATE

## Purpose

This roadmap entry updates the active Market Engine direction after ME-SA05. The goal is to move quickly from safe `company_profile` Source Context consumption toward complete local output that can be inspected by the operator.

This entry preserves existing planned sprints as deferred candidates. It does not delete, reject, or block them.

## Current completed basis

```text
ME-SA03  company_profile compatibility contract                   completed
ME-SA04  company_profile compatibility gate                       completed
ME-SA05  company_profile Source Context consumption               completed
```

Current company-profile path:

```text
company_profile acquisition
  -> staging/import validation
  -> compatibility gate
  -> Source Context consumption
```

## Roadmap correction

The project should now avoid spending too many sprints on refinement before seeing complete output. The active roadmap is corrected toward the fastest safe route to a full local analysis/reporting artifact.

Updated near-term sequence:

```text
ME-SA06
  -> ME-RUN27
  -> ME-SA07
  -> ME-DL03
  -> ME-RUN28
  -> ME-QA01
```

## Sprint sequence

### 1. ME-SA06 — Derive basic company_profile observations from Source Context

Next active sprint.

Purpose: transform consumed `company_profile` Source Context into descriptive, non-advisory Fundamental Observations.

Boundary: no recommendations, target prices, ranking, urgency, conviction, setup/trade-plan impact, portfolio impact, or Decision Engine authority.

Ticker-agnostic rule: implementation and tests must not be optimized only for NVDA. Multiple tickers or synthetic profile variants should be used where practical.

### 2. ME-RUN27 — Run NVDA/AMD/ASML through company_profile Source Context and Fundamental Observations with cross-ticker audit

Immediate run sprint after ME-SA06.

Purpose: prove the company-profile Source Context and observation path across multiple tickers and produce per-ticker evidence.

This is a system test, not a ticker-specific optimization sprint. NVDA, AMD, and ASML are bounded validation tickers only.

Expected outcome categories:

```text
acquired
validated
blocked
consumed
observation_ready
dry_run_partial
dry_run_completed
```

### 3. ME-SA07 — Allow company_profile observations into Analysis Review as descriptive context only

Purpose: make descriptive profile observations visible to Analysis Review without introducing investment authority.

Boundary: Analysis Review may use company identity/profile context to explain what source data says, but may not infer recommendation, valuation, price target, ranking, conviction, urgency, tradeability, or allocation from it.

### 4. ME-DL03 — Generate Telegram preview artifact from dry-run report without sending

Purpose: create an operator-visible Telegram-format preview artifact from the dry-run report so output readability can be inspected early.

Approved mode:

```text
preview_only
send_enabled: false
telegram_side_effects: none
```

Boundary: no real Telegram API call, no scheduler, no production delivery, no external send behavior.

### 5. ME-RUN28 — Run expanded supported-universe acquisition and dry-run classification

Purpose: scale the path beyond a small bounded validation set and classify outcomes per ticker.

This sprint should expose whether failures are caused by acquisition, staging validation, compatibility gating, Source Context consumption, Fundamental Observations, Analysis Review, or reporting.

It must not introduce ticker-specific fixes. Any repeated failure class should become a general contract/code/data-quality follow-up.

### 6. ME-QA01 — Human quality review of full analysis outputs

Purpose: inspect real full-output artifacts before further optimization.

The review should decide whether the next highest-value work is source coverage, Analysis Review semantics, output readability, Telegram preview refinement, portfolio-context persistence, Decision Engine handoff hardening, or broader universe support.

## Preserved deferred candidates

The following remain valid roadmap candidates, but are intentionally below the fast full-output path unless concrete run evidence shows they are blocking progress:

* ME-CANDIDATE03 — Candidate classification QA/review contract.
* ME-OUT03 — Operator report readability/polish improvements.
* ME-PR03 — Approved portfolio context source/persistence contract.
* ME-DE03 — Decision Engine handoff review hardening.
* Additional ME-QAxx / ME-GOVxx — concrete-evidence driven only.
* Manual/operator-supplied source package work — fallback/manual diagnostic only.

## Universal ticker rule

Future sprint prompts in this chain must include the following guardrail:

```text
The implementation must be ticker-agnostic. Do not hardcode NVDA, AMD, ASML, or any known symbol into runtime behavior. Use bounded tickers only as validation fixtures. Failures must be classified by general cause and resolved through general contracts or code paths, not symbol-specific patches.
```

## Non-goals

This roadmap update does not authorize:

* runtime changes;
* provider calls;
* live data retrieval;
* Telegram sending;
* production writes;
* portfolio/watchlist writes;
* broker or order behavior;
* target prices;
* recommendations;
* allocation or position sizing;
* ranking, urgency, conviction, scoring, or tradeability authority;
* Decision Engine authority changes.

## Expected benefit

This sequence should reduce the risk of building clean contracts without knowing whether the complete operator-facing output is useful. It prioritizes complete local evidence, cross-ticker behavior, and human output inspection before further optimization.
