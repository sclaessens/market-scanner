# ME-RM04 — Fast full-output sprint sequence backlog entry

Owner roles: Product Owner / Scrum Master / Technical Architect / Data Steward / QA Lead / Governance Auditor

Job family: ME-RM — Roadmap / Governance

Status: ACTIVE PLANNING UPDATE

## Goal

Reorder the active Market Engine sprint direction toward the fastest safe path to complete, inspectable, local end-to-end analysis output while preserving existing planned sprints as valid deferred candidates.

This update is a planning/governance change only. It does not remove existing sprints, does not block deferred work, and does not authorize runtime/provider/Telegram/portfolio/broker side effects.

## Background

After ME-SA03, ME-SA04, and ME-SA05, `company_profile` snapshots can now move through:

```text
company_profile acquisition
  -> staging/import validation
  -> compatibility gate
  -> Source Context consumption
```

The project should now avoid an extended refinement loop before seeing complete practical output. The next goal is to reach a full local output path quickly enough to inspect whether the application is producing useful, understandable, and trustworthy analysis artifacts.

## Planning decision

The active sprint order is updated to prioritize a fast full-output path:

```text
1. ME-SA06 — Derive basic company_profile observations from Source Context
2. ME-RUN27 — Run NVDA/AMD/ASML through company_profile Source Context and Fundamental Observations with cross-ticker audit
3. ME-SA07 — Allow company_profile observations into Analysis Review as descriptive context only
4. ME-DL03 — Generate Telegram preview artifact from dry-run report without sending
5. ME-RUN28 — Run expanded supported-universe acquisition and dry-run classification
6. ME-QA01 — Human quality review of full analysis outputs
```

This sequence is intended to shorten the distance between source ingestion and real operator-visible output. It does not authorize investment actions, target prices, broker behavior, production delivery, or live Telegram sending.

## Ticker-agnostic implementation rule

From this planning update onward, implementation sprints in the `company_profile`, run, analysis, output, and delivery-preview path must remain ticker-agnostic.

Required guardrails:

* no hardcoded `NVDA`, `AMD`, `ASML`, or other ticker-specific runtime branches;
* no field mapping that only works for one known company profile;
* no provider-specific or exchange-specific workaround unless explicitly modeled as a contract rule;
* no US-only assumption unless the contract declares it and tests protect non-US or non-standard cases;
* tests should use multiple tickers or synthetic ticker fixtures where practical;
* NVDA/AMD/ASML may be used as bounded system-test tickers, but not as special-case implementation targets;
* ASML or an equivalent non-US/non-standard profile should be used as a governance case where available;
* failures must be classified per ticker instead of patched away for one known symbol.

## Revised near-term candidates

### ME-SA06 — Derive basic company_profile observations from Source Context

Status: NEXT ACTIVE CANDIDATE

Goal: Produce descriptive, non-advisory `company_profile` observations from consumed Source Context.

Additional RM04 guardrail: ME-SA06 must be ticker-agnostic and must not optimize for only NVDA or a narrow handpicked profile shape.

### ME-RUN27 — Run NVDA/AMD/ASML through company_profile Source Context and Fundamental Observations with cross-ticker audit

Status: CANDIDATE AFTER ME-SA06

Goal: Execute the current company-profile path across multiple bounded tickers and produce an audit that classifies per-ticker acquisition, validation, gate, Source Context, and Fundamental Observations outcomes.

Scope:

* bounded local run for `NVDA`, `AMD`, and `ASML` or equivalent available bounded tickers;
* cross-ticker classification, not ticker-specific remediation;
* local artifacts only;
* no live provider calls unless already explicitly allowed by an approved acquisition job mode;
* no Telegram send;
* no portfolio/watchlist writes;
* no broker actions;
* no recommendation, allocation, target price, ranking, urgency, conviction, or tradeability authority.

### ME-SA07 — Allow company_profile observations into Analysis Review as descriptive context only

Status: CANDIDATE AFTER ME-RUN27

Goal: Allow `company_profile` observations to be visible to Analysis Review as descriptive context only, without changing recommendation, setup, portfolio, or Decision Engine authority.

Scope must remain non-advisory. This sprint may not introduce buy/sell/hold, target prices, rankings, urgency, conviction, position sizing, allocation, or trade authority.

### ME-DL03 — Generate Telegram preview artifact from dry-run report without sending

Status: CANDIDATE AFTER ME-SA07 OR AFTER A SUCCESSFUL FULL-OUTPUT RUN

Goal: Generate a Telegram-format preview artifact from local dry-run reporting output without sending anything to Telegram.

Approved output mode:

```text
preview_only
send_enabled: false
telegram_side_effects: none
```

This sprint remains delivery-preview only and may not introduce production delivery, scheduler behavior, real Telegram API calls, or user-facing send behavior.

### ME-RUN28 — Run expanded supported-universe acquisition and dry-run classification

Status: CANDIDATE AFTER ME-DL03 OR EARLIER IF RUN27 EXPOSES UNIVERSE-SCALE BLOCKERS

Goal: Run the expanded supported universe through acquisition/import/dry-run classification and produce a per-ticker outcome report.

Required classifications should include, where applicable:

```text
acquired
missing
invalid
blocked
consumed
observation_ready
analysis_ready
dry_run_completed
dry_run_partial
```

The sprint must report why each ticker failed or progressed. It must not hardcode fixes for individual symbols.

### ME-QA01 — Human quality review of full analysis outputs

Status: CANDIDATE AFTER FIRST COMPLETE OR NEAR-COMPLETE FULL-OUTPUT RUN

Goal: Review actual output quality before further optimization. The review should answer whether the analysis is understandable, trustworthy, useful, and correctly bounded.

This is the preferred point to decide whether to prioritize output readability, source coverage, Analysis Review behavior, Telegram preview, portfolio-context persistence, Decision Engine handoff hardening, or further data-quality work.

## Deferred but preserved candidates

The following candidates remain valid. They are not rejected and not blocked; they are intentionally placed below the fast full-output path unless concrete run evidence proves they are blocking progress:

* ME-CANDIDATE03 — Candidate classification QA/review contract.
* ME-OUT03 — Operator report readability/polish improvements.
* ME-PR03 — Approved portfolio context source/persistence contract.
* ME-DE03 — Decision Engine handoff review hardening.
* Additional ME-QAxx / ME-GOVxx — only when concrete run evidence justifies them.
* Manual/operator-supplied source package diagnostics — fallback only, not the primary route.

## Non-goals

ME-RM04 does not authorize:

* runtime code changes;
* provider calls;
* live data retrieval;
* yfinance, SEC/EDGAR, or network access changes;
* Telegram sending;
* production writes;
* portfolio/watchlist writes;
* broker/order actions;
* scheduler behavior;
* UI behavior;
* buy/sell/hold recommendations;
* target prices;
* allocation or position sizing;
* ranking, urgency, conviction, scoring, or tradeability authority;
* Decision Engine authority changes.

## Acceptance criteria

* The active direction is documented as a fast full-output path.
* ME-SA06 remains the next active sprint.
* ME-RUN27 is explicitly placed immediately after ME-SA06.
* Telegram preview is preserved but placed after analysis/full-output progress, not as an immediate distraction.
* Expanded-universe execution remains a near-term goal.
* Existing planned sprints are preserved as deferred candidates rather than deleted or blocked.
* Ticker-agnostic implementation guardrails are explicit.
