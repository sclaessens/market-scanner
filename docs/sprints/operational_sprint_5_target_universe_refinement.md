# Operational Sprint 5 Target-Universe Refinement

## 1. Status and Scope

This document is a documentation-only governance refinement for Operational Sprint 5.

It refines the target-universe strategy for data coverage expansion after the diagnostics-only data coverage audit utility became available and after the operator clarified that current portfolio and watchlist artifacts should not be treated as the preferred first expansion target.

This document does not implement:

- code changes
- tests
- CSV files
- generated artifacts
- reports
- GitHub Actions workflows
- provider integration
- runtime orchestration
- Reporting changes
- Telegram changes
- Decision Engine changes
- scanner changes
- validation, context, timing, fundamental, or portfolio intelligence runtime changes

No sprint is closed or certified complete by this document.

This document records governance direction only. It does not authorize implementation.

## 2. Background

Operational Sprint 5 includes the Data Coverage Expansion Plan, which depends on reliable visibility into portfolio metadata coverage, fundamentals coverage, target-universe selection, missing-field coverage, and freshness diagnostics.

The diagnostics-only data coverage audit utility has now been implemented and merged. The utility supports coverage measurement across the following target modes:

- explicit ticker/date targets
- scanner-derived targets
- scanner A/B-grade targets
- portfolio-derived targets
- portfolio-watchlist targets
- full scanner targets

The utility remains diagnostics-only. It measures and summarizes coverage. It does not change data, call providers, write runtime artifacts, alter source contracts, or authorize Decision Engine behavior.

## 3. Current Finding

Portfolio metadata and fundamentals coverage can now be measured correctly.

The explicit six-ticker baseline for `C`, `GM`, `GS`, `PLD`, `TT`, and `WELL` on target date `2026-05-07` is useful for validating the utility because it confirms that ticker/date matching, portfolio metadata freshness diagnostics, fundamentals sufficiency diagnostics, and missing-field coverage metrics are being reported.

Current explicit six-ticker baseline:

- portfolio metadata coverage: `100.0%`
- metadata complete count: `6`
- metadata invalid count: `0`
- metadata last updated after target date count: `6`
- fundamentals coverage: `66.67%`
- fundamentals sufficient count: `4`
- fundamentals partial count: `2`
- missing `eps_growth_yoy`: `2`
- missing `operating_margin`: `2`
- date mismatch count: `0`

However, the current portfolio and watchlist should not be used as the next primary expansion target.

The operator clarified that:

- the portfolio has changed;
- the watchlist may no longer be up to date;
- portfolio and watchlist artifacts should be rebuilt later once the application produces better source-supported analysis;
- expanding coverage around stale portfolio or watchlist artifacts risks optimizing the data layer around outdated operator inputs.

The six-ticker explicit baseline remains valid as a utility validation baseline. It should not be treated as the preferred next operational expansion universe.

## 4. Refined Target-Universe Decision

The preferred next expansion target is scanner A/B-grade rows.

The secondary expansion target is the broader scanner output.

The deferred target is portfolio holdings and watchlist repair or rebuild.

Portfolio and watchlist artifacts should be rebuilt later, after scanner-based data coverage has improved and after the application can provide stronger source-supported analysis for portfolio and watchlist reconstruction.

This refinement does not change strategy authority. Scanner A/B-grade selection is coverage prioritization only.

Scanner A/B-grade coverage prioritization must not become:

- allocation priority
- tradeability status
- ranking authority
- eligibility filtering
- urgency classification
- conviction classification
- hidden filtering
- Decision Engine bypass

## 5. Governance Rationale

The refined target-universe strategy protects data stewardship discipline.

The project should avoid building coverage expansion around stale operator inputs. Outdated portfolio and watchlist artifacts should not become authoritative merely because they are measurable. Optimizing the data layer around stale holdings or an old watchlist could produce misleading completeness, misleading diagnostics, and pressure to repair downstream behavior before the source universe is trustworthy.

Scanner output is a more neutral current opportunity universe for the next data coverage expansion step. Scanner A/B-grade rows provide a practical first coverage slice because they are current scanner outputs and can expose missing metadata and fundamentals patterns without implying allocation authority.

This document preserves the certified doctrine:

- classification upstream
- allocation downstream
- Decision Engine is the only allocation authority
- upstream layers classify and enrich only
- reporting communicates only
- no hidden filtering
- no ranking authority outside approved Decision Engine logic
- no scoring authority outside approved Decision Engine logic

A/B scanner coverage is a coverage-prioritization mechanism only. It is not a portfolio decision, action signal, tradeability gate, ranking rule, or allocation input by itself.

## 6. Allowed Next Actions

The following actions are allowed as separate validation-only or planning-only work:

- run the data coverage audit utility in `scanner-ab` mode;
- run the data coverage audit utility in `scanner` mode;
- optionally run the data coverage audit utility in `full-scanner` mode;
- summarize missing portfolio metadata coverage patterns;
- summarize missing fundamentals coverage patterns;
- identify missing fields;
- identify potentially sector-sensitive missing fields;
- prepare source-data expansion templates;
- defer portfolio repair until stronger scanner-based coverage exists;
- defer watchlist repair until stronger scanner-based coverage exists.

These actions must remain diagnostics-only or documentation-only unless a separate governed implementation task explicitly authorizes changes.

## 7. Forbidden Scope

This refinement must not:

- change Decision Engine logic;
- change Reporting logic;
- change Telegram logic;
- change scanner logic;
- change validation runtime logic;
- change context runtime logic;
- change timing runtime logic;
- change fundamental runtime logic;
- change portfolio intelligence runtime logic;
- change provider-assisted prefill behavior;
- call provider APIs;
- add credentials or secrets;
- edit portfolio CSV files;
- edit watchlist CSV files;
- edit fundamentals CSV files;
- write generated processed artifacts;
- write runtime reports;
- introduce ranking semantics;
- introduce scoring semantics;
- introduce tradeability semantics;
- introduce urgency semantics;
- introduce conviction semantics;
- introduce allocation semantics;
- introduce eligibility semantics;
- introduce hidden filtering semantics.

## 8. Sector-Aware Fundamentals Note

Scanner-based expansion may reveal sector-sensitive fundamentals coverage issues.

For example, some missing or less comparable fields may matter differently across sectors, industries, capital structures, or asset types. A scanner-based target universe can expose those gaps more naturally than a stale portfolio or watchlist universe.

Sector-aware fundamentals strategy is a future design topic. This document does not implement sector-specific fields, change the current fundamentals contract, modify the Fundamental Layer, or authorize sector-specific Decision Engine logic.

## 9. Backlog Impact Assessment

Existing related backlog items are sufficient for this refinement.

Relevant existing backlog coverage includes:

- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- existing portfolio and watchlist repair or contract items in `docs/sprints/project_backlog.md`, including backlog coverage for authoritative active portfolio source repair and portfolio/watchlist-related future governance

This refinement narrows the next coverage-validation direction but does not create a new deferred implementation requirement beyond the existing backlog scope.

Backlog impact assessment:
- No new backlog items identified.

## 10. Recommended Next Step

Prepare a separate validation-only Codex prompt to run:

```bash
python scripts/core/audit_data_coverage.py --target-mode scanner-ab
```

Optionally follow with:

```bash
python scripts/core/audit_data_coverage.py --target-mode scanner
```

The validation-only prompt should require Codex to:

- summarize missing metadata patterns;
- summarize missing fundamentals patterns;
- identify missing fields;
- highlight sector-sensitive coverage issues if visible;
- avoid modifying data;
- avoid modifying runtime behavior;
- avoid writing generated artifacts unless the existing audit utility inherently produces diagnostics-only output under an explicitly approved path;
- avoid code changes;
- avoid tests unless explicitly needed for diagnostics validation.

The next step should remain validation-only. It should not repair portfolio data, rebuild the watchlist, call provider APIs, alter runtime orchestration, or loosen Decision Engine behavior.
