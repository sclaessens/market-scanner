# Operational Sprint 5 Manual Intake Pilot

## 1. Status and Scope

This document defines a limited manual source-supported intake pilot setup for Operational Sprint 5.

This task creates pilot intake files only.

No factual metadata values were populated.

No factual fundamentals values were populated.

No source URLs, source references, freshness dates, business values, or financial values were populated.

No sprint is closed or certified complete by this document.

This document and the pilot files do not implement:

- code changes
- tests
- generated reports
- GitHub Actions workflows
- provider/API integration
- credentials or secrets
- runtime orchestration
- Decision Engine changes
- Reporting changes
- Telegram changes
- scanner changes
- validation, context, timing, fundamental, or portfolio intelligence runtime changes
- portfolio repair
- watchlist repair
- fundamentals source repair

## 2. Background

This pilot follows:

- `docs/sprints/operational_sprint_5_target_universe_refinement.md`
- `docs/sprints/operational_sprint_5_scanner_coverage_audit_followup.md`
- `docs/sprints/operational_sprint_5_source_data_expansion_plan.md`
- `docs/sprints/operational_sprint_5_source_data_intake_worklist.md`

Operational Sprint 5 established scanner A/B-grade rows as the preferred next coverage target.

Scanner A/B selection is coverage prioritization only. It is not ranking authority, allocation authority, tradeability status, eligibility filtering, urgency, conviction, or a Decision Engine bypass.

Portfolio and watchlist repair or rebuild remain deferred.

## 3. Pilot Objective

The pilot exists to test the manual intake structure on `10` tickers before any larger source-supported collection effort.

This pilot does not authorize full `145`-row completion.

It also does not authorize provider/API calls, runtime changes, existing source CSV edits, or data population beyond a separately approved manual data-steward task.

## 4. Pilot Subset Selection

The pilot subset copies the first `10` rows from the scanner A/B intake templates.

The selected tickers are:

- `AAPL`
- `ABBV`
- `ADBE`
- `ADI`
- `ADP`
- `ADSK`
- `AIG`
- `AKAM`
- `ALL`
- `AMAT`

The pilot target date is `2026-05-19`.

This ordering is operational only. It does not imply ranking, priority, conviction, tradeability, urgency, allocation, or eligibility.

## 5. Pilot File Descriptions

The metadata pilot file is:

```text
data/intake/os5_scanner_ab_metadata_intake_pilot.csv
```

It contains `10` rows.

The populated fields are:

- `ticker`
- `target_date`
- `metadata_collection_status`

`metadata_collection_status` is set to `SOURCE_REQUIRED`.

All factual metadata fields remain empty.

The fundamentals pilot file is:

```text
data/intake/os5_scanner_ab_fundamentals_intake_pilot.csv
```

It contains `10` rows.

The populated fields are:

- `ticker`
- `target_date`
- `fundamentals_collection_status`

`fundamentals_collection_status` is set to `SOURCE_REQUIRED`.

All factual fundamentals fields remain empty.

## 6. Manual Collection Rules for a Future Data Steward

Every future populated value must have source support.

Every future populated value must have a source reference or URL.

Every future populated value must have a freshness or last-updated date.

Unverified fields must remain empty.

Unavailable fields may be documented with a note only after manual source review.

Guessed, inferred, approximate, convenience, scraped, synthesized, or unsourced values are not allowed.

Provider/API calls require separate authorization.

No credentials or secrets may be added to the repository, pilot files, source artifacts, generated outputs, reports, logs, or documentation.

## 7. Forbidden Scope

This pilot must not:

- change Decision Engine logic;
- change Reporting logic;
- change Telegram logic;
- change scanner logic;
- change validation runtime logic;
- change context runtime logic;
- change timing runtime logic;
- change fundamental runtime logic;
- change portfolio intelligence runtime logic;
- edit existing portfolio files;
- edit existing watchlist files;
- edit existing fundamentals source files;
- write generated processed artifacts;
- call provider APIs;
- add credentials or secrets;
- introduce ranking;
- introduce scoring;
- introduce tradeability;
- introduce urgency;
- introduce conviction;
- introduce allocation;
- introduce eligibility;
- introduce hidden filtering;
- bypass the Decision Engine.

## 8. Backlog Impact Assessment

Existing backlog items are sufficient for this pilot.

Relevant existing backlog coverage includes:

- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- existing portfolio and watchlist-related backlog items in `docs/sprints/project_backlog.md`

Backlog impact assessment:
- No new backlog items identified.

## 9. Recommended Next Step

Perform a separate human data-steward review step.

That future step may manually collect source-supported values for the `10` pilot tickers only.

The future step must:

- avoid provider/API calls unless separately authorized;
- avoid runtime changes;
- avoid existing source CSV changes unless separately authorized;
- avoid Decision Engine changes;
- avoid Reporting changes;
- avoid Telegram changes;
- avoid scanner changes;
- preserve source support and freshness metadata for every populated value.
