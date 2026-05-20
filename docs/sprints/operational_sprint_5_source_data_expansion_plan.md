# Operational Sprint 5 Source-data Expansion Plan

## 1. Status and Scope

This document is a documentation-only planning note for Operational Sprint 5.

It prepares a governed source-data expansion approach for the scanner A/B target universe identified by the Operational Sprint 5 target-universe refinement and scanner coverage audit follow-up.

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
- scanner changes
- Decision Engine changes
- validation, context, timing, fundamental, or portfolio intelligence runtime changes

No sprint is closed or certified complete by this document.

This document records planning direction only. It does not authorize implementation, data edits, provider/API calls, or runtime behavior changes.

## 2. Background

`docs/sprints/operational_sprint_5_target_universe_refinement.md` established scanner A/B-grade rows as the preferred next data coverage expansion target.

`docs/sprints/operational_sprint_5_scanner_coverage_audit_followup.md` recorded the validation-only scanner coverage audit findings:

- scanner A/B target universe: `145` tickers / `145` ticker-date rows;
- scanner A/B portfolio metadata coverage: `2.07%`;
- scanner A/B fundamentals coverage: `1.38%`;
- scanner-wide target universe: `291` tickers / `291` ticker-date rows;
- scanner-wide portfolio metadata coverage: `2.06%`;
- scanner-wide fundamentals coverage: `1.37%`.

The current interpretation is that the issue is source-data coverage, not runtime behavior.

Scanner A/B rows are now the preferred governed data completion target.

Scanner A/B selection is coverage prioritization only. It is not allocation authority, ranking authority, tradeability status, eligibility filtering, urgency, conviction, or a Decision Engine bypass.

## 3. Planning Objective

The objective is to prepare a governed data-completion approach for the `145` scanner A/B rows.

The plan should prepare portfolio metadata and fundamentals source expansion without editing data yet.

All future data entry must preserve:

- source provenance;
- source freshness metadata;
- deterministic ticker identity;
- traceability to the target universe;
- separation from Decision Engine authority.

This planning note does not authorize filling source CSV files, committing generated artifacts, calling providers, or changing runtime logic.

## 4. Target Universe

Primary target:

- `145` scanner A/B rows from the `scanner-ab` audit.

Secondary target:

- broader `291`-row scanner universe after A/B coverage improves.

Deferred target:

- portfolio repair or rebuild;
- watchlist repair or rebuild.

Portfolio and watchlist repair remain deferred because current scanner-based coverage should improve before stale operator-maintained portfolio or watchlist artifacts become expansion targets.

## 5. Required Portfolio Metadata Planning Fields

Future source-data expansion should prepare the following descriptive portfolio metadata fields:

- `ticker`
- company name
- `sector`
- `industry`
- `asset_class`
- `currency`
- `metadata_source`
- metadata source URL or reference
- `metadata_last_updated`
- data steward notes, if needed

Current governed artifact fields should remain aligned with `data/portfolio/portfolio_metadata.csv` unless a separate governed contract change authorizes additional runtime fields.

Company name, source URL or reference, and data steward notes are planning fields unless separately approved for a runtime source-artifact schema.

All portfolio metadata fields are descriptive only. They must not create allocation, ranking, scoring, tradeability, urgency, conviction, eligibility, or hidden filtering semantics.

## 6. Required Fundamentals Planning Fields

Future fundamentals source expansion should prepare the following source-supported fields:

- `ticker`
- `as_of_date`
- `report_period`
- revenue growth
- EPS growth
- gross margin
- operating margin
- debt or leverage field currently required by the contract
- free cash flow or cash-flow field currently required by the contract
- `currency`
- `source_name`
- fundamentals source URL or reference
- `source_last_updated`
- data quality notes, if needed

Current governed artifact fields should remain aligned with `data/raw/fundamentals.csv` unless a separate governed contract change authorizes additional runtime fields.

Source URL or reference and data quality notes are planning fields unless separately approved for a runtime source-artifact schema.

This document does not invent new runtime contract fields. Any additional runtime fields must remain future design candidates until separately governed.

## 7. Source Provenance and Freshness Rules

Every future data row must have source support.

Every future data row must have a freshness date.

Manual entries must remain traceable to a source document, provider export, filing, or other operator-approved reference.

Provider-assisted entries require separate authorization.

No provider/API calls are authorized by this planning note.

No credentials or secrets may be added to the repository, generated artifacts, reports, logs, source CSV files, or documentation.

Source provenance and freshness metadata are diagnostic and descriptive. They do not create allocation authority or Decision Engine bypass.

## 8. Governance Constraints

Future source-data expansion must not:

- change Decision Engine logic;
- change Reporting logic;
- change Telegram logic;
- change scanner logic;
- change validation runtime logic;
- change context runtime logic;
- change timing runtime logic;
- change fundamental runtime logic;
- change portfolio intelligence runtime logic;
- introduce ranking;
- introduce scoring;
- introduce tradeability;
- introduce urgency;
- introduce conviction;
- introduce allocation;
- introduce eligibility;
- introduce hidden filtering;
- bypass the Decision Engine.

The Decision Engine remains the only allocation authority.

Reporting and Telegram remain communication-only.

Scanner A/B selection remains a data coverage planning target only.

## 9. Data Completion Workflow Proposal

The future governed workflow should be:

1. Export or identify the `145` scanner A/B tickers from the diagnostics target universe.
2. Prepare a portfolio metadata template aligned with approved source-artifact contracts.
3. Prepare a fundamentals template aligned with approved source-artifact contracts.
4. Populate source-supported values in a separate authorized data-preparation step.
5. Preserve source provenance and freshness metadata for every row.
6. Validate coverage with:

```bash
.venv/bin/python scripts/diagnostics/audit_data_coverage.py --target-mode scanner-ab
```

7. Review missing, partial, stale, invalid, and date-mismatch diagnostics.
8. Only after scanner A/B coverage improves, consider broader scanner coverage.
9. Defer portfolio and watchlist rebuild until scanner-based coverage improves.

This workflow proposal does not implement templates, edit CSV files, call providers, change runtime behavior, or authorize pipeline changes.

## 10. Sector-aware Fundamentals Note

Sector-aware fundamentals may become relevant after more source coverage exists.

For example, some fields may be more meaningful or less comparable across sectors, industries, capital structures, asset classes, or reporting models.

This document does not implement sector-specific fields or sector-specific logic.

Sector-aware fundamentals strategy remains a future design topic and must not be routed into Decision Engine, Reporting, Telegram, or scanner behavior without separate governance.

## 11. Backlog Impact Assessment

Existing backlog items are sufficient for this planning note.

Relevant existing backlog coverage includes:

- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- existing portfolio and watchlist-related backlog items in `docs/sprints/project_backlog.md`

Backlog impact assessment:
- No new backlog items identified.

## 12. Recommended Next Step

Prepare a separate implementation-authorized data-preparation prompt only after this plan is reviewed.

That future prompt may prepare source-data expansion templates for the `145` scanner A/B rows, but it must still avoid runtime changes, Decision Engine changes, Reporting changes, Telegram changes, scanner changes, portfolio repair, and watchlist repair unless separately authorized.

Any later data-preparation prompt must explicitly define whether it is allowed to:

- create templates only;
- edit source CSV files;
- use provider-assisted prefill scripts;
- add source references;
- validate coverage after data preparation;
- commit source artifacts.

Until that separate authorization exists, this document remains planning-only.
