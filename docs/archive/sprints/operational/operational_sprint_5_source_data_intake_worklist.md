# Operational Sprint 5 Source-data Intake Worklist

## 1. Status and Scope

This document is a documentation-only worklist note for Operational Sprint 5 source-data intake.

It accompanies empty source-data intake templates for the `145` scanner A/B rows identified by the scanner-based data coverage audit.

This task creates empty intake templates only. It does not populate factual source data.

This document and the accompanying templates do not implement:

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

No sprint is closed or certified complete by this document.

## 2. Created Intake Templates

The following empty intake templates were created:

```text
data/intake/os5_scanner_ab_metadata_intake_template.csv
data/intake/os5_scanner_ab_fundamentals_intake_template.csv
```

Each template contains `145` rows.

The only populated target fields are:

- `ticker`
- `target_date`
- collection status placeholder

Ticker and target date came from the existing scanner A/B output in:

```text
data/processed/scanner_ranked.csv
```

The scanner A/B target date in the templates is `2026-05-19`.

## 3. Data Population Boundary

No factual metadata fields were populated.

No factual fundamentals fields were populated.

No company names, sectors, industries, asset classes, revenue growth values, EPS growth values, operating margin values, leverage values, free cash flow values, provider facts, or source-supported business facts were inferred, guessed, scraped, fetched, synthesized, or manually filled.

No provider/API calls were made.

The templates are intake structures only. They are not runtime source artifacts for the Decision Engine, Reporting, Telegram, scanner, Fundamental Layer, or Portfolio Intelligence.

## 4. Metadata Intake Fields

The metadata intake template contains:

- `ticker`
- `target_date`
- `company_name`
- `sector`
- `industry`
- `asset_class`
- `metadata_source`
- `metadata_source_url`
- `metadata_last_updated`
- `metadata_collection_status`
- `data_steward_notes`

The following fields remain empty until a later source-supported data collection task is authorized:

- `company_name`
- `sector`
- `industry`
- `asset_class`
- `metadata_source`
- `metadata_source_url`
- `metadata_last_updated`
- `data_steward_notes`

`metadata_collection_status` is set to `TO_COLLECT`.

## 5. Fundamentals Intake Fields

The fundamentals intake template contains:

- `ticker`
- `target_date`
- `fundamentals_period`
- `revenue_growth_yoy`
- `eps_growth_yoy`
- `operating_margin`
- `debt_leverage_metric`
- `free_cash_flow_metric`
- `fundamentals_source`
- `fundamentals_source_url`
- `fundamentals_last_updated`
- `fundamentals_collection_status`
- `data_quality_notes`

The following fields remain empty until a later source-supported data collection task is authorized:

- `fundamentals_period`
- `revenue_growth_yoy`
- `eps_growth_yoy`
- `operating_margin`
- `debt_leverage_metric`
- `free_cash_flow_metric`
- `fundamentals_source`
- `fundamentals_source_url`
- `fundamentals_last_updated`
- `data_quality_notes`

`fundamentals_collection_status` is set to `TO_COLLECT`.

## 6. Data Steward Usage

A future data steward may use these templates to organize manual source-supported collection.

Before any factual values are populated, a separate authorized task must define whether the work may:

- edit intake templates only;
- transform intake templates into governed source artifacts;
- edit `data/portfolio/portfolio_metadata.csv`;
- edit `data/raw/fundamentals.csv`;
- use provider-assisted prefill scripts;
- commit populated source artifacts.

Every future populated value must include source support and freshness metadata.

Manual entries must be traceable to a source document, provider export, filing, or other approved reference.

Provider-assisted entries require separate authorization.

No credentials or secrets may be added to the repository, templates, source artifacts, generated outputs, reports, logs, or documentation.

## 7. Governance Constraints

Scanner A/B remains coverage prioritization only.

The intake templates must not become:

- ranking authority
- allocation authority
- tradeability status
- eligibility filtering
- urgency classification
- conviction classification
- hidden filtering
- Decision Engine bypass

Portfolio and watchlist repair remain deferred.

This worklist does not authorize Decision Engine, Reporting, Telegram, scanner, portfolio, watchlist, fundamental, timing, context, validation, provider, or runtime behavior changes.

## 8. Backlog Impact Assessment

Existing backlog items are sufficient for this worklist.

Relevant existing backlog coverage includes:

- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- existing portfolio and watchlist-related backlog items in `docs/sprints/project_backlog.md`

Backlog impact assessment:
- No new backlog items identified.

## 9. Recommended Next Step

Review the empty intake templates before authorizing any data collection.

A future prompt may authorize one of the following limited next steps:

- template review only;
- manual data collection into intake templates;
- provider-assisted prefill preparation;
- transformation from reviewed intake templates into governed source artifacts.

That future prompt must explicitly preserve runtime boundaries and must not authorize Decision Engine, Reporting, Telegram, scanner, portfolio, watchlist, or fundamentals runtime logic changes unless separately governed.
