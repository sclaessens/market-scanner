# Numerical Fundamentals Source Lookup Preview 1

## Status and Scope

This document is the first governed Numerical Fundamentals Source Lookup Preview for the 15 selected fundamentals tickers.

This is a governed source-data lookup preview and steward classification artifact.

This is not a coding sprint, runtime-logic change, local raw fundamentals update task, automated ingestion implementation, or approval-to-write task.

This document does not:

- update `data/raw/fundamentals.csv`;
- update any CSV file;
- write source-data values to any artifact;
- authorize automatic numerical writes;
- run the full pipeline;
- modify code;
- modify tests;
- modify generated outputs;
- change runtime behavior;
- change Decision Engine logic;
- change Reporting logic;
- change Telegram logic;
- change scanner logic;
- change Fundamental Layer logic;
- change Portfolio Intelligence logic.

No numerical values are approved for write by this task.

## Protocol References

Governance and audit references:

- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`
- `docs/sprints/numerical_fundamentals_source_method_preview_1.md`
- `docs/sprints/fundamentals_source_data_expansion_preview_1.md`
- `docs/sprints/fundamentals_provenance_only_update_1.md`
- `docs/sprints/project_backlog.md`

This task follows the Automated Source Data Steward protocol and the Numerical Fundamentals Source Method Preview.

The selected batch is limited to the 15 approved tickers documented below.

## Selected Batch

Selected tickers:

- `AMAT`
- `ANET`
- `ASML`
- `COST`
- `DELL`
- `ENPH`
- `EOG`
- `EQIX`
- `EW`
- `EXPD`
- `FDX`
- `FTNT`
- `HAL`
- `HLT`
- `HPE`

## MVP Metrics

MVP metrics for this preview:

- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `net_margin`
- `debt_to_equity`
- `return_on_equity`
- `free_cash_flow_margin`

Alignment note:

- `docs/sprints/numerical_fundamentals_contract_scaling_alignment.md` supersedes this broad exploratory list for scaling.
- Current source-data scaling may write only the writable MVP metrics supported by the current raw fundamentals schema.
- `net_margin`, `return_on_equity`, and `free_cash_flow_margin` remain candidate/future metrics until explicitly supported.

Percentage metrics use decimal convention when eventually approved.

Examples:

- 25% becomes `0.25`
- -8% becomes `-0.08`

`debt_to_equity` uses a unitless numeric ratio.

## Current Local Fundamentals State

Local ignored source artifact inspected:

- `data/raw/fundamentals.csv`

Observed state:

- The file exists locally.
- The file remains ignored through `.gitignore`.
- The selected 15 tickers each have a provenance-only row.
- The selected 15 tickers have blank numerical metric fields.
- No file modifications were made during inspection.

## Approved Source Method Used

Recommended source method from `docs/sprints/numerical_fundamentals_source_method_preview_1.md`:

- Manual approved-source extraction for a small batch preview.

Allowed source classes for a later value collection task include:

- public company investor-relations financial statements or annual reports;
- SEC filings where applicable;
- company-published annual reports;
- reputable public financial data pages only if terms permit manual viewing and no scraping/API use occurs.

## Source Lookup Limitations

This preview did not collect approved numerical values.

A limited manual source-method smoke check found that public financial data pages can expose the required concepts, but those pages may rely on third-party data providers and require explicit source-method, terms, and metric-mapping review before values can be approved for local ignored source-data updates.

Because this task must not use paid or restricted APIs, credentials, automated scraping, unapproved providers, or unguided calculations, all metric rows are classified as `REVIEW_REQUIRED`.

No provider APIs were called.

No scraping was performed.

No credentials or secrets were created.

No numerical values were written to any CSV file.

## Source Lookup Preview Table

| ticker | metric_name | proposed_value | unit_convention | fiscal_period | source_name | source_reference | source_freshness_date | value_origin | metric_definition_status | period_status | parse_status | steward_state | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AMAT | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| AMAT | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| AMAT | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| AMAT | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| AMAT | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| AMAT | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| AMAT | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| AMAT | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| ANET | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| ANET | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| ANET | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| ANET | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| ANET | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| ANET | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| ANET | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| ANET | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| ASML | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| ASML | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| ASML | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| ASML | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| ASML | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| ASML | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| ASML | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| ASML | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| COST | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| COST | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| COST | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| COST | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| COST | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| COST | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| COST | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| COST | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| DELL | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| DELL | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| DELL | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| DELL | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| DELL | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| DELL | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| DELL | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| DELL | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| ENPH | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| ENPH | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| ENPH | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| ENPH | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| ENPH | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| ENPH | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| ENPH | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| ENPH | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| EOG | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| EOG | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| EOG | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EOG | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EOG | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EOG | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| EOG | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| EOG | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| EQIX | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| EQIX | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| EQIX | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EQIX | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EQIX | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EQIX | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| EQIX | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| EQIX | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| EW | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| EW | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| EW | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EW | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EW | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EW | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| EW | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| EW | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| EXPD | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| EXPD | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| EXPD | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EXPD | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EXPD | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| EXPD | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| EXPD | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| EXPD | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| FDX | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| FDX | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| FDX | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| FDX | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| FDX | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| FDX | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| FDX | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| FDX | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| FTNT | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| FTNT | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| FTNT | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| FTNT | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| FTNT | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| FTNT | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| FTNT | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| FTNT | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| HAL | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| HAL | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| HAL | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| HAL | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| HAL | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| HAL | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| HAL | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| HAL | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| HLT | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| HLT | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| HLT | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| HLT | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| HLT | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| HLT | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| HLT | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| HLT | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |
| HPE | revenue_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; source and period require later governed lookup. |
| HPE | eps_growth_yoy |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; EPS definition and period require later governed lookup. |
| HPE | gross_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| HPE | operating_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| HPE | net_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; period and source mapping require later governed lookup. |
| HPE | debt_to_equity |  | unitless_ratio |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; debt/equity definition requires later governed lookup. |
| HPE | return_on_equity |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; equity denominator and period require later governed lookup. |
| HPE | free_cash_flow_margin |  | decimal_percent |  |  |  |  | MISSING | CLEAR | UNCLEAR | MISSING | REVIEW_REQUIRED | No approved numerical value captured; free-cash-flow definition requires later governed lookup. |

## Row-Level Summary

| ticker | approved_metric_count | review_required_metric_count | rejected_metric_count | row_steward_state | eligible_for_later_local_update | notes |
|---|---:|---:|---:|---|---|---|
| AMAT | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| ANET | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| ASML | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| COST | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| DELL | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| ENPH | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| EOG | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| EQIX | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| EW | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| EXPD | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| FDX | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| FTNT | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| HAL | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| HLT | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |
| HPE | 0 | 8 | 0 | REVIEW_REQUIRED | NO | All metrics require later governed source lookup before local update consideration. |

## Approval State Distribution

Metric-level distribution:

| steward_state | metric_count |
|---|---:|
| APPROVED | 0 |
| REVIEW_REQUIRED | 120 |
| REJECTED | 0 |

Row-level distribution:

| row_steward_state | row_count |
|---|---:|
| REVIEW_REQUIRED | 15 |

## Metrics Eligible For Later Local Ignored Update Consideration

No metrics are eligible for later local ignored update consideration from this preview.

Eligibility requires metric-level `APPROVED` classification, which was not assigned in this task.

## Metrics Requiring Human Review

All 120 metric rows require human review before local ignored update consideration.

Review is required because no numerical values were approved, no source references were accepted for metric-level write consideration, and source-method/period/definition mapping must be completed under a later explicit numerical sourcing task.

## Rejected Metrics

No metrics were rejected.

## Source-Method Observations

Manual approved-source extraction remains the recommended source method for the next task.

The next task should prefer company annual reports, investor-relations financial statements, and SEC filings where applicable.

If a reputable public financial data page is used, the task must confirm that manual viewing is permitted, source terms are acceptable, provider lineage is acceptable, and metric definitions map cleanly to the MVP contract.

Provider/API-assisted extraction remains deferred under `BL-0017`.

## Validation Notes

Validation was documentation-only.

Inspected local ignored `data/raw/fundamentals.csv` only to confirm selected rows are present and numerical metric fields are blank.

No runtime tests were run.

No pipeline was run.

No source-data values were written to any CSV.

No generated files were modified.

No CSV files were modified.

No raw fundamentals file was modified.

No provider APIs were called.

No scraping was performed.

No credentials or secrets were created.

## Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Existing backlog coverage remains sufficient:

- `BL-0015 — Define and implement approved Fundamental data source and quality classification contract`
- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- `BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract`
- `BL-0011 — Define and repair authoritative active portfolio source`

## Recommended Next Step

Launch a narrower numerical fundamentals source lookup task that explicitly selects one approved source class and a smaller pilot subset.

Recommended pilot:

- Use company annual reports or SEC filings as the primary source class.
- Start with 3 to 5 tickers from the selected batch.
- Capture preview values only.
- Classify each metric as `APPROVED`, `REVIEW_REQUIRED`, or `REJECTED`.
- Do not update `data/raw/fundamentals.csv` until a later explicit local ignored update task.
