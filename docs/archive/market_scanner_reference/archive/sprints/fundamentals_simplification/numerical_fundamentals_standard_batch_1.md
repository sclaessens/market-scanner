# Numerical Fundamentals Standard Batch 1

## Status and Scope

This document records the first standard controlled Numerical Fundamentals Batch under the aligned numerical fundamentals scaling contract.

This was a governed data-steward batch task. It was not a coding sprint, runtime-logic change, automated ingestion implementation, provider/API integration task, or source-data commit task.

The task combined manual approved-source extraction, metric-level steward classification, a local ignored raw fundamentals update for fully approved writable MVP metrics, Fundamental Layer validation, and documentation-only commit preparation.

This document does not approve any change to runtime behavior, Decision Engine logic, Reporting logic, Telegram logic, scanner logic, Fundamental Layer logic, Portfolio Intelligence logic, tests, source-controlled portfolio metadata, generated outputs, or GitHub workflows.

## Protocol References

- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`
- `docs/sprints/numerical_fundamentals_contract_scaling_alignment.md`
- `docs/sprints/numerical_fundamentals_pilot_1.md`
- `docs/sprints/fundamentals_provenance_only_update_1.md`
- `docs/sprints/project_backlog.md`

## Scaling Contract Reference

The batch follows the aligned scaling contract:

- standard controlled batch size: up to 10 tickers;
- writable MVP metrics only;
- candidate/future metrics are not written;
- approved values may be written only to local ignored `data/raw/fundamentals.csv`;
- raw fundamentals and backups must not be committed;
- generated outputs must not be committed;
- Fundamental Layer validation is required after local raw updates.

## Batch Selection Rationale

The selected batch uses metadata-complete tickers from the previous metadata expansion batch that were not part of Pilot 1.

Selected batch:

- `ANET`
- `DELL`
- `ENPH`
- `EOG`
- `EQIX`
- `EW`
- `EXPD`
- `FDX`
- `FTNT`
- `HAL`

Excluded from this batch because they were part of Pilot 1:

- `AMAT`
- `ASML`
- `COST`

## Writable MVP Metrics

Writable metrics for this batch:

- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `debt_to_equity`

Candidate/future metrics were not written:

- `net_margin`
- `return_on_equity`
- `free_cash_flow_margin`

## Source Method Used

Manual approved-source extraction was used.

Allowed source classes used in this batch:

- SEC annual filings;
- company investor relations annual results releases;
- company annual report files.

No provider APIs, paid or restricted APIs, credentials, automated scraping, bulk provider extraction, analyst opinions, or price-action-derived values were used.

## Source Lookup Preview Table

Percentage metrics use decimal convention. For example, `0.25` means 25%.

| ticker | metric_name | proposed_value | unit_convention | fiscal_period | source_name | source_reference | source_freshness_date | value_origin | calculation_inputs | calculation_formula | metric_definition_status | period_status | parse_status | steward_state | notes |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ANET | revenue_growth_yoy | 0.285959 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/0001596532/000159653226000013/anet-20251231.htm | 2026-05-24 | CALCULATED | revenue 2025 9005.7; revenue 2024 7003.1 | `(current_period_revenue - prior_period_revenue) / prior_period_revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Official filing values support the governed formula. |
| ANET | eps_growth_yoy | 0.233184 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/0001596532/000159653226000013/anet-20251231.htm | 2026-05-24 | CALCULATED | diluted EPS 2025 2.75; diluted EPS 2024 2.23 | `(current_period_diluted_eps - prior_period_diluted_eps) / abs(prior_period_diluted_eps)` | CLEAR | CLEAR | PARSEABLE | APPROVED | Prior EPS was positive and comparable. |
| ANET | gross_margin | 0.640561 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/0001596532/000159653226000013/anet-20251231.htm | 2026-05-24 | CALCULATED | gross profit 2025 5768.7; revenue 2025 9005.7 | `gross_profit / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period filing values support the governed formula. |
| ANET | operating_margin | 0.428184 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/0001596532/000159653226000013/anet-20251231.htm | 2026-05-24 | CALCULATED | operating income 2025 3856.1; revenue 2025 9005.7 | `operating_income / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period filing values support the governed formula. |
| ANET | debt_to_equity |  | unitless ratio | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/0001596532/000159653226000013/anet-20251231.htm | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Total debt definition was not clear enough for auto-approval. |
| DELL | revenue_growth_yoy | 0.19 | decimal | FY2026 | Dell Technologies FY2026 results release | https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-3 | 2026-05-24 | DIRECT_REPORTED |  |  | CLEAR | CLEAR | PARSEABLE | APPROVED | Official results release directly reports annual revenue growth. |
| DELL | eps_growth_yoy | 0.36 | decimal | FY2026 | Dell Technologies FY2026 results release | https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-3 | 2026-05-24 | DIRECT_REPORTED |  |  | CLEAR | CLEAR | PARSEABLE | APPROVED | Official results release directly reports GAAP diluted EPS growth. |
| DELL | gross_margin |  | decimal | FY2026 | Dell Technologies FY2026 results release | https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-3 | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Readily available margin disclosure emphasized non-GAAP presentation. |
| DELL | operating_margin | 0.071773 | decimal | FY2026 | Dell Technologies FY2026 results release | https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-3 | 2026-05-24 | CALCULATED | operating income 8149; net revenue 113538 | `operating_income / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period official results values support the governed formula. |
| DELL | debt_to_equity |  | unitless ratio | FY2026 | Dell Technologies FY2026 results release | https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-3 | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Debt and equity definition requires review before use. |
| ENPH | revenue_growth_yoy | 0.107189 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm | 2026-05-24 | CALCULATED | revenue 2025 1472985; revenue 2024 1330383 | `(current_period_revenue - prior_period_revenue) / prior_period_revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Official filing values support the governed formula. |
| ENPH | eps_growth_yoy | 0.72 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm | 2026-05-24 | CALCULATED | diluted EPS 2025 1.29; diluted EPS 2024 0.75 | `(current_period_diluted_eps - prior_period_diluted_eps) / abs(prior_period_diluted_eps)` | CLEAR | CLEAR | PARSEABLE | APPROVED | Prior EPS was positive and comparable. |
| ENPH | gross_margin | 0.466403 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm | 2026-05-24 | CALCULATED | gross profit 2025 687004; revenue 2025 1472985 | `gross_profit / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period filing values support the governed formula. |
| ENPH | operating_margin | 0.106943 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm | 2026-05-24 | CALCULATED | operating income 2025 157526; revenue 2025 1472985 | `operating_income / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period filing values support the governed formula. |
| ENPH | debt_to_equity | 1.107959 | unitless ratio | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm | 2026-05-24 | CALCULATED | current debt 632183; noncurrent debt 572194; equity 1087023 | `total_debt / total_equity` | CLEAR | CLEAR | PARSEABLE | APPROVED | Total debt components and equity were visible and period-consistent. |
| EOG | revenue_growth_yoy | -0.044983 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/0000821189/000082118926000054/eog-20251231.htm | 2026-05-24 | CALCULATED | operating revenues 2025 22632; operating revenues 2024 23698 | `(current_period_revenue - prior_period_revenue) / prior_period_revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Official filing values support the governed formula. |
| EOG | eps_growth_yoy | -0.189333 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/0000821189/000082118926000054/eog-20251231.htm | 2026-05-24 | CALCULATED | diluted EPS 2025 9.12; diluted EPS 2024 11.25 | `(current_period_diluted_eps - prior_period_diluted_eps) / abs(prior_period_diluted_eps)` | CLEAR | CLEAR | PARSEABLE | APPROVED | Prior EPS was positive and comparable. |
| EOG | gross_margin |  | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/0000821189/000082118926000054/eog-20251231.htm | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Gross profit definition was not clear enough for oil and gas reporting. |
| EOG | operating_margin | 0.282123 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/0000821189/000082118926000054/eog-20251231.htm | 2026-05-24 | CALCULATED | operating income 2025 6385; operating revenues 2025 22632 | `operating_income / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period filing values support the governed formula. |
| EOG | debt_to_equity |  | unitless ratio | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/0000821189/000082118926000054/eog-20251231.htm | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Debt definition requires review before use. |
| EQIX | revenue_growth_yoy | 0.053612 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1101239/000110123926000032/eqix-20251231.htm | 2026-05-24 | CALCULATED | revenues 2025 9217; revenues 2024 8748 | `(current_period_revenue - prior_period_revenue) / prior_period_revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Official filing values support the governed formula. |
| EQIX | eps_growth_yoy | 0.618824 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1101239/000110123926000032/eqix-20251231.htm | 2026-05-24 | CALCULATED | diluted EPS 2025 13.76; diluted EPS 2024 8.50 | `(current_period_diluted_eps - prior_period_diluted_eps) / abs(prior_period_diluted_eps)` | CLEAR | CLEAR | PARSEABLE | APPROVED | Prior EPS was positive and comparable. |
| EQIX | gross_margin | 0.510904 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1101239/000110123926000032/eqix-20251231.htm | 2026-05-24 | CALCULATED | revenues 2025 9217; cost of revenues 2025 4508 | `(revenue - cost_of_revenues) / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period filing values support the governed formula. |
| EQIX | operating_margin | 0.200499 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1101239/000110123926000032/eqix-20251231.htm | 2026-05-24 | CALCULATED | income from operations 2025 1848; revenues 2025 9217 | `operating_income / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period filing values support the governed formula. |
| EQIX | debt_to_equity |  | unitless ratio | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1101239/000110123926000032/eqix-20251231.htm | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | REIT debt presentation and lease treatment require review before use. |
| EW | revenue_growth_yoy | 0.115470 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1099800/000109980026000009/ew-20251231.htm | 2026-05-24 | CALCULATED | net sales 2025 6067.6; net sales 2024 5439.5 | `(current_period_revenue - prior_period_revenue) / prior_period_revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Official filing values support the governed formula. |
| EW | eps_growth_yoy |  | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1099800/000109980026000009/ew-20251231.htm | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | EPS comparability requires review because of discontinued-operations presentation. |
| EW | gross_margin | 0.804964 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1099800/000109980026000009/ew-20251231.htm | 2026-05-24 | CALCULATED | net sales 2025 6067.6; cost of sales 2025 1183.4 | `(revenue - cost_of_sales) / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period filing values support the governed formula. |
| EW | operating_margin | 0.208353 | decimal | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1099800/000109980026000009/ew-20251231.htm | 2026-05-24 | CALCULATED | operating income 2025 1264.2; net sales 2025 6067.6 | `operating_income / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period filing values support the governed formula. |
| EW | debt_to_equity |  | unitless ratio | FY2025 | SEC Form 10-K | https://www.sec.gov/Archives/edgar/data/1099800/000109980026000009/ew-20251231.htm | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Debt definition requires review before use. |
| EXPD | revenue_growth_yoy | 0.044195 | decimal | FY2025 | Expeditors FY2025 results release | https://investor.expeditors.com/~/media/Files/E/Expeditors-IR-V2/press-release/2026/EXPDQ41.PDF | 2026-05-24 | CALCULATED | revenues 2025 11069009; revenues 2024 10600515 | `(current_period_revenue - prior_period_revenue) / prior_period_revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Official results release values support the governed formula. |
| EXPD | eps_growth_yoy | 0.040210 | decimal | FY2025 | Expeditors FY2025 results release | https://investor.expeditors.com/~/media/Files/E/Expeditors-IR-V2/press-release/2026/EXPDQ41.PDF | 2026-05-24 | CALCULATED | diluted EPS 2025 5.95; diluted EPS 2024 5.72 | `(current_period_diluted_eps - prior_period_diluted_eps) / abs(prior_period_diluted_eps)` | CLEAR | CLEAR | PARSEABLE | APPROVED | Prior EPS was positive and comparable. |
| EXPD | gross_margin |  | decimal | FY2025 | Expeditors FY2025 results release | https://investor.expeditors.com/~/media/Files/E/Expeditors-IR-V2/press-release/2026/EXPDQ41.PDF | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Logistics gross margin definition requires review before use. |
| EXPD | operating_margin | 0.095087 | decimal | FY2025 | Expeditors FY2025 results release | https://investor.expeditors.com/~/media/Files/E/Expeditors-IR-V2/press-release/2026/EXPDQ41.PDF | 2026-05-24 | CALCULATED | operating income 2025 1052546; revenues 2025 11069009 | `operating_income / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period release values support the governed formula. |
| EXPD | debt_to_equity |  | unitless ratio | FY2025 | Expeditors FY2025 results release | https://investor.expeditors.com/~/media/Files/E/Expeditors-IR-V2/press-release/2026/EXPDQ41.PDF | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Debt definition requires review before use. |
| FDX | revenue_growth_yoy | 0.002281 | decimal | FY2025 | FedEx FY2025 results release | https://investor.fedex.com/news-and-events/investor-news/investor-news-details/2025/FedEx-Reports-Fourth-Quarter-Diluted-EPS-of-6-88-and-Adjusted-Diluted-EPS-of-6-07/default.aspx | 2026-05-24 | CALCULATED | revenue 2025 87.9; revenue 2024 87.7 | `(current_period_revenue - prior_period_revenue) / prior_period_revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Official results release values support the governed formula. |
| FDX | eps_growth_yoy | -0.023242 | decimal | FY2025 | FedEx FY2025 results release | https://investor.fedex.com/news-and-events/investor-news/investor-news-details/2025/FedEx-Reports-Fourth-Quarter-Diluted-EPS-of-6-88-and-Adjusted-Diluted-EPS-of-6-07/default.aspx | 2026-05-24 | CALCULATED | diluted EPS 2025 16.81; diluted EPS 2024 17.21 | `(current_period_diluted_eps - prior_period_diluted_eps) / abs(prior_period_diluted_eps)` | CLEAR | CLEAR | PARSEABLE | APPROVED | Prior EPS was positive and comparable. |
| FDX | gross_margin |  | decimal | FY2025 | FedEx FY2025 results release | https://investor.fedex.com/news-and-events/investor-news/investor-news-details/2025/FedEx-Reports-Fourth-Quarter-Diluted-EPS-of-6-88-and-Adjusted-Diluted-EPS-of-6-07/default.aspx | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Gross profit definition requires review before use. |
| FDX | operating_margin | 0.059 | decimal | FY2025 | FedEx FY2025 results release | https://investor.fedex.com/news-and-events/investor-news/investor-news-details/2025/FedEx-Reports-Fourth-Quarter-Diluted-EPS-of-6-88-and-Adjusted-Diluted-EPS-of-6-07/default.aspx | 2026-05-24 | DIRECT_REPORTED |  |  | CLEAR | CLEAR | PARSEABLE | APPROVED | Official results release directly reports FY2025 operating margin. |
| FDX | debt_to_equity |  | unitless ratio | FY2025 | FedEx FY2025 results release | https://investor.fedex.com/news-and-events/investor-news/investor-news-details/2025/FedEx-Reports-Fourth-Quarter-Diluted-EPS-of-6-88-and-Adjusted-Diluted-EPS-of-6-07/default.aspx | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Debt definition requires review before use. |
| FTNT | revenue_growth_yoy | 0.141677 | decimal | FY2025 | Fortinet FY2025 annual report | https://investor.fortinet.com/static-files/7058aea1-a50c-4b7b-905f-ba5aedb92f40 | 2026-05-24 | CALCULATED | revenue 2025 6799.6; revenue 2024 5955.8 | `(current_period_revenue - prior_period_revenue) / prior_period_revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Official annual report values support the governed formula. |
| FTNT | eps_growth_yoy | 0.070796 | decimal | FY2025 | Fortinet FY2025 annual report | https://investor.fortinet.com/static-files/7058aea1-a50c-4b7b-905f-ba5aedb92f40 | 2026-05-24 | CALCULATED | diluted EPS 2025 2.42; diluted EPS 2024 2.26 | `(current_period_diluted_eps - prior_period_diluted_eps) / abs(prior_period_diluted_eps)` | CLEAR | CLEAR | PARSEABLE | APPROVED | Prior EPS was positive and comparable. |
| FTNT | gross_margin | 0.804562 | decimal | FY2025 | Fortinet FY2025 annual report | https://investor.fortinet.com/static-files/7058aea1-a50c-4b7b-905f-ba5aedb92f40 | 2026-05-24 | CALCULATED | gross profit 2025 5470.7; revenue 2025 6799.6 | `gross_profit / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period annual report values support the governed formula. |
| FTNT | operating_margin | 0.306592 | decimal | FY2025 | Fortinet FY2025 annual report | https://investor.fortinet.com/static-files/7058aea1-a50c-4b7b-905f-ba5aedb92f40 | 2026-05-24 | CALCULATED | operating income 2025 2084.7; revenue 2025 6799.6 | `operating_income / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period annual report values support the governed formula. |
| FTNT | debt_to_equity | 0.805091 | unitless ratio | FY2025 | Fortinet FY2025 annual report | https://investor.fortinet.com/static-files/7058aea1-a50c-4b7b-905f-ba5aedb92f40 | 2026-05-24 | CALCULATED | current debt 499.7; long-term debt 496.6; equity 1237.5 | `total_debt / total_equity` | CLEAR | CLEAR | PARSEABLE | APPROVED | Debt components and equity were visible and period-consistent. |
| HAL | revenue_growth_yoy | -0.033124 | decimal | FY2025 | Halliburton FY2025 results release | https://ir.halliburton.com/news-releases/news-release-details/halliburton-announces-fourth-quarter-2025-results | 2026-05-24 | CALCULATED | revenue 2025 22184; revenue 2024 22944 | `(current_period_revenue - prior_period_revenue) / prior_period_revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Official results release values support the governed formula. |
| HAL | eps_growth_yoy | -0.469965 | decimal | FY2025 | Halliburton FY2025 results release | https://ir.halliburton.com/news-releases/news-release-details/halliburton-announces-fourth-quarter-2025-results | 2026-05-24 | CALCULATED | diluted EPS 2025 1.50; diluted EPS 2024 2.83 | `(current_period_diluted_eps - prior_period_diluted_eps) / abs(prior_period_diluted_eps)` | CLEAR | CLEAR | PARSEABLE | APPROVED | Prior EPS was positive and comparable. |
| HAL | gross_margin |  | decimal | FY2025 | Halliburton FY2025 results release | https://ir.halliburton.com/news-releases/news-release-details/halliburton-announces-fourth-quarter-2025-results | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Gross profit definition was not clear enough for oilfield-services reporting. |
| HAL | operating_margin | 0.101875 | decimal | FY2025 | Halliburton FY2025 results release | https://ir.halliburton.com/news-releases/news-release-details/halliburton-announces-fourth-quarter-2025-results | 2026-05-24 | CALCULATED | operating income 2025 2260; revenue 2025 22184 | `operating_income / revenue` | CLEAR | CLEAR | PARSEABLE | APPROVED | Same-period official results values support the governed formula. |
| HAL | debt_to_equity |  | unitless ratio | FY2025 | Halliburton FY2025 results release | https://ir.halliburton.com/news-releases/news-release-details/halliburton-announces-fourth-quarter-2025-results | 2026-05-24 | MISSING |  |  | UNCLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Debt definition requires review before use. |

## Metric-Level Steward Classification

Approval distribution:

- `APPROVED`: 36
- `REVIEW_REQUIRED`: 14
- `REJECTED`: 0

Approved metrics:

- `ANET`: `revenue_growth_yoy`, `eps_growth_yoy`, `gross_margin`, `operating_margin`
- `DELL`: `revenue_growth_yoy`, `eps_growth_yoy`, `operating_margin`
- `ENPH`: `revenue_growth_yoy`, `eps_growth_yoy`, `gross_margin`, `operating_margin`, `debt_to_equity`
- `EOG`: `revenue_growth_yoy`, `eps_growth_yoy`, `operating_margin`
- `EQIX`: `revenue_growth_yoy`, `eps_growth_yoy`, `gross_margin`, `operating_margin`
- `EW`: `revenue_growth_yoy`, `gross_margin`, `operating_margin`
- `EXPD`: `revenue_growth_yoy`, `eps_growth_yoy`, `operating_margin`
- `FDX`: `revenue_growth_yoy`, `eps_growth_yoy`, `operating_margin`
- `FTNT`: `revenue_growth_yoy`, `eps_growth_yoy`, `gross_margin`, `operating_margin`, `debt_to_equity`
- `HAL`: `revenue_growth_yoy`, `eps_growth_yoy`, `operating_margin`

Review-required metrics:

- `ANET`: `debt_to_equity`
- `DELL`: `gross_margin`, `debt_to_equity`
- `EOG`: `gross_margin`, `debt_to_equity`
- `EQIX`: `debt_to_equity`
- `EW`: `eps_growth_yoy`, `debt_to_equity`
- `EXPD`: `gross_margin`, `debt_to_equity`
- `FDX`: `gross_margin`, `debt_to_equity`
- `HAL`: `gross_margin`, `debt_to_equity`

Rejected metrics:

- None.

## Local Ignored Update Decision

At least one metric was approved, so the local ignored raw fundamentals file was updated.

Updated local ignored artifact:

- `data/raw/fundamentals.csv`

Backup created before editing:

- `data/raw/fundamentals_backup_before_standard_numerical_batch_1.csv`

Backup row count:

- 36

Raw fundamentals row count before update:

- 36

Raw fundamentals row count after update:

- 36

Rows added:

- 0

Rows updated locally:

- 10

Only approved writable MVP metrics were written. Review-required fields were left blank. Candidate/future metrics were not added or written.

## Local Update Table

| ticker | metric_name | approved_for_local_update | value_written_locally | source_reference | validation_state |
|---|---|---|---:|---|---|
| ANET | revenue_growth_yoy | YES | 0.285959 | https://www.sec.gov/Archives/edgar/data/0001596532/000159653226000013/anet-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| ANET | eps_growth_yoy | YES | 0.233184 | https://www.sec.gov/Archives/edgar/data/0001596532/000159653226000013/anet-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| ANET | gross_margin | YES | 0.640561 | https://www.sec.gov/Archives/edgar/data/0001596532/000159653226000013/anet-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| ANET | operating_margin | YES | 0.428184 | https://www.sec.gov/Archives/edgar/data/0001596532/000159653226000013/anet-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| ANET | debt_to_equity | NO |  | https://www.sec.gov/Archives/edgar/data/0001596532/000159653226000013/anet-20251231.htm | REVIEW_REQUIRED |
| DELL | revenue_growth_yoy | YES | 0.19 | https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-3 | VALIDATED_NUMERICAL_STANDARD_BATCH |
| DELL | eps_growth_yoy | YES | 0.36 | https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-3 | VALIDATED_NUMERICAL_STANDARD_BATCH |
| DELL | gross_margin | NO |  | https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-3 | REVIEW_REQUIRED |
| DELL | operating_margin | YES | 0.071773 | https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-3 | VALIDATED_NUMERICAL_STANDARD_BATCH |
| DELL | debt_to_equity | NO |  | https://investors.delltechnologies.com/news-releases/news-release-details/dell-technologies-delivers-fourth-quarter-and-full-year-fiscal-3 | REVIEW_REQUIRED |
| ENPH | revenue_growth_yoy | YES | 0.107189 | https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| ENPH | eps_growth_yoy | YES | 0.72 | https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| ENPH | gross_margin | YES | 0.466403 | https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| ENPH | operating_margin | YES | 0.106943 | https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| ENPH | debt_to_equity | YES | 1.107959 | https://www.sec.gov/Archives/edgar/data/1463101/000146310126000013/enph-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EOG | revenue_growth_yoy | YES | -0.044983 | https://www.sec.gov/Archives/edgar/data/0000821189/000082118926000054/eog-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EOG | eps_growth_yoy | YES | -0.189333 | https://www.sec.gov/Archives/edgar/data/0000821189/000082118926000054/eog-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EOG | gross_margin | NO |  | https://www.sec.gov/Archives/edgar/data/0000821189/000082118926000054/eog-20251231.htm | REVIEW_REQUIRED |
| EOG | operating_margin | YES | 0.282123 | https://www.sec.gov/Archives/edgar/data/0000821189/000082118926000054/eog-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EOG | debt_to_equity | NO |  | https://www.sec.gov/Archives/edgar/data/0000821189/000082118926000054/eog-20251231.htm | REVIEW_REQUIRED |
| EQIX | revenue_growth_yoy | YES | 0.053612 | https://www.sec.gov/Archives/edgar/data/1101239/000110123926000032/eqix-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EQIX | eps_growth_yoy | YES | 0.618824 | https://www.sec.gov/Archives/edgar/data/1101239/000110123926000032/eqix-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EQIX | gross_margin | YES | 0.510904 | https://www.sec.gov/Archives/edgar/data/1101239/000110123926000032/eqix-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EQIX | operating_margin | YES | 0.200499 | https://www.sec.gov/Archives/edgar/data/1101239/000110123926000032/eqix-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EQIX | debt_to_equity | NO |  | https://www.sec.gov/Archives/edgar/data/1101239/000110123926000032/eqix-20251231.htm | REVIEW_REQUIRED |
| EW | revenue_growth_yoy | YES | 0.115470 | https://www.sec.gov/Archives/edgar/data/1099800/000109980026000009/ew-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EW | eps_growth_yoy | NO |  | https://www.sec.gov/Archives/edgar/data/1099800/000109980026000009/ew-20251231.htm | REVIEW_REQUIRED |
| EW | gross_margin | YES | 0.804964 | https://www.sec.gov/Archives/edgar/data/1099800/000109980026000009/ew-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EW | operating_margin | YES | 0.208353 | https://www.sec.gov/Archives/edgar/data/1099800/000109980026000009/ew-20251231.htm | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EW | debt_to_equity | NO |  | https://www.sec.gov/Archives/edgar/data/1099800/000109980026000009/ew-20251231.htm | REVIEW_REQUIRED |
| EXPD | revenue_growth_yoy | YES | 0.044195 | https://investor.expeditors.com/~/media/Files/E/Expeditors-IR-V2/press-release/2026/EXPDQ41.PDF | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EXPD | eps_growth_yoy | YES | 0.040210 | https://investor.expeditors.com/~/media/Files/E/Expeditors-IR-V2/press-release/2026/EXPDQ41.PDF | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EXPD | gross_margin | NO |  | https://investor.expeditors.com/~/media/Files/E/Expeditors-IR-V2/press-release/2026/EXPDQ41.PDF | REVIEW_REQUIRED |
| EXPD | operating_margin | YES | 0.095087 | https://investor.expeditors.com/~/media/Files/E/Expeditors-IR-V2/press-release/2026/EXPDQ41.PDF | VALIDATED_NUMERICAL_STANDARD_BATCH |
| EXPD | debt_to_equity | NO |  | https://investor.expeditors.com/~/media/Files/E/Expeditors-IR-V2/press-release/2026/EXPDQ41.PDF | REVIEW_REQUIRED |
| FDX | revenue_growth_yoy | YES | 0.002281 | https://investor.fedex.com/news-and-events/investor-news/investor-news-details/2025/FedEx-Reports-Fourth-Quarter-Diluted-EPS-of-6-88-and-Adjusted-Diluted-EPS-of-6-07/default.aspx | VALIDATED_NUMERICAL_STANDARD_BATCH |
| FDX | eps_growth_yoy | YES | -0.023242 | https://investor.fedex.com/news-and-events/investor-news/investor-news-details/2025/FedEx-Reports-Fourth-Quarter-Diluted-EPS-of-6-88-and-Adjusted-Diluted-EPS-of-6-07/default.aspx | VALIDATED_NUMERICAL_STANDARD_BATCH |
| FDX | gross_margin | NO |  | https://investor.fedex.com/news-and-events/investor-news/investor-news-details/2025/FedEx-Reports-Fourth-Quarter-Diluted-EPS-of-6-88-and-Adjusted-Diluted-EPS-of-6-07/default.aspx | REVIEW_REQUIRED |
| FDX | operating_margin | YES | 0.059 | https://investor.fedex.com/news-and-events/investor-news/investor-news-details/2025/FedEx-Reports-Fourth-Quarter-Diluted-EPS-of-6-88-and-Adjusted-Diluted-EPS-of-6-07/default.aspx | VALIDATED_NUMERICAL_STANDARD_BATCH |
| FDX | debt_to_equity | NO |  | https://investor.fedex.com/news-and-events/investor-news/investor-news-details/2025/FedEx-Reports-Fourth-Quarter-Diluted-EPS-of-6-88-and-Adjusted-Diluted-EPS-of-6-07/default.aspx | REVIEW_REQUIRED |
| FTNT | revenue_growth_yoy | YES | 0.141677 | https://investor.fortinet.com/static-files/7058aea1-a50c-4b7b-905f-ba5aedb92f40 | VALIDATED_NUMERICAL_STANDARD_BATCH |
| FTNT | eps_growth_yoy | YES | 0.070796 | https://investor.fortinet.com/static-files/7058aea1-a50c-4b7b-905f-ba5aedb92f40 | VALIDATED_NUMERICAL_STANDARD_BATCH |
| FTNT | gross_margin | YES | 0.804562 | https://investor.fortinet.com/static-files/7058aea1-a50c-4b7b-905f-ba5aedb92f40 | VALIDATED_NUMERICAL_STANDARD_BATCH |
| FTNT | operating_margin | YES | 0.306592 | https://investor.fortinet.com/static-files/7058aea1-a50c-4b7b-905f-ba5aedb92f40 | VALIDATED_NUMERICAL_STANDARD_BATCH |
| FTNT | debt_to_equity | YES | 0.805091 | https://investor.fortinet.com/static-files/7058aea1-a50c-4b7b-905f-ba5aedb92f40 | VALIDATED_NUMERICAL_STANDARD_BATCH |
| HAL | revenue_growth_yoy | YES | -0.033124 | https://ir.halliburton.com/news-releases/news-release-details/halliburton-announces-fourth-quarter-2025-results | VALIDATED_NUMERICAL_STANDARD_BATCH |
| HAL | eps_growth_yoy | YES | -0.469965 | https://ir.halliburton.com/news-releases/news-release-details/halliburton-announces-fourth-quarter-2025-results | VALIDATED_NUMERICAL_STANDARD_BATCH |
| HAL | gross_margin | NO |  | https://ir.halliburton.com/news-releases/news-release-details/halliburton-announces-fourth-quarter-2025-results | REVIEW_REQUIRED |
| HAL | operating_margin | YES | 0.101875 | https://ir.halliburton.com/news-releases/news-release-details/halliburton-announces-fourth-quarter-2025-results | VALIDATED_NUMERICAL_STANDARD_BATCH |
| HAL | debt_to_equity | NO |  | https://ir.halliburton.com/news-releases/news-release-details/halliburton-announces-fourth-quarter-2025-results | REVIEW_REQUIRED |

## Fundamental Layer Validation Results

Command run:

```bash
PYTHONPATH=. .venv/bin/python scripts/core/build_fundamental_layer.py
```

Result:

- Success.
- `data/processed/fundamental_quality.csv` was written.
- `data/logs/fundamental_layer_log.csv` was written.

Post-run Fundamental Layer output:

- Row count: 6
- `quality_state` distribution:
  - `SUFFICIENT_DATA`: 4
  - `PARTIAL_DATA`: 2
- `quality_metadata_status` distribution:
  - `complete`: 4
  - `partial`: 2
- `source_data_status` distribution:
  - `source_available`: 4
  - `partial_data`: 2

Selected batch post-validation status:

| ticker | raw_row_present | approved_metrics_written | present_in_direct_builder_output | quality_state_after_validation | source_data_status_after_validation | validation_state |
|---|---:|---:|---:|---|---|---|
| ANET | YES | 4 | NO | not observed | not observed | REVIEW_REQUIRED |
| DELL | YES | 3 | NO | not observed | not observed | REVIEW_REQUIRED |
| ENPH | YES | 5 | NO | not observed | not observed | REVIEW_REQUIRED |
| EOG | YES | 3 | NO | not observed | not observed | REVIEW_REQUIRED |
| EQIX | YES | 4 | NO | not observed | not observed | REVIEW_REQUIRED |
| EW | YES | 3 | NO | not observed | not observed | REVIEW_REQUIRED |
| EXPD | YES | 3 | NO | not observed | not observed | REVIEW_REQUIRED |
| FDX | YES | 3 | NO | not observed | not observed | REVIEW_REQUIRED |
| FTNT | YES | 5 | NO | not observed | not observed | REVIEW_REQUIRED |
| HAL | YES | 3 | NO | not observed | not observed | REVIEW_REQUIRED |

Validation limitation:

- The direct Fundamental Layer run consumed the currently available upstream input with 6 rows.
- The 10 selected batch tickers were not present in that direct builder output.
- The local raw update and CSV validation confirm that approved metrics were written in the ignored raw source artifact, but per-ticker Fundamental Layer output for this batch was not observable without refreshing broader upstream generated artifacts.
- The full pipeline was not run by default because this task is a controlled source-data batch and the instructions do not require broad pipeline execution unless safe and useful.

## Generated Artifact Handling

The Fundamental Layer builder produced generated runtime artifacts:

- `data/processed/fundamental_quality.csv`
- `data/logs/fundamental_layer_log.csv`

No generated artifacts were committed.

`git status --short` showed no tracked generated-file changes after the builder run.

## Git Ignored and Untracked Confirmation

Local ignored raw fundamentals artifacts:

- `data/raw/fundamentals.csv`
- `data/raw/fundamentals_backup_before_standard_numerical_batch_1.csv`

Both paths are ignored by `.gitignore` through `data/raw/`.

The raw fundamentals update and backup remain local ignored operator artifacts and are not committed.

## Validation Limitations

- Manual approved-source extraction was limited to the selected 10 tickers and five writable MVP metrics.
- Review-required metrics were left blank.
- Candidate/future metrics were not collected or written.
- The direct Fundamental Layer builder did not include the selected batch tickers in its current output universe.
- The full pipeline was not run.
- No runtime tests were run because this was a governed source-data batch with documentation-only commit scope, not a code or test change.

## Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

## Recommended Next Step

Review and merge this documentation report, then run a follow-up validation task that refreshes the broader upstream pipeline only if explicitly approved. After that validation, continue with either:

- a second standard controlled numerical fundamentals batch for up to 10 metadata-complete tickers; or
- a targeted review pass for the 14 review-required metrics from this batch.
