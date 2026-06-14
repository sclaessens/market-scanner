# SEC Fundamental Analysis Rationalization

Status: ACTIVE RATIONALIZATION
Backlog driver: BL-0015
Sprint: SEC-5 — Fundamental Analysis Rationalization

## 1. Purpose

SEC-5 rationalizes current fundamental analysis expectations against the SEC-supported field model documented in SEC-4.

The goal is to decide which fundamental fields and metrics remain reliable enough for future implementation, which require approved derivation rules, which should be optional or review-required, and which should be deferred before SEC-6 transformation begins.

SEC-5 does not implement runtime changes.

## 2. Scope Boundary

This sprint is documentation-only.

It does not include:

- code changes;
- tests;
- SEC calls;
- SEC downloads;
- generated data;
- pipeline integration;
- SEC-to-fundamentals transformation;
- changes to metrics, quality, or analysis runtime logic;
- Decision Engine or Reporting changes.

## 3. Source Inputs

SEC-5 uses these source documents:

- SEC strategy document: `docs/active/source_data/sec_edgar_fundamentals_source_strategy.md`
- SEC source architecture document: `docs/active/source_data/sec_edgar_source_architecture.md`
- SEC XBRL mapping investigation document: `docs/active/source_data/sec_xbrl_mapping_investigation.md`
- current fundamentals platform contract: `docs/active/contracts/fundamentals_platform_contract.md`
- current calculation technical spec: `docs/active/contracts/fundamental_calculations_technical_spec.md`
- SEC-2 implementation note: `docs/sprints/sec_2_sec_bulk_intake_implementation.md`
- SEC-3 implementation note: `docs/sprints/sec_3_sec_ticker_cik_coverage.md`

## 4. Rationalization Classification Model

- CORE: Supported enough to remain a direct future implementation candidate when source evidence is present.
- CORE_WITH_ALTERNATES: Supported enough to remain a candidate, but alternate source tags or review paths are expected.
- DERIVED_REQUIRES_APPROVED_RULES: Potentially supported only through approved deterministic derivation rules.
- OPTIONAL: Useful when present, but not required for baseline SEC-supported fundamentals.
- REVIEW_REQUIRED: Requires human or policy review before implementation readiness can be confirmed.
- UNSUPPORTED_FOR_NOW: Not ready for SEC-supported implementation under the current mapping and policy state.
- DEFER_TO_FUTURE_SOURCE: Better handled by a later source, policy, or data model decision.

These statuses are descriptive only and do not imply allocation, ranking, scoring, eligibility, tradeability, urgency, conviction, buy/sell, final action, or hidden filtering.

## 5. Field-Level Rationalization

| field | SEC mapping basis | rationalized_status | reason | SEC-6 readiness | notes |
|---|---|---|---|---|---|
| revenue | Primary and alternate SEC revenue candidates are identified. | CORE_WITH_ALTERNATES | Revenue has plausible SEC tag coverage, but issuer wording and alternate tag usage require controlled mapping. | Ready for non-derived transformation spec. | Preserve tag evidence and avoid silent selection where multiple candidates conflict. |
| gross_profit | `us-gaap:GrossProfit` is identified when available. | OPTIONAL | Gross profit can be useful, but availability may vary by issuer and statement presentation. | Conditional only. | Missing gross profit should not be treated as negative analysis. |
| operating_income | `us-gaap:OperatingIncomeLoss` is identified. | CORE | Operating income has a direct candidate, subject to sign and definition review. | Ready for non-derived transformation spec. | Keep review behavior for issuer-specific definitions. |
| net_income | Primary and alternate SEC net income candidates are identified. | CORE_WITH_ALTERNATES | Net income has plausible SEC tag coverage, but alternate candidate use requires review. | Ready for non-derived transformation spec. | Preserve sign conventions and period evidence. |
| diluted_eps | `us-gaap:EarningsPerShareDiluted` is identified. | REVIEW_REQUIRED | EPS uses per-share units and requires careful annual versus quarterly period handling. | Not ready without unit and period policy. | EPS must not be mixed with monetary facts. |
| total_debt | SEC-4 identifies debt components, not an approved single direct field. | DERIVED_REQUIRES_APPROVED_RULES | Total debt likely requires controlled component rules and double-counting prevention. | Blocked until derivation rules are approved. | Do not infer missing components or mix lease-inclusive and lease-exclusive tags silently. |
| total_equity | Primary and alternate equity candidates are identified. | CORE_WITH_ALTERNATES | Equity has direct candidates, but noncontrolling interest and entity type require review. | Ready for non-derived transformation spec with review conditions. | Instant balance-sheet period-end alignment is required. |
| free_cash_flow | SEC-4 identifies operating cash flow and capex components. | DERIVED_REQUIRES_APPROVED_RULES | Free cash flow is not approved as a direct SEC XBRL field. | Blocked until derivation rules are approved. | Missing capex must not be treated as zero. |

## 6. Metric-Level Rationalization

| metric | required_fields | rationalized_status | reason | SEC-6 dependency | notes |
|---|---|---|---|---|---|
| gross_margin | `gross_profit`, `revenue` | OPTIONAL | The formula remains valid, but gross profit may not be consistently available. | Revenue mapping plus conditional gross profit handling. | Null when required inputs are missing. |
| operating_margin | `operating_income`, `revenue` | CORE_WITH_ALTERNATES | Direct candidate fields appear viable with source evidence and period alignment. | Revenue and operating income mapping. | Remains descriptive only. |
| net_margin | `net_income`, `revenue` | CORE_WITH_ALTERNATES | Direct candidate fields appear viable with alternate net income review. | Revenue and net income mapping. | Null on missing inputs or invalid revenue. |
| debt_to_equity | `total_debt`, `total_equity` | DERIVED_REQUIRES_APPROVED_RULES | Equity may be mapped, but total debt requires approved derivation. | Approved total debt derivation and equity mapping. | Must not run from partial or inferred debt components. |
| return_on_equity | `net_income`, `total_equity` | CORE_WITH_ALTERNATES | Required fields have plausible direct SEC candidates with review conditions. | Net income and equity mapping. | Equity denominator edge cases remain review-required. |
| free_cash_flow_margin | `free_cash_flow`, `revenue` | DERIVED_REQUIRES_APPROVED_RULES | Free cash flow requires approved derivation before use. | Approved free cash flow derivation and revenue mapping. | Capex sign policy is required first. |
| revenue_growth_yoy | current and prior `revenue` | CORE_WITH_ALTERNATES | Revenue appears viable, but comparable periods and currency consistency are required. | Multi-period revenue mapping and period policy. | No invented TTM or mixed annual/quarterly contexts. |
| eps_growth_yoy | current and prior `diluted_eps` | REVIEW_REQUIRED | EPS unit and period handling require careful policy before implementation. | EPS mapping, per-share unit validation, comparable periods. | Sign changes remain review-required. |
| free_cash_flow_growth_yoy | current and prior `free_cash_flow` | DERIVED_REQUIRES_APPROVED_RULES | Depends on approved free cash flow derivation across comparable periods. | Approved free cash flow derivation and period policy. | Do not calculate from inferred capex. |
| revenue_cagr_3y | start and end `revenue` | CORE_WITH_ALTERNATES | Revenue can support CAGR when three-year period comparability is proven. | Multi-year revenue coverage and period consistency. | Start revenue must be positive and source-supported. |
| eps_cagr_3y | start and end `diluted_eps` | REVIEW_REQUIRED | EPS CAGR is sensitive to sign changes, units, and comparable periods. | EPS policy and multi-year period consistency. | Null and review-required for unsafe sign behavior. |
| free_cash_flow_cagr_3y | start and end `free_cash_flow` | DERIVED_REQUIRES_APPROVED_RULES | Depends on approved free cash flow derivation and comparable multi-year facts. | Approved free cash flow derivation. | Not implementation-ready before formula approval. |
| average_gross_margin_3y | three comparable `gross_margin` values | OPTIONAL | Useful when gross profit coverage exists, but not baseline required. | Gross margin availability across three comparable years. | Missing gross profit should preserve null output. |
| average_operating_margin_3y | three comparable `operating_margin` values | CORE_WITH_ALTERNATES | Operating margin remains viable if revenue and operating income are mapped across comparable years. | Operating margin availability across three comparable years. | Descriptive smoothing only. |
| operating_margin_trend_3y | three comparable `operating_margin` values | CORE_WITH_ALTERNATES | Trend is compatible when underlying periods and fields are comparable. | Operating margin availability and trend policy. | No scoring or ranking semantics. |
| debt_to_equity_trend_3y | three comparable `debt_to_equity` values | DERIVED_REQUIRES_APPROVED_RULES | Depends on derived total debt and valid equity over comparable periods. | Approved total debt derivation and multi-year equity mapping. | Invalid equity remains review-required. |
| free_cash_flow_margin_trend_3y | three comparable `free_cash_flow_margin` values | DERIVED_REQUIRES_APPROVED_RULES | Depends on approved free cash flow derivation and margin availability. | Approved free cash flow derivation. | Sign changes remain review-required for interpretation. |
| fiscal_year_count_available | `ticker`, `fiscal_year`, `fiscal_period` | CORE | Coverage helper is compatible with SEC source evidence. | Period metadata extraction. | Duplicate years require review. |
| consecutive_years_available | `ticker`, `fiscal_year`, `fiscal_period` | CORE | Coverage helper is compatible with SEC source evidence. | Period metadata extraction. | Mixed periods require review. |
| missing_required_raw_fields_count | required raw fields | CORE_WITH_ALTERNATES | Completeness helper remains useful, but required fields may need rationalized required/optional policy. | SEC-5 field classification and SEC-6 output contract. | Should distinguish missing optional fields from missing core fields. |
| missing_metric_inputs_count | metric dependency list | CORE_WITH_ALTERNATES | Helper remains useful if metric dependencies are rationalized by field readiness. | Approved metric registry and dependency policy. | Derived metrics must surface blocked dependencies. |
| period_consistency_flag | `fiscal_year`, `fiscal_period`, `period_end_date` | CORE | Period consistency is required by SEC-4 policy. | Period metadata extraction. | Annual and quarterly contexts must not be mixed silently. |
| currency_consistency_flag | `currency` | CORE | Currency consistency is required for monetary comparability. | Unit/currency extraction. | No currency conversion unless future policy approves it. |

## 7. Analysis Simplification Principles

- prefer reliable SEC-supported fields over complex workaround logic;
- do not invent missing values;
- do not treat missing optional fields as negative analysis;
- use review-required states for incomplete or ambiguous data;
- do not create hidden scoring, ranking, tradeability, urgency, conviction, or allocation semantics;
- keep upstream analysis descriptive;
- Decision Engine remains the only allocation authority.

## 8. Derived Metric and Derived Field Dependencies

The following cannot be treated as implementation-ready until formula rules are approved:

- total_debt
- free_cash_flow
- debt_to_equity
- free_cash_flow_margin

SEC-6 must not implement these derived values without approved deterministic rules.

## 9. Quality and Analysis Layer Implications

Future implementation should treat missing core fields as source-readiness or review-required conditions, not as business weakness.

Missing optional fields should remain explicit and should not penalize analysis. Missing derived components should block the derived field or derived metric rather than infer a value.

Ambiguous SEC tags, conflicting facts, unit conflicts, period conflicts, and review-required fields should produce descriptive review-required behavior. They must not silently select a winner without a documented deterministic rule.

Any future quality or analysis output must remain descriptive and row-preserving.

## 10. SEC-6 Readiness Decision

Decision:

```text
READY_ONLY_FOR_NON_DERIVED_FIELDS
```

SEC-6 may prepare transformation specifications for source-supported direct fields first.

`total_debt` and `free_cash_flow` require approved derivation rules before implementation. Metrics depending on those derived fields also remain blocked until the formulas and conflict policies are approved.

## 11. SEC-6 Handoff

SEC-6 should:

- transform only source-supported fields;
- preserve source evidence;
- preserve missing values honestly;
- avoid deriving `total_debt` and `free_cash_flow` unless rules are approved;
- avoid pipeline integration unless separately approved;
- not commit generated outputs;
- use fixture-based tests;
- avoid live SEC dependency in tests.

## 12. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

BL-0015 covers fundamentals source-data and quality classification. BL-0017 covers future governed automated ingestion strategy. The SEC sprint sequence already covers the immediate work.

## 13. No-Runtime-Change Confirmation

- no scripts changed;
- no tests changed;
- no data changed;
- no reports changed;
- no CSV files changed;
- no generated files changed;
- no workflow files changed;
- no runtime behavior changed;
- no SEC/network calls performed;
- no SEC data downloaded;
- no scraping performed;
- no Decision Engine behavior changed;
- no Reporting behavior changed;
- no Telegram behavior changed;
- no portfolio behavior changed.
