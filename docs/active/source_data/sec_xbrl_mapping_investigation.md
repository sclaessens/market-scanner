# SEC XBRL Mapping Investigation

Status: ACTIVE INVESTIGATION SKELETON
Backlog driver: BL-0015
Sprint: SEC-4A — SEC XBRL Mapping Investigation Skeleton

## Purpose

This document will become the source-data mapping investigation for mapping SEC XBRL Company Facts into the internal fundamentals model.

## Scope Boundary

This investigation is documentation-only.

It does not include:

- code changes;
- tests;
- SEC calls;
- SEC downloads;
- generated data;
- pipeline integration;
- SEC-to-fundamentals transformation;
- changes to metrics, quality, or analysis logic.

## Internal Target Fields

- revenue
- gross_profit
- operating_income
- net_income
- diluted_eps
- total_debt
- total_equity
- free_cash_flow

## Mapping Status Definitions

- PRIMARY_CANDIDATE: A candidate mapping that may be considered as a primary descriptive source during review.
- ALTERNATE_CANDIDATE: A candidate mapping that may be considered as an alternate descriptive source during review.
- DERIVED_COMPONENT: A candidate that may contribute to a derived value if future derivation rules are approved.
- REVIEW_REQUIRED: A candidate or field that requires further human review before any implementation decision.
- REJECTED_CANDIDATE: A candidate that has been reviewed and documented as unsuitable for the intended mapping.
- UNSUPPORTED: A field or candidate that is not supported by the current investigation state.

These statuses are descriptive only and do not imply ranking, scoring, eligibility, tradeability, urgency, conviction, buy/sell, allocation, final action, or hidden filtering.

## Field Reliability Status Definitions

- CORE_IF_AVAILABLE: The field may be treated as core when a reviewed source is available.
- CORE_WITH_ALTERNATES: The field may require reviewed alternate candidates to support consistent coverage.
- DERIVED_REQUIRES_RULES: The field may require explicit derivation rules before implementation.
- OPTIONAL: The field may remain optional for source-data mapping purposes.
- REVIEW_REQUIRED: The field requires further review before reliability can be classified.
- UNSUPPORTED_UNTIL_FURTHER_REVIEW: The field remains unsupported until additional review is completed.

## Mapping Table

| internal_field | candidate_tag | candidate_role | unit_expectation | period_type | derivation_needed | reliability_classification | review_notes | implementation_status |
|---|---|---|---|---|---|---|---|---|
| revenue | us-gaap:Revenues | PRIMARY_CANDIDATE | monetary | duration | no | CORE_WITH_ALTERNATES | Review issuer wording and whether the reported concept represents total revenue. | INVESTIGATION_ONLY |
| revenue | us-gaap:SalesRevenueNet | ALTERNATE_CANDIDATE | monetary | duration | no | CORE_WITH_ALTERNATES | Use as an alternate revenue candidate where issuer reporting uses sales revenue terminology. | INVESTIGATION_ONLY |
| revenue | us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax | ALTERNATE_CANDIDATE | monetary | duration | no | CORE_WITH_ALTERNATES | Review applicability under revenue recognition disclosures and issuer-specific presentation. | INVESTIGATION_ONLY |
| gross_profit | us-gaap:GrossProfit | PRIMARY_CANDIDATE | monetary | duration | no | CORE_IF_AVAILABLE | Primary candidate when reported, but availability may vary by issuer and statement presentation. | INVESTIGATION_ONLY |
| operating_income | us-gaap:OperatingIncomeLoss | PRIMARY_CANDIDATE | monetary | duration | no | CORE_IF_AVAILABLE | Review sign conventions and issuer-specific definitions before implementation. | INVESTIGATION_ONLY |
| net_income | us-gaap:NetIncomeLoss | PRIMARY_CANDIDATE | monetary | duration | no | CORE_WITH_ALTERNATES | Primary net income candidate, subject to review of issuer presentation and sign conventions. | INVESTIGATION_ONLY |
| net_income | us-gaap:ProfitLoss | ALTERNATE_CANDIDATE | monetary | duration | no | CORE_WITH_ALTERNATES | Alternate candidate requiring review before any runtime mapping decision. | INVESTIGATION_ONLY |
| diluted_eps | us-gaap:EarningsPerShareDiluted | PRIMARY_CANDIDATE | per-share | duration | no | REVIEW_REQUIRED | EPS uses per-share units, must not be mixed with monetary facts, and requires careful annual versus quarterly context review. | INVESTIGATION_ONLY |
| total_debt | us-gaap:DebtCurrent | DERIVED_COMPONENT | monetary | instant | yes | DERIVED_REQUIRES_RULES | Current debt component; controlled derivation rules are required to avoid double-counting and missing-component inference. | INVESTIGATION_ONLY |
| total_debt | us-gaap:LongTermDebtCurrent | DERIVED_COMPONENT | monetary | instant | yes | DERIVED_REQUIRES_RULES | Current portion of long-term debt; review overlap with other current debt concepts before derivation. | INVESTIGATION_ONLY |
| total_debt | us-gaap:LongTermDebtNoncurrent | DERIVED_COMPONENT | monetary | instant | yes | DERIVED_REQUIRES_RULES | Noncurrent long-term debt component; derivation rules must define inclusion boundaries. | INVESTIGATION_ONLY |
| total_debt | us-gaap:LongTermDebtAndFinanceLeaseObligationsCurrent | DERIVED_COMPONENT | monetary | instant | yes | DERIVED_REQUIRES_RULES | Lease-inclusive current component; do not mix with lease-exclusive debt tags without explicit rules. | INVESTIGATION_ONLY |
| total_debt | us-gaap:LongTermDebtAndFinanceLeaseObligationsNoncurrent | DERIVED_COMPONENT | monetary | instant | yes | DERIVED_REQUIRES_RULES | Lease-inclusive noncurrent component; requires review before combining with other debt concepts. | INVESTIGATION_ONLY |
| total_debt | us-gaap:ShortTermBorrowings | DERIVED_COMPONENT | monetary | instant | yes | DERIVED_REQUIRES_RULES | Short-term borrowing component; derivation rules must prevent overlap with current debt tags. | INVESTIGATION_ONLY |
| total_debt | us-gaap:FinanceLeaseLiabilityCurrent | DERIVED_COMPONENT | monetary | instant | yes | DERIVED_REQUIRES_RULES | Current finance lease liability component; lease treatment must be explicitly approved. | INVESTIGATION_ONLY |
| total_debt | us-gaap:FinanceLeaseLiabilityNoncurrent | DERIVED_COMPONENT | monetary | instant | yes | DERIVED_REQUIRES_RULES | Noncurrent finance lease liability component; lease treatment must be explicitly approved. | INVESTIGATION_ONLY |
| total_equity | us-gaap:StockholdersEquity | PRIMARY_CANDIDATE | monetary | instant | no | CORE_WITH_ALTERNATES | Primary equity candidate for corporate issuers; equity is an instant balance-sheet fact. | INVESTIGATION_ONLY |
| total_equity | us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest | ALTERNATE_CANDIDATE | monetary | instant | no | CORE_WITH_ALTERNATES | Alternate equity candidate requiring review of noncontrolling interest treatment. | INVESTIGATION_ONLY |
| total_equity | us-gaap:PartnersCapital | ALTERNATE_CANDIDATE | monetary | instant | no | CORE_WITH_ALTERNATES | Alternate candidate for partnerships and similar entities requiring special review. | INVESTIGATION_ONLY |
| free_cash_flow | us-gaap:NetCashProvidedByUsedInOperatingActivities | DERIVED_COMPONENT | monetary | duration | yes | DERIVED_REQUIRES_RULES | Operating cash flow component; free cash flow requires approved derivation rules before implementation. | INVESTIGATION_ONLY |
| free_cash_flow | us-gaap:PaymentsToAcquirePropertyPlantAndEquipment | DERIVED_COMPONENT | monetary | duration | yes | DERIVED_REQUIRES_RULES | Capital expenditure component; signs must be reviewed carefully and no formula is implemented in SEC-4C. | INVESTIGATION_ONLY |

## Income Statement Field Recommendations

- revenue: Use a reviewed primary revenue candidate with alternate candidates available for issuer-specific reporting patterns.
- gross_profit: Treat as core when available, with review required for issuers that do not report gross profit consistently.
- operating_income: Use operating income or loss as the primary candidate after sign convention and definition review.
- net_income: Use net income or loss as the primary candidate, with profit or loss retained as a reviewed alternate.

## Balance Sheet, EPS, and Cash-Flow Field Recommendations

- diluted_eps: Treat diluted EPS as a reviewed per-share duration fact and keep it separate from monetary mappings.
- total_debt: Treat total debt as a derived field that requires approved component and double-counting rules.
- total_equity: Use stockholders' equity as the primary candidate with alternates reviewed for noncontrolling interest or entity type.
- free_cash_flow: Treat free cash flow as a derived field requiring approved operating cash flow and capital expenditure rules.

## SEC-4D Handoff

SEC-4D should cover:

- derived-field policy;
- approved derivation formulas;
- duplicate/amended fact handling;
- unit and period conflict policy;
- SEC-5 analysis-rationalization handoff.

## No-Runtime-Change Confirmation

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
- no scraping performed.

SEC-4B changed documentation only and did not change code, tests, data, generated files, SEC access, pipeline behavior, or downstream runtime behavior.

SEC-4C changed documentation only and did not change code, tests, data, generated files, SEC access, pipeline behavior, or downstream runtime behavior.
