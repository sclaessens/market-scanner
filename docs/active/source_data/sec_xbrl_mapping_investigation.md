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

## Income Statement Field Recommendations

- revenue: Use a reviewed primary revenue candidate with alternate candidates available for issuer-specific reporting patterns.
- gross_profit: Treat as core when available, with review required for issuers that do not report gross profit consistently.
- operating_income: Use operating income or loss as the primary candidate after sign convention and definition review.
- net_income: Use net income or loss as the primary candidate, with profit or loss retained as a reviewed alternate.

## SEC-4C Handoff

SEC-4C should cover:

- diluted_eps
- total_debt
- total_equity
- free_cash_flow

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
