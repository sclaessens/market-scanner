# Operational Sprint Planning — Approved Fundamental Data Source

## 1. Status and Scope

This document is a documentation-only PM / Functional Analyst / Technical Analyst / Governance planning artifact for the approved real fundamental data source workstream.

This is the ChatGPT-owned planning step before any Codex implementation work.

This document prepares the scope, governance boundaries, candidate-source evaluation criteria, data-contract expectations, implementation handoff, and backlog reconciliation for a future implementation sprint.

This document does not implement:

- code;
- tests;
- CSV files;
- generated artifacts;
- reports;
- workflows;
- provider integration;
- provider/API calls;
- scraping;
- credentials or secrets;
- runtime orchestration;
- daily ingestion;
- backtesting code;
- Reporting changes;
- Telegram changes;
- scanner changes;
- Decision Engine changes;
- portfolio files;
- watchlist files;
- runtime behavior changes.

No sprint is closed or certified complete by this document.

No fundamental data source is approved by this document.

No provider/API integration is authorized by this document.

No data collection is authorized by this document.

## 2. Background

The current pipeline can run end-to-end technically, but final decisions remain constrained because the Fundamental Layer lacks an approved real fundamental data source.

Prior investigation found that `data/processed/fundamental_quality.csv` contained 291 rows where all rows had insufficient fundamental data states, including source-missing metadata and source-missing source-data status.

This means the Fundamental Layer can preserve rows and classify absence of data, but it cannot yet provide meaningful source-supported fundamental quality metadata.

The relevant backlog items are:

- `BL-0015 — Define and implement approved Fundamental data source and quality classification contract`;
- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`.

The recently closed analyst-expectations provider strategy concluded that analyst expectations provider execution should be paused and that the project should refocus on higher-priority source-data foundations, especially approved real fundamental data sourcing.

## 3. Objective

The objective of the next governed workstream is to define and prepare implementation of an approved real fundamental data source path that allows the Fundamental Layer to classify source-supported fundamental data without introducing decision authority upstream.

The workstream must answer:

- which source or source-artifact path is allowed for real fundamentals;
- which fields are required for the Fundamental Layer to classify data quality;
- how source provenance and freshness are represented;
- how missing, partial, stale, and unsupported data are classified;
- how provider/API, manual CSV, or hybrid source paths are governed;
- what Codex may implement later;
- which tests and validation controls Codex must include;
- how the work remains descriptive upstream and Decision Engine-authoritative downstream.

## 4. Research-Only and Classification-Only Boundary

The Fundamental Layer remains classification-only and descriptive.

Fundamental data must not become upstream buy/sell advice.

Fundamental data must not create or imply:

- ranking authority;
- scoring authority outside approved classification semantics;
- allocation authority;
- tradeability;
- urgency;
- conviction;
- eligibility;
- hidden filtering;
- Reporting recommendations;
- Telegram recommendations;
- Decision Engine bypass.

Any future use of fundamental classifications by the Decision Engine must occur only through the already governed Decision Engine authority boundary.

The upstream Fundamental Layer may classify data state and quality, but it must not decide actions.

## 5. Candidate Source Paths

No source path is approved by this document.

The future design should compare the following candidate paths.

| Candidate path | Description | Potential advantage | Key governance concern |
|---|---|---|---|
| Manual governed CSV source artifact | A human-maintained raw fundamentals CSV with explicit source, freshness, and field definitions. | Simple, auditable, low implementation complexity, no credentials required. | Manual drift, freshness discipline, and limited scalability. |
| Provider-assisted local export | A provider or platform export transformed into a governed raw artifact. | More structured than manual entry while avoiding direct runtime API integration. | Terms, export rights, source provenance, and repeatability. |
| API/provider integration | Automated provider API or package-based access for fundamental fields. | Better refresh potential and lower manual burden. | Credentials, rate limits, licensing, nondeterminism, failure handling, caching, tests, and operational governance. |
| Hybrid MVP | Start with a governed manual CSV contract and later upgrade to provider-assisted ingestion. | Allows immediate source-supported classification without premature provider complexity. | Must avoid locking in weak schema or manual-only assumptions. |

Planning recommendation: prefer a hybrid MVP path unless a provider is already clearly approved.

## 6. Minimum Required Fundamental Source Fields

The future source contract should define a minimal field set before implementation.

Candidate fields:

| Field | Purpose |
|---|---|
| `ticker` | Required row identity key. |
| `as_of_date` | Date the fundamental values represent. |
| `source_name` | Human-readable source name or provider name. |
| `source_reference` | URL, export reference, provider reference, or audit reference. |
| `source_freshness_date` | Date the source was last updated or confirmed. |
| `currency` | Currency for monetary fields where relevant. |
| `revenue_growth_yoy` | Candidate growth metric, if source-supported. |
| `eps_growth_yoy` | Candidate profitability-growth metric, if source-supported. |
| `gross_margin` | Candidate profitability metric, if source-supported. |
| `operating_margin` | Candidate operating-quality metric, if source-supported. |
| `net_margin` | Candidate profitability metric, if source-supported. |
| `free_cash_flow_margin` | Candidate cash-generation metric, if source-supported. |
| `debt_to_equity` | Candidate balance-sheet risk metric, if source-supported. |
| `current_ratio` | Candidate liquidity metric, if source-supported. |
| `return_on_equity` | Candidate capital-efficiency metric, if source-supported. |
| `return_on_invested_capital` | Candidate quality metric, if source-supported. |
| `fundamental_notes` | Human-readable notes about limitations, missing data, source caveats, or manual review. |

The final field list must be confirmed by a developer specification before implementation.

## 7. Fundamental Quality Classification Contract

The future Fundamental Layer should classify source status without introducing decision authority.

Candidate output states:

| State | Meaning |
|---|---|
| `SUFFICIENT_DATA` | Required source fields are present, fresh enough, and internally coherent for descriptive classification. |
| `PARTIAL_DATA` | Some source fields are present but key required fields are missing. |
| `STALE_DATA` | Source data exists but freshness rules are not satisfied. |
| `INSUFFICIENT_DATA` | Source data is missing, unsupported, or too incomplete for classification. |
| `SOURCE_UNSUPPORTED` | Ticker/security cannot currently be supported by the approved source path. |
| `SOURCE_ERROR` | Source artifact exists but cannot be parsed or validated. |

Candidate metadata fields:

| Field | Purpose |
|---|---|
| `quality_state` | Descriptive quality state. |
| `quality_reason` | Human-readable classification reason. |
| `quality_metadata_status` | Metadata completeness status. |
| `source_data_status` | Source data availability status. |
| `source_freshness_status` | Freshness status. |
| `missing_fundamentals_count` | Count of missing required fields. |
| `partial_data_count` | Count or flag for partial source support. |
| `stale_data_count` | Count or flag for stale source support. |

These names must be reconciled with the existing implementation before Codex changes code.

## 8. Source Evaluation Criteria

Before approving a source or source path, the project should evaluate:

| Criterion | Required assessment |
|---|---|
| Coverage | Does the source cover the scanner universe and portfolio tickers? |
| Field availability | Are required fundamental fields available consistently? |
| Definition clarity | Are metrics defined clearly enough to compare across tickers? |
| Freshness | Is update date or fiscal period visible and reliable? |
| Source provenance | Can every value be traced to a source reference? |
| Licensing / terms | Is manual storage, export use, or API use permitted? |
| Access requirements | Does the source require account, credentials, subscription, or API key? |
| Rate limits | Are access limits compatible with the project? |
| Stability | Is the source stable enough for repeated operation? |
| Auditability | Can raw and normalized values be inspected or reconstructed? |
| Failure modes | What happens when data is missing, stale, malformed, or unavailable? |
| Automation suitability | Can the path later be automated safely? |

## 9. Recommended Sprint Shape

Recommended sprint name:

`Operational Sprint — Approved Fundamental Data Source MVP`

Recommended scope:

1. Confirm the approved source path.
2. Define the raw source artifact or provider boundary.
3. Define required input fields and validation rules.
4. Define source freshness and provenance rules.
5. Define missing, partial, stale, unsupported, and error states.
6. Produce a developer specification for Codex implementation.
7. Preserve Fundamental Layer classification-only authority.
8. Preserve Decision Engine as the only allocation authority.
9. Validate output row preservation and no hidden filtering.
10. Confirm that Reporting and Telegram remain communication-only.

Recommended implementation sequence for Codex after governance approval:

1. Add or update raw source contract only after source path approval.
2. Update fundamental builder to read approved source artifact or provider wrapper.
3. Add fail-fast validation for required columns and row identity.
4. Preserve all upstream opportunity rows.
5. Emit descriptive quality states only.
6. Add focused tests for source present, source missing, partial, stale, malformed, and unsupported cases.
7. Run full test suite.
8. Run full pipeline locally.
9. Confirm final decision behavior remains Decision Engine-controlled.

## 10. Codex Handoff Boundaries

Codex may be asked later to implement only after a separate approved developer specification exists.

Codex implementation must not:

- invent a source without approval;
- infer missing fundamentals from price action;
- scrape websites without explicit approval;
- create credentials or secrets without governance;
- introduce ranking, conviction, urgency, tradeability, or allocation upstream;
- loosen Decision Engine authority;
- modify Reporting or Telegram to infer missing fundamentals;
- change portfolio files unless explicitly scoped;
- write generated CSV outputs as source-of-truth manually;
- bypass tests.

Codex implementation should be constrained to code, tests, data contract validation, and controlled runtime verification only after governance planning is complete.

## 11. Acceptance Criteria for Future Implementation

A future implementation sprint should be accepted only if:

- an approved source path is documented;
- required input fields are defined;
- source provenance is preserved;
- freshness rules are implemented;
- missing and partial data are classified explicitly;
- malformed source artifacts fail fast;
- all upstream rows are preserved;
- no upstream allocation or tradeability semantics are introduced;
- `fundamental_quality.csv` no longer classifies all rows as source missing when valid source data is provided;
- tests cover source-present and source-problem scenarios;
- full pipeline execution succeeds;
- final decisions remain under Decision Engine authority;
- Reporting and Telegram remain communication-only.

## 12. Risks and Controls

| Risk | Control |
|---|---|
| Provider data creates hidden scoring pressure | Keep Fundamental Layer classification-only and route any action use through Decision Engine governance. |
| Manual CSV becomes stale | Require source freshness fields and stale classification. |
| Missing data is silently filled | Prohibit silent inference and classify missing fields explicitly. |
| API access introduces credential risk | Do not approve credentials until separate secrets governance exists. |
| Source terms prohibit storage or automation | Require licensing/terms review before source approval. |
| Row loss occurs during merge | Require row-preserving tests and duplicate-key validation. |
| Reporting displays fundamentals as recommendations | Keep Reporting communication-only and source-supported. |

## 13. Backlog Impact Assessment

Existing backlog items `BL-0015` and `BL-0017` remain sufficient.

`BL-0015` captures the approved Fundamental data source and quality classification contract.

`BL-0017` captures the governed automated or provider-assisted ingestion strategy for fundamentals and portfolio metadata.

This planning document prepares those backlog items for future sprint planning but does not identify additional deferred work.

Backlog impact assessment:
- No new backlog items identified.

## 14. Recommended Next Step

The recommended next step is a documentation-only developer-spec preparation prompt for Codex.

That prompt should ask Codex to create the implementation plan and developer specification only, not to implement code yet.

After that, a separate Codex implementation sprint can be launched if the source path and developer specification are approved.
