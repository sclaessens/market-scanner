# Analyst Expectations Source Comparison Matrix

## 1. Status and Scope

This document is a documentation-only source comparison matrix for analyst expectations research.

It follows `docs/research/analyst_expectations_source_policy_and_validation_design.md` and supports backlog item `BL-0018 — Define governed analyst expectations and historical validation research strategy`.

This document compares candidate source categories and source-evaluation dimensions only.

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
- fundamentals files;
- runtime behavior changes.

No sprint is closed or certified complete by this document.

No source is approved by this document.

No data collection is authorized by this document.

## 2. Purpose

The purpose of this matrix is to prepare a controlled source-selection discussion before analyst expectations data is collected or automated.

The matrix helps compare candidate source categories against the governed source-policy criteria already defined in `docs/research/analyst_expectations_source_policy_and_validation_design.md`.

The goal is to identify which source categories are likely suitable for:

- research-only analyst expectation review;
- historical point-in-time validation;
- source provenance and freshness tracking;
- future exception-based human review;
- later governed automation proposals.

The matrix is not a provider selection decision.

The matrix is not a license review.

The matrix is not a runtime integration plan.

## 3. Research-Only Boundary

Analyst expectations remain research-only.

Analyst expectations must not become buy/sell advice.

Source comparison must not create or imply:

- ranking authority;
- scoring authority;
- allocation authority;
- tradeability;
- urgency;
- conviction;
- eligibility;
- hidden filtering;
- Reporting recommendations;
- Telegram recommendations;
- Decision Engine bypass.

Candidate source strength, weakness, or suitability labels in this document are research planning labels only. They do not classify securities, opportunities, or portfolio actions.

Future Decision Engine use requires separate governance, separate approval, explicit design, tests, and audit controls.

## 4. Candidate Source Categories

The following source categories are candidates pending evaluation.

They are not approved sources.

| Candidate source category | Candidate role | Potential usefulness | Key limitations | Current approval status |
|---|---|---|---|---|
| Official company investor relations materials | Company-supported context for guidance, earnings materials, presentations, and management outlook. | Useful for official company context, financial disclosure interpretation, and management guidance review. | Usually not independent analyst consensus; may not provide comparable rating or price-target fields; may be narrative rather than structured. | Candidate only; not approved. |
| SEC filings or equivalent regulatory filings | Official regulatory disclosure source for reported financials, risks, management discussion, and historical disclosures. | Useful for audit-backed company history and reported fundamentals context. | Generally not a direct analyst expectations source; may not contain price targets, analyst counts, or estimate consensus. | Candidate only; not approved. |
| Exchange or market-profile pages | Market-profile source for company/ticker metadata, summaries, and sometimes analyst-related fields. | May offer stable market context and potentially visible consensus fields for some tickers. | Coverage, field definitions, licensing, point-in-time support, and URL stability must be validated. | Candidate only; not approved. |
| Market data platforms publishing analyst ratings, price targets, or estimates | Direct candidate source for analyst consensus, rating distributions, price targets, EPS estimates, and revenue estimates. | Likely most relevant for current analyst expectations research if definitions, terms, and coverage are suitable. | Current-page values may not be point-in-time historical truth; licensing and automation constraints may be significant. | Candidate only; not approved. |
| Paid, licensed, or API-based providers | Structured data source candidate for analyst expectations and possible automated access if later approved. | May provide cleaner fields, identifiers, rate-limit clarity, and historical access depending on provider. | Requires provider governance, licensing review, credentials/secrets policy, cost review, and automation approval. | Candidate only; not approved. |
| Historical point-in-time data providers | Candidate source for historical reconstruction and look-ahead-safe validation. | Most relevant for valid historical validation if they preserve analyst expectations as known on past dates. | May be expensive, licensed, incomplete, or complex to integrate; must prove point-in-time semantics. | Candidate only; not approved. |

## 5. Comparison Criteria Matrix

This matrix defines what future source comparison should evaluate.

It intentionally does not score actual providers.

| Criterion | Official investor relations | Regulatory filings | Exchange or market-profile pages | Market data platforms | Paid/API providers | Historical point-in-time providers |
|---|---|---|---|---|---|---|
| Coverage by ticker | Evaluate whether each portfolio/scanner ticker has company materials available. | Evaluate jurisdiction and filing availability by ticker. | Evaluate ticker coverage across regions and exchanges. | Evaluate analyst field coverage by ticker. | Evaluate contractual universe and ticker mapping. | Evaluate historical ticker coverage and delisted/security-change support. |
| Field availability | Likely guidance and narrative context; analyst consensus unlikely. | Reported financials and disclosures; analyst fields unlikely. | May include selected summary fields. | May include consensus rating, analyst count, price targets, EPS/revenue estimates. | May include structured fields, metadata, and provider-specific identifiers. | Must include historical snapshots or point-in-time values. |
| Definition clarity | Guidance definitions may be company-specific. | Filing definitions are formal but not analyst-consensus oriented. | Must confirm source definitions and calculation origin. | Must confirm rating categories, target definitions, estimate windows, and revision logic. | Must document provider field definitions and calculation rules. | Must document historical availability and revision/versioning semantics. |
| Historical point-in-time support | Materials can be dated, but analyst consensus may be absent. | Strong for filed documents, weak for analyst expectations. | Often unclear unless archived or provider-supported. | Often weak if only current web pages are available. | Depends on provider. | Core requirement and primary evaluation focus. |
| Freshness | Usually clear for releases and presentations. | Filing dates are clear. | Must verify source freshness date. | Must distinguish source freshness from page access date. | Must expose freshness metadata. | Must distinguish original as-of date, revision date, and collection date. |
| Update frequency | Event-driven. | Event-driven and regulatory-calendar driven. | Varies. | Varies by platform and analyst update activity. | Provider-defined. | Provider-defined and historically versioned. |
| Licensing and terms of use | Must confirm permitted storage and quotation. | Public filings are accessible but reuse rules still require review. | Must review site terms. | Must review site terms and redistribution limits. | Requires formal licensing review. | Requires formal licensing review. |
| Rate limits | Usually manual access only. | Usually source-specific. | Must evaluate. | Must evaluate. | Provider-defined. | Provider-defined. |
| Stable references/URLs | Investor pages may change; document URLs may persist or move. | Filing references are often stable. | URLs may change. | URLs may change; provider identifiers may be better. | Provider identifiers may be stable. | Provider identifiers and snapshot IDs should be required. |
| Consistency across tickers | Company-specific and inconsistent. | Structured by jurisdiction but not uniform globally. | May vary by exchange and region. | Must test field consistency. | Contract-dependent. | Contract-dependent and essential for validation. |
| Conflict handling | Must separate company guidance from analyst views. | Must separate official reported facts from expectations. | Must reconcile with data platform values if used. | Must compare across platforms if multiple sources are evaluated. | Must define provider precedence. | Must preserve historical conflicts rather than overwrite. |
| Auditability | Good if source documents are preserved as references. | Strong if filing references are stable. | Depends on stable references and page versioning. | Depends on source references, timestamps, and terms. | Stronger if provider supports metadata and audit logs. | Essential; must preserve snapshot provenance. |
| Suitability for automation | Low until separate governance approves scraping or source access. | Possible for filings only after separate governance. | Unclear and usually requires terms review. | Unclear; public pages may not be suitable. | Possible only after provider and credential governance. | Possible only after provider and research-pipeline governance. |

## 6. Preliminary Suitability View

This section provides a planning-level suitability view.

It does not approve sources and does not assign securities, rankings, scores, or investment actions.

| Candidate source category | Current research suitability | Historical validation suitability | Automation suitability | Notes |
|---|---|---|---|---|
| Official company investor relations materials | Medium for context; low for analyst consensus. | Medium for company disclosures; low for analyst consensus. | Low unless future governance defines document handling. | Useful context source, but not sufficient as an analyst consensus source. |
| SEC filings or equivalent regulatory filings | Medium for official disclosure context; low for analyst consensus. | High for filed historical facts; low for analyst expectation history. | Medium only for filings workflows if separately approved. | Useful validation context, not a direct analyst expectations source. |
| Exchange or market-profile pages | Medium if analyst fields are present and defined. | Low unless point-in-time or archived data is available. | Low until terms, rate limits, and access stability are approved. | Candidate for current snapshot comparison only, pending terms and definitions. |
| Market data platforms publishing analyst ratings, price targets, or estimates | High for current analyst expectations research if definitions and terms are acceptable. | Low to medium unless historical point-in-time data is available. | Low until provider/API or access governance is approved. | Most likely current analyst expectations source category, but not enough for look-ahead-safe backtesting by itself. |
| Paid, licensed, or API-based providers | High if fields, coverage, and licensing are acceptable. | Medium to high depending on historical support. | Medium to high after separate governance. | Requires provider selection, cost, credential, secrets, and usage-policy review. |
| Historical point-in-time data providers | Medium for current review; high for historical validation. | High if true point-in-time analyst estimate snapshots are available. | Medium after provider and research-pipeline governance. | Most relevant for proper backtesting and look-ahead-bias prevention. |

## 7. Minimum Approval Questions Before Any Source Is Used

Before any source is used for analyst expectations research collection, the project should answer the following questions.

1. What exact fields are available?
2. Are the source definitions clear enough to compare across tickers?
3. Does the source publish source freshness or estimate dates?
4. Does the source support historical point-in-time reconstruction?
5. Are current web pages being incorrectly treated as historical truth?
6. Are licensing and terms of use compatible with the intended research use?
7. Is storage of raw and normalized values permitted?
8. Are stable source references or provider identifiers available?
9. Are analyst counts and rating category mappings visible and auditable?
10. Are price-target and rating consensus definitions separate or blended?
11. Are EPS and revenue estimate periods clearly defined?
12. How are stale, missing, and conflicting values handled?
13. Can the operator review only exceptions and samples rather than every ticker?
14. Would automation require credentials, secrets, caching, or rate-limit controls?
15. Does any proposed use risk Decision Engine bypass or hidden recommendation semantics?

## 8. Candidate Field Coverage Checklist

A future source review should record whether each candidate field is available, unavailable, unclear, or not applicable.

| Candidate field | Required source review |
|---|---|
| `ticker` | Confirm ticker mapping and exchange handling. |
| `as_of_date` | Confirm whether the source value has a valid as-of date. |
| `collection_date` | Confirm how project collection date would be recorded later. |
| `source_name` | Confirm provider/source naming convention. |
| `source_url_or_reference` | Confirm stable URL, filing reference, provider ID, or audit reference. |
| `source_freshness_date` | Confirm whether source update date exists and what it means. |
| `consensus_rating` | Confirm whether this is source-published or would require internal calculation. |
| `analyst_count` | Confirm whether analyst count is available for each consensus value. |
| `buy_count` | Confirm category definition and whether buy-equivalent mappings are required. |
| `hold_count` | Confirm category definition and whether neutral/hold mappings are required. |
| `sell_count` | Confirm category definition and whether sell-equivalent mappings are required. |
| `average_price_target` | Confirm currency, time horizon, and calculation definition. |
| `low_price_target` | Confirm source availability and outlier handling. |
| `high_price_target` | Confirm source availability and outlier handling. |
| `current_price_at_collection` | Confirm whether this is source-provided or must come from a separate approved price source. |
| `implied_upside_pct` | Confirm whether source-provided or internally calculated; internal formula requires later approval. |
| `current_year_eps_estimate` | Confirm fiscal-year definition and estimate date. |
| `next_year_eps_estimate` | Confirm fiscal-year definition and estimate date. |
| `current_year_revenue_estimate` | Confirm fiscal-year definition and currency. |
| `next_year_revenue_estimate` | Confirm fiscal-year definition and currency. |
| `estimate_revision_direction` | Confirm source definition and lookback period. |
| `raw_source_value` | Confirm whether raw source values can be stored. |
| `normalized_value` | Confirm whether normalization is necessary and separately approved. |
| `normalization_note` | Confirm required explanation for any mapping or transformation. |
| `data_quality_state` | Confirm future quality-state definitions before use. |
| `data_quality_notes` | Confirm exception notes and review workflow. |

## 9. Source Review Output Template

A future documentation-only source review may use the following template without collecting runtime data.

| Field | Value |
|---|---|
| Candidate source name | TBD |
| Source category | TBD |
| Access method reviewed | Public documentation / public page description / provider documentation / other, without runtime collection |
| Analyst ratings available | Unknown / yes / no / partial |
| Price targets available | Unknown / yes / no / partial |
| EPS estimates available | Unknown / yes / no / partial |
| Revenue estimates available | Unknown / yes / no / partial |
| Analyst count available | Unknown / yes / no / partial |
| Source freshness available | Unknown / yes / no / partial |
| Point-in-time support | Unknown / yes / no / partial |
| Licensing review needed | Yes |
| Automation review needed | Yes if proposed beyond manual documentation review |
| Credential/secrets review needed | Yes if API or paid provider is proposed |
| Conflict-handling review needed | Yes |
| Research-only status | Required |
| Approved for runtime use | No |
| Approved for Decision Engine use | No |
| Approved for Reporting or Telegram use | No |
| Recommended next governance action | TBD |

## 10. Human Review Model

The operator should not review every ticker.

Human review should remain exception-based and should focus on:

- source-policy approval;
- provider or source category approval;
- field-definition approval;
- licensing and terms-of-use review;
- point-in-time capability review;
- conflict review;
- missing-data and stale-data exception review;
- limited sampling;
- future automation proposal approval.

Manual review should not become a permanent ticker-by-ticker operating model.

## 11. Not Approved by This Matrix

This matrix does not approve:

- any source or provider;
- runtime data collection;
- manual ticker-by-ticker analyst data collection;
- scraping;
- provider/API calls;
- credentials or secrets;
- automated ingestion;
- CSV creation;
- historical backtesting code;
- source-derived scoring;
- source-derived ranking;
- source-derived tradeability;
- source-derived conviction;
- source-derived urgency;
- source-derived eligibility;
- hidden filtering;
- Decision Engine integration;
- Reporting recommendations;
- Telegram recommendations.

## 12. Backlog Impact Assessment

Existing backlog item `BL-0018` remains sufficient.

This document executes the recommended documentation-only source comparison step already implied by the source-policy and validation-design document.

It does not identify additional deferred work beyond the existing governed analyst expectations research path.

Backlog impact assessment:
- No new backlog items identified.

## 13. Recommended Next Step

The recommended next step is a documentation-only shortlist review of named candidate sources.

That future shortlist review should:

- compare named candidate sources against this matrix;
- rely on public documentation, terms summaries, or provider documentation review only;
- avoid runtime data collection;
- avoid API calls;
- avoid scraping;
- avoid credentials or secrets;
- avoid CSV creation;
- avoid backtesting code;
- avoid runtime integration;
- keep all analyst expectations work research-only.

If any named source appears promising, a later governance step should define whether limited manual sample collection is permitted and under what audit controls.
