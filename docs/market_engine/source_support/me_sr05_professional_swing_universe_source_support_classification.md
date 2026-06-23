# ME-SR05 - Professional Swing Universe source-support classification

Sprint: ME-SR05 - Classify source support for Professional Swing Universe

Job family: ME-SR - Source Refresh / Source Coverage

Status: Implemented

## Purpose

ME-SR05 implements deterministic source-support classification for the editable Professional Swing Universe.

The classifier answers whether each Professional Swing Universe row is currently supported by approved local SEC CompanyFacts source artifacts that already exist in the Market Engine data flow.

It does not fetch, refresh, infer, rank, recommend, report, or execute anything.

## Public API

Runtime module:

```text
src/market_engine/source_support/professional_swing.py
```

Primary API:

```text
classify_professional_swing_universe_source_support(...)
```

Output format:

```text
market-engine-professional-swing-source-support-v1
```

Default inputs:

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
data/market_engine/source_snapshots
```

The function also accepts explicit path overrides for tests and future local runtime integration.

## Classification Inputs

The classifier consumes:

* the validated editable Professional Swing Universe loader output from ME-UNI06;
* local SEC CompanyFacts raw snapshot envelopes;
* local SEC CompanyFacts provider error records when present;
* the approved SEC CompanyFacts field mapping for:
  * `revenue`;
  * `net_income`;
  * `operating_cash_flow`;
  * `capital_expenditures`.

The classifier does not treat the editable-universe `source_policy_hint` column as authoritative source truth. That column remains operator metadata only.

## Statuses Implemented

The classifier emits explicit status values:

```text
supported_cached
missing_snapshot
unsupported_sec_companyfacts
missing_required_source_field
malformed_or_unreadable_source_artifact
ambiguous_identity
manual_review_only
excluded
```

No staleness status is emitted in ME-SR05 because the current Professional Swing Universe source-support contract does not define a freshness threshold. A later sprint may add freshness policy if needed.

## Source Evidence

Each ticker-level classification preserves:

* ticker;
* Professional Swing Universe row reference;
* active status;
* universe status;
* source policy hint;
* operator priority;
* source artifact references;
* provider error references when available;
* required mapped source fields;
* missing required source fields;
* selected SEC tag, taxonomy namespace, unit, filing form, fiscal year, period end date, accession number, and source value when available.

Numeric zero source values are preserved as valid observed values and are not treated as missing.

## Fail-Closed Behavior

The classifier fails closed when:

* the Professional Swing Universe CSV is invalid;
* a matching local snapshot cannot be read safely;
* required mapped fields are absent;
* multiple local snapshots create ambiguous ticker identity.

Missing or malformed artifacts are represented explicitly. They are not converted into supported rows.

## Non-Scope

ME-SR05 does not add:

* provider calls;
* SEC or EDGAR live access;
* yfinance or external data access;
* source refresh;
* synthetic source facts;
* cached-source execution;
* production data writes;
* Telegram or email delivery;
* portfolio or watchlist mutation;
* reporting output;
* Recommendation Review behavior;
* Portfolio Review behavior;
* Decision Engine behavior;
* BUY / SELL / HOLD semantics;
* allocation, target price, ranking, scoring, urgency, conviction, tradeability, position sizing, order, or execution behavior.

## Relationship To ME-RUN20

ME-SR05 creates the source-support classification layer needed before ME-RUN20.

ME-RUN20 may consume the `supported_cached` subset for a clean local cached-source scan, while preserving the blocked and unsupported classifications for operator review.
