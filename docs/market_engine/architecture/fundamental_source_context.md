# Market Engine Fundamental Source Context

Owner role: Technical Architect / Financial Analyst / Data Steward / QA / Test Lead

Status: ME11 IMPLEMENTED

## Purpose

This document describes the first Market Engine fundamental source context created by ME11.

The context converts approved SEC CompanyFacts source evidence into canonical source fields with provenance and source-readiness status. It is a source-only context. It is not financial analysis.

## Source-Only Boundary

The fundamental source context may expose:

- ticker;
- provider;
- source readiness;
- canonical source field values;
- missing canonical fields;
- provider error category and message when applicable;
- period metadata;
- provenance for selected SEC facts.

The context must not emit:

- free cash flow;
- growth;
- margins;
- valuation metrics;
- scores;
- rankings;
- recommendations;
- BUY / SELL / HOLD;
- allocation;
- conviction;
- urgency;
- tradeability;
- position sizing;
- execution advice.

## Canonical Fields

ME11 implements the ME10-approved canonical SEC fields:

- `revenue`
- `net_income`
- `operating_cash_flow`
- `capital_expenditures`

These fields are source coverage fields only. They do not authorize analysis conclusions.

## Readiness States

Allowed readiness states:

- `AVAILABLE`: all four approved canonical fields are present.
- `PARTIAL`: at least one approved canonical field is present and at least one is missing.
- `MISSING`: no approved canonical fields are present.
- `UNSUPPORTED`: the ticker is outside supported source coverage or lacks supported CIK mapping.
- `INVALID_TICKER`: ticker input is malformed.
- `PROVIDER_ERROR`: controlled provider, network, HTTP, JSON, or request failure.

## Provenance Requirements

For each available canonical field, provenance preserves:

- canonical field name;
- selected SEC tag;
- provider name;
- taxonomy namespace;
- unit;
- raw value;
- fiscal year;
- fiscal period;
- filing form;
- filing date;
- period start date;
- period end date;
- accession number;
- frame when available;
- selection reason;
- fallback alias when used.

Missing values remain missing and are not converted to zero, false, estimates, derived values, or previous-period fallbacks.

## Relation To Source Intake

Source intake remains responsible for explicit provider access, ticker-level failure isolation, readiness summaries, and bounded smoke behavior.

The fundamental source context consumes already-fetched SEC CompanyFacts evidence. It does not perform live provider calls by itself.

## Future Analysis Boundary

ME11 does not approve analysis. A later sprint may use this source context to build a non-decision analysis pass only after the source context, missing-data behavior, provenance, and testing boundaries remain intact.
