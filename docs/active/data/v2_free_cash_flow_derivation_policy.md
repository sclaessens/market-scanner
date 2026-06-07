# V2 FreeCashFlow Derivation or Missingness Policy

Status: ACTIVE
Reset stage: RESET-10L-BL21

## Purpose

This document governs how the v2 fundamentals layer may handle `FreeCashFlow`
when a direct source field is missing.

This policy selects option C:

```text
Support both directly sourced FreeCashFlow and governed derived FreeCashFlow,
while preserving provenance, derivation status, validation conditions, and
explicit missingness.
```

This policy exists because the first NVDA real fundamental analysis review
failed closed into `LIMITED_ANALYSIS` with `CASH_FLOW_UNKNOWN` and
`REVIEW_DATA_LIMITATION` when `FreeCashFlow` was not provided as a direct source
field.

The policy allows future implementation to derive `FreeCashFlow` only under
strict conditions. It does not authorize silent inference, missing-to-zero
conversion, final investment recommendations, reports, Telegram delivery,
portfolio/watchlist updates, or production pipeline behavior.

## Decision

Decision: approve governed FreeCashFlow derivation as an allowed v2 fundamentals
normalization behavior.

The governed derivation formula is:

```text
FreeCashFlow = NetCashProvidedByUsedInOperatingActivities - CapitalExpenditures
```

Equivalent source field naming may be accepted only if the existing provider and
normalization contracts clearly map the source fields to:

```text
operating_cash_flow
capital_expenditures
```

This derived value must be treated as a derived normalized metric, not as a
source-reported value.

## Allowed FreeCashFlow States

A normalized `free_cash_flow` value may have one of these states:

```text
source_reported
source_derived
missing
invalid
not_parseable
not_derivable
```

### source_reported

Use when a valid direct `FreeCashFlow` source field is present.

Requirements:

- source field is present;
- value is valid and parseable;
- currency and unit are known;
- reported period is known;
- provenance references the direct source field.

### source_derived

Use when direct `FreeCashFlow` is missing but the governed derivation inputs are
valid.

Requirements:

- direct `FreeCashFlow` is absent or explicitly missing;
- operating cash flow is present and valid;
- capital expenditures is present and valid;
- both values use the same currency;
- both values use the same unit;
- both values belong to the same reported period;
- both values share compatible fiscal year and fiscal quarter context;
- provenance references both source input fields;
- derivation formula is recorded;
- derivation status is recorded;
- validation warnings include that the value is derived.

### missing

Use when direct `FreeCashFlow` is absent and one or more derivation inputs are
also absent.

Requirements:

- missing fields remain explicit;
- no fallback to zero;
- readiness reflects incomplete cash-flow data.

### invalid

Use when one or more fields are present but invalid.

Requirements:

- invalid fields remain visible;
- no derived value is produced;
- readiness reflects invalid cash-flow data.

### not_parseable

Use when one or more input values cannot be parsed safely.

Requirements:

- parse failure is recorded;
- no derived value is produced;
- readiness reflects parse failure.

### not_derivable

Use when input fields exist but cannot be safely combined because of mismatch.

Examples:

- currency mismatch;
- unit mismatch;
- period mismatch;
- fiscal context mismatch;
- ambiguous sign convention;
- missing provenance.

## Required Provenance

A derived `free_cash_flow` record must preserve provenance for both input fields:

```text
NetCashProvidedByUsedInOperatingActivities
CapitalExpenditures
```

The normalized record must show:

- the metric is derived;
- the derivation formula used;
- the source fields used;
- the source reference or raw evidence identifier;
- reported period;
- fiscal year and quarter where applicable;
- currency;
- unit;
- validation warnings.

A derived metric must not be indistinguishable from a directly sourced metric.

## Sign Convention

The implementation must verify the sign convention for capital expenditures.

If capital expenditures is already represented as a negative cash outflow, the
implementation must not subtract it again unless the source contract explicitly
normalizes it as a positive outflow amount.

Future implementation must document the chosen convention in tests.

If sign convention is ambiguous, derivation must fail closed as:

```text
not_derivable
```

## Missing-Value Rule

Missing values must remain explicit.

Forbidden substitutions:

```text
0
0.0
"0"
False
""
```

The system must not compute FreeCashFlow when either required input is missing.

The system must not assume capital expenditures is zero when absent.

The system must not assume operating cash flow is zero when absent.

## Readiness Behavior

Readiness may improve from `partial` only if a derived FreeCashFlow value is
validly produced and all required provenance and validation conditions are met.

If FreeCashFlow remains missing, invalid, not parseable, or not derivable,
readiness must continue to reflect a cash-flow limitation.

Suggested readiness warnings:

```text
free_cash_flow:source_reported
free_cash_flow:source_derived
free_cash_flow:missing_required_input
free_cash_flow:invalid_required_input
free_cash_flow:not_parseable
free_cash_flow:not_derivable
free_cash_flow:period_mismatch
free_cash_flow:currency_mismatch
free_cash_flow:unit_mismatch
free_cash_flow:sign_convention_ambiguous
```

## Analysis Behavior

A derived FreeCashFlow value may be used by future analysis only if:

- it is validly derived;
- provenance is complete;
- derivation status is visible;
- no missing value was substituted;
- readiness exposes that the value is derived.

A derived value must not produce a final investment recommendation by itself.

This policy does not authorize:

- BUY, SELL, or HOLD output;
- allocation guidance;
- conviction labels;
- urgency labels;
- target prices;
- tradeability conclusions;
- Telegram delivery;
- report generation;
- portfolio/watchlist updates.

## Implementation Requirements for BL22

The next implementation sprint should update existing modules where possible and
must obey the Python file creation policy.

Default expectation:

```text
Update existing provider, normalization, readiness, or fundamentals modules.
Do not create a new Python file unless formally justified.
```

Required tests:

- direct FreeCashFlow source field remains `source_reported`;
- missing direct FreeCashFlow with valid operating cash flow and capex becomes
  `source_derived`;
- missing operating cash flow keeps FreeCashFlow missing or not derivable;
- missing capex keeps FreeCashFlow missing or not derivable;
- invalid operating cash flow fails closed;
- invalid capex fails closed;
- currency mismatch fails closed;
- unit mismatch fails closed;
- period mismatch fails closed;
- ambiguous sign convention fails closed;
- no missing values are converted to zero;
- derived metric records provenance for both inputs;
- readiness warnings expose derived or not-derivable status;
- no investment behavior is produced.

## Non-Goals

RESET-10L-BL21 does not:

- implement FreeCashFlow derivation;
- modify Python code;
- modify tests;
- execute provider calls;
- write production data;
- run the production pipeline;
- generate reports;
- create Telegram artifacts;
- modify portfolio/watchlist files;
- produce investment recommendations.

## Next Step

Proceed to:

```text
RESET-10L-BL22 — Implement Governed FreeCashFlow Derivation
```

BL22 should implement this policy in existing modules wherever possible, add
contract and unit tests, and preserve all guardrails from this document and the
Python file creation policy.
