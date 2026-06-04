# V2 Real Analysis Output Defect Review

Status: ACTIVE
Reset stage: RESET-10L-BL24

## Purpose

This document reviews the real-analysis output after the NVDA one-ticker analysis
rerun with governed derived FreeCashFlow.

The goal is to identify the next true blocker after `CASH_FLOW_UNKNOWN` was
resolved, define what `prior-year growth evidence` must mean, and determine the
next implementation step without broadening scope to recommendations, reporting,
Telegram delivery, portfolio/watchlist updates, or production pipeline behavior.

This is a review/governance artifact only. It does not implement code, modify
tests, run provider calls, write production data, generate reports, create
Telegram artifacts, update portfolio/watchlist files, or invoke Decision Engine
investment behavior.

## Reviewed Inputs

This review is based on:

- `RESET-10L-BL20 — Run First One-Ticker Real Fundamental Analysis`;
- `RESET-10L-BL21 — Govern FreeCashFlow Derivation or Missingness Policy`;
- `RESET-10L-BL22 — Implement Governed FreeCashFlow Derivation`;
- `RESET-10L-BL23 — Re-run NVDA One-Ticker Real Fundamental Analysis with Derived FreeCashFlow`.

Relevant records:

```text
docs/active/v2_nvda_first_real_fundamental_analysis_review.md
docs/active/v2_free_cash_flow_derivation_policy.md
docs/active/v2_free_cash_flow_derivation_implementation.md
docs/active/v2_nvda_real_analysis_rerun_with_derived_fcf.md
BL23 Finding

The BL23 rerun confirmed that governed derived FreeCashFlow works for the NVDA
analysis path.

Resolved blocker:

CASH_FLOW_UNKNOWN

Updated cash-flow profile:

CASH_FLOW_POSITIVE

Updated source-data readiness:

source readiness: available
missing_fundamentals_count: 0
free_cash_flow status: source_derived

The analysis still remains limited, but no longer because cash-flow data is
unknown.

Remaining limitation:

LIMITED_ANALYSIS due to missing governed prior-year growth evidence
Defect Classification

This is not a generic missing-data problem anymore. It is a specific analysis
coverage limitation.

Defect class:

REAL_ANALYSIS_OUTPUT_LIMITATION

Defect name:

MISSING_GOVERNED_PRIOR_YEAR_GROWTH_EVIDENCE

Affected area:

fundamental analysis / growth evidence / real-source comparison data

Severity:

HIGH for real analysis usefulness
LOW for safety

Reasoning:

The system correctly avoids producing a final recommendation.
The system correctly carries derived FreeCashFlow with provenance.
The system correctly improves cash-flow readiness.
The system still cannot produce a fuller real fundamental analysis because it
lacks governed prior-year comparison evidence.
What Prior-Year Growth Evidence Must Mean

Prior-year growth evidence should mean a governed comparison between a current
period value and a prior comparable period value.

At minimum, a future implementation should support growth evidence for:

revenue
net_income
operating_income
free_cash_flow

The first implementation may start with revenue and free_cash_flow only if scope
must remain narrow.

A governed prior-year growth evidence record must preserve:

current period metric name;
current period value;
prior period value;
prior period reference;
fiscal year and fiscal period context;
currency;
unit;
source or raw evidence linkage for both values;
comparison formula;
growth value or status;
validation warnings;
explicit missing/invalid/not-comparable status when needed.
Allowed Growth Evidence States

The future implementation should support explicit states such as:

growth_available
growth_missing_prior_period
growth_missing_current_period
growth_invalid_current_period
growth_invalid_prior_period
growth_not_parseable
growth_not_comparable
growth_period_mismatch
growth_currency_mismatch
growth_unit_mismatch
growth_provenance_gap

Missing or invalid prior-year data must not be converted to zero.

Growth Formula

For numeric values where comparison is valid:

growth_rate = (current_value - prior_value) / abs(prior_value)

If the prior value is zero, missing, invalid, not parseable, or sign-ambiguous,
then growth must fail closed as not comparable or invalid rather than producing a
misleading value.

Analysis Behavior Requirements

A future implementation may allow real analysis to move beyond the current
LIMITED_ANALYSIS state only when enough governed growth evidence is available.

However, growth evidence must not by itself produce:

BUY;
SELL;
HOLD;
allocation;
conviction;
urgency;
target price;
tradeability;
recommendation behavior.

Any analysis output must remain review-oriented until a separate approved sprint
authorizes recommendation semantics.

Next Implementation Scope

The next implementation step should be narrow and should not add broad analysis
architecture.

Approved next candidate:

RESET-10L-BL25 — Implement Governed Prior-Year Growth Evidence

BL25 should:

inspect existing fundamentals, provider adapter, analysis, and readiness
modules;
update existing Python files first;
obey docs/active/v2_python_file_creation_policy.md;
add governed prior-year growth evidence states;
add tests for available, missing, invalid, not comparable, and mismatch states;
keep missing values explicit;
avoid reports, Telegram, production pipeline execution, portfolio/watchlist
updates, and recommendation behavior.
Python Cleanup Roadmap Note

The cleanup goal remains active.

However, the project should not start deleting or archiving Python files before
one more real-analysis iteration verifies the prior-year growth evidence path.

Recommended cleanup step after BL25 or BL26:

Python Usage and Legacy Cleanup Review

That review should identify:

Python files used by the new v2 real analysis path;
Python files that are legacy-only;
duplicate responsibilities;
files that can be archived or removed after certified replacement.
Non-Goals

RESET-10L-BL24 does not:

implement prior-year growth evidence;
modify Python code;
modify tests;
execute provider calls;
write production data;
run the production pipeline;
generate reports;
create Telegram artifacts;
modify portfolio/watchlist files;
produce BUY, SELL, HOLD, allocation, conviction, urgency, target-price,
tradeability, scoring, or recommendation behavior.
Conclusion

The real analysis path has improved. CASH_FLOW_UNKNOWN is resolved for the
NVDA rerun through governed derived FreeCashFlow.

The next real blocker is governed prior-year growth evidence. The project should
implement that next, narrowly, using existing modules where possible and keeping
all missingness, comparability, provenance, and safety guardrails explicit.