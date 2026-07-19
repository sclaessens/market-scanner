# ME-DATA08 Operator Fundamental Metric Package Audit

## Outcome

ME-DATA08 implements a local, deterministic preparation and fail-closed
validation flow for operator-supplied fundamental metric evidence. An accepted
package uses the existing
`market-engine-data07-operator-fundamental-metrics-v1` import boundary. The
preparation input contract is
`market-engine-data08-operator-fundamental-metric-input-v1`; the validator and
report contracts are independently versioned as
`market-engine-data08-operator-fundamental-metric-validator-v1` and
`market-engine-data08-operator-fundamental-metric-validation-report-v1`.

## Governance-approved metric boundary

The normative allowlist is the ME-DATA06/ME-DATA07 `MVP_METRIC_FIELDS` tuple:

```text
revenue_growth_yoy
eps_growth_yoy
gross_margin
operating_margin
debt_to_equity
```

These are duration ratio metrics. `ratio` and `percent` are the only accepted
input units; percent is transparently normalized by the existing ME-DATA07
consumer. Currency is not applicable and a non-empty currency is rejected.
Aliases, inferred periods, guessed units, scale multipliers, currency
conversion, derived ratios, estimates, interpolation, scoring, ranking, and
qualitative classifications are not supported.

## Package contract

The top level requires `schema_version`, `package_id`, `created_at`, and a
non-empty `records` array. Each record requires:

- ticker and company identity (`name` and canonical `instrument_id`);
- canonical metric identity, finite numeric value, and explicit unit;
- duration period type, start/end dates, fiscal year, and fiscal period;
- primary-source provenance: source name/reference, raw source field, source
  date, observation/acquisition timestamps, and parser version.

Preparation groups records by ticker and reporting period, orders records and
metrics canonically, preserves source references and raw values, and emits the
existing ME-DATA07 metric-map structure. Raw operator input is read-only.
Identical inputs produce identical package bytes. Output paths must be new, so
the flow never silently overwrites an existing accepted package or report.

## Operator workflow

```bash
PYTHONPATH=src python -m market_engine.data.operator_fundamental_metric_package_command \
  --input operator_input/market_engine/me-data08/fundamental_metric_input.json \
  --package-output operator_input/market_engine/me-data07/fundamental_metrics.json \
  --report-output artifacts/market_engine/operator_fundamental_metric_packages/validation_report.json
```

Exit code `0` means the package is accepted and was written. Exit code `1`
means governance validation rejected the entire package; only the validation
report is written. Exit code `2` identifies malformed/unreadable operator
input or command usage. Exit code `3` identifies an output filesystem error.
Expected rejection paths do not emit a traceback.

## Accepted and rejected semantics

Acceptance means only
`eligible_for_explicit_me_data07_operator_import`. It does not execute that
import and does not establish completeness, analysis readiness, actionability,
recommendation readiness, or tradeability. Any blocking issue rejects the
whole package and sets `downstream_consumability` to `not_consumable`; partial
acceptance is forbidden. The report contains package identity, contract and
validator versions, input checksum, counts, errors/warnings, JSON paths,
metric identities where relevant, artifact paths, and the explicit downstream
boundary.

Stable blocking reason codes are:

```text
AMBIGUOUS_METRIC_ALIAS
AMBIGUOUS_SCALE
AUTHORITY_FIELD_FORBIDDEN
COMPANY_IDENTITY_MISSING
COMPANY_TICKER_MISMATCH
CONFLICTING_RECORD_CONTEXT
CURRENCY_NOT_APPLICABLE
DUPLICATE_METRIC_CONFLICT
DUPLICATE_METRIC_RECORD
EMPTY_METRIC_SET
INCOMPATIBLE_PERIOD_TYPE
INVALID_FISCAL_CONTEXT
INVALID_REPORTING_PERIOD
INVALID_TICKER
INVALID_TIMESTAMP
INVALID_UNIT
METRIC_NOT_ALLOWLISTED
PACKAGE_NOT_OBJECT
PROVENANCE_INCOMPLETE
RECORD_NOT_OBJECT
RECORDS_MISSING
REQUIRED_FIELD_MISSING
UNSUPPORTED_SCHEMA_VERSION
UNKNOWN_FIELD
VALUE_NOT_FINITE
VALUE_NOT_NUMERIC
```

`PERCENT_NORMALIZED_TO_RATIO` is the only normalization warning. A warning
does not override a blocking error.

## Side-effect and downstream boundary

ME-DATA08 reads one explicit local file and writes only the two explicit local
output paths. It introduces no provider or network call, credential, canonical
datastore import, ME-DATA07 execution, ME-DATA06 execution, ME-RUN31
execution, portfolio/watchlist mutation, delivery send, scheduler, model
invocation, broker action, recommendation rule, or Decision Engine authority.

## Verification evidence

The focused tests cover accepted preparation, deterministic serialization,
canonical ordering, the existing ME-DATA07 validator handoff, raw-input
immutability, stable summaries and reason codes, duplicate/conflict behavior,
whole-package rejection, malformed JSON including NaN/Infinity, output
behavior, and CLI exit codes. Final command results are recorded in the sprint
PR and closeout report.

## Remaining boundary

An operator must still supply genuine governance-approved primary-source
evidence. Acceptance by ME-DATA08 permits only an explicit ME-DATA07 operator
import attempt. Production import and automatic downstream consumption remain
blocked.
