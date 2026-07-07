# ME-CI01 - Structured Decision Output Contract v1

## Purpose

Structured Decision Output v1 defines the stable machine-readable layer between
Market Engine decision artifacts and downstream consumers such as:

* ChatGPT Advisory Layer;
* Notification Layer;
* dashboards;
* future frontends;
* audit and replay tooling.

The contract is an output contract only. It does not implement Decision Engine
runtime behavior, portfolio mutation, order generation, broker integration,
notification delivery, dashboard rendering, or ChatGPT API integration.

The approved schema version is:

```text
structured-decision-output-v1
```

The approved artifact type is:

```text
market-engine-structured-decision-output
```

## Design principles

* Versioned contract: every artifact must declare `schema_version`.
* Machine-readable first: structured fields are authoritative.
* Stable field names: consumers must not depend on presentation text.
* No UI coupling: the contract contains no layout, color, widget, markdown, or
  channel formatting requirements.
* No channel coupling: ChatGPT, Notification Layer, dashboards, and future
  adapters consume the same structured contract.
* No free-form advice as source of truth: human summaries may explain but must
  not override structured fields.
* Explicit data coverage: coverage, freshness, missing families, and blockers
  are first-class fields.
* Explicit confidence: scores and levels must carry status, scale, confidence,
  or reason codes where applicable.
* Explicit risk: risk fields must be present even when values are unavailable.
* Explicit provenance and reference hooks: important values must link to source,
  artifact, observation, review, or Decision Engine references.
* Fail closed when required fields are missing, malformed, unsupported, stale, or
  internally inconsistent.
* Backward-compatible additive evolution: compatible v1 changes may add optional
  fields but must not rename, remove, or change the meaning of existing fields.
* Conservative financial-language governance: consumers must preserve
  `is_actionable`, `review_required`, `blocked`, and `descriptive_only` states
  and must not fabricate buy, sell, allocation, or position-sizing guidance.

## Top-level object

| Field | Type | Required | Meaning | Fail-closed behavior |
| --- | --- | --- | --- | --- |
| `schema_version` | string | yes | Contract identity. Must equal `structured-decision-output-v1`. | Unsupported, missing, or malformed versions fail closed. |
| `artifact_type` | string | yes | Artifact family. Must equal `market-engine-structured-decision-output`. | Missing or mismatched type fails closed. |
| `generated_at` | string or null | yes | ISO-8601 generation timestamp when available. | Missing key fails closed. Null is allowed only when timestamp generation is not available and provenance explains why. |
| `run_id` | string | yes | Stable run identifier for replay and traceability. | Missing or empty values fail closed. |
| `ticker` | string | yes | Primary ticker identity. | Missing or mismatch with `instrument.ticker` fails closed. |
| `instrument` | object | yes | Structured instrument identity. | Missing required instrument fields fail closed. |
| `data_coverage` | object | yes | Coverage, freshness, missing evidence, and readiness. | Missing object or unsupported coverage state fails closed. |
| `decision` | object | yes | Machine-readable decision result or blocked/non-actionable state. | Missing object, unsupported action, or missing actionability state fails closed. |
| `scores` | object | yes | Versioned score slots. Individual values may be null. | Missing score slots fail closed; null values are allowed with status/reasons. |
| `portfolio_context` | object | yes | Portfolio context and position-sizing availability. | Missing object fails closed. Unknown portfolio data must remain explicit. |
| `risk` | object | yes | Risk score, risk level, flags, and invalidation triggers. | Missing object fails closed. Missing risk values must not be inferred. |
| `levels` | object | yes | Price, zone, support/resistance, and invalidation levels. | Missing object fails closed. Numeric levels without provenance or confidence are invalid. |
| `thesis` | object | yes | Thesis status, drivers, and change indicators. | Missing object fails closed. |
| `evidence` | object | yes | Source, artifact, observation, review, and Decision Engine references. | Missing object fails closed. Required lineage gaps must be represented explicitly. |
| `explainability` | object | yes | Reason codes, change reasons, blockers, and summary permission. | Missing object fails closed. |
| `consumer_guidance` | object | yes | Consumer permissions and forbidden uses. | Missing object fails closed. Consumers must respect forbidden uses. |
| `validation` | object | yes | Contract validation status and fail-closed reason. | Missing object fails closed. Invalid contract status blocks consumption. |

## Field semantics

### `schema_version`

Required string.

Allowed value:

```text
structured-decision-output-v1
```

Consumers must fail closed for missing, unknown, unsupported, malformed, or
future schema versions unless a later compatibility document explicitly approves
that version.

### `artifact_type`

Required string.

Allowed value:

```text
market-engine-structured-decision-output
```

Consumers must not treat other artifact types as Structured Decision Output.

### `instrument`

Required object.

Minimum fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `ticker` | string | yes | Instrument ticker. Must match top-level `ticker`. |
| `name` | string or null | yes | Instrument display name when known. |
| `asset_type` | string | yes | Example values: `equity`, `etf`, `fund`, `crypto`, `unknown`. |
| `exchange` | string or null | yes | Exchange when known. |
| `currency` | string or null | yes | Trading or reporting currency when known. |

Unknown fields must be explicit as `null` or controlled values such as
`unknown`. Consumers must not infer exchange, name, or currency from ticker.

### `data_coverage`

Required object.

Minimum fields:

| Field | Type | Required | Allowed values / meaning |
| --- | --- | --- | --- |
| `coverage_status` | string | yes | `ready`, `partial`, `descriptive_only`, `blocked` |
| `coverage_score` | number or null | yes | Coverage score on an explicitly declared scale, or null when unavailable. |
| `freshness_status` | string | yes | `fresh`, `stale`, `mixed`, `unknown`, `blocked` |
| `missing_families` | array of strings | yes | Source or evidence families missing from the run. |
| `stale_families` | array of strings | yes | Source or evidence families known to be stale. |
| `blocked_reason` | string or null | yes | Required when `coverage_status` is `blocked`. |

Coverage status semantics:

| Status | Meaning |
| --- | --- |
| `ready` | Required data families are present enough for the declared decision state. |
| `partial` | Some decision-relevant data is missing, but the artifact can still express a controlled limited state. |
| `descriptive_only` | Data supports description or interpretation only, not actionability. |
| `blocked` | Required data is missing, stale, invalid, unsupported, or insufficient. |

### `decision`

Required object.

Minimum fields:

| Field | Type | Required | Allowed values / meaning |
| --- | --- | --- | --- |
| `action` | string | yes | `no_action`, `watch`, `hold`, `buy_candidate`, `add_candidate`, `trim_candidate`, `exit_candidate`, `blocked` |
| `action_strength` | string or null | yes | `none`, `low`, `medium`, `high`, or null when not applicable. |
| `time_horizon` | string or null | yes | Example values: `short_term`, `swing`, `medium_term`, `long_term`, `unknown`. |
| `is_actionable` | boolean | yes | Whether downstream approved action semantics exist. |
| `actionability_blockers` | array of strings | yes | Reasons actionability is unavailable or constrained. |
| `review_required` | boolean | yes | Whether human or downstream Decision Engine review is required. |

The contract supports candidate states, but it does not force blind buy or sell
guidance. `is_actionable=false` is normal when data, confidence, portfolio
context, Decision Engine authority, or validation is incomplete.

Consumers must not convert `buy_candidate`, `add_candidate`, `trim_candidate`,
or `exit_candidate` into execution instructions unless a later approved
Decision Engine and execution contract explicitly allows it.

### `scores`

Required object. Each score slot is required, but each score `value` may be
`null` until the responsible engine exists and has approved evidence.

Required score slots:

* `conviction`;
* `confidence`;
* `quality`;
* `momentum`;
* `valuation`;
* `risk_reward`;
* `portfolio_fit`.

Each score slot must be an object with:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `value` | number or null | yes | Score value or null when unavailable. |
| `scale` | string | yes | Example: `0_100`, `not_available`. |
| `status` | string | yes | `available`, `not_available`, `insufficient_evidence`, `blocked`, `not_applicable`. |
| `reason_codes` | array of strings | yes | Machine-readable reasons for the score or absence. |

Consumers must not invent missing score values. Null score values must remain
null in downstream interpretation.

### `portfolio_context`

Required object.

Minimum fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `position_status` | string | yes | `held`, `not_held`, `partially_known`, `unknown`, `stale`, `invalid`, `not_available`. |
| `current_weight` | number or null | yes | Current portfolio weight when approved context provides it. |
| `target_weight` | number or null | yes | Target weight only when an approved Decision Engine contract provides it. |
| `max_weight` | number or null | yes | Maximum allowed weight when approved portfolio policy provides it. |
| `exposure_flags` | array of strings | yes | Exposure, overlap, or policy flags. |
| `concentration_risk` | string | yes | `low`, `medium`, `high`, `unknown`, `not_available`, `blocked`. |
| `cash_dependency` | string | yes | `none`, `low`, `medium`, `high`, `unknown`, `not_available`. |
| `position_sizing_available` | boolean | yes | Whether approved position-sizing output is available. |

This object reserves future Portfolio Intelligence support without implementing
Portfolio Intelligence in ME-CI01.

### `risk`

Required object.

Minimum fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `risk_level` | string | yes | `low`, `medium`, `high`, `unknown`, `blocked`. |
| `risk_score` | number or null | yes | Risk score when available. |
| `volatility_flag` | string | yes | `none`, `watch`, `elevated`, `unknown`, `blocked`. |
| `liquidity_flag` | string | yes | `none`, `watch`, `elevated`, `unknown`, `blocked`. |
| `drawdown_flag` | string | yes | `none`, `watch`, `elevated`, `unknown`, `blocked`. |
| `thesis_break_flags` | array of strings | yes | Machine-readable thesis-break flags. |
| `invalidation_triggers` | array of objects | yes | Evidence-backed invalidation triggers. |

Risk values must not be inferred from price movement, ticker identity, or
consumer preference.

### `levels`

Required object.

Minimum fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `current_price` | level object or null | yes | Current price only when approved evidence exists. |
| `buy_zone` | level range object or null | yes | Candidate buy zone when approved. |
| `add_zone` | level range object or null | yes | Candidate add zone when approved. |
| `trim_zone` | level range object or null | yes | Candidate trim zone when approved. |
| `stop_or_invalidation_level` | level object or null | yes | Invalidation level, not an order instruction. |
| `support_levels` | array of level objects | yes | Approved support levels. |
| `resistance_levels` | array of level objects | yes | Approved resistance levels. |

Level object minimum fields:

* `value`;
* `currency`;
* `confidence`;
* `evidence_refs`;
* `as_of`.

Level range object minimum fields:

* `lower`;
* `upper`;
* `currency`;
* `confidence`;
* `evidence_refs`;
* `as_of`.

Numeric levels are invalid without evidence references and confidence. A stop or
invalidation level is analytical context only and must not be interpreted as a
stop order.

### `thesis`

Required object.

Minimum fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `summary_codes` | array of strings | yes | Machine-readable thesis summary codes. |
| `positive_drivers` | array of strings | yes | Positive driver codes. |
| `negative_drivers` | array of strings | yes | Negative driver codes. |
| `changed_since_previous_run` | boolean or null | yes | Null when no comparable previous run exists. |
| `thesis_status` | string | yes | `intact`, `improving`, `deteriorating`, `mixed`, `unknown`, `blocked`. |

### `evidence`

Required object.

Minimum fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `source_refs` | array of strings | yes | Raw or normalized source references. |
| `artifact_refs` | array of strings | yes | Market Engine artifact references. |
| `observation_refs` | array of strings | yes | Observation IDs or paths. |
| `recommendation_review_ref` | string or null | yes | Upstream Recommendation Review reference. |
| `portfolio_review_ref` | string or null | yes | Upstream Portfolio Review reference. |
| `decision_engine_ref` | string or null | yes | Decision Engine artifact reference when available. |

Consumers must not rely on unreferenced values when reference fields indicate
missing or incomplete evidence.

### `explainability`

Required object.

Minimum fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `primary_reason_codes` | array of strings | yes | Primary reason codes for the decision state. |
| `score_change_reasons` | array of strings | yes | Score-change codes. Empty when unavailable. |
| `decision_change_reasons` | array of strings | yes | Decision-change codes. Empty when unavailable. |
| `blocking_reasons` | array of strings | yes | Blocking reason codes. |
| `human_summary_allowed` | boolean | yes | Whether consumers may produce human summaries from the structured output. |

ChatGPT may formulate explanations from codes and references, but this
Structured Decision Output remains the source of truth.

### `consumer_guidance`

Required object.

Minimum fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `chatgpt_allowed_use` | array of strings | yes | Allowed ChatGPT Advisory Layer uses. |
| `notification_allowed_use` | array of strings | yes | Allowed Notification Layer uses. |
| `dashboard_allowed_use` | array of strings | yes | Allowed dashboard/frontend uses. |
| `requires_human_review` | boolean | yes | Whether human review is required before action interpretation. |
| `forbidden_uses` | array of strings | yes | Explicit forbidden uses. |

Allowed ChatGPT uses may include explanation, comparison, scenario discussion,
and artifact-grounded question answering. Notification Layer may consume only
compact signal semantics. Dashboards may display structured states but must not
invent missing values.

Forbidden uses must include consumer-side fabrication of missing scores,
portfolio context, price levels, actionability, allocation, orders, or broker
instructions when absent from the artifact.

### `validation`

Required object.

Minimum fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `contract_status` | string | yes | `valid`, `valid_with_warnings`, `fail_closed`. |
| `required_fields_present` | boolean | yes | Whether required top-level and required nested fields are present. |
| `semantic_warnings` | array of strings | yes | Non-blocking warnings. |
| `fail_closed_reason` | string or null | yes | Required when `contract_status` is `fail_closed`. |

Consumers must not process an artifact with `contract_status=fail_closed` as an
actionable or ready decision. Consumers must preserve fail-closed reasons in
downstream summaries.

## Compatibility rules

Compatible v1 changes may:

* add optional fields;
* add new reason codes;
* add new non-breaking evidence references;
* add new allowed consumer-use labels when they do not weaken forbidden uses.

Incompatible changes require a new schema version when they:

* rename or remove existing fields;
* change allowed values or meanings;
* make optional fields required;
* weaken fail-closed behavior;
* weaken consumer forbidden-use rules;
* introduce action, allocation, order, broker, or channel semantics not approved
  by this contract.

## Fail-closed rules

Consumers and future producers must fail closed when:

* required top-level fields are missing;
* required nested fields are missing;
* `schema_version` or `artifact_type` is unsupported;
* ticker identity conflicts with `instrument.ticker`;
* required provenance is absent for numeric levels;
* `validation.contract_status` is `fail_closed`;
* data coverage is `blocked`;
* `decision.action` is unsupported;
* `decision.is_actionable=true` while actionability blockers or required
  Decision Engine references are missing;
* consumers encounter unsupported future versions.

Fail-closed behavior must produce an explicit blocked or non-consumable state,
not a default action.

## Acceptance criteria

* Structured Decision Output v1 is versioned.
* The contract defines a top-level object and field semantics.
* Required and optional fields are identified.
* Fail-closed behavior is documented.
* Consumer rules for ChatGPT Advisory Layer, Notification Layer, and dashboards
  are documented.
* Coverage, readiness, and actionability are explicit.
* Scores may be null while engines are absent.
* Consumers may not invent missing values.
* The contract supports future Conviction, Position Sizing, and Portfolio
  Intelligence without implementing them.
* Example artifacts exist for actionable-candidate and blocked/descriptive-only
  cases.
* Roadmap and backlog are synchronized.
* No runtime code is changed.
