# ME-CI02 - ChatGPT Advisory Context Contract v1

## Purpose

ME-CI02 defines the ChatGPT Advisory Context Contract: the controlled,
evidence-backed context envelope that Market Engine may provide to an external
ChatGPT Advisory Layer.

The approved contract identity is:

```text
contract_name: chatgpt_advisory_context
contract_version: v1
schema_version: chatgpt-advisory-context-v1
artifact_type: market-engine-chatgpt-advisory-context
```

The contract composes existing Market Engine contracts. It does not replace
Structured Decision Output, Governor output, Dispatch Station output,
Recommendation Review, Portfolio Review, Decision Engine handoff, or source
coverage/readiness contracts.

## Architectural position

ChatGPT is an advisory interpretation layer above controlled Market Engine
artifacts.

ChatGPT is not:

* a Market Engine stage;
* a source-data acquisition layer;
* a fundamental analysis engine;
* a Setup Detection engine;
* a Recommendation Review replacement;
* a Portfolio Review replacement;
* a Decision Engine replacement;
* a Governor policy authority;
* a hidden production-action override path.

ChatGPT may explain, compare, summarize, and discuss scenarios only within the
structured context boundary. It must not silently expand, replace, or override
upstream evidence, governance, readiness, or decision semantics.

## Approved upstream context families

An advisory context may include or reference only approved, versioned Market
Engine outputs:

| Family | Include mode | Notes |
| --- | --- | --- |
| Structured Decision Output | embedded summary plus artifact reference | ME-CI01 is the canonical decision-facing input. |
| Governor output | referenced and selectively summarized | Preserve readiness, blockers, warnings, authority flags, and policy constraints. |
| Dispatch Station report | summary or artifact reference | Use ME-DS01 output as presentation context only; do not duplicate conflicting semantics. |
| Recommendation Review | reference or lineage summary | Preserve non-actionable/review semantics. |
| Portfolio Review | reference or lineage summary | Include only when approved context exists. |
| Decision Engine handoff/output | reference or embedded structured status when approved | ChatGPT may interpret, not override. |
| Source/readiness artifacts | references and machine-readable status summaries | No raw provider payloads unless a future contract explicitly authorizes them. |

## Contract identity

Every context artifact must contain:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `schema_version` | string | yes | Must equal `chatgpt-advisory-context-v1`. |
| `artifact_type` | string | yes | Must equal `market-engine-chatgpt-advisory-context`. |
| `generated_at` | string or null | yes | Context assembly timestamp. This is not market-data freshness. |
| `run_id` | string | yes | Advisory context run identity. |
| `ticker` | string | yes | Primary instrument ticker. |
| `instrument` | object | yes | Instrument identity copied from approved upstream context. |
| `source_artifact_refs` | array of strings | yes | References to all source artifacts used to assemble the context. |

Missing, malformed, unsupported, or conflicting identity fields fail closed.

## Top-level object

Required top-level fields:

```text
schema_version
artifact_type
generated_at
run_id
ticker
instrument
source_artifact_refs
advisory_eligibility
structured_decision_output
governor_context
dispatch_station_context
provenance_context
freshness_context
uncertainty_context
portfolio_context_boundary
recommendation_boundary
human_advisory_guidance
prohibited_inputs
prohibited_inferences
fail_closed
validation
```

## Advisory eligibility

ME-CI02 reuses existing readiness concepts and does not create a parallel
investment taxonomy.

Allowed advisory eligibility states are:

| State | Meaning |
| --- | --- |
| `eligible` | Required context, provenance, freshness, and Structured Decision Output are sufficient for advisory interpretation within the allowed scope. |
| `descriptive_only` | Context can support explanation of observations, blockers, or descriptive state, but not recommendation or action-oriented advisory wording. |
| `blocked` | Context assembly or upstream governance prevents advisory interpretation. ChatGPT must lead with blockers and avoid conclusions. |

Required fields:

```text
advisory_eligibility.state
advisory_eligibility.reason_codes
advisory_eligibility.allowed_scope
advisory_eligibility.required_disclosures
advisory_eligibility.blocking_reasons
```

`eligible` does not mean order-ready, broker-ready, allocation-ready, or
Decision Engine override-ready. It means ChatGPT may interpret the approved
context for the declared allowed scope.

## Provenance context

The context must preserve:

* evidence families present;
* evidence families missing;
* source artifact references;
* observation references;
* Governor references;
* Dispatch Station references;
* Structured Decision Output references;
* Recommendation Review references;
* Portfolio Review references;
* Decision Engine or handoff references when available;
* acquisition, import, run, and lineage identifiers when available.

Required fields:

```text
provenance_context.evidence_families_present
provenance_context.evidence_families_missing
provenance_context.artifact_refs
provenance_context.lineage_refs
provenance_context.missing_provenance
provenance_context.raw_payload_included
```

`raw_payload_included` must be false unless a later explicit contract approves
raw provider payloads for advisory context.

## Freshness context

Freshness must be represented per evidence family. A single global timestamp is
not sufficient when component freshness differs.

Required fields:

```text
freshness_context.global_freshness_status
freshness_context.family_freshness
freshness_context.stale_markers
freshness_context.stale_reasons
freshness_context.unknown_freshness
```

Allowed freshness statuses:

```text
fresh
mixed
stale
unknown
blocked
```

Unknown freshness is not equivalent to fresh. Stale or unknown critical evidence
must downgrade eligibility to `descriptive_only` or `blocked` according to the
fail-closed matrix.

## Uncertainty context

Required fields:

```text
uncertainty_context.confidence
uncertainty_context.uncertainty_level
uncertainty_context.missing_evidence
uncertainty_context.contradictory_evidence
uncertainty_context.unresolved_blockers
uncertainty_context.assumptions
uncertainty_context.limitations
```

Uncertainty must use machine-readable reason codes where possible. Free text may
explain but must not be the only representation of a blocker, missing evidence,
contradiction, or assumption.

## Structured Decision Output consumption

ME-CI01 Structured Decision Output is the canonical upstream decision-facing
input.

The advisory context must include:

```text
structured_decision_output.schema_version
structured_decision_output.artifact_ref
structured_decision_output.embedded_fields
structured_decision_output.fields_referenced_only
structured_decision_output.semantic_override_allowed
```

Rules:

* `schema_version` must be `structured-decision-output-v1`.
* `semantic_override_allowed` must be false.
* ChatGPT may explain Structured Decision Output but must not rewrite it.
* `decision`, `data_coverage`, `scores`, `portfolio_context`, `risk`,
  `levels`, `thesis`, `evidence`, `explainability`, `consumer_guidance`, and
  `validation` may be embedded as compact structured summaries.
* Full upstream artifacts should be referenced when embedding would create
  duplication or conflict.
* Missing Structured Decision Output blocks advisory eligibility unless the
  context is explicitly scoped to pre-decision descriptive explanation.

## Governor context boundary

Governor context may include:

* governance state;
* evaluation readiness;
* factor states and score availability;
* recommendation-state mapping;
* buy-zone explanation;
* position-management explanation;
* blockers;
* warnings;
* policy constraints;
* unresolved exceptions;
* non-actionability reasons;
* authority flags.

ChatGPT may explain Governor signals but must not ignore, override, or soften
them. Governor fixed-false authority fields remain fixed-false in advisory
interpretation.

## Dispatch Station context boundary

ME-DS01 Dispatch Station output may serve as advisory context in three modes:

| Mode | Meaning |
| --- | --- |
| `artifact_reference_only` | Use only a stable reference to the report artifact. |
| `summary_allowed` | Include selected summary sections while preserving source semantics. |
| `not_included` | Dispatch Station output is unavailable or unnecessary. |

Dispatch Station output is presentation context, not a second source of
decision truth. If Dispatch Station and Structured Decision Output conflict,
the advisory context must fail closed or mark the conflict explicitly.

## Human advisory guidance

The context must provide machine-readable instructions, not hidden prompt text.

Required fields:

```text
human_advisory_guidance.allowed_interpretation_scope
human_advisory_guidance.prohibited_inference_scope
human_advisory_guidance.mandatory_caveats
human_advisory_guidance.required_blocker_disclosure
human_advisory_guidance.required_stale_data_disclosure
human_advisory_guidance.required_uncertainty_disclosure
```

ChatGPT may turn these into user-facing prose, but the contract carries the
semantics.

## Portfolio context boundary

Portfolio context may be included only when upstream context proves it through
approved contracts such as `market-engine-portfolio-context-v1`, Portfolio
Review, Structured Decision Output, or a future Portfolio Intelligence contract.

Rules:

* Missing portfolio context must remain missing.
* Position size, exposure, cash, target weight, max weight, allocation, or
  portfolio fit must not be invented.
* Advisory output without portfolio context may still explain non-portfolio
  evidence and blockers.
* Advisory output without portfolio context must not answer portfolio-sizing or
  add/reduce questions as if current holdings were known.

## Recommendation boundary

The advisory context separates:

| Layer | Meaning |
| --- | --- |
| `evidence_summary` | What approved evidence says. |
| `analytical_interpretation` | What approved analysis/review context means. |
| `recommendation_interpretation` | Explanation of approved recommendation or decision states. |
| `advisory_wording` | Human-facing language generated from approved context. |

ChatGPT may not produce a higher actionability claim than upstream permits.

Examples:

```text
upstream descriptive_only
-> ChatGPT may explain observations and limitations
-> ChatGPT must not present a buy/sell/hold recommendation as validated
```

```text
upstream blocked
-> advisory context must be blocker-first
-> ChatGPT must not produce an actionable conclusion
```

## Prohibited inputs

Inputs that must never be included in v1 context:

* raw unvalidated web snippets;
* unprovenanced analyst opinions;
* stale snapshots without stale markers;
* current price without timestamp or evidence reference;
* portfolio holdings without run/source identity;
* unsupported target price;
* unsupported stop-loss;
* unsupported probability;
* broker secrets, credentials, or production action handles;
* prompt-only instructions that bypass structured fields.

Inputs allowed only as descriptive evidence when explicitly marked:

* company-profile-only context;
* operator notes;
* incomplete source-family summaries;
* historical report snippets;
* stale but clearly marked context.

Inputs that can support advisory interpretation:

* approved Structured Decision Output;
* approved Governor context;
* approved Dispatch Station report summaries;
* approved readiness, provenance, freshness, and blocker summaries.

Inputs that can support actionability only when upstream contracts allow:

* approved Decision Engine output;
* approved Portfolio Intelligence output;
* approved Position Sizing output;
* approved execution or broker contract output if a future sprint creates it.

## Prohibited inferences

ChatGPT must not infer:

* missing facts;
* current freshness from old timestamps;
* actionability from descriptive-only output;
* readiness by ignoring blockers;
* portfolio state from ticker ownership assumptions;
* price, target, stop-loss, or position size from incomplete context;
* certainty from incomplete evidence;
* recommendation from company profile alone;
* fundamentals by extrapolating missing values;
* allocation or order instructions from candidate states;
* Decision Engine approval from handoff readiness.

## Fail-closed matrix

| Condition | Required result |
| --- | --- |
| Missing or unsupported context schema version | `context_invalid`; advisory blocked |
| Missing ticker identity | `context_invalid`; advisory blocked |
| Missing run identity | `context_invalid`; advisory blocked |
| Missing required provenance | advisory blocked unless scope is explicitly descriptive-only and missing provenance is disclosed |
| Unknown freshness for critical evidence | descriptive-only or blocked with mandatory disclosure |
| Stale critical evidence | blocked when action/recommendation interpretation is requested; descriptive-only allowed only with disclosure |
| Missing Structured Decision Output | blocked for decision advisory; descriptive-only allowed only for pre-decision evidence explanation |
| Malformed Governor output | blocked for Governor-based advisory |
| Conflicting readiness states | blocked until conflict is resolved or explicitly disclosed as non-consumable |
| Blocked Recommendation Review | blocker-first descriptive explanation only |
| Missing portfolio context | portfolio-specific advisory blocked; non-portfolio explanation may remain descriptive |
| Incomplete Dispatch Station report | advisory may use artifact references only or block summary-based context |

Required fail-closed fields:

```text
fail_closed.context_status
fail_closed.advisory_status
fail_closed.reason_codes
fail_closed.recoverable
fail_closed.allowed_fallback_scope
```

Allowed `context_status` values:

```text
valid
valid_with_warnings
context_invalid
```

Allowed `advisory_status` values:

```text
eligible
descriptive_only
blocked
```

## Examples

Example artifacts:

```text
docs/market_engine/contracts/examples/chatgpt_advisory_context_v1_eligible.json
docs/market_engine/contracts/examples/chatgpt_advisory_context_v1_descriptive_only.json
docs/market_engine/contracts/examples/chatgpt_advisory_context_v1_blocked.json
```

The examples are synthetic and non-production. They illustrate contract shape
only and do not make real market claims.

## Acceptance criteria

* Contract name and version are explicit.
* Advisory eligibility states are defined.
* Provenance, freshness, uncertainty, readiness, and blockers are represented.
* ME-CI01 Structured Decision Output consumption is defined.
* Governor and Dispatch Station context boundaries are defined.
* Portfolio and recommendation boundaries are explicit.
* Prohibited inputs and prohibited inferences are listed.
* Fail-closed behavior is documented.
* Eligible, descriptive-only, and blocked examples exist.
* No runtime code is changed.
