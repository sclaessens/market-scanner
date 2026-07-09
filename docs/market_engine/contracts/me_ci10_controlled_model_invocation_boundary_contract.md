# ME-CI10 - Controlled Model Invocation Boundary Contract v1

## Purpose

ME-CI10 defines the formal boundary for a future controlled model invocation
between the existing local advisory pipeline and a future external model
provider.

Approved contract identity:

```text
contract_name: controlled_model_invocation_boundary
contract_version: v1
schema_version: market-engine-controlled-model-invocation-boundary-v1
artifact_type: market-engine-controlled-model-invocation-boundary-contract
```

This is a contract document. It is not a runtime artifact, provider adapter,
model API call, SDK integration, authentication layer, retry loop, timeout
handler, prompt executor, response parser, delivery integration, broker layer,
portfolio mutation path, allocation layer, sizing layer, or autonomous decision
system.

## Architecture position

Approved sequence:

```text
ME-CI05 - advisory artifact assembly
  -> ME-CI06 - advisory artifact validation
  -> ME-CI07 - prompt + response grounding contract
  -> ME-CI08 - local synthetic dry-run scaffold
  -> ME-CI09 - grounding validator hardening
  -> ME-CI10 - controlled model invocation boundary contract
  -> future controlled model invocation implementation
  -> CI09 grounding validator
  -> future grounded-response delivery boundary
```

Current pre-model boundary:

```text
validated source artifact
  -> prompt package
  -> synthetic response fixture
  -> grounding validation
```

Future desired extension:

```text
validated source artifact
  -> prompt package
  -> controlled invocation request
  -> provider/model
  -> raw response capture
  -> parser boundary
  -> grounding validation
```

ME-CI10 defines only the boundary. A later sprint must separately approve any
runtime implementation.

## Contract identity

The CI10 contract identity names the boundary and the required future envelope
semantics. The contract document is not itself a produced runtime artifact.

Future runtime artifacts may reference this contract using:

```text
schema_version: market-engine-controlled-model-invocation-boundary-v1
contract_name: controlled_model_invocation_boundary
contract_version: v1
```

## Invocation eligibility

A future invocation may start only when all conditions hold:

```text
source advisory artifact is CI06-valid
AND prompt package is valid
AND question class is approved
AND permitted use case is approved
AND authority boundary is present
AND mandatory disclosures are derivable
AND required context selection succeeded
AND invocation request passes deterministic validation
```

Hard rules:

```text
file exists != invocation eligible
prompt package exists != invocation eligible
valid source artifact != every question allowed
question accepted != authority expansion
```

Invalid input must stop before the provider boundary. No best-effort invocation
with incomplete validation is allowed.

## Pre-invocation gates

A future deterministic pre-invocation validator must check at least:

1. source artifact CI06-valid;
2. prompt package valid;
3. prompt package identity match;
4. source run identity match;
5. instrument identity match;
6. approved question class;
7. approved permitted use case;
8. approved response contract identity;
9. authority boundary present;
10. required context present or explicitly absent according to question class;
11. mandatory disclosure policy present;
12. forbidden inference set present;
13. provider identity valid;
14. model identity valid;
15. capability profile acceptable;
16. token and cost budgets valid;
17. timeout policy valid;
18. retry policy bounded;
19. idempotency key valid;
20. no forbidden input class present.

ME-CI10 does not implement this validator.

## Approved input package

The approved primary runtime input is the validated prompt package emitted by
the CI08 prompt-package boundary:

```text
schema_version: market-engine-advisory-prompt-package-v1
artifact_type: market-engine-advisory-prompt-package
contract_name: controlled_advisory_prompt_package
contract_version: v1
```

Allowed prompt package families are:

* prompt package identity;
* source artifact identity;
* instrument identity;
* user question;
* question classification;
* permitted use case;
* selected context;
* mandatory disclosures;
* forbidden inferences;
* required response contract;
* grounding requirements;
* authority boundary.

The invocation boundary must not accept raw provider payloads, the entire data
lake, raw portfolio files, arbitrary local files, raw secrets, broker
credentials, environment dumps, or unvalidated user prose as facts.

## Forbidden input classes

Forbidden invocation inputs include:

* invalid advisory artifact;
* invalid prompt package;
* unvalidated user claims treated as facts;
* raw broker credentials;
* API keys inside prompt payload;
* secrets;
* arbitrary environment variables;
* raw production database dumps;
* unrelated portfolio data;
* unrelated user personal data;
* unsupported model instructions;
* instructions that override system authority boundaries;
* direct order instructions;
* target weight instructions;
* position sizing authority;
* unsupported causal conclusions;
* unsupported materiality conclusions;
* hidden external retrieval results;
* provider-side browsing output.

## Context minimization

The invocation request must send only context required for the approved question
class and use case.

It must not send the complete advisory artifact by default.

Always-required context:

* contract identity;
* source artifact identity;
* validation evidence;
* instrument identity;
* run identity;
* advisory eligibility;
* Structured Decision Output context;
* blockers;
* missing context;
* freshness;
* uncertainty;
* provenance;
* authority boundary.

Question-relevant context:

* Portfolio Intelligence for portfolio, position, sizing, and allocation
  questions;
* Explainability / Change-Rationale for change, why, and comparison questions;
* Governor context for recommendation, buy-zone, and position-management
  explanations;
* Dispatch only as presentation/reference context.

Conditionally required context:

* comparable baseline for change questions;
* proven holding state for ownership questions;
* approved level evidence for price-level questions;
* upstream attribution level for reason claims.

Prohibited context:

* raw provider payloads unless they are already inside an approved prompt
  package context family;
* secrets or credentials;
* unrelated user or portfolio data;
* external browsing results;
* data added after prompt-package validation.

Minimization must never silently remove blockers, uncertainty, freshness,
missingness, provenance, or authority boundaries.

## Invocation request envelope

A future invocation request envelope should be machine-checkable and contain:

| Field | Required | Nullable | Semantics |
|---|---:|---:|---|
| `invocation_request_identity` | yes | no | Request schema, artifact type, contract identity, request id, generated timestamp. |
| `source_prompt_package_identity` | yes | no | Prompt package schema, artifact type, contract name, contract version, package id, hash. |
| `source_artifact_identity` | yes | no | CI06-valid advisory artifact identity and validation status. |
| `instrument_identity` | yes | no | Ticker and instrument identity copied from the prompt package. |
| `run_identity` | yes | no | Market Engine source run identity. |
| `invocation_identity` | yes | no | Invocation id and logical invocation grouping. |
| `attempt_identity` | yes | no | Attempt id and attempt sequence. |
| `question_classification` | yes | no | Approved question class and required context families. |
| `permitted_use_case` | yes | no | CI08 permitted use case. |
| `provider_identity` | yes | no | Provider name, API family, and trace capability declaration. |
| `model_identity` | yes | no | Model name, version or snapshot, and reporting policy. |
| `model_capability_profile` | yes | no | Required/preferred/optional/unsupported capabilities. |
| `instruction_contract` | yes | no | System/application/user hierarchy and forbidden overrides. |
| `input_context` | yes | no | Minimized selected context from the validated prompt package. |
| `response_schema_contract` | yes | no | Required CI07 response envelope identity. |
| `budget_limits` | yes | no | Token and cost budgets. |
| `timeout_policy` | yes | no | Connect, response, and total timeout semantics. |
| `retry_policy` | yes | no | Bounded retry and non-retryable failure semantics. |
| `idempotency_policy` | yes | no | Idempotency key and duplicate behavior. |
| `data_handling_policy` | yes | no | Secret exclusion, persistence, retention, and logging rules. |
| `grounding_handoff_contract` | yes | no | Mandatory parser-to-CI09 grounding handoff. |
| `audit_context` | yes | no | Source SHA, operator/runtime context, and audit evidence requirements. |

## Run/invocation/attempt identity

The Market Engine run identity is distinct from model invocation identity:

```text
Market Engine run_id != invocation_id
```

Required identities:

| Identity | Meaning |
|---|---|
| `run_id` | Upstream Market Engine run that produced the advisory artifact and prompt package. |
| `invocation_id` | Logical model invocation for one prompt package and configuration profile. |
| `attempt_id` | One concrete provider call attempt under an invocation. |

Rules:

```text
retry attempt != new Market Engine analysis run
new model response != new upstream market analysis
same prompt package + new invocation attempt must preserve source lineage
```

## Idempotency

The future idempotency key must derive from stable inputs:

* source advisory artifact identity;
* prompt package identity;
* prompt package content hash;
* question class;
* permitted use case;
* provider identity;
* model identity;
* configuration profile;
* response schema contract;
* budget policy version.

Repeated identical requests may reuse a prior result only when the idempotency
policy explicitly allows reuse and the persisted result is complete and
auditable.

Retries use the same `invocation_id` and distinct `attempt_id` values.
Duplicate concurrent invocation for the same key must be blocked or attached to
the existing in-flight invocation according to an explicit policy. The audit
trail must distinguish cache/reuse from a new provider call.

## Provider identity

Provider identity fields:

```text
provider_name
provider_api_family
provider_account_boundary
provider_region_or_endpoint_policy
provider_trace_id_policy
provider_usage_metadata_policy
```

ME-CI10 is provider-neutral. It does not require a specific provider.
Provider-neutral does not mean provider metadata may be discarded.

## Model identity

Model identity fields:

```text
model_name
model_version_or_snapshot
model_release_channel
capability_profile_id
configuration_profile_id
model_identity_reported_by_provider
```

If a provider cannot report stable model identity or equivalent versioning, the
future implementation must treat this as a capability limitation and record it
explicitly.

## Capability profile

Capability categories:

| Capability | v1 requirement |
|---|---|
| Structured JSON output support | required |
| Stable model identity reporting | required or explicit limitation |
| Provider request/trace id | required or explicit limitation |
| Bounded output size | required |
| Timeout support | required |
| Non-streaming response option | required |
| Deterministic or constrained schema mode | preferred |
| Refusal signaling | required |
| Usage metadata | required or explicit limitation |
| Token accounting | required or explicit limitation |
| Tool use | unsupported for v1 |
| External browsing | unsupported for v1 |
| Streaming | unsupported for v1 |

Provider-neutral means these requirements are expressed as capabilities rather
than provider-specific APIs.

## Prompt payload boundary

Future prompt payload structure:

```text
system_instructions
application_instructions
user_question
selected_context
required_response_schema
mandatory_disclosures
forbidden_inferences
authority_boundary
grounding_requirements
```

Rules:

```text
user question cannot override system boundary
selected context cannot remove mandatory uncertainty/blocker data
prompt text != source of truth
```

The source of truth remains the validated prompt package and upstream
artifacts.

## Instruction hierarchy

Instruction precedence:

```text
system contract
  > application/developer instruction
  > user question
```

A user question may not ignore blockers, add sizing authority, add allocation
authority, add execution authority, force unsupported certainty, require an
unsupported target, disable grounding, bypass the parser, or publish raw output
directly.

## Token budgets

Future budget policy must define:

```text
max_input_tokens
max_output_tokens
max_total_tokens
soft_warning_threshold
```

If the prompt package exceeds budget, the implementation must fail closed or
perform an approved deterministic context-minimization step. It must not
arbitrarily truncate until the prompt fits.

Context that may never be silently dropped:

* source identity;
* validation evidence;
* authority boundary;
* forbidden inferences;
* mandatory disclosures;
* blockers;
* uncertainty;
* freshness;
* missing context;
* provenance;
* required response schema;
* grounding requirements.

## Cost budgets

Future budget policy must define:

```text
per_invocation_cost_limit
per_run_cost_limit
daily_nonproduction_cost_limit
budget_policy_source
budget_policy_version
```

ME-CI10 does not hardcode price values. A future implementation must fail
closed before invocation when a request or retry would exceed budget. Usage and
estimated cost metadata must be persisted when available.

## Timeout policy

Timeout policy must define:

```text
connect_timeout
response_timeout
total_invocation_timeout
timeout_state
retry_eligibility
persistence_behavior
```

A provider timeout is a technical invocation failure. It is not an investment
conclusion and must not be translated into `unable_to_determine`.

## Retry policy

Retryable technical failures may include transient timeout, transient provider
transport failure, and provider rate limit when policy allows.

Non-retryable failures include invalid prompt package, invalid schema contract,
authority violation, malformed invocation request, unsupported question class,
budget exceeded, forbidden input detected, and policy-blocked requests.

Retries must be bounded. No unbounded loop, autonomous indefinite retry, or
silent repeated provider call is allowed.

## Backoff policy

Backoff is contract-only in ME-CI10. A future implementation must define:

* bounded retry count;
* bounded backoff;
* maximum elapsed retry window;
* audit trail per attempt;
* no autonomous indefinite retry.

## Raw response capture

Raw provider response must be captured separately as a future
`raw_model_response` artifact family.

Required provenance:

```text
invocation_id
attempt_id
provider_identity
model_identity
provider_request_id
received_at
finish_reason
usage_metadata
raw_output
response_content_type
raw_output_hash
```

Raw output is:

```text
not grounded
not parsed
not delivery eligible
not advisory truth
```

## Sensitive data handling

Secrets must never enter prompt payloads or persisted artifacts.

Rules:

* API keys must not be persisted;
* auth headers must not be persisted;
* raw provider response metadata must exclude secrets;
* prompt packages must be checked for forbidden secret-bearing fields by a
  future implementation;
* logs must not contain credentials;
* environment variables are not prompt context.

ME-CI10 does not implement a secret scanner.

## Invocation state taxonomy

Technical invocation states:

```text
request_validated
request_blocked
invocation_started
response_received
raw_response_persisted
provider_refusal
empty_response
truncated_response
malformed_response
schema_invalid_response
transport_failure
timeout
rate_limited
provider_error
budget_blocked
policy_blocked
parse_pending
parsed
parse_failed
grounding_pending
grounding_completed
grounding_ungrounded
grounding_blocked
```

These are not advisory response modes. They must not be confused with
`unable_to_determine`, `refused_outside_authority`, or
`blocked_invalid_context`.

## Provider refusal

Provider/model refusal is a runtime/provider outcome.

Advisory refusal is a structured response mode under CI07.

```text
model refuses request != refused_outside_authority
```

A provider refusal must be captured as raw response provenance and must not be
converted into an advisory refusal unless a parsed structured response passes
the CI07 parser boundary and CI09 grounding validation.

## Empty response

Empty provider response handling:

```text
empty provider response
  -> invocation failure
  -> no parser success
  -> no grounding validation success
  -> no delivery eligibility
```

No fallback advisory text may be generated.

## Truncated response

Known truncated output handling:

```text
finish reason indicates truncation
  -> truncated_response
  -> no successful structured parsing
  -> no grounding validation
  -> no delivery eligibility
```

Retry eligibility is policy-dependent. Best-effort grounding of known-truncated
output is not allowed in v1.

## Malformed JSON

Malformed JSON handling:

```text
raw response malformed JSON
  -> parser failure
  -> no grounding validator invocation on fabricated repaired content
```

No silent JSON repair or LLM self-repair loop is approved by ME-CI10.

## Schema mismatch

Valid JSON with the wrong response schema is a parser/contract failure.

```text
valid JSON + wrong response schema != close enough
```

The CI07 response envelope remains the source of truth.

## Parser boundary

Required future sequence:

```text
raw response capture
  -> strict parser
  -> structured response candidate
  -> CI09 grounding validator
```

Parser responsibilities:

* syntax;
* schema shape;
* primitive types;
* required fields;
* enum values;
* rejection of unknown fields when the response contract disallows them.

Parser non-responsibilities:

* evidence validation;
* authority validation;
* investment semantics;
* claim truth;
* grounding status determination.

Those remain CI09 grounding validator responsibilities.

## Grounding handoff

Parser success is not grounding success.

Rules:

```text
parsed response candidate -> CI09 grounding validator
parser success != grounded
grounded status must be validator-computed
model-declared grounding status != source of truth
raw output must never bypass grounding validation
```

## Downstream eligibility

Future downstream eligibility:

| CI09 grounding status | Delivery eligibility |
|---|---|
| `grounded` | Eligible for future controlled downstream display after a delivery contract exists. |
| `grounded_with_mandatory_caveats` | Eligible only with preserved disclosures. |
| `partially_grounded` | Eligible only as bounded partial response with unsupported parts preserved. |
| `ungrounded` | Not delivery eligible. |
| `blocked` | Not delivery eligible. |

ME-CI10 implements no delivery.

## Invocation request validation requirements

The future request validator must be deterministic, fail-closed, and
machine-readable. It must produce structured issue codes rather than free-text
only failures. It must not call a provider while validating.

## Invocation result contract

Conceptual future result envelope:

| Field | Required | Semantics |
|---|---:|---|
| `invocation_result_identity` | yes | Result schema, artifact type, result id, generated timestamp. |
| `invocation_request_identity` | yes | Linked request id and hash. |
| `run_identity` | yes | Source Market Engine run identity. |
| `invocation_identity` | yes | Invocation id. |
| `attempt_identity` | yes | Attempt id and sequence. |
| `provider_identity` | yes | Provider used or blocked provider metadata. |
| `model_identity` | yes | Model identity used or blocked model metadata. |
| `invocation_state` | yes | Technical invocation state. |
| `timing_metadata` | yes | Start, end, latency, timeout policy. |
| `usage_metadata` | yes | Token or provider usage metadata when available. |
| `cost_metadata` | yes | Estimated or unavailable cost metadata. |
| `provider_trace_metadata` | yes | Provider request or trace IDs when available. |
| `raw_response_reference` | yes | Reference to raw response artifact when available. |
| `parser_state` | yes | Parser outcome. |
| `grounding_handoff_state` | yes | Whether CI09 validator was invoked and result reference. |
| `audit_context` | yes | Source SHA, policy versions, hashes, and validation evidence. |

## State machine

Success path:

```text
request_validated
  -> invocation_started
  -> response_received
  -> raw_response_persisted
  -> parse_pending
  -> parsed
  -> grounding_pending
  -> grounding_completed
```

Failure branches:

```text
request_blocked
invocation_timeout
provider_failure
rate_limited
response_empty
response_truncated
response_malformed
schema_invalid
parse_failed
grounding_ungrounded
grounding_blocked
```

No ambiguous success state is allowed.

## Persistence contract

Conceptual future local/non-production artifacts:

```text
invocation_request.json
invocation_attempt.json
raw_model_response.json
parser_result.json
grounding_result.json
invocation_summary.json
manifest.json
```

ME-CI10 creates none of these runtime artifacts.

## Auditability

Required audit evidence:

* source main SHA;
* source artifact identity;
* prompt package identity;
* invocation request ID;
* invocation ID;
* attempt ID;
* provider/model identity;
* provider request/trace ID;
* timestamps;
* timeout policy;
* retry policy;
* idempotency key;
* token usage;
* cost metadata;
* finish reason;
* parser status;
* grounding status;
* final downstream eligibility.

## Observability requirements

Future implementation must be able to measure:

```text
invocation_count
success_count
timeout_count
provider_error_count
rate_limit_count
parse_failure_count
schema_failure_count
grounding_ungrounded_count
grounding_blocked_count
token_usage
estimated_cost
latency
```

ME-CI10 implements no monitoring runtime.

## Failure taxonomy

| Failure | Retryable? | Raw response available? | Grounding invoked? | Delivery eligible? | Severity |
|---|---|---:|---:|---:|---|
| `pre_invocation_contract_failure` | no | no | no | no | error |
| `policy_failure` | no | no | no | no | error |
| `budget_failure` | no | no | no | no | error |
| `transport_failure` | policy-dependent | no | no | no | error |
| `timeout_failure` | policy-dependent | no | no | no | error |
| `provider_failure` | policy-dependent | maybe | no | no | error |
| `rate_limit_failure` | policy-dependent | no | no | no | warning/error by policy |
| `empty_response_failure` | policy-dependent | yes | no | no | error |
| `truncated_response_failure` | policy-dependent | yes | no | no | error |
| `malformed_response_failure` | no automatic repair | yes | no | no | error |
| `schema_failure` | no automatic repair | yes | no | no | error |
| `parser_failure` | no automatic repair | yes | no | no | error |
| `grounding_ungrounded_failure` | no | yes | yes | no | error |
| `grounding_blocked_failure` | no | yes | yes | no | error |

## Fail-closed matrix

| Condition | Invocation | Retry | Parse | Grounding | Delivery eligibility |
|---|---|---|---|---|---|
| invalid source artifact | blocked | no | no | no | no |
| invalid prompt package | blocked | no | no | no | no |
| forbidden input | blocked | no | no | no | no |
| budget exceeded | blocked | no | no | no | no |
| timeout | failed | policy-dependent | no | no | no |
| provider error | failed | policy-dependent | no | no | no |
| empty response | failed | policy-dependent | no | no | no |
| truncated response | failed | policy-dependent | no successful parse | no | no |
| malformed JSON | failed | no automatic repair | parse failed | no | no |
| schema mismatch | failed | no automatic repair | parse failed | no | no |
| parsed but ungrounded | completed technically | no | yes | ungrounded | no |
| parsed but blocked | completed technically | no | yes | blocked | no |
| grounded | completed | no | yes | grounded | future eligible |
| caveated | completed | no | yes | grounded_with_mandatory_caveats | future eligible with caveats |
| partial | completed | no | yes | partially_grounded | future bounded eligibility |

## Implementation test matrix

Future implementation acceptance scenarios:

| Area | Scenario |
|---|---|
| Pre-invocation | valid source plus valid prompt package |
| Pre-invocation | invalid source |
| Pre-invocation | invalid prompt package |
| Pre-invocation | unsupported question class |
| Pre-invocation | authority boundary missing |
| Pre-invocation | budget exceeded |
| Pre-invocation | forbidden secret-bearing field |
| Invocation transport | success |
| Invocation transport | timeout |
| Invocation transport | provider 5xx |
| Invocation transport | rate limit |
| Invocation transport | empty body |
| Invocation transport | connection failure |
| Model output | valid structured output |
| Model output | model refusal |
| Model output | malformed JSON |
| Model output | truncated JSON |
| Model output | valid JSON wrong schema |
| Model output | valid schema ungrounded |
| Model output | valid schema blocked |
| Model output | valid grounded |
| Model output | valid caveated |
| Model output | valid partial |
| Retry/idempotency | retryable timeout |
| Retry/idempotency | non-retryable validation failure |
| Retry/idempotency | duplicate idempotency key |
| Retry/idempotency | second attempt preserves source lineage |
| Retry/idempotency | retry budget exhausted |

## Provider neutrality

ME-CI10 defines a provider-neutral boundary:

* common invocation envelope;
* common identity;
* common failure taxonomy;
* common raw response capture;
* common parser handoff;
* common grounding handoff.

Provider-neutral does not mean provider-specific metadata is lost. Provider
trace IDs, model identity, usage metadata, and capability differences must be
preserved.

## Model configuration profile

Conceptual controlled configuration profile:

```text
configuration_profile_id
structured_output_required
streaming_allowed
temperature_policy
max_output_tokens
seed_policy
tool_use_allowed
external_browsing_allowed
```

For v1:

```text
tool_use_allowed: false
external_browsing_allowed: false
streaming_allowed: false
```

## Tool-use boundary

Future model invocation for this advisory pipeline must not use external tools
unless a later sprint approves them.

V1 disabled tools:

```text
model tools: disabled
web browsing: disabled
code execution: disabled
broker tools: disabled
portfolio mutation tools: disabled
notification tools: disabled
```

The model response must be based only on supplied validated context.

## External browsing boundary

Model-side web browsing would break the closed grounding model.

Therefore v1 allows no browser tool, provider-side search, retrieval outside
the approved prompt package, or hidden external enrichment. Future
web-enrichment requires a separate contract.

## Provenance preservation

The model does not create new source lineage:

```text
model interpretation != new evidence source
model claim must map back to upstream evidence references
raw model response is not provenance for investment facts
```

## Nondeterminism boundary

Model output may be nondeterministic:

```text
same source run
same prompt package
same model
same config
different response possible
```

Future artifacts must preserve invocation identity, model identity,
configuration profile, raw output hash, parser result hash, grounding result,
and timestamps.

## Reproducibility

ME-CI10 distinguishes:

```text
reproducible inputs
traceable invocation
deterministic validation
```

from:

```text
identical model output
```

Identical model output is not guaranteed unless a future provider/model
contract proves it.

## Integrity requirements

Future implementation should support hashes for:

* prompt package;
* invocation request;
* raw response;
* parsed response;
* grounding result.

ME-CI10 does not implement hashing runtime.

## Implementation decision

ME-CI10 is docs-first and contract-only.

It adds no Python runtime, tests, provider adapter, SDK, environment variables,
API key loading, HTTP client, retry code, timeout code, token estimator, cost
calculator, parser runtime, model call, or grounding changes.

## Residual gaps

Residual gaps:

* no provider implementation;
* no model invocation;
* no API key management;
* no HTTP runtime;
* no retry runtime;
* no timeout runtime;
* no parser runtime;
* no production response artifact;
* no delivery;
* no full natural-language semantic scanner;
* no external browsing contract;
* no multi-provider routing;
* no fallback model policy.

## Recommended next sprint

Recommended next sprint:

```text
ME-CI11 - Implement controlled local model invocation adapter scaffold
```

ME-CI11 should remain non-production, single-provider, single-model-profile,
delivery-free, grounding-mandatory, and fail-closed. It should implement only
the minimum local adapter scaffold needed to exercise the CI10 boundary after
human review.
