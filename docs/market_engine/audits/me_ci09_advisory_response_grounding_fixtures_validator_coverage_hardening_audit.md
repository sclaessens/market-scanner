# ME-CI09 - Advisory Response Grounding Fixtures and Validator Coverage Hardening Audit

Owner roles: Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: COMPLETED LOCAL HARDENING SPRINT

## Objective

ME-CI09 hardens the ME-CI08 local advisory response grounding scaffold before a
future model-invocation boundary is introduced.

The sprint focus is:

```text
existing CI08 local scaffold
  -> coverage audit
  -> adversarial fixture matrix
  -> validator edge-case hardening
  -> issue taxonomy hardening
  -> deterministic regression coverage
  -> local dry-run regression evidence
```

## Architecture position

ME-CI09 keeps the approved advisory architecture:

```text
ME-CI05 advisory artifact
  -> ME-CI06 artifact validation
  -> ME-CI07 prompt/response-grounding contract
  -> ME-CI08 local grounding scaffold
  -> ME-CI09 fixture and validator hardening
  -> future model invocation boundary
```

ME-CI09 is not an investment reasoning engine. It does not generate advisory
content, call a model, execute prompts, score securities, allocate capital,
derive target weights, size positions, route orders, mutate portfolio state, or
change Decision Engine, Governor, Recommendation Review, or Portfolio Review
semantics.

## Inspected sources

The sprint inspected ME-CI07 prompt and response-grounding contracts and
examples, ME-CI08 audit and roadmap material, advisory prompt package,
grounding validator, dry-run runner, advisory package exports, CI08 advisory
tests, runtime fixture conventions, CI03 Portfolio Intelligence, CI04
Explainability / Change-Rationale, CI05 advisory artifact structure, CI06
validator semantics, Structured Decision Output, Governor, Dispatch, and the
central backlog and roadmap.

The source `main` revision at sprint start was:

```text
cad108cb06369864dce9868be4aba50602b131d0
```

## CI08 baseline

ME-CI08 introduced the local deterministic dry-run scaffold and fail-closed
grounding validator. It already covered envelope validation, basic claim and
evidence reference checks, restricted path resolution, disclosures, missing
context, authority violations, freshness, blocker omissions, contradictions,
and dry-run artifact persistence.

The CI09 audit found hardening gaps in duplicate evidence references,
support-type compatibility, declared grounding status consistency, broad parent
paths, context-family path containment, referenced versus embedded context,
partial-answer completeness, targeted summary consistency, mixed freshness,
blocker neutralization, Dispatch contradiction cherry-picking, subtle portfolio
misuse, and deterministic issue ordering.

## Coverage matrix

| Contract rule | Current implementation | Current test | Fixture coverage | Edge-case gap | False-positive risk | False-negative risk | Planned hardening |
|---|---|---|---|---|---|---|---|
| Material claims require references | Existing validator required references | CI08 grounding tests | Basic one-to-one | One-to-many and many-to-one source reuse not explicit | Shared paths could be over-rejected | Duplicate ambiguity could pass | Add one-to-many, many-to-one, duplicate, orphan, and wrong-ID fixtures |
| Evidence support type must fit claim type | CI08 validated enum only | Basic enum tests | Limited | Incompatible combinations passed | Overly strict matrix could reject valid interpretation | Causal or conditional claims could be under-supported | Add compatibility matrix |
| Evidence path must resolve safely | Restricted resolver existed | Basic path tests | Limited | Broad parent and family mismatch paths could pass | Rejecting valid arrays | Broad context object could ground material claim | Add path family containment and broad parent rejection |
| Context mode must be preserved | Missing context checks existed | Basic absent-context tests | Limited | Referenced-only context could be treated as embedded proof | Referenced metadata-only response could be over-rejected | Referenced or absent payload could prove facts | Reject material proof from referenced or absent context containers |
| Response mode must match content | Basic mode checks existed | CI08 mode tests | Limited | Partial answer and blocked content consistency gaps | Valid refusal/inability could be over-blocked | Hidden action content could pass | Add targeted mode and partial completeness checks |
| Response grounding summary is not source of truth | Validator computed status | Basic status mapping | Limited | Response-declared status mismatch was not flagged | Valid caveat status could be over-flagged | Response could claim grounded despite issues | Add `declared_grounding_status_mismatch` |
| Freshness disclosure is context relevant | Global freshness checks existed | Basic stale/unknown tests | Limited | Mixed family freshness not precise | Irrelevant stale family could force disclosure | Relevant stale family could be omitted | Use referenced families for family-level freshness checks |
| Blockers must remain visible | Blocker omission check existed | Basic omission tests | Limited | Summary could neutralize blockers | Generic prose could be over-blocked | Blocker minimization could pass | Add targeted neutralization phrases |
| Dispatch cannot override canonical SDO | Contradiction disclosure existed | Basic contradiction tests | Limited | Disclosed contradiction could still cherry-pick Dispatch | Real non-conflict disclosure should pass | Canonical state could be hidden | Add cherry-pick summary check |
| Issue ordering must be deterministic | Sorting existed | Implicit | Limited | Multi-error ordering not pinned | None found | Unordered additions could regress | Add deterministic issue-order test and sort key |

## Adversarial fixture strategy

ME-CI09 uses mutation-style fixtures over valid baseline advisory artifacts and
responses. This keeps each invalid case tied to one contract rule and avoids
large near-duplicate JSON fixture files.

The added hardening module covers valid tricky cases and invalid subtle cases
with parameterized tests. Existing CI08 tests continue to cover envelope,
schema, basic grounding, authority, dry-run, CLI, artifact persistence, and
status mapping behavior.

## Test matrix

| Fixture id | Question class | Response mode | Expected status | Expected issue codes | Blocking/non-blocking | Expected disclosure requirements | Expected dry-run state |
|---|---|---|---|---|---|---|---|
| CI09-A01 | current_state_explanation | advisory_interpretation | grounded | none | none | none | dry_run_completed_grounded |
| CI09-A02 | current_state_explanation | advisory_interpretation | grounded | none | none | none | dry_run_completed_grounded |
| CI09-A03 | current_state_explanation | advisory_interpretation | ungrounded | duplicate_evidence_reference | non-blocking ungrounded | none | dry_run_failed_ungrounded |
| CI09-A04 | current_state_explanation | advisory_interpretation | ungrounded | unknown_claim_reference | non-blocking ungrounded | none | dry_run_failed_ungrounded |
| CI09-A05 | current_state_explanation | advisory_interpretation | ungrounded | missing_evidence_reference, unknown_claim_reference | non-blocking ungrounded | none | dry_run_failed_ungrounded |
| CI09-A06 | change_rationale_question | advisory_interpretation | grounded_with_mandatory_caveats or ungrounded | support_type_incompatible or unsupported_causal_claim for invalid combinations | non-blocking ungrounded when invalid | causality or freshness disclosure when required | caveated or failed |
| CI09-A07 | current_state_explanation | advisory_interpretation | grounded | none | none | none | dry_run_completed_grounded |
| CI09-A08 | current_state_explanation | advisory_interpretation | ungrounded | evidence_path_not_allowed or evidence_path_not_found | non-blocking ungrounded | none | dry_run_failed_ungrounded |
| CI09-A09 | current_state_explanation | advisory_interpretation | ungrounded | evidence_path_not_allowed | non-blocking ungrounded | none | dry_run_failed_ungrounded |
| CI09-A10 | current_state_explanation | advisory_interpretation | ungrounded | evidence_path_not_allowed | non-blocking ungrounded | none | dry_run_failed_ungrounded |
| CI09-A11 | portfolio_context_question | partial_answer | ungrounded | evidence_path_not_allowed | non-blocking ungrounded | missing_portfolio_disclosure | dry_run_failed_ungrounded |
| CI09-A12 | portfolio_context_question | partial_answer | ungrounded | missing_context_used_as_fact | non-blocking ungrounded | missing_portfolio_disclosure | dry_run_failed_ungrounded |
| CI09-A13 | portfolio_context_question | advisory_interpretation | blocked or ungrounded | semantic_override_detected, unsupported_sizing_claim, or unsupported_allocation_claim | blocking or ungrounded | none | dry_run_blocked or dry_run_failed_ungrounded |
| CI09-A14 | change_rationale_question | advisory_interpretation | ungrounded | unsupported_causal_claim | non-blocking ungrounded | none | dry_run_failed_ungrounded |
| CI09-A15 | current_state_explanation | advisory_interpretation | grounded | none | none | none | dry_run_completed_grounded |
| CI09-A16 | portfolio_context_question | advisory_interpretation | ungrounded | required_disclosure_missing | non-blocking ungrounded | staleness_disclosure | dry_run_failed_ungrounded |
| CI09-A17 | current_state_explanation | advisory_interpretation | ungrounded | blocker_omission | non-blocking ungrounded | none | dry_run_failed_ungrounded |
| CI09-A18 | current_state_explanation | partial_answer | blocked | contradiction_not_disclosed | blocking | contradiction_disclosure | dry_run_blocked |
| CI09-A19 | current_state_explanation | advisory_interpretation | blocked or ungrounded | source_artifact_identity_mismatch, instrument_identity_mismatch, question_classification_mismatch, invalid_artifact_reference, run_identity_mismatch | blocking for identity, ungrounded for evidence lineage | none | blocked or failed |
| CI09-A20 | current_state_explanation | advisory_interpretation | ungrounded | declared_grounding_status_mismatch | non-blocking ungrounded | none | dry_run_failed_ungrounded |
| CI09-A21 | portfolio_context_question | partial_answer | ungrounded | partial_answer_incomplete | non-blocking ungrounded | missing_portfolio_disclosure | dry_run_failed_ungrounded |
| CI09-A22 | current_state_explanation | advisory_interpretation | blocked or ungrounded | deterministic multi-error set | deterministic by issue sort key | none | blocked or failed |

## Claim/reference graph coverage

ME-CI09 validates one claim with multiple valid evidence references, multiple
claims sharing one valid source path, orphan references to unknown claims,
material claims whose claim ID does not match the only reference, exact
duplicate evidence references, claim type mismatch through existing CI08 tests,
and evidence lineage mismatch by artifact reference and run ID.

Repository policy: exact duplicate evidence references are rejected with
`duplicate_evidence_reference`. The validator does not silently deduplicate
because duplicated evidence can hide fixture mistakes and make issue counts
ambiguous.

## Claim collection coverage

The validator continues to collect structured claim dictionaries from response
sections that can carry claim IDs. CI09 deliberately does not add a general
natural-language scanner over free-form `summary` text. Instead, it adds
targeted deterministic summary consistency checks for authority expansion,
descriptive-only upgrade wording, unsupported change-rationale statements,
blocked/inability affirmative conclusions, blocker neutralization, and Dispatch
cherry-picking.

Residual semantic misuse that cannot be tied to deterministic structured fields
remains outside ME-CI09 scope.

## Response mode consistency

ME-CI09 hardens `descriptive_only`, `partial_answer`, `unable_to_determine`,
`refused_outside_authority`, and `blocked_invalid_context` consistency.
Partial answers require at least one grounded supported part and explicit
`unable_to_determine` content when unsupported scopes or missing context remain.
Blocked and inability responses must not publish affirmative advisory
assessment content.

## Grounding summary consistency

ME-CI09 adds `declared_grounding_status_mismatch`. The deterministic validator
remains the source of truth. A response may include its own
`grounding_summary.status`, but that declaration cannot override the computed
validator status.

## Support-type compatibility

| Claim type | Accepted support types |
|---|---|
| direct_artifact_fact | direct |
| upstream_state_description | direct, summarized |
| evidence_summary | direct, summarized |
| explicit_upstream_reason | direct, summarized |
| supported_interpretation | direct, summarized, interpreted |
| conditional_interpretation | conditional |
| associated_change | associated_only, summarized |

`associated_only` remains prohibited as causal proof for reason-like claims.

## Path resolver edge cases

ME-CI09 keeps the restricted CI07-style resolver. It does not add JSONPath,
wildcards, recursive descent, fuzzy matching, nearest-key logic, or a new
dependency.

Coverage includes valid nested object paths, valid array index paths, array
index out of bounds, negative index syntax, wildcard syntax, recursive descent
syntax, empty paths, trailing dot syntax, wrong container type,
context-family path mismatch, and broad parent paths for material claims.

Material claims may not be grounded by broad parent context paths such as
`$.structured_decision_context`, `$.portfolio_intelligence_context`, or
`$.explainability_change_rationale_context`.

## Embedded/reference/absent handling

Embedded context payload paths may be used when the concrete path resolves.
Referenced context may preserve lineage/reference metadata but cannot be used
as local embedded proof for material claims. Absent context cannot be used as a
known-fact source.

## Portfolio boundary coverage

ME-CI09 adds tests and targeted checks for subtle portfolio misuse:

* known holding quantity does not prove full portfolio fit;
* cash balance does not prove deployable cash for sizing;
* current weight does not become target weight;
* concentration warnings do not become sell instructions.

No portfolio model, allocation model, target-weight engine, or sizing logic was
added.

## Explainability attribution coverage

ME-CI09 preserves CI04 boundaries. Explicit upstream reason claims require
compatible evidence. Associated-only support cannot prove causality.
Not-comparable or absent explainability cannot support unchanged, root-cause,
or single-main-reason summary claims. Materiality wording remains rejected
unless explicitly supported by the existing contract.

## Freshness coverage

ME-CI09 adds family-level relevance. Irrelevant stale family context does not
force an unnecessary disclosure. Relevant stale family context requires
`staleness_disclosure`. Unknown relevant family freshness requires
`unknown_freshness_disclosure`. Global stale or unknown state remains enforced.

## Blocker preservation coverage

ME-CI09 preserves source blockers and detects targeted neutralization in the
summary, including phrases such as no relevant blockers, no blockers, resolved
blockers, and main blocker. Existing blocker omission behavior remains.

The validator does not implement a general semantic blocker scanner.

## Contradiction coverage

ME-CI09 keeps Structured Decision Output as canonical over Dispatch
presentation context. Coverage includes undisclosed conflict, disclosed
conflict that still cherry-picks Dispatch in the summary, and valid disclosed
contradiction cases from existing CI08 tests.

The validator does not add majority voting, source voting, score averaging, or
cross-family contradiction inference outside the CI07/CI08 Dispatch boundary.

## Lineage coverage

ME-CI09 tests schema version mismatch, artifact type mismatch, source run ID
mismatch, instrument ticker mismatch, question class mismatch, evidence
artifact reference mismatch, and evidence run ID mismatch.

Identity mismatches remain blocking where they invalidate the response/source
relationship. Evidence reference lineage mismatches remain ungrounded failures.

## Issue taxonomy audit

| Issue code | Issue family | Blocking? | Ungrounded? | Warning? | Current test coverage | Fixture coverage | Deterministic order? |
|---|---|---:|---:|---:|---|---|---|
| missing_required_field | envelope | no | yes | no | CI08 | existing | yes |
| invalid_field_type | envelope | no | yes | no | CI08 | existing | yes |
| invalid_enum_value | envelope | no | yes | no | CI08 | existing | yes |
| source_artifact_identity_mismatch | identity | yes | yes | no | CI08 + CI09 | lineage matrix | yes |
| instrument_identity_mismatch | identity | yes | yes | no | CI08 + CI09 | lineage matrix | yes |
| question_classification_mismatch | identity | yes | yes | no | CI08 + CI09 | lineage matrix | yes |
| response_mode_invalid | envelope | no | yes | no | CI08 | existing | yes |
| duplicate_claim_id | grounding | no | yes | no | CI08 | existing | yes |
| duplicate_evidence_reference | grounding | no | yes | no | CI09 | duplicate ref | yes |
| missing_evidence_reference | grounding | no | yes | no | CI08 + CI09 | wrong claim ID | yes |
| unknown_claim_reference | grounding | no | yes | no | CI08 + CI09 | orphan ref | yes |
| claim_type_mismatch | grounding | no | yes | no | CI08 | existing | yes |
| support_type_incompatible | grounding | no | yes | no | CI09 | support matrix | yes |
| invalid_context_family | grounding | no | yes | no | CI08 | existing | yes |
| invalid_artifact_reference | grounding | no | yes | no | CI08 + CI09 | lineage matrix | yes |
| run_identity_mismatch | grounding | no | yes | no | CI08 + CI09 | lineage matrix | yes |
| evidence_path_not_found | grounding | no | yes | no | CI08 + CI09 | path matrix | yes |
| evidence_path_not_allowed | grounding | no | yes | no | CI08 + CI09 | path/family/context modes | yes |
| missing_context_used_as_fact | missingness | no | yes | no | CI08 + CI09 | absent context | yes |
| required_disclosure_missing | disclosure | no | yes | no | CI08 + CI09 | freshness/support | yes |
| forbidden_claim_type | authority | yes | yes | no | CI08 | existing | yes |
| authority_violation | authority | yes | yes | no | CI08 | existing | yes |
| unsupported_sizing_claim | authority | no | yes | no | CI08 + CI09 | portfolio misuse | yes |
| unsupported_allocation_claim | authority | no | yes | no | CI08 + CI09 | portfolio misuse | yes |
| unsupported_execution_claim | authority | no | yes | no | CI08 | existing | yes |
| unsupported_causal_claim | explainability | no | yes | no | CI08 + CI09 | support/explainability | yes |
| unsupported_materiality_claim | explainability | no | yes | no | CI08 | existing | yes |
| freshness_conflict | freshness | no | yes | no | CI08 | existing | yes |
| blocker_omission | blocker | no | yes | no | CI08 + CI09 | neutralization | yes |
| contradiction_not_disclosed | contradiction | yes | yes | no | CI08 + CI09 | cherry-pick | yes |
| semantic_override_detected | authority/mode | yes | yes | no | CI08 + CI09 | summary/portfolio | yes |
| recommendation_remapping_detected | authority | yes | yes | no | CI08 | existing | yes |
| declared_grounding_status_mismatch | grounding | no | yes | no | CI09 | summary mismatch | yes |
| partial_answer_incomplete | mode | no | yes | no | CI09 | partial answer | yes |

## Status mapping audit

| Issue class | Grounding status | Dry-run state | CLI exit |
|---|---|---|---:|
| No issue | grounded | dry_run_completed_grounded | 0 |
| Mandatory caveat only | grounded_with_mandatory_caveats | dry_run_completed_with_caveats | 0 |
| Valid partial response | partially_grounded | dry_run_completed_partial | 0 |
| Ungrounded issue | ungrounded | dry_run_failed_ungrounded | non-zero |
| Blocking issue | blocked | dry_run_blocked | non-zero |

Blocking issue codes remain dominant over non-blocking ungrounded issue codes.

## False-positive audit

Confirmed valid cases preserved include one claim with multiple valid evidence
references, multiple claims sharing one valid evidence path, interpreted support
for supported interpretations, conditional support for conditional
interpretations, associated-only support for associated changes with required
disclosure, valid partial answers from existing CI08 tests, irrelevant stale
portfolio family for current-state questions, valid descriptive-only and
refusal/inability cases from existing CI08 tests, absent portfolio context for
non-portfolio current-state questions, absent Dispatch when irrelevant, and
absent explainability for current-state questions.

The main false-positive reduction is family-level freshness relevance: unused
stale context no longer forces a disclosure.

## False-negative audit

Confirmed and fixed gaps include duplicate evidence references, support-type
incompatibility, broad parent evidence paths, evidence path family mismatch,
referenced-only context used as embedded proof, absent context used as known
fact, declared grounding status mismatch, partial answer with missing
`unable_to_determine`, blocker neutralization in summary, Dispatch
contradiction cherry-picking, subtle portfolio-fit, target-weight,
deployable-cash, and sell-instruction misuse, and unsupported explainability
summary wording.

Not deterministically solvable in ME-CI09: comprehensive free-form prose
semantic scanning, general causality inference, general materiality inference,
arbitrary cross-family contradiction discovery, and model-output quality
judgment.

## Partial-answer completeness

ME-CI09 requires partial answers to include at least one grounded supported
part, at least one explicit unsupported part, and `unable_to_determine` content
when required context is missing. This preserves partial answers as bounded
responses rather than disguised complete answers.

## Determinism

ME-CI09 adds a deterministic issue sort key:

```text
path, code, claim_id, issue_family
```

A multi-error regression fixture validates that repeated validation of the same
input produces identical status, issue codes, paths, claim IDs, issue families,
and payload order.

## Dry-run regression evidence

The required local dry-run smoke matrix for final validation is:

| Smoke | Expected grounding status | Expected dry-run state | Expected exit |
|---|---|---|---:|
| grounded | grounded | dry_run_completed_grounded | 0 |
| caveated | grounded_with_mandatory_caveats | dry_run_completed_with_caveats | 0 |
| partial | partially_grounded | dry_run_completed_partial | 0 |
| ungrounded evidence misuse | ungrounded | dry_run_failed_ungrounded | non-zero |
| blocked authority violation | blocked | dry_run_blocked | non-zero |
| blocked contradiction | blocked | dry_run_blocked | non-zero |

Each smoke must validate JSON output with `json.tool`, confirm the manifest,
and compare grounding status, dry-run state, issue count, and exit code.

## Artifact persistence regression

ME-CI09 relies on the ME-CI08 artifact persistence layer and revalidates it
through smoke runs. Successful states write auditable dry-run artifacts. Failed
states preserve grounding results and issues. Successful grounded artifacts are
not created for failed states. Manifests match dry-run summaries. Issue counts
match grounding result payloads. Reruns with identical run IDs fail safe through
overwrite protection.

## Runtime changes

Runtime changes are limited to:

```text
src/market_engine/advisory/advisory_response_grounding.py
```

The sprint adds targeted validator checks only where the CI09 coverage audit
found a deterministic gap. No broad refactor, dependency, JSONPath library,
NLP dependency, model abstraction, provider integration, or delivery layer was
added.

## Governance boundary

ME-CI09 remains:

```text
local-only
deterministic
model-free
provider-free
delivery-free
non-production
fail-closed
```

It does not introduce OpenAI API usage, ChatGPT API usage, model invocation,
prompt execution, temperature, token limits, streaming, provider retry,
external network calls, live prices, Telegram, email, Messenger, Signal,
broker integration, order generation, order routing, portfolio mutation,
watchlist mutation, allocation, target weights, position sizing, execution
advice, scheduler behavior, UI behavior, or autonomous loops.

## Residual gaps

Residual gaps intentionally preserved:

* no comprehensive natural-language semantic scanner;
* no real model invocation;
* no prompt template execution;
* no production response artifact;
* no full JSONPath support;
* no causal reasoning engine;
* no materiality reasoning engine;
* no arbitrary cross-context contradiction engine;
* no channel delivery.

These gaps should not be patched with brittle prose scanning. Future sprints
should define stronger structured contracts when broader semantic coverage is
required.

## Recommended next sprint

```text
ME-CI10 - Define controlled model invocation boundary contract
```

ME-CI10 should be contract-only. It should define model invocation ownership,
allowed inputs, forbidden inputs, response capture, redaction, retry limits,
failure states, artifact persistence, grounding handoff, and governance
controls before any implementation sprint may call a model.
