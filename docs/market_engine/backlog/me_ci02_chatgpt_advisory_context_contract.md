# ME-CI02 - ChatGPT Advisory Context Contract

Sprint ID: ME-CI02

Status: COMPLETED DOCS-ONLY CONTRACT

Job family: ME-CI / ChatGPT Advisory Integration

Date: 2026-07-07

## Summary

ME-CI02 defines `chatgpt-advisory-context-v1`, the controlled context envelope
that may be supplied to a future ChatGPT Advisory Layer.

## Context

ME-CI02 follows ME-CI01 Structured Decision Output. It defines how Structured
Decision Output, Governor context, Dispatch Station context, provenance,
freshness, uncertainty, readiness, blockers, and consumer boundaries may be
composed for advisory interpretation.

## Contract outputs

ME-CI02 adds:

* ChatGPT Advisory Context contract;
* eligible, descriptive-only, and blocked example artifacts;
* audit documentation;
* roadmap/backlog synchronization.

## Acceptance criteria

* Contract identity and version are explicit.
* Advisory eligibility states are defined.
* Provenance, freshness, uncertainty, readiness, and blockers are represented.
* Structured Decision Output consumption is defined.
* Governor and Dispatch Station boundaries are documented.
* Portfolio and recommendation boundaries are explicit.
* Prohibited inputs and prohibited inferences are listed.
* Fail-closed behavior is documented.
* Examples exist for eligible, descriptive-only, and blocked contexts.
* No runtime code is changed.

## Validation

Required validation:

```text
git diff --check
git status --short
git diff --stat
.venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_advisory_context_v1_eligible.json
.venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_advisory_context_v1_descriptive_only.json
.venv/bin/python -m json.tool docs/market_engine/contracts/examples/chatgpt_advisory_context_v1_blocked.json
```

Runtime tests are not required because this is docs-only.

## Follow-up backlog

* ME-CI03 - ChatGPT-readable Portfolio Intelligence context.
* ME-CI04 - explainability/change-rationale contract.
* ME-CI05 - daily ChatGPT-ready advisory artifact.
* ME-PI01 - Portfolio Intelligence exposure contract.
* ME-PS01 - Position Sizing decision contract.
* ME-NL01 - channel-neutral Notification Layer contract.
