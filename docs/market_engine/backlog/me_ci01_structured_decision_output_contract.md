# ME-CI01 - Structured Decision Output Contract

Sprint ID: ME-CI01

Status: COMPLETED DOCS-FIRST CONTRACT

Job family: ME-CI / ChatGPT Advisory Integration

Date: 2026-07-07

## Summary

ME-CI01 defines Structured Decision Output v1 as the machine-readable interface
between Market Engine decision artifacts and consumers such as ChatGPT Advisory
Layer, Notification Layer, dashboards, future frontends, and audit/replay
tooling.

## Context

ME-RM06 established that ChatGPT Advisory Layer is the primary interactive
interface above reproducible Market Engine artifacts. Structured Decision Output
must therefore precede ChatGPT context, Portfolio Intelligence, Position Sizing,
and notification adapter work.

## Contract outputs

ME-CI01 adds:

* `docs/market_engine/contracts/me_ci01_structured_decision_output_contract.md`;
* `docs/market_engine/contracts/examples/structured_decision_output_v1_actionable_candidate.json`;
* `docs/market_engine/contracts/examples/structured_decision_output_v1_blocked_descriptive_only.json`;
* ME-CI01 audit documentation;
* roadmap and backlog synchronization.

## Acceptance criteria

* Structured Decision Output v1 is versioned.
* Contract has a top-level object and field semantics.
* Required and optional fields are identified.
* Fail-closed behavior is described.
* Consumer rules for ChatGPT Advisory Layer, Notification Layer, and dashboards
  are described.
* Coverage, readiness, and actionability are explicit.
* Scores can be null while engines are absent.
* Consumers may not invent missing values.
* Contract supports future Conviction, Position Sizing, and Portfolio
  Intelligence.
* Example artifacts exist.
* Roadmap and backlog are synchronized.
* No runtime code is changed.

## Validation

Required validation:

```text
git diff --check
rg validation for Structured Decision Output / ChatGPT Advisory Layer / Notification Layer / fail-closed terms
python -m json.tool docs/market_engine/contracts/examples/structured_decision_output_v1_actionable_candidate.json
python -m json.tool docs/market_engine/contracts/examples/structured_decision_output_v1_blocked_descriptive_only.json
```

Runtime tests are not required for this docs-first contract sprint.

## Follow-up backlog

* ME-CI02 - ChatGPT Advisory Context Contract.
* ME-CI03 - ChatGPT-readable Portfolio Intelligence context.
* ME-CI04 - explainability/change-rationale contract.
* ME-CI05 - daily ChatGPT-ready advisory artifact.
* ME-PI01 - Portfolio Intelligence exposure contract.
* ME-PS01 - Position Sizing decision contract.
* ME-NL01 - channel-neutral Notification Layer contract.
