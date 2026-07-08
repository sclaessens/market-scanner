# ME-CI05 - Daily ChatGPT-ready Advisory Artifact Audit

## Objective

ME-CI05 produces the first controlled runtime composition layer for the
ChatGPT advisory architecture:

```text
schema_version: market-engine-chatgpt-ready-advisory-artifact-v1
artifact_type: market-engine-chatgpt-ready-advisory-artifact
```

The artifact is deterministic, local, JSON-only, and fail-closed. It composes
already-approved Market Engine artifacts into a ChatGPT-ready advisory artifact
without calling ChatGPT, generating prompts, fetching market data, mutating
portfolio or watchlist state, delivering messages, producing broker payloads,
or changing Decision Engine authority.

## Start state

ME-CI05 started from fresh `main` after ME-CI04 was merged through PR #436.

Source main SHA:

```text
ea3ad8218201beb2894974c753edd8e48cd97ee2
```

## Inspected upstream contracts and runtime conventions

Inspected sources included:

* ME-CI01 Structured Decision Output contract and examples;
* ME-CI02 ChatGPT Advisory Context contract and examples;
* ME-CI03 Portfolio Intelligence Context contract and examples;
* ME-CI04 Explainability / Change-Rationale Context contract and examples;
* ME-RUN04 and ME-RUN05 local dry-run artifact persistence contracts and
  implementation;
* end-to-end dry-run command and payload conventions;
* source, observation, setup, analysis, recommendation, portfolio, handoff,
  Governor, Dispatch Station, delivery/reporting, and output-report contract
  families;
* active architecture and governance documentation.

## Composition map

ME-CI05 composes only approved local JSON artifacts:

| Input family | Runtime handling | Authority |
| --- | --- | --- |
| Structured Decision Output | required; embedded as canonical structured decision context | source of truth for decision-facing structured state |
| ChatGPT Advisory Context | optional; embedded when schema and identity validate | source of truth for advisory eligibility and advisory boundary |
| Portfolio Intelligence Context | optional; embedded when schema and identity validate | source of truth for portfolio-specific advisory context |
| Explainability / Change-Rationale Context | optional; embedded when schema and identity validate | source of truth for current-state explanation and change rationale |
| Governor context | optional; embedded when identity validates | source of truth for supplied Governor state only |
| Dispatch context | optional; embedded when identity validates | presentation-only context, never an override |

## Runtime files

ME-CI05 adds:

```text
src/market_engine/advisory/advisory_artifact.py
src/market_engine/advisory/daily_artifact.py
tests/market_engine/advisory/test_advisory_artifact.py
```

The assembler exposes:

* `assemble_chatgpt_ready_advisory_artifact`;
* `load_chatgpt_ready_advisory_inputs`;
* `persist_chatgpt_ready_advisory_artifact`;
* `compose_chatgpt_ready_advisory_artifact_from_directory`.

The command module exposes:

```text
PYTHONPATH=src python -m market_engine.advisory.daily_artifact \
  --input-artifact-dir <path> \
  --output-dir <path> \
  --generated-at <timestamp>
```

Input discovery is intentionally non-ambiguous. The required input filename is:

```text
structured_decision_output.json
```

Optional companion filenames are:

```text
chatgpt_advisory_context.json
chatgpt_portfolio_intelligence_context.json
chatgpt_explainability_change_rationale_context.json
governor_context.json
dispatch_context.json
```

## Persistence contract

The local output path is:

```text
<output_root>/<run_id>/<ticker>/chatgpt_ready_advisory.json
<output_root>/<run_id>/<ticker>/manifest.json
```

The default output root is:

```text
artifacts/market_engine/chatgpt_ready_advisory
```

Persistence is local-only, non-production, JSON sorted-key deterministic, and
overwrite-protected unless explicitly allowed.

## Fail-closed behavior

Required Structured Decision Output failures raise a controlled error and do
not produce an artifact. Examples include unsupported schema version, unsupported
artifact type, missing run identity, missing ticker, and instrument ticker
conflicts.

Optional context conflicts produce a blocked artifact instead of silently
dropping or rewriting context. Examples include ticker conflicts, run identity
conflicts, and unsupported optional schema versions.

Composition states are:

```text
blocked_artifact_produced
descriptive_only_artifact_produced
eligible_artifact_produced
```

## Source precedence

ME-CI05 preserves this precedence order:

```text
Structured Decision Output > Dispatch presentation summary
Portfolio Intelligence Context > narrative text
Governor canonical state > Governor explanation summary
machine-readable blocker state > free text
```

Dispatch context is presentation-only. It can be included or absent, but it
cannot override Structured Decision Output or Governor state.

## Missing context behavior

ME-CI05 preserves absence without inventing facts:

* absent portfolio context stays absent and unavailable;
* absent portfolio context is not converted to `holdings=[]` as a fact;
* unavailable cash stays `null` / unavailable, not zero;
* absent explainability context leaves change rationale unavailable;
* a non-comparable explainability context remains `not_comparable`, not
  unchanged;
* absent Dispatch context is optional and not blocking.

## Advisory eligibility behavior

ME-CI05 does not upgrade upstream eligibility.

* blocked upstream state produces a blocked artifact;
* descriptive-only upstream state remains descriptive-only;
* eligible state is preserved only when required identity and schema validation
  pass;
* portfolio availability affects portfolio-specific advisory availability, not
  standalone Structured Decision Output validity;
* explainability availability affects change-rationale availability, not the
  existence of current-state Structured Decision Output.

## Freshness behavior

Freshness is represented per family. The artifact `generated_at` timestamp does
not refresh upstream evidence. Mixed upstream freshness remains mixed.

## Test coverage

ME-CI05 adds synthetic runtime tests for:

* eligible artifact assembly;
* descriptive-only no-upgrade behavior;
* blocked upstream behavior;
* optional portfolio absence;
* optional explainability absence;
* optional Dispatch absence;
* ticker conflicts;
* run identity conflicts;
* unsupported required and optional schema versions;
* Structured Decision Output instrument conflict;
* partial holdings without false `not_held` inference;
* unavailable cash not becoming zero;
* `not_comparable` explainability not becoming unchanged;
* Governor conflict;
* Dispatch non-override;
* mixed freshness;
* deterministic source reference ordering;
* deterministic assembly with reordered references;
* persistence path and manifest;
* overwrite protection;
* unsafe output roots and unsafe run IDs;
* explicit input loading;
* directory composition;
* CLI success and failure;
* side-effect dependency guard.

## Governance boundary

ME-CI05 does not introduce:

* market-data fetching;
* yfinance, SEC, EDGAR, or provider calls;
* ChatGPT, OpenAI, prompt, or LLM calls;
* Telegram, email, broker, portfolio, watchlist, scheduler, UI, or production
  writes;
* allocation, position sizing, target weight, order generation, or rebalancing;
* new recommendation, Governor, Portfolio Review, Decision Engine, Dispatch, or
  reporting semantics;
* temporal diff, causality, or materiality inference.

## Result

ME-CI05 creates a small deterministic composition layer that can produce a
daily ChatGPT-ready advisory artifact from explicit local Market Engine JSON
inputs while preserving the project doctrine:

```text
classification upstream
allocation downstream
Decision Engine = ONLY allocation authority
```
