# ME-GH01 - Disable Default Provider Invocation Audit

## Objective

Remove the accidental baseline behavior where the grounded advisory command could call the OpenAI Responses API from environment configuration.

The GitHub-first baseline must prepare deterministic Market Engine artifacts and must not require, attempt, or encourage paid OpenAI API calls.

## Context

ME-GH01 locked the current baseline direction:

```text
GitHub/local runtime prepares data and deterministic analysis artifacts
  -> ChatGPT interprets those artifacts interactively
  -> provider-based advisory output remains optional/deferred
```

The existing `grounded_advisory_orchestration.py` still contained this default path:

```text
valid source + valid request + no injected invoker
  -> OpenAIResponsesInvoker.from_environment().invoke(...)
```

That meant a normal command could attempt an API call whenever `OPENAI_API_KEY` and a model variable were present. That behavior conflicted with the GitHub-first no-API baseline.

## Changes

- Updated `src/market_engine/advisory/grounded_advisory_orchestration.py` so the default command path no longer falls back to `OpenAIResponsesInvoker.from_environment()`.
- The default path now fail-closes with `request_blocked` and an explicit message that provider invocation is disabled by default for the GitHub-first no-API baseline.
- Explicit test invokers remain supported so deterministic fixture tests can still validate parser, CI09 grounding, reporting, and artifact persistence without network calls.
- Added `tests/market_engine/advisory/test_grounded_advisory_no_api_baseline.py` to assert that even with `OPENAI_API_KEY` and `MARKET_ENGINE_ADVISORY_MODEL` present, the default path does not call `urllib.request.urlopen`.

## Guardrail

Baseline commands must not call the OpenAI API.

Provider invocation may only return in a future explicit, user-approved, optional provider sprint. It must not be part of the 500-ticker baseline path, ME-GH02 through ME-GH06, or any GitHub-first batch execution.

## Expected status behavior

Default grounded advisory command path:

```text
invocation_state = request_blocked
advisory_status = blocked_invocation_not_configured
validation_status = invalid
raw_provider_response = null
```

This is intentional for the no-API baseline.

## Validation to run locally

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/market_engine/advisory/grounded_advisory_output.py \
  src/market_engine/advisory/grounded_advisory_orchestration.py \
  src/market_engine/advisory/grounded_advisory_runtime.py \
  src/market_engine/advisory/grounded_advisory_output_command.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/market_engine/advisory/test_grounded_advisory_no_api_baseline.py \
  tests/market_engine/advisory/test_grounded_advisory_output.py -q

git diff --check
```

## Outcome

The accidental baseline provider invocation path has been removed from the default code path.

The optional provider adapter class still exists as historical/optional CI11 implementation code, but the normal command path no longer uses it automatically.
