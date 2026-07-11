# ME-GH01 - Default Provider Invocation Disabled

## Status

Active baseline guardrail.

## Decision

The grounded advisory command path must not call the OpenAI API by default.

The Market Engine baseline is GitHub-first and no-paid-API:

```text
source artifacts
  -> deterministic readiness/status/index/ranking artifacts
  -> ChatGPT interactive interpretation
```

Provider-generated advisory reports are optional/deferred and are not part of the current baseline roadmap.

## Practical consequence

Running:

```text
python -m market_engine.advisory.grounded_advisory_output_command ...
```

must fail closed before provider invocation unless a deterministic test invoker is injected by tests.

It must not use `OPENAI_API_KEY`, `MARKET_ENGINE_ADVISORY_MODEL`, or `OPENAI_MODEL` as an automatic trigger for real provider calls.

## Next baseline work

Continue with:

```text
ME-GH02 - Batch artifact discovery and ticker status index
```

Do not resume ME-CI11D / provider invocation work unless the user explicitly re-approves paid API usage later.
