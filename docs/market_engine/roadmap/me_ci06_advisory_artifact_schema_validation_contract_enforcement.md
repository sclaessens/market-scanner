# ME-CI06 - Advisory Artifact Schema Validation and Contract Enforcement Roadmap Entry

## Status

COMPLETED RUNTIME CONTRACT ENFORCEMENT

## Roadmap position

ME-CI06 follows ME-CI05:

```text
ME-RM06 - ChatGPT advisory delivery roadmap reposition
  -> ME-CI01 - Structured Decision Output contract
  -> ME-CI02 - ChatGPT Advisory Context Contract
  -> ME-CI03 - ChatGPT-readable Portfolio Intelligence Context
  -> ME-CI04 - Explainability / Change-Rationale Contract
  -> ME-CI05 - Daily ChatGPT-ready advisory artifact
  -> ME-CI06 - Advisory artifact schema validation and contract enforcement
  -> ME-CI07 - Prompt and response-grounding contract
  -> controlled advisory dry run
```

## Summary

ME-CI06 adds deterministic validation for CI05 ChatGPT-ready advisory artifacts.
It enforces top-level artifact shape, contract identity, embedded/referenced
context compatibility, cross-context ticker/run consistency, missingness
semantics, freshness semantics, validation metadata, and contextual forbidden
authority boundaries.

ME-CI06 makes this possible:

```text
validated deterministic ChatGPT-ready advisory artifact
```

ME-CI06 does not make this possible:

```text
ChatGPT advisory answer generation
```

## Runtime boundary

ME-CI06 validates and gates local artifact persistence only. It does not call
ChatGPT, execute prompts, fetch data, deliver notifications, modify portfolio
or watchlist state, contact brokers, or add allocation authority.

## Next

```text
ME-CI07 - Define ChatGPT advisory prompt and response-grounding contract
```

ME-CI07 should define prompt inputs, allowed wording, required disclosures,
refusal behavior, grounding checks, and response constraints before any live
ChatGPT runtime or notification delivery is introduced.
