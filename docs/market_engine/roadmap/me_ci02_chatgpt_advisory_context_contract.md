# ME-CI02 - ChatGPT Advisory Context Contract Roadmap Entry

Sprint ID: ME-CI02

Status: COMPLETED DOCS-ONLY CONTRACT

Job family: ME-CI / ChatGPT Advisory Integration

Date: 2026-07-07

## Roadmap position

ME-CI02 follows ME-CI01:

```text
ME-DS01
  -> ME-RM06
  -> ME-CI01 - Structured Decision Output contract
  -> ME-CI02 - ChatGPT Advisory Context Contract
  -> ME-CI03 - ChatGPT-readable Portfolio Intelligence context
```

## Roadmap decision

ME-CI02 defines `chatgpt-advisory-context-v1` as the controlled, evidence-backed
context envelope for a future ChatGPT Advisory Layer.

The context composes Structured Decision Output, Governor context, Dispatch
Station context, provenance, freshness, uncertainty, readiness, blockers, and
consumer rules. It does not implement a ChatGPT API call, prompt runner,
notification adapter, runtime context assembler, or production delivery flow.

## Gate for ME-CI03

ME-CI03 may define ChatGPT-readable Portfolio Intelligence context only inside
the ME-CI02 advisory boundary.

ME-CI03 must not allow ChatGPT to invent missing portfolio state, holdings,
exposure, cash, position size, target weight, allocation, or add/reduce
guidance.

## Next sprint

```text
ME-CI03 - ChatGPT-readable Portfolio Intelligence context
```
