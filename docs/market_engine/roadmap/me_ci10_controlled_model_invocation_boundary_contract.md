# ME-CI10 - Controlled Model Invocation Boundary Contract Roadmap Entry

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: COMPLETED DOCS-FIRST CONTRACT

## Roadmap position

ME-CI10 follows the completed advisory artifact, prompt package, dry-run, and
grounding hardening chain:

```text
ME-CI05 - Daily ChatGPT-ready advisory artifact
  -> ME-CI06 - Advisory artifact schema validation and contract enforcement
  -> ME-CI07 - ChatGPT advisory prompt and response-grounding contract
  -> ME-CI08 - Controlled advisory response dry run and grounding validator scaffold
  -> ME-CI09 - Advisory response grounding fixtures and validator coverage hardening
  -> ME-CI10 - Controlled model invocation boundary contract
```

## Purpose

ME-CI10 defines how a future model invocation may be requested, bounded,
provenanced, persisted, parsed, grounded, and fail-closed without making raw
model output directly delivery eligible.

## Governance boundary

ME-CI10 is docs-only. It introduces no runtime provider call, SDK, API key,
HTTP client, retry loop, timeout handling, prompt execution, parser runtime,
delivery path, broker integration, portfolio/write behavior, watchlist/write
behavior, allocation, sizing, target weight, execution, scheduler, UI, or
autonomous loop.

## Output-first roadmap pivot

ME-CI10 closes the contract-building phase for the first advisory-output milestone.
The next sprint is no longer defined as a generic adapter scaffold. The roadmap
now requires a visible user-facing result before additional advisory
infrastructure or hardening work is added.

The governing product rule is:

```text
Every next sprint must create a new visible result that the user can inspect or use.
A sprint that adds only a contract, validator, taxonomy, scaffold, or boundary is
not allowed ahead of the first real grounded advisory report unless a concrete
blocking defect from an actual run proves it necessary.
```

The approved output-first sequence is:

```text
ME-CI11 - Generate first real grounded advisory output from a Market Engine artifact
  -> ME-CI12 - Render grounded response as a high-quality human-readable stock report
  -> ME-CI13 - Run real advisory reports for NVDA, AMD, and ASML
  -> ME-CI14 - Generate portfolio-aware advisory output from approved portfolio context
  -> ME-CI15 - Expose advisory output through the application interface
```

## Next

```text
ME-CI11 - Generate first real grounded advisory output from a Market Engine artifact
```

ME-CI11 is an implementation-and-run sprint. Its Definition of Done is not that
an adapter or invocation boundary exists. It is complete only when one real
Market Engine ticker artifact has traversed the controlled invocation,
structured parsing, mandatory CI09 grounding validation, and local report
rendering path, producing a readable `advisory_report.md` together with its
source, prompt package, invocation, raw response, parsed response, grounding
result, and manifest lineage.

ME-CI11 remains non-production, single-provider, single-model-profile,
non-streaming, tool-free, browsing-free, delivery-free, broker-free,
portfolio/write-free, allocation-free, sizing-free, and fail-closed. Failed,
ungrounded, or blocked responses must not produce a successful advisory report.