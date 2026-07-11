# ME-CI11 - First Real Grounded Advisory Output Roadmap Entry

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: IMPLEMENTED WITH REAL INVOCATION BLOCKED BY LOCAL CONFIGURATION; PR REVIEW REMEDIATION APPLIED

## Roadmap position

ME-CI11 follows the completed advisory contract and validation chain:

```text
ME-CI05 - Daily ChatGPT-ready advisory artifact
  -> ME-CI06 - Advisory artifact validation
  -> ME-CI07 - Prompt and response-grounding contract
  -> ME-CI08 - Local synthetic advisory dry-run scaffold
  -> ME-CI09 - Grounding validator hardening
  -> ME-CI10 - Controlled model invocation boundary contract
  -> ME-CI11 - First real grounded advisory output flow and blocked real invocation evidence
  -> ME-CI11B - Configured real grounded model invocation
```

## Purpose

ME-CI11 must produce the first user-readable advisory output flow from the application for one real ticker and persist fail-closed evidence when a real provider invocation cannot be configured.

The sprint is output-first. It is not complete when an adapter, parser, provider wrapper, or invocation abstraction merely exists.

ME-CI11 selected a real NVDA Market Engine artifact and produced a local blocked advisory report. The real model response portion of the original hard Definition of Done remains open because the local environment did not provide `OPENAI_API_KEY` and did not provide `MARKET_ENGINE_ADVISORY_MODEL` or `OPENAI_MODEL`.

## Required path

```text
real Market Engine ticker artifact
  -> source validation
  -> deterministic evidence catalog and prompt package
  -> deterministic CI10 pre-invocation validation
  -> controlled non-production model invocation
  -> raw provider response capture
  -> strict structured parse
  -> structured claim/evidence projection
  -> mandatory ME-CI09 grounding validation
  -> readable advisory_report.md
```

## Hard Definition of Done

ME-CI11 is complete only when:

- one real ticker artifact is selected from actual Market Engine output;
- the source artifact is valid for the controlled advisory path;
- the prompt package is built from approved context;
- one real model invocation occurs under CI10 boundaries;
- invocation metadata and raw response are persisted locally;
- the raw response is strictly parsed into the approved structured response shape;
- ME-CI09 grounding validation executes on the parsed response projection;
- failed, malformed, ungrounded, or blocked responses do not produce a successful advisory report;
- at least one grounded path produces `advisory_report.md`;
- the report is readable without opening the JSON internals;
- the report remains traceable to source artifact, prompt package, invocation, raw response, parsed response, grounding result, and manifest.

## Implementation note

The first PR implementation used a parallel evidence-reference containment validator. PR review classified that as a merge blocker because it bypassed the established ME-CI09 grounding authority.

The PR branch was corrected before merge:

```text
strict parser
  -> explicit claim/evidence linkage
  -> ME-CI09 validate_advisory_response_grounding(...)
  -> accepted advisory output
```

The remediation also added exact evidence source paths, deterministic CI10 request validation, provider-side strict structured output, bounded output tokens, model/provider/configuration-sensitive idempotency, expanded raw provider response capture, safe output path containment, and stronger regression coverage.

The selected NVDA run still failed closed at invocation configuration with `blocked_invocation_not_configured`. The persisted run evidence therefore remains blocker evidence, not successful model-output evidence.

The first successful real model output is explicitly moved to ME-CI11B rather than being silently treated as complete.

## Required visible content

The first report should, where supported by source evidence, present:

- concise current-state summary;
- supporting evidence;
- opposing evidence;
- blockers;
- uncertainty;
- freshness caveats;
- current upstream recommendation state and interpretation, when available;
- what would improve the case;
- what would weaken the case;
- what cannot be determined from available evidence.

The report must preserve descriptive-only, partial, blocked, missing, stale, unknown, and not-comparable states.

## Governance boundary

ME-CI11 remains:

- non-production;
- single-provider;
- single-model-profile;
- non-streaming;
- tool-free;
- browsing-free;
- local artifact output only;
- delivery-free;
- broker-free;
- portfolio-write-free;
- watchlist-write-free;
- allocation-free;
- sizing-free;
- execution-free;
- fail-closed.

No generic multi-provider routing, fallback-model policy, notification adapter, autonomous loop, or theoretical hardening sprint may displace this visible output milestone unless the actual CI11 run exposes a concrete blocker.

## Next

```text
ME-CI11B - Execute configured real grounded advisory model invocation
```

ME-CI11B must use the corrected ME-CI11 runtime and approved non-production model configuration to obtain, strictly parse, ME-CI09-ground, validate, and render one successful real provider response before ME-CI12 starts report-quality polishing.
