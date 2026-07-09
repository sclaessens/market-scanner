# ME-CI11 - First Real Grounded Advisory Output Roadmap Entry

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor / Operator

Status: NEXT ACTIVE ADVISORY SPRINT

## Roadmap position

ME-CI11 follows the completed advisory contract and validation chain:

```text
ME-CI05 - Daily ChatGPT-ready advisory artifact
  -> ME-CI06 - Advisory artifact validation
  -> ME-CI07 - Prompt and response-grounding contract
  -> ME-CI08 - Local synthetic advisory dry-run scaffold
  -> ME-CI09 - Grounding validator hardening
  -> ME-CI10 - Controlled model invocation boundary contract
  -> ME-CI11 - First real grounded advisory output
```

## Purpose

ME-CI11 must produce the first real user-readable advisory output from the application for one real ticker.

The sprint is output-first. It is not complete when an adapter, parser, provider wrapper, or invocation abstraction merely exists.

## Required path

```text
real Market Engine ticker artifact
  -> source validation
  -> deterministic prompt package
  -> controlled non-production model invocation
  -> raw response capture
  -> strict structured parse
  -> mandatory CI09 grounding validation
  -> readable advisory_report.md
```

## Hard Definition of Done

ME-CI11 is complete only when:

- one real ticker artifact is selected from actual Market Engine output;
- the source artifact is valid for the controlled advisory path;
- the prompt package is built from approved context;
- one real model invocation occurs under CI10 boundaries;
- invocation metadata and raw response are persisted locally;
- the raw response is strictly parsed into the CI07 response shape;
- CI09 grounding validation executes on the parsed response;
- failed, malformed, ungrounded, or blocked responses do not produce a successful advisory report;
- at least one grounded path produces `advisory_report.md`;
- the report is readable without opening the JSON internals;
- the report remains traceable to source artifact, prompt package, invocation, raw response, parsed response, grounding result, and manifest.

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
ME-CI12 - Render grounded response as a high-quality human-readable stock report
```

ME-CI12 must use CI11 real-run evidence to improve report readability and consistency. It must not reopen pre-output architecture loops.