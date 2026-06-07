# Project Charter

Status: ACTIVE
Reset stage: RESET-1

## Purpose

The market-scanner project exists to build a deterministic, auditable, institutional market-scanning and decision-support platform.

The project does not exist to create hidden discretionary trading shortcuts. It exists to preserve opportunity evidence, classify source and market conditions, and allow a single downstream Decision Engine to produce final decision semantics under explicit contracts.

## RESET-1 Direction

RESET-0 decided to proceed with a controlled clean rebuild. The old repository is treated as a knowledge base, not as a failed implementation. RESET-1 converts certified knowledge into a concise canonical documentation set for v2 planning.

## Product Boundary

The platform may:

- discover market opportunities;
- preserve opportunity rows and evidence;
- classify validation, context, fundamentals, timing, and portfolio state;
- maintain explicit data contracts;
- emit final decisions only through the Decision Engine;
- communicate decisions through reporting without altering them;
- support historical learning and diagnostics when governed as research.

The platform must not:

- create allocation semantics outside the Decision Engine;
- hide filtering in upstream layers;
- treat generated artifacts as source-of-truth inputs without approval;
- promote source-data review artifacts into pipeline inputs before contracts exist;
- reuse old Python files as the v2 implementation base.

## Certified Doctrine

- Classification upstream.
- Allocation downstream.
- Decision Engine is the only allocation, execution, arbitration, and final-action authority.
- Reporting communicates only.
- No hidden filtering.
- No upstream tradeability.
- Deterministic behavior.
- Row preservation where contractually required.
- Auditability and source traceability.
- Explicit contracts before implementation.
- English-only repository content.

## Development Posture

New feature work on the old active architecture is paused. Old code remains available for legacy operation and reference. v2 implementation must start only after canonical documentation, repository structure, data contracts, and test strategy are approved.

## Source-of-Truth Order

1. `AGENTS.md` for repository-level AI and governance boundaries until it is explicitly updated.
2. RESET-1 canonical active documents in `docs/active/`.
3. `docs/resets/reset_0_full_repository_knowledge_extraction_and_rebuild_decision.md` for reset rationale.
4. Legacy active, sprint, audit, and archive documents as source material only.

## Success Criteria

The project succeeds when v2 can run from clean contracts and newly written implementation while preserving the certified doctrine learned from the old repository.
