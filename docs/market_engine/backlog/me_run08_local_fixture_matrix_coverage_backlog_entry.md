# ME-RUN08 - Expand local fixture matrix coverage

Status: NEXT APPROVED SPRINT AFTER ME-RUN07

Goal: expand the local non-production dry-run fixture coverage from one realistic fixture into a deterministic fixture matrix for completed, limited, blocked, stale-data, missing-data, numeric-zero, unsupported-input, and provenance-heavy states.

Scope: local fixtures, local tests, command documentation, and audit documentation only.

Non-goals: no provider calls, live data, broker calls, message delivery, portfolio or watchlist writes, production artifacts, new financial logic, action semantics, allocation behavior, order generation, ranking, scoring, urgency, conviction, or tradeability authority.

Acceptance criteria: fixture matrix is non-production, reuses existing Market Engine payload contracts, uses `local_snapshot_fixture`, preserves existing input modes, keeps artifact writing opt-in, and synchronizes roadmap and backlog after completion.
