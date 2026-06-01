# PM Operating Model

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines how planning, scope control, backlog handling, and reset-stage governance work during the v2 rebuild.

## Operating Principle

The rebuild is staged. A stage authorizes only its explicitly allowed changes. Documentation stages do not authorize implementation. Implementation stages do not authorize architecture changes beyond their approved contracts.

## Current Stage

RESET-1 authorizes canonical documentation rewrite only.

Allowed:

- create or update canonical documentation;
- define source-of-truth order;
- define future reset roadmap;
- capture backlog impact as draft items;
- preserve old documents as reference material.

Forbidden:

- code changes;
- test changes;
- CSV/data changes;
- generated outputs;
- file moves;
- archive/delete actions;
- workflow changes;
- pipeline runs;
- SEC diagnostics or live SEC calls;
- runtime behavior changes.

## Reset Stage Lifecycle

Each reset stage should state:

- purpose;
- allowed changes;
- forbidden changes;
- expected outputs;
- validation method;
- backlog impact;
- next action.

## Backlog Handling

The backlog is a planning tool. It does not authorize implementation.

RESET-1 captures the controlled rebuild as the active planning direction. The detailed backlog may be updated in a later dedicated documentation step if needed. Until then, draft backlog items in reset documents are planning candidates only.

## Scope Control

A reset stage must not mix unrelated work. For example, canonical documentation rewrite must not include code scaffolding, tests, fixtures, workflows, file moves, or generated artifact cleanup.

## Development Pause

New feature work on the old active architecture remains paused. Legacy maintenance may continue only if required to preserve the current run path or prevent data loss.

## Validation Policy

Documentation-only GitHub connector work should report that local commands such as `git diff --check` and `git status` cannot be executed unless a local working tree is available. Tests are not required when no code or tests changed.
