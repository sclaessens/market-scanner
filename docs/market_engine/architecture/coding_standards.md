# Coding Standards

Owner role: Development Lead / Technical Architect

Status: MARKET ENGINE BASELINE

## Purpose

These standards preserve coding lessons already learned in the repository and define baseline constraints for future Market Engine implementation sprints.

ME01 does not authorize implementation. These standards guide later approved work.

## Module Ownership

- Do not create a new Python file for every new step.
- Extend existing modules when ownership is clear.
- Create new Python files only when architecture explicitly justifies a new ownership boundary.
- Keep module names aligned with stable product responsibilities, not transient sprint tasks.
- Do not use temporary quick scripts as canonical runtime.
- Manual smoke harnesses must be explicit, bounded, and non-canonical unless promoted by architecture.

## Runtime Safety

- Do not introduce hidden provider calls.
- Do not perform production side effects at import time.
- Do not mutate portfolio data from lower layers.
- Do not mutate watchlist data from lower layers.
- Do not trigger Telegram delivery from source, scanner, data, or analysis layers.
- Do not generate reports from source, scanner, data, or analysis layers.

## Data Semantics

- Missing data must remain missing and must not be converted to zero.
- Provider and source access must stay separated from analysis and decision logic.
- Raw source evidence, normalized data, source-readiness state, and analysis output must remain distinguishable.
- Data intake must not imply recommendation, allocation, or operator action.

## Decision Boundary

- No BUY / SELL / HOLD / recommendation leakage is allowed in source, data, scanner, or fundamental layers.
- No allocation gating is allowed outside the Decision Engine.
- No tradeability, conviction, urgency, or execution logic is allowed outside the Decision Engine.
- Reporting remains communication only.

## Source Access Boundary

Provider access must be explicit, bounded, and reviewable. Future live-provider smoke checks must be separated from normal runtime and automated tests.

