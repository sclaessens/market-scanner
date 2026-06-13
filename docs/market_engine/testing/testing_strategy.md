# Testing Strategy

Owner role: QA / Test Lead

Status: MARKET ENGINE BASELINE

## Purpose

This strategy preserves testing lessons already learned in the repository and defines baseline expectations for future Market Engine implementation sprints.

ME01 does not authorize new tests or test changes.

## Test Placement

- Do not create new test files when an existing test family is suitable.
- Extend existing test families where possible.
- Create new test files only for new ownership boundaries.
- Keep test naming aligned with behavior and ownership, not sprint ceremony.

## Provider Safety

- No live provider calls are allowed in normal automated tests.
- Use fake or synthetic provider responses.
- Future live-provider smoke checks must be explicit manual smoke harnesses, not normal automated tests.
- Manual smoke harnesses must be bounded, documented, and non-canonical unless promoted by architecture.

## Required Future Proofs

Future Market Engine tests must prove:

- missing data remains missing;
- ticker failures do not stop the whole batch;
- source intake does not become recommendation logic;
- BUY / SELL / HOLD / recommendation language does not leak into source, data, scanner, or fundamental layers;
- Decision Engine side effects do not occur outside authorized decision paths;
- Telegram and reporting side effects do not occur from lower layers;
- portfolio and watchlist mutation does not occur from lower layers.

## Side-Effect Controls

Tests must guard against:

- hidden provider calls;
- production writes;
- import-time side effects;
- report generation;
- Telegram delivery;
- portfolio mutation;
- watchlist mutation;
- Decision Engine authority expansion.

