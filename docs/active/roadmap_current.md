# Current Operational Roadmap

Status: ACTIVE

This document replaces migration-era roadmap material for forward planning.

## Phase

The project is in the operational intelligence platform evolution phase.

Sprints 0 through 8 completed the architecture purification phase and certified the core pipeline architecture.

## Strategic Direction

The next roadmap should focus on operating, observing, hardening, and learning from the certified architecture.

## Priority Areas

### 1. Runtime Reliability

Improve failure handling, input validation, repeatability, and deterministic recovery.

### 2. Orchestration

Strengthen end-to-end pipeline execution, scheduling, dependency handling, and local versus automated runs.

### 3. GitHub Actions Recovery

Stabilize workflow automation and make failures easier to diagnose and repair.

### 4. Operational Visibility

Improve logs, run summaries, diagnostics, and operator-facing status reporting.

### 5. Historical Decision Storage

Persist decision outputs for later analysis without changing Decision Engine authority.

### 6. Prediction Tracking

Track outcomes of decisions and classifications as observational feedback. Prediction tracking is research and diagnostics unless a later governed change explicitly routes an approved signal into the Decision Engine.

Prediction tracking must not introduce hidden allocation logic upstream.

### 7. Feedback Loops

Introduce research and diagnostic loops that help evaluate system behavior while preserving separation between observation and allocation.

Feedback loops and historical performance analysis are observational by default. They may inform future proposals, tests, diagnostics, or Decision Engine design reviews, but they must not create upstream tradeability, hidden filtering, or allocation authority outside the Decision Engine.

### 8. Portfolio Intelligence Evolution

Expand descriptive portfolio intelligence without moving allocation authority outside the Decision Engine.

### 9. Reporting Redesign

Improve communication clarity and usability while preserving reporting neutrality.

### 10. Production Readiness

Improve operational runbooks, environment setup, validation, monitoring, and release discipline.

## Governance Rule

Roadmap items do not authorize implementation. Each item must still pass Governance v2 level classification before development begins.
