# Repository Structure

Status: ACTIVE
Reset stage: RESET-1

## Purpose

This document defines the intended v2 repository structure. RESET-1 does not move, delete, archive, or refactor files. This document is the planning baseline for RESET-2 and later.

## Current Rule

Until RESET-2 approves a structure and archive plan, existing files remain in place. Old files are reference material unless explicitly carried forward into the new canonical set.

## Proposed V2 Structure

```text
docs/
  active/
    project_charter.md
    product_vision.md
    roles_and_responsibilities.md
    pm_operating_model.md
    functional_analysis.md
    technical_architecture.md
    data_architecture.md
    source_data_strategy.md
    pipeline_contract.md
    decision_engine_contract.md
    reporting_contract.md
    testing_strategy.md
    repository_structure.md
    backlog.md
    roadmap.md
  legacy/
    sprints/
    audits/
    superseded/
    source_data/
  resets/
    reset_0_full_repository_knowledge_extraction_and_rebuild_decision.md
    reset_1_canonical_documentation_rewrite_closeout.md
src/
  market_scanner/
    discovery/
    validation/
    context/
    fundamentals/
    timing/
    portfolio/
    decisions/
    reporting/
    orchestration/
    shared/
tests/
  contract/
  unit/
  integration/
  fixtures/
data/
  input/
    portfolio/
    universe/
  fixtures/
  generated/
    processed/
    logs/
reports/
  generated/
```

## Tracking Policy

Track:

- canonical documentation;
- newly written v2 source code after approval;
- approved fixtures;
- minimal configuration needed for reproducible development and CI.

Ignore by default:

- generated CSV outputs;
- logs;
- reports;
- provider caches;
- SEC cache files;
- local diagnostics;
- temporary run outputs.

## Legacy Policy

Legacy files must not be deleted during RESET-1. RESET-2 should classify old active, sprint, audit, archive, code, test, data, and workflow paths before any movement happens.

## V2 Code Placement

New v2 Python code should live under `src/market_scanner/` after RESET-3. Old `scripts/` files remain reference-only and legacy-run surfaces until a governed cutover.

## V2 Test Placement

New v2 tests should live under contract, unit, integration, and fixture-focused groups. Old tests are reference material and should not force old implementation paths into v2.
