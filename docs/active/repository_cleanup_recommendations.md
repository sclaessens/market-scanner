# Repository Cleanup Recommendations

Status: ACTIVE

This document records recommended cleanup actions for the post-purification repository.

## Completed in This Restructuring

- Added active documentation tier.
- Added current-state architecture source of truth.
- Added Governance v2 operational lifecycle.
- Added simplified sprint lifecycle.
- Added archive strategy.
- Added active/reference/archive classification.
- Added operational development model.
- Added current operational roadmap.

## Recommended Follow-Up Cleanup

### 1. Create Physical Archive Folders

Recommended structure:

```text
docs/archive/audits/
docs/archive/sprints/
docs/archive/migration/
docs/archive/superseded/
```

### 2. Move Historical Sprint Files

Move completed sprint artifacts from `docs/sprints/` into `docs/archive/sprints/` after confirming links and references.

Suggested candidates:

- completed sprint plans
- developer specs
- closeouts
- sprint-specific governance notes
- historical roadmap drafts

### 3. Move Historical Audit Files

Move completed sprint audits from `docs/audits/` into `docs/archive/audits/` after confirming references.

### 4. Preserve Certification Index

Keep a compact certification index active or reference-level so Sprints 0 through 8 remain easy to audit without keeping every historical file in the active path.

### 5. Consolidate Contracts

Create active contract files for each runtime layer if contract details remain scattered across sprint documents.

Recommended future folder:

```text
docs/active/contracts/
```

### 6. Add Operational Runbooks

Recommended future folder:

```text
docs/active/runbooks/
```

Initial runbooks:

- local development
- full pipeline run
- GitHub Actions recovery
- runtime failure triage
- daily operation checklist

### 7. Reduce README Doctrine Duplication

README should remain concise and link to active documentation instead of restating detailed governance doctrine.

### 8. Avoid New Sprint Document Sprawl

Future sprints should use the simplified sprint lifecycle unless Governance v2 escalation is required.

## Cleanup Boundary

This restructuring should not change runtime code, generated CSVs, Decision Engine logic, reporting semantics, or pipeline outputs.

## Principle

Prefer staged cleanup over large irreversible moves. Preserve institutional history while making the active repository easier to understand.