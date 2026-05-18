# Archive Strategy

Status: ACTIVE

This document defines how historical and superseded materials are preserved without remaining operationally authoritative.

## Objective

The repository must preserve institutional history while reducing active documentation clutter.

Archiving means removing a document from the operational surface. It does not mean deleting institutional reasoning.

## Archive Categories

### Sprint Artifacts

Historical sprint plans, developer specs, execution plans, closeouts, and implementation reviews belong in archive once the sprint is certified complete.

### Audits

Historical audits remain important evidence. They should be preserved but not treated as active development instructions unless referenced by an active document.

### Migration Documents

Architecture migration and governance purification documents are historical after Sprints 0 through 8.

### Superseded Documents

Documents whose doctrine has been consolidated into `docs/active/` should be archived or marked as superseded.

## Recommended Archive Structure

```text
docs/archive/
  audits/
  sprints/
  migration/
  superseded/
```

## Current Transition Rule

Existing legacy locations may remain physically in place during the first restructuring PR to avoid large path churn. Their authority is reduced by documentation hierarchy:

- active documentation defines current truth
- reference documentation explains rationale
- archive or legacy sprint/audit documents preserve history

A later mechanical cleanup PR may move files physically after the active structure has been accepted.

## Do Not Delete

Do not delete:

- sprint closeouts
- final governance audits
- implementation audits
- historical roadmap artifacts
- certified migration rationale
- architecture transition evidence

These files preserve institutional traceability.

## Archive Header Recommendation

When legacy files are physically moved or marked in place, add this header:

```markdown
> Archive status: HISTORICAL
> This document is preserved for traceability. It is not an active operational source of truth. Current operational doctrine is maintained in `docs/active/`.
```

## Authority Rule

If archived or historical documents conflict with active documents, the active documents win.

## Cleanup Strategy

Use staged cleanup:

1. Add active documentation structure.
2. Update README to point to active documentation.
3. Classify legacy files.
4. Move or mark historical files in a separate cleanup PR if path churn is large.
5. Keep all certified evidence preserved.