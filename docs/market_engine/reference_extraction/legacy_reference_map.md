# Legacy Reference Map

Owner role: Governance Auditor / Scrum Master

Status: ACTIVE MARKET ENGINE REFERENCE MAP

## Purpose

This map records the documentation-root transition after ME04-PREP.

## Path Mapping

```text
docs/active/
-> docs/archive/market_scanner_reference/active/

docs/archive/ historical contents
-> docs/archive/market_scanner_reference/archive/

docs/audits/
-> docs/archive/market_scanner_reference/audits/

docs/legacy/
-> docs/archive/market_scanner_reference/legacy/

docs/resets/
-> docs/archive/market_scanner_reference/resets/

docs/project_roles_and_responsibilities.md
-> docs/archive/market_scanner_reference/project_roles_and_responsibilities.md
```

`docs/templates/` remains in place after ME04-PREP-C. It contains reusable documentation templates and is treated as a manual decision / shared documentation infrastructure area, not a clear legacy archive candidate.

## Authority Rule

From ME04-PREP onward, `docs/market_engine/` is the active authority for Market Engine documentation.

Archived documents under `docs/archive/market_scanner_reference/` are reference only. They may inform Market Engine work only through explicit extraction.

## Citation Rule

Future Market Engine documentation should cite the archived path when referring to old v2, BL, or reset documents that formerly lived under `docs/active/`.

Existing Market Engine documents may still mention `docs/active/...` because those were historical paths at extraction time. Those references should be read as historical extraction records unless a later sprint explicitly updates them.

New Market Engine specifications must not treat the archived path as active implementation authority.
