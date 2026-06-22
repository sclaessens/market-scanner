# ME-UNI08 Roadmap Entry - Professional Swing Universe CLI Flag

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI08

## Roadmap Position

ME-UNI08 follows:

```text
ME-UNI06 - Implement editable universe loader and validation
ME-UNI07 - Wire editable universe into local Market Engine runtime input
```

ME-UNI08 turns the ME-UNI07 bridge into a first-class operator CLI choice.

## Completed Outcome

The cached-source batch dry-run command now supports:

```text
--professional-swing-universe
```

The flag uses the approved default Professional Swing Universe path and routes through existing loader/validation behavior.

## Preserved Behavior

Custom `--canonical-ticker-universe <path>` input remains supported.

`--professional-swing-universe` and `--canonical-ticker-universe <path>` are mutually exclusive and fail closed when combined.

## Next Roadmap Item

```text
ME-SR05 - Classify source support for Professional Swing Universe
```

ME-SR05 should classify actual source support for Professional Swing Universe rows. It must not treat operator source-policy hints as authoritative source support.
