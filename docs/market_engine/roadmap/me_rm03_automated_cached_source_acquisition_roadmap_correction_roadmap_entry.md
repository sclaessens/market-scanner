# ME-RM03 - Automated cached-source acquisition roadmap correction roadmap entry

COMPLETED BY ME-RM03.

## Active Next Direction

Active next direction after ME-SR13:

```text
ME-SA01 - Define automated cached-source acquisition job contract
```

ME-SR13A is no longer the primary next sprint. It remains available only as a fallback/manual diagnostic candidate.

## Target Architecture

The target architecture is:

```text
automated acquisition
-> cached-source snapshot
-> existing validation
-> cached_source_snapshot dry-run
-> operator preview
```

The application must own recurring source acquisition. Manually supplied source packages are not the primary operating model.

ME-RUN25 remains the proven import/staging/dry-run bridge. ME-SR13 remains blocked evidence that missing manual operator input is not a reliable primary path.

## Recommended Sequence

```text
ME-SA01 - Define automated cached-source acquisition job contract
ME-SA02 - Implement first bounded automated cached-source acquisition job for approved sample tickers/source families
ME-RUN26 - Run automated cached-source acquisition for NVDA, AMD, ASML through staging validation and local dry-run
ME-TP01 - Produce terminal-visible operator preview from real cached-source dry-run artifacts
```

ME-SA01 must preserve local-only validation boundaries, fail closed on missing or malformed source material, remain compatible with the existing cached-source snapshot manifest/import/staging flow, and introduce no downstream analysis or decision authority.
