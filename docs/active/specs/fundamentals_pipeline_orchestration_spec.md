# Fundamentals Pipeline Orchestration Specification

Status: ACTIVE SPECIFICATION
Backlog driver: BL-0015
Related sequence: Sprint E1 through Sprint E5

## 1. Purpose

This document specifies how the new fundamentals builders may later be connected to the normal pipeline.

This is a documentation-only specification. It does not authorize code changes, pipeline wiring, generated artifact commits, provider/API usage, scraping, or runtime behavior changes.

The goal is to avoid connecting the new fundamentals surfaces too quickly or in the wrong order.

## 2. Current Fundamentals Surfaces

The project now has these standalone or compatibility-safe surfaces:

| Surface | File | Purpose | Current status |
|---|---|---|---|
| Raw history validation | `scripts/core/build_fundamentals_history_intake.py` | Validate raw fundamentals history schema and source evidence. | Implemented, standalone. |
| Fundamental metrics | `scripts/core/build_fundamental_metrics.py` | Compute deterministic metrics from validated raw history. | Implemented, standalone. |
| Fundamental quality compatibility | `scripts/core/build_fundamental_layer.py` | Preserve existing `fundamental_quality.csv` contract and optionally map raw/metrics evidence. | Implemented, pipeline-facing compatibility surface. |
| Fundamental analysis | `scripts/core/build_fundamental_analysis.py` | Produce descriptive fundamental analysis states from quality and optional metrics. | Implemented, standalone. |

## 3. Existing Pipeline Position

The current operational pipeline already has a Fundamental Layer position before Timing State.

Current protected sequence remains:

```text
scanner
-> validation
-> context
-> fundamental quality
-> timing state
-> portfolio intelligence
-> decision engine
-> reporting
```

The existing `fundamental_quality.csv` contract must remain protected until an explicit downstream migration is approved.

## 4. Proposed Future Fundamentals Subsequence

Future controlled sequence inside the Fundamental Layer area:

```text
raw fundamentals history validation
-> fundamental metrics
-> fundamental quality compatibility
-> fundamental analysis
```

This does not mean all four must immediately run inside the full pipeline.

The first orchestration implementation should be narrow and should preserve current downstream behavior.

## 5. Artifact Contract Direction

Potential future artifacts:

| Artifact | Status after E5 | Future orchestration role |
|---|---|---|
| `data/raw/fundamentals_history.csv` | Future source/input artifact. | Read only if explicitly present and validated. |
| `data/processed/fundamental_metrics.csv` | Future generated processed artifact. | Produced from raw history only when input exists. |
| `data/processed/fundamental_quality.csv` | Existing protected pipeline artifact. | Must remain downstream-compatible. |
| `data/processed/fundamental_analysis.csv` | Future generated processed artifact. | May remain standalone at first; downstream consumption requires later approval. |

Do not commit generated CSV outputs as part of orchestration implementation unless the repository explicitly approves generated artifact tracking.

## 6. Orchestration Principles

Future implementation must follow these principles:

1. Preserve `fundamental_quality.csv` as the downstream contract.
2. Do not break Timing State, Portfolio Intelligence, Decision Engine, Reporting, Telegram, or scanner behavior.
3. Do not make `fundamental_analysis.csv` a required downstream dependency until explicitly approved.
4. Do not treat missing raw fundamentals as a pipeline failure unless the file path is explicitly supplied and structurally invalid.
5. Do not call providers or scrape sources during orchestration.
6. Do not introduce ticker-category runtime logic.
7. Do not introduce decision authority outside the Decision Engine.
8. Keep generated artifacts deterministic.
9. Keep tests fixture-based where possible.
10. Keep orchestration changes separate from Python runtime cleanup.

## 7. Recommended First Wiring Step

Recommended first implementation after this specification:

```text
Sprint E7 — Controlled Fundamentals Pipeline Wiring
```

Recommended scope:

- add explicit optional orchestration support for raw history path;
- if raw history path exists, validate it;
- build metrics to an explicit generated path;
- build quality using optional raw/metrics paths;
- optionally build analysis to an explicit generated path;
- keep `fundamental_quality.csv` as the only downstream-required artifact;
- keep `fundamental_analysis.csv` non-blocking and non-consumed downstream;
- add tests for ordering, optional input handling, and no downstream contract breakage.

Do not combine this with runtime file reorganization.

## 8. Proposed Ordering for Full Pipeline Integration

A later implementation may use this order:

```text
build context
-> validate raw fundamentals history if configured
-> build fundamental metrics if raw history is configured and valid
-> build fundamental quality compatibility
-> build fundamental analysis if quality exists and metrics are available or optional
-> continue to timing state using the protected quality-compatible surface
```

Timing State should not be changed to require `fundamental_analysis.csv` in the first wiring sprint.

## 9. Failure Behavior

Recommended failure behavior:

| Situation | Future behavior |
|---|---|
| No raw-history path configured | Preserve current pipeline behavior. |
| Raw-history path configured but file missing | Fail clearly before generated outputs. |
| Raw-history path configured but invalid | Fail clearly using E1 validation. |
| Metrics cannot be built from valid raw history | Fail clearly in the fundamentals stage. |
| Metrics path absent but quality can still run | Preserve compatibility behavior. |
| Analysis cannot be built | Do not block existing downstream flow unless analysis was explicitly required. |

The first wiring sprint should avoid making analysis mandatory.

## 10. Test Requirements for Future Wiring

A future wiring sprint should test:

1. existing pipeline path still works without raw-history configuration;
2. raw-history validation is called when explicit raw-history input is configured;
3. invalid raw-history input fails before metrics output;
4. metrics builder runs after raw-history validation;
5. quality builder receives optional raw/metrics paths when configured;
6. quality output contract remains compatible;
7. analysis builder can run after quality/metrics without downstream consumption;
8. no rows are filtered by fundamentals enrichment;
9. generated output paths are explicit and deterministic;
10. no provider/API calls are introduced;
11. no ticker-category runtime input is required;
12. Decision Engine, Reporting, Telegram, Timing State, and Portfolio Intelligence behavior remain unchanged.

## 11. Documentation Updates Needed After Future Wiring

After a future wiring implementation, update only if needed:

- pipeline contracts if new generated artifact contracts become active;
- calculation registry if orchestration status changes calculation availability;
- backlog if new items are discovered;
- sprint closeout after merge.

Do not rewrite active doctrine unless the implementation reveals a real conflict.

## 12. Cleanup Boundary

Python file organization cleanup remains separate.

Potential future cleanup may reorganize fundamentals builders into a dedicated folder or package, but that belongs under BL-0023 or a dedicated runtime organization sprint.

Do not combine cleanup with orchestration wiring.

## 13. Recommended Next Decision

Choose one:

```text
A. Sprint E7 — Controlled Fundamentals Pipeline Wiring
B. R1 / BL-0023 — Python Runtime Organization Cleanup
```

Recommended default:

```text
A. Sprint E7 — Controlled Fundamentals Pipeline Wiring
```

Reason: the fundamentals builders are implemented and can now be connected carefully while still preserving downstream contracts.

## 14. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

BL-0015 remains active until the new fundamentals platform is operationally wired, validated, and closed out.

## 15. Validation

Documentation-only validation for this sprint should confirm:

- only documentation files changed;
- no code files changed;
- no tests changed;
- no CSV files changed;
- no raw data changed;
- no generated files changed;
- no workflow files changed;
- no provider APIs called;
- no scraping performed;
- no pipeline run;
- no tests run.