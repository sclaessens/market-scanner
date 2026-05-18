# Historical Evidence Contracts

Status: ACTIVE

## Purpose

Historical evidence artifacts preserve observational run, artifact, decision, and reporting evidence for audit and later research. They are separate from live processed artifacts and do not affect live pipeline behavior.

## Boundary

These artifacts are observational only.

They are not Decision Engine inputs. They do not affect live decisions, allocation logic, execution logic, upstream classification, reporting semantics, or pipeline sequencing.

## Generated Artifacts

| Artifact | Purpose | Authority boundary |
|---|---|---|
| `data/history/pipeline_runs.csv` | One row per historical evidence capture run with capture status, row counts, linkage status, and diagnostics | Observational run evidence only |
| `data/history/pipeline_artifacts.csv` | One row per expected source artifact with existence, row count, path, size, modified time, hash, and diagnostics | Observational artifact lineage only |
| `data/history/decision_reporting_observations.csv` | Decision Engine rows linked to reporting representation rows where possible, including unmatched diagnostics | Observational decision/reporting evidence only |

Generated live CSVs under `data/history/` should not be committed unless a future implementation explicitly approves a tiny source-controlled fixture or contract artifact.

## Schema Rules

Historical evidence schemas are additive and separate from existing `data/processed/` and `data/logs/` schemas.

The capture utility may read latest-state processed artifacts and logs. It must not mutate those source artifacts, write run identity back into them, or change existing schemas.

## Row Identity

Historical row identity is diagnostic and audit evidence only. It may record:

- ticker
- date
- source artifact path
- source row index
- source row identity when available
- input row hash when available
- computed observational row hash
- deterministic source ordering
- duplicate or missing identity diagnostics

Duplicate handling is diagnostic only. Historical capture must not resolve duplicates through filtering, ranking, suppression, or prioritization.

## Run Identity

Historical run identity is generated at capture time and confined to `data/history/*.csv`.

Run identity must not be consumed by the Decision Engine, live decisions, allocation logic, reporting decision semantics, upstream classification, or hidden filtering.

## Forbidden Semantics

Historical evidence must not introduce:

- hidden filtering
- ranking authority
- scoring authority
- prioritization authority
- conviction semantics
- urgency semantics
- upstream tradeability
- allocation gates
- reporting decision authority

Historical analysis may support audit, diagnostics, research, and future governed proposals only.
