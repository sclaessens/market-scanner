# ME-DATA05 Incremental Market Data Refresh

Run ID: me-data05-incremental-refresh-20260713T140000Z
Status: incremental_refresh_operational
Cutoff date: 2026-07-10
Overlap calendar days: 7

## Refresh Summary

| Metric | Value |
|---|---:|
| Histories checked | 952 |
| Already current | 946 |
| Incrementally updated | 0 |
| New snapshots | 0 |
| Full rebuilds | 0 |
| Failed or blocked updates | 2 |
| Insufficient history | 4 |
| Rows downloaded | 12 |
| Rows added | 0 |
| Rows replaced within overlap | 0 |
| Files rewritten | 0 |
| Files unchanged | 952 |

## Coverage

| Metric | Before | After |
|---|---:|---:|
| Valid histories | 946 | 946 |
| Insufficient histories | 6 | 6 |
| Missing histories | 0 | 0 |
| Invalid histories | 0 | 0 |
| Unsupported mappings | 0 | 0 |

## Evaluation

| Metric | Before | After |
|---|---:|---:|
| Selected outcomes | 12 | 12 |
| Resolved | 0 | 0 |
| Still unresolved | 12 | 12 |
| Newly resolved |  | 0 |

Block reasons after:

```json
{
  "insufficient_forward_data": 12
}
```

## Recommended Next Sprint

ME-ANALYSIS01 - Broad canonical-universe analysis execution and reporting over the now-operational local market dataset.
