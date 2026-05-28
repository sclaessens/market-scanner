# Sprint E2 Closeout — Fundamental Metrics Builder

Status: CLOSED
Backlog driver: BL-0015
Date: 2026-05-27

## 1. Purpose

This document closes Sprint E2 after the implementation of the Fundamental Metrics builder.

Sprint E2 followed Sprint E1 and added deterministic financial metric calculation from validated raw fundamentals history data.

## 2. Implemented Scope

Sprint E2 added or updated:

- `scripts/core/build_fundamental_metrics.py`
- `tests/core/test_build_fundamental_metrics.py`
- `docs/active/logic/calculation_registry.md`

Implemented metrics:

- `gross_margin`
- `operating_margin`
- `net_margin`
- `free_cash_flow_margin`
- `debt_to_equity`
- `return_on_equity`
- `revenue_yoy_growth`
- `eps_yoy_growth`
- `free_cash_flow_yoy_growth`

The builder validates raw history input through the Sprint E1 raw-history validator before computing metrics.

## 3. Output Contract Review

Sprint E2 output is:

- row-preserving;
- input-order preserving;
- deterministic;
- written only when an explicit output path is supplied;
- based on temporary test fixtures in tests;
- not dependent on real local raw data.

Output includes identity/source columns, metric columns, and descriptive helper columns:

- `metric_status`
- `metric_missing_inputs`
- `metric_warnings`

These helper columns are descriptive only. They do not create quality states, ranking authority, tradeability, urgency, conviction, allocation semantics, or final decisions.

## 4. Non-Scope Confirmation

Sprint E2 did not implement:

- Fundamental Quality refactor;
- Fundamental Analysis layer;
- ticker-category runtime logic;
- category source artifact;
- source-data automation;
- provider/API integration;
- full pipeline orchestration;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- portfolio changes;
- generated data commits;
- Python runtime cleanup or file deletion.

## 5. Validation Summary

Sprint E2 validation reported:

- focused test command passed: 13 tests passed;
- full test suite passed: 280 tests passed;
- `git diff --check` passed;
- governance grep checks were run and only reported pre-existing references outside Sprint E2 scope.

## 6. Backlog Review

BL-0015 remains active as the broader fundamentals implementation driver.

Sprint E2 completed only the deterministic metrics-builder substep. It does not complete BL-0015 as a whole because future work still includes:

- compatibility strategy for the existing Fundamental Layer surface;
- future quality mapping from raw-history and metrics evidence;
- future descriptive Fundamental Analysis layer;
- controlled integration decisions;
- eventual pipeline orchestration decisions after compatibility is protected.

No backlog item should be marked implemented solely because of Sprint E2.

## 7. Calculation Registry Impact

The calculation registry was updated during Sprint E2.

Registry impact:

- Fundamental Metrics is now implemented/current as a deterministic calculation family.
- Current implementation: `scripts/core/build_fundamental_metrics.py`.
- Future artifact: `data/processed/fundamental_metrics.csv`.
- Semantics: deterministic calculations only.

No additional registry correction is required in this closeout.

## 8. Logic and Code Placement Review

Code placement is acceptable for the current scope.

Reason:

- the metrics builder is separate from `build_fundamental_layer.py`;
- it reuses the E1 validation helper instead of duplicating raw schema checks;
- it keeps raw validation separate from metrics;
- it keeps metrics separate from quality classification and business analysis;
- it is not wired into the pipeline;
- it does not touch downstream Decision Engine, Reporting, Telegram, Timing, or Portfolio Intelligence surfaces.

Future implementation may later decide whether fundamentals modules should move into a dedicated fundamentals package or subfolder as part of a broader cleanup/refactor sprint. That is not part of Sprint E2.

## 9. Documentation Review

The calculation registry was updated during implementation. No additional active contract rewrite is required for this closeout.

Current documentation remains aligned with:

- `docs/active/contracts/fundamental_calculations_technical_spec.md`;
- `docs/active/specs/fundamentals_history_implementation_spec.md`;
- `docs/active/contracts/fundamentals_platform_contract.md`;
- `docs/active/logic/calculation_registry.md`;
- `docs/active/logic/strategy_logic_rationalization.md`;
- `docs/active/logic/ticker_category_model.md`.

## 10. Recommended Next Sprint

Recommended next sprint:

```text
Sprint E3 — Fundamental Quality Compatibility Wrapper
```

Recommended scope:

- protect the existing `data/processed/fundamental_quality.csv` downstream contract;
- preserve `scripts/core/build_fundamental_layer.py` as the pipeline-facing compatibility surface;
- begin mapping validated raw-history and/or metrics evidence into compatible quality metadata only if explicitly scoped;
- keep downstream row shape and required columns stable;
- avoid Decision Engine, Reporting, Telegram, Timing, Portfolio Intelligence, and scanner changes;
- avoid ticker-category runtime logic;
- avoid provider/API integration;
- avoid generated data commits.

Sprint E3 must not become a broad Fundamental Analysis sprint.

## 11. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

## 12. Closeout Decision

Sprint E2 is closed.

The project may proceed to a separately scoped Sprint E3 developer prompt after Product Owner approval.