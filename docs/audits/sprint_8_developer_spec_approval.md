# Sprint 8 Developer Specification Approval

Sprint 8: Reporting Layer  
Approval date: 2026-05-11  
Approval authority: Sprint 8 Technical Lead Approval Authority  
Approval decision: DEVELOPER SPECIFICATION APPROVED

## Executive Summary

The Sprint 8 Developer Specification was reviewed against the certified architecture, approved execution plan, execution review, Reporting Layer governance doctrine, Decision Engine authority boundary, row-preservation rules, traceability requirements, auditability requirements, observability requirements, Telegram governance separation, legacy remediation requirements, test strategy, CI enforcement requirements, and repository language governance.

The Developer Specification is approved for implementation.

## Approval Scope

Reviewed artifact:

- `docs/sprints/sprint_8_developer_spec.md`

Supporting artifacts:

- `docs/sprints/sprint_8_reporting_preparation.md`
- `docs/audits/sprint_8_governance_audit.md`
- `docs/sprints/sprint_8_execution_plan.md`
- `docs/audits/sprint_8_execution_review.md`
- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/project_backlog.md`

This approval did not modify runtime code, tests, generated CSV/data files, or Telegram outputs.

## Governance Compliance Approval

Finding: Approved.

The Developer Specification preserves:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority
- upstream layers classify only
- reporting communicates only
- no hidden filtering
- no hidden allocation semantics outside Decision Engine
- no decision semantics outside Decision Engine
- no ranking authority outside Decision Engine
- no scoring authority outside Decision Engine

The specification correctly authorizes Reporting implementation as communication, representation, observability, traceability, and audit metadata only.

## Runtime Contract Approval

Finding: Approved.

The specification defines `scripts/reporting/build_reporting_layer.py` as the authoritative Reporting builder.

It defines the mandatory source:

- `data/processed/final_decisions.csv`

It defines the mandatory outputs:

- `data/processed/reporting_dashboard_data.csv`
- `data/logs/reporting_layer_log.csv`
- `reports/daily/telegram_message.txt`

It defines exact schemas for reporting dashboard data and reporting log output.

## Determinism Approval

Finding: Approved.

The specification requires stable source order, zero-based source row index, fixed section order, lexicographic source action grouping, source-order row display inside groups, deterministic missing-value rendering, no randomized sampling, and no score/rank/urgency/actionability ordering.

The determinism contract is sufficient for implementation and testing.

## Traceability Approval

Finding: Approved.

The specification defines source row identity as:

```text
<source_artifact_path>#<source_row_index>#<ticker>#<date>#<input_row_hash>
```

It requires source artifact path, source row identity, source row index, pass-through source fields, dashboard artifact path, and log metadata for every generated report. This is sufficient to prove Reporting did not create an alternate decision universe.

## Auditability Approval

Finding: Approved.

The specification requires machine-readable dashboard data and log metadata for every Telegram message. It requires row counts, preservation status, grouping rule, truncation rule, deterministic ordering rule, forbidden semantics status, English-only status, and upstream mutation status.

The auditability contract is implementation-ready.

## Observability Approval

Finding: Approved.

The specification defines observability for source availability, schema status, optional stability availability, artifact write status, source row count, dashboard row count, displayed row count, summarized row count, omitted row count, row preservation, ticker/date universe preservation, source order preservation, forbidden semantics status, English-only status, and source mutation status.

The observability scope is sufficient.

## Telegram Governance Approval

Finding: Approved.

The specification correctly separates:

- outbound Telegram Reporting as communication-only, source-traceable, deterministic, and audit-safe
- inbound Telegram command handling as operational command infrastructure only

The specification does not authorize inbound Telegram command expansion and does not treat inbound command handling as Reporting authority or allocation authority.

## Legacy Remediation Approval

Finding: Approved.

The specification explicitly treats legacy `market_scan_*.md` reports as historical artifacts, prohibits regenerating them as active Sprint 8 output, removes active ranked/scored setup language, removes omission-based Telegram summaries, and requires English-only normalization for touched Reporting and Telegram content.

This satisfies the required legacy remediation path from the governance audit and execution plan.

## Test Strategy Approval

Finding: Approved.

The specification requires tests for schema validation, row-count preservation, source-universe preservation, deterministic output, deterministic ordering, deterministic grouping, deterministic truncation, no hidden omission, no hidden prioritization, no hidden ranking, no hidden scoring, no hidden execution semantics, no hidden urgency semantics, forbidden keyword scanning, English-only output, traceability, source-row identity preservation, fail-fast validation, optional Stability Layer handling, duplicate row identity handling, Telegram representation metadata, outbound/inbound Telegram separation, and source mutation detection.

The test scope is sufficient for implementation authorization.

## CI Enforcement Approval

Finding: Approved.

The specification defines CI or mandatory local validation for:

- forbidden reporting semantics
- forbidden keywords
- English-only validation
- deterministic output validation
- reporting schema validation
- reporting traceability validation
- reporting row-preservation validation
- no source mutation validation

The enforcement strategy is sufficient for Sprint 8 implementation.

## Governance Risks

Residual implementation risks:

- implementation must not recreate low-information omission under a new label
- implementation must not let Telegram compact formatting imply priority or urgency
- implementation must not allow legacy `build_telegram_summary.py` logic to remain an independent reporting path
- implementation must not expand inbound Telegram command processing
- forbidden keyword scans must distinguish runtime output from negative assertions in tests

These risks are controlled by the Developer Specification and must be verified during implementation audit.

## Required Corrections

No required corrections identified.

## Technical Lead Approval Decision

DEVELOPER SPECIFICATION APPROVED

Sprint 8 is authorized to proceed to implementation under `docs/sprints/sprint_8_developer_spec.md`.

## Backlog Impact Assessment

Backlog impact assessment: No new backlog items identified.

BL-0006 already captures the authorized Reporting and Telegram semantic drift remediation. No additional backlog item was identified during Developer Specification approval.

## Final Approval Conclusion

The Sprint 8 Developer Specification is governance-complete, implementation-authoritative, and approved.

Implementation may proceed only within the specified runtime boundaries, output contracts, deterministic rules, traceability requirements, auditability controls, observability controls, Telegram separation rules, legacy remediation requirements, English-only enforcement requirements, and test requirements.
