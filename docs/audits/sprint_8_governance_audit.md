# Sprint 8 Governance Audit

Sprint 8: Reporting Layer  
Audit date: 2026-05-10  
Audit authority: Sprint 8 Governance Audit Authority  
Certification decision: CERTIFIED WITH REQUIRED CORRECTIONS

## Executive Summary

Sprint 8 Reporting Layer preparation was audited against the certified market-scanner architecture, reporting-only governance doctrine, Decision Engine authority boundary, distribution-preservation requirements, and repository language governance.

The preparation correctly identifies that Reporting must remain a communication and observability layer only. It explicitly prohibits hidden filtering, hidden prioritization, hidden ranking, hidden allocation, execution signalling, tradeability classification, urgency classification, and Decision Engine output overrides.

The preparation also correctly identifies material legacy semantic drift in current Reporting and Telegram artifacts, including row omission, compact grouping, active decision grouping, legacy ranked/scored reports, entry/stop/target language, actionable headings, mixed-language operator text, and Telegram command-processing risks.

The preparation is certified for the next lifecycle phase only with required corrections. These corrections must be addressed in Sprint 8 execution planning and developer specification before implementation may be authorized.

## Audit Scope

Reviewed artifacts:

- `AGENTS.md`
- `README.md`
- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/execution_roadmap_v2.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/sprint_8_reporting_layer.md`
- `docs/sprints/sprint_8_reporting_preparation.md`
- `docs/sprints/sprint_7_closeout.md`
- `docs/technical/decision_engine_design_v2.md`
- `docs/functional/Functional_Analysis_v2.md`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `scripts/reporting/reporter.py`
- `scripts/telegram/process_telegram_commands.py`
- `tests/reporting/test_build_telegram_summary.py`
- `reports/daily/telegram_message.txt`
- legacy daily reporting outputs under `reports/daily/`

This audit is governance-only. No runtime code, tests, generated CSV/data files, reports, or implementation artifacts were modified.

## Certified Governance Context

The certified architecture is:

```text
scanner
-> validation_layer
-> context_layer
-> fundamental_layer
-> timing_state_layer
-> portfolio_intelligence_layer
-> watchlist
-> portfolio
-> decision_engine
-> reporting
```

Certified doctrine:

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

Repository language governance:

- repository content must be English-only
- Dutch is allowed only in direct chat with the user
- mixed-language repository artifacts are forbidden

## Reporting Boundary Audit

Finding: Pass with required execution-planning corrections.

The preparation correctly defines Reporting as communication-only and states that Reporting may summarize, visualize, organize, explain, format, and aggregate already-approved outputs.

The preparation correctly forbids Reporting from:

- creating allocation decisions
- modifying allocation decisions
- suppressing opportunities
- introducing hidden filtering
- reprioritizing opportunities
- creating tradeability logic
- introducing urgency semantics
- overriding Decision Engine outputs
- mutating upstream artifacts
- creating ranking authority
- creating scoring authority
- deriving execution semantics from upstream metadata combinations

Required correction:

- Sprint 8 execution planning must convert these boundaries into enforceable implementation contracts, tests, and forbidden-field scans before developer execution is authorized.

## Decision Engine Authority Audit

Finding: Pass with required execution-planning corrections.

The preparation preserves Decision Engine authority. It requires Reporting to display Decision Engine fields exactly as produced and forbids recalculation of tradeability, conviction, allocation priority, final actions, execution semantics, or urgency.

Required correction:

- Future reporting outputs must include source provenance and row identity for every displayed or summarized Decision Engine row so the communication layer cannot become a parallel decision authority.

## Reporting Semantics Audit

Finding: Pass with required corrections.

The preparation correctly identifies semantic categories that Reporting must avoid:

- hidden prioritization
- hidden ranking
- hidden execution semantics
- hidden suppression
- implicit recommendation language
- action reinterpretation
- conviction recalculation
- allocation priority recalculation

Special risk assessment:

- Aggregation logic can become hidden prioritization if sections imply importance.
- Grouping logic can become hidden filtering if groups determine display visibility.
- Omission logic can become hidden suppression if omitted rows lack audit metadata.
- Compact reporting can become distribution collapse if full source visibility is not traceable.
- Telegram formatting can imply urgency through section order, active-decision headings, or action labels.
- Legacy ranked/scored reports can create hidden allocation authority by presenting scores, grades, and ranked setup lists.

Required correction:

- Execution planning must define neutral, source-provenanced section naming and deterministic display rules that do not imply urgency, priority, or execution readiness.

## Legacy Semantic Drift Audit

Finding: Pass with required corrections.

The preparation correctly identifies legacy drift in current artifacts:

- `reports/daily/market_scan_*.md` includes actionable headings.
- Legacy reports include entry, stop, target, risk/reward, score, grade, and ranked setup sections.
- `reports/daily/telegram_message.txt` reports low-information scanner observations as omitted.
- `scripts/reporting/build_telegram_summary.py` groups active decisions and observation candidates.
- `scripts/reporting/build_telegram_summary.py` uses low-information logic based on source, setup, tradeability, validation, timing, portfolio, and reason combinations.
- `scripts/reporting/build_telegram_summary.py` limits examples in observation summaries.
- `tests/reporting/test_build_telegram_summary.py` currently validates omission of low-information scanner observations.
- `scripts/reporting/send_telegram.py` contains non-English operator-facing strings and comments.
- `scripts/telegram/process_telegram_commands.py` contains BUY/SELL command handling, non-English comments, and non-ASCII operator messages.

Required correction:

- Sprint 8 execution planning must explicitly decide whether legacy daily reports are archived, replaced, or converted to Decision Engine-traceable communication artifacts.
- Sprint 8 developer specification must remove or quarantine reporting behaviours that encode hidden omission, ranking, scoring, or execution-like presentation.

## Distribution Preservation Audit

Finding: Pass with required corrections.

The preparation correctly states:

- Reporting must not silently suppress rows, opportunities, or Decision Engine outputs for readability.
- Aggregation, grouping, truncation, prioritization, or summarization rules must be deterministic, documented, and auditable.
- If a communication surface cannot display every row, it must state source row count, displayed row count, summarized row count, omitted row count, deterministic display rule, and full source artifact path.

Required correction:

- Execution planning must define a concrete row-preservation reporting contract and tests proving no silent row loss.
- Any "omitted" concept must be replaced by auditable representation metadata or explicitly governed display-limit metadata.

## Reporting Determinism Audit

Finding: Pass with required corrections.

The preparation defines deterministic requirements for:

- stable input ordering
- stable grouping order
- stable section order
- stable tie handling
- stable text templates
- deterministic missing-value rendering
- deterministic output schema
- deterministic row-count logging

Required correction:

- Execution planning must define exact deterministic ordering for all report sections and rows, with tests preventing accidental priority-like ordering.

## Reporting Auditability Audit

Finding: Pass with required corrections.

The preparation requires source traceability, source row identity, explicit grouping rules, display-limit documentation, and clear distinction between displayed and summarized rows.

Required correction:

- Future reporting artifacts must include machine-readable audit metadata, not only human-readable summary text.
- Telegram output must be backed by an auditable reporting log or dashboard data artifact proving source coverage.

## Reporting Observability Audit

Finding: Pass with required corrections.

The preparation requires logging for:

- input artifact paths
- schema status
- input row counts
- output row counts
- row preservation status
- grouping rules
- truncation rules
- omitted or summarized row counts
- output artifact paths
- generation timestamp
- contract version
- forbidden field scan status
- source traceability status

Required correction:

- Execution planning must specify the exact reporting log schema and distinguish operator progress messages from report content.

## Data Contract Audit

Finding: Pass with required corrections.

The preparation defines candidate inputs and a minimum output contract. It correctly avoids recalculated tradeability, recalculated conviction, hidden filter flags, new ranking fields, new scoring fields, execution urgency fields, and recommendation fields.

Required correction:

- The future execution plan must reconcile the Sprint 8 roadmap's legacy conceptual `decision_output.csv` references with the current authoritative `data/processed/final_decisions.csv`.
- The future execution plan must avoid output fields such as `conviction_score` unless they are direct pass-through Decision Engine fields and explicitly source-provenanced.

## Telegram Governance Audit

Finding: Pass with required corrections.

The preparation correctly isolates Telegram as a communication channel, not a decision authority.

Current Telegram risks:

- active decision section ordering can imply priority
- observation grouping can imply lower importance
- low-information omission can become hidden suppression
- compact example limits can hide source rows
- BUY/SELL command processing is governance-adjacent and execution-like
- non-English Telegram messages violate repository language governance

Required correction:

- Sprint 8 execution planning must define separate governance treatment for outbound reporting and inbound Telegram command processing.
- Outbound Telegram reporting must include deterministic traceability metadata or a companion audit artifact.
- Inbound Telegram command handling must not be treated as Reporting Layer implementation unless explicitly governed as a separate operational command surface.

## Repository Language Governance Audit

Finding: Pass with required corrections.

The preparation document itself is English-only. It correctly identifies mixed-language repository drift in legacy reporting and Telegram files.

Known language governance drift:

- `scripts/reporting/build_telegram_summary.py` contains non-English text.
- `scripts/reporting/send_telegram.py` contains non-English comments and operator-facing messages.
- `scripts/telegram/process_telegram_commands.py` contains non-English comments, non-ASCII symbols, and mixed-language command responses.
- Legacy sprint and roadmap documentation contains mixed-language text already captured by `BL-0005`.

Required correction:

- Sprint 8 execution planning must include English-only remediation for Reporting and Telegram code paths in scope.
- New Sprint 8 artifacts must remain English-only.

## Governance Risks

Critical governance risks:

- Reporting could become hidden filtering through omitted rows.
- Reporting could become hidden prioritization through active section ordering.
- Reporting could become hidden ranking through legacy ranked reports.
- Reporting could become hidden scoring authority through grades and scores.
- Reporting could become execution signalling through entry, stop, target, actionable, and watch-for-retest language.
- Telegram command processing could blur reporting with operational trade command handling.
- Compact Telegram formatting could collapse distribution visibility.
- Mixed-language reporting artifacts could violate permanent repository language governance.

## Required Corrections

Required before Sprint 8 may proceed to implementation authorization:

1. Define an execution plan that replaces or quarantines legacy reporting semantics that imply actionability, ranking, scoring, execution, or prioritization.
2. Define exact Reporting input contracts using `data/processed/final_decisions.csv` as the authoritative Decision Engine output.
3. Define exact Reporting output contracts and log schemas with row-count, row-identity, grouping, summarization, and traceability metadata.
4. Define deterministic display, grouping, truncation, and section-order rules that cannot imply allocation priority.
5. Define tests for no hidden filtering, no hidden prioritization, no hidden ranking, no hidden scoring, no hidden execution semantics, no source mutation, and English-only reporting output.
6. Define how Telegram outbound reporting is separated from inbound Telegram command processing.
7. Define remediation requirements for mixed-language reporting and Telegram artifacts.
8. Define treatment for legacy daily reports: archive, replace, or convert to Decision Engine-traceable communication artifacts.

## Non-Blocking Recommendations

- Prefer a single `build_reporting_layer.py` as the authoritative reporting artifact builder.
- Use companion audit logs for Telegram messages so compact human text does not carry the full audit burden.
- Treat display truncation as a first-class auditable event.
- Keep report wording neutral and descriptive.
- Prefer section labels based on source fields, not operator action implications.
- Preserve full source artifact references in every report output.

## Certification Decision

CERTIFIED WITH REQUIRED CORRECTIONS

The preparation is governance-aligned and sufficient to proceed to Sprint 8 execution planning only if the required corrections are carried into the execution plan and developer specification.

Implementation is not authorized by this audit.

## Backlog Impact Assessment

Backlog impact assessment: No new backlog items identified.

Rationale: `BL-0006` already captures the reporting and Telegram semantic drift identified by this audit, and `BL-0005` already captures legacy mixed-language documentation normalization.

## Final Audit Conclusion

Sprint 8 preparation correctly frames Reporting as a communication-only layer and accurately identifies legacy semantic drift that could otherwise compromise the certified architecture.

The preparation preserves the Decision Engine authority boundary, establishes distribution-preservation expectations, and defines auditability, observability, determinism, traceability, and language-governance requirements.

The next required lifecycle action is Sprint 8 execution planning with the required corrections from this audit. No implementation work may begin until execution planning and developer specification explicitly certify the reporting contracts and remediation scope.
