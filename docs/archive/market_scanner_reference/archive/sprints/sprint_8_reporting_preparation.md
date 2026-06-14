# Sprint 8 Reporting Layer Preparation

Sprint 8: Reporting Layer  
Preparation date: 2026-05-10  
Status: PREPARATION COMPLETE  
Scope: Documentation and governance preparation only

## Executive Summary

Sprint 8 prepares the Reporting Layer as an institutional communication and observability layer.

Reporting must communicate already-approved outputs. It must not create decisions, modify decisions, suppress opportunities, introduce hidden filtering, reprioritize opportunities, create execution signals, create urgency semantics, or derive new allocation meaning from combinations of upstream metadata.

This preparation reviewed the current reporting architecture, Telegram summary generation, generated reports, current Decision Engine doctrine, Sprint 7 closeout, and active backlog governance. It identifies material legacy semantic drift in current reporting and Telegram artifacts. These findings are preparation inputs only. No implementation is authorized by this document.

## Strategic Context

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

The institutional doctrine is:

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

Sprint 7 added a Stability Layer that produces persistence metadata only. Sprint 8 reporting may consume this metadata for communication and observability, but may not reinterpret it into execution, allocation, ranking, or urgency semantics.

## Current Reporting Architecture

Current reporting artifacts reviewed:

- `scripts/reporting/reporter.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `scripts/telegram/process_telegram_commands.py`
- `reports/daily/telegram_message.txt`
- `reports/daily/market_scan_*.md`
- `tests/reporting/test_build_telegram_summary.py`

Current reporting behaviour is split across:

- legacy markdown market scan reports generated from setup-style scanner outputs
- Telegram summary generation from `data/processed/final_decisions.csv`
- Telegram delivery through `scripts/reporting/send_telegram.py`
- Telegram command processing through `scripts/telegram/process_telegram_commands.py`

The current Telegram summary builder reads `data/processed/final_decisions.csv`, groups rows by `final_action`, summarizes scanner observations, omits low-information scanner observations, and writes `reports/daily/telegram_message.txt`.

The current legacy markdown reports contain setup scores, grades, entries, stops, targets, risk/reward values, ranked setup sections, and actionable headings. These artifacts predate the current certified architecture and must be treated as legacy reporting evidence, not as approved Sprint 8 target behaviour.

## Reporting Governance Doctrine

Reporting is a communication layer only.

Reporting may:

- summarize already-approved outputs
- visualize already-approved outputs
- communicate Decision Engine outputs
- organize information for operators
- explain source-provenanced fields
- format outputs for Telegram, dashboards, files, or operator views
- aggregate already-approved outputs when aggregation rules are explicit and auditable
- expose observability about reporting generation

Reporting may never:

- create allocation decisions
- modify allocation decisions
- suppress opportunities
- introduce hidden filtering
- reprioritize opportunities
- create tradeability logic
- introduce urgency semantics
- override Decision Engine outputs
- mutate upstream artifacts
- create ranking authority
- create scoring authority
- derive execution semantics from upstream metadata combinations

## Reporting Layer Boundaries

Reporting may consume certified artifacts only.

Approved candidate inputs for future Sprint 8 implementation planning:

- `data/processed/final_decisions.csv`
- `data/processed/stability_state.csv`
- `data/processed/portfolio_intelligence.csv`
- `data/logs/decision_engine_log.csv`
- `data/logs/stability_layer_log.csv`
- certified source metadata or logs required for traceability

Reporting must not write back to:

- scanner outputs
- validation outputs
- context outputs
- fundamental outputs
- timing outputs
- portfolio intelligence outputs
- Decision Engine outputs
- Stability Layer outputs

Reporting outputs must be separate communication artifacts, such as:

- Telegram message text
- reporting dashboard data
- reporting logs
- reporting audit summaries
- reporting observability snapshots

## Allowed Reporting Behaviour

Allowed behaviours:

- display `final_action` exactly as produced by the Decision Engine
- display `allocation_decision` exactly as produced by the Decision Engine
- display `execution_decision` exactly as produced by the Decision Engine
- display `stability_state` exactly as produced by the Stability Layer
- display portfolio context when it is clearly labelled as source metadata
- group rows by existing Decision Engine fields when group rules are documented
- count rows by existing fields
- show source artifact paths
- show row counts and preservation checks
- show distribution summaries
- show missing-source warnings
- create deterministic operator-friendly formatting

Allowed reporting aggregation must remain communication-only. Aggregation cannot determine which opportunities matter more, which decisions deserve action, or which rows should be hidden.

## Forbidden Reporting Behaviour

Forbidden behaviours:

- hidden filtering
- hidden prioritization
- hidden ranking
- hidden allocation
- execution signalling
- tradeability classification
- urgency classification
- implicit recommendation language
- action reinterpretation
- conviction recalculation
- allocation priority recalculation
- new entry, stop, target, or sizing instructions
- derived buy, sell, trim, accumulate, or remove logic
- suppressing rows for readability without explicit audit metadata
- mutating source artifacts
- using scanner scores or grades to create reporting order
- using stability metadata to delay, accelerate, suppress, or emphasize decisions as execution instructions

Reporting must not contain language such as:

- "actionable"
- "best"
- "top"
- "ranked"
- "urgent"
- "execute now"
- "buy now"
- "recommended"
- "priority" unless it is a direct Decision Engine field label and source-provenanced

## Reporting Data Contracts

Minimum future reporting input contract:

- source artifact path
- source artifact contract version when available
- `ticker`
- `date`
- Decision Engine output fields required for communication
- Stability Layer metadata fields required for persistence communication
- row identity fields

Minimum future reporting output contract:

- `ticker`
- `date`
- `report_section`
- `source_artifact`
- `source_row_identity`
- `display_action`
- `display_reason`
- `stability_state`
- `reporting_reason`
- `reporting_contract_version`

Output contract constraints:

- no recalculated tradeability fields
- no recalculated conviction fields
- no hidden filter flags
- no new ranking fields
- no new scoring fields
- no execution urgency fields
- no recommendation fields

If a report groups, summarizes, truncates, or aggregates rows, the output must include enough metadata to prove which source rows were included, summarized, or represented.

## Reporting Observability Requirements

Reporting must log:

- input artifact paths
- input schema status
- input row counts
- output row counts
- row preservation status
- source universe preservation status
- grouping rules used
- truncation rules used
- omitted row counts, if any
- summarized row counts, if any
- output artifact paths
- generation timestamp
- reporting contract version
- forbidden field scan status
- source traceability status

Operator-facing runtime messages must describe pipeline progress only. They must not imply execution urgency, trade recommendations, allocation priority, or hidden filtering.

## Reporting Auditability Requirements

Every reporting output must be traceable to certified source artifacts.

Audit requirements:

- one source row identity per displayed row when row-level display is used
- deterministic mapping from source rows to report sections
- explicit documentation of grouping and summary rules
- explicit documentation of any truncation or display limits
- clear distinction between displayed rows and summarized rows
- no silent row loss
- no untracked omission
- no report-only interpretation of allocation meaning

Telegram messages must be reproducible from source files and reporting configuration.

## Reporting Determinism Requirements

Reporting must be deterministic for identical inputs and configuration.

Required deterministic controls:

- stable input ordering
- stable grouping order
- stable section order
- stable tie handling
- stable text templates
- deterministic missing-value rendering
- deterministic output schema
- deterministic row-count logging

Non-deterministic timestamps may appear only in reporting logs as run metadata. They must not alter reporting content or reporting semantics unless explicitly documented as report generation time.

## Reporting Distribution Preservation

Reporting must preserve the Decision Engine output universe for audit purposes.

Reporting must not silently suppress rows, opportunities, or Decision Engine outputs for readability purposes.

Any aggregation, grouping, truncation, prioritization, or summarization rules must remain deterministic, explicitly documented, and governance-auditable.

If a communication surface cannot display every row, it must:

- state the total source row count
- state the displayed row count
- state the summarized row count
- state the omitted row count, if any
- explain the deterministic display rule
- provide the full source artifact path

The reporting layer may use compact communication formats, but compactness cannot become hidden filtering.

## Reporting Risks

Primary risks:

- reporting becomes hidden allocation logic
- reporting becomes hidden prioritization
- reporting becomes hidden ranking
- reporting becomes execution signalling
- reporting suppresses rows for readability
- reporting converts metadata into recommendations
- reporting displays legacy scanner scores as decision authority
- reporting uses words that imply urgency or trade recommendations
- reporting mutates source artifacts or generated decision outputs
- reporting remains mixed-language and violates repository language governance

The highest-risk current behaviour is row omission in Telegram summaries without a full audit contract proving row-level representation.

## Legacy Semantic Drift Analysis

Current legacy drift identified:

- `reports/daily/market_scan_*.md` contains "Pullback setups (actionable)", which implies execution readiness.
- `reports/daily/market_scan_*.md` contains entries, stops, targets, risk/reward, grades, scores, and ranked setup sections, which can imply execution, ranking, scoring, and recommendation semantics.
- `reports/daily/telegram_message.txt` includes "Low-information scanner observations omitted", which is an explicit row omission behaviour that requires audit-safe replacement or traceability.
- `scripts/reporting/build_telegram_summary.py` groups rows into active decisions and observation candidates, which may be acceptable only if future implementation documents deterministic grouping and row preservation.
- `scripts/reporting/build_telegram_summary.py` uses low-information logic based on field combinations, which is a potential hidden filtering and hidden suppression risk.
- `scripts/reporting/build_telegram_summary.py` uses observation example limits, which can create display truncation risk if not accompanied by source row counts and traceability.
- `scripts/reporting/build_telegram_summary.py` contains a non-English missing-file message and non-ASCII display separators.
- `scripts/reporting/send_telegram.py` contains non-English comments and operator-facing error messages.
- `scripts/telegram/process_telegram_commands.py` contains non-English comments, non-ASCII symbols, and command-processing behaviour for BUY and SELL text commands. That command-processing path is operationally separate from reporting, but it is a governance-adjacent Telegram risk because it handles execution-like user commands.
- `tests/reporting/test_build_telegram_summary.py` currently validates omission of low-information scanner observations, which confirms legacy suppression-like reporting behaviour is encoded in tests.

Hidden urgency semantics:

- No direct "urgent" language was found in current reporting output, but action section ordering and active decision grouping can imply priority if not explicitly defined as formatting only.

Hidden execution semantics:

- Legacy reports expose entry, stop, target, risk/reward, "actionable", and "watch for pullback/retest" language.
- Telegram command processing accepts BUY and SELL text commands in a separate Telegram path.

Hidden prioritization semantics:

- Legacy reports include ranked setup sections, scores, grades, and top-style ordering.
- Telegram active decision sections may imply higher importance than observation sections.

Implicit recommendation language:

- "actionable", "entry", "stop", "target", and risk/reward formatting can imply recommended trading action.

Legacy filtering behaviour:

- Telegram summary logic identifies low-information scanner observations and omits row-level display.

Hidden suppression behaviour:

- The current Telegram message reports omitted scanner observations. This is disclosed in text, but not governed by a formal reporting contract with row-level traceability and deterministic omission metadata.

## Non-Scope

Sprint 8 preparation does not authorize:

- runtime code changes
- test changes
- generated CSV or report changes
- implementation of `build_reporting_layer.py`
- Telegram summary rewrites
- dashboard generation
- historical reporting generation
- command-processing changes
- strategy changes
- threshold changes
- allocation logic
- ranking logic
- scoring logic
- execution logic

This document is preparation only.

## Governance Constraints

Future Sprint 8 execution planning must enforce:

- Reporting communicates only.
- Decision Engine remains the only allocation authority.
- Reporting must not recalculate tradeability.
- Reporting must not recalculate conviction.
- Reporting must not create urgency classifications.
- Reporting must not create hidden priority.
- Reporting must not create hidden ranking.
- Reporting must not suppress rows silently.
- Reporting must not mutate upstream or Decision Engine artifacts.
- Reporting outputs must be English-only.
- Reporting logs and operator messages must be English-only.
- Any display limit must have explicit audit metadata.

## Required Future Inputs

Future Sprint 8 execution planning should define the authoritative reporting input set from:

- `data/processed/final_decisions.csv`
- `data/processed/stability_state.csv`
- `data/logs/decision_engine_log.csv`
- `data/logs/stability_layer_log.csv`
- any certified portfolio context artifact approved for communication

The execution plan must resolve whether legacy market scan markdown reports remain in scope, are archived, or are replaced by Decision Engine-traceable reports.

## Expected Reporting Outputs

Candidate future outputs:

- `reports/daily/telegram_message.txt`
- `data/processed/reporting_dashboard_data.csv`
- `data/logs/reporting_layer_log.csv`
- reporting audit summary artifacts
- dashboard-ready distribution artifacts

Expected outputs must:

- be source-traceable
- preserve row visibility through audit metadata
- display Decision Engine fields without reinterpretation
- display Stability Layer fields without reinterpretation
- document grouping and summary rules
- remain deterministic
- remain English-only

## Backlog Impact Assessment

Backlog impact assessment: New backlog items identified and added to project_backlog.md.

Added backlog item:

- `BL-0006`: Remediate legacy Reporting and Telegram semantic drift.

## Sprint 8 Preparation Conclusion

Sprint 8 preparation is complete and recommends governance audit before execution planning.

The current reporting layer contains legacy semantic drift that must be corrected before Sprint 8 can be certified for implementation. The correction must be governed through execution planning and developer specification, not through ad hoc code changes.

Sprint 8 may proceed to governance audit. Implementation is not authorized by this preparation document.
