# Sprint 8 Execution Plan

Sprint 8: Reporting Layer  
Planning date: 2026-05-11  
Status: EXECUTION PLANNING COMPLETE  
Scope: Planning and governance only

## Executive Summary

Sprint 8 execution planning converts the certified Reporting Layer preparation and governance audit into an implementation-ready plan.

Reporting remains a communication and observability layer only. It may display, summarize, organize, format, and audit already-approved outputs, but it may not create or modify decisions, suppress opportunities, rank opportunities, score opportunities, reprioritize rows, create execution signals, create urgency semantics, or override Decision Engine output.

This plan incorporates the required corrections from `docs/audits/sprint_8_governance_audit.md`. It defines authoritative inputs, output contracts, row-preservation rules, deterministic ordering, grouping, truncation, traceability, auditability, observability, logging, Telegram governance boundaries, legacy remediation, failure handling, and future test requirements.

No implementation is authorized by this document.

## Sprint Scope

Sprint 8 implementation scope is limited to Reporting Layer communication artifacts and reporting observability artifacts.

Authorized future implementation scope:

- replace legacy reporting semantics with Decision Engine-traceable reporting
- produce deterministic Reporting Layer outputs from certified source artifacts
- produce machine-readable reporting audit metadata
- produce operator-facing Telegram communication from reporting artifacts
- preserve row identity and source-universe traceability
- remove or quarantine legacy report-only hidden filtering, ranking, scoring, and execution-like semantics
- normalize reporting and Telegram repository content to English-only

Out-of-scope work remains prohibited unless separately authorized through sprint governance.

## Execution Objectives

Execution objectives:

- preserve Decision Engine authority
- preserve source row universe visibility
- remove hidden omission behaviour
- convert compact communication into governed representation metadata
- make all grouping and truncation deterministic and auditable
- separate outbound reporting from inbound Telegram command handling
- produce English-only repository content and generated reporting output
- provide tests proving no hidden filtering, prioritization, ranking, scoring, urgency semantics, execution semantics, row loss, or source mutation

## Reporting Runtime Architecture

Target runtime flow:

```text
data/processed/final_decisions.csv
        |
        v
Reporting contract validation
        |
        v
Reporting representation builder
        |
        +--> data/processed/reporting_dashboard_data.csv
        +--> data/logs/reporting_layer_log.csv
        +--> reports/daily/telegram_message.txt
```

Optional source metadata flow:

```text
data/processed/stability_state.csv
        |
        v
Reporting representation builder
```

Runtime ownership:

- Decision Engine owns allocation decisions in `data/processed/final_decisions.csv`.
- Stability Layer owns persistence metadata in `data/processed/stability_state.csv`.
- Reporting owns only derived communication representation, reporting logs, reporting audit metadata, and outbound message text.
- Reporting must never write back to certified upstream or Decision Engine artifacts.

## Reporting Input Contracts

Authoritative Decision Engine reporting source:

- `data/processed/final_decisions.csv`

Optional certified metadata source:

- `data/processed/stability_state.csv`

Required Decision Engine source fields for future implementation:

- `ticker`
- `date`
- Decision Engine action or decision fields displayed exactly as produced by the source artifact
- Decision Engine reason or rationale fields displayed exactly as produced by the source artifact when available

Required source identity fields:

- `ticker`
- `date`
- `source_row_index`
- `source_row_identity`
- `source_artifact_path`

If `source_row_identity` is not present in the source artifact, Reporting must deterministically derive it from source artifact path, zero-based source row index, ticker, and date. Derived row identity is traceability metadata only and must not affect decision semantics.

Input validation rules:

- fail fast on missing required columns when `final_decisions.csv` exists
- fail fast on duplicate source row identities
- fail fast on duplicate ticker/date/source row combinations when they make row identity ambiguous
- do not fabricate missing source rows
- do not infer missing allocation fields
- do not mutate input artifacts
- treat missing optional Stability Layer input as `SOURCE_UNAVAILABLE` metadata only

## Reporting Output Contracts

Required future reporting outputs:

- `data/processed/reporting_dashboard_data.csv`
- `data/logs/reporting_layer_log.csv`
- `reports/daily/telegram_message.txt`

Required `data/processed/reporting_dashboard_data.csv` schema:

```text
ticker
date
source_artifact_path
source_row_identity
source_row_index
reporting_contract_version
report_section
display_mode
source_decision_action
source_decision_state
source_decision_reason
stability_state
display_text
representation_reason
grouping_rule
truncation_rule
deterministic_ordering_rule
```

Required `data/logs/reporting_layer_log.csv` schema:

```text
run_id
generated_at
reporting_contract_version
input_artifact
stability_artifact
dashboard_output_artifact
telegram_output_artifact
input_status
stability_status
source_row_count
dashboard_row_count
displayed_row_count
summarized_row_count
omitted_row_count
row_count_preserved
ticker_date_universe_preserved
source_order_preserved
grouping_rule
truncation_rule
deterministic_ordering_rule
source_artifact_path
source_traceability_status
forbidden_semantics_status
english_only_status
upstream_artifacts_mutated
classification_rationale
```

Output contract constraints:

- no new ranking fields
- no new scoring fields
- no tradeability classification fields
- no urgency classification fields
- no execution instruction fields
- no recommendation fields
- no allocation override fields
- no hidden filter flags

Any pass-through field from the Decision Engine must retain source provenance and must not be renamed into stronger semantics.

## Reporting Distribution-Preservation Rules

Reporting must preserve the source universe.

Rules:

- every row from `data/processed/final_decisions.csv` must appear in `data/processed/reporting_dashboard_data.csv`
- reporting dashboard row count must equal source row count unless a fail-fast validation error prevents output
- ticker/date universe must be preserved
- source row identity must be preserved or deterministically derived
- no row may disappear for readability, compactness, layout, or operator convenience
- Telegram compactness must be represented through display metadata, not row omission
- `omitted_row_count` must be `0` for compliant outputs
- any row not individually rendered in Telegram must remain represented through group coverage metadata and dashboard data

Reporting must not silently suppress rows, opportunities, or Decision Engine outputs for readability purposes. Any aggregation, grouping, truncation, prioritization, or summarization rules must remain deterministic, explicitly documented, and governance-auditable.

## Reporting Determinism Rules

Determinism rules:

- input rows are processed in source file order
- source file order is preserved in `source_row_index`
- report section order is fixed by contract version
- group ordering is fixed by explicit grouping rules
- row ordering inside groups is source order unless a future developer specification explicitly defines a non-semantic deterministic order
- missing values are rendered as `SOURCE_UNAVAILABLE`
- timestamps appear only in logs, not as row ordering inputs
- no randomized sampling is allowed
- no score-based, ranking-based, urgency-based, or actionability-based ordering is allowed

Fixed report section order:

```text
DECISION_OUTPUTS
STABILITY_METADATA
SOURCE_COVERAGE
OPERATIONAL_NOTES
```

The section order is operational and must not imply priority, urgency, rank, or allocation importance.

## Reporting Grouping Rules

Allowed grouping fields:

- source Decision Engine action fields, displayed as source facts only
- source Decision Engine state fields, displayed as source facts only
- Stability Layer metadata fields, displayed as persistence metadata only
- source artifact status fields

Forbidden grouping behaviour:

- grouping by derived urgency
- grouping by derived tradeability
- grouping by derived score
- grouping by derived rank
- grouping by hidden eligibility
- grouping that changes row visibility
- grouping that changes row inclusion

Default grouping rule:

```text
GROUP_BY_SOURCE_DECISION_ACTION_THEN_SOURCE_ORDER
```

Default grouping is for communication structure only. It is not an allocation priority, recommendation order, or execution sequence.

## Reporting Truncation Rules

Dashboard data must not be truncated.

Telegram output may be compact only when all of the following are true:

- every source row remains represented in `data/processed/reporting_dashboard_data.csv`
- Telegram text states source row count and represented row count
- rows not individually shown in Telegram are marked as `GROUP_REPRESENTED` in dashboard data
- `omitted_row_count` remains `0`
- truncation rule is recorded in `data/logs/reporting_layer_log.csv`
- group counts and source artifact path are included

Default Telegram truncation rule:

```text
TELEGRAM_GROUP_SUMMARY_WITH_SOURCE_ORDER_EXAMPLES
```

Example limits must use source order only and must not select examples by score, rank, urgency, or actionability.

## Reporting Traceability Rules

Traceability requirements:

- every dashboard row must carry `source_artifact_path`
- every dashboard row must carry `source_row_identity`
- every dashboard row must carry `source_row_index`
- every Telegram message must identify the source artifact path
- every Telegram message must state source row count and represented row count
- every report generation run must write a log row with `run_id`
- every run must declare `reporting_contract_version`
- any optional source absence must be logged as source status, not hidden

Traceability must prove that Reporting communicated existing outputs and did not create an alternate decision universe.

## Reporting Auditability Rules

Auditability requirements:

- machine-readable dashboard data must exist for every generated Telegram message
- machine-readable log metadata must exist for every generated Telegram message
- grouping rule must be explicit
- truncation rule must be explicit
- deterministic ordering rule must be explicit
- source row count must be logged
- displayed row count must be logged
- summarized row count must be logged
- omitted row count must be logged
- source-universe preservation must be logged
- forbidden semantics scan status must be logged
- English-only scan status must be logged

Human-readable Telegram output is not sufficient audit evidence by itself.

## Reporting Observability Rules

Observability must distinguish operator pipeline status from report content.

Required observability dimensions:

- input availability
- input schema status
- optional source availability
- row-count preservation
- ticker/date universe preservation
- output artifact write status
- forbidden semantics scan status
- English-only status
- source mutation status
- reporting contract version

Operator-facing runtime messages must remain neutral. They must not imply trading urgency, recommendation strength, priority, ranking, scoring, or actionability.

## Reporting Logging Schema

Future implementation must write `data/logs/reporting_layer_log.csv` using this exact schema:

```text
run_id
generated_at
reporting_contract_version
input_artifact
stability_artifact
dashboard_output_artifact
telegram_output_artifact
input_status
stability_status
source_row_count
dashboard_row_count
displayed_row_count
summarized_row_count
omitted_row_count
row_count_preserved
ticker_date_universe_preserved
source_order_preserved
grouping_rule
truncation_rule
deterministic_ordering_rule
source_artifact_path
source_traceability_status
forbidden_semantics_status
english_only_status
upstream_artifacts_mutated
classification_rationale
```

The log is reporting observability only. It must not contain allocation decisions created by Reporting.

## Telegram Governance Boundaries

Outbound Telegram reporting:

- may communicate already-approved Decision Engine outputs
- may communicate source row counts and representation counts
- may communicate source artifact paths
- may communicate Stability Layer metadata as persistence metadata only
- may compact formatting through governed group representation
- must not imply urgency, ranking, actionability, priority, recommendation, or execution readiness

Inbound Telegram command handling:

- is operational command infrastructure, not Reporting Layer decision generation
- must remain separate from outbound reporting implementation
- must not be used as evidence that Reporting can create BUY, SELL, REMOVE, urgency, or execution semantics
- requires separate governance treatment if execution-like commands remain in repository scope

Sprint 8 Reporting implementation must not expand inbound Telegram command capabilities.

## Legacy Reporting Remediation Strategy

Decision: legacy `reports/daily/market_scan_*.md` outputs must be archived as historical artifacts and replaced for active reporting by Decision Engine-traceable Reporting Layer artifacts.

Execution strategy for future implementation:

- do not regenerate legacy market scan format as active Sprint 8 output
- remove active reliance on legacy ranked setup sections
- remove active reliance on setup scores, grades, entry, stop, target, and risk/reward presentation
- replace legacy report generation with source-provenanced communication artifacts
- preserve historical files unless explicit archival cleanup is separately authorized
- document legacy artifacts as pre-certified architecture outputs when referenced

Legacy remediation is governed by BL-0006.

## English-Only Remediation Scope

Sprint 8 future implementation must normalize newly touched Reporting and Telegram repository content to English-only.

Required scope:

- reporting scripts touched by Sprint 8
- Telegram outbound reporting text touched by Sprint 8
- reporting tests touched by Sprint 8
- generated reporting outputs produced by Sprint 8
- reporting logs produced by Sprint 8
- reporting documentation produced by Sprint 8

Existing legacy mixed-language content outside Sprint 8 touch scope remains governed by BL-0005 unless it directly affects Reporting or Telegram remediation under BL-0006.

## Reporting Test Strategy

Future implementation tests must cover:

- no hidden filtering
- no hidden prioritization
- no hidden ranking
- no hidden scoring
- no hidden execution semantics
- no hidden urgency semantics
- no silent row loss
- deterministic output generation
- deterministic ordering
- deterministic grouping
- deterministic truncation
- English-only reporting output
- traceability preservation
- source-universe preservation
- no source mutation
- schema enforcement
- missing input handling
- missing optional Stability Layer handling
- duplicate row identity handling
- forbidden keyword scanning
- Telegram compact representation metadata
- inbound and outbound Telegram separation

Forbidden keyword tests must avoid treating negative assertions in tests as runtime violations.

## Forbidden Semantics Enforcement

Future implementation must enforce forbidden semantics through:

- schema denylist checks for Reporting Layer outputs
- runtime content scans for generated text
- tests scanning Reporting Layer source files
- tests scanning generated reporting artifacts
- review of section names and headings
- review of grouping and truncation rules

Forbidden runtime semantics include:

- hidden filtering
- hidden prioritization
- hidden ranking
- hidden scoring
- execution signalling
- urgency classification
- tradeability classification
- recommendation language
- allocation override
- source mutation

## Failure Handling Rules

Failure handling rules:

- missing `data/processed/final_decisions.csv` produces empty schema-compliant reporting outputs and a log row with `input_status` set to `SOURCE_MISSING`
- empty `data/processed/final_decisions.csv` produces empty schema-compliant reporting outputs and a log row with `input_status` set to `SOURCE_EMPTY`
- missing required source columns fails fast and must not produce partial reporting outputs
- duplicate source row identity fails fast
- missing optional `data/processed/stability_state.csv` does not fail the run and must be logged as `stability_status` set to `SOURCE_UNAVAILABLE`
- generated output write failure fails the run
- English-only scan failure fails the run before publication
- forbidden semantics scan failure fails the run before publication
- source artifact mutation detection fails the run

No failure path may fabricate reporting rows or allocation meaning.

## Non-Scope

Sprint 8 execution planning does not authorize:

- runtime code changes
- test changes
- generated data changes
- Decision Engine changes
- Stability Layer changes
- upstream layer changes
- strategy optimization
- threshold tuning
- new allocation logic
- new filtering logic
- new ranking logic
- new scoring logic
- new execution command behaviour
- Telegram inbound command expansion
- production deployment

## Governance Constraints

Sprint 8 implementation must obey:

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
- repository content is English-only

## Backlog Impact Assessment

Backlog impact assessment: No new backlog items identified.

BL-0006 already captures the Reporting and Telegram semantic drift remediation required for Sprint 8 execution. No additional deferred work was identified during this execution planning phase.

## Sprint 8 Execution Planning Conclusion

Sprint 8 execution planning is complete and ready for Technical Lead execution review.

The plan incorporates the governance audit's required corrections, defines enforceable reporting contracts, preserves Decision Engine authority, requires deterministic and auditable reporting outputs, removes hidden omission as an allowed behaviour, separates outbound Telegram reporting from inbound command handling, and prepares Sprint 8 for Developer Specification after execution review approval.
