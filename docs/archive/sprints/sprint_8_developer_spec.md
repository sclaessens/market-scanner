# Sprint 8 Developer Specification

Sprint 8: Reporting Layer  
Specification date: 2026-05-11  
Status: DEVELOPER SPECIFICATION COMPLETE  
Authority: Sprint 8 Developer Specification Authority

## Executive Summary

This Developer Specification authorizes Sprint 8 Reporting Layer implementation scope.

The implementation must create a deterministic, source-traceable, audit-safe Reporting Layer that communicates Decision Engine output without creating allocation authority, hidden filtering, hidden prioritization, hidden ranking, hidden scoring, hidden urgency semantics, hidden execution semantics, silent row loss, or source mutation.

The authoritative Reporting builder must be `scripts/reporting/build_reporting_layer.py`. It must read `data/processed/final_decisions.csv`, optionally enrich communication with `data/processed/stability_state.csv` as persistence metadata only, write `data/processed/reporting_dashboard_data.csv`, write `data/logs/reporting_layer_log.csv`, and write `reports/daily/telegram_message.txt`.

This specification authorizes implementation only within the boundaries below.

## Approved Governance Context

Certified architecture:

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
- repository content is English-only

Sprint 8 implementation must follow:

- `docs/sprints/sprint_8_reporting_preparation.md`
- `docs/audits/sprint_8_governance_audit.md`
- `docs/sprints/sprint_8_execution_plan.md`
- `docs/audits/sprint_8_execution_review.md`

## Sprint 8 Implementation Scope

Authorized implementation files and responsibilities:

- `scripts/reporting/build_reporting_layer.py`: create as authoritative Reporting Layer builder.
- `scripts/reporting/build_telegram_summary.py`: refactor into a compatibility wrapper that delegates to `build_reporting_layer.py`; it must not retain independent grouping, omission, filtering, or decision presentation logic.
- `scripts/reporting/send_telegram.py`: normalize English-only comments and operator-facing messages; keep delivery-only responsibility.
- `scripts/reporting/reporter.py`: quarantine legacy markdown reporting semantics from active Sprint 8 reporting; do not use legacy setup, score, grade, entry, stop, target, or risk/reward presentation as active Reporting Layer output.
- `scripts/telegram/process_telegram_commands.py`: isolate inbound Telegram command handling from outbound Reporting; do not expand command behaviour; normalize English-only comments and operator-facing messages if touched.
- `tests/reporting/`: add or update tests to enforce this specification.
- `reports/daily/telegram_message.txt`: generated outbound communication artifact.
- `data/logs/reporting_layer_log.csv`: generated Reporting Layer observability and audit log.
- `data/processed/reporting_dashboard_data.csv`: generated machine-readable reporting representation artifact.

No implementation is authorized outside these files unless strictly required for imports, packaging, or test fixtures and still consistent with this specification.

## Runtime Architecture Specification

Runtime flow:

```text
data/processed/final_decisions.csv
        |
        v
scripts/reporting/build_reporting_layer.py
        |
        +--> data/processed/reporting_dashboard_data.csv
        +--> data/logs/reporting_layer_log.csv
        +--> reports/daily/telegram_message.txt
```

Optional metadata flow:

```text
data/processed/stability_state.csv
        |
        v
scripts/reporting/build_reporting_layer.py
```

Runtime ownership boundaries:

- Decision Engine owns allocation output in `data/processed/final_decisions.csv`.
- Stability Layer owns persistence metadata in `data/processed/stability_state.csv`.
- Reporting owns representation data, logs, and outbound communication text only.
- Telegram outbound delivery sends already generated text only.
- Telegram inbound command processing is operational command infrastructure only and must remain separate from Reporting authority.

## Reporting Builder Specification

`scripts/reporting/build_reporting_layer.py` must become the authoritative Reporting builder.

Required functions:

- `load_final_decisions(path: Path) -> pandas.DataFrame`
- `load_stability_state(path: Path) -> pandas.DataFrame`
- `validate_final_decisions_schema(df: pandas.DataFrame) -> None`
- `validate_stability_schema(df: pandas.DataFrame) -> None`
- `build_source_row_identity(row: pandas.Series, source_row_index: int, source_path: Path) -> str`
- `build_reporting_dashboard(final_decisions: pandas.DataFrame, stability_state: pandas.DataFrame | None) -> pandas.DataFrame`
- `build_telegram_message(reporting_dashboard: pandas.DataFrame, log_metadata: dict) -> str`
- `build_reporting_log_row(...) -> dict`
- `write_outputs(...) -> None`
- `main() -> None`

The builder must not contain allocation, eligibility, tradeability, urgency, ranking, scoring, or recommendation logic.

`scripts/reporting/build_telegram_summary.py` must call the authoritative builder or expose a compatibility function that uses the authoritative builder. It must not maintain separate legacy section order, low-information omission, scanner observation suppression, or action-priority lists.

## Reporting Input Contract Specification

Authoritative reporting source:

- `data/processed/final_decisions.csv`

Required columns in `data/processed/final_decisions.csv`:

```text
ticker
date
final_action
allocation_decision
execution_decision
portfolio_decision_state
opportunity_decision_state
arbitration_state
allocation_rationale
execution_rationale
arbitration_reason
conflict_resolution_reason
source_provenance
decision_contract_version
input_row_hash
```

Optional certified metadata source:

- `data/processed/stability_state.csv`

Required columns when `data/processed/stability_state.csv` exists and is non-empty:

```text
ticker
date
stability_state
conviction_persistence
action_persistence
behavioural_stability
transition_frequency
escalation_frequency
stability_reason
persistence_duration
```

Input validation:

- if `final_decisions.csv` is missing, write empty schema-compliant outputs and log `SOURCE_MISSING`
- if `final_decisions.csv` is empty, write empty schema-compliant outputs and log `SOURCE_EMPTY`
- if `final_decisions.csv` exists with missing required columns, fail fast
- if duplicate source row identities are generated, fail fast
- if optional `stability_state.csv` is missing, continue and log `SOURCE_UNAVAILABLE`
- if optional `stability_state.csv` exists with missing required columns, fail fast only for stability enrichment; no fabricated stability data is allowed

## Reporting Output Contract Specification

Mandatory output artifacts:

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
source_final_action
source_allocation_decision
source_execution_decision
source_portfolio_decision_state
source_opportunity_decision_state
source_arbitration_state
source_allocation_rationale
source_execution_rationale
source_arbitration_reason
source_conflict_resolution_reason
source_provenance
source_decision_contract_version
source_input_row_hash
stability_state
display_text
representation_reason
grouping_rule
truncation_rule
deterministic_ordering_rule
```

Required output values:

- `reporting_contract_version`: `REPORTING_CONTRACT_V1`
- `report_section`: one of `DECISION_OUTPUTS`, `STABILITY_METADATA`, `SOURCE_COVERAGE`, `OPERATIONAL_NOTES`
- `display_mode`: one of `ROW_DISPLAYED`, `GROUP_REPRESENTED`, `SOURCE_UNAVAILABLE`
- `grouping_rule`: `GROUP_BY_SOURCE_FINAL_ACTION_THEN_SOURCE_ORDER`
- `truncation_rule`: `TELEGRAM_GROUP_SUMMARY_WITH_SOURCE_ORDER_EXAMPLES`
- `deterministic_ordering_rule`: `SOURCE_ORDER_WITH_FIXED_SECTION_ORDER`

Forbidden output fields:

- ranking fields
- scoring fields
- tradeability classification fields
- urgency classification fields
- execution instruction fields
- recommendation fields
- allocation override fields
- hidden filter flags

## Reporting Determinism Specification

Implementation must enforce:

- stable source file ordering
- zero-based `source_row_index` assigned before any grouping
- fixed section order: `DECISION_OUTPUTS`, `STABILITY_METADATA`, `SOURCE_COVERAGE`, `OPERATIONAL_NOTES`
- grouping by source `final_action` for communication only
- group order sorted lexicographically by source `final_action`
- row order inside each group preserved by `source_row_index`
- deterministic missing value rendering as `SOURCE_UNAVAILABLE`
- no timestamp-driven row ordering
- no randomized sampling
- no sorting by score, rank, urgency, actionability, price, return, exposure, or any derived allocation meaning

## Reporting Grouping Specification

Grouping implementation:

- default group key is source `final_action`
- group labels must be neutral and formatted as `Decision output: <source_final_action>`
- group counts must be displayed
- grouping must not change row inclusion
- grouping must not change dashboard row count
- grouping must not produce priority, rank, recommendation, or execution sequence

Allowed grouping inputs:

- source `final_action`
- source Decision Engine state fields
- source artifact status
- Stability Layer metadata for display only

Forbidden grouping inputs:

- derived urgency
- derived tradeability
- derived rank
- derived score
- hidden eligibility
- hidden suppression state

## Reporting Truncation Specification

`data/processed/reporting_dashboard_data.csv` must never be truncated.

Telegram compact representation:

- Telegram may show group summaries and deterministic examples by source order
- Telegram must state source row count
- Telegram must state represented row count
- Telegram must state dashboard artifact path
- Telegram must state source artifact path
- rows individually shown in Telegram must have `display_mode` set to `ROW_DISPLAYED`
- rows represented only through group counts must have `display_mode` set to `GROUP_REPRESENTED`
- `omitted_row_count` must be `0`

Telegram examples:

- example rows must be selected by ascending `source_row_index`
- example count must be a fixed constant
- the constant must be named `TELEGRAM_GROUP_EXAMPLE_LIMIT`
- the value must be covered by tests
- examples must never be selected by score, rank, urgency, actionability, or price movement

## Reporting Traceability Specification

Source row identity:

```text
<source_artifact_path>#<source_row_index>#<ticker>#<date>#<input_row_hash>
```

Traceability requirements:

- every dashboard row must include `source_artifact_path`
- every dashboard row must include `source_row_identity`
- every dashboard row must include `source_row_index`
- every dashboard row must include source Decision Engine fields as pass-through `source_*` fields
- every Telegram message must include the source artifact path
- every Telegram message must include the dashboard artifact path
- every log row must include the source artifact path
- every log row must include row-count preservation status

Traceability metadata must never be used for allocation, priority, urgency, ranking, scoring, or filtering.

## Reporting Auditability Specification

Auditability requirements:

- every Telegram message must be reproducible from `reporting_dashboard_data.csv`
- every reporting run must append or write a machine-readable log row
- every run must record row counts and preservation booleans
- every run must record grouping and truncation rules
- every run must record forbidden semantics scan status
- every run must record English-only status
- every run must record upstream mutation status
- generated artifacts must be deterministic for the same source inputs except for log timestamp and run identifier

`reports/daily/telegram_message.txt` is not sufficient audit evidence without dashboard data and log metadata.

## Reporting Observability Specification

Observability must include:

- source availability
- source schema status
- optional stability availability
- dashboard write status
- Telegram write status
- source row count
- dashboard row count
- displayed row count
- summarized row count
- omitted row count
- row preservation status
- ticker/date universe preservation status
- source order preservation status
- forbidden semantics status
- English-only status
- source mutation status

Runtime console output must be English-only and operationally neutral.

## Reporting Logging Specification

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

Required log values:

- `reporting_contract_version`: `REPORTING_CONTRACT_V1`
- `omitted_row_count`: `0` for compliant output
- `upstream_artifacts_mutated`: `False`
- `grouping_rule`: `GROUP_BY_SOURCE_FINAL_ACTION_THEN_SOURCE_ORDER`
- `truncation_rule`: `TELEGRAM_GROUP_SUMMARY_WITH_SOURCE_ORDER_EXAMPLES`
- `deterministic_ordering_rule`: `SOURCE_ORDER_WITH_FIXED_SECTION_ORDER`

## Telegram Governance Specification

Outbound Telegram Reporting:

- must be generated from `reporting_dashboard_data.csv`
- must be source-traceable
- must be deterministic
- must be audit-safe
- must be English-only
- must state source row count
- must state represented row count
- must state omitted row count as `0`
- must state source artifact path
- must state dashboard artifact path
- must not imply urgency, ranking, priority, actionability, recommendation, or execution readiness

Inbound Telegram Commands:

- remain operational command infrastructure only
- are not Reporting Layer authority
- are not Decision Engine authority
- are not allocation authority
- must not be expanded by Sprint 8
- must remain isolated from outbound Reporting implementation

`scripts/telegram/process_telegram_commands.py` may be normalized for English-only repository governance and boundary comments, but Sprint 8 must not add command capabilities.

## Legacy Reporting Remediation Specification

Legacy `reports/daily/market_scan_*.md` reports:

- are historical artifacts
- must not be regenerated as active Sprint 8 output
- must not be treated as certified Reporting Layer output
- must not be used as input to new Reporting Layer artifacts

Legacy ranked and scored setup language:

- must be removed from active Reporting Layer output
- must not appear in generated Telegram output
- must not appear in `reporting_dashboard_data.csv`
- must not appear in `reporting_layer_log.csv`

Omission-based Telegram summaries:

- must be removed
- `Low-information scanner observations omitted` must not appear in generated output
- low-information classification must not be recreated under another name
- compactness must be represented by `GROUP_REPRESENTED` display metadata

Legacy non-English reporting strings and comments:

- must be normalized to English in touched Reporting and Telegram files
- must be covered by English-only validation tests

## English-Only Enforcement Specification

Implementation must enforce English-only repository content for touched files and generated outputs.

Required checks:

- generated Telegram output contains ASCII-compatible English operator text
- runtime log values are English-only
- comments added or modified in touched files are English-only
- tests added or modified are English-only
- generated reporting CSV column names are English-only

Non-ASCII characters must not be introduced into touched Sprint 8 files or generated Reporting outputs unless strictly required by external API payloads. No such exception is currently authorized.

## Reporting Validation Specification

Validation requirements:

- validate required final decision columns
- validate optional stability columns when source exists
- validate generated dashboard schema exactly
- validate generated log schema exactly
- validate source row identity uniqueness
- validate dashboard row count equals source row count
- validate ticker/date universe preservation
- validate source order preservation
- validate `omitted_row_count` equals `0`
- validate forbidden output fields are absent
- validate generated text passes forbidden semantics scan
- validate generated text is English-only
- validate source artifacts are not mutated

## Reporting Test Specification

Future tests in `tests/reporting/` must cover:

- reporting schema validation
- row-count preservation
- source-universe preservation
- deterministic output generation
- deterministic ordering
- deterministic grouping
- deterministic truncation
- no hidden omission
- no hidden prioritization
- no hidden ranking
- no hidden scoring
- no hidden execution semantics
- no hidden urgency semantics
- forbidden keyword scanning
- English-only output validation
- traceability validation
- source-row identity preservation
- fail-fast validation
- optional Stability Layer handling
- duplicate row identity handling
- Telegram representation metadata validation
- outbound/inbound Telegram separation validation
- source artifact mutation detection

Existing tests that validate omission-based Telegram summaries must be replaced or updated to validate row-preserving representation metadata.

## CI Enforcement Specification

CI or mandatory local validation must enforce:

- forbidden reporting semantics scan
- forbidden keyword scan
- English-only validation
- deterministic output validation
- reporting schema validation
- reporting traceability validation
- reporting row-preservation validation
- no source mutation validation

Minimum commands after implementation:

```bash
pytest
python scripts/reporting/build_reporting_layer.py
git diff --check
grep -R "Low-information scanner observations omitted" scripts/reporting tests/reporting reports/daily
grep -R "BUY NOW" scripts/reporting tests/reporting reports/daily
grep -R "urgent" scripts/reporting tests/reporting reports/daily
grep -R "ranked" scripts/reporting tests/reporting reports/daily
grep -R "score" scripts/reporting tests/reporting reports/daily
```

Keyword scans must distinguish forbidden runtime semantics from negative assertions in tests where needed.

## Failure Handling Specification

Failure handling:

- missing `final_decisions.csv`: create empty schema-compliant outputs and log `SOURCE_MISSING`
- empty `final_decisions.csv`: create empty schema-compliant outputs and log `SOURCE_EMPTY`
- missing required final decision columns: fail fast
- duplicate source row identity: fail fast
- missing optional `stability_state.csv`: continue and log `SOURCE_UNAVAILABLE`
- invalid optional stability schema: fail fast for enrichment validation
- generated output write failure: fail fast
- forbidden semantics scan failure: fail fast before publication
- English-only scan failure: fail fast before publication
- source mutation detection failure: fail fast

Failure modes must not fabricate rows or allocation meaning.

## Forbidden Semantics Enforcement Specification

Forbidden runtime terms and semantics include:

- hidden filtering
- hidden prioritization
- hidden ranking
- hidden scoring
- urgency classification
- execution signalling
- tradeability classification
- recommendation language
- allocation override
- row suppression
- source mutation

Generated Reporting outputs must not contain:

- `BUY NOW`
- `urgent`
- `ranked`
- `score`
- `best`
- `top`
- `recommended`
- `priority`
- `tradeable`
- `omitted`
- `actionable`

The word `omitted` may appear only in the required log column name `omitted_row_count` and in governance/test negative assertions, not as generated human-facing suppression language.

## Runtime Boundary Specification

`scripts/reporting/build_reporting_layer.py`:

- owns reporting generation
- owns dashboard output
- owns reporting log output
- owns Telegram message text generation

`scripts/reporting/build_telegram_summary.py`:

- compatibility entrypoint only
- delegates to `build_reporting_layer.py`
- no independent semantics

`scripts/reporting/send_telegram.py`:

- delivery only
- reads existing Telegram message text
- sends message through Telegram API
- no report generation
- no decision semantics

`scripts/reporting/reporter.py`:

- legacy markdown reporter
- not active Sprint 8 output authority
- no new Sprint 8 semantics

`scripts/telegram/process_telegram_commands.py`:

- inbound command infrastructure only
- not Reporting authority
- not allocation authority
- no command expansion in Sprint 8

## Non-Scope

Sprint 8 implementation must not:

- modify Decision Engine logic
- modify upstream classification logic
- modify Stability Layer logic
- create allocation logic
- create tradeability logic
- create urgency logic
- create ranking logic
- create scoring logic
- suppress opportunities
- remove source rows
- mutate source artifacts
- expand inbound Telegram commands
- redesign architecture
- tune strategy thresholds

## Governance Constraints

Implementation must preserve:

- Decision Engine allocation authority
- Reporting communication-only responsibility
- deterministic outputs
- source-universe preservation
- row identity traceability
- auditability
- observability
- English-only repository content
- no hidden filtering
- no hidden prioritization
- no hidden ranking
- no hidden scoring
- no hidden execution semantics
- no hidden urgency semantics

## Backlog Impact Assessment

Backlog impact assessment: No new backlog items identified.

BL-0006 already captures the Reporting and Telegram semantic drift remediation authorized for Sprint 8 implementation. No additional deferred work was identified during Developer Specification.

## Sprint 8 Developer Specification Conclusion

Sprint 8 Developer Specification is complete and implementation-authoritative.

The approved implementation path is to create `scripts/reporting/build_reporting_layer.py` as the authoritative Reporting Layer builder, refactor legacy Telegram summary generation into a compatibility wrapper, preserve source row universe and row identity, generate deterministic dashboard, log, and Telegram artifacts, enforce English-only output, remove omission-based summaries, isolate inbound Telegram commands, and test all governance boundaries before implementation audit.
