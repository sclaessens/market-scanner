# Operational Sprint 5 Data Steward / Data Analyst Role Governance

## 1. Status and Scope

This document is a documentation-only role governance note for Operational Sprint 5.

It defines the Data Steward / Data Analyst responsibility boundary for future manual, source-supported data collection connected to the OS5 scanner A/B intake pilot.

This document does not implement:

- code changes
- tests
- CSV changes
- generated artifacts
- reports
- GitHub Actions workflows
- provider integration
- runtime orchestration
- Reporting changes
- Telegram changes
- scanner changes
- Decision Engine changes
- validation, context, timing, fundamental, or portfolio intelligence runtime changes
- portfolio file changes
- watchlist file changes
- fundamentals source file changes

No sprint is closed or certified complete by this document.

This document records role governance only. It does not authorize implementation, source-data population, provider/API usage, or runtime behavior changes.

## 2. Background

Operational Sprint 5 has established a governed source-data intake path for improving source-supported coverage without weakening architecture boundaries.

Relevant OS5 artifacts include:

- `docs/sprints/operational_sprint_5_target_universe_refinement.md`
- `docs/sprints/operational_sprint_5_scanner_coverage_audit_followup.md`
- `docs/sprints/operational_sprint_5_source_data_expansion_plan.md`
- `docs/sprints/operational_sprint_5_source_data_intake_worklist.md`
- `docs/sprints/operational_sprint_5_manual_intake_pilot.md`
- `data/intake/os5_scanner_ab_metadata_intake_template.csv`
- `data/intake/os5_scanner_ab_fundamentals_intake_template.csv`
- `data/intake/os5_scanner_ab_metadata_intake_pilot.csv`
- `data/intake/os5_scanner_ab_fundamentals_intake_pilot.csv`

The 145-row scanner A/B intake templates exist. The 10-row manual intake pilot files exist. No factual metadata or fundamentals values have been populated by those templates or pilot files.

The next possible phase is a human/manual source-supported data collection pilot for the 10 pilot tickers only. A role boundary is required before that work begins so that source evidence, uncertainty handling, ChatGPT assistance, Codex processing, and governance limits remain explicit.

## 3. Data Steward / Data Analyst Role Definition

The Data Steward / Data Analyst role is responsible for preparing reviewed, source-supported values for later controlled insertion into the OS5 pilot intake files.

The role may:

- identify acceptable external sources;
- compare multiple sources when field interpretation is ambiguous;
- collect source-supported values;
- record source names;
- record source references or URLs;
- record freshness dates or last-updated dates;
- leave uncertain or unsupported values empty;
- document ambiguity in notes fields;
- prepare reviewed values for later technical insertion by Codex or another implementation actor;
- recommend whether a field should remain unpopulated because source support is insufficient.

The role is a data stewardship and evidence-review function only. It is not an allocation authority, scanner authority, runtime implementation authority, or portfolio/watchlist reconstruction authority.

## 4. Explicit Role Limitations

The Data Steward / Data Analyst role must not:

- invent values;
- infer values without source support;
- treat ChatGPT output as a primary source;
- use unsourced values;
- fill convenience values because they appear obvious;
- make allocation decisions;
- change scanner selection;
- change scanner grading;
- change Decision Engine logic;
- change Reporting logic;
- change Telegram logic;
- rebuild portfolio files;
- rebuild watchlist files;
- call provider APIs unless separately authorized;
- add credentials or secrets;
- bypass existing governance;
- introduce ranking, scoring, tradeability, urgency, conviction, allocation, eligibility, or hidden filtering semantics.

Unsupported values must remain empty. Ambiguous values must remain empty unless the ambiguity is resolved by a documented source interpretation and the interpretation is recorded in a notes field.

## 5. ChatGPT Assistance Boundary

ChatGPT may assist the Data Steward / Data Analyst with:

- research planning;
- identifying candidate source types;
- comparing sources;
- summarizing source evidence;
- drafting structured review tables;
- highlighting uncertainty;
- identifying fields that require human review;
- preparing Codex prompts after human review.

ChatGPT must not be treated as the factual source of record.

Every factual value must be traceable to an external source. A value is not acceptable merely because ChatGPT suggests it, remembers it, infers it, or presents it confidently.

If ChatGPT assists with research, the retained record must still identify the external source name, URL or stable reference where possible, freshness date or last-updated date, and any interpretation notes required for the value.

## 6. Source Evidence Rules

Every populated value must include sufficient source evidence.

At minimum, each populated row or field must record:

- source name;
- source URL or stable reference where possible;
- freshness date, as-of date, reporting period, or last-updated date;
- interpretation note if the field requires judgment or mapping;
- data quality note if the source is partial, stale, inconsistent, or ambiguous.

If source support is missing, the value must remain empty.

If source support conflicts across sources, the value must remain empty unless a documented source hierarchy or governance decision resolves the conflict.

If a source provides a conceptually adjacent field rather than the exact target field, the value must remain empty unless the mapping is explicitly documented and later approved as compatible with the contract.

## 7. Pilot Limitation

The first manual source-supported collection phase is limited to the 10 OS5 pilot tickers.

This document does not authorize filling all 145 scanner A/B rows.

This document does not authorize broader scanner-wide completion.

This document does not authorize portfolio repair, watchlist repair, or portfolio/watchlist rebuild.

Broader completion requires separate authorization after the pilot proves that the intake structure, source-evidence rules, and review workflow are practical.

## 8. Governance Constraints

The certified project doctrine remains unchanged:

- classification upstream;
- allocation downstream;
- Decision Engine is the only allocation authority;
- upstream layers classify and enrich only;
- reporting communicates only;
- no hidden filtering;
- no ranking authority outside approved Decision Engine logic;
- no scoring authority outside approved Decision Engine logic.

Scanner A/B remains coverage-prioritization only.

This role definition introduces no:

- ranking authority;
- scoring authority;
- tradeability semantics;
- urgency semantics;
- conviction semantics;
- allocation semantics;
- eligibility semantics;
- hidden filtering semantics;
- Decision Engine bypass.

The Data Steward / Data Analyst may prepare source-supported data values for review. The role may not transform those values into trading decisions, portfolio actions, or reporting recommendations.

## 9. Codex Processing Boundary

Codex may be used later to technically insert reviewed, source-supported values into pilot CSV files only after explicit authorization.

Codex must not independently research, infer, guess, or populate values unless a future prompt explicitly authorizes a specific governed data collection process and source policy.

For the OS5 manual pilot, the intended separation is:

1. Data Steward / Data Analyst researches and reviews source-supported values.
2. Human operator approves the values for pilot insertion.
3. Codex inserts exactly the approved values into the pilot CSV files.
4. Codex validates structure and confirms no runtime files were changed.

This document does not authorize step 3. It only defines the role boundary required before step 1 begins.

## 10. Backlog Impact Assessment

Existing related backlog items are sufficient for this role governance note.

Relevant existing backlog coverage includes:

- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- existing portfolio and watchlist-related backlog items already present in `docs/sprints/project_backlog.md`
- existing data-contract and operational intelligence backlog coverage for source-supported metadata and fundamentals

This document defines a role boundary for the already-governed OS5 intake path. It does not identify a new deferred implementation requirement beyond the existing backlog scope.

Backlog impact assessment:
- No new backlog items identified.

## 11. Recommended Next Step

Run a separate human/manual Data Steward pilot for the 10 OS5 pilot tickers.

That future pilot should:

- collect source-supported values for the 10 pilot tickers only;
- keep unsupported values empty;
- record source names;
- record source URLs or stable references where possible;
- record freshness dates or last-updated dates;
- document ambiguity in notes fields;
- avoid provider/API calls unless separately authorized;
- avoid runtime behavior changes;
- avoid changes to existing source CSVs outside the explicitly authorized pilot files.

Codex should only be used later to insert reviewed, source-supported values into the pilot CSVs after explicit authorization.
