# SEC Code/Data Alignment Audit Before SEC-7F

Status: DOCUMENTATION-ONLY AUDIT
Repository: sclaessens/market-scanner
Branch: docs/sec-code-data-alignment-audit
Date: 2026-06-01

## 1. Executive Summary

This audit reviewed SEC-related code, committed CSV/data surfaces, documentation, and generated-output boundaries before any SEC-7F implementation.

Conclusion:

- SEC-7F should proceed, but only under strict implementation boundaries.
- The SEC code should be reviewed during SEC-7F implementation, but this audit does not identify a hard blocker requiring a broad pre-SEC-7F refactor.
- The CSV/data state is not a blocker for SEC-7F if SEC-7F uses fixtures, explicit local inputs, and temporary output paths only.
- Existing committed CSVs should not be used as SEC-7F source-of-truth inputs.
- Generated, stale, backup, and operational CSVs should not be cleaned up in SEC-7F.
- Any CSV/data cleanup should be handled by a separate governed cleanup sprint.
- Pipeline integration remains blocked.

Recommended path:

```text
Path A — Proceed to SEC-7F implementation with strict boundaries.
```

This audit is documentation-only. It does not modify code, tests, CSV files, data files, generated files, reports, workflows, runtime behavior, or pipeline integration.

## 2. Repository Governance Review

The following governance constraints remain binding:

- classification upstream;
- allocation downstream;
- Decision Engine is the only allocation authority;
- Reporting is communication-only;
- SEC review output is source-data readiness evidence only;
- `review_required` does not mean business weakness;
- missing values are not zero;
- no hidden filtering;
- no generated data commits unless explicitly approved;
- real SEC cache files remain local-only;
- English-only repository content;
- no pipeline integration before review semantics are stable.

SEC-7F must not create ranking, scoring, eligibility, tradeability, urgency, conviction, allocation, buy/sell, or final-action semantics.

## 3. SEC Documentation State Reviewed

Recent SEC sequence:

- SEC-7C improved fact tolerance so messy individual SEC facts no longer fail an entire ticker by default.
- SEC-7D documented the controlled local diagnostic and confirmed SEC output is not pipeline-ready.
- SEC-7E defined review modes and annual-first policy.
- SEC-7F framing was refined to require a structured review model rather than cosmetic period filtering and note cleanup.

Current SEC-7F title:

```text
SEC-7F — SEC Review Model and Period Selection Implementation
```

Required SEC-7F review model direction:

```text
AcceptedFact
RejectedFact
PeriodReviewResult
RunReviewSummary
```

The review model must separate accepted facts, rejected/skipped facts, fact-level evidence, period-level evidence, row-level evidence, ticker-level evidence, run-level diagnostic summaries, and explicit review-mode period selection.

## 4. SEC Code Inventory

### 4.1 `scripts/fundamentals/sec_companyfacts_bulk_intake.py`

Purpose:

- handles SEC Company Facts bulk ZIP source validation;
- manages local cache path family;
- validates official SEC URL and required User-Agent policy;
- supports explicit download/validation behavior;
- generates local manifest-style evidence when invoked.

Inputs:

- explicit SEC Company Facts source URL;
- explicit local cache paths;
- explicit User-Agent configuration.

Outputs:

- local SEC cache artifacts and manifests when invoked;
- no operational pipeline output by default.

CSV behavior:

- not primarily a CSV producer or consumer.

Network behavior:

- capable of live SEC/download behavior only when explicitly invoked by an operator;
- must not be invoked during SEC-7F tests.

Operational boundary:

- isolated from Decision Engine, Reporting, Telegram, portfolio, scanner, validation, context, timing, and portfolio intelligence.

Audit conclusion:

- no pre-SEC-7F code change required;
- SEC-7F should not modify this file unless implementation proves a narrow review-model dependency, which is currently unlikely.

### 4.2 `scripts/fundamentals/sec_ticker_cik_index.py`

Purpose:

- reads ticker/CIK source inputs;
- normalizes CIK values;
- builds ticker-to-CIK coverage;
- preserves requested tickers with descriptive mapping statuses.

Inputs:

- local ticker/CIK source files;
- project ticker inputs.

Outputs:

- in-memory coverage DataFrames or explicitly written local outputs if invoked by surrounding utilities.

CSV behavior:

- can read local CSV-like source files depending on operator-provided inputs;
- should not depend on generated operational CSVs for SEC-7F.

Network behavior:

- no live SEC/network call expected in the mapping logic itself.

Operational boundary:

- isolated from Decision Engine, Reporting, Telegram, portfolio, scanner, validation, context, timing, and portfolio intelligence.

Audit conclusion:

- no pre-SEC-7F code change required;
- SEC-7F may rely on this module only through fixture/temp-path tests and explicit local inputs.

### 4.3 `scripts/fundamentals/sec_companyfacts_transform.py`

Purpose:

- transforms local SEC Company Facts JSON payloads into internal fundamentals-history-style rows;
- maps direct fields;
- derives approved fields such as total debt and free cash flow under conservative formula policy;
- records notes and evidence for missing, conflicting, or review-required facts.

Inputs:

- explicit local Company Facts JSON file;
- ticker;
- CIK;
- source freshness date;
- extraction date.

Outputs:

- internal rows compatible with fundamentals history intake contract;
- optional explicit output only when called by a wrapper/runner.

CSV behavior:

- not a default operational CSV writer;
- should not write to `data/processed/` during SEC-7F.

Network behavior:

- no live SEC calls; it transforms local payloads only.

Current technical state:

- SEC-7C introduced skipped-fact evidence and fact-selection outcomes;
- problematic facts are skipped or marked review-required rather than failing entire tickers by default;
- conflicts remain review-required;
- missing values are not inferred as zero.

Known gap for SEC-7F:

- skipped facts are still conceptually collected at payload/run scope and can be repeated into row notes;
- SEC-7F should introduce a structured review model separating accepted facts, rejected facts, period review, row evidence, ticker evidence, and run-level summaries.

Audit conclusion:

- this is the primary SEC-7F implementation file;
- no pre-SEC-7F cleanup is required, but SEC-7F should change this file within the approved boundaries.

### 4.4 `scripts/fundamentals/run_sec_transformation_review.py`

Purpose:

- composes project ticker input, ticker/CIK coverage, local Company Facts JSON files, and transformation output into a controlled SEC review DataFrame;
- preserves missing CIK and missing Company Facts cases as review rows;
- emits review metadata columns such as mapping status, transformation status, review-required status, missing fields, and derived-field status.

Inputs:

- explicit `--project-tickers` path;
- explicit `--ticker-cik-source` path;
- explicit `--companyfacts-dir` path;
- explicit source freshness date;
- explicit extraction date.

Outputs:

- review DataFrame;
- CSV only when an explicit output path is provided;
- CLI requires `--output` unless `--validate-only` is supplied.

CSV behavior:

- reads explicit local inputs;
- writes explicit output only;
- must not write to operational paths by default.

Network behavior:

- no live SEC call; it uses local files.

Operational boundary:

- isolated from Decision Engine, Reporting, Telegram, scanner, validation, context, timing, portfolio, and portfolio intelligence.

Known gap for SEC-7F:

- review output currently behaves as a row emitter with review columns, but it does not yet expose a structured run-level summary model;
- review-mode selection and row-level versus run-level evidence separation should be implemented or coordinated here.

Audit conclusion:

- this is a primary SEC-7F implementation file;
- no pre-SEC-7F refactor required, but SEC-7F should update it with strict fixture/temp-path tests.

### 4.5 `scripts/fundamentals/build_history_intake.py`

Purpose:

- defines or enforces the fundamentals-history intake schema;
- exposes required columns consumed by the SEC review runner.

Inputs:

- fundamentals history/intake records or configured source files depending on invocation.

Outputs:

- governed raw fundamentals-history-compatible outputs when invoked.

CSV behavior:

- may read/write fundamentals history data in the broader fundamentals pipeline;
- SEC-7F should not use operational outputs from this module as source-of-truth input.

Audit conclusion:

- no pre-SEC-7F change required;
- SEC-7F may import schema constants but should not modify runtime ingestion behavior.

### 4.6 `scripts/fundamentals/build_metrics.py`

Purpose:

- builds derived metrics from fundamentals history.

Inputs:

- governed fundamentals history input.

Outputs:

- metrics output for later fundamentals quality/analysis stages.

SEC-7F relevance:

- downstream of SEC review;
- not part of SEC-7F.

Audit conclusion:

- do not modify in SEC-7F;
- pipeline integration remains blocked before this stage receives SEC-derived operational data.

### 4.7 `scripts/fundamentals/build_quality.py`

Purpose:

- classifies fundamentals quality/readiness.

Inputs:

- metrics/history-derived fundamentals outputs.

Outputs:

- fundamentals quality classification output.

SEC-7F relevance:

- downstream operational layer;
- must not consume SEC review output until a separate integration sprint approves it.

Audit conclusion:

- do not modify in SEC-7F.

### 4.8 `scripts/fundamentals/build_analysis.py`

Purpose:

- builds fundamentals analysis output after history, metrics, and quality are available.

Inputs:

- governed fundamentals metrics/quality outputs.

Outputs:

- analysis output for later layers.

SEC-7F relevance:

- downstream and out of scope;
- SEC review output remains source-data readiness evidence, not investment analysis.

Audit conclusion:

- do not modify in SEC-7F.

### 4.9 Relevant compatibility wrappers under `scripts/core/`

Purpose:

- preserve legacy/runtime compatibility for fundamentals pipeline entry points.

SEC-7F relevance:

- out of scope;
- must not be used to introduce SEC pipeline integration.

Audit conclusion:

- do not modify in SEC-7F.

## 5. CSV/Data Inventory

Repository search found committed CSVs under these path families:

- `data/logs/`
- `data/processed/`
- `data/watchlist/`
- `data/portfolio/`
- `data/intake/`

No committed real SEC Company Facts cache files were identified by repository search. Searches for `companyfacts`, `data/local`, and `sec_edgar` returned code/docs references rather than committed local SEC payloads.

### 5.1 `data/logs/context_layer_log.csv`

Classification: generated runtime output / historical log.

Producer/consumer:

- produced by context-layer runtime logging;
- not an SEC-7F input.

Recommendation:

- do not touch in SEC-7F;
- evaluate in a separate generated-artifact hygiene or cleanup sprint.

Risk if removed too early:

- may break tests, historical review, or developer expectations.

### 5.2 `data/processed/market_regime.csv`

Classification: active governed output / generated runtime output.

Producer/consumer:

- produced by market-regime/runtime processing;
- consumed by scanner/context pipeline components.

Recommendation:

- do not touch in SEC-7F;
- not a SEC source-of-truth input.

Risk if removed too early:

- may break operational pipeline runs or tests.

### 5.3 `data/processed/scanner_ranked.csv`

Classification: active governed output / generated runtime output.

Producer/consumer:

- produced by scanner;
- consumed by validation/context flow.

Recommendation:

- do not touch in SEC-7F;
- not a SEC source-of-truth input.

Risk if removed too early:

- may break end-to-end pipeline runs or validation tests.

### 5.4 `data/processed/context_strength.csv`

Classification: active governed output / generated runtime output.

Producer/consumer:

- produced by context layer;
- consumed by downstream technical/fundamental/timing layers depending on current pipeline wiring.

Recommendation:

- do not touch in SEC-7F.

Risk if removed too early:

- may break downstream pipeline tests or runtime assumptions.

### 5.5 `data/watchlist/watchlist_status.csv`, `data/watchlist/watchlist_active.csv`, `data/watchlist/watchlist_transactions.csv`

Classification: active governed output or historical/watchlist state; cleanup candidate only after owner review.

Producer/consumer:

- produced/consumed by watchlist or historical watchlist workflows;
- not part of SEC-7F.

Recommendation:

- do not touch in SEC-7F;
- evaluate in a separate watchlist artifact cleanup sprint if desired.

Risk if removed too early:

- may break watchlist persistence or historical review assumptions.

### 5.6 `data/portfolio/portfolio_review.csv`, `portfolio_metadata.csv`, `portfolio_positions.csv`, `portfolio_transactions.csv`

Classification: active governed input/state and output depending on file.

Producer/consumer:

- portfolio runtime and portfolio intelligence workflow;
- manual portfolio state may exist in this family.

Recommendation:

- SEC work must not touch these files;
- do not cleanup as part of SEC audit or SEC-7F.

Risk if removed too early:

- may destroy portfolio state or break portfolio intelligence.

### 5.7 `data/portfolio/portfolio_positions_backup.csv`, `portfolio_transactions_backup.csv`

Classification: backup / historical candidate / needs owner decision.

Producer/consumer:

- likely manual or prior workflow backups;
- not SEC-related.

Recommendation:

- do not remove now;
- evaluate in a separate data cleanup sprint with explicit owner approval.

Risk if removed too early:

- may remove recovery information for portfolio state.

### 5.8 `data/intake/os5_scanner_ab_*_template.csv`

Classification: governed intake template.

Producer/consumer:

- source-data/intake workflow template;
- may be used for manual or governed ingestion.

Recommendation:

- keep committed;
- do not touch in SEC-7F.

Risk if removed too early:

- may break source-data operating workflow or documentation examples.

### 5.9 `data/intake/os5_scanner_ab_*_pilot.csv`

Classification: pilot intake data / historical or active governed input candidate / needs owner decision.

Producer/consumer:

- source-data/intake pilot workflow;
- not SEC-7F input.

Recommendation:

- do not remove now;
- classify in separate intake-data rationalization sprint if desired.

Risk if removed too early:

- may remove pilot evidence needed for backlog/source-data continuity.

### 5.10 `data/local/sec_edgar/` and Company Facts cache files

Classification: local-only ignored SEC data.

Audit finding:

- no committed real SEC cache files were identified by repository search;
- documentation and code reference local SEC paths, but committed search results did not show real Company Facts payloads.

Recommendation:

- keep real SEC cache local and ignored;
- do not commit raw SEC JSON, real SEC payload excerpts, or generated review CSVs.

## 6. SEC-Specific CSV/Data Concerns

| Concern | Audit finding | Recommendation |
|---|---|---|
| Real SEC cache committed accidentally | Not identified by repository search. | Continue keeping `data/local/sec_edgar/` local-only. |
| Generated SEC review outputs committed accidentally | Not identified in current CSV search results. | Keep generated SEC review outputs local/temp only. |
| Local diagnostic outputs committed accidentally | Not identified as SEC review CSVs. | Continue aggregate-only documentation policy. |
| Stale SEC review CSVs | None identified in committed CSV search results. | No SEC CSV cleanup needed before SEC-7F. |
| Valid fixture CSVs | Fixture-specific inventory should remain test-owned; no cleanup in this audit. | Keep fixtures unless a test-fixture rationalization sprint is approved. |
| Operational CSVs SEC-7F must not touch | `data/processed`, `data/portfolio`, `data/watchlist`, `data/logs`. | Exclude from SEC-7F. |
| Portfolio CSVs SEC work must not touch | `data/portfolio/*`. | Exclude from SEC-7F. |
| Reporting CSVs SEC work must not touch | `reports/` and reporting outputs. | Exclude from SEC-7F. |
| Processed CSVs as source-of-truth inputs | Existing `data/processed/*` files are pipeline artifacts, not SEC-7F source inputs. | SEC-7F must use fixtures/temp paths/local explicit inputs only. |

## 7. Code/Data Dependency Matrix

| Script | Inputs | Outputs | Generated output status | Fixture dependency | Operational pipeline dependency | Risk | Recommendation |
|---|---|---|---|---|---|---|---|
| `sec_companyfacts_bulk_intake.py` | SEC URL, User-Agent, local cache path | local cache/manifest when invoked | local-only | use mocked/temp files | none for SEC-7F | medium if live calls used | Do not call in SEC-7F tests. |
| `sec_ticker_cik_index.py` | local ticker/CIK source, project tickers | coverage data | explicit/local | fixture CSVs | none | low | Safe with fixture inputs. |
| `sec_companyfacts_transform.py` | local Company Facts JSON | transformed rows | no default operational write | JSON fixtures | none | medium | Primary SEC-7F target. |
| `run_sec_transformation_review.py` | explicit project tickers, ticker/CIK source, companyfacts dir | review CSV only with explicit output | explicit/local | fixture/temp paths | none | medium | Primary SEC-7F target. |
| `build_history_intake.py` | fundamentals history/intake | governed history output | pipeline artifact | fixture/unit tests | yes | medium | Do not modify in SEC-7F except schema import awareness. |
| `build_metrics.py` | fundamentals history | metrics output | pipeline artifact | tests | yes | medium | Do not modify in SEC-7F. |
| `build_quality.py` | metrics/history | quality output | pipeline artifact | tests | yes | medium | Do not modify in SEC-7F. |
| `build_analysis.py` | metrics/quality/history | analysis output | pipeline artifact | tests | yes | medium | Do not modify in SEC-7F. |
| `scripts/core/*` wrappers | pipeline paths | runtime outputs | pipeline artifact | tests | yes | high | Do not modify in SEC-7F. |

## 8. Staleness and Cleanup Assessment

Are there CSV files that are likely outdated?

- Yes, some committed CSVs are likely historical, generated, backup, or pilot artifacts, especially backup portfolio CSVs, logs, and old intake pilot files.
- This audit does not verify semantic freshness of file contents because no data files were opened, modified, regenerated, or normalized.

Are there CSV files that can be removed now?

- No. Do not remove any CSV in this audit.

Are there CSV files that should only be removed after a cleanup sprint?

- Yes. Backup CSVs, logs, generated runtime outputs, and pilot files should only be removed or archived after a separate approved cleanup sprint.

Are there CSV files that should be moved to archive?

- Possibly. `portfolio_*_backup.csv`, old logs, and pilot intake files are archive candidates, but require owner decision and dependency checks.

Are there CSV files that should stay because tests or governance depend on them?

- Yes. Intake templates, portfolio state, processed pipeline artifacts, and test fixtures should be preserved until dependency checks prove otherwise.

Are there generated CSV files that should be ignored rather than committed?

- Possibly. Runtime outputs under `data/logs/` and some `data/processed/` artifacts look generated. A separate Generated Artifact Hygiene sprint should review `.gitignore` and committed artifact policy.

Are code changes needed to stop relying on stale CSVs?

- No immediate pre-SEC-7F hard blocker was identified.
- Future cleanup may require code changes if current scripts/tests depend on committed generated files.

Are `.gitignore` changes needed later?

- Possibly. A later cleanup sprint should compare `.gitignore` against actual generated runtime path families and decide which generated artifacts should be ignored versus kept as governed fixtures/contracts.

## 9. SEC-7F Readiness Conclusion

Is the SEC code ready for SEC-7F implementation?

```text
Yes, with strict boundaries.
```

The SEC transformer and controlled review runner are the correct isolated surfaces for SEC-7F. They already use explicit local inputs and explicit output paths. SEC-7F can proceed if it remains focused on structured review-model semantics, review modes, and temp/fixture-based tests.

Is the data/CSV state ready for SEC-7F implementation?

```text
Yes, if SEC-7F avoids repository CSV reliance.
```

The committed CSV state should not be treated as SEC source-of-truth. SEC-7F should use fixture JSON/CSV inputs and temp output paths only.

What must be fixed before SEC-7F?

- Nothing identified as a hard blocker.
- SEC-7F must respect the audit conclusion that committed CSVs are not implementation inputs.

What can be handled during SEC-7F?

- structured review model;
- accepted/rejected fact separation;
- row/period/ticker/run evidence separation;
- explicit review modes;
- deterministic recent annual period selection;
- local run-summary structures or output only behind explicit paths.

What should be deferred until after SEC-7F?

- generated artifact cleanup;
- CSV archive/delete decisions;
- `.gitignore` rationalization;
- pipeline integration;
- full SEC re-run readiness assessment.

Does pipeline integration remain blocked?

```text
Yes.
```

Pipeline integration remains blocked until a controlled SEC-7F implementation and a later controlled review re-run demonstrate stable semantics and acceptable review-required patterns.

## 10. Role-Based Analysis

### Product Owner

The audit reduces product risk before SEC-7F by preventing implementation work from being built around stale generated CSVs or ambiguous review output. The business value is cleaner future SEC fundamentals ingestion, not immediate pipeline activation.

### PM / Scrum Master

Recommended sequencing:

1. Merge this audit.
2. Proceed to SEC-7F with strict boundaries.
3. After SEC-7F, run a controlled review re-run/readiness sprint.
4. Separately schedule cleanup/hygiene only if owner-approved.

This audit should not mix cleanup with SEC-7F implementation.

### Functional Analyst

Operator risks:

- committed CSVs may appear authoritative even when they are generated runtime artifacts;
- SEC review output may look like investment analysis even though it is source-data readiness evidence;
- review-required rows may be misread as business weakness;
- broad all-period SEC review output may confuse operators without explicit review modes.

Operator guidance:

- SEC-7F outputs remain local review material;
- `RECENT_ANNUAL_REVIEW` is an explicit review contract;
- `ALL_PERIODS_REVIEW` remains diagnostic only.

### Data Steward

Data policy:

- real SEC JSON stays local-only;
- generated SEC review CSVs stay local/temp-only;
- committed operational CSVs should not be used by SEC-7F;
- generated and stale CSV cleanup requires a separate sprint;
- no CSV should be deleted without producer/consumer dependency review.

### Technical Analyst / Architect

Architecture conclusion:

- SEC local review modules remain isolated;
- the core gap is review-model semantics, not code/data cleanup;
- output-path discipline is mostly sound in SEC review runner;
- SEC-7F should not touch downstream fundamentals analysis, core pipeline orchestration, Reporting, Telegram, or Decision Engine.

### Developer

Implementation implications:

- target `sec_companyfacts_transform.py` and `run_sec_transformation_review.py` only;
- use fixtures and temp paths;
- do not use committed runtime CSVs as tests or inputs;
- do not generate real SEC outputs;
- no live SEC/network calls;
- no downstream runtime changes.

### Governance Auditor

Governance checks:

- no forbidden investment semantics found in the intended SEC review path;
- no accidental pipeline integration identified in SEC audit scope;
- no committed real SEC Company Facts payloads identified by search;
- Decision Engine / Reporting / Telegram remain isolated;
- pipeline integration remains blocked.

## 11. Backlog Impact Assessment

Backlog impact assessment:

```text
No backlog modification performed.
```

BL-0015 and BL-0017 still cover SEC source-data, quality classification, and governed ingestion strategy work.

Potential future backlog draft if the owner wants cleanup:

```text
BL-XXXX — Generated Artifact and CSV Hygiene Sprint

Audit and rationalize committed CSV/data artifacts across data/processed, data/logs, data/intake, data/watchlist, data/portfolio, tests, and reports. Classify each file as governed input, governed output, generated runtime artifact, test fixture, archive candidate, or delete candidate. Update documentation and .gitignore only after producer/consumer dependency review. Do not modify runtime behavior unless separately approved.
```

This backlog item is optional. It is not required before SEC-7F.

## 12. Recommended Next Steps

Recommended path:

```text
Path A — Proceed to SEC-7F implementation with strict boundaries.
```

Rationale:

- no hard blocker was identified;
- SEC-7F can operate with fixtures and temp paths;
- committed CSV cleanup is useful but separable;
- delaying SEC-7F for broad cleanup would mix concerns;
- pipeline integration remains blocked either way.

Recommended future SEC-7F title:

```text
SEC-7F — SEC Review Model and Period Selection Implementation
```

Recommended sprint after SEC-7F:

```text
SEC-7G — Controlled SEC Review Re-Run and Integration Readiness Assessment
```

## 13. Validation

Requested validation for this audit:

```bash
git diff --check
git status
```

Validation status:

- Not run from a local checkout because this task was executed through the GitHub connector.
- Repository changes were limited to a single Markdown audit file.
- No tests were run because no code or tests were changed.
- No SEC diagnostics were run.
- No live SEC calls were performed.
- No generated outputs were created.

## 14. Boundary Confirmation

This audit changed:

- `docs/audits/sec_code_data_alignment_audit_before_sec_7f.md`

This audit did not change:

- runtime code;
- tests;
- CSV files;
- data files;
- generated files;
- reports;
- workflows;
- backlog;
- Decision Engine;
- Reporting;
- Telegram;
- scanner;
- validation;
- context;
- timing;
- portfolio intelligence;
- runtime behavior;
- pipeline integration.
