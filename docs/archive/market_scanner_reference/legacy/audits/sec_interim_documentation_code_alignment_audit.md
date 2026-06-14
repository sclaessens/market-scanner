# SEC Interim Documentation and Code Alignment Audit

Status: DOCUMENTATION-ONLY AUDIT
Date: 2026-05-31
Repository: sclaessens/market-scanner
Branch: docs/sec-interim-documentation-code-alignment-audit

## 1. Executive Summary

This audit reviewed the current SEC documentation, backlog direction, sprint sequence, and relevant fundamentals code after the SEC-2 through SEC-7A work stream.

Conclusion:

- The current SEC documentation direction is broadly aligned with the implemented code.
- The implemented SEC flow is coherent and properly staged: source strategy, local Company Facts intake/cache, ticker/CIK coverage, XBRL mapping, analysis rationalization, direct transformation, derived formula support, controlled local review, and controlled review input preparation.
- The current code remains standalone and does not integrate SEC outputs into the active operational pipeline.
- The current transformer is intentionally conservative, but the latest local SEC-7B diagnostic shows it is too strict for messy real SEC facts.
- The next sprint should be `SEC-7C — Real SEC Fact Tolerance and Period Selection`.
- No backlog modification is made by this audit. BL-0015 and BL-0017 still cover the broad data-source and ingestion strategy work, but this audit recommends documenting SEC-7C as a narrow sprint under the existing backlog drivers rather than adding a new item now.

Key finding:

The local SEC-7B diagnostic should not be interpreted as a failure of SEC data. It shows that the current transformation logic can access mapped CIKs and valid local Company Facts JSON files, but it fails whole-ticker transformation when individual facts have missing fiscal-year metadata or conflicting same-tag/unit/period values. The next implementation step should make the transformer tolerant at the fact or period level, preserving transformable periods and isolating problematic facts as review-required.

## 2. Repository State Reviewed

Reviewed documentation and governance surfaces:

- `AGENTS.md`
- `README.md`
- `docs/active/repository_structure.md`
- `docs/active/backlog_and_sprint_operating_model.md`
- `docs/active/roles_and_responsibilities.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/sprint_status_tracker.md` when present
- `docs/active/source_data/sec_edgar_fundamentals_source_strategy.md`
- `docs/active/source_data/sec_edgar_source_architecture.md`
- `docs/active/source_data/sec_xbrl_mapping_investigation.md`
- `docs/active/source_data/sec_fundamental_analysis_rationalization.md`
- `docs/active/source_data/sec_derived_formula_policy.md`
- `docs/sprints/sec_2_sec_bulk_intake_implementation.md`
- `docs/sprints/sec_3_sec_ticker_cik_coverage.md`
- `docs/sprints/sec_6a_direct_sec_fundamentals_transform.md`
- `docs/sprints/sec_6c_derived_formula_support.md`
- `docs/sprints/sec_6d_controlled_local_transformation_review.md`
- `docs/sprints/sec_7a_controlled_review_input_preparation.md`
- `docs/active/contracts/fundamentals_platform_contract.md`
- `docs/active/contracts/fundamental_calculations_technical_spec.md`

Reviewed code and test surfaces without modifying them:

- `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
- `scripts/fundamentals/sec_ticker_cik_index.py`
- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- `scripts/fundamentals/build_history_intake.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/fundamentals/build_analysis.py`
- relevant compatibility wrappers under `scripts/core/`
- relevant tests under `tests/fundamentals/`

Repository governance anchors observed:

- `AGENTS.md` defines `classification upstream` and `allocation downstream`.
- `AGENTS.md` states that the Decision Engine is the only allocation authority.
- `AGENTS.md` defines the English-only repository standard.
- `AGENTS.md` allows ChatGPT to perform documentation-only governance work directly in GitHub while prohibiting runtime code, test, generated data, CSV, report, workflow, and behavior changes.
- `docs/sprints/project_backlog.md` remains the operational deferred-work source of truth and does not itself authorize implementation.

## 3. Role-Based Findings

### 3.1 Product Owner

Product direction remains coherent.

The current SEC path supports a broad SEC data strategy rather than manual five-ticker entry. The review ticker set should remain a controlled diagnostic sample, not a product-definition limit.

Controlled review before pipeline integration remains the correct product sequence. The project should not feed real SEC Company Facts output into the operational Fundamental Layer until real-data tolerance and evidence requirements are clearer.

Real fundamentals analysis should remain blocked from investment interpretation until SEC fact quality issues are understood. The current review outputs are source-data readiness evidence, not investment analysis.

Product Owner decision needed next:

- approve SEC-7C as the next narrow implementation sprint;
- keep pipeline integration blocked;
- accept that review-required rows are useful evidence rather than a failure state.

### 3.2 PM / Scrum Master

Sprint sequencing remains logical, but the next sprint must be tightly bounded.

SEC-7C is the right next sprint because the blocker is implementation tolerance against real SEC facts, not another broad documentation sprint. The necessary documentation correction is captured in this audit.

Recommended sprint shape:

- one primary theme;
- no pipeline integration;
- no generated real data committed;
- no broad data-source automation;
- fixture-based tests only;
- no Decision Engine or Reporting changes.

The sprint should not mix tolerance changes with output summarization policy, full SEC dataset processing, pipeline wiring, or investment-analysis interpretation.

### 3.3 Functional Analyst

The operator workflow is mostly understandable:

1. download or validate SEC Company Facts locally;
2. prepare or verify ticker/CIK source input;
3. prepare a project ticker input file;
4. run the controlled review against explicit local files;
5. interpret output statuses as source-data readiness, not investment decisions.

Potential confusion remains around real SEC diagnostics:

- `TRANSFORM_REVIEW_REQUIRED` can look like a complete data failure, even when local JSON exists and mapping works.
- Missing fiscal-year facts and conflicting facts currently surface as whole-ticker transform failures.
- Operators may not know which periods were transformable if older or irrelevant periods fail first.

Functional clarification needed in SEC-7C:

- explain fact-level, period-level, and ticker-level review-required handling;
- distinguish no data, untransformable data, partially transformable data, and fully transformed data;
- define whether a recent-year filter is optional, required, or deferred.

### 3.4 Data Steward

Data governance remains sound.

Current local SEC cache direction is appropriate:

- Company Facts bulk cache artifacts remain local under `data/local/sec_edgar/companyfacts/`.
- `data/local/` is ignored and should remain uncommitted.
- Review outputs generated from real SEC data should remain local unless a future governance decision explicitly permits summarized, sanitized documentation.

Current risks:

- real SEC Company Facts payloads may include historical, amended, duplicated, incomplete, or conflicting facts;
- not every SEC fact has the same metadata completeness;
- source evidence must be preserved for both transformed and review-required facts;
- skipping or isolating facts must not remove auditability.

Data Steward requirement for SEC-7C:

- skipped facts, isolated conflicts, and review-required periods need evidence notes;
- transformable periods must preserve source references, extraction date, source freshness date, CIK, tag, unit, period, accession when available, and reason for any omission or review state.

### 3.5 Financial Analyst

The current fundamental fields are financially reasonable as a baseline SEC-supported model:

- `revenue`
- `gross_profit`
- `operating_income`
- `net_income`
- `diluted_eps`
- `total_debt`
- `total_equity`
- `free_cash_flow`

The direct-field approach is financially appropriate for revenue, operating income, net income, total equity, and conditional gross profit when source evidence is present.

`diluted_eps` remains useful but requires careful unit and period handling because it is a per-share value and can be reported across different period contexts.

`total_debt` derivation is financially reasonable only under conservative component rules. The current logic correctly avoids mixing simple debt, lease-inclusive debt, short-term borrowings, and finance lease components without evidence.

`free_cash_flow` derivation from operating cash flow and capital expenditures is financially reasonable, but only when capex sign conventions and source tags are clear.

Financial review still needed before investment analysis:

- distinguish annual facts from quarterly facts;
- ensure recent periods are selected consistently;
- avoid using older conflicting facts to block current-year analysis unnecessarily;
- treat missing optional fields as missing data, not business weakness;
- keep derived formulas null or review-required when source evidence is insufficient.

### 3.6 Technical Analyst / Architect

Architecture remains coherent.

Module boundaries are clean:

- `sec_companyfacts_bulk_intake.py` handles SEC Company Facts bulk ZIP source validation, local cache, ZIP validation, manifest generation, and optional explicit download.
- `sec_ticker_cik_index.py` handles local ticker/CIK source reading, CIK normalization, mapping status, and row-preserving coverage.
- `sec_companyfacts_transform.py` handles local Company Facts payload transformation into internal raw fundamentals history rows.
- `run_sec_transformation_review.py` composes ticker/CIK coverage and local Company Facts transformation into controlled review output.
- downstream fundamentals metrics, quality, and analysis builders remain separate.

CLI design is explicit and safe:

- input paths are caller-provided;
- output paths are explicit;
- validate-only paths exist where relevant;
- no module should write operational outputs by default.

The main technical gap is exception granularity. Current transformation raises exceptions for missing fiscal-year metadata, unsupported fiscal periods, unit conflicts, and conflicting facts. The review runner catches transformer exceptions at ticker level and emits a ticker-level `TRANSFORM_REVIEW_REQUIRED` row. That is safe, but too coarse for real SEC data.

SEC-7C should move from whole-ticker failure to controlled fact/period tolerance.

### 3.7 Governance Auditor

Governance compliance is preserved.

No reviewed SEC module creates allocation authority, tradeability, urgency, conviction, ranking, scoring, eligibility, buy/sell, final action, or hidden filtering.

The Decision Engine remains the sole allocation authority.

Reporting remains communication-only and is not modified by the SEC flow.

Generated SEC data remains uncommitted by policy.

The reviewed code and sprint documentation consistently state no live SEC calls in tests, no generated operational output commits, no pipeline integration, and no downstream Decision Engine or Reporting changes.

English-only repository content remains required and this audit document is written in English.

### 3.8 Developer / Codex Handoff

Recommended next sprint:

```text
SEC-7C — Real SEC Fact Tolerance and Period Selection
```

Allowed files:

- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/run_sec_transformation_review.py` only if review status/output columns require narrow adjustment
- `tests/fundamentals/test_sec_companyfacts_transform.py`
- `tests/fundamentals/test_run_sec_transformation_review.py`
- `docs/sprints/sec_7c_real_sec_fact_tolerance_and_period_selection.md`
- optionally this audit document only if the sprint records a follow-up correction

Forbidden files:

- `scripts/core/decision_engine.py`
- `scripts/reporting/`
- `scripts/telegram/`
- `scripts/run_full_pipeline.py`
- `scripts/run_scan.py`
- `data/`
- `reports/`
- `.github/workflows/`
- portfolio runtime files
- scanner, validation, context, timing, and portfolio intelligence runtime behavior

Implementation boundaries:

- no live SEC calls in tests;
- fixture-based tests only;
- no real SEC data committed;
- no generated output committed;
- no pipeline integration;
- no Decision Engine changes;
- no Reporting changes;
- no Telegram changes;
- no portfolio behavior changes;
- no investment-analysis interpretation.

Required tests:

- missing fiscal year fact is isolated or skipped with review evidence and does not fail the whole ticker;
- conflicting same-tag/unit/period facts are isolated or marked review-required without failing transformable periods;
- older-period conflicts do not block recent clean periods when period filtering or isolation is active;
- transformable periods remain represented;
- review-required evidence is preserved;
- optional recent-year filtering is deterministic if implemented;
- no live SEC/network call on import or tests;
- no output writes unless explicit temporary output path is supplied;
- review-only columns remain free of allocation/trade/action semantics.

Validation commands for Codex/local execution:

```bash
git diff --check
.venv/bin/python -m pytest tests/fundamentals/test_sec_companyfacts_transform.py
.venv/bin/python -m pytest tests/fundamentals/test_run_sec_transformation_review.py
```

Optional broader validation, if no runtime boundaries are crossed:

```bash
.venv/bin/python -m pytest tests/fundamentals
```

Expected PR body points:

- implements SEC-7C real SEC fact tolerance and period selection;
- no live SEC calls;
- no real SEC data committed;
- no generated data committed;
- no pipeline integration;
- no Decision Engine changes;
- no Reporting changes;
- no Telegram changes;
- tests use fixtures/temp paths;
- transformable periods are preserved while problematic facts are review-required.

## 4. Documentation / Code Alignment Matrix

| Area | Documentation expectation | Code observation | Alignment | Notes |
|---|---|---|---|---|
| SEC source strategy | SEC Company Facts is the preferred initial broad source direction. | SEC modules are built around Company Facts bulk ZIP and Company Facts JSON payloads. | Aligned | The five review tickers remain diagnostic samples, not the full strategy. |
| SEC-2 bulk intake/cache | Local cache only, official SEC URL validation, User-Agent requirement, manifest support, no pipeline integration. | `sec_companyfacts_bulk_intake.py` defines the official ZIP URL, default local cache family, URL allowlisting, User-Agent validation, ZIP validation, and manifest generation. | Aligned | Download exists only when explicitly invoked. |
| SEC-3 ticker/CIK coverage | Local ticker/CIK source reading, CIK normalization, row-preserving coverage, descriptive statuses. | `sec_ticker_cik_index.py` normalizes CIKs, reads local source shapes, and preserves missing or ambiguous rows. | Aligned | Mapping statuses remain descriptive. |
| SEC XBRL mapping investigation | Map candidate tags and identify derived/review-required fields before transformation. | Transformer candidate maps follow the documented direct and derived tag families. | Aligned | Real SEC messiness now requires tolerance policy. |
| SEC analysis rationalization | Direct fields first; derived debt and free cash flow require conservative rules. | Transformer maps direct fields and derives only approved `total_debt` and `free_cash_flow` component paths. | Aligned | Conservative behavior is appropriate, but currently too coarse on exceptions. |
| SEC-6A direct transformation | Local JSON input, source evidence, no pipeline integration, no generated output by default. | `transform_companyfacts_file` accepts explicit local JSON and explicit context; output path is optional. | Aligned | Missing optional fields are preserved in notes. |
| SEC-6C derived formulas | `total_debt` and `free_cash_flow` derived only from approved clean components. | Debt and free cash flow derivation functions enforce conservative component selection and no inferred missing values. | Aligned | Missing or conflicting components should become review evidence rather than whole-ticker failure where possible. |
| SEC-6D controlled review | Compose mapping plus local transformation with explicit inputs and output path. | Review runner requires project tickers, ticker/CIK source, Company Facts directory, dates, and explicit output unless validate-only. | Aligned | Ticker-level exception handling is safe but not granular enough. |
| SEC-7A review preparation | Controlled real-data review should remain local and uncommitted. | Code supports explicit local review inputs and local outputs. | Aligned | SEC-7B diagnostic confirms local workflow works but exposes tolerance gap. |
| Backlog | BL-0015 and BL-0017 govern source-data/fundamentals and future ingestion. | Backlog still lists BL-0015 as approved for planning and BL-0017 as blocked until schema/source policy are proven. | Mostly aligned | New SEC-7C can be handled under BL-0015/BL-0017 without immediate backlog edit. |

## 5. Current SEC Flow Validation

The current SEC flow is logically coherent:

```text
SEC source strategy
-> SEC Company Facts local intake/cache
-> ticker/CIK indexing and coverage
-> XBRL mapping investigation
-> fundamental analysis rationalization
-> direct SEC-supported transformation
-> derived formula support
-> controlled local transformation review
-> controlled review input preparation
-> real SEC fact tolerance and period selection
```

Boundary validation:

- SEC Company Facts intake/cache does not transform fundamentals.
- Ticker/CIK coverage does not read Company Facts or decide eligibility.
- XBRL mapping remains analysis/documentation until consumed by transformer rules.
- Direct transformation does not integrate with the pipeline.
- Derived formula support remains local/fixture scoped and conservative.
- Controlled review composes existing utilities but still does not write to operational processed artifacts by default.
- SEC-7A/SEC-7B review stays local under `data/local/`.
- No reviewed step creates Decision Engine authority.

The flow should not advance to pipeline integration until SEC-7C resolves real fact tolerance and period-selection behavior.

## 6. SEC-7B Local Review Diagnostic Interpretation

Latest local SEC-7B diagnostic finding:

- ticker/CIK mapping worked for the five review tickers;
- local Company Facts JSON files were downloaded and valid;
- the review runner produced five rows;
- all five rows were `TRANSFORM_REVIEW_REQUIRED`;
- observed issues included missing fiscal year facts and conflicting SEC facts;
- generated output stayed local under `data/local/`;
- repository working tree remained clean.

Interpretation:

This is not a complete failure of SEC data.

It means the controlled local review path is operational enough to map tickers, find local Company Facts files, parse valid JSON, and produce review output. The failure mode is the transformer's strict handling of messy real facts.

Current transformer behavior is safe but too strict:

- missing `fy` currently raises an exception;
- conflicting facts currently raise an exception;
- the review runner catches the exception and marks the entire ticker as `TRANSFORM_REVIEW_REQUIRED`;
- transformable periods may be lost when unrelated older or problematic facts fail first.

Required direction:

- isolate bad facts;
- isolate conflicting periods;
- preserve clean transformable periods;
- mark problematic facts or periods as review-required;
- keep source evidence for all skipped or review-required facts;
- avoid silently choosing winners without policy.

## 7. Identified Gaps and Risks

### 7.1 Real SEC Fact Tolerance

The transformer needs controlled tolerance for:

- facts missing fiscal year metadata;
- unsupported or missing fiscal periods;
- missing period end dates;
- conflicting facts for the same tag/unit/period;
- conflicting units;
- old-period conflicts that should not block recent clean periods;
- duplicated same-value facts that should remain deterministic and evidenced.

### 7.2 Period Selection

The project needs a policy for:

- recent-year selection;
- annual-only versus mixed annual/quarterly handling;
- transformable period preservation;
- whether optional period filtering is exposed through CLI;
- whether default behavior should transform all safe periods or a controlled recent window.

### 7.3 Evidence Requirements

Skipped or review-required facts must preserve enough evidence to audit later:

- ticker;
- CIK;
- tag;
- unit;
- fiscal year when available;
- fiscal period when available;
- period end date when available;
- accession when available;
- filed date when available;
- reason for skip/review;
- whether the fact was excluded from transformation or retained as review-required.

### 7.4 Documentation-Safe Review Summaries

Real SEC review outputs should remain local. A future documentation policy may allow aggregate summaries, but only if they avoid committing raw SEC payloads, generated review CSVs, or overly detailed real-data extracts.

Potential safe summary fields:

- ticker count;
- row count;
- status distribution;
- generic reason categories;
- no raw fact values;
- no downloaded JSON content;
- no generated CSV content.

This audit recommends deferring documentation-safe review summary policy until after SEC-7C clarifies output semantics.

## 8. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

Rationale:

- BL-0015 still covers approved fundamentals source-data and quality classification contract work.
- BL-0017 still covers governed automated ingestion strategy, and remains blocked until local/manual architecture and validation are proven.
- SEC-7C is a narrow implementation-hardening sprint that can be governed under the existing SEC source-data sequence and BL-0015/BL-0017 context.
- A separate backlog item for documentation-safe review summaries may become justified later, but it is not necessary before SEC-7C.
- Updating `docs/sprints/project_backlog.md` now would risk expanding backlog scope before the real-data tolerance sprint defines the precise output semantics.

Recommended backlog note for a future sprint closeout, if SEC-7C confirms the need:

```text
Candidate future backlog item:
Documentation-safe SEC review summary policy.
Purpose: define whether and how generated real SEC review outputs may be summarized in docs without committing raw SEC data, generated CSVs, or source-sensitive extracts.
```

## 9. Recommended Sprint Sequence

Recommended sequence:

1. `SEC-7C — Real SEC Fact Tolerance and Period Selection`
2. `SEC-7D — Documentation-Safe Review Summary Policy` only if SEC-7C produces stable review-status semantics that require publication guidance
3. `SEC-8 — Controlled SEC Review Re-run` using local inputs only, no generated output committed unless explicitly approved
4. `SEC-9 — SEC-to-Fundamentals Pipeline Integration Specification` documentation-only, only after real-data tolerance is proven
5. later implementation sprint for controlled integration, if approved

Not recommended now:

- full SEC dataset automation;
- writing to `data/raw/fundamentals_history.csv`;
- feeding SEC outputs into `fundamental_metrics.csv`, `fundamental_quality.csv`, or `fundamental_analysis.csv`;
- Decision Engine or Reporting consumption;
- investment interpretation of SEC review outputs.

## 10. Recommended Next Sprint Specification

Sprint name:

```text
SEC-7C — Real SEC Fact Tolerance and Period Selection
```

Sprint purpose:

Make the local SEC Company Facts transformer robust enough for messy real SEC facts while preserving deterministic behavior, audit evidence, and governance boundaries.

In scope:

- isolate facts with missing fiscal-year metadata;
- isolate conflicting facts for the same tag/unit/period;
- preserve clean transformable periods;
- mark problematic facts or periods as review-required;
- add deterministic optional recent-year or period filtering if approved;
- preserve source evidence for skipped/review-required facts;
- keep all tests fixture-based and local.

Out of scope:

- live SEC calls in tests;
- new SEC downloads;
- generated real SEC output commits;
- pipeline integration;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- portfolio changes;
- scanner/validation/context/timing/portfolio-intelligence changes;
- investment analysis or recommendation logic.

Acceptance criteria:

- missing fiscal-year facts do not fail the whole ticker;
- conflicting facts do not fail the whole ticker;
- clean periods remain transformable;
- review-required facts or periods are explicit and evidenced;
- optional period filtering, if implemented, is deterministic and documented;
- generated output is still written only to explicit caller paths;
- no allocation/trade/action semantics are introduced.

## 11. No-Runtime-Change Confirmation

This audit changed documentation only.

Confirmed scope:

- no `scripts/` files changed;
- no `tests/` files changed;
- no `data/` files changed;
- no `reports/` files changed;
- no CSV files changed;
- no generated files changed;
- no `.github/workflows/` files changed;
- no Decision Engine logic changed;
- no Reporting semantics changed;
- no Telegram delivery or formatting changed;
- no portfolio behavior changed;
- no scanner behavior changed;
- no validation/context/timing/portfolio intelligence behavior changed;
- no fundamental metrics, quality, or analysis runtime behavior changed;
- no SEC/network call performed by this audit;
- no generated data created or committed;
- no runtime behavior changed.

Validation performed or simulated:

- Documentation-only file creation planned under `docs/audits/`.
- Changed-file scope verified by GitHub compare after file creation.
- `git diff --check` was not executed locally because this audit was performed through the GitHub connector rather than a local checkout. The audit document was manually reviewed for documentation-only scope and markdown/plain-text consistency.
- Runtime tests were not run because no runtime files were changed.

Expected PR title:

```text
docs: audit SEC documentation and code alignment
```

Expected PR body points:

- documentation-only audit;
- no code changes;
- no tests required or run;
- no SEC calls;
- no generated data committed;
- no runtime behavior changed;
- recommended next sprint: `SEC-7C — Real SEC Fact Tolerance and Period Selection`.
