# Fundamentals History Implementation Specification

Status: ACTIVE DEVELOPER SPECIFICATION

## 1. Purpose

This document is the developer-ready implementation specification for a future Sprint E of the Fundamentals Simplification Program.

This document does not authorize implementation by itself. Implementation requires explicit Sprint E approval.

Sprint D changes no code, tests, CSV files, generated outputs, raw data, provider integrations, scraping behavior, pipeline execution, or runtime behavior.

## 2. Architecture Baseline

The target fundamentals architecture is:

```text
raw fundamentals history
-> calculated metrics
-> fundamental quality classification
-> fundamental analysis classification
-> downstream portfolio/decision layers
```

Raw history stores source-supported statement facts only. It must not store quality states, analysis states, ratios, ranking, scoring, allocation fields, buy/sell fields, urgency, conviction, eligibility, tradeability, or hidden filtering semantics.

The metrics layer calculates deterministic values from raw history. It must follow `docs/active/contracts/fundamental_calculations_technical_spec.md` and must not interpret business quality or create allocation authority.

The quality layer classifies completeness, readiness, reliability, and review-required states. It remains data-readiness classification only.

The analysis layer classifies descriptive business and fundamental characteristics from approved metrics and quality states. It remains descriptive and has no allocation authority.

The Decision Engine remains the only allocation, execution, arbitration, and final-action authority.

## 3. Current-State Compatibility Problem

The current runtime is a metric-like MVP, not the target raw-history architecture:

- `data/raw/fundamentals.csv` stores selected metric-like values rather than raw multi-year statement history.
- `scripts/core/build_fundamental_layer.py` currently builds `data/processed/fundamental_quality.csv`.
- Downstream layers expect the current `fundamental_quality.csv` shape.
- No `data/processed/fundamental_metrics.csv` artifact exists.
- No `data/processed/fundamental_analysis.csv` artifact exists.
- Current `as_of_date` semantics are overloaded and mix period, report, source freshness, extraction, and opportunity-date matching concerns.
- Migration must avoid breaking Timing State, Portfolio Intelligence, Decision Engine, Reporting, or Telegram flows.

The immediate implementation risk is downstream schema breakage. Sprint E must preserve compatibility first and only then introduce cleaner internal layers.

## 4. Target Artifacts

### 4.1 Raw History Artifact

Target:

```text
data/raw/fundamentals_history.csv
```

Required columns:

| Column | Requirement |
|---|---|
| `ticker` | Normalized security ticker. |
| `fiscal_year` | Reported fiscal year. |
| `fiscal_period` | Approved period label, such as annual, quarterly, trailing, or another governed value. |
| `period_end_date` | Fiscal period end date. |
| `report_date` | Company report, filing, or publication date. |
| `currency` | Reporting currency. |
| `revenue` | Raw reported revenue. |
| `gross_profit` | Raw reported gross profit. |
| `operating_income` | Raw reported operating income. |
| `net_income` | Raw reported net income. |
| `diluted_eps` | Raw reported diluted earnings per share. |
| `total_debt` | Raw reported total debt. |
| `total_equity` | Raw reported total equity. |
| `free_cash_flow` | Raw reported or source-supported free cash flow. |
| `source_name` | Source name. |
| `source_reference` | Specific source reference, URL, filing name, or evidence pointer. |
| `source_freshness_date` | Date the source was checked. |
| `extraction_date` | Local extraction date. |
| `notes` | Data-steward notes, caveats, or review comments. |

Source-data policy:

- raw reported data only;
- source reference required for every row;
- currency required for every row;
- date semantics must remain split across period end, report, source freshness, and extraction fields;
- no inferred values;
- no raw ratios unless explicitly approved by a later governed contract;
- no quality, analysis, ranking, scoring, allocation, buy/sell, urgency, conviction, eligibility, or tradeability fields.

Local ignored policy:

- `data/raw/fundamentals_history.csv` should remain local ignored unless repository policy changes;
- tracked docs and tests may define schemas and fixtures, but live raw source data must not be committed without explicit approval;
- local raw data must be treated as source evidence, not active doctrine.

### 4.2 Metrics Artifact

Target:

```text
data/processed/fundamental_metrics.csv
```

Required output groups:

- single-period metrics;
- year-over-year metrics;
- multi-year metrics;
- data-quality helper metrics.

Formula authority:

- `docs/active/contracts/fundamental_calculations_technical_spec.md`

Expected metric families include:

- gross margin;
- operating margin;
- net margin;
- debt to equity;
- return on equity;
- free cash flow margin;
- revenue year-over-year growth;
- diluted EPS year-over-year growth;
- free cash flow year-over-year growth;
- 3-year CAGR metrics where mathematically valid;
- average and trend helper metrics where approved by the calculation spec.

Rules:

- calculations only;
- deterministic outputs for the same inputs;
- null or review-helper outputs for missing, invalid, zero-denominator, sign-change, period-mismatch, or currency-mismatch cases;
- no allocation, ranking, scoring, eligibility, tradeability, urgency, conviction, buy/sell, or hidden filtering semantics.

### 4.3 Quality Artifact

Target:

```text
data/processed/fundamental_quality.csv
```

Compatibility requirement:

- preserve downstream-required columns until downstream consumers are explicitly updated;
- preserve current source-missing and insufficient-data style compatibility where downstream tests or runtime consumers expect it;
- preserve row identity from the current context/opportunity universe;
- add new descriptive fields only if explicitly approved by a governed contract or Sprint E scope;
- remain data-readiness and classification-only.

Potential future quality states may include:

- `SOURCE_MISSING`
- `RAW_HISTORY_INCOMPLETE`
- `RAW_HISTORY_PARTIAL`
- `RAW_HISTORY_READY`
- `METRICS_PARTIAL`
- `METRICS_READY`
- `REVIEW_REQUIRED`

Any state-name migration must include a compatibility bridge or downstream contract update.

### 4.4 Analysis Artifact

Target:

```text
data/processed/fundamental_analysis.csv
```

Descriptive output groups:

- growth state;
- margin state;
- profitability state;
- debt state;
- cashflow state;
- consistency state;
- trend state;
- review-required state and reasons.

Rules:

- descriptive classification only;
- no allocation authority;
- no ranking authority;
- no scoring authority;
- no tradeability;
- no urgency;
- no conviction;
- no eligibility;
- no buy/sell semantics;
- no hidden filtering.

## 5. Proposed Future Scripts

### 5.1 `scripts/core/build_fundamentals_history_intake.py`

Purpose:

Normalize or validate raw-history intake.

Inputs:

- approved local intake or staging artifact;
- existing local `fundamentals.csv` only if a future migration spec explicitly approves it.

Outputs:

- `data/raw/fundamentals_history.csv` or a validation report, depending on local ignored policy.

Must not:

- calculate metrics;
- analyze business characteristics;
- allocate;
- rank;
- score;
- infer missing values.

Implementation notes for Sprint E:

- validate required columns;
- normalize tickers;
- parse date fields;
- validate currency presence;
- validate source reference presence;
- detect duplicate `ticker` + `fiscal_year` + `fiscal_period` rows;
- reject or report forbidden semantic fields.

### 5.2 `scripts/core/build_fundamental_metrics.py`

Purpose:

Calculate deterministic metrics from raw history.

Inputs:

- `data/raw/fundamentals_history.csv`

Outputs:

- `data/processed/fundamental_metrics.csv`

Formula authority:

- `docs/active/contracts/fundamental_calculations_technical_spec.md`

Must handle:

- missing inputs;
- zero denominators;
- negative denominators;
- sign changes;
- period inconsistency;
- currency inconsistency;
- duplicate ticker-period rows.

Must not:

- source new data;
- infer missing raw values;
- interpret business quality;
- allocate;
- rank;
- score;
- create eligibility, tradeability, urgency, conviction, buy/sell, or hidden filtering semantics.

### 5.3 `scripts/core/build_fundamental_quality.py`

Purpose:

Build data-readiness classification.

Inputs:

- raw history;
- metrics;
- helper fields.

Outputs:

- `data/processed/fundamental_quality.csv`

Compatibility:

- preserve the current downstream-required shape where needed;
- continue source-missing and insufficient-data style compatibility if downstream consumers expect it;
- add new quality states only under a governed contract;
- keep output descriptive and data-readiness focused.

Must not:

- classify business attractiveness;
- allocate;
- rank;
- score;
- create eligibility, tradeability, urgency, conviction, buy/sell, or hidden filtering semantics.

### 5.4 `scripts/core/build_fundamental_analysis.py`

Purpose:

Build descriptive financial analysis classifications.

Inputs:

- `data/processed/fundamental_metrics.csv`
- `data/processed/fundamental_quality.csv`

Outputs:

- `data/processed/fundamental_analysis.csv`

Must not:

- allocate;
- rank;
- score;
- create buy/sell;
- create urgency;
- create conviction;
- create eligibility;
- create tradeability.

## 6. Compatibility Strategy

This section is the central Sprint D decision point.

### Option A - Compatibility Wrapper First

Keep the existing `scripts/core/build_fundamental_layer.py` as the pipeline-facing builder for Sprint E.

Add new internal or helper builders for:

- raw history validation;
- metric calculation;
- quality-state mapping;
- analysis output.

The existing `data/processed/fundamental_quality.csv` remains compatible with current downstream consumers.

Pros:

- lower pipeline risk;
- preserves downstream tests;
- avoids immediate Timing State, Portfolio Intelligence, Decision Engine, Reporting, and Telegram changes;
- allows raw-history and metrics work to be tested before orchestration replacement;
- keeps the current operator pipeline stable during migration.

Cons:

- temporary duplication;
- old MVP artifact may remain during transition;
- `build_fundamental_layer.py` may remain overloaded until a later replacement sprint;
- compatibility mapping must be documented and tested.

### Option B - Full Replacement

Replace the current Fundamental Layer flow with:

```text
raw history
-> metrics
-> quality
-> analysis
```

Pros:

- cleaner architecture;
- less compatibility code after completion;
- more direct alignment with the target contracts.

Cons:

- higher downstream breakage risk;
- requires more tests before first merge;
- likely touches orchestration and downstream expectations;
- increases risk of accidental runtime behavior change;
- may blur Sprint E scope if not split carefully.

### Recommendation

Sprint E should use Option A unless there is a strong reviewed reason not to.

The first implementation should preserve `build_fundamental_layer.py` as the pipeline-facing compatibility surface, add focused helpers or builders behind it, and keep `data/processed/fundamental_quality.csv` compatible with current downstream consumers.

Full replacement should be deferred until raw history, metrics, quality compatibility, and analysis outputs are independently tested and accepted.

## 7. Migration Strategy from `data/raw/fundamentals.csv`

Current MVP data may be treated only as compatibility or migration evidence if explicitly approved.

Rules:

- current `data/raw/fundamentals.csv` may be read only for compatibility or migration if explicitly approved;
- metric-like fields must not be treated as raw history;
- `as_of_date` must not be blindly mapped to `period_end_date`, `report_date`, `source_freshness_date`, or `extraction_date`;
- missing period metadata must produce a review-required migration state;
- no values may be inferred;
- migration must be reversible or isolated;
- current MVP data must not silently become source truth for raw history;
- any migration output must remain documentation-approved and operator-reviewable.

Possible future migration outputs:

- no migration in first implementation;
- optional manual staging template;
- optional migration report only;
- optional local ignored draft `fundamentals_history.csv` generated from source-supported rows only.

This specification does not authorize actual data migration.

## 8. Required Test Plan for Sprint E

Sprint E should add or update focused tests only after implementation is explicitly approved.

### Raw History Tests

Potential file:

```text
tests/core/test_build_fundamentals_history_intake.py
```

Cover:

- required columns;
- duplicate ticker/fiscal-year/fiscal-period detection;
- date parsing;
- missing source reference;
- missing currency;
- forbidden fields;
- local ignored policy behavior where relevant.

### Metrics Tests

Potential file:

```text
tests/core/test_build_fundamental_metrics.py
```

Cover:

- gross margin;
- operating margin;
- net margin;
- debt to equity;
- return on equity;
- free cash flow margin;
- revenue year-over-year growth;
- EPS year-over-year growth;
- free cash flow year-over-year growth;
- 3-year CAGR;
- missing inputs;
- zero denominators;
- negative denominators;
- sign changes;
- non-consecutive fiscal years;
- period inconsistency;
- currency inconsistency.

### Quality Compatibility Tests

Potential file:

```text
tests/core/test_build_fundamental_quality_compatibility.py
```

Cover:

- current expected output columns;
- source missing compatibility;
- insufficient data compatibility;
- partial or ready states if introduced;
- row preservation;
- ticker/date identity;
- no forbidden semantics.

### Analysis Tests

Potential file:

```text
tests/core/test_build_fundamental_analysis.py
```

Cover:

- growth state;
- margin state;
- debt state;
- profitability state;
- cashflow state;
- review-required propagation;
- no allocation, tradeability, ranking, or scoring fields.

### Pipeline Compatibility Tests

Update only in future implementation if needed:

- timing state tests;
- portfolio intelligence tests;
- decision engine tests;
- reporting tests only if artifact shape changes.

Compatibility tests must prove that downstream layers continue to receive the fields they require unless a governed contract change explicitly updates those requirements.

## 9. Orchestration and Pipeline Considerations

Expected future orchestration:

```text
raw history validation or intake
-> metrics
-> quality
-> analysis
-> downstream pipeline compatibility surface
```

Required orchestration properties:

- raw history validation or intake before metrics;
- metrics before quality;
- quality before analysis;
- current downstream pipeline compatibility preserved;
- deterministic sequencing;
- no provider calls without separate authorization;
- no scraping without separate authorization;
- no generated artifact commits unless explicitly approved.

Likely orchestration touchpoints:

- `scripts/run_full_pipeline.py`
- `scripts/run_scan.py`
- any current call to `build_fundamental_layer.py`

Sprint D does not authorize orchestration changes.

## 10. Generated Artifact Policy

Sprint D generates no runtime artifacts.

Sprint E may generate artifacts only if explicitly approved.

Generated files must not be committed unless repository policy explicitly allows them.

Validation should use fixtures and focused outputs wherever possible.

Generated outputs are evidence, not doctrine.

## 11. Forbidden Semantics

Outside the Decision Engine, fields or logic must not imply:

- buy;
- sell;
- final action;
- allocation;
- tradeability;
- urgency;
- conviction;
- eligibility;
- ranking;
- scoring authority;
- hidden filtering;
- reporting-based decision logic.

Outside the Decision Engine these terms may appear only in negative tests, documentation of forbidden terms, or downstream final decision artifacts governed by Decision Engine contracts.

## 12. Sprint E Implementation Boundary

After explicit approval, Sprint E may implement:

- raw history schema validation or intake helper;
- metrics builder;
- quality compatibility refactor or wrapper;
- analysis builder;
- focused tests;
- optional fixtures;
- documentation updates needed to align approved implementation details.

Sprint E may not implement without separate approval:

- provider/API integration;
- scraping;
- automatic source-data collection;
- portfolio metadata changes;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- generated artifact commits;
- raw CSV data commits.

Sprint E must remain compatible with the current downstream pipeline unless a governed contract change explicitly authorizes otherwise.

## 13. Acceptance Criteria for Sprint E

Sprint E will be acceptable only if:

- existing pipeline behavior remains compatible or governed changes are explicitly documented;
- existing downstream tests still pass or are updated under approved contract;
- row preservation remains protected;
- forbidden semantics remain absent outside approved Decision Engine surfaces;
- formulas match the calculation spec;
- missing or invalid data produces review-required or null outputs, not inferred values;
- source-data policy is respected;
- no Decision Engine authority is weakened;
- generated artifacts are not committed unless explicitly approved;
- provider calls and scraping are not introduced unless separately approved.

## 14. Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Existing backlog items `BL-0015`, `BL-0017`, `BL-0019`, `BL-0020`, and `BL-0021` already cover the current fundamentals gaps.

## 15. Validation

Commands run for Sprint D:

- `git checkout main`
- `git pull origin main`
- `git status`
- `git checkout -b docs/sprint-d-fundamentals-history-implementation-spec`
- `sed -n '1,220p' docs/active/contracts/fundamentals_platform_contract.md`
- `sed -n '1,260p' docs/active/contracts/fundamental_calculations_technical_spec.md`
- `sed -n '1,220p' docs/active/analysis/financial_analysis_contract.md`
- `sed -n '1,220p' docs/active/analysis/functional_analysis_contract.md`
- `sed -n '1,220p' docs/active/analysis/technical_analysis_contract.md`
- `sed -n '1,260p' docs/active/roles_and_responsibilities.md`
- `sed -n '1,280p' docs/active/inventory/fundamentals_code_inventory.md`
- `sed -n '1,240p' docs/active/contracts/pipeline_contracts.md`
- `sed -n '1,260p' docs/active/architecture_current_state.md`
- `sed -n '1,260p' docs/active/governance_v2.md`
- `sed -n '1,260p' docs/sprints/fundamentals_simplification_sprint_plan.md`

Validation confirmation:

- only documentation files changed;
- no code files changed;
- no tests changed;
- no CSV files changed;
- no raw data changed;
- no generated files changed;
- no workflow files changed;
- no provider APIs called;
- no scraping performed;
- no pipeline run;
- no tests run.
