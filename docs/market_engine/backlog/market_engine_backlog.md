# Market Engine Backlog

Owner role: Scrum Master / PM / Product Owner

Status: ACTIVE MARKET ENGINE BACKLOG

## Purpose

This backlog captures the Market Engine sprint line.

`ME01–ME13` are the historical foundation phase. From `ME-GOV01` onward, all future Market Engine work must use the job-scoped sprint naming convention defined in:

```text
docs/market_engine/governance/me_gov01_job_scoped_sprint_naming_convention.md
```

Backlog items do not authorize implementation unless the sprint scope explicitly does so and repository governance allows it.

## Backlog Rules

* Preserve old repository assets as reference material.
* Do not blindly copy old script-era code.
* Do not use old quick scripts as canonical runtime.
* Do not continue legacy cleanup as the active implementation path.
* Do not delete, archive, rename, or ignore old files as part of Market Engine backlog work unless a sprint explicitly authorizes it.
* Keep classification upstream and allocation downstream.
* Preserve Decision Engine authority as the only allocation authority.
* Keep source readiness separate from investment quality.
* Keep missing data explicit.
* Do not convert missing numeric values to zero.
* Do not introduce BUY / SELL / HOLD, recommendation, allocation, urgency, conviction, tradeability, or hidden ranking semantics outside an approved Decision Engine or recommendation-review boundary.
* Do not introduce Telegram, reporting, portfolio, watchlist, provider, or runtime side effects unless a sprint explicitly authorizes them.
* Future Market Engine sprints must use job-scoped sprint IDs.
* Future sprints must be preserved in the backlog and roadmap as soon as they are identified as logical next steps.
* Planned sprint sequence may only be interrupted when a real problem, blocker, architectural gap, governance risk, test gap, data-quality issue, or newly discovered dependency requires insertion.
* When a sprint is inserted ahead of the planned sequence, the insertion reason must be documented in the backlog and roadmap.
* Code changes should usually happen inside one job family at a time.
* Cross-job work must be explicitly labeled as governance, QA, data governance, or integration contract work.
* Analysis, recommendation, portfolio review, and delivery authority must remain separated.

## Active Results-first Advice Baseline Chain

The active baseline chain for 500-ticker progress is now:

```text
ME-GH02 - Batch artifact discovery and ticker status index (completed)
  -> ME-ADV01 - Minimal deterministic advice engine v1 (completed)
  -> ME-ADV02 - 500-ticker advice batch output (completed)
  -> ME-DATA01 - Close highest-impact advice data coverage gaps (completed)
  -> ME-EVAL01 - Advice outcome tracking and feedback loop (completed)
  -> ME-EVAL02 - Scheduled/future outcome refresh using local snapshots (completed)
  -> ME-DATA02 - Import missing and forward local price snapshots for unresolved outcomes (implementation complete / coverage partial)
  -> ME-BOOT03 - Bootstrap authoritative universe and local price-history coverage (implementation complete / coverage partial)
  -> ME-DATA04 - Build complete canonical local market dataset (operational dataset partial)
  -> ME-DATA05 - Incremental market data refresh and forward evaluation (completed / incremental_refresh_operational)
  -> ME-RUN30 - Full canonical-universe analysis and candidate ranking (completed / completed_with_blockers)
  -> ME-RUN31 - Add broader non-price evidence to canonical-universe ranking (completed / completed_with_blockers)
  -> ME-DATA06 - Expand canonical fundamental evidence coverage from local approved evidence sources (implemented / local_coverage_improved_with_remaining_blockers)
  -> ME-DATA07 - Expand validated MVP fundamental metric sourcing for remaining canonical-universe blockers (implemented / operator_import_operational / pilot_blocked_missing_operator_evidence)
  -> ME-DATA08 - Prepare and validate a governance-approved operator fundamental metric package (implemented / local_package_preparation_operational)
```

ME-ADV01 implemented the first minimal deterministic advice engine. It consumes
the ME-GH02 `ticker_status_index.json` and linked dry-run artifacts, writes
`manifest.json`, `advice_index.json`, `advice_index.md`,
`advice_summary.json`, and `unable_to_advise.json` under
`artifacts/market_engine/advice_runs/<run_id>/`, and produces one advice label
per ticker.

ME-ADV01 requires no OpenAI API key, performs no provider invocation, performs
no source acquisition or live data refresh, and adds no broker/order execution,
portfolio/watchlist mutation, Telegram, or delivery side effects.

ME-ADV02 scaled the deterministic advice output path into batch output. It
writes batch advice artifacts under
`artifacts/market_engine/advice_batches/<run_id>/`, including per-label
markdown reports, `missing_data_report.md`, and `coverage_report.md`.

The ME-ADV02 sample run produced advice output for all 12 tickers in the
widest available ME-GH02 status index and reported 2.40% coverage against a
500-ticker target. All 12 labels were `watchlist`, with `portfolio_context`
and `setup_price_market_context` missing for all covered tickers.

ME-DATA01 closed the highest-impact setup/price/market coverage gap enough to
break watchlist-only output. It reused existing local dry-run artifacts and
local price-history CSVs, added deterministic setup/price/market context, and
reran the batch output. The sample changed from `watchlist: 12` to
`buy_candidate: 4`, `wait_for_price: 2`, `avoid_for_now: 1`, and
`watchlist: 5`.

ME-EVAL01 created the first deterministic advice feedback loop. It records the
advice label and advice-date anchor, reads local price-history CSVs only,
computes 5/21/63 trading-day outcomes where possible, writes unresolved
outcomes separately, aggregates label performance, and produces rule feedback.
The sample evaluation run
`me-eval01-advice-outcomes-20260712T120000Z` evaluated 12 tickers, resolved 0
outcomes, and marked 12 unresolved outcomes. The unresolved reasons were
`insufficient_forward_data: 8` and `missing_price_history: 4`.

ME-EVAL02 added the local refresh flow for existing unresolved advice outcomes.
It consumes the ME-EVAL01 `advice_outcome_index.json`, selects unresolved
outcomes, reloads the original advice context without regenerating advice,
uses the latest supplied local price-history root, and writes
`refresh_outcome_index.json`, `refresh_report.md`,
`missing_price_history.json`, and `manifest.json` under
`artifacts/market_engine/evaluation_refresh_runs/<run_id>/`.

The ME-EVAL02 real-world run
`me-eval02-refresh-local-snapshots-20260712T130000Z` selected 12 unresolved
outcomes, resolved 0, kept 8 as `insufficient_forward_data`, and reported
4 as `missing_price_history`: `CLS`, `CRDO`, `IREN`, and `VRT`.

ME-DATA04 then built the complete canonical local market dataset for 952
instruments. The completed dataset established 946 valid current histories,
6 explicit insufficient-history or insufficient-forward-data blockers, no
missing histories, no invalid histories, and no unsupported mappings.

ME-DATA05 converted the ME-DATA04 dataset flow into an operational
incremental refresh. The command reads existing local histories, requests only
stale recent windows with overlap, writes only changed files, refreshes
coverage automatically, and runs ME-EVAL02 automatically. The two same-cutoff
runs `me-data05-incremental-refresh-20260713T140000Z` and
`me-data05-idempotency-refresh-20260713T141000Z` both checked 952 histories,
reported 946 `already_current`, 2 `stale_after_update`, 4
`insufficient_history`, 0 files rewritten, 0 rows added, and 12 ME-EVAL02
outcomes still unresolved due to `insufficient_forward_data`.

PR review follow-up kept ME-DATA05 price-refresh focused: `--refresh-universe`
now fails closed because no supported in-place canonical membership refresh
implementation exists for this flow. Persisted artifacts are compacted:
`per_ticker_status.json` is the only full per-ticker detail list,
`refresh_summary.json` is aggregate-only, and the duplicate
`already_current.json` artifact was removed.

ME-RUN30 then executed the first full canonical-universe technical setup
screening and candidate ranking over the 952-instrument dataset. It attempted
all 952 instruments, analysed 946 eligible local price histories, split 4
insufficient-history blockers from 2 stale-history blockers, reported
setup/screening/blocker distributions, produced 330 ranked technical review
candidates, and generated a top-candidate review package. The PR review fix
removed canonical-looking advice labels from ME-RUN30 output and made the
ranking scope explicit as `technical_setup_screening`. Missing fundamental,
portfolio, and market context remain visible and penalised; all ranked
candidates are `full_advice_ready: false`.

ME-RUN31 then attached available local non-price evidence to the ME-RUN30
technical screening result and handed the generated ticker status index to the
existing deterministic advice engine. The full run
`me-run31-broad-non-price-evidence-full-advice-readiness-20260715T154103Z`
attempted all 952 canonical instruments, completed the advice engine for all
952, found 4 canonical advice-input-ready instruments, produced 4
`wait_for_price` partial advice outputs, kept 948 instruments
`unable_to_advise`, and produced 0 full-advice-ready ranking candidates. The
review fix removed hardcoded freshness, made technical input explicit,
validated source dates and duplicate rows, and wrote a compact committed
evidence package under
`artifacts/market_engine/run_evidence/me-run31-broad-non-price-evidence-full-advice-readiness-20260715T154103Z/`.
The dominant blocker remains fundamental evidence coverage: 931 missing and
17 partial fundamental contexts.

ME-DATA06 then inventoried local fundamental evidence sources, normalized
consumeable local evidence into the existing ME-RUN31 fundamental quality CSV
contract, and reran ME-RUN31 using the normalized artifact. The run
`me-data06-fundamental-evidence-coverage-expansion-20260715T163629Z`
discovered 5 source families, consumed 3, rejected 2 as not runtime
fundamental-quality evidence, and reduced missing fundamental contexts from
931 to 907. Complete contexts improved from 4 to 6, partial contexts from
17 to 39, canonical advice-input-ready instruments from 4 to 6, and
unable-to-advise instruments from 948 to 946. Full-advice-ready remained 0.
The newly advice-input-ready tickers were `ENPH` and `FTNT`.
PR #462 review follow-up reran the comparison as
`me-data06-fundamental-evidence-coverage-review-fix-20260718T113254Z` with an
explicit validated ME-RUN31 per-ticker baseline. Aggregate coverage remained
unchanged, no regressions were found, and the corrected transition set contains
22 rather than 18 `missing_to_partial` tickers because `CLS`, `CRDO`, `IREN`,
and `VRT` were omitted by the earlier CSV-based transition comparison.

### ME-EVAL02 - Scheduled/future outcome refresh using local snapshots

Owner roles: Product Owner / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-EVAL / Outcome evaluation

Status: COMPLETED

Goal: rerun deterministic advice outcome evaluation against later local
snapshots so unresolved ME-EVAL01 horizons can become resolved without live
provider acquisition.

Scope: local evaluation refresh only. No OpenAI API, provider invocation, live
download, broker/order execution, portfolio/watchlist mutation, Telegram,
delivery side effects, Decision Engine allocation authority changes, or advice
rule rewrites.

### ME-DATA02 - Import missing and forward local price snapshots for unresolved outcomes

Owner roles: Product Owner / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-DATA / Local data coverage

Status: IMPLEMENTATION COMPLETE / ACQUISITION COVERAGE PARTIAL

Goal: make local price-history CSVs available for unresolved evaluation
tickers that currently have no `data/processed/<ticker>.csv`, and make later
local snapshots available for tickers that still have insufficient forward
data after the advice date.

Scope: local data import or fixture/snapshot ingestion only when explicitly
approved. No live download, provider refresh, API use, broker/order execution,
portfolio/watchlist mutation, Telegram, delivery side effects, or advice rule
changes.

ME-DATA02 created a canonical local market-data universe config and local
coverage command. The full report-only run
`me-data02-full-coverage-report-only-20260712T142000Z` produced 308 canonical
instruments, 299 unique equities, 9 ETFs, and 3 market-context instruments.
It did not claim full 1,000+ index coverage because no reproducible local S&P
500, Nasdaq-100, S&P MidCap 400, or STOXX Europe membership source was present.

Coverage remains partial:

- valid current snapshots: 0
- imported: 0
- refreshed: 0
- missing: 12
- insufficient: 293
- invalid: 1
- unsupported mappings: 2

The ME-EVAL02 critical run remained blocked: 8
`insufficient_forward_data`, 4 `missing_price_history` (`CLS`, `CRDO`,
`IREN`, `VRT`).

### ME-DATA03 - Operator-supplied local price snapshot import for ME-EVAL blockers

Owner roles: Product Owner / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-DATA / Local data coverage

Status: SUPERSEDED BY ME-DATA04 / RETAINED FOR BACKLOG HISTORY

Goal: provide approved local CSV snapshots for the 12 critical ME-EVAL
unresolved outcome tickers, including missing snapshots for `CLS`, `CRDO`,
`IREN`, and `VRT`, and sufficient forward rows after the original advice date
for the other 8 tickers.

Scope: operator-supplied local snapshot import and validation only. No live
download, new provider architecture, broker/order execution, portfolio or
watchlist mutation, advice generation, outcome-rule changes, Telegram,
scheduler, queue, daemon, machine learning, or recommendation threshold
changes.

ME-DATA04 and ME-DATA05 superseded this as the active baseline path by creating
the complete canonical local dataset and the repeatable incremental refresh
flow. The item remains documented because it was a valid intermediate baseline
candidate before the broader dataset path landed.

### ME-DATA05 - Incremental market data refresh and forward evaluation

Owner roles: Product Owner / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-DATA / Local data coverage

Status: COMPLETED / incremental_refresh_operational

Goal: provide a safe, repeatable operator command for refreshing the complete
local market dataset without full re-downloads of already valid histories,
then automatically refresh coverage and ME-EVAL02 outcomes.

Scope: local price-history refresh, validation, coverage reporting, and
ME-EVAL02 refresh only. No broker/order execution, portfolio or watchlist
mutation, advice generation, synthetic forward data, Telegram, scheduler,
Decision Engine authority change, or recommendation threshold change.

### ME-RUN30 - Full canonical-universe analysis and candidate ranking

Owner roles: Product Owner / Operator / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN / Broad analysis execution

Status: COMPLETED / completed_with_blockers

Goal: use the now-operational 952-instrument local market dataset for broad
Market Engine technical setup screening and deterministic candidate ranking.

Scope: broad local technical screening execution and reporting over existing
canonical dataset artifacts. Reuse canonical setup/price/market context
contracts where available. No full deterministic advice labels, allocation
authority, broker/order execution, portfolio/watchlist mutation, Telegram
delivery, synthetic data, or Decision Engine changes.

### ME-RUN31 - Add broader non-price evidence to canonical-universe ranking

Owner roles: Product Owner / Operator / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN / Broad analysis execution

Status: COMPLETED / completed_with_blockers

Goal: connect available non-price evidence families into broad
canonical-universe full-advice readiness so the existing deterministic advice
engine can be used without broad technical-screening candidates masquerading
as full advice.

Scope: local deterministic evidence integration and adapter work only. No
allocation authority, broker/order execution, portfolio/watchlist mutation,
Telegram delivery, synthetic data, live provider calls, or Decision Engine
changes.

Result: the implementation added
`src/market_engine/run/broad_non_price_evidence_advice_readiness.py`, broad
evidence coverage tests, and the full advice-readiness run under
`artifacts/market_engine/full_advice_readiness_runs/me-run31-broad-non-price-evidence-full-advice-readiness-20260715T112146Z/`.
The compact committed evidence package is under
`artifacts/market_engine/run_evidence/me-run31-broad-non-price-evidence-full-advice-readiness-20260715T112146Z/`.
The sprint confirms the adapter and canonical advice handoff are operational,
but full-advice ranking remains empty because local fundamental evidence is
mostly missing.

### ME-DATA06 - Expand canonical fundamental evidence coverage from local approved evidence sources

Owner roles: Product Owner / Operator / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-DATA / Local evidence coverage

Status: IMPLEMENTED / local_coverage_improved_with_remaining_blockers

Goal: increase canonical-universe fundamental context coverage from approved
local evidence sources so more ME-RUN31 candidates can reach deterministic
canonical advice readiness.

Scope: local evidence inventory, import, normalization, validation, and
coverage reporting only. No provider side effects unless explicitly approved,
no allocation authority, no advice rule changes, no broker/order execution, no
portfolio or watchlist mutation, no Telegram delivery, and no Decision Engine
changes.

Result: ME-DATA06 implemented
`src/market_engine/data/fundamental_evidence_coverage.py`, wrote the full run
artifact under
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-fundamental-evidence-coverage-expansion-20260715T163629Z/`,
and reran ME-RUN31 as
`me-run31-after-me-data06-fundamental-evidence-coverage-20260715T163629Z`.
Measured before/after:

```text
fundamental_complete: 4 -> 6
fundamental_partial: 17 -> 39
fundamental_missing: 931 -> 907
canonical_advice_input_ready: 4 -> 6
full_advice_ready: 0 -> 0
unable_to_advise: 948 -> 946
```

Remaining dominant blocker:

```text
missing_fundamental_context: 907
```

PR #462 review follow-up status: `completed`. The validated rerun
`me-data06-fundamental-evidence-coverage-review-fix-20260718T113254Z` preserved
the aggregate counts above, derived readiness from the baseline artifact,
found no regressions, and corrected inventory freshness to three current
sources, one unknown source, and one not-assessed source.

### ME-DATA07 - Expand validated MVP fundamental metric sourcing for remaining canonical-universe blockers

Owner roles: Product Owner / Operator / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-DATA / Local evidence coverage

Status: IMPLEMENTED / operator_import_operational / pilot_blocked_missing_operator_evidence

Goal: expand approved, current, normalized MVP fundamental metrics for the
remaining canonical-universe instruments still blocked by missing fundamental
context after ME-DATA06.

Scope: source approval, local evidence acquisition or operator-supplied local
evidence import, validation, normalization, and coverage reporting only. No
recommendation-rule changes, allocation authority, broker/order execution,
portfolio/watchlist mutation, Telegram delivery, scheduler behavior, or
Decision Engine changes.

Result: ME-DATA07 confirmed the five-metric MVP contract, reconciled 952
canonical tickers, and implemented the source-approval, symbol-mapping,
operator-import, validation, immutable-snapshot, normalization, and explicit
ME-DATA06 downstream gates. The actual pilot run
`me-data07-validated-mvp-fundamental-metric-sourcing-20260718T122028Z` selected
12 tickers and failed closed because the explicit operator evidence package was
absent. It performed zero provider calls, imported and normalized zero records,
created no raw snapshot, and did not run ME-DATA06 or ME-RUN31. Coverage
therefore remains 6 complete, 39 partial, 907 missing, 6 canonical
advice-input-ready, 0 full-advice-ready, and 946 unable-to-advise; no coverage
improvement or regression claim was made.

PR #463 review follow-up: the corrected blocked pilot
`me-data07-validated-mvp-fundamental-metric-sourcing-review-fix-20260718T141045Z`
now derives terminal batch counts from the canonical per-ticker statuses. It
reconciles 12 selected as 0 successful, 12 blocked, 0 failed, and 0 pending,
with 940 not selected. The missing operator input records one presence check
and zero actual import attempts. Coverage and readiness remain unchanged.

Remaining blocker: a governance-approved operator package with primary-source
metric lineage is required. Recommended next sprint:

```text
ME-DATA08 - Prepare and validate a governance-approved operator fundamental metric package
```

### ME-DATA08 - Prepare and validate a governance-approved operator fundamental metric package

Owner roles: Product Owner / Operator / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-DATA / Local evidence coverage

Status: IMPLEMENTED / local_package_preparation_operational

Goal: give an operator one deterministic, versioned, fail-closed local flow to
prepare governance-approved primary-source metric evidence for the existing
ME-DATA07 operator-import boundary.

Result: ME-DATA08 implemented the
`market-engine-data08-operator-fundamental-metric-input-v1` contract, strict
validation against the normative five-metric ME-DATA06/07 allowlist,
transparent percent-to-ratio normalization metadata, deterministic
ME-DATA07-compatible package serialization, a machine-readable validation
report, stable reason codes, and explicit CLI exit behavior. Any blocking
record rejects the whole package. Acceptance means only eligibility for an
explicit ME-DATA07 operator-import step; it does not import evidence or grant
analysis, recommendation, tradeability, or decision authority.

Remaining blocker: genuine governance-approved primary-source operator
evidence is still required before the existing ME-DATA07 pilot can be run.
Recommended next story: execute a bounded ME-DATA07 pilot with an accepted
ME-DATA08 package and review the resulting local coverage evidence. Production
import, automatic downstream execution, live acquisition, and authority
changes remain blocked.

## Current ChatGPT Advisory Artifact Chain

The current ME-CI advisory chain is:

```text
ME-CI01 - Structured Decision Output contract (completed)
  -> ME-CI02 - ChatGPT Advisory Context Contract (completed)
  -> ME-CI03 - ChatGPT-readable Portfolio Intelligence Context (completed)
  -> ME-CI04 - Explainability / Change-Rationale Contract (completed)
  -> ME-CI05 - Daily ChatGPT-ready advisory artifact (completed)
  -> ME-CI06 - Advisory artifact schema validation and contract enforcement (completed)
  -> ME-CI07 - ChatGPT advisory prompt and response-grounding contract (completed)
  -> ME-CI08 - Controlled advisory response dry run and grounding validator scaffold (completed)
  -> ME-CI09 - Harden advisory response grounding fixtures and validator coverage (completed)
  -> ME-CI10 - Define controlled model invocation boundary contract (completed)
  -> ME-CI11 - First real grounded advisory output flow (implemented with invocation blocked by local configuration)
  -> ME-CI11B - Execute configured real grounded advisory model invocation (blocked by missing local OPENAI_API_KEY)
  -> ME-CI11C - Run configured provider invocation with local API key (blocked by Codex command process env propagation)
  -> ME-CI11D - Fix Codex command process provider environment propagation (next)
```

ME-CI08 implements the first local deterministic dry-run and grounding
validator scaffold over explicit synthetic response fixtures. It does not
enable real ChatGPT advisory answer generation, prompt execution, OpenAI API
integration, notification delivery, broker integration, portfolio mutation,
watchlist mutation, allocation, sizing, execution, or autonomous decision
making.

ME-CI09 hardens the ME-CI08 local scaffold with adversarial fixture coverage,
claim/reference graph validation, support-type compatibility, path-family
containment, referenced/absent context handling, grounding-summary consistency,
partial-answer completeness, family-level freshness relevance, blocker
preservation, Dispatch contradiction coverage, lineage validation, and
deterministic issue ordering. It remains local-only, deterministic, model-free,
provider-free, delivery-free, non-production, and fail-closed.

ME-CI10 defines the controlled model invocation boundary contract for a future
provider/model implementation. It documents invocation eligibility, input
boundaries, context minimization, request and result envelopes, run/invocation
identity, idempotency, provider/model identity, capability requirements,
budgets, timeout/retry semantics, raw response capture, parser boundary,
grounding handoff, failure taxonomy, fail-closed behavior, and future
implementation tests. It is docs-only and does not add model invocation,
provider calls, parser runtime, delivery, or downstream authority.

ME-CI11 implements the local grounded advisory output flow from a real Market
Engine artifact through source validation, deterministic input packaging,
CI10-shaped invocation request generation, raw-response capture, strict parser
behavior, allowed-evidence grounding validation, structured output persistence,
readable report rendering, and manifest persistence. The first real NVDA run
failed closed before provider invocation because required local model
configuration was absent. ME-CI11 therefore does not claim a successful real
model advisory response.

ME-CI11B used the corrected ME-CI11 runtime with explicit non-production model
configuration and attempted the primary NVDA run plus an AMD smoke run through
the same command path. Both runs failed closed before provider invocation
because the local environment did not contain `OPENAI_API_KEY`. The exact
blocker is local provider credentials, not source artifacts, parser behavior,
CI09 grounding logic, or ticker-specific runtime behavior.

ME-CI11C attempted the same universal CI11 command path with
`MARKET_ENGINE_ADVISORY_MODEL=gpt-4.1-mini` explicitly set. The persisted
invocation requests prove the model was included, but the Codex command process
did not contain a non-empty `OPENAI_API_KEY`. NVDA and AMD therefore failed
closed before provider invocation with `blocked_invocation_not_configured`.

ME-CI11D is the next advisory sprint. It must fix only the process-environment
propagation issue so the exact command process used by Codex sees a non-empty
`OPENAI_API_KEY`, then rerun the same universal CI11 command path before any
batch or report-quality expansion.

## Historical Foundation Phase

`ME01–ME13` remain historical foundation sprints and must not be renumbered.

They may be referenced as:

```text
ME01–ME13 foundation phase
```

or individually where needed.

Future work must not continue as `ME14`, `ME15`, etc.

## Foundation Sprint Roadmap

### ME01 - Reset Market Engine documentation structure and knowledge extraction policy

Owner roles: PM / Product Owner, Scrum Master, Governance Auditor, Technical Architect

Status: COMPLETED BY ME01

Goal: Create the Market Engine documentation root, knowledge extraction policy, source inventory, baseline coding and testing standards, placeholders, audit record, and backlog.

Scope: Documentation and backlog only.

Outcome: `docs/market_engine/` was established as the active Market Engine documentation root with required baseline documentation and backlog structure.

### ME02 - Extract and write Market Engine functional flow

Owner roles: Functional Analyst, Product Owner, Scrum Master, Governance Auditor

Status: COMPLETED BY ME02

Goal: Extract the Market Engine functional flow from existing documentation, code, tests, audits, and backlog items.

Scope: Functional flow specification, role responsibilities, user/operator workflows, classification flow, state boundaries, and implementation/testing implications.

Outcome: Source intake, analysis, operator review, and downstream layers were separated and documented.

### ME03 - Extract and write Market Engine financial, scanner, and fundamental logic

Owner roles: Financial Analyst, Data Steward, Functional Analyst, Governance Auditor

Status: COMPLETED BY ME03

Goal: Extract financial, scanner, fundamental, and source-readiness logic for Market Engine specifications.

Scope: Financial logic, scanner classification lessons, fundamental data lessons, provider/source readiness, data implications, missing-data rules, quality-state rules, ticker failure handling, source-intake boundaries, analysis boundaries, and failure modes.

Outcome: Financial/scanner/fundamental rules were documented while preserving the boundary between source intake, analysis, recommendation, and allocation authority.

### ME04-PREP - Archive old active documentation and make Market Engine the only active docs root

Owner roles: Scrum Master, Governance Auditor, Technical Architect

Status: COMPLETED BY ME04-PREP

Goal: Preserve former active v2, BL, and reset documentation as historical reference material while making `docs/market_engine/` the only active Market Engine documentation root.

Scope: Documentation structure only.

Outcome: Former active documentation was preserved under `docs/archive/market_scanner_reference/active/`.

### ME04-PREP-B - Inventory remaining legacy documentation outside Market Engine

Owner roles: Scrum Master, Governance Auditor

Status: COMPLETED BY ME04-PREP-B

Goal: Inventory remaining documentation and reference material outside `docs/market_engine/` and outside the Market Scanner reference archive.

Scope: Documentation inventory only.

Outcome: Remaining legacy documentation candidates were inventoried before consolidation.

### ME04-PREP-C - Consolidate remaining legacy documentation under Market Scanner reference archive

Owner roles: Scrum Master, Governance Auditor

Status: COMPLETED BY ME04-PREP-C

Goal: Move clear legacy documentation candidates under `docs/archive/market_scanner_reference/` while keeping `docs/market_engine/` as the only active Market Engine documentation root.

Scope: Documentation structure only.

Outcome: Legacy documentation/reference areas were preserved under `docs/archive/market_scanner_reference/`.

### ME04-PREP-D - Inventory legacy runtime, tests, and data before Market Engine cutover

Owner roles: Technical Architect, Development Lead, Data Steward, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME04-PREP-D

Goal: Inventory legacy runtime, scripts, tests, data, reports, and root-level files before Market Engine cutover.

Scope: Documentation-only inventory.

Outcome: Old runtime, scripts, tests, data, reports, and root-level files were classified without moving, deleting, or modifying runtime assets.

### ME04 - Extract and write Market Engine technical, coding, and testing architecture

Owner roles: Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME04

Goal: Extract technical architecture, coding rules, and testing architecture for Market Engine.

Scope: Module ownership, provider/data/analysis/decision separation, runtime boundaries, side-effect controls, test-family conventions, manual smoke harness standards, forbidden field policy, and file strategy.

Outcome: Market Engine technical ownership, provider boundaries, test boundaries, and file/module strategy were documented.

### ME05 - Build all-ticker source intake smoke

Owner roles: Development Lead, Data Steward, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME05

Goal: Build an explicit all-ticker source intake smoke harness after ME02 through ME04 specifications authorize the boundary.

Scope: Bounded manual source intake smoke harness, source availability capture, per-ticker failure capture, raw evidence feasibility, normalized data feasibility, missingness preservation, and source-readiness states.

Outcome: A clean `src/market_engine/source_intake/` package, fake provider scenarios, readiness statuses, per-ticker intake results, batch summaries, missing-field frequency tracking, targeted tests, fake-provider manual smoke entrypoint, and audit/documentation updates were added.

### ME06 - Add bounded real provider source intake smoke and coverage review

Owner roles: Data Steward, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME06

Goal: Add a bounded real-provider source intake smoke and review coverage evidence without entering analysis or recommendation behavior.

Scope: First real provider selection, explicit manual invocation, ticker limit, source coverage evidence, failure triage, source-readiness implications, missing-data observations, provider/source limitations, data-owner review, generated-output/archive decision inputs, and backlog follow-up.

Outcome: A SEC CompanyFacts provider adapter, mocked provider tests, explicit real-provider manual smoke flags, ticker limit enforcement, and local source coverage review were added.

### ME07 - Review real-provider coverage and define source-data owner decisions

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME07

Goal: Review ME06 real-provider coverage behavior and define source-data owner decisions before building first fundamental source context.

Scope: Provider availability review, SEC access/user-agent/network follow-up, ticker-to-CIK ownership decision, smoke evidence retention policy, required-field alias review, source artifact handling, and readiness criteria for first fundamental source context.

Outcome: The bounded SEC smoke failure was triaged as a controlled network/DNS access failure in the environment; SEC CompanyFacts remained approved for bounded smoke only until access and ownership decisions were resolved.

### ME08 - Repair SEC CompanyFacts network access and rerun bounded coverage review

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME08

Goal: Resolve the SEC CompanyFacts network/request access issue and rerun bounded source coverage before approving source context work.

Scope: SEC access diagnostics, User-Agent/contact policy review, environment/network review, bounded manual SEC smoke rerun, ticker-to-CIK ownership decision, source evidence retention decision, and coverage review documentation.

Outcome: Local runtime DNS and HTTPS access succeeded. The bounded SEC CompanyFacts smoke reached `AVAILABLE=4` for `NVDA`, `AMD`, `META`, and `COST` with no missing fields or provider errors. SEC CompanyFacts was approved for bounded coverage review only.

### ME09 - Run bounded multi-ticker SEC CompanyFacts coverage artifact review

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME09

Goal: Run a bounded multi-ticker SEC CompanyFacts coverage review and evaluate isolated smoke artifacts before any source context or analysis sprint.

Scope: Explicit bounded ticker set, SEC CompanyFacts coverage review, isolated non-production smoke artifacts if explicitly requested, missing-field evidence, provider-error evidence, ticker-to-CIK source ownership review, artifact retention review, and readiness criteria for first source context.

Outcome: A bounded 10-ticker SEC CompanyFacts coverage review reached `AVAILABLE=10` for `NVDA`, `AMD`, `META`, `COST`, `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`, and `AVGO`. Non-production smoke artifacts were written locally and intentionally not committed.

### ME10 - Define approved SEC CompanyFacts field mapping and source coverage contract

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME10

Goal: Convert bounded SEC CompanyFacts smoke evidence into an approved source-field mapping and source coverage contract before source context or analysis work.

Scope: SEC field alias review, required-field contract, ticker-to-CIK ownership decision, source coverage contract, artifact retention policy, missing-field semantics, provider-error semantics, and readiness criteria for first fundamental source context.

Outcome: The first SEC CompanyFacts field mapping and source coverage contract was approved for `revenue`, `net_income`, `operating_cash_flow`, and `capital_expenditures`. SEC CompanyFacts was approved for field mapping implementation, but not for analysis.

### ME11 - Implement SEC field mapping and first fundamental source context

Owner roles: Financial Analyst, Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME11

Goal: Implement the approved SEC CompanyFacts field mapping contract and create the first source-only Market Engine fundamental context.

Scope: Deterministic SEC alias priority, canonical source field mapping, raw source value preservation, SEC fact provenance, missing-data preservation, source readiness, and source-only context objects.

Outcome: SEC CompanyFacts contract mapping, a source-only fundamental context, tests for approved mappings and forbidden substitutions, provenance checks, missing-data checks, and documentation/audit updates were added.

### ME12 - Build first non-decision fundamental analysis pass

Owner roles: Financial Analyst, Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME12

Goal: Build the first non-decision fundamental analysis pass from approved Market Engine source context.

Scope: Source-backed financial observations, explicit missing-data handling, source limitation flags, and deterministic non-decision context suitable for later operator review.

Outcome: The first non-decision fundamental analysis pass was added. It consumes ME11 source context and emits source-grounded observations without free cash flow, growth, margins, ratios, valuation metrics, scores, rankings, recommendations, or Decision Engine behavior.

### ME13 - Define Market Engine job architecture and data persistence contract

Owner roles: Product Owner / Technical Architect / Data Steward / Development Lead / QA Lead / Governance Auditor

Status: COMPLETED BY ME13

Goal: Define the Market Engine job architecture and data persistence contract before additional derived analysis layers.

Scope: Job-oriented architecture, independent input/output contracts, independent persistence paths, independent execution cadences, authority boundaries, side-effect boundaries, tests, upgrade policy, and GitHub Actions direction.

Outcome: Market Engine is governed as a job-oriented system with independent jobs, independent input/output contracts, independent persistence paths, independent execution cadences, and independent upgrade paths. The previously generic post-ME13 `ME14` next-sprint label is superseded by ME-GOV01.

## Job-Scoped Sprint Governance

### ME-GOV01 — Define job-scoped sprint naming convention

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: Governance / architecture / working method

Status: COMPLETED BY ME-GOV01

Goal: Define the job-scoped sprint naming convention for all future Market Engine work after the ME01–ME13 foundation phase.

Scope: Sprint naming convention, job-family prefixes, numbering rules, cross-job sprint rules, split rules, backlog rules, documentation rules, audit rules, testing rules, Codex prompt rules, foundation sprint reference rules, and next sprint approval.

Not in scope: Python code changes, tests, provider calls, runtime execution, generated data, reports, Telegram, portfolio/watchlist mutation, recommendation behavior, or Decision Engine behavior.

Acceptance criteria:

* `ME` remains the project prefix.
* `ME01–ME13` are preserved as historical foundation sprints.
* Future sprints do not continue as `ME14`, `ME15`, etc.
* Future sprints use job-family prefixes.
* Each job family has its own numbering sequence starting at `01`.
* Cross-job work is explicitly labeled as governance, QA, data governance, or integration contract work.
* Split rules protect job independence.
* Analysis, recommendation, portfolio review, and delivery authority remain separated.
* The next approved sprint is `ME-SR01`.

Outcome: Job-scoped sprint naming is approved and documented in `docs/market_engine/governance/me_gov01_job_scoped_sprint_naming_convention.md`.

## Approved Job Families

| Prefix | Job family | Scope |
|---|---|---|
| `ME-GOV` | Governance / architecture / working method | Governance decisions, architecture rules, job-boundary doctrine, sprint rules, working method, roadmap structure, authority separation |
| `ME-SR` | Source Refresh jobs | Fetch, refresh, cache, validate, and persist raw external source data |
| `ME-SC` | Source Context jobs | Convert raw source data into source-aware context, availability states, metadata, and diagnostics |
| `ME-FO` | Fundamental Observation jobs | Produce non-decision fundamental observations from approved source context |
| `ME-DO` | Derived Observation jobs | Produce derived observations, trends, deltas, ratios, comparisons, and computed analytical signals |
| `ME-AR` | Analysis Review jobs | Review observations into analytical interpretation without recommendation authority |
| `ME-RR` | Recommendation Review jobs | Produce recommendation review output from approved analysis inputs |
| `ME-SD` | Setup Detection jobs | Detect non-actionable setups and patterns from approved observation inputs |
| `ME-PR` | Portfolio Review jobs | Apply portfolio-specific context such as positions, allocation, exposure, concentration, and portfolio fit |
| `ME-DL` | Delivery jobs | Deliver already-approved outputs through reports, Telegram, dashboards, exports, or other user-facing channels |
| `ME-CI` | ChatGPT Advisory Integration jobs | Define ChatGPT Advisory Layer and Structured Decision Output contracts for artifact-grounded interactive explanation |
| `ME-PI` | Portfolio Intelligence jobs | Define approved portfolio exposure and context contracts for downstream interpretation |
| `ME-PS` | Position Sizing jobs | Define downstream position-sizing decision contracts without bypassing Decision Engine authority |
| `ME-NL` | Notification Layer jobs | Define channel-neutral compact notification payloads and later adapters after structured outputs stabilize |
| `ME-QA` | Cross-job quality / testing / CI | Contract tests, regression tests, compatibility checks, CI gates, cross-job quality enforcement |
| `ME-DATA` | Data governance / persistence / retention | Shared data layout, persistence policy, retention policy, cache lifecycle, schema storage, and data governance |

## Completed Job-Scoped Sprints

### ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading

Owner roles: Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Source Refresh

Status: COMPLETED BY ME-SR01

Goal: Persist bounded raw SEC CompanyFacts provider responses so future source mapping, context building, and observations can run from cached source snapshots instead of repeatedly calling SEC.

Scope:

* bounded SEC CompanyFacts raw JSON snapshot writing;
* snapshot metadata;
* ticker manifest;
* provider error manifest;
* cached snapshot loading;
* source refresh documentation;
* local source refresh tests;
* old path prohibition tests;
* ME-SR01 audit note.

Approved persistence path:

```text
data/market_engine/source_snapshots/sec_companyfacts/<run_id>/
```

Recommended snapshot structure:

```text
raw/
  NVDA_companyfacts.json
  AMD_companyfacts.json
snapshot_metadata.json
ticker_manifest.csv
provider_errors.csv
```

Explicit non-scope:

* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no broad pipeline refactor;
* no Decision Engine behavior;
* no monolithic run-everything implementation;
* no derived observations;
* no free cash flow;
* no growth;
* no margins;
* no valuation metrics;
* no score;
* no ranking;
* no BUY / SELL / HOLD;
* no portfolio mutation;
* no watchlist mutation;
* no reporting.

Acceptance criteria:

* Raw SEC CompanyFacts snapshots are persisted as raw JSON, not CSV.
* Cached source loading is supported for downstream mapping/context/observation jobs.
* Provider errors are persisted separately from successful raw payloads.
* Snapshot metadata and ticker manifest are written in the approved source snapshot path.
* No old data/report paths are written.
* Tests remain local to Source Refresh unless an explicit `ME-QA` sprint is created.
* No analysis, recommendation, portfolio, delivery, Telegram, or Decision Engine behavior is introduced.

Outcome: ME-SR01 added SEC CompanyFacts raw snapshot persistence, cached raw snapshot loading, provider error manifest writing, metadata validation, latest cached snapshot selection, and an explicit cached SEC provider path that avoids provider/network calls when supplied a cached snapshot file. Tests use temporary local payloads only.

### ME-SC01 — Define SEC CompanyFacts Source Context contract from cached raw snapshots

Owner roles: Data Steward / Technical Architect / Financial Analyst / QA Lead / Governance Auditor

Job family: Source Context

Status: COMPLETED BY ME-SC01

Goal: Define the Source Context contract for building SEC CompanyFacts source context from cached raw source snapshots produced by ME-SR01.

Scope:

* Source Context input contract from cached raw SEC CompanyFacts snapshot envelopes;
* Source Context output contract;
* approved context-level source availability states;
* approved field-level states;
* provenance requirements;
* missingness and provider-error rules;
* persistence paths;
* test requirements for later implementation;
* authority boundaries;
* next implementation sprint identification.

Approved input path:

```text
data/market_engine/source_snapshots/sec_companyfacts/<source_refresh_run_id>/
```

Approved output path:

```text
data/market_engine/source_contexts/fundamentals/<source_context_run_id>/
```

Approved context format version:

```text
sec-companyfacts-source-context-v1
```

Approved context-level states:

* `AVAILABLE`;
* `PARTIAL`;
* `MISSING`;
* `INVALID`;
* `PROVIDER_ERROR`;
* `UNSUPPORTED`.

Approved field-level states:

* `PRESENT`;
* `MISSING`;
* `INVALID`;
* `UNSUPPORTED`.

Approved initial canonical fields:

* `revenue`;
* `net_income`;
* `operating_cash_flow`;
* `capital_expenditures`.

Explicit non-scope:

* no Python implementation;
* no tests;
* no provider calls;
* no runtime behavior;
* no source refresh behavior;
* no fundamental observations;
* no derived observations;
* no free cash flow;
* no growth;
* no margins;
* no valuation metrics;
* no analysis review;
* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no Decision Engine behavior.

Acceptance criteria:

* Source Context contract is documented.
* Input contract from ME-SR01 raw snapshots is defined.
* Output contract for source-only context is defined.
* Context-level and field-level states are defined.
* Provenance requirements are defined.
* Missingness and provider-error rules are defined.
* Persistence paths are defined.
* Test requirements are defined for implementation.
* Authority boundaries are explicit.
* Next implementation sprint is identified as `ME-SC02`.
* No runtime, code, test, provider, data, generated artifact, recommendation, portfolio, delivery, Telegram, or Decision Engine behavior is changed.

Outcome: ME-SC01 approved the SEC CompanyFacts Source Context contract in `docs/market_engine/source_context/me_sc01_sec_companyfacts_source_context_contract.md` and recorded the audit in `docs/market_engine/audits/me_sc01_sec_companyfacts_source_context_contract_audit.md`.

## Next Approved Sprint

### ME-SC02 — Implement SEC CompanyFacts Source Context from cached raw snapshots

Owner roles: Data Steward / Technical Architect / Financial Analyst / Development Lead / QA Lead / Governance Auditor

Job family: Source Context

Status: COMPLETED BY ME-SC02

Goal: Implement the SEC CompanyFacts Source Context contract defined by ME-SC01.

Scope:

* load cached raw SEC CompanyFacts snapshot envelopes from the ME-SR01 persistence path;
* build source-only context output;
* emit implemented context-level states: `AVAILABLE`, `PARTIAL`, and `MISSING`;
* reserve contract-level states: `INVALID`, `PROVIDER_ERROR`, and `UNSUPPORTED`;
* emit implemented field-level states: `PRESENT` and `MISSING`;
* reserve contract-level field states: `INVALID` and `UNSUPPORTED`;
* preserve source provenance;
* preserve source refresh snapshot metadata;
* preserve missingness explicitly;
* fail safely with controlled `SecCompanyFactsContextBuildError` when cached snapshot loading fails;
* persist Source Context output under `data/market_engine/source_contexts/fundamentals/<source_context_run_id>/<ticker>/source_context.json`;
* add local Source Context tests using synthetic/temporary cached payloads only;
* document implementation and audit results.

Explicit non-scope:

* no live provider calls in automated tests;
* no source refresh job runner;
* no source refresh behavior change;
* no source intake provider behavior change;
* no fundamental observations;
* no derived observations;
* no free cash flow;
* no growth;
* no margins;
* no valuation metrics;
* no score;
* no ranking;
* no BUY / SELL / HOLD;
* no analysis review;
* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no Decision Engine behavior.

Acceptance criteria:

* ME-SC02 implements the ME-SC01 contract for cached raw snapshot consumption.
* Source Context can be built from cached raw SEC CompanyFacts snapshots without live provider calls.
* Source Context output remains source-only.
* Missingness remains explicit.
* Numeric zero is treated as present, not missing.
* Raw source provenance and period metadata are preserved.
* Source refresh snapshot metadata is preserved.
* Cached snapshot failures are controlled and explicit.
* Tests prove boundary compliance and old path prohibition.
* Documentation, backlog, and audit are updated.

Outcome: ME-SC02 added a job-scoped Source Context implementation in `src/market_engine/source_context/`, with tests in `tests/market_engine/source_context/`. The implementation consumes ME-SR01 cached raw SEC CompanyFacts snapshots, emits source-only context output, preserves canonical field values, field states, source provenance, source refresh metadata, and missingness, and can persist context JSON under the approved Source Context path. Automated tests use temporary local cached snapshots only and do not make live provider calls.

## Candidate Follow-Up Sprints

### ME-SR02 — Build bounded SEC CompanyFacts source refresh job runner

Owner roles: Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Source Refresh

Status: CANDIDATE FOLLOW-UP

Goal: Build a bounded Source Refresh job runner that fetches a controlled ticker set, persists raw SEC CompanyFacts snapshots, and records provider errors under the approved source snapshot path.

Scope:

* bounded ticker input;
* explicit SEC CompanyFacts provider use;
* raw snapshot writing;
* provider error manifest writing;
* run metadata;
* no downstream source context, observations, analysis, recommendation, portfolio, delivery, Telegram, or Decision Engine behavior.

Acceptance criteria:

* Job runner is explicit and bounded.
* Raw successful payloads are persisted under `data/market_engine/source_snapshots/sec_companyfacts/<run_id>/`.
* Provider errors are persisted separately.
* No old data/report paths are written.
* Automated tests do not call live providers.
* No analysis, recommendation, portfolio, delivery, Telegram, or Decision Engine behavior is introduced.

### ME-FO01 — Define Fundamental Observation contract from SEC CompanyFacts Source Context

Owner roles: Financial Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Fundamental Observation

Status: COMPLETED BY ME-FO01

Goal: Define the non-decision Fundamental Observation contract from approved SEC CompanyFacts Source Context.

Scope:

* define approved Fundamental Observation input contract;
* define approved Fundamental Observation output contract;
* define observation categories;
* define observation states;
* define Source Context state handling;
* define provenance requirements;
* define forbidden authority semantics;
* define persistence path recommendation;
* define ME-FO02 implementation boundaries.

Explicit non-scope:

* no Python code changes;
* no tests;
* no data files;
* no provider calls;
* no runtime behavior;
* no Source Refresh changes;
* no Source Context changes;
* no derived calculations;
* no analysis review;
* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no Decision Engine behavior.

Acceptance criteria:

* Fundamental Observation job boundary is defined.
* Source Context input contract is defined.
* Fundamental Observation output contract is defined.
* Approved observation categories and states are defined.
* Source Context state handling is defined.
* Provenance requirements are defined.
* Forbidden authority semantics are defined.
* Persistence path recommendation is defined.
* ME-FO02 implementation scope is clear.
* Sprint remains documentation/contract only.

Outcome: ME-FO01 defined the Fundamental Observation contract from SEC CompanyFacts Source Context. Implementation is deferred to `ME-FO02 — Implement Fundamental Observations from SEC CompanyFacts Source Context`.

### ME-FO02 — Implement Fundamental Observations from SEC CompanyFacts Source Context

Owner roles: Financial Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Fundamental Observation

Status: COMPLETED BY ME-FO02

Goal: Implement non-decision Fundamental Observations from approved SEC CompanyFacts Source Context.

Scope:

* consume `SecCompanyFactsSourceContext` objects from ME-SC02;
* emit `sec-companyfacts-fundamental-observations-v1` output;
* implement approved ME-FO01 observation categories;
* implement approved ME-FO01 observation states;
* preserve source context state;
* preserve source refresh metadata;
* preserve source values;
* preserve source provenance;
* preserve missingness explicitly;
* treat numeric zero as present;
* persist Fundamental Observation output under `data/market_engine/fundamental_observations/<fundamental_observation_run_id>/<ticker>/fundamental_observations.json`;
* refuse overwrite of existing Fundamental Observation output;
* add local tests using synthetic/temporary cached Source Context input only;
* document implementation and audit results.

Explicit non-scope:

* no raw SEC CompanyFacts fetching;
* no cached raw snapshot loading as a primary input;
* no Source Refresh behavior changes;
* no Source Context behavior changes;
* no derived calculations;
* no free cash flow;
* no growth;
* no margins;
* no ratios;
* no valuation metrics;
* no peer comparison;
* no trend analysis;
* no scoring;
* no ranking;
* no BUY / SELL / HOLD;
* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no reporting;
* no Decision Engine behavior;
* no position sizing;
* no execution advice.

Acceptance criteria:

* Available Source Context produces approved observations.
* Partial Source Context preserves missingness.
* Missing Source Context produces `NOT_ASSESSED` and `MISSING_DATA` observations.
* Positive, negative, zero, and missing source values are handled correctly.
* Numeric zero remains present and produces `ZERO_SOURCE_VALUE` where applicable.
* Source values and provenance are preserved.
* Source refresh metadata is preserved.
* Derived calculations are not emitted.
* Recommendation, score, ranking, portfolio, delivery, Telegram, and Decision Engine authority are not emitted.
* Persistence writes JSON under the approved Fundamental Observation path.
* Persistence refuses overwrite.
* Tests do not use live SEC/provider calls.
* Tests do not import legacy runtime modules.
* Documentation, backlog, and audit are updated.

Outcome: ME-FO02 implemented non-decision Fundamental Observations from SEC CompanyFacts Source Context in `src/market_engine/fundamental_observations/`, with tests in `tests/market_engine/fundamental_observations/`. The implementation consumes ME-SC02 Source Context objects, emits source-grounded observation output, preserves source values, missingness, Source Context state, source refresh metadata, and provenance, and stays inside the ME-FO job family without introducing derived calculations, analysis review, recommendation review, portfolio review, delivery, Telegram, or Decision Engine behavior.

### ME-DO01 — Add first derived cash-generation observation layer

Owner roles: Financial Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Derived Observation

Status: COMPLETED BY ME-DO01

Goal: Add the first derived but still non-decision cash-generation observation layer from approved ME-FO02 Fundamental Observations.

Scope:

* consume `SecCompanyFactsFundamentalObservationSet` objects from ME-FO02;
* emit `sec-companyfacts-derived-cash-generation-observations-v1` output;
* calculate only `free_cash_flow = operating_cash_flow - capital_expenditures`;
* preserve upstream Fundamental Observation references;
* preserve upstream source values;
* preserve upstream source references;
* preserve source context state;
* preserve source refresh metadata;
* preserve missingness explicitly;
* treat numeric zero as present;
* emit positive, negative, and zero derived source-value states;
* emit limitation observations when required source fields are missing;
* persist Derived Cash Generation output under `data/market_engine/derived_observations/cash_generation/<derived_observation_run_id>/<ticker>/derived_cash_generation_observations.json`;
* refuse overwrite of existing Derived Cash Generation output;
* add local tests using synthetic/temporary upstream observations only;
* document implementation and audit results.

Explicit non-scope:

* no raw SEC CompanyFacts fetching;
* no cached raw snapshot loading as a primary input;
* no Source Refresh behavior changes;
* no Source Context behavior changes;
* no Fundamental Observation behavior changes;
* no FCF yield;
* no margins;
* no growth;
* no ratios;
* no valuation metrics;
* no peer comparison;
* no trend analysis;
* no scoring;
* no ranking;
* no BUY / SELL / HOLD;
* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no reporting;
* no Decision Engine behavior;
* no position sizing;
* no execution advice.

Acceptance criteria:

* Positive free cash flow is derived correctly.
* Negative free cash flow is derived correctly.
* Zero free cash flow is derived correctly.
* Zero operating cash flow remains present and can be used in derivation.
* Missing operating cash flow limits derivation explicitly.
* Missing capital expenditures limits derivation explicitly.
* Upstream Fundamental Observation references are preserved.
* Upstream source values and source references are preserved.
* Source context metadata is preserved.
* Source refresh metadata is preserved.
* Persistence writes JSON under the approved Derived Observation path.
* Persistence refuses overwrite.
* Analysis, recommendation, score, ranking, portfolio, delivery, Telegram, and Decision Engine authority are not emitted.
* Tests do not use live SEC/provider calls.
* Tests do not import legacy runtime modules.
* Documentation, backlog, and audit are updated.

Outcome: ME-DO01 implemented the first non-decision Derived Observation layer in `src/market_engine/derived_observations/`, with tests in `tests/market_engine/derived_observations/`. The implementation consumes ME-FO02 Fundamental Observations, derives only free cash flow from operating cash flow and capital expenditures, preserves upstream source values, source references, Source Context state, source refresh metadata, and missingness, and stays inside the ME-DO job family without introducing analysis review, recommendation review, portfolio review, delivery, Telegram, reporting, or Decision Engine behavior.

### ME-AR01 — Define Analysis Review contract from Fundamental and Derived Observations

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Analysis Review

Status: COMPLETED BY ME-AR01

Goal: Define the non-recommendation Analysis Review contract from approved ME-FO02 Fundamental Observations and ME-DO01 Derived Observations.

Scope:

* define approved Analysis Review input families;
* define approved upstream input formats;
* define recommended Analysis Review output format;
* define recommended Analysis Review persistence path;
* define approved Analysis Review categories;
* define approved Analysis Review states;
* define state semantics;
* define recommended review item structure;
* define approved and forbidden message style;
* define provenance requirements;
* define persistence requirements;
* define ME-AR02 implementation requirements;
* preserve recommendation, portfolio, delivery, Telegram, reporting, and Decision Engine boundaries.

Approved input families:

* ME-FO — Fundamental Observations;
* ME-DO — Derived Observations.

Approved initial input formats:

* `sec-companyfacts-fundamental-observations-v1`;
* `sec-companyfacts-derived-cash-generation-observations-v1`.

Recommended output format:

* `sec-companyfacts-analysis-review-v1`.

Recommended output path:

* `data/market_engine/analysis_reviews/<analysis_review_run_id>/<ticker>/analysis_review.json`.

Approved Analysis Review categories:

* `SOURCE_AVAILABILITY_REVIEW`;
* `FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW`;
* `CASH_GENERATION_REVIEW`;
* `FREE_CASH_FLOW_REVIEW`;
* `DATA_LIMITATION_REVIEW`;
* `HUMAN_REVIEW_REQUIREMENT`.

Approved Analysis Review states:

* `SOURCE_HEALTHY`;
* `SOURCE_LIMITED`;
* `OBSERVATIONS_COMPLETE`;
* `OBSERVATIONS_LIMITED`;
* `CASH_GENERATION_POSITIVE`;
* `CASH_GENERATION_NEGATIVE`;
* `CASH_GENERATION_NEUTRAL`;
* `DATA_LIMITED`;
* `REQUIRES_HUMAN_REVIEW`;
* `NOT_ASSESSED`.

Explicit non-scope:

* no Python implementation;
* no tests;
* no runtime behavior;
* no provider calls;
* no data writes;
* no generated artifacts;
* no raw SEC CompanyFacts fetching;
* no Source Refresh changes;
* no Source Context changes;
* no Fundamental Observation changes;
* no Derived Observation changes;
* no Recommendation Review behavior;
* no Portfolio Review behavior;
* no Delivery behavior;
* no Telegram;
* no reporting;
* no Decision Engine behavior;
* no BUY / SELL / HOLD;
* no target price;
* no score;
* no ranking;
* no rating;
* no conviction;
* no urgency;
* no tradeability;
* no allocation;
* no position sizing;
* no execution advice;
* no watchlist mutation;
* no portfolio mutation.

Acceptance criteria:

* Analysis Review job boundary is defined.
* Approved upstream observation families are defined.
* Approved upstream input formats are defined.
* Recommended Analysis Review output format is defined.
* Recommended Analysis Review persistence path is defined.
* Approved Analysis Review categories are defined.
* Approved Analysis Review states are defined.
* State semantics are documented.
* Approved message style is documented.
* Forbidden message style is documented.
* Provenance requirements are documented.
* Persistence requirements are documented.
* ME-AR02 implementation requirements are documented.
* Recommendation, portfolio, delivery, Telegram, reporting, and Decision Engine boundaries remain explicit.
* Sprint remains documentation/contract only.

Outcome: ME-AR01 defined the non-recommendation Analysis Review contract from ME-FO02 Fundamental Observations and ME-DO01 Derived Observations. The contract approves initial Analysis Review categories and states, defines provenance and persistence requirements, and prepares ME-AR02 implementation without introducing Python code, tests, runtime behavior, provider calls, data writes, Recommendation Review, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine authority.

### ME-AR02 — Implement Analysis Review from Fundamental and Derived Observations

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Analysis Review

Status: COMPLETED BY ME-AR02

Goal: Implement non-recommendation Analysis Review from approved ME-FO02 Fundamental Observations and ME-DO01 Derived Observations.

Scope:

* consume `SecCompanyFactsFundamentalObservationSet` objects from ME-FO02;
* consume `SecCompanyFactsDerivedCashGenerationObservationSet` objects from ME-DO01;
* emit `sec-companyfacts-analysis-review-v1` output;
* implement approved ME-AR01 Analysis Review categories;
* implement approved ME-AR01 Analysis Review states;
* validate upstream observation-set alignment;
* preserve upstream Fundamental Observation references;
* preserve upstream Derived Observation references;
* preserve upstream source values;
* preserve upstream derived values;
* preserve source context state;
* preserve source refresh metadata;
* preserve missingness and limitation states;
* emit data limitation review when upstream observations are limited;
* emit human review requirement when upstream observations are incomplete or limited;
* persist Analysis Review output under `data/market_engine/analysis_reviews/<analysis_review_run_id>/<ticker>/analysis_review.json`;
* refuse overwrite of existing Analysis Review output;
* add local tests using synthetic/temporary upstream observations only;
* document implementation and audit results.

Approved Analysis Review categories implemented:

* `SOURCE_AVAILABILITY_REVIEW`;
* `FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW`;
* `CASH_GENERATION_REVIEW`;
* `FREE_CASH_FLOW_REVIEW`;
* `DATA_LIMITATION_REVIEW`;
* `HUMAN_REVIEW_REQUIREMENT`.

Approved Analysis Review states implemented:

* `SOURCE_HEALTHY`;
* `SOURCE_LIMITED`;
* `OBSERVATIONS_COMPLETE`;
* `OBSERVATIONS_LIMITED`;
* `CASH_GENERATION_POSITIVE`;
* `CASH_GENERATION_NEGATIVE`;
* `CASH_GENERATION_NEUTRAL`;
* `DATA_LIMITED`;
* `REQUIRES_HUMAN_REVIEW`;
* `NOT_ASSESSED`.

Explicit non-scope:

* no raw SEC CompanyFacts fetching;
* no provider calls;
* no Source Refresh behavior changes;
* no Source Context behavior changes;
* no Fundamental Observation behavior changes;
* no Derived Observation behavior changes;
* no Recommendation Review behavior;
* no Portfolio Review behavior;
* no Delivery behavior;
* no Telegram;
* no reporting;
* no Decision Engine behavior;
* no BUY / SELL / HOLD;
* no target price;
* no score;
* no ranking;
* no rating;
* no conviction;
* no urgency;
* no tradeability;
* no allocation;
* no position sizing;
* no execution advice;
* no watchlist mutation;
* no portfolio mutation.

Acceptance criteria:

* Analysis Review output is emitted using `sec-companyfacts-analysis-review-v1`.
* Complete positive upstream observations produce non-recommendation Analysis Review.
* Negative cash generation is reviewed without recommendation authority.
* Neutral cash generation is reviewed without recommendation authority.
* Limited upstream observations emit data limitation review.
* Limited upstream observations emit human review requirement.
* Upstream Fundamental Observation references are preserved.
* Upstream Derived Observation references are preserved.
* Upstream source values and derived values are preserved.
* Source context metadata is preserved.
* Source refresh metadata is preserved.
* Upstream observation-set mismatch fails safely.
* Persistence writes JSON under the approved Analysis Review path.
* Persistence refuses overwrite.
* Recommendation, score, ranking, portfolio, delivery, Telegram, reporting, and Decision Engine authority are not emitted.
* Tests do not use live SEC/provider calls.
* Tests do not import legacy runtime modules.
* Documentation, backlog, and audit are updated.

Outcome: ME-AR02 implemented non-recommendation Analysis Review in `src/market_engine/analysis_review/`, with tests in `tests/market_engine/analysis_review/`. The implementation consumes ME-FO02 Fundamental Observations and ME-DO01 Derived Cash Generation Observations, emits approved ME-AR01 review categories and states, preserves upstream observation references, source values, derived values, Source Context state, source refresh metadata, missingness, and limitation states, and stays inside the ME-AR job family without introducing Recommendation Review, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine behavior.

### ME-RR01 — Define Recommendation Review contract from Analysis Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Recommendation Review

Status: COMPLETED BY ME-RR01

Goal: Define the Recommendation Review contract from approved Analysis Review output.

Scope:

* define the Recommendation Review contract boundary;
* define allowed input contract sec-companyfacts-analysis-review-v1;
* define recommended output contract sec-companyfacts-recommendation-review-v1;
* define recommended future output path data/market_engine/recommendation_reviews/<recommendation_review_run_id>/<ticker>/recommendation_review.json;
* define review states;
* define review categories;
* define allowed message semantics;
* define forbidden message semantics;
* define missing-data and numeric-zero requirements;
* define provenance requirements;
* define boundaries with Analysis Review, Portfolio Review, Decision Engine, Delivery, Reporting, Telegram, providers, and legacy runtime;
* define ME-RR02 implementation requirements.

Approved input contract:

* sec-companyfacts-analysis-review-v1.

Recommended output contract:

* sec-companyfacts-recommendation-review-v1.

Recommended future output path:

* data/market_engine/recommendation_reviews/<recommendation_review_run_id>/<ticker>/recommendation_review.json.

Approved review states:

* human_review_required;
* insufficient_evidence;
* blocked_by_missing_data;
* not_applicable.

Approved review categories:

* analysis_supportive_but_not_actionable;
* analysis_mixed_or_conflicted;
* analysis_blocked_by_missing_data;
* analysis_not_supported;
* input_contract_invalid.

Explicit non-scope:

* no Python implementation;
* no tests;
* no runtime behavior;
* no provider calls;
* no data writes;
* no generated artifacts;
* no portfolio review;
* no portfolio action;
* no allocation;
* no position sizing;
* no execution advice;
* no Telegram;
* no reporting;
* no delivery;
* no Decision Engine behavior;
* no BUY / SELL / HOLD as direct trading instructions;
* no score;
* no ranking;
* no conviction;
* no urgency;
* no tradeability;
* no watchlist mutation;
* no portfolio mutation.

Acceptance criteria:

* Recommendation Review job boundary is defined.
* Approved Analysis Review input contract is defined.
* Recommended Recommendation Review output contract is defined.
* Recommended future persistence path is defined.
* Review states are defined.
* Review categories are defined.
* Allowed message semantics are defined.
* Forbidden message semantics are defined.
* Missing-data rules are defined.
* Numeric-zero rules are defined.
* Provenance requirements are defined.
* Boundaries with Portfolio Review, Delivery, Reporting, Telegram, and Decision Engine remain explicit.
* ME-RR02 implementation requirements are documented.
* Sprint remains documentation/contract only.

Outcome: ME-RR01 defined Recommendation Review as a non-actionable, source-grounded, human-review routing layer from sec-companyfacts-analysis-review-v1. The contract approves initial review states and review categories, defines provenance and boundary requirements, and prepares ME-RR02 implementation without introducing Python code, tests, runtime behavior, provider calls, data writes, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine authority.

### ME-RR02 — Implement Recommendation Review from Analysis Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Recommendation Review

Status: COMPLETED BY ME-RR02

Goal: Implement the minimum viable non-actionable Recommendation Review builder from approved Analysis Review output.

Scope remained inside the ME-RR job family and did not introduce portfolio review, delivery, Telegram, reporting, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, tradeability, watchlist mutation, or portfolio mutation.

Implemented input contract:

* sec-companyfacts-analysis-review-v1.

Implemented output contract:

* sec-companyfacts-recommendation-review-v1.

Implemented runtime module:

* `src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py`.

Implemented package export:

* `src/market_engine/recommendation_review/__init__.py`.

Implemented tests:

* `tests/market_engine/recommendation_review/test_sec_companyfacts_recommendation_review.py`.

Implemented audit:

* `docs/market_engine/audits/me_rr02_recommendation_review_implementation_audit.md`.

Implemented review states:

* human_review_required;
* insufficient_evidence;
* blocked_by_missing_data;
* not_applicable.

Implemented review categories:

* analysis_supportive_but_not_actionable;
* analysis_mixed_or_conflicted;
* analysis_blocked_by_missing_data;
* analysis_not_supported;
* input_contract_invalid.

Implemented behavior:

* supportive Analysis Review input creates a non-actionable human-review candidate;
* limited Analysis Review input blocks Recommendation Review with explicit missing data;
* unsupported Analysis Review contracts fail closed;
* Recommendation Review JSON can be persisted under `data/market_engine/recommendation_reviews`;
* persistence refuses overwrite;
* normal review text does not emit action-authority terms;
* legacy `scripts` and `market_scanner` imports are not introduced.

Validation:

* targeted Recommendation Review tests passed: 7 passed;
* full Market Engine test suite passed: 136 passed.

Outcome: ME-RR02 implemented the first non-actionable SEC CompanyFacts Recommendation Review layer. The layer consumes `sec-companyfacts-analysis-review-v1`, emits `sec-companyfacts-recommendation-review-v1`, preserves upstream provenance and missing-data state, persists JSON safely, refuses overwrite, and keeps portfolio, delivery, Telegram, reporting, and Decision Engine authority out of scope.

### ME-RM01 — Align Market Engine roadmap and sprint sequence

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: Roadmap / Governance

Status: COMPLETED BY ME-RM01

Goal: Align the Market Engine roadmap and backlog after Recommendation Review implementation and preserve the required future sprint sequence.

Scope: Roadmap documentation, backlog sequence update, Setup Detection insertion before Portfolio Review, and governance rule for preserving future sprint sequence.

Outcome: ME-RM01 created `docs/market_engine/roadmap/market_engine_roadmap.md`, inserted Setup Detection before Portfolio Review, moved Portfolio Review after Setup Detection-aware Analysis Review and Recommendation Review work, and added the governance rule that future logical next sprints must be preserved in the backlog and roadmap when identified.

Insertion reason: Setup Detection was identified as a missing architectural layer between Derived Observations and downstream Analysis Review / Recommendation Review / Portfolio Review. Without this insertion, the roadmap would jump too quickly from Recommendation Review to Portfolio Review and skip a required pattern/setup layer.

## Completed Sprint

### ME-SD01 — Define Setup Detection contract

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Setup Detection

Status: COMPLETED BY ME-SD01

Goal: Define the contract for detecting patterns and setups from Fundamental Observations and Derived Observations.

Scope: Documentation-only contract sprint.

Implemented input contracts:

* `sec-companyfacts-fundamental-observations-v1`;
* `sec-companyfacts-derived-cash-generation-observations-v1`.

Implemented output contract:

* `sec-companyfacts-setup-detection-v1`.

Implemented documentation:

* `docs/market_engine/setup_detection/me_sd01_setup_detection_contract.md`.

Implemented audit:

* `docs/market_engine/audits/me_sd01_setup_detection_contract_audit.md`.

ME-SD01 defined:

* Setup Detection job-family boundary;
* setup definition;
* initial setup families;
* setup categories;
* setup states;
* required evidence model;
* missing-data handling;
* non-actionable boundary;
* provenance requirements;
* future ME-SD02 implementation requirements;
* future ME-SD02 persistence requirements;
* future ME-SD02 test requirements;
* relationship to Analysis Review, Recommendation Review, Portfolio Review, Decision Engine, and Delivery / Reporting.

ME-SD01 did not introduce Python code, tests, provider calls, data writes, BUY / SELL / HOLD action semantics, portfolio mutation, Decision Engine behavior, Telegram, reporting, recommendation authority, allocation, execution advice, or delivery behavior.

Outcome: ME-SD01 defined Setup Detection as the missing non-actionable pattern/setup layer between Derived Observations and Analysis Review. The contract allows future ME-SD02 implementation to detect structured setups from approved Fundamental Observations and Derived Cash Generation Observations while preserving provenance, missing-data state, source grounding, numeric-zero semantics, and authority boundaries.

### ME-SD02 — Implement first Setup Detection layer

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Setup Detection

Status: COMPLETED BY ME-SD02

Goal: Implement the first non-actionable Setup Detection builder from approved observation inputs.

Scope: Local synthetic tests only; no live providers; no portfolio mutation; no Decision Engine behavior; no BUY / SELL / HOLD action semantics.

Implemented runtime module:

* `src/market_engine/setup_detection/sec_companyfacts_setup_detection.py`

Implemented package export:

* `src/market_engine/setup_detection/__init__.py`

Implemented tests:

* `tests/market_engine/setup_detection/test_sec_companyfacts_setup_detection.py`

Implemented audit:

* `docs/market_engine/audits/me_sd02_setup_detection_implementation_audit.md`

Implemented input contracts:

* `sec-companyfacts-fundamental-observations-v1`;
* `sec-companyfacts-derived-cash-generation-observations-v1`.

Implemented output contract:

* `sec-companyfacts-setup-detection-v1`.

ME-SD02 implemented:

* Setup Detection runtime module under the active `market_engine` package;
* builder equivalent to `build_sec_companyfacts_setup_detection(...)`;
* output contract `sec-companyfacts-setup-detection-v1`;
* setup items using the categories and states defined by ME-SD01;
* source and derived observation references;
* explicit missing-data preservation;
* numeric-zero preservation;
* fail-closed behavior for unsupported input contracts;
* JSON persistence equivalent to `persist_sec_companyfacts_setup_detection(...)`;
* overwrite refusal for persisted setup detection output;
* local synthetic tests.

ME-SD02 tested:

* complete positive setup evidence produces setup detection output;
* partial evidence produces `setup_partially_detected`;
* missing required observations produce `setup_blocked_by_missing_data`;
* conflicted evidence produces `setup_conflicted`;
* unsupported input contract fails closed;
* numeric zero is preserved and not treated as missing;
* source and derived references are preserved;
* forbidden action-authority terms are not emitted in normal setup text;
* persistence writes JSON under temporary root;
* persistence refuses overwrite;
* no legacy `scripts` or old `market_scanner` imports are introduced.

ME-SD02 did not introduce:

* live provider calls;
* SEC or EDGAR calls;
* yfinance calls;
* production data writes;
* Analysis Review behavior changes;
* Recommendation Review behavior changes;
* Portfolio Review behavior;
* Decision Engine behavior;
* Telegram delivery;
* reporting output;
* BUY / SELL / HOLD action semantics;
* allocation;
* position sizing;
* execution advice;
* ranking;
* scoring;
* conviction scoring;
* urgency scoring;
* tradeability scoring.

Outcome: ME-SD02 implemented the first non-actionable Setup Detection layer in `src/market_engine/setup_detection/`, with tests in `tests/market_engine/setup_detection/`. The implementation consumes approved SEC CompanyFacts Fundamental Observations and Derived Cash Generation Observations, emits `sec-companyfacts-setup-detection-v1`, preserves source and derived observation references, preserves missing-data and numeric-zero semantics, implements JSON persistence under `data/market_engine/setup_detections/<run_id>/<ticker>/setup_detection.json`, refuses overwrite, and does not introduce Analysis Review, Recommendation Review, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine behavior.

### ME-AR03 — Extend Analysis Review contract for Setup Detection input

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Analysis Review

Status: COMPLETED BY ME-AR03

Goal: Define how Analysis Review can consume Setup Detection output without recommendation authority.

Scope: Documentation-only contract update.

Implemented contract:

* `docs/market_engine/analysis_review/me_ar03_setup_detection_input_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_ar03_setup_detection_input_contract_audit.md`

ME-AR03 defined:

* how `sec-companyfacts-setup-detection-v1` becomes an approved Analysis Review input;
* how Setup Detection evidence is referenced from Analysis Review items;
* how setup limitations and missing-data states are preserved;
* how Setup Detection categories map to Analysis Review categories;
* how Setup Detection states map to Analysis Review states;
* how conflicted, partial, blocked, and not-assessed setup evidence should be handled;
* how numeric zero remains present and must not be treated as missing;
* how Analysis Review remains non-recommendation and non-actionable;
* how ME-AR04 must implement Setup Detection-aware Analysis Review behavior.

ME-AR03 did not introduce Python code, tests, provider calls, data writes, Recommendation Review behavior, Portfolio Review behavior, Telegram, reporting, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability authority.

Outcome: ME-AR03 extended the Analysis Review contract so a future ME-AR04 implementation can consume `sec-companyfacts-setup-detection-v1` while preserving Analysis Review as descriptive, provenance-preserving, missing-data-aware, numeric-zero-safe, non-recommendation, and non-actionable.

### ME-AR04 — Implement Analysis Review consumption of Setup Detection

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Analysis Review

Status: COMPLETED BY ME-AR04

Goal: Implement Analysis Review support for Setup Detection input.

Scope: Local synthetic tests only; no provider calls; no Recommendation Review, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine behavior.

ME-AR04 must implement Analysis Review consumption of Setup Detection according to the ME-AR03 contract.

ME-AR04 must:

* consume `sec-companyfacts-setup-detection-v1`;
* preserve existing Fundamental Observation and Derived Observation behavior;
* validate input alignment across Fundamental Observations, Derived Observations, and Setup Detection;
* emit Setup Detection-aware Analysis Review items;
* preserve Setup Detection categories and states;
* preserve Setup Detection evidence;
* preserve Setup Detection limitations;
* preserve missing observations;
* preserve source and derived references;
* preserve numeric-zero semantics;
* preserve non-recommendation and non-actionable boundary markers;
* fail closed or emit controlled limitation output for unsupported Setup Detection input contracts;
* add local synthetic tests only.

ME-AR04 must test:

* complete Setup Detection input creates Setup Detection-aware Analysis Review;
* partial setup input creates partial Setup Detection-aware review;
* missing setup evidence creates blocked or data-limited review;
* conflicted setup input creates conflicted review and human-review routing;
* not-assessed setup input remains not assessed;
* unsupported Setup Detection input contract fails closed;
* numeric zero remains present and is not treated as missing;
* Setup Detection references are preserved;
* Fundamental Observation and Derived Observation references remain preserved;
* existing Analysis Review behavior is not broken;
* forbidden action-authority terms are not emitted;
* no legacy `scripts` or old `market_scanner` imports are introduced.

ME-AR04 must not introduce recommendation authority, portfolio mutation, delivery behavior, Telegram behavior, reporting behavior, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability authority.

Implemented runtime:

* `src/market_engine/analysis_review/sec_companyfacts_analysis_review.py`

Implemented tests:

* `tests/market_engine/analysis_review/test_sec_companyfacts_analysis_review.py`

Implemented documentation:

* `docs/market_engine/analysis_review/me_ar04_analysis_review_setup_detection_implementation.md`
* `docs/market_engine/audits/me_ar04_analysis_review_setup_detection_implementation_audit.md`

Outcome: ME-AR04 extended the existing `sec-companyfacts-analysis-review-v1` implementation with optional Setup Detection input. The implementation preserves existing ME-AR02 behavior without Setup Detection input, validates Setup Detection alignment and contract version, emits Setup Detection-aware Analysis Review items, preserves setup evidence, setup limitations, missing observations, source and derived references, numeric-zero semantics, and remains non-recommendation and non-actionable.

## Completed Sprint

### ME-RR03 — Extend Recommendation Review contract for Setup Detection-aware Analysis Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Recommendation Review

Status: COMPLETED BY ME-RR03

Goal: Define how Recommendation Review consumes Setup Detection-aware Analysis Review.

Scope: Documentation-only contract update.

Recommendation Review remains non-actionable.

Implemented contract:

* `docs/market_engine/recommendation_review/me_rr03_setup_detection_aware_analysis_review_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_rr03_setup_detection_aware_analysis_review_contract_audit.md`

ME-RR03 defined:

* how Setup Detection-aware Analysis Review becomes approved Recommendation Review input;
* how `sec-companyfacts-analysis-review-v1` remains the approved input contract;
* how setup-aware evidence is preserved in Recommendation Review provenance;
* how detected setup states route only to non-actionable human review;
* how partial setup states preserve uncertainty;
* how conflicted setup states preserve conflict;
* how blocked setup states preserve explicit missing-data blocking;
* how not-assessed setup states remain not assessed or insufficient evidence;
* how not-detected setup states must not become negative recommendations;
* how missing setup data remains explicit;
* how numeric zero remains present and must not be treated as missing;
* how Recommendation Review remains downstream of Analysis Review;
* ME-RR04 implementation requirements.

ME-RR03 did not introduce Python code, tests, provider calls, data writes, Recommendation Review runtime changes, Analysis Review runtime changes, Setup Detection runtime changes, Portfolio Review behavior, Telegram, reporting, delivery, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability authority.

Outcome: ME-RR03 extended the Recommendation Review contract on paper so ME-RR04 can later implement consumption of Setup Detection-aware `sec-companyfacts-analysis-review-v1` output. The contract preserves setup-aware evidence and provenance, keeps missing setup data explicit, preserves numeric-zero semantics, routes setup states only to non-actionable human-review or blocked/insufficient-evidence outcomes, and prevents action, portfolio, delivery, ranking, scoring, or Decision Engine authority.

## Recommended Next Sprint

### ME-RR04 — Implement Setup Detection-aware Recommendation Review behavior

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Recommendation Review

Status: COMPLETED BY ME-RR04

Goal: Implement Setup Detection-aware Recommendation Review behavior.

Scope: Non-actionable Recommendation Review only; no action authority, portfolio mutation, delivery behavior, Telegram, reporting, or Decision Engine behavior.

ME-RR04 must implement Setup Detection-aware Recommendation Review only after ME-RR03 defines the contract.

ME-RR04 must:

* consume only validated `sec-companyfacts-analysis-review-v1`;
* preserve existing ME-RR02 behavior when Setup Detection-aware Analysis Review items are absent;
* detect Setup Detection-aware Analysis Review items where present;
* preserve setup-aware provenance;
* preserve setup categories and states;
* preserve setup evidence and limitations;
* preserve missing setup observations;
* preserve source and derived references;
* preserve numeric-zero semantics;
* preserve non-actionable boundary markers;
* route detected setup evidence to human review only;
* route partial setup evidence to human review with explicit uncertainty;
* route conflicted setup evidence to human review with explicit conflict;
* route blocked setup evidence to blocked-by-missing-data;
* route not-assessed setup evidence to insufficient-evidence or blocked routing;
* fail closed for unsupported Analysis Review input contracts;
* add local synthetic tests only;
* avoid live provider calls;
* avoid production data writes;
* avoid legacy `scripts` or old `market_scanner` imports.

ME-RR04 must not introduce portfolio mutation, watchlist mutation, delivery behavior, Telegram behavior, reporting behavior, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability authority.

Implemented runtime:

* `src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py`

Implemented tests:

* `tests/market_engine/recommendation_review/test_sec_companyfacts_recommendation_review.py`

Implemented documentation:

* `docs/market_engine/recommendation_review/me_rr04_setup_detection_aware_recommendation_review_implementation.md`
* `docs/market_engine/audits/me_rr04_setup_detection_aware_recommendation_review_implementation_audit.md`

Outcome: ME-RR04 implemented Recommendation Review consumption of Setup Detection-aware `sec-companyfacts-analysis-review-v1` output. The implementation preserves existing ME-RR02 behavior when setup-aware fields are absent, preserves setup categories, setup states, setup evidence, setup limitations, missing setup observations, source and derived references, numeric-zero semantics, and routes setup-aware evidence only to non-actionable human-review, blocked-by-missing-data, or insufficient-evidence Recommendation Review outcomes.

## Completed Sprint

### ME-PR01 — Define Portfolio Review contract from Recommendation Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Portfolio Review

Status: COMPLETED BY ME-PR01

Goal: Define the Portfolio Review contract after Setup Detection-aware Recommendation Review exists.

Scope: Documentation-only contract sprint.

Implemented contract:

* `docs/market_engine/portfolio_review/me_pr01_portfolio_review_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_pr01_portfolio_review_contract_audit.md`

Implemented backlog update:

* `docs/market_engine/backlog/me_pr01_backlog_update.md`

ME-PR01 defined:

* Portfolio Review job-family boundary;
* approved Recommendation Review input requirements;
* required explicit portfolio-context input family;
* position, exposure, concentration, and fit review semantics;
* allowed portfolio-review states;
* allowed portfolio-review categories;
* missing-data and stale-data rules;
* numeric-zero preservation rules;
* provenance requirements;
* authority boundary between Portfolio Review and Decision Engine;
* ME-PR02 implementation requirements.

Approved Recommendation Review input contract:

* `sec-companyfacts-recommendation-review-v1`

Approved portfolio-context input family:

* `market-engine-portfolio-context-v1`

Approved Portfolio Review output contract:

* `sec-companyfacts-portfolio-review-v1`

Outcome: ME-PR01 defined Portfolio Review as a non-actionable, explicit-portfolio-context-dependent review layer downstream of Recommendation Review and upstream of Decision Engine handoff. The contract preserves Recommendation Review provenance, Setup Detection-aware provenance when present, portfolio-context evidence, missing portfolio-context data, stale portfolio-context data, and numeric-zero semantics.

ME-PR01 did not introduce Python code, tests, runtime behavior, provider calls, broker calls, data writes, generated artifacts, portfolio mutation, watchlist mutation, Telegram, reporting, delivery, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation execution, target weights, order generation, position sizing instructions, ranking, scoring, conviction, urgency, or tradeability authority.

## Recommended Next Sprint

### ME-PR02 — Implement Portfolio Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Portfolio Review

Status: COMPLETED BY ME-PR02

Goal: Implement Portfolio Review after the ME-PR01 contract definition.

Scope: Non-actionable Portfolio Review only. No portfolio mutation, no broker calls, no Decision Engine behavior, no delivery behavior, no Telegram, no reporting, no BUY / SELL / HOLD action semantics, no allocation execution, no target weights, no order generation, no ranking, no scoring, no conviction, no urgency, and no tradeability authority.

ME-PR02 must implement Portfolio Review only after ME-PR01 defines the contract.

ME-PR02 must:

* consume approved `sec-companyfacts-recommendation-review-v1` input;
* consume explicitly supplied `market-engine-portfolio-context-v1` input;
* emit `sec-companyfacts-portfolio-review-v1`;
* preserve Recommendation Review provenance;
* preserve Setup Detection-aware provenance when present;
* preserve portfolio-context provenance;
* preserve missing portfolio-context data explicitly;
* preserve stale portfolio-context data explicitly;
* preserve numeric-zero semantics;
* produce non-actionable position, exposure, concentration, and portfolio-fit review output;
* fail closed for unsupported input contracts;
* add local synthetic tests only;
* avoid live provider calls;
* avoid broker calls;
* avoid production data writes;
* avoid legacy `scripts` or old `market_scanner` imports.

ME-PR02 must preserve Decision Engine authority and must not execute allocations, orders, rebalances, alerts, reports, delivery actions, portfolio mutations, or watchlist mutations.

Implemented runtime:

* `src/market_engine/portfolio_review/sec_companyfacts_portfolio_review.py`
* `src/market_engine/portfolio_review/__init__.py`

Implemented tests:

* `tests/market_engine/portfolio_review/test_sec_companyfacts_portfolio_review.py`

Implemented documentation:

* `docs/market_engine/portfolio_review/me_pr02_portfolio_review_implementation.md`
* `docs/market_engine/audits/me_pr02_portfolio_review_implementation_audit.md`

Outcome: ME-PR02 implemented non-actionable Portfolio Review from validated `sec-companyfacts-recommendation-review-v1` input and explicitly supplied `market-engine-portfolio-context-v1` input. The implementation emits `sec-companyfacts-portfolio-review-v1`, preserves Recommendation Review provenance and Setup Detection-aware provenance when present, preserves portfolio-context provenance, missing and stale portfolio-context markers, numeric-zero semantics, and produces review-only position, exposure, concentration, portfolio-fit, data-limitation, and downstream-handoff-readiness items.

Possible future Portfolio Review follow-up candidate: `ME-PR03 — Define approved portfolio context source and persistence contract`. This candidate is not inserted ahead of ME-DE01 because ME-PR02 did not uncover a blocker; it should be added formally only if a later Decision Engine handoff or portfolio-context sprint requires persisted portfolio-context sourcing beyond caller-supplied context.

## Completed Sprint

### ME-DE01 — Define Decision Engine handoff contract

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Decision Engine handoff

Status: COMPLETED BY ME-DE01

Goal: Define the boundary between Market Engine review output and actual decision/action authority.

Scope: Documentation-only contract sprint unless explicitly re-scoped.

ME-DE01 defined:

* approved upstream input from Portfolio Review;
* handoff payload requirements;
* what the Market Engine may request from the Decision Engine;
* what only the Decision Engine may decide;
* action/allocation authority boundaries;
* fail-closed rules;
* audit and traceability requirements;
* ME-DE02 implementation requirements.

Implemented contract:

* `docs/market_engine/decision_engine/me_de01_decision_engine_handoff_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_de01_decision_engine_handoff_contract_audit.md`

Outcome: ME-DE01 defined `market-engine-decision-engine-handoff-v1` as the future handoff payload downstream of `sec-companyfacts-portfolio-review-v1`. The contract defines Portfolio Review eligibility, blocked handoff states, fail-closed rules, numeric-zero preservation, provenance requirements, prohibited payload fields, and ME-DE02 implementation requirements while preserving Decision Engine as the only future action and allocation authority.

ME-DE01 did not introduce Python code, tests, provider calls, data writes, Telegram, reporting, delivery behavior, portfolio mutation, BUY / SELL / HOLD execution semantics, allocation execution, order generation, or live Decision Engine behavior.

## Completed Sprint

### ME-DE02 — Implement controlled Decision Engine handoff

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Decision Engine handoff

Status: COMPLETED BY ME-DE02

Goal: Implement controlled handoff according to the ME-DE01 contract.

Scope: Must preserve Decision Engine as the only action/allocation authority.

ME-DE02 must implement handoff behavior only after ME-DE01 defines the contract.

ME-DE02 must not bypass Portfolio Review, Recommendation Review, Analysis Review, Setup Detection, or authority boundaries.

ME-DE02 must:

* consume approved `sec-companyfacts-portfolio-review-v1` input only;
* emit `market-engine-decision-engine-handoff-v1`;
* validate Portfolio Review contract and version;
* validate ticker identity;
* validate portfolio-context version and state;
* validate Portfolio Review handoff-readiness evidence;
* preserve Recommendation Review and Setup Detection-aware provenance when present;
* preserve missing-data markers;
* preserve stale-data markers;
* preserve numeric-zero semantics;
* produce only a handoff-readiness payload;
* avoid Decision Engine decisions, actions, allocation, ranking, scoring, execution, delivery, Telegram, and reporting behavior.

Implemented runtime:

* `src/market_engine/decision_engine_handoff/sec_companyfacts_handoff.py`
* `src/market_engine/decision_engine_handoff/__init__.py`

Implemented tests:

* `tests/market_engine/decision_engine_handoff/test_sec_companyfacts_handoff.py`

Implemented documentation:

* `docs/market_engine/decision_engine/me_de02_decision_engine_handoff_implementation.md`
* `docs/market_engine/audits/me_de02_decision_engine_handoff_implementation_audit.md`

Outcome: ME-DE02 implemented deterministic `market-engine-decision-engine-handoff-v1` construction from approved `sec-companyfacts-portfolio-review-v1` input. Eligible Portfolio Review output produces `ready_for_decision_engine_review`; ineligible input produces explicit blocked handoff states with deterministic blocked reasons. The implementation preserves Portfolio Review, portfolio-context, Recommendation Review, Analysis Review, Setup Detection-aware, missing-data, stale-data, and numeric-zero evidence without introducing Decision Engine action or allocation authority.

ME-DE02 did not introduce provider calls, broker calls, live data access, portfolio writes, watchlist writes, Telegram, reporting, delivery behavior, Decision Engine runtime decisions, trade instructions, allocation advice, target weights, order generation, position sizing, urgency, conviction, tradeability, ranking, scoring, or execution advice.

## Completed Sprint

### ME-DL01 — Define Delivery / Reporting contract

Owner roles: Product Owner / Operator / User / Technical Architect / QA Lead / Governance Auditor

Job family: Delivery / Reporting

Status: COMPLETED BY ME-DL01

Goal: Define how approved outputs may be delivered or reported.

Scope: Documentation-only contract sprint unless explicitly re-scoped.

ME-DL01 defined:

* approved upstream input requirements;
* delivery eligibility;
* reporting eligibility;
* Telegram/reporting boundaries;
* user-facing output contract;
* audit and traceability requirements;
* fail-closed delivery rules;
* ME-DL02 implementation requirements.

Implemented contract:

* `docs/market_engine/delivery_reporting/me_dl01_delivery_reporting_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_dl01_delivery_reporting_contract_audit.md`

Outcome: ME-DL01 defined `market-engine-delivery-report-v1` as the future Delivery / Reporting payload downstream of `market-engine-decision-engine-handoff-v1`. The contract defines approved input, delivery states, allowed reporting categories, forbidden reporting behavior, presentation rules, blocked/upstream handling, missing-data handling, stale-data handling, numeric-zero safety, provenance preservation, and ME-DL02 implementation requirements.

ME-DL01 did not introduce Python code, tests, provider calls, data writes, portfolio mutation, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, tradeability authority, Telegram delivery, email delivery, broker integration, scheduler behavior, report generation, or user-facing alerts.

## Completed Sprint

### ME-DL02 — Implement controlled Delivery / Reporting output

Owner roles: Product Owner / Operator / User / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Delivery / Reporting

Status: COMPLETED BY ME-DL02

Goal: Implement the Delivery / Reporting contract defined by ME-DL01.

Scope: Must not bypass Recommendation Review, Portfolio Review, or Decision Engine handoff authority boundaries.

ME-DL02 must implement delivery/reporting only after ME-DL01 defines the contract.

ME-DL02 must:

* consume only approved `market-engine-decision-engine-handoff-v1` payloads;
* emit `market-engine-delivery-report-v1`;
* preserve blocked upstream states as blocked;
* preserve missing-data markers;
* preserve stale-data markers;
* preserve numeric-zero semantics;
* preserve upstream provenance;
* emit only allowed non-actionable reporting categories;
* use local synthetic tests only;
* avoid provider calls;
* avoid Telegram, email, broker, portfolio, watchlist, scheduler, and production report writes;
* avoid ranking, conviction, urgency, target-price, BUY / SELL / HOLD, allocation, or execution semantics.

Implemented runtime:

* `src/market_engine/delivery_reporting/sec_companyfacts_delivery_report.py`
* `src/market_engine/delivery_reporting/__init__.py`

Implemented tests:

* `tests/market_engine/delivery_reporting/test_sec_companyfacts_delivery_report.py`

Implemented documentation:

* `docs/market_engine/delivery_reporting/me_dl02_delivery_reporting_implementation.md`
* `docs/market_engine/audits/me_dl02_delivery_reporting_implementation_audit.md`

Outcome: ME-DL02 implemented deterministic `market-engine-delivery-report-v1` construction from approved `market-engine-decision-engine-handoff-v1` input. The implementation preserves blocked upstream states, missing-data markers, stale-data markers, numeric-zero evidence, and upstream provenance while emitting only non-actionable reporting payload sections.

ME-DL02 did not introduce provider calls, live market data calls, Telegram delivery, email delivery, broker integration, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, trade instructions, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Completed Sprint

### ME-RUN05 — Implement local dry-run artifact persistence

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN05

Goal: Implement local dry-run artifact persistence for generated Market Engine runtime artifacts.

Scope: Optional local non-production JSON persistence for already-built `market-engine-end-to-end-dry-run-v1` payloads only.

ME-RUN05 implemented:

* local dry-run artifact format: `market-engine-local-dry-run-artifact-v1`;
* local dry-run artifact manifest format: `market-engine-local-dry-run-artifact-manifest-v1`;
* approved path category: `artifacts/market_engine/dry_runs/`;
* explicit `--write-local-artifact` command behavior;
* deterministic artifact metadata through caller-supplied artifact timestamp;
* safe path validation;
* overwrite refusal by default;
* stable, human-readable JSON serialization;
* local synthetic tests only.

Implemented runtime:

* `src/market_engine/run/local_dry_run_artifacts.py`
* `src/market_engine/run/end_to_end_dry_run_command.py`
* `src/market_engine/run/__init__.py`

Implemented tests:

* `tests/market_engine/run/test_local_dry_run_artifacts.py`

Implemented documentation:

* `docs/market_engine/run/me_run05_local_dry_run_artifact_persistence_implementation.md`
* `docs/market_engine/audits/me_run05_local_dry_run_artifact_persistence_audit.md`

Outcome: ME-RUN05 lets local dry-run executions persist deterministic, inspectable, non-production JSON artifacts while preserving the stdout-only default command behavior. The writer preserves dry-run contract identity, missing-data markers, stale-data markers, blocked states, blocked reasons, numeric-zero values, provenance, delivery report references, forbidden-side-effect confirmation, and authority-boundary confirmation.

ME-RUN05 did not introduce provider calls, live market data calls, Telegram delivery, email delivery, broker integration, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, trade instructions, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Completed Sprint

### ME-RUN10 — Implement cached-source end-to-end local execution path

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN10

Goal: Implement the cached-source end-to-end local execution path defined by ME-RUN09.

Scope: Local cached-source input loading, command input mode, fail-closed validation, downstream contract construction through approved builders, tests, documentation, and audit only.

ME-RUN10 implemented:

* `cached_source_snapshot` dry-run input mode;
* `market-engine-cached-source-local-execution-input-v1` wrapper support;
* cached SEC CompanyFacts source snapshot path containment validation;
* cached source snapshot to Source Context construction;
* downstream contract construction through the implemented Market Engine chain;
* explicit local portfolio-context input support;
* optional local dry-run artifact writing through the existing `--write-local-artifact` flag;
* local synthetic tests only.

Implemented runtime:

* `src/market_engine/run/cached_source_execution.py`
* `src/market_engine/run/end_to_end_dry_run.py`
* `src/market_engine/run/end_to_end_dry_run_command.py`
* `src/market_engine/run/local_dry_run_artifacts.py`
* `src/market_engine/run/__init__.py`

Implemented tests:

* `tests/market_engine/run/test_me_run10_cached_source_local_execution.py`

Implemented documentation:

* `docs/market_engine/run/me_run10_cached_source_local_execution_implementation.md`
* `docs/market_engine/audits/me_run10_cached_source_local_execution_implementation_audit.md`
* `docs/market_engine/backlog/me_run10_cached_source_local_execution_backlog_entry.md`
* `docs/market_engine/roadmap/me_run10_cached_source_local_execution_roadmap_entry.md`

Outcome: ME-RUN10 proves Market Engine can run the local dry-run chain from an already-existing cached SEC CompanyFacts source snapshot and explicitly supplied local portfolio context without live provider calls or production side effects. The final output remains `market-engine-end-to-end-dry-run-v1`, and artifact persistence remains opt-in through the existing local artifact path.

ME-RUN10 did not introduce provider calls, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, all-ticker production runs, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Completed Sprint

### ME-RUN11 — Run cached-source local execution against a broader deterministic ticker bundle

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN11

Goal: Validate cached-source local execution against a small deterministic ticker bundle.

Scope: Local synthetic cached-source fixtures, ticker-by-ticker command coverage, opt-in artifact validation, fail-closed malformed snapshot validation, tests, documentation, and audit only.

ME-RUN11 implemented:

* deterministic bundle coverage for `NVDA`, `MSFT`, and `AMD`;
* per-ticker validation of `market-engine-end-to-end-dry-run-v1`;
* per-ticker validation of `cached_source_snapshot`;
* cached-source provenance checks;
* source refresh snapshot ID provenance checks;
* numeric-zero source and portfolio-context evidence checks;
* artifact writing default-off validation across bundle runs;
* opt-in artifact writing validation for one selected ticker;
* malformed cached-source fail-closed validation.

Implemented tests:

* `tests/market_engine/run/test_me_run11_cached_source_ticker_bundle_execution.py`

Implemented documentation:

* `docs/market_engine/run/me_run11_cached_source_ticker_bundle_execution.md`
* `docs/market_engine/audits/me_run11_cached_source_ticker_bundle_execution_audit.md`
* `docs/market_engine/backlog/me_run11_cached_source_ticker_bundle_execution_backlog_entry.md`
* `docs/market_engine/roadmap/me_run11_cached_source_ticker_bundle_execution_roadmap_entry.md`

Outcome: ME-RUN11 proves the ME-RUN10 cached-source local execution path can run against a small deterministic ticker bundle by invoking the approved command path ticker-by-ticker. The sprint does not add a broad batch runner or production execution contract. The final per-ticker output remains `market-engine-end-to-end-dry-run-v1`, and artifact persistence remains opt-in through `--write-local-artifact`.

ME-RUN11 did not introduce provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, all-ticker production execution, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Future Sprint Candidates

Recommended next sprint after ME-RUN11:

```text
ME-RUN12 - Define safe all-ticker cached-source batch dry-run contract
```

Rationale: ME-RUN11 validates a small deterministic per-ticker bundle. Any broader cached-source batch behavior should be contract-defined before implementation so that production boundaries, cached-source discovery, artifact semantics, failure isolation, and operator visibility remain explicit.

## Completed Sprint

### ME-RUN13 - Implement safe all-ticker cached-source batch dry-run behavior

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN13

Goal: Implement the safe local cached-source batch dry-run path defined by ME-RUN12.

Scope: Local cached-source batch dry-run implementation, deterministic local tests, documentation, and audit only.

ME-RUN13 implemented:

* batch contract: `market-engine-cached-source-batch-dry-run-v1`;
* per-ticker output preservation as `market-engine-end-to-end-dry-run-v1`;
* deterministic cached-source discovery under an explicit local root;
* explicit requested ticker support;
* deterministic cached ticker discovery mode;
* missing cached-source ticker blocking;
* invalid cached-source ticker blocking;
* unsupported cached-source ticker blocking;
* ambiguous cached-source ticker blocking;
* downstream contract failure isolation;
* unexpected local error isolation;
* batch counts and per-ticker result summaries;
* numeric-zero evidence preservation;
* opt-in local batch artifact writing;
* overwrite protection for batch artifacts.

Implemented runtime:

* `src/market_engine/run/cached_source_batch_execution.py`
* `src/market_engine/run/__init__.py`

Implemented tests:

* `tests/market_engine/run/test_me_run13_cached_source_batch_dry_run.py`

Implemented documentation:

* `docs/market_engine/run/me_run13_safe_all_ticker_cached_source_batch_dry_run_implementation.md`
* `docs/market_engine/audits/me_run13_safe_all_ticker_cached_source_batch_dry_run_implementation_audit.md`

Outcome: ME-RUN13 implements the ME-RUN12 contract as a local cached-source batch wrapper over approved per-ticker dry-runs. It preserves per-ticker failure isolation, deterministic counts, local-only provenance, numeric-zero evidence, opt-in artifact behavior, and non-actionable boundaries.

ME-RUN13 did not introduce live provider calls, SEC/EDGAR fetches, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Historical Implementation Candidate

### ME-RUN14 - Add cached-source batch dry-run command interface

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: HISTORICAL CANDIDATE SUPERSEDED BY COMPLETED RUN COMMAND WORK

Goal: Add a narrow operator-facing command interface for the ME-RUN13 cached-source batch dry-run runtime behavior.

Rationale: ME-RUN13 implements the safe batch behavior as a runtime function and artifact writer. A separate sprint should add any operator-facing command interface so command arguments, terminal output, artifact flags, and failure messages remain explicit and reviewable.

Scope: Command interface, local argument parsing, terminal JSON output, opt-in artifact wiring, deterministic local tests, documentation, and audit only unless explicitly re-scoped.

Non-goals: no provider refresh, live market data, external API calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, new financial logic, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Completed Sprint

### ME-UNI02 - Implement canonical ticker universe loading and validation

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI02

Goal: implement the canonical ticker universe loader and validation layer defined by ME-UNI01.

Scope: loader, validator, typed result models, deterministic selection, synthetic tests, documentation, backlog, roadmap and audit only.

ME-UNI02 implemented:

* canonical contract version: `market-engine-canonical-ticker-universe-v1`;
* canonical default path: `data/market_engine/ticker_universe/ticker_universe.csv`;
* explicit path override support;
* required-column validation;
* required field validation;
* allowed-value validation;
* ticker trim and uppercase normalization only;
* duplicate normalized ticker and market rejection;
* active cached-source default selection;
* explicit inactive row inclusion when requested;
* optional metadata preservation;
* deterministic ordering;
* operator-readable validation errors.

Implemented runtime:

```text
src/market_engine/ticker_universe/
```

Implemented tests:

```text
tests/market_engine/ticker_universe/test_canonical_ticker_universe.py
```

Implemented documentation:

```text
docs/market_engine/ticker_universe/me_uni02_canonical_ticker_universe_loader_implementation.md
docs/market_engine/audits/me_uni02_canonical_ticker_universe_loader_audit.md
```

Outcome: the canonical ticker universe can be loaded and validated deterministically before downstream RUN consumption.

ME-UNI02 did not introduce provider calls, live network calls, source refresh jobs, batch execution, Telegram behavior, email delivery, broker integration, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next Canonical-Universe RUN Candidate

### ME-RUN16 - Execute first real cached-source batch dry-run using canonical ticker universe

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: CANDIDATE AFTER ME-UNI02

Goal: consume the ME-UNI02 canonical ticker universe loader in the cached-source batch dry-run path and execute the first real cached-source batch dry-run using the canonical universe.

Scope: cached-source/local-only RUN integration, canonical universe visibility, fail-closed invalid-universe behavior, local tests, documentation and audit only unless explicitly re-scoped.

ME-RUN16 must remain blocked from provider refresh, live data calls, Telegram delivery, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability and execution advice.

## Completed Sprint

### ME-RUN16 - Execute first real cached-source batch dry-run using canonical ticker universe

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED WITH BLOCKED TICKER OUTCOME BY ME-RUN16

Goal: execute the first cached-source batch dry-run selected from the canonical ticker universe.

Outcome:

* canonical universe loaded from `data/market_engine/ticker_universe/ticker_universe.csv`;
* 14 canonical rows loaded;
* 13 active `cached_source_only` tickers selected;
* SMCI excluded because `source_policy=manual_review_only`;
* all 13 selected tickers returned `blocked_missing_cached_source`;
* no provider or live data fallback occurred;
* generated local batch manifest under `artifacts/market_engine/...`, not committed.

Implemented runtime change:

```text
src/market_engine/run/cached_source_batch_dry_run_command.py
```

Implemented tests:

```text
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

Implemented documentation:

```text
docs/market_engine/run/me_run16_first_canonical_universe_cached_source_batch_dry_run_execution.md
docs/market_engine/audits/me_run16_first_canonical_universe_cached_source_batch_dry_run_audit.md
docs/market_engine/backlog/me_run16_first_canonical_universe_cached_source_batch_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run16_first_canonical_universe_cached_source_batch_dry_run_roadmap_entry.md
```

ME-RUN16 did not introduce provider calls, live data calls, Telegram delivery, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next Source Refresh Candidate

### ME-SR02 - Produce bounded canonical-universe SEC CompanyFacts cached source snapshots

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh

Status: CANDIDATE AFTER ME-RUN16

Goal: produce or validate bounded local SEC CompanyFacts cached source snapshots for the canonical universe so a later RUN sprint can execute downstream dry-runs from real cached source evidence.

Rationale: ME-RUN16 proved canonical-universe RUN selection and fail-closed behavior. It also showed that this checkout has no cached source snapshots under `data/market_engine/source_snapshots`, so every selected ticker blocks before downstream dry-run execution.

Scope: Source Refresh only. No portfolio writes, watchlist writes, Telegram delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Sprint

### ME-RUN24 - Non-production portfolio-context fixture for expanded scans

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN24

Goal: add an explicit non-production portfolio-context fixture path for expanded cached-source scans so ME-RUN23-style `supported_cached` tickers can pass portfolio context into Portfolio Review without broker data, live portfolio data, or portfolio/watchlist mutations.

Outcome:

* reused the existing `market-engine-local-portfolio-context-batch-v1` contract;
* added `--non-production-portfolio-context-fixture <path>` to the expanded supported-universe cached-source scan command;
* kept default behavior unchanged when the fixture argument is omitted;
* added output provenance for `portfolio_context_source=absent` and `portfolio_context_source=non_production_fixture`;
* preserved no broker or live portfolio access and no portfolio/watchlist mutation confirmations;
* added fail-closed tests for missing, malformed, and unsupported fixture input.

Implemented documentation:

```text
docs/market_engine/run_reports/me_run24_non_production_portfolio_context_fixture_expanded_scans.md
docs/market_engine/audits/me_run24_non_production_portfolio_context_fixture_expanded_scans_audit.md
docs/market_engine/backlog/me_run24_non_production_portfolio_context_fixture_expanded_scans_backlog_entry.md
docs/market_engine/roadmap/me_run24_non_production_portfolio_context_fixture_expanded_scans_roadmap_entry.md
```

ME-RUN24 does not canonicalize the universe, add provider calls, use broker APIs, mutate portfolio or watchlist state, add Telegram or delivery side effects, add Decision Engine action authority, or add trading/action semantics.

## Completed Sprint

### ME-SR07 - Cached-source snapshot acquisition plan for missing expanded universe entries

Owner roles: Product Owner / Operator / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR07

Goal: prepare a precise acquisition plan for cached-source snapshots needed by expanded-universe entries that are currently missing usable cached input data.

Outcome:

* documented the current expanded-universe coverage baseline from `data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv` and `data/market_engine/source_snapshots`;
* inventoried 53 expanded Professional Swing Universe entries: 12 `supported_cached`, 38 `missing_snapshot`, and 3 `manual_review_only`;
* defined required source families for the current SEC CompanyFacts cached-source path;
* defined allowed and disallowed future acquisition modes;
* defined required acquisition metadata and validation gates;
* documented that missing, partial, stale, malformed, unsupported, ambiguous, or manual-review-only data cannot be silently promoted to supported.

No snapshots were acquired. No runtime provider or live-fetch behavior was added. No unavailable data was marked acquired or supported.

Implemented documentation:

```text
docs/market_engine/audits/me_sr07_cached_source_snapshot_acquisition_plan.md
docs/market_engine/backlog/me_sr07_cached_source_snapshot_acquisition_plan_backlog_entry.md
docs/market_engine/roadmap/me_sr07_cached_source_snapshot_acquisition_plan_roadmap_entry.md
```

Follow-up sprint candidates:

* ME-SR08 - Define cached-source snapshot acquisition manifest contract.
* ME-SR09 - Implement missing expanded-universe snapshot coverage inventory command.
* ME-SR10 - Implement manual cached-source snapshot staging validator.
* ME-SR11 - Implement approved bounded acquisition or import workflow.
* ME-SR12 - Define non-US ticker source-family and source-mapping governance contract.
* ME-RUN25 - Rerun expanded cached-source coverage audit after staged snapshots exist.

ME-SR12 is future work only. It must define how non-US tickers, ADRs, foreign listings, dual listings, and `needs_source_mapping` entries can be admitted into cached-source coverage through explicit source-family rules and source identity mapping. It must define requirements for ticker, exchange/listing context, company identity, source entity identifier, and CIK only when applicable. It must cover entries such as ASML, NVO, RHM, RR, ADYEN, and similar future entries without acquiring snapshots, implementing provider access, or marking non-US tickers supported merely because a current classifier can load a snapshot.

## Completed Sprint

### ME-SR08 - Cached-source snapshot acquisition manifest contract

Owner roles: Product Owner / Operator / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR08

Goal: define the cached-source snapshot acquisition manifest contract before any acquisition, staging, validation, or import implementation begins.

Outcome:

* defined snapshot-level acquisition manifest scope;
* defined optional batch-level acquisition manifest scope;
* specified required manifest fields, field semantics, allowed values, validation behavior, governance constraints, local artifact relationships, and failure modes;
* included syntactically valid JSON examples for snapshot and batch manifests;
* aligned follow-up implementation candidates with the existing ME-SR07 sequence.

ME-SR08 is docs-only. No snapshots were acquired. No runtime acquisition, provider access, live-fetch behavior, dry-run behavior, Decision Engine behavior, broker behavior, Telegram behavior, portfolio/watchlist behavior, production writes, or action authority were added.

Implemented documentation:

```text
docs/market_engine/audits/me_sr08_cached_source_snapshot_acquisition_manifest_contract.md
docs/market_engine/backlog/me_sr08_cached_source_snapshot_acquisition_manifest_contract_backlog_entry.md
docs/market_engine/roadmap/me_sr08_cached_source_snapshot_acquisition_manifest_contract_roadmap_entry.md
```

Follow-up implementation candidates:

* ME-SR09 - Implement missing expanded-universe snapshot coverage inventory command.
* ME-SR10 - Implement manual cached-source snapshot staging validator.
* ME-SR11 - Implement approved bounded acquisition or import workflow.
* ME-SR12 - Define non-US ticker source-family and source-mapping governance contract.
* ME-RUN25 - Rerun expanded cached-source coverage audit after staged snapshots exist.

Implementation must not bypass ME-SR08 governance constraints.

## Completed Sprint

### ME-SR09 - Cached-source snapshot inventory command

Owner roles: Product Owner / Operator / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR09

Goal: implement the first local deterministic inventory command for cached-source snapshot acquisition manifests.

Outcome:

* added local cached-source snapshot inventory logic;
* added module command `market_engine.source_refresh.cached_source_snapshot_inventory_command`;
* defined JSON report format `market-engine-cached-source-snapshot-inventory-v1`;
* reports usable, unusable, missing-manifest, malformed-manifest, unknown-format, missing-payload, stale, and synthetic/test-fixture states;
* added deterministic fixture-based tests.

ME-SR09 did not acquire snapshots, stage snapshots, implement provider access, fetch live data, call SEC/EDGAR/yfinance, mutate portfolio/watchlist state, send Telegram output, write production data, modify cached-source dry-run semantics, change Decision Engine behavior, or add action authority.

Implemented documentation:

```text
docs/market_engine/audits/me_sr09_cached_source_snapshot_inventory_command_audit.md
docs/market_engine/backlog/me_sr09_cached_source_snapshot_inventory_command_backlog_entry.md
docs/market_engine/roadmap/me_sr09_cached_source_snapshot_inventory_command_roadmap_entry.md
```

Follow-up implementation candidates:

* ME-SR10 - Implement manual cached-source snapshot staging validator.
* ME-SR11 - Implement approved bounded acquisition or import workflow.
* ME-SR12 - Define non-US ticker source-family and source-mapping governance contract.
* ME-RUN25 - Rerun expanded cached-source coverage audit after staged snapshots exist.

ME-SR10 remains the next logical sprint because manually staged payloads and acquisition manifests must be validated against the ME-SR08 contract before acquisition/import work begins.

## Completed Sprint

### ME-SR10 - Manual cached-source snapshot staging validator

Owner roles: Product Owner / Operator / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR10

Goal: implement a local deterministic validator for manually staged cached-source snapshot manifests and payloads.

Outcome:

* added local cached-source snapshot staging validation logic;
* added module command `market_engine.source_refresh.cached_source_snapshot_staging_validator_command`;
* defined JSON report format `market-engine-cached-source-snapshot-staging-validation-v1`;
* reports accepted/rejected staging decisions with deterministic issue codes;
* blocks missing manifests, malformed manifests, unknown formats, missing payloads, hash/size mismatches, stale snapshots, failed/not-validated states, false usable flags, and fixture/test material;
* added deterministic fixture-based tests.

ME-SR10 did not acquire snapshots, stage snapshots, import snapshots, implement provider access, fetch live data, call SEC/EDGAR/yfinance, mutate portfolio/watchlist state, send Telegram output, write production data, modify cached-source dry-run semantics, change Decision Engine or Recommendation Review behavior, or add action authority.

Implemented documentation:

```text
docs/market_engine/audits/me_sr10_manual_cached_source_snapshot_staging_validator_audit.md
docs/market_engine/backlog/me_sr10_manual_cached_source_snapshot_staging_validator_backlog_entry.md
docs/market_engine/roadmap/me_sr10_manual_cached_source_snapshot_staging_validator_roadmap_entry.md
```

Follow-up implementation candidates:

* ME-SR11 - Implement cached-source snapshot acquisition dry-run command.
* ME-SR12 - Define non-US ticker source-family and source-mapping governance contract.
* ME-RUN25 - Rerun expanded cached-source coverage audit after staged snapshots exist.

ME-SR11 remains the next logical sprint because any acquisition/import dry-run must preserve ME-SR08 manifest requirements and ME-SR10 staging validation gates.

## Completed Sprint

### ME-SR11 - Cached-source snapshot acquisition dry-run command

Owner roles: Product Owner / Operator / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR11

Goal: implement a local deterministic cached-source snapshot acquisition dry-run command.

Outcome:

* added local cached-source snapshot acquisition dry-run planning logic;
* added module command `market_engine.source_refresh.cached_source_snapshot_acquisition_dry_run_command`;
* defined JSON report format `market-engine-cached-source-snapshot-acquisition-dry-run-v1`;
* reports planned/blocked entries for ticker and source-family requests;
* blocks invalid tickers, unsupported source families, and missing output roots fail-closed;
* reports ME-SR08 required manifest fields and ME-SR10 staging validator handoff;
* added deterministic fixture-based tests.

ME-SR11 did not acquire snapshots, fetch data, stage payloads, import payloads, write acquisition manifests, implement provider access, call SEC/EDGAR/yfinance, mutate portfolio/watchlist state, send Telegram output, write production data, modify cached-source dry-run semantics, change Decision Engine or Recommendation Review behavior, or add action authority.

Implemented documentation:

```text
docs/market_engine/audits/me_sr11_cached_source_snapshot_acquisition_dry_run_command_audit.md
docs/market_engine/backlog/me_sr11_cached_source_snapshot_acquisition_dry_run_command_backlog_entry.md
docs/market_engine/roadmap/me_sr11_cached_source_snapshot_acquisition_dry_run_command_roadmap_entry.md
```

Follow-up implementation candidates:

* ME-SR12 - Implement operator-supplied cached-source snapshot import command.
* Future source-family governance - Define non-US ticker source-family and source-mapping governance.
* ME-RUN25 - Rerun expanded cached-source coverage audit after staged snapshots exist.

ME-SR12 is the next logical sprint because acquisition intent is now visible, but the repo still needs a controlled local import path for operator-supplied payloads before broader cached-source coverage can expand.

## Completed Sprint

### ME-SR12 - Operator-supplied cached-source snapshot import command

Owner roles: Product Owner / Operator / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR12

Goal: implement a safe local command for importing operator-supplied cached-source snapshots into the cached-source snapshot workspace.

Outcome:

* added local cached-source snapshot import logic;
* added module command `market_engine.source_refresh.cached_source_snapshot_import_command`;
* validates one snapshot directory or direct `manifest.json` path with the ME-SR10 staging validator before copy;
* imports to `<destination-root>/<batch_id>/<ticker>/<snapshot_id>/`;
* preserves manifest and payload files without mutating the source;
* blocks missing sources, missing/malformed manifests, validation failures, ambiguous manifests, and existing destinations fail-closed;
* prints stable operator-readable terminal success and failure summaries;
* added deterministic fixture-based tests.

ME-SR12 did not call providers, fetch data, use SEC/EDGAR/yfinance, send Telegram output, mutate portfolio/watchlist state, write outside the configured destination root, modify cached-source dry-run semantics, change Decision Engine or Recommendation Review behavior, or add action authority.

Implemented documentation:

```text
docs/market_engine/audits/me_sr12_operator_supplied_cached_source_snapshot_import_command_audit.md
docs/market_engine/backlog/me_sr12_operator_supplied_cached_source_snapshot_import_command_backlog_entry.md
docs/market_engine/roadmap/me_sr12_operator_supplied_cached_source_snapshot_import_command_roadmap_entry.md
```

Follow-up implementation candidates:

* ME-RUN25 - Rerun expanded cached-source coverage audit after validated local imports exist.
* Future source-family governance - Define non-US ticker source-family and source-mapping governance.
* Future import enhancement - Add explicitly tested overwrite or batch-import behavior only if operator workflow requires it.

## Completed Run

### ME-RUN25 - Operator-supplied cached-source snapshot import validation flow

Owner roles: Product Owner / Operator / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: ME-RUN - Local Run / Validation

Status: COMPLETED BY ME-RUN25

Goal: run the first operator-supplied cached-source snapshot import/staging validation flow and prove whether it can feed the existing local dry-run path.

Outcome:

* created a temporary non-production operator-supplied SEC CompanyFacts fixture under `/private/tmp`;
* imported the fixture with the ME-SR12 import command;
* validated the imported workspace with the ME-SR10 staging validator;
* confirmed the imported payload can feed `--input-mode cached_source_snapshot`;
* confirmed the local dry-run blocks at `portfolio_review` without portfolio context;
* confirmed the local dry-run completes when non-production local portfolio context is supplied;
* recorded contract gaps, tooling gaps, safety state, and next sprint recommendation.

Conclusion:

```text
PASS
```

The run is fixture-backed and non-production. Real-world source quality remains unproven until operator-supplied local files are used.

Implemented documentation:

```text
docs/market_engine/audits/me_run25_operator_supplied_cached_source_snapshot_import_validation_flow.md
docs/market_engine/backlog/me_run25_operator_supplied_cached_source_snapshot_import_validation_flow_backlog_entry.md
docs/market_engine/roadmap/me_run25_operator_supplied_cached_source_snapshot_import_validation_flow_roadmap_entry.md
```

Follow-up implementation candidates:

* ME-SR13 - Run real-world operator-supplied cached-source sample import for NVDA, AMD, ASML.
* ME-SR14 - Run first real cached-source Market Engine analysis for accepted sample tickers.
* ME-SR15 - Render Telegram-style terminal preview from real cached-source analysis output.

## Blocked Sprint

### ME-SR13 - Real-world operator-supplied cached-source sample import

Owner roles: Product Owner / Operator / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: BLOCKED BY MISSING OPERATOR INPUT

Goal: run the first real-world operator-supplied cached-source sample import for `NVDA`, `AMD`, and `ASML`.

Outcome:

* checked for the expected local operator input root at `operator_input/market_engine/me-sr13-real-world-sample/`;
* confirmed the `operator_input` root is absent from the workspace;
* did not fabricate `NVDA`, `AMD`, or `ASML` source files;
* did not substitute the ME-RUN25 fixture;
* did not run import, staging validation, or local cached-source dry-runs because no real input was available;
* preserved a documentation-only blocked result.

Conclusion:

```text
BLOCKED
```

Implemented documentation:

```text
docs/market_engine/audits/me_sr13_real_world_operator_supplied_cached_source_sample_import.md
docs/market_engine/backlog/me_sr13_real_world_operator_supplied_cached_source_sample_import_backlog_entry.md
docs/market_engine/roadmap/me_sr13_real_world_operator_supplied_cached_source_sample_import_roadmap_entry.md
```

Follow-up implementation candidate:

* ME-SA01 - Define automated cached-source acquisition job contract.
* ME-SR13A - Prepare real-world operator-supplied cached-source input package for NVDA, AMD, ASML as a fallback/manual diagnostic candidate only.

ME-SR13A is superseded as the primary next sprint by the ME-RM03 product-owner correction. The application-owned automated cached-source acquisition path is now the active direction.

## Completed Roadmap Governance Sprint

### ME-RM03 - Automated cached-source acquisition roadmap correction

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RM / Roadmap Governance

Status: COMPLETED BY ME-RM03

Goal: lock the product-owner decision that automated cached-source acquisition by an application job is now the primary direction after ME-SR13.

Decision:

* automated cached-source acquisition is now primary;
* ME-SR13A is superseded as the primary next sprint;
* ME-SR13A remains available only as a fallback/manual diagnostic candidate;
* ME-SA01 is now the next active sprint;
* the existing ME-SR12 / ME-RUN25 import, staging validation, and `cached_source_snapshot` dry-run bridge remains valid and should be consumed by the future acquisition job.

Recommended sequence:

```text
ME-SA01 - Define automated cached-source acquisition job contract
ME-SA02 - Implement first bounded automated cached-source acquisition job for approved sample tickers/source families
ME-RUN26 - Run automated cached-source acquisition for NVDA, AMD, ASML through staging validation and local dry-run
ME-TP01 - Produce terminal-visible operator preview from real cached-source dry-run artifacts
```

ME-SA01 must define acquisition job inputs, bounded ticker or universe input, approved source families, approved provider/source adapters, provenance, retrieval timestamp, source timestamp, freshness/staleness policy, missing-data handling, cached-source snapshot output location, manifest compatibility with the existing validator/import flow, fail-closed behavior, no downstream side effects, and no analysis or decision authority.

Implemented documentation:

```text
docs/market_engine/audits/me_rm03_automated_cached_source_acquisition_roadmap_correction.md
docs/market_engine/backlog/me_rm03_automated_cached_source_acquisition_roadmap_correction_backlog_entry.md
docs/market_engine/roadmap/me_rm03_automated_cached_source_acquisition_roadmap_correction_roadmap_entry.md
```

### ME-SR03 - Resolve canonical-universe cached-source coverage blockers

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR03

Goal: resolve or precisely document the canonical-universe cached-source coverage blockers for HO, ASML, and TSM.

Outcome:

* ASML resolved by preserving annual `20-F` `us-gaap` facts in `EUR`;
* TSM resolved by preserving annual `20-F` `ifrs-full` facts in `USD`;
* HO remains blocked because no approved cached SEC CompanyFacts snapshot exists locally;
* canonical rerun improved to 12 completed tickers and 1 blocked ticker.

Implemented runtime change:

```text
src/market_engine/source_intake/sec_companyfacts_fields.py
```

Implemented tests:

```text
tests/market_engine/source_intake/test_sec_companyfacts_field_mapping.py
```

Implemented documentation:

```text
docs/market_engine/source_data/me_sr03_canonical_universe_cached_source_coverage_blockers.md
docs/market_engine/audits/me_sr03_canonical_universe_cached_source_coverage_blockers_audit.md
docs/market_engine/backlog/me_sr03_canonical_universe_cached_source_coverage_blockers_backlog_entry.md
docs/market_engine/roadmap/me_sr03_canonical_universe_cached_source_coverage_blockers_roadmap_entry.md
```

ME-SR03 did not introduce provider calls, live data calls, Telegram delivery, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Sprint

### ME-SR04 - Resolve HO canonical-universe source identity or exclusion decision

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR04

Goal: decide whether HO should receive an approved source identity/backfill path or be moved out of default cached-source execution until a valid source exists.

Rationale: ME-SR03 resolved ASML and TSM using existing cached source data. HO remains blocked because ME-SR02 recorded it as unsupported and no local cached SEC CompanyFacts snapshot exists.

Outcome:

* HO remains in the canonical universe as Thales on Euronext;
* HO source policy changed from `cached_source_only` to `manual_review_only`;
* HO is excluded from default canonical SEC CompanyFacts cached-source execution;
* HO is not eligible for future Telegram preview or delivery until a separate approved source identity decision changes that status;
* canonical cached-source rerun selected 12 supported tickers and completed 12 with zero blocked tickers.

Implemented configuration/test changes:

```text
data/market_engine/ticker_universe/ticker_universe.csv
tests/market_engine/ticker_universe/test_canonical_ticker_universe.py
```

Implemented documentation:

```text
docs/market_engine/source_data/me_sr04_ho_canonical_universe_source_identity_decision.md
docs/market_engine/audits/me_sr04_ho_canonical_universe_source_identity_decision_audit.md
docs/market_engine/backlog/me_sr04_ho_canonical_universe_source_identity_decision_backlog_entry.md
docs/market_engine/roadmap/me_sr04_ho_canonical_universe_source_identity_decision_roadmap_entry.md
```

Scope: Source Refresh / source identity only. No portfolio writes, watchlist writes, Telegram delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Universe Governance Sprint

### ME-UNI08 - Add first-class Professional Swing Universe CLI flag

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI08

Goal: expose the approved editable Professional Swing Universe as a first-class local cached-source batch runtime CLI choice.

Outcome:

* added `--professional-swing-universe` to the cached-source batch dry-run command;
* routed the flag through the existing ME-UNI07 runtime-input builder;
* preserved the ME-UNI06 Professional Swing Universe loader and validation path;
* preserved custom `--canonical-ticker-universe <path>` behavior;
* made the Professional Swing flag and custom universe path mutually exclusive;
* added focused command tests and sprint documentation.

Implemented runtime/test changes:

```text
src/market_engine/run/cached_source_batch_dry_run_command.py
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

Implemented documentation:

```text
docs/market_engine/ticker_universe/me_uni08_professional_swing_universe_cli_flag.md
docs/market_engine/audits/me_uni08_professional_swing_universe_cli_flag_audit.md
docs/market_engine/backlog/me_uni08_professional_swing_universe_cli_flag_backlog_entry.md
docs/market_engine/roadmap/me_uni08_professional_swing_universe_cli_flag_roadmap_entry.md
```

Scope: ME-UNI08 did not introduce provider calls, source refresh, output/reporting behavior, delivery behavior, scheduler behavior, portfolio/watchlist writes, Decision Engine action semantics, or trading authority.

## Next Source Support Candidate

### ME-SR05 - Classify source support for Professional Swing Universe

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR05

Goal: classify actual cached-source support for Professional Swing Universe rows before broad supported-universe cached-source scanning.

Rationale: ME-UNI08 makes the editable Professional Swing Universe easy to select at runtime, but operator source-policy hints are not authoritative source-support truth.

Outcome:

* implemented deterministic Professional Swing Universe source-support classification;
* consumed the validated editable Professional Swing Universe loader;
* classified local SEC CompanyFacts source support from approved cached snapshot artifacts and provider error records;
* emitted explicit statuses for `supported_cached`, `missing_snapshot`, `unsupported_sec_companyfacts`, `missing_required_source_field`, `malformed_or_unreadable_source_artifact`, `ambiguous_identity`, `manual_review_only`, and `excluded`;
* preserved source artifact references, provider error references, required source field status, missing-field evidence, universe row references, and numeric-zero evidence;
* preserved the source-support-only boundary with no provider calls, source refresh, reporting, portfolio/watchlist mutation, Decision Engine behavior, action semantics, allocation, ranking, scoring, urgency, conviction, tradeability, position sizing, order, or execution behavior.

Planned sequence after ME-SR05:

```text
ME-RUN20 - Execute clean supported-universe cached-source scan
ME-RUN21 - Inspect and summarize supported-universe cached-source scan outputs
ME-RUN22 - Produce first human-readable Market Engine interpretation report from cached-source supported-universe outputs
ME-OUT01 - Define readable operator report contract from dry-run artifacts
ME-OUT02 - Implement readable operator report from dry-run artifacts
ME-CANDIDATE01 - Define non-actionable candidate classification contract
ME-CANDIDATE02 - Implement non-actionable candidate classification from readable operator output
```

### ME-RUN20 - Execute clean supported-universe cached-source scan

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN20

Goal: execute a local cached-source scan against the currently supported active subset of the editable Professional Swing Universe and produce inspectable local artifacts.

Scope: ME-RUN20 should consume ME-SR05 source-support classification results. Unsupported, missing, malformed, ambiguous, manual-review-only, and excluded rows must remain visible but must not be silently treated as clean supported cached-source rows.

Outcome:

* executed the 12 ME-SR05-supported cached-source tickers through the existing local cached-source batch dry-run path;
* requested 12, discovered 12 cached snapshots, executed 12, completed 12;
* observed 0 blocked, 0 failed, 0 skipped, 0 missing, 0 ambiguous, 0 unsupported, and 0 stale source results inside the supported subset;
* wrote local non-production artifacts under `artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/`;
* did not commit generated artifacts by default;
* preserved all cached-source/local-only and non-actionable boundaries.

### ME-RUN21 - Inspect and summarize supported-universe cached-source scan outputs

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN21

Goal: inspect the ME-RUN20 supported-universe cached-source scan artifacts and summarize whether the outputs are complete, consistent, and usable as the basis for first human-readable Market Engine interpretation.

Outcome:

* inspected the ME-RUN20 artifact root under `artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/`;
* confirmed 12 ticker directories contain valid `dry_run.json` and `manifest.json`;
* confirmed all 12 ticker payloads use `market-engine-end-to-end-dry-run-v1`;
* confirmed all 12 ticker payloads completed all expected dry-run stages;
* observed no missing-data markers, stale-data markers, blocked stages, malformed JSON, or structural inconsistency in the supported subset;
* documented readiness for the next non-actionable interpretation/reporting sprint.

### ME-RUN22 - Produce first human-readable Market Engine interpretation report from cached-source supported-universe outputs

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN22

Goal: produce the first human-readable, non-actionable Market Engine interpretation report from the ME-RUN20 cached-source supported-universe artifacts.

Scope: ME-RUN22 must preserve the non-actionable boundary. It may summarize and explain generated artifacts, but it must not introduce BUY / SELL / HOLD advice, allocation, ranking, scoring, target prices, urgency, conviction, tradeability, position sizing, execution instructions, broker-ready output, Telegram delivery, or production writes.

Outcome:

* implemented `market-engine-interpretation-report-v1`;
* added a deterministic local report generator for cached-source dry-run artifacts;
* generated Markdown and companion JSON summary outputs;
* preserved per-ticker artifact paths, stage states, missing/stale/blocked markers, and provenance references;
* handled missing and malformed ticker artifacts with explicit skipped reasons;
* added focused tests for happy path, missing files, malformed JSON, deterministic ordering, guardrail metadata, and CLI output;
* generated a local sample report under `artifacts/market_engine/me-run22-human-readable-report-me-run20-supported-universe-20260623T120000Z/`;
* did not commit generated local report artifacts by default;
* preserved the non-actionable, provider-free, local-only boundary.

### ME-OUT01 - Define readable operator report contract from dry-run artifacts

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-OUT - Output / Operator Reporting

Status: COMPLETED BY ME-OUT01

Goal: define a readable, non-actionable operator report contract from generated dry-run artifacts without introducing delivery, trading authority, ranking, scoring, allocation, or execution behavior.

Outcome:

* defined `market-engine-readable-operator-report-v1`;
* listed approved local dry-run and interpretation report input families;
* defined required Markdown operator report sections;
* defined required JSON companion summary metadata;
* documented missing-data, stale-data, blocked-state, numeric-zero, provenance, deterministic-output, fail-closed, and advisory-language guardrail requirements;
* preserved all provider, broker, delivery, portfolio, watchlist, runtime, and action-authority boundaries.

### ME-OUT02 - Implement readable operator report from dry-run artifacts

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-OUT - Output / Operator Reporting

Status: COMPLETED BY ME-OUT02

Goal: implement the readable operator report contract defined by ME-OUT01 as deterministic local Markdown and JSON report generation.

Outcome:

* implemented `market-engine-readable-operator-report-v1`;
* added deterministic local operator report generation from existing dry-run artifacts;
* emitted `operator_report.md` and `operator_report_summary.json`;
* preserved artifact integrity, stage completion, output-family, missing-data, stale-data, blocked-state, provenance, and numeric-zero markers;
* skipped incomplete or malformed per-ticker artifacts with explicit reasons;
* refused missing roots, unsafe report run ids, path traversal, and output overwrite;
* added focused tests and implementation documentation;
* preserved the local-only, provider-free, non-production, non-actionable output boundary.

### ME-CANDIDATE01 - Define non-actionable candidate classification contract

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-CANDIDATE - Candidate Classification

Status: COMPLETED BY ME-CANDIDATE01

Goal: define a non-actionable candidate classification contract from readable operator output and dry-run artifacts without introducing action authority, ranking, scoring, allocation, urgency, conviction, tradeability, or execution behavior.

Outcome:

* defined `market-engine-candidate-classification-v1`;
* listed approved local readable operator, dry-run artifact, and interpretation report inputs;
* defined allowed non-actionable candidate buckets;
* documented forbidden action, ranking, scoring, urgency, conviction, tradeability, target-price, allocation, and execution semantics;
* defined required machine-readable summary metadata;
* documented missing-data, stale-data, blocked-state, malformed-artifact, unsupported-input, numeric-zero, provenance, advisory-language guardrail, and fail-closed requirements.

### ME-CANDIDATE02 - Implement non-actionable candidate classification from readable operator output

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-CANDIDATE - Candidate Classification

Status: COMPLETED BY ME-CANDIDATE02

Goal: implement `market-engine-candidate-classification-v1` from readable operator output without introducing action authority, ranking, scoring, allocation, urgency, conviction, tradeability, or execution behavior.

Outcome:

* implemented deterministic pure candidate classification from readable operator output;
* emitted local `candidate_classification_report.md` and `candidate_classification_summary.json`;
* used the exact ME-CANDIDATE01 candidate bucket set;
* preserved evidence references, blocking reasons, safety flags, missing-data markers, stale-data markers, blocked-state markers, provenance presence, and numeric-zero evidence presence;
* detected forbidden action-oriented wording and failed closed into an unclassified bucket;
* added focused tests and implementation/audit documentation;
* preserved local-only, provider-free, non-production, non-actionable boundaries.

Next planning note: ME-CANDIDATE02 does not insert an immediate blocking follow-up. Candidate-classification QA/review, output readability polish, delivery-preview work, portfolio-context persistence, stronger Decision Engine handoff review, and additional governance remain valid deferred follow-up candidates. They should be picked up only after expanded-universe execution produces concrete inspection, QA, governance, or delivery evidence that justifies them, or if such a concrete blocker is discovered earlier. The active next direction is to scale from the current supported subset toward a larger Professional Swing Universe / target analysis universe and then execute readable/candidate outputs over that larger universe.

## Completed Sprint

### ME-UNI09 - Expand Professional Swing Universe from candidate classification output

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI09

Goal: implement controlled, deterministic Professional Swing Universe expansion from non-actionable candidate-classification output.

Outcome:

* added `market-engine-professional-swing-universe-expansion-v1`;
* implemented a pure builder that consumes `market-engine-candidate-classification-v1`;
* preserved existing editable Professional Swing Universe entries;
* included only eligible `ready_for_manual_candidate_review` candidates with valid proposed universe rows;
* excluded already-present, duplicated, manual-review-only, ambiguous, unsupported, non-equity, missing-source, malformed, and ineligible candidates with explicit reasons;
* failed closed on malformed summaries, unsupported format versions, unknown candidate buckets, invalid tickers, conflicting identities, unsafe paths, and invalid proposed universe entries;
* returned deterministic summary counts and auditable per-candidate decisions;
* performed no file writes and did not mutate the canonical universe CSV;
* preserved local-only, provider-free, non-production, non-actionable boundaries.

Implemented runtime:

```text
src/market_engine/ticker_universe/professional_swing_expansion.py
```

Implemented tests:

```text
tests/market_engine/ticker_universe/test_professional_swing_universe_expansion.py
```

Implemented documentation:

```text
docs/market_engine/ticker_universe/me_uni09_professional_swing_universe_expansion_from_candidates.md
docs/market_engine/audits/me_uni09_professional_swing_universe_expansion_from_candidates_audit.md
docs/market_engine/backlog/me_uni09_professional_swing_universe_expansion_from_candidates_backlog_entry.md
docs/market_engine/roadmap/me_uni09_professional_swing_universe_expansion_from_candidates_roadmap_entry.md
```

## Active Next Direction

### ME-SA08 - Define safe descriptive Analysis Review continuation beyond the Recommendation Review boundary

Owner roles: Product Owner / Operator / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SA / Pipeline continuation contract

Status: NEXT ACTIVE CANDIDATE AFTER ME-SA07

Goal: define how descriptive Analysis Review context may continue toward reportability without creating recommendation or allocation authority.

Scope: continuation contract only. No provider calls, live data, investment evaluation, recommendation semantics, portfolio behavior, allocation, or Decision Engine authority.

Rationale: ME-SA07 reaches Analysis Review safely and exposes Recommendation Review as the next controlled authority boundary before reportability.

ME-SR13A remains available only as a fallback/manual diagnostic candidate.

## Deferred Follow-up Candidates

These sprints are not rejected and not blocked. They remain valid later-candidates and are intentionally deferred below ME-SA01:

* ME-CANDIDATE03 - Candidate classification QA/review contract.
* ME-OUT03 - Operator report readability/polish improvements.
* ME-DL03 - Non-production delivery preview, only after expanded-universe report usefulness is proven.
* ME-PR03 - Approved portfolio context source/persistence contract, only if larger runs show repeated portfolio-context friction.
* ME-DE03 - Decision Engine handoff review hardening, only if expanded candidate/report outputs expose a concrete downstream handoff gap.
* ME-QAxx / ME-GOVxx - Additional governance or QA only when concrete evidence justifies it.

## Scale-first Rule

After ME-CANDIDATE02 and ME-UNI09, planning should prefer expanded-universe source-support classification and execution over additional polish, QA, governance, or delivery layers unless a concrete blocker is discovered. The project should first prove that the full local pipeline can run over a materially larger ticker universe before refinement layers are prioritized.

## Completed Sprint

### ME-RUN17 - Canonical-universe cached-source batch dry-run with ME-SR02 snapshots

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED WITH DOWNSTREAM BLOCKED OUTCOME BY ME-RUN17

Goal: execute and fix canonical-universe cached-source batch dry-run behavior using ME-SR02 snapshots.

Outcome:

* fixed RUN discovery for `sec_companyfacts/<snapshot_id>/raw/*.json`;
* preserved older `*/raw/*.json` discovery;
* selected 13 canonical active `cached_source_only` tickers;
* excluded SMCI as `manual_review_only`;
* discovered 12 ME-SR02 raw snapshots;
* executed 12 local end-to-end dry-run payloads;
* kept HO blocked as missing cached source;
* generated 12 local per-ticker artifacts plus a batch manifest;
* preserved provider-free, local-only, non-actionable boundaries.

Implemented runtime change:

```text
src/market_engine/run/cached_source_batch_execution.py
```

Implemented test change:

```text
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

Implemented documentation:

```text
docs/market_engine/run/me_run17_canonical_universe_cached_source_batch_dry_run_with_me_sr02_snapshots.md
docs/market_engine/audits/me_run17_canonical_universe_cached_source_batch_dry_run_audit.md
docs/market_engine/backlog/me_run17_canonical_universe_cached_source_batch_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run17_canonical_universe_cached_source_batch_dry_run_roadmap_entry.md
```

ME-RUN17 did not introduce provider calls, live data calls, Telegram delivery, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Historical RUN Candidate

### ME-RUN18 - Provide portfolio context for canonical-universe cached-source dry-runs

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: HISTORICAL CANDIDATE SUPERSEDED BY COMPLETED ME-RUN18 / ME-RUN19 PATH

Goal: provide approved local portfolio context to canonical-universe cached-source dry-runs so downstream review stages can progress without production portfolio writes.

Rationale: ME-RUN17 now discovers ME-SR02 snapshots and executes 12 dry-run payloads, but the chain remains blocked downstream because required local portfolio context is not provided.

Scope: local cached-source RUN behavior only. No provider refresh, live market data, portfolio writes, watchlist writes, Telegram/email delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Sprint

### ME-RUN19 - Portfolio-context-aware canonical cached-source dry-run

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN19

Goal: execute the canonical-universe cached-source batch dry-run with ME-SR02 snapshots and approved local non-production portfolio context.

Outcome:

* created the local non-production portfolio context file at `data/market_engine/portfolio_contexts/local_portfolio_context.json`;
* ran the existing ME-RUN18 `--portfolio-context` command path without runtime code changes;
* selected 13 active `cached_source_only` tickers from the canonical universe;
* excluded SMCI because `source_policy=manual_review_only`;
* discovered 12 ME-SR02 cached source snapshots;
* executed 12 per-ticker dry-runs;
* completed 10 tickers through Portfolio Review, Decision Engine handoff, Delivery / Reporting, and dry-run summary;
* preserved 3 blocked tickers: ASML, HO, and TSM;
* kept generated artifacts under `artifacts/market_engine/me-run19-20260622T103000Z/` uncommitted.

Implemented documentation:

```text
docs/market_engine/run/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_execution.md
docs/market_engine/audits/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_audit.md
docs/market_engine/backlog/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_roadmap_entry.md
```

ME-RUN19 did not introduce provider calls, live data calls, Telegram delivery, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next Source Refresh Candidate

### ME-SR03 - Resolve canonical-universe cached-source coverage blockers

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh

Status: CANDIDATE AFTER ME-RUN19

Goal: resolve the remaining cached-source coverage blockers exposed by ME-RUN19 before broader canonical-universe validation or Telegram preview work.

Rationale: ME-RUN19 proved that local portfolio context unlocks Portfolio Review, Decision Engine handoff, and Delivery / Reporting for complete cached-source tickers. Remaining blockers are source-coverage issues: HO lacks a cached source snapshot, while ASML and TSM preserve upstream missing-field evidence and block at Recommendation Review.

Scope: Source Refresh only. No portfolio writes, watchlist writes, Telegram delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

### ME-SA01 — Define automated cached-source acquisition job contract

Status: COMPLETED BY ME-SA01

ME-SA01 defined the docs-only contract for the automated cached-source acquisition job.

The primary route is now:

```text
automated acquisition job
  -> cached-source snapshot package
  -> existing import/staging validation
  -> cached_source_snapshot dry-run
  -> operator preview
```

This follows the product-owner decision recorded after ME-SR13 and ME-RM03: the application must be able to retrieve source data itself through an automated job. Manual operator-supplied input packages are no longer the primary route and remain only a possible fallback/manual diagnostic candidate.

ME-SA01 added:

```text
docs/market_engine/source_data/me_sa01_automated_cached_source_acquisition_job_contract.md
docs/market_engine/audits/me_sa01_automated_cached_source_acquisition_job_contract_audit.md
docs/market_engine/backlog/me_sa01_automated_cached_source_acquisition_job_contract_backlog_entry.md
docs/market_engine/roadmap/me_sa01_automated_cached_source_acquisition_job_contract_roadmap_entry.md
```

ME-SA01 defined:

- request format `market-engine-automated-cached-source-acquisition-request-v1`;
- result format `market-engine-automated-cached-source-acquisition-result-v1`;
- approved run modes;
- ticker input rules;
- source family rules;
- provider/source adapter policy;
- snapshot package compatibility;
- provenance requirements;
- freshness/staleness policy;
- missing-data policy;
- failure model;
- safety and side-effect constraints;
- handoff to existing import/staging/dry-run flow;
- ME-SA02 acceptance criteria.

ME-SA01 did not introduce runtime code, tests, provider calls, live data retrieval, yfinance, SEC/EDGAR access, source data files, fake NVDA/AMD/ASML data, Telegram sending, portfolio/watchlist writes, production writes, or downstream Decision Engine / Recommendation Review / Portfolio Review / Delivery semantic changes.

ME-SA01 also did not introduce BUY / SELL / HOLD, target price, allocation, position sizing, ranking, urgency, conviction, or tradeability authority.

Next logical sprint:

```text
ME-SA02 — Implement first bounded automated cached-source acquisition job
```

Follow-up route:

```text
ME-SA02
  -> ME-RUN26 — Run automated cached-source acquisition for NVDA/AMD/ASML through staging validation and local dry-run
  -> ME-TP01 — Produce terminal-visible operator preview from real cached-source dry-run artifacts
```

### ME-SA02 — Implement first bounded automated cached-source acquisition job

Status: COMPLETED BY ME-SA02

ME-SA02 implemented the first bounded automated cached-source acquisition job according to the ME-SA01 contract.

Implemented runtime:

```text
src/market_engine/source_acquisition/__init__.py
src/market_engine/source_acquisition/automated_cached_source_acquisition.py
```

Implemented tests:

```text
tests/market_engine/source_acquisition/test_automated_cached_source_acquisition.py
```

Implemented documentation:

```text
docs/market_engine/source_data/me_sa02_bounded_automated_cached_source_acquisition_job_implementation.md
docs/market_engine/audits/me_sa02_bounded_automated_cached_source_acquisition_job_audit.md
docs/market_engine/backlog/me_sa02_bounded_automated_cached_source_acquisition_job_backlog_entry.md
docs/market_engine/roadmap/me_sa02_bounded_automated_cached_source_acquisition_job_roadmap_entry.md
```

ME-SA02 supports explicit bounded ticker lists, the initial `company_profile` source family, deterministic fake adapter behavior, per-ticker snapshot package writing, result payload writing, manifest/payload hash and size recording, provenance preservation, freshness state preservation, and fail-closed validation.

Validation:

```text
12 passed - tests/market_engine/source_acquisition/test_automated_cached_source_acquisition.py
19 passed - tests/market_engine/source_refresh/test_cached_source_snapshot_staging_validator.py
492 passed - tests/market_engine
1159 passed - full pytest
```

ME-SA02 did not perform provider calls, network calls, yfinance calls, SEC/EDGAR calls, live data retrieval, Telegram sends, production writes, portfolio/watchlist writes, broker actions, or downstream semantic changes.

Next logical sprint:

```text
ME-RUN26 - Run automated cached-source acquisition for NVDA, AMD, ASML through staging validation and local dry-run
```

### ME-RUN26 - Run automated cached-source acquisition through staging validation and local dry-run

Status: COMPLETED WITH BLOCKED OUTCOME BY ME-RUN26

ME-RUN26 executed the ME-SA02 automated cached-source acquisition job for `NVDA`, `AMD`, and `ASML`, validated the generated packages with the existing staging validator, and attempted the existing `cached_source_snapshot` local dry-run path.

Artifact root:

```text
artifacts/market_engine/me-run26-automated-cached-source-acquisition-20260626T120200Z
```

Outcome:

```text
Acquisition: PASS - 3 completed entries
Staging validation: PASS - 3 accepted entries, 0 rejected entries
cached_source_snapshot dry-run: BLOCKED - 3 blocked entries
Overall: BLOCKED
```

Exact blocker:

```text
The existing cached_source_snapshot local dry-run path attempts to build SEC CompanyFacts Source Context and blocks on ME-SA02 company_profile payloads with: SEC CompanyFacts snapshot metadata is missing.
```

Implemented files:

```text
scripts/market_engine/me_run26_run_automated_cached_source_acquisition.sh
docs/market_engine/audits/me_run26_automated_cached_source_acquisition_staging_and_dry_run_audit.md
docs/market_engine/backlog/me_run26_automated_cached_source_acquisition_staging_and_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run26_automated_cached_source_acquisition_staging_and_dry_run_roadmap_entry.md
```

Next active sprint:

```text
ME-SA03 - Define company_profile cached-source dry-run consumption compatibility contract
```

### ME-SA03 - Define company_profile cached-source dry-run consumption compatibility contract

Status: COMPLETED BY ME-SA03

ME-SA03 defined the compatibility contract for consuming or explicitly rejecting `company_profile` cached-source snapshots through local `cached_source_snapshot` dry-run flows.

Implemented documentation:

```text
docs/market_engine/audits/me_sa03_company_profile_cached_source_dry_run_consumption_compatibility_contract.md
docs/market_engine/backlog/me_sa03_company_profile_cached_source_dry_run_consumption_compatibility_contract_backlog_entry.md
docs/market_engine/roadmap/me_sa03_company_profile_cached_source_dry_run_consumption_compatibility_contract_roadmap_entry.md
```

Next active sprint:

```text
ME-SA04 - Implement company_profile cached-source dry-run consumption compatibility gate
```

### ME-SA04 - Implement company_profile cached-source dry-run compatibility gate

Status: COMPLETED BY ME-SA04

ME-SA04 implemented a deterministic local compatibility gate for `company_profile` cached-source snapshots in the existing `cached_source_snapshot` dry-run route.

Implemented:

```text
src/market_engine/run/cached_source_execution.py
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
docs/market_engine/audits/me_sa04_company_profile_cached_source_dry_run_compatibility_gate_audit.md
docs/market_engine/backlog/me_sa04_company_profile_cached_source_dry_run_compatibility_gate_backlog_entry.md
docs/market_engine/roadmap/me_sa04_company_profile_cached_source_dry_run_compatibility_gate_roadmap_entry.md
```

Validation:

```text
12 passed - tests/market_engine/run/test_me_run10_cached_source_local_execution.py
103 passed - tests/market_engine/run
496 passed - tests/market_engine
1163 passed - full pytest
```

Next active sprint:

```text
ME-SA05 - Consume company_profile into Source Context
```

### ME-SA05 - Consume company_profile into Source Context

Status: COMPLETED BY ME-SA05

ME-SA05 implemented deterministic local consumption of compatible
`company_profile` snapshots into an explicit Source Context contract.

Implemented:

```text
src/market_engine/source_context/company_profile_context.py
src/market_engine/run/cached_source_execution.py
src/market_engine/run/end_to_end_dry_run.py
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
docs/market_engine/audits/me_sa05_company_profile_source_context_consumption_audit.md
docs/market_engine/backlog/me_sa05_company_profile_source_context_consumption_backlog_entry.md
docs/market_engine/roadmap/me_sa05_company_profile_source_context_consumption_roadmap_entry.md
```

Valid profiles are consumed, rejected profiles remain fail-closed without
trusted profile content, and SEC CompanyFacts input records profile absence as
optional. Profile-only execution stops before Fundamental Observations.

Validation:

```text
21 passed - tests/market_engine/run/test_me_run10_cached_source_local_execution.py
112 passed - tests/market_engine/run
505 passed - tests/market_engine
1172 passed - full pytest
```

Next active sprint:

```text
ME-SA06 - Derive basic company_profile observations from Source Context
```

### ME-SA06 - Derive basic company_profile observations from Source Context

Status: COMPLETED BY ME-SA06

ME-SA06 implemented deterministic, informational profile observations from
consumed Company Profile Source Context.

Implemented:

```text
src/market_engine/fundamental_observations/company_profile_observations.py
src/market_engine/run/cached_source_execution.py
src/market_engine/run/end_to_end_dry_run.py
tests/market_engine/fundamental_observations/test_company_profile_observations.py
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
docs/market_engine/audits/me_sa06_company_profile_fundamental_observations_audit.md
docs/market_engine/backlog/me_sa06_company_profile_fundamental_observations_backlog_entry.md
docs/market_engine/roadmap/me_sa06_company_profile_fundamental_observations_roadmap_entry.md
```

Profile-only runs now complete Fundamental Observations and stop at Derived
Observations. SEC CompanyFacts observations remain unchanged.

Validation:

```text
4 passed - company-profile Fundamental Observations tests
21 passed - cached-source local execution tests
112 passed - tests/market_engine/run
509 passed - tests/market_engine
1176 passed - full pytest
```

Next active sprint:

```text
ME-RUN27 - Run NVDA/AMD/ASML with company_profile Source Context and Fundamental Observations
```

### ME-RUN27 - Run NVDA/AMD/ASML through company_profile Source Context and Fundamental Observations

Status: COMPLETED WITH CONTROLLED STOP BY ME-RUN27

ME-RUN27 executed one deterministic local path for the bounded NVDA/AMD/ASML
validation set.

Outcome:

```text
acquisition: 3 completed
staging validation: 3 accepted
compatibility gate: 3 allowed
Source Context: 3 consumed
Fundamental Observations: 3 completed
stop stage: Derived Observations for all 3
overall: completed_with_controlled_stop
```

Implemented evidence:

```text
scripts/market_engine/me_run27_company_profile_cross_ticker_dry_run.py
tests/market_engine/run/test_me_run27_company_profile_cross_ticker_dry_run.py
docs/market_engine/audits/me_run27_company_profile_cross_ticker_dry_run_audit.md
docs/market_engine/backlog/me_run27_company_profile_cross_ticker_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run27_company_profile_cross_ticker_dry_run_roadmap_entry.md
```

Next active sprint:

```text
ME-SA07 - Allow company_profile observations into Analysis Review as descriptive context only
```

### ME-SA07 - Allow company_profile observations into Analysis Review as descriptive context only

Status: COMPLETED BY ME-SA07

ME-SA07 implemented a non-financial context bridge, setup-not-applicable
boundary, and descriptive Analysis Review context for consumed company-profile
observations.

Profile-only runs now complete Analysis Review and stop at Recommendation Review
with:

```text
company_profile_descriptive_analysis_context_has_no_recommendation_input
```

Implemented:

```text
src/market_engine/derived_observations/company_profile_context_bridge.py
src/market_engine/setup_detection/company_profile_not_applicable.py
src/market_engine/analysis_review/company_profile_analysis_context.py
tests/market_engine/analysis_review/test_company_profile_analysis_context.py
docs/market_engine/audits/me_sa07_company_profile_analysis_review_descriptive_context_audit.md
docs/market_engine/backlog/me_sa07_company_profile_analysis_review_descriptive_context_backlog_entry.md
docs/market_engine/roadmap/me_sa07_company_profile_analysis_review_descriptive_context_roadmap_entry.md
```

Validation:

```text
7 passed - Company Profile Analysis Context tests
21 passed - cached-source local execution tests
114 passed - tests/market_engine/run
518 passed - tests/market_engine
1185 passed - full pytest
```

Next active sprint:

```text
ME-SA08 - Define safe descriptive Analysis Review continuation beyond the Recommendation Review boundary
```

### ME-SA08 - Add company_profile-only Recommendation Review boundary

Status: COMPLETED BY ME-SA08

ME-SA08 implemented an explicit fail-closed Recommendation Review result for
profile-only analysis context:

```text
review_state: blocked_by_missing_data
review_category: company_profile_only_context_non_actionable
```

Company metadata remains descriptive provenance only. It cannot populate
recommendation, price, conviction, sizing, trade, or Decision Engine-ready
fields and cannot upgrade limited SEC CompanyFacts evidence.

Implemented:

```text
src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py
src/market_engine/run/cached_source_execution.py
tests/market_engine/recommendation_review/test_sec_companyfacts_recommendation_review.py
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
tests/market_engine/run/test_me_run27_company_profile_cross_ticker_dry_run.py
docs/market_engine/audits/me_sa08_company_profile_only_recommendation_review_boundary.md
docs/market_engine/backlog/me_sa08_company_profile_only_recommendation_review_boundary_backlog_entry.md
docs/market_engine/roadmap/me_sa08_company_profile_only_recommendation_review_boundary_roadmap_entry.md
```

Validation:

```text
16 passed - Recommendation Review tests
21 passed - cached-source local execution tests
2 passed - ME-RUN27 cross-ticker tests
520 passed - tests/market_engine
1187 passed - full pytest
```

### ME-SA09 - Define multi-source analysis-context readiness contract

Status: COMPLETED DOCS-ONLY CONTRACT

ME-SA09 defines five distinct evidence-readiness levels:

```text
descriptive_only
partial_analysis
recommendation_eligible
actionable_review
decision_ready
```

The contract defines source families, a readiness matrix, deterministic blocked
reasons, prohibited inferences, downstream implications, and transition
invariants.

Company-profile-only context remains `descriptive_only` and blocked with
`company_profile_only_context_non_actionable`. Recommendation eligibility does
not imply a favorable or actionable result. `actionable_review` and ME-SA09
`decision_ready` are reserved under current governance.

Implemented documentation:

```text
docs/market_engine/analysis_review/me_sa09_multi_source_analysis_context_readiness_contract.md
docs/market_engine/audits/me_sa09_multi_source_analysis_context_readiness_contract_audit.md
docs/market_engine/backlog/me_sa09_multi_source_analysis_context_readiness_contract_backlog_entry.md
docs/market_engine/roadmap/me_sa09_multi_source_analysis_context_readiness_contract_roadmap_entry.md
```

Next sprint:

```text
ME-SA10 - Implement multi-source analysis-context readiness classifier
```

### ME-SA10 - Implement typed fail-closed analysis-context readiness classifier

Status: COMPLETED BY ME-SA10

ME-SA10 implements the ME-SA09 contract as a pure typed classifier.

Reachable levels:

```text
descriptive_only
partial_analysis
recommendation_eligible
```

Reserved and unreachable:

```text
actionable_review
decision_ready
```

Company-profile-only context remains descriptive and blocked with
`company_profile_only_context_non_actionable`. Descriptive context cannot
upgrade missing fundamental or setup/price/market evidence.

The classifier is exported as a standalone Analysis Review module. Existing
runtime artifacts and downstream contracts are unchanged.

Implemented:

```text
src/market_engine/analysis_review/analysis_context_readiness.py
src/market_engine/analysis_review/__init__.py
tests/market_engine/analysis_review/test_analysis_context_readiness.py
docs/market_engine/audits/me_sa10_analysis_context_readiness_classifier_audit.md
docs/market_engine/backlog/me_sa10_analysis_context_readiness_classifier_backlog_entry.md
docs/market_engine/roadmap/me_sa10_analysis_context_readiness_classifier_roadmap_entry.md
```

Validation:

```text
15 passed - new readiness classifier tests
40 passed - Analysis Review tests
16 passed - Recommendation Review tests
535 passed - tests/market_engine
1202 passed - full pytest
```

### ME-SA11 - Implement readiness adapter and artifact metadata

Status: COMPLETED BY ME-SA11

ME-SA11 adds a conservative adapter from existing dry-run stage payloads to the
ME-SA10 evidence families and persists the result at:

```text
artifact["payload"]["analysis_context_readiness"]
```

Company-profile-only output remains `descriptive_only` with
`company_profile_only_context_non_actionable`. Complete fundamental plus
setup/price/market evidence with valid provenance reaches at most
`recommendation_eligible`.

`actionable_review` and `decision_ready` remain reserved and unreachable.
Existing dry-run and local artifact format versions remain unchanged.

Implemented:

```text
src/market_engine/analysis_review/analysis_context_readiness_adapter.py
src/market_engine/analysis_review/__init__.py
src/market_engine/run/end_to_end_dry_run.py
tests/market_engine/analysis_review/test_analysis_context_readiness_adapter.py
tests/market_engine/run/test_end_to_end_dry_run.py
tests/market_engine/run/test_local_dry_run_artifacts.py
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
docs/market_engine/audits/me_sa11_analysis_context_readiness_adapter_artifact_metadata_audit.md
docs/market_engine/backlog/me_sa11_analysis_context_readiness_adapter_artifact_metadata_backlog_entry.md
docs/market_engine/roadmap/me_sa11_analysis_context_readiness_adapter_artifact_metadata_roadmap_entry.md
```

Validation:

```text
11 passed - readiness adapter tests
51 passed - Analysis Review tests
16 passed - Recommendation Review tests
114 passed - run tests
546 passed - tests/market_engine
1213 passed - full pytest
```

Next sprint:

```text
ME-RUN28A - Run NVDA/AMD/ASML through persisted readiness and Recommendation Review boundary
```

### ME-RUN28A - Validate persisted readiness and Recommendation Review boundary

Status: COMPLETED WITH CONTROLLED STOP BY ME-RUN28A

ME-RUN28A reused the existing deterministic cached-source/local dry-run flow
for `NVDA`, `AMD`, and `ASML`.

All three persisted artifacts contain
`artifact["payload"]["analysis_context_readiness"]`. Every result is
`descriptive_only`, with company-profile evidence present and fundamental,
setup/price/market, and provenance/staleness evidence missing. Recommendation
Review is blocked with `company_profile_only_context_non_actionable`.

Every artifact confirms:

```text
actionable_review_allowed: false
decision_engine_ready: false
```

No actionable recommendation fields were produced. `actionable_review` and
`decision_ready` remain reserved and unreachable.

Validation:

```text
51 passed - Analysis Review tests
16 passed - Recommendation Review tests
114 passed - run tests
1213 passed - full pytest
PASS - three persisted artifacts inspected
PASS - no actionable recommendation-field keys
PASS - git diff --check
```

Next sprint:

```text
ME-RUN28 - Expanded supported-universe acquisition and dry-run classification
```

The non-production Telegram preview remains unsent and deferred to ME-DL03.

### ME-RUN28 - Expanded supported-universe acquisition and dry-run classification

Status: COMPLETED WITH BLOCKED OUTCOME BY ME-RUN28

ME-RUN28 classified 16 active Professional Swing Universe tickers across
automated acquisition, staging validation, existing local cached-source
coverage, batch dry-run, persisted readiness, Recommendation Review, and
Decision Engine readiness.

```text
automated acquisition completed: 3
automated acquisition unsupported_ticker: 13
staging accepted: 3
direct acquisition-package dry-runs: 3 descriptive_only
existing SEC cached source found: 12
missing cached source snapshot: 4
partial_analysis: 12
Recommendation Review completed: 12
actionable: 0
Decision Engine-ready: 0
```

The automated acquisition job is the primary universe-scale blocker: it
supports only NVDA, AMD, and ASML and only `company_profile`. Those three
packages validate and run successfully to a controlled, descriptive-only
Recommendation Review stop.

The 12 existing SEC-cache dry-runs all persist fundamentals and
provenance/staleness evidence but lack setup/price/market evidence. They stop
at Portfolio Review because portfolio context was intentionally absent.

No staging defect, ticker-specific dry-run defect, provider/network call,
production write, Telegram send, portfolio/watchlist mutation, actionable
review, or Decision Engine-ready state was observed.

Validation:

```text
546 passed - tests/market_engine
1213 passed - full pytest
PASS - 16-ticker artifact classification assertions
PASS - git diff --check
```

Next sprint:

```text
ME-SA12 - Expanded supported-universe cached-source acquisition coverage contract
```

Setup/price/market evidence and portfolio-context readiness remain separate
follow-ups.

### ME-SA12 - Generic supported-universe cached-source coverage contract

Status: COMPLETED DOCS-ONLY CONTRACT

ME-SA12 defines a future-ticker-safe contract for supported-universe
cached-source coverage.

```text
tickers are data, not logic
```

The contract separates:

* universe membership;
* source-family support and availability;
* manifest and staging validation;
* provenance and freshness;
* consumability and completeness;
* analysis readiness;
* Recommendation Review eligibility;
* Portfolio Review and handoff readiness;
* reserved actionable and Decision Engine-ready capabilities.

Coverage requirements are selected through generic capability profiles, never
ticker names. ME-RUN28 tickers remain regression examples only.

Current ME-SA09/10/11 readiness values remain authoritative.
`actionable_review`, `actionable`, `decision_ready`, and `de_ready` remain
reserved and unreachable.

No runtime or test file changed.

Validation:

```text
546 passed - tests/market_engine
1213 passed - full pytest
PASS - git diff --check
PASS - governance grep; no new runtime hit
```

Next sprint:

```text
ME-SA13 - Implement generic cached-source coverage classification model
```

Expanded acquisition coverage must not precede the generic classifier or use
ticker-specific shortcuts.

### ME-SA13 - Implement generic cached-source coverage classification model

Status: COMPLETED BY ME-SA13

ME-SA13 implements the ME-SA12 contract as a pure deterministic classifier:

```text
market-engine-supported-universe-cached-source-coverage-v1
```

The model classifies generic source-family requirements across support,
availability, manifest validity, provenance, freshness, consumability,
completeness, readiness, and blockers.

Public API:

```text
classify_cached_source_coverage(...)
classify_cached_source_coverage_batch(...)
```

Ticker strings remain data attributes. No ticker-specific runtime branch or
allowlist determines classification.

Company-profile-only input remains `descriptive_only`. Partial analytical
coverage remains non-actionable. Reserved `actionable` and `de_ready` states
remain unreachable under current governance.

Validation:

```text
39 passed - new classifier tests
63 passed - source-support tests
585 passed - tests/market_engine
1252 passed - full pytest
PASS - git diff --check
PASS - governance greps; no new ticker-specific runtime logic
```

Next sprint:

```text
ME-SA14 - Adapt cached-source staging validation into generic coverage input
```

Expanded acquisition or dry-run reporting integration must follow the generic
adapter and may not use ticker-specific shortcuts.

### ME-SA14 - Staging-validation evidence adapter

Status: COMPLETED BY ME-SA14

ME-SA14 adds the deterministic Refinery bridge from existing cached-source
staging-validation entries to ME-SA13 `CachedSourceCoverageInput` values.

The adapter preserves ticker and market as data, maps only approved generic
source-family aliases, and fails closed for invalid manifests, missing
provenance, stale evidence, unsupported families, non-consumable snapshots,
and incomplete evidence.

Next sprint:

```text
ME-RUN29 - Run expanded generic coverage classification from staging-validation evidence
```

No expanded run, provider call, Governor/Dispatch Station behavior, actionable
state, or Decision Engine authority is part of ME-SA14.

### ME-RUN29 - Expanded generic coverage classification

Status: COMPLETED BY ME-RUN29

ME-RUN29 runs deterministic committed staging-validation fixture evidence
through the ME-SA14 adapter and ME-SA13 classifier. It writes local JSON and
Markdown evidence with per-entry coverage/readiness, dominant blockers, and
zero reserved authority states.

Next:

```text
ME-GV01 - Define The Governor investment evaluation contract
```

### ME-GV03 - Governor non-actionable dry-run scaffold

Status: COMPLETED BY ME-GV03

ME-GV03 implements deterministic ME-GV02 factor-state classification inside
the ME-GV01 output envelope. Evidence gaps, blockers, limitations, provenance,
and conflicts remain explicit. All scoring and reserved authority output stays
null, false, or blocked.

Next:

```text
ME-GV04 - Implement factor scoring from approved analysis evidence
```

### ME-GV04 - Governor factor scoring

Status: COMPLETED BY ME-GV04

ME-GV04 implements deterministic 0-100 factor scoring for explicit approved
fundamentals, growth, risk, and data-confidence evidence. Components,
normalization rules, contributions, evidence references, and limitations are
inspectable. Missing or conflicting evidence remains unscored.

Factor weights, overall score, rank, recommendation mapping, actionability,
allocation, and Decision Engine readiness remain unavailable.

Next:

```text
ME-GV05 - Implement recommendation-state mapping under approved boundary
```

### ME-GV05 - Governor recommendation-state mapping

Status: COMPLETED BY ME-GV05

ME-GV05 separates recommendation eligibility from direction and maps complete
approved Governor evidence into deterministic `avoid`, `watch`, `consider`, or
`preferred` interpretive states. Critical factors, data confidence, risk,
Recommendation Review boundaries, conflicts, and limitations remain explicit.

All recommendation output is non-actionable. Overall scoring, ranking,
allocation, execution, and Decision Engine readiness remain unavailable.

Next:

```text
ME-GV06 - Implement buy-zone and position-management explanation contract
```

### ME-GV06 - Governor buy-zone and position-management explanation

Status: COMPLETED BY ME-GV06

ME-GV06 adds fail-closed explanation from approved price/setup and position
context after recommendation mapping. Pullback, breakout, invalidation, and
hold/add/reduce/exit review context preserve evidence references and never
invent price levels.

Execution, orders, stops, allocation, mutation, and Decision Engine authority
remain unavailable.

Next:

```text
ME-DS01 - Define Dispatch Station output contract for Governor reports
```

### ME-RM06 - Reposition Delivery Layer around ChatGPT Advisory Integration

Status: COMPLETED DOCS-ONLY ROADMAP REPOSITION

ME-RM06 formally replaces the old implicit Telegram-first delivery direction
with a structured advisory sequence:

```text
Market Engine
-> Structured Artifacts / Structured Decision Output
-> ChatGPT Advisory Layer
-> user
```

Compact channel output is repositioned as Notification Layer:

```text
Market Engine
-> Notification Layer
-> Messenger / Signal / Telegram / email / later adapters
```

The backlog now treats ChatGPT Advisory Layer as the primary interactive user
interface over reproducible Market Engine artifacts. GitHub and artifacts remain
the source of truth. Notification adapters are deferred until structured output,
conviction, Position Sizing, Portfolio Intelligence, and ChatGPT context
contracts are stable.

Follow-up backlog items:

* ME-CI01 - Define Structured Decision Output contract for ChatGPT consumption. Completed by ME-CI01.
* ME-CI02 - Define ChatGPT Advisory Context Contract. Completed by ME-CI02.
* ME-CI03 - Add ChatGPT-readable Portfolio Intelligence context. Completed by ME-CI03.
* ME-CI04 - Define explainability/change-rationale contract. Completed by ME-CI04.
* ME-CI05 - Produce daily ChatGPT-ready advisory artifact.
* ME-PI01 - Define Portfolio Intelligence exposure contract.
* ME-PS01 - Define Position Sizing decision contract.
* ME-NL01 - Reframe notification layer as channel-neutral compact summary.
* ME-NL02 - Define daily notification payload contract.
* ME-NL03 - Select first notification adapter after structured outputs stabilize.

ME-RM06 changes documentation only. It does not add runtime code, tests,
scripts, CLI behavior, provider calls, ChatGPT API integration, Telegram,
Messenger, Signal, email implementation, portfolio/watchlist writes, production
integrations, or Decision Engine semantic changes.

### ME-CI01 - Structured Decision Output contract

Status: COMPLETED DOCS-FIRST CONTRACT

Goal: define Structured Decision Output v1 as the stable machine-readable
interface between Market Engine decision artifacts and consumers including
ChatGPT Advisory Layer, Notification Layer, dashboards, future frontends, and
audit/replay tooling.

Scope:

* contract documentation;
* JSON examples;
* field semantics;
* required and optional field behavior;
* versioning;
* compatibility rules;
* fail-closed rules;
* consumer guidance;
* roadmap/backlog/audit updates.

Non-goals:

* no runtime code;
* no Decision Engine semantic changes;
* no ChatGPT API integration;
* no notification integration;
* no provider, yfinance, SEC, or EDGAR changes;
* no portfolio/watchlist writes;
* no UI or dashboard implementation.

Acceptance criteria:

* Structured Decision Output v1 is versioned as `structured-decision-output-v1`.
* Top-level object and field semantics are documented.
* Required and optional fields are identified.
* Fail-closed behavior is documented.
* Consumer rules for ChatGPT Advisory Layer, Notification Layer, and dashboards
  are documented.
* Coverage, readiness, and actionability are explicit.
* Scores may be null while engines are absent.
* Consumers may not invent missing values.
* Future Conviction, Position Sizing, and Portfolio Intelligence are supported
  without being implemented.
* Example artifacts exist.
* No runtime code is changed.

Dependencies:

* ME-RM06 delivery roadmap reposition.

Follow-ups:

* ME-CI02 - ChatGPT Advisory Context Contract. Completed by ME-CI02.
* ME-CI03 - ChatGPT-readable Portfolio Intelligence context. Completed by ME-CI03.
* ME-CI04 - explainability/change-rationale contract. Completed by ME-CI04.
* ME-CI05 - daily ChatGPT-ready advisory artifact.
* ME-PI01 - Portfolio Intelligence exposure contract.
* ME-PS01 - Position Sizing decision contract.
* ME-NL01 - channel-neutral Notification Layer contract.

### ME-CI02 - ChatGPT Advisory Context Contract

Status: COMPLETED DOCS-ONLY CONTRACT

Goal: define `chatgpt-advisory-context-v1`, the controlled, evidence-backed
context envelope that a future ChatGPT Advisory Layer may consume.

Scope:

* contract documentation;
* eligible, descriptive-only, and blocked examples;
* provenance, freshness, uncertainty, readiness, and blocker semantics;
* ME-CI01 Structured Decision Output consumption rules;
* Governor and Dispatch Station context boundaries;
* portfolio and recommendation boundaries;
* prohibited inputs and prohibited inferences;
* fail-closed matrix;
* audit, roadmap, and backlog updates.

Non-goals:

* no ChatGPT API integration;
* no prompt runner;
* no LLM runtime;
* no Telegram, Messenger, Signal, email, dashboard, or notification
  integration;
* no provider, yfinance, SEC, or EDGAR changes;
* no portfolio/watchlist writes;
* no Decision Engine, Governor, Dispatch Station, Recommendation Review, or
  Portfolio Review semantic redesign.

Acceptance criteria:

* Contract name and version are explicit.
* Advisory eligibility states are defined.
* Provenance, freshness, uncertainty, readiness, and blockers are represented.
* ME-CI01 Structured Decision Output consumption is defined.
* Governor and Dispatch Station integration boundaries are documented.
* Portfolio context and recommendation boundaries are explicit.
* Prohibited inputs and prohibited inferences are listed.
* Fail-closed behavior is documented.
* Examples exist for eligible, descriptive-only, and blocked advisory contexts.
* No runtime code is changed.

Dependencies:

* ME-DS01 - Dispatch Station Governor report output contract.
* ME-RM06 - ChatGPT advisory delivery roadmap reposition.
* ME-CI01 - Structured Decision Output contract.

Follow-ups:

* ME-CI03 - ChatGPT-readable Portfolio Intelligence context. Completed by ME-CI03.
* ME-CI04 - explainability/change-rationale contract.
* ME-CI05 - daily ChatGPT-ready advisory artifact.
* ME-PI01 - Portfolio Intelligence exposure contract.
* ME-PS01 - Position Sizing decision contract.
* ME-NL01 - channel-neutral Notification Layer contract.

### ME-CI03 - ChatGPT-readable Portfolio Intelligence Context

Status: COMPLETED DOCS-ONLY CONTRACT

Goal: define `chatgpt-portfolio-intelligence-context-v1`, the controlled
Portfolio Intelligence subcontext that future ChatGPT advisory context assembly
may embed or reference inside the ME-CI02 advisory boundary.

Scope:

* source-of-truth matrix for portfolio identity, holdings, position value,
  weight, cash, exposure, concentration, constraints, portfolio fit,
  recommendation-to-position relationship, missingness, freshness, and
  provenance;
* relationship to `chatgpt-advisory-context-v1`;
* availability states: `available`, `partial`, `unavailable`, `blocked`;
* holdings and position semantics;
* exposure and concentration boundaries;
* cash and allocation boundaries;
* portfolio-fit boundary;
* missingness, provenance, and freshness contracts;
* advisory permission boundary;
* prohibited inputs and prohibited inferences;
* fail-closed matrix;
* complete, partial, blocked, and held-with-sizing-unavailable examples.

Non-goals:

* no runtime assembler;
* no typed schema or validator;
* no ChatGPT API integration;
* no prompt template;
* no provider, yfinance, SEC, or EDGAR change;
* no broker integration;
* no portfolio or watchlist writes;
* no allocation engine;
* no position sizing engine;
* no rebalancing engine;
* no Decision Engine, Governor, Portfolio Review, Dispatch Station, reporting,
  delivery, or notification semantic change.

Acceptance criteria:

* Contract identity is explicit:
  `chatgpt-portfolio-intelligence-context-v1`.
* Canonical upstream sources are documented.
* Unknown, missing, zero, partial, stale, and blocked semantics are separated.
* ChatGPT advisory permissions and prohibited determinations are explicit.
* Fail-closed behavior is documented.
* JSON examples validate.
* No runtime code is changed.

Dependencies:

* ME-RM06 - ChatGPT advisory delivery roadmap reposition.
* ME-CI01 - Structured Decision Output contract.
* ME-CI02 - ChatGPT Advisory Context Contract.
* ME-PR01 / ME-PR02 - Portfolio Review contract and implementation.
* ME-DE01 / ME-DE02 - Decision Engine handoff contract and implementation.
* ME-GV06 - Governor buy-zone and position-management explanation contract.
* ME-DS01 - Dispatch Station Governor report output contract.

Follow-ups:

* ME-CI04 - explainability/change-rationale contract. Completed by ME-CI04.
* ME-CI05 - daily ChatGPT-ready advisory artifact.
* ME-PI01 - Portfolio Intelligence exposure contract.
* future typed schema / validator.
* future deterministic advisory context assembler.

### ME-CI04 - Explainability / Change-Rationale Contract

Status: COMPLETED DOCS-ONLY CONTRACT

Goal: define `chatgpt-explainability-change-rationale-context-v1`, the
controlled explainability and change-rationale contract for the ChatGPT advisory
architecture.

Scope:

* source-of-truth matrix for explanation families;
* relation to ME-CI01, ME-CI02, and ME-CI03;
* explanation availability states;
* current-state rationale;
* change classification;
* state transition and evidence delta semantics;
* reason attribution levels;
* blocker delta and uncertainty delta;
* freshness-driven rationale;
* portfolio rationale;
* unchanged conclusion semantics;
* contradiction handling;
* permission matrix;
* use-case matrix;
* temporal comparison rules;
* fail-closed matrix;
* JSON examples.

Non-goals:

* no runtime explainability engine;
* no temporal diff runtime;
* no artifact comparison engine;
* no causal attribution engine;
* no materiality engine;
* no typed schema or validator;
* no ChatGPT API integration;
* no prompt template;
* no provider, yfinance, SEC, or EDGAR change;
* no broker integration;
* no portfolio or watchlist writes;
* no allocation, position sizing, or rebalancing engine;
* no Decision Engine, Governor, Recommendation Review, Portfolio Review,
  Dispatch Station, reporting, delivery, or notification semantic change.

Acceptance criteria:

* Contract identity is explicit:
  `chatgpt-explainability-change-rationale-context-v1`.
* Canonical explanation sources are documented.
* Reason attribution prevents unsupported causal claims.
* Temporal comparison distinguishes current, reference, previous comparable,
  previous chronological, and baseline runs.
* Unchanged conclusion does not imply no evidence changed.
* Portfolio rationale remains separate from standalone recommendation rationale.
* Fail-closed behavior is documented.
* JSON examples validate.
* No runtime code is changed.

Dependencies:

* ME-RM06 - ChatGPT advisory delivery roadmap reposition.
* ME-CI01 - Structured Decision Output contract.
* ME-CI02 - ChatGPT Advisory Context Contract.
* ME-CI03 - ChatGPT-readable Portfolio Intelligence Context.
* Recommendation Review, Portfolio Review, Decision Engine handoff, Governor,
  Dispatch Station, readiness, provenance, freshness, and uncertainty contracts.

Follow-ups:

* ME-CI05 - daily ChatGPT-ready advisory artifact.
* future typed schema / validator.
* future deterministic advisory context assembler.
* future prompt contract.
* future controlled advisory dry run.

### ME-BOOT03 - Bootstrap authoritative universe and local price-history coverage

Status: IMPLEMENTATION COMPLETE / MEMBERSHIP AND PRICE-HISTORY COVERAGE PARTIAL

ME-BOOT03 combines the planned ME-UNIV03 authoritative membership import and
ME-DATA03 local price-history import responsibilities into one bootstrap sprint.

Completed:

* versioned local membership source snapshots under
  `config/market_engine/universes/sources/`;
* central symbol overrides under
  `config/market_engine/universes/symbol_overrides.json`;
* deterministic universe build command
  `market_engine.data.canonical_universe_bootstrap_command`;
* rebuilt canonical universe with 314 instruments;
* source inventory, layer summary, overlap, symbol mapping, unsupported mapping,
  excluded instrument, and report artifacts;
* expanded data-run artifacts for imported, refreshed, insufficient, invalid,
  unsupported, missing, and unresolved readiness outputs;
* ME-EVAL02 refresh rerun after the bootstrap scan.

Known partial status:

* local source snapshots are controlled partials, not complete official index
  membership files;
* full 1,000+ canonical universe coverage is not claimed;
* no operator-supplied forward price-history import root was present;
* ME-EVAL02 remains unresolved with 8 insufficient-forward outcomes and 4
  missing price histories.

Next:

```text
ME-DATA04 - Operator-supplied forward price-history import for ME-EVAL blockers
```

### ME-DATA04 - Build complete canonical local market dataset

Status: OPERATIONAL DATASET PARTIAL

ME-DATA04 built a real local market dataset from reproducible membership and
price-history sources:

* membership: Wikipedia S&P 500 and S&P MidCap 400 tables plus explicit
  project/ETF supplements;
* price history: Yahoo Finance daily OHLCV through the existing `yfinance`
  dependency;
* cutoff date: `2026-07-10`;
* canonical universe: 952 instruments;
* valid local histories: 946;
* valid coverage: 99.37%;
* missing histories: 0;
* invalid histories: 0;
* unsupported mappings: 0.

The sprint remains partial because ME-EVAL02 still reports `resolved: 0`.
The four missing-price-history blockers (`CLS`, `CRDO`, `IREN`, `VRT`) now have
valid local histories, but all 12 evaluation candidates still require future
forward-horizon data.

Next:

```text
ME-DATA05 - Post-cutoff forward outcome refresh after future trading days become available
```
