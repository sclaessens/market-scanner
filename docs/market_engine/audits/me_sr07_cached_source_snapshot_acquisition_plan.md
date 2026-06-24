# ME-SR07 — Cached-source snapshot acquisition plan for missing expanded universe entries

## 1. Purpose

ME-SR07 prepares a controlled acquisition plan for cached-source snapshots needed by expanded Professional Swing Universe entries that currently lack usable cached input data.

This sprint is planning only. It does not acquire source data, implement provider access, add live fetching, create snapshots, stage snapshots, change runtime analysis semantics, or change Decision Engine behavior.

ME-SR07 does not make any security, recommendation, ranking, order, trade, allocation, or execution decision. Missing data remains missing until a later approved sprint supplies valid evidence and passes validation gates.

## 2. Current Repository Baseline

The current repository contains the following relevant baseline:

| Area | Exact path | Baseline finding |
|---|---|---|
| Editable Professional Swing Universe | `data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv` | Current tracked universe contains 53 entries after the expanded-universe path. |
| SEC CompanyFacts raw snapshot contract | `docs/market_engine/source_refresh/me_sr01_sec_companyfacts_raw_snapshot_cache.md` | Defines `sec-companyfacts-raw-v1` envelopes under `data/market_engine/source_snapshots/sec_companyfacts/<run_id>/raw/`. |
| Snapshot persistence implementation | `src/market_engine/source_refresh/sec_companyfacts_snapshots.py` | Defines raw snapshot envelope metadata, no-overwrite behavior, manifest writing, cached loading, and fail-closed load errors. |
| Required source field mapping | `src/market_engine/source_intake/sec_companyfacts_fields.py` | Requires `revenue`, `net_income`, `operating_cash_flow`, and `capital_expenditures` for source-support classification. |
| Source-support classifier | `src/market_engine/source_support/professional_swing.py` | Classifies rows as `supported_cached`, `missing_snapshot`, `missing_required_source_field`, `manual_review_only`, and related fail-closed states. |
| Expanded source-support wrapper | `src/market_engine/source_support/expanded_professional_swing.py` | Reuses the Professional Swing source-support classifier for expanded-universe classification. |
| Expanded source-support documentation | `docs/market_engine/source_support/me_sr06_expanded_professional_swing_universe_source_support.md` | Documents `market-engine-expanded-professional-swing-source-support-v1`. |
| Expanded cached-source scan documentation | `docs/market_engine/run_reports/me_run23_expanded_supported_universe_cached_source_scan.md` | Documents `market-engine-expanded-supported-universe-cached-source-scan-v1`. |
| Portfolio-context fixture follow-up | `docs/market_engine/run_reports/me_run24_non_production_portfolio_context_fixture_expanded_scans.md` | Confirms ME-RUN24 addressed the Portfolio Review context blocker only. |
| Tracked source snapshots | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/` | Contains 12 raw SEC CompanyFacts snapshots plus `snapshot_metadata.json`, `ticker_manifest.csv`, and `provider_errors.csv`. |
| Provider error evidence | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/provider_errors.csv` | Records HO as unsupported for the ME-SR02 SEC CompanyFacts bounded snapshot bundle. |

Known prior sprint facts:

- ME-SR05 introduced deterministic Professional Swing Universe source-support classification.
- ME-SR06 reused ME-SR05 classification for the expanded Professional Swing Universe.
- ME-RUN23 proved expanded cached-source selection and selected 12 `supported_cached` entries.
- ME-RUN24 added an explicit non-production portfolio-context fixture path and did not address snapshot acquisition.

Local untracked `artifacts/market_engine/` content was inspected only as local run evidence. It is not treated as repository source truth and is not committed by this sprint.

## 3. Expanded-Universe Snapshot Coverage Inventory

Inventory source of truth for ME-SR07:

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
data/market_engine/source_snapshots
```

Inspection method:

```text
market_engine.source_support.expanded_professional_swing.classify_expanded_professional_swing_universe_source_support
```

Classification summary observed locally:

| Status | Count |
|---|---:|
| total expanded universe entries | 53 |
| supported cached | 12 |
| missing snapshot | 38 |
| manual review only | 3 |
| unsupported SEC CompanyFacts | 0 |
| missing required source field | 0 |
| malformed or unreadable source artifact | 0 |
| ambiguous identity | 0 |
| excluded | 0 |

Coverage inventory:

| Entry / ticker | Universe status | Cached-source snapshot present? | Snapshot path if present | Missing source families | Blocking reason | Proposed acquisition status |
|---|---|---|---|---|---|---|
| NVDA | candidate | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/NVDA_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| AMD | candidate | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/AMD_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| ASML | needs_source_mapping | present and usable for current cached-source classifier only; broader non-US source-family governance remains unresolved | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/ASML_companyfacts.json` | none for current classifier; future non-US source-family/source-mapping governance still required | none for current classifier; broader non-US governance unresolved | currently usable by existing classifier; do not treat ME-SR07 as resolving future non-US source-family governance |
| META | candidate | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/META_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| MSFT | candidate | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/MSFT_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| VRT | watching | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/VRT_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| CLS | watching | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/CLS_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| CRDO | watching | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/CRDO_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| IREN | watching | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/IREN_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| COST | candidate | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/COST_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| AVGO | candidate | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/AVGO_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| TSM | candidate | present and usable | `data/market_engine/source_snapshots/sec_companyfacts/me-sr02-canonical-universe-20260619T000000Z/raw/TSM_companyfacts.json` | none | none | already supported; no ME-SR07 acquisition |
| AAPL | candidate | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| AMZN | candidate | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| GOOGL | candidate | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| TSLA | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| PLTR | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| ARM | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| MU | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| MRVL | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| ANET | candidate | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| DELL | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| HPE | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| ETN | candidate | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| GEV | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| CEG | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| NRG | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| APP | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| RDDT | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| NET | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| DDOG | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| SNOW | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| MDB | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| SHOP | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| MELI | candidate | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| UBER | candidate | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| COIN | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| MSTR | manual_review_only | not eligible for cached-source support while manual-review-only | none | none until governance status changes | Universe row is marked manual_review_only. | not eligible; no acquisition until explicit governance change |
| RKLB | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| ASTS | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| SOFI | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| HOOD | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| HIMS | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| ELF | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| CELH | watching | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| LLY | candidate | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| NVO | needs_source_mapping | missing for current SEC CompanyFacts local support | none | approved source identity mapping; SEC CompanyFacts raw snapshot or future approved equivalent; run metadata; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | blocked pending source identity/source-family decision |
| ISRG | candidate | missing | none | SEC CompanyFacts raw snapshot; run metadata; ticker manifest row; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | candidate for future controlled acquisition |
| SMCI | manual_review_only | not eligible for cached-source support while manual-review-only | none | none until governance status changes | Universe row is marked manual_review_only. | not eligible; no acquisition until explicit governance change |
| HO | manual_review_only | not eligible for cached-source support while manual-review-only | none | none until governance status changes | Universe row is marked manual_review_only. | not eligible; no acquisition until explicit governance change |
| RHM | needs_source_mapping | missing for current SEC CompanyFacts local support | none | approved source identity mapping; future approved source family; run metadata; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | blocked pending source identity/source-family decision |
| RR | needs_source_mapping | missing for current SEC CompanyFacts local support | none | approved source identity mapping; future approved source family; run metadata; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | blocked pending source identity/source-family decision |
| ADYEN | needs_source_mapping | missing for current SEC CompanyFacts local support | none | approved source identity mapping; future approved source family; run metadata; validation evidence | No approved local SEC CompanyFacts source snapshot was found for the universe ticker. | blocked pending source identity/source-family decision |

## 4. Required Cached-Source Families

The existing cached-source local execution path currently recognizes SEC CompanyFacts raw snapshots as the source family for this universe path.

| Source family | Purpose | Expected shape | Minimum metadata | Validation requirement | Failure behavior |
|---|---|---|---|---|---|
| Company identity / ticker identity metadata | Tie the universe entry to the cached source entity. | Existing snapshot envelope metadata and `ticker_manifest.csv` row. | ticker, CIK when applicable, source name, snapshot id, local path. | Ticker and CIK must match expected entity when supplied; source name must be recognized. | Missing or mismatched identity fails closed and cannot be treated as supported. |
| SEC CompanyFacts raw snapshot | Preserve raw fundamental source evidence for current cached-source execution. | JSON envelope with `metadata` and `raw_payload`, format `sec-companyfacts-raw-v1`. | ticker, CIK, source name, fetched timestamp, snapshot id, payload format version. | `load_sec_companyfacts_raw_snapshot` must load the file; field mapping must find required fields. | Missing, invalid, unsupported, mismatched, or incomplete snapshots remain blocked. |
| Source manifest | Provide run-level inventory and reviewability. | `ticker_manifest.csv`, `snapshot_metadata.json`, and provider error rows when applicable. | run id, source name, payload format version, created timestamp, ticker, CIK, snapshot id, path. | Manifest rows must point to existing files and use recognized source family/version. | Missing manifest blocks promotion even if a raw file exists. |
| Acquisition metadata | Record how the snapshot entered local cached-source storage. | Future manifest or sidecar contract; not yet implemented. | acquisition mode, source origin, acquisition timestamp UTC, as-of date if known, operator notes, real/synthetic/derived classification. | Metadata must be complete before an entry is promoted. | Missing metadata leaves status missing or review-blocked. |
| Validation report | Show that cached-source local execution can identify and validate the snapshot without live fetch. | Future validation artifact; command not implemented in ME-SR07. | validation command, validation timestamp, status, missing/stale notes, checksum, source family. | Must pass schema, identity, manifest, checksum, freshness, field mapping, and dry-run discovery gates. | Failed validation cannot be upgraded to supported. |

ME-SR07 does not introduce price, market context, analyst, broker, or execution source families because the existing cached-source contract for this path does not require them.

## 5. Acquisition Rules

Allowed future acquisition modes:

| Mode | Status | Conditions |
|---|---|---|
| Approved bounded source-refresh run | Allowed only in a later approved implementation sprint | Must define ticker set, source family, no-overwrite behavior, manifest writing, validation gates, and operator review. |
| Manual local staging of real source snapshots | Allowed only after a manifest/staging contract exists | Must include explicit provenance, checksum, source origin, acquisition timestamp, as-of date if known, and validation status. |
| Import from previously approved internal cached-source bundle | Allowed only if manifest and checksums are preserved | Must not silently rewrite paths or mark unsupported entries as supported. |
| Synthetic fixture generation | Allowed only for tests | Must be marked synthetic fixture and must never be treated as real cached-source coverage. |

Disallowed modes:

- no ad-hoc web scraping;
- no silent provider fallback;
- no unversioned manual copy-paste;
- no overwriting existing snapshots without manifest update;
- no source file mutation without checksum/provenance update;
- no treating generated test fixtures as real source snapshots;
- no marking missing, stale, partial, malformed, unsupported, or ambiguous evidence as supported;
- no trade/actionable recommendation generation from missing or unvalidated snapshots;
- no broker, Telegram, portfolio, watchlist, or production-output side effects.

Fail-closed rules:

- missing data remains missing;
- stale data remains stale;
- partial data remains partial;
- manual-review-only rows remain non-acquisition candidates until governance changes;
- `needs_source_mapping` rows require an approved source identity/source-family decision before acquisition.

## 6. Snapshot Acquisition Metadata Contract

Existing contracts already define part of the metadata:

- raw snapshot envelope metadata in `src/market_engine/source_refresh/sec_companyfacts_snapshots.py`;
- run metadata in `snapshot_metadata.json`;
- ticker inventory in `ticker_manifest.csv`;
- provider errors in `provider_errors.csv`.

ME-SR07 proposes that every future acquired or staged snapshot must carry the following metadata before it can be considered for support:

| Field | Required? | Notes |
|---|---|---|
| ticker / entry identifier | yes | Must match universe entry. |
| market / listing context | yes | Required for non-US or `needs_source_mapping` cases. |
| source family | yes | Example: `sec_companyfacts`. |
| source origin | yes | Provider, approved bundle, or manually staged source origin. |
| acquisition mode | yes | Bounded source-refresh run, manual staging, approved bundle import, or test fixture. |
| acquisition timestamp UTC | yes | When the local snapshot was acquired or staged. |
| source publication or as-of date | yes when known | Required to support stale-data review. |
| local file path | yes | Must stay under approved source snapshot root. |
| checksum or content hash | yes | Required for no-silent-mutation evidence. |
| schema / format version | yes | Example: `sec-companyfacts-raw-v1`. |
| validation status | yes | Pass/fail/review-blocked with validation timestamp. |
| stale-data status | yes | Explicit state, not implicit. |
| missing-data notes | yes | Empty only when validation proves completeness. |
| operator notes | optional but supported | Must remain non-actionable. |
| real source data / synthetic fixture / derived artifact | yes | Synthetic fixtures cannot count as real coverage. |
| no-overwrite confirmation | yes | Existing snapshots must not be overwritten silently. |
| manifest reference | yes | Must identify run-level manifest or future acquisition manifest. |

This metadata should be formalized in a later sprint rather than embedded ad hoc in ME-SR07 prose.

## 7. Validation Gates

Before a missing expanded-universe entry can be promoted to usable cached-source coverage, all gates below must pass:

1. The canonical universe entry exists in `data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv`.
2. The row is active and not `manual_review_only`, `blocked`, or `rejected`.
3. `needs_source_mapping` rows have an approved source identity/source-family decision.
4. Required source families are present for the approved source path.
5. Raw snapshot envelopes use recognized schema/format versions.
6. Run-level manifests exist and reference the local files.
7. Checksums or content hashes exist.
8. Source dates and acquisition timestamps are recorded.
9. Missing and stale notes are explicit.
10. The source-support classifier can discover the snapshot without provider or network access.
11. Required mapped source fields are present for the current SEC CompanyFacts path: `revenue`, `net_income`, `operating_cash_flow`, and `capital_expenditures`.
12. Ambiguous ticker identity is absent.
13. Malformed or unreadable artifacts are absent.
14. Verification commands pass.
15. Report output remains non-actionable and does not introduce recommendation, ranking, order, allocation, or execution wording.

Any failed gate keeps the entry missing, review-blocked, or unsupported. No gate may silently promote a row.

## 8. Local Operator Workflow Proposal

Future implementation sprint workflow:

1. Generate a deterministic missing expanded-universe coverage list from the current universe and source snapshot root.
2. Split entries into direct SEC CompanyFacts candidates, `needs_source_mapping` blockers, and manual-review-only exclusions.
3. Prepare an acquisition checklist for only eligible missing entries.
4. Acquire or stage source snapshots outside runtime execution according to the approved acquisition mode.
5. Write raw snapshot files under an approved source snapshot run directory.
6. Write or update run metadata, ticker manifest, provider-error records, and future acquisition metadata.
7. Compute checksums.
8. Validate snapshot schema and identity.
9. Run source-support classification without live fetch.
10. Run cached-source batch dry-run only for entries that pass source-support validation.
11. Write local artifacts only when explicitly requested.
12. Audit the result and update coverage inventory.

This workflow is provider-neutral. ME-SR07 does not approve a new provider or source-access mechanism.

## 9. Follow-Up Sprint Candidates

Recommended sequence:

| Sprint | Purpose | Notes |
|---|---|---|
| ME-SR08 — Define cached-source snapshot acquisition manifest contract | Formalize acquisition metadata, checksum, stale-data, validation, and real/synthetic/derived classification fields. | Contract-only sprint before any staging implementation. |
| ME-SR09 — Implement missing expanded-universe snapshot coverage inventory command | Produce deterministic inventory from universe + source snapshot root. | No provider calls; output only. |
| ME-SR10 — Implement manual cached-source snapshot staging validator | Validate manually staged snapshots against manifests, checksums, schema, identity, and required fields. | No live fetch; no acquisition. |
| ME-SR11 — Implement approved bounded acquisition or import workflow | Only after ME-SR08 through ME-SR10 define and validate the gates. | Scope depends on approved source mode. |
| ME-SR12 — Define non-US ticker source-family and source-mapping governance contract | Define how non-US tickers, ADRs, foreign listings, dual listings, and `needs_source_mapping` entries can be admitted into cached-source coverage. | Future governance work only; must define approved source-family rules and source identity mapping before entries such as ASML, NVO, RHM, RR, ADYEN, or similar future rows can be broadly governed. |
| ME-RUN25 — Rerun expanded cached-source coverage audit after staged snapshots exist | Run existing cached-source path after new evidence exists. | Does not belong before actual validated snapshots exist. |

ME-RUN23 should not be reused as a new follow-up name because it is already completed. A later run sprint should use the next available ME-RUN identifier.

ME-SR12 is intentionally future work. It must not acquire snapshots, implement provider access, or mark non-US tickers supported merely because the current classifier can load a snapshot. It must define source identity mapping requirements, including ticker, exchange/listing context, company identity, source entity identifier, and CIK only when applicable. Until that governance exists, non-US and `needs_source_mapping` rows remain fail-closed for broader source-family purposes.

## Backlog and Roadmap Consistency Check

ME-SR07 also checked the canonical backlog and roadmap plus the standalone ME-SR07 backlog and roadmap entries:

- `docs/market_engine/backlog/market_engine_backlog.md`
- `docs/market_engine/backlog/me_sr07_cached_source_snapshot_acquisition_plan_backlog_entry.md`
- `docs/market_engine/roadmap/market_engine_roadmap.md`
- `docs/market_engine/roadmap/me_sr07_cached_source_snapshot_acquisition_plan_roadmap_entry.md`

ME-SR08 remains the next logical sprint because the acquisition manifest contract must be formalized before inventory commands, staging validators, acquisition/import workflows, or non-US source-family governance are implemented.

The non-US ticker/source-family governance item is intentionally recorded as future work. It does not change current ME-SR07 coverage counts, does not promote `needs_source_mapping` entries, and does not change runtime behavior.

## 10. Acceptance Criteria

ME-SR07 is complete when:

- the missing expanded-universe snapshot problem is documented;
- the current coverage state is inventoried as far as the repo allows;
- exact blockers are explicit;
- acquisition metadata requirements are defined;
- validation gates are defined;
- disallowed behaviors are explicit;
- backlog and roadmap are updated;
- no live fetching or runtime provider logic is added;
- no snapshots are acquired or staged;
- no missing entries are marked supported;
- docs checks or tests pass as applicable.

## Verification

ME-SR07 is docs/planning only. No runtime tests are required by code changes because no runtime code or tests changed.

Recommended verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
git diff -- docs/market_engine
git status
```
