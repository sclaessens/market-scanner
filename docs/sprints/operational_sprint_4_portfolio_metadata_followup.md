# Operational Sprint 4 Portfolio Metadata Follow-up

## 1. Status

Status: INVESTIGATION FOLLOW-UP

This document records the Portfolio Metadata / Sector Exposure blocker discovered after the staged MVP Fundamental data source implementation.

This is a documentation-only governance follow-up. It does not authorize implementation, runtime code changes, tests, generated artifact changes, CSV edits, provider integration, Reporting changes, Telegram changes, or Decision Engine loosening.

No sprint is closed or certified complete by this document.

## 2. Investigation Context

The staged MVP Fundamental data source implementation has been merged.

The Fundamental Layer now supports optional manually maintained raw fundamentals through:

```text
data/raw/fundamentals.csv
```

A local operational verification was performed with a manually prefilled raw fundamentals file.

The Fundamental Layer produced mixed quality states:

- `SUFFICIENT_DATA`: 4 rows
- `PARTIAL_DATA`: 2 rows
- `INSUFFICIENT_DATA`: 285 rows

Confirmed `SUFFICIENT_DATA` tickers:

- GM
- PLD
- TT
- WELL

Confirmed `PARTIAL_DATA` tickers:

- C
- GS

This confirms that the Fundamental data MVP works and that `SUFFICIENT_DATA` can now reach downstream artifacts.

## 3. Decision Engine Evidence

Despite confirmed `SUFFICIENT_DATA` reaching Decision Engine inputs, final decisions remained:

```text
final_action = REVIEW for all 291 rows
```

Focused Decision Engine review for C, GM, GS, PLD, TT, and WELL showed:

- `allocation_decision = REVIEW_REQUIRED`
- `execution_decision = REVIEW_REQUIRED`
- `portfolio_decision_state = PORTFOLIO_METADATA_REVIEW_REQUIRED`
- `opportunity_decision_state = INSUFFICIENT_DECISION_METADATA`
- `arbitration_state = MISSING_METADATA`
- `portfolio_metadata_status = PARTIAL`

The conflict resolution rationale included:

```text
portfolio_metadata_reason=portfolio source available with partial sector metadata
```

Therefore, the Fundamental data blocker has been resolved for the tested rows, but the Decision Engine remains correctly conservative because portfolio metadata is incomplete.

## 4. Portfolio Metadata Evidence

Portfolio metadata investigation found:

```text
data/processed/portfolio_intelligence.csv
```

- 291 rows
- `portfolio_metadata_status = PARTIAL` for all 291 rows

```text
data/processed/final_decisions.csv
```

- 291 rows
- `portfolio_metadata_status = PARTIAL` for all 291 rows

The active portfolio source files do not contain explicit sector metadata.

```text
data/portfolio/portfolio_positions.csv
```

Current columns:

- `ticker`
- `quantity`
- `avg_cost`
- `last_price`
- `market_value`
- `unrealized_pnl`
- `pnl_pct`
- `status`
- `last_action`
- `last_action_at`

```text
data/portfolio/portfolio_review.csv
```

Current columns:

- `ticker`
- `quantity`
- `avg_cost`
- `last_price`
- `pnl_pct`
- `ma20`
- `ma50`
- `ma200`
- `exposure_state`
- `drawdown_state`
- `risk_state`
- `portfolio_reason`
- `reviewed_at`

There is currently no explicit portfolio metadata source with:

- `sector`
- `portfolio_sector`
- `industry`
- `asset_class`
- metadata source
- metadata freshness timestamp
- sector metadata freshness contract

## 5. Root Cause

The root cause is incomplete portfolio sector metadata.

Portfolio Intelligence can identify that portfolio sources are available, but it cannot complete sector exposure classification because the source data has no explicit sector, industry, asset class, metadata source, or metadata freshness fields.

The Decision Engine correctly preserves `REVIEW` because required decision metadata is incomplete.

This is not a Fundamental Layer bug, not a Reporting bug, not a Telegram bug, and not a Decision Engine path bug.

## 6. Current Portfolio Source Limitations

Current portfolio source files provide holdings, prices, exposure state, drawdown state, and risk review fields.

They do not provide a governed descriptive metadata contract for sector exposure.

Current limitations:

- no approved sector source
- no approved industry source
- no asset-class metadata
- no portfolio metadata freshness rule
- no metadata source provenance
- no deterministic handling of missing sector metadata
- no complete portfolio metadata status criteria

These limitations prevent the Decision Engine from distinguishing complete portfolio metadata from partial metadata.

## 7. Governance Interpretation

The Decision Engine must not be loosened to work around partial portfolio metadata.

Portfolio metadata must remain descriptive input. It must not create allocation authority, ranking authority, urgency, conviction, tradeability semantics, hidden filtering, Reporting-based decision logic, or Decision Engine bypass.

Reporting and Telegram must remain communication-only. They must not compensate for missing Decision Engine metadata or infer decisions from sector metadata.

Any fix must define a governed Portfolio Metadata / Sector Exposure contract before implementation.

## 8. Possible Safe Design Directions

### 8.1 Option A — Add Sector Fields Directly to `portfolio_positions.csv`

Description: extend the active holdings file with sector and industry fields.

Potential fields:

- `sector`
- `industry`
- `asset_class`
- `metadata_source`
- `metadata_last_updated`

Advantages:

- simple to read from existing portfolio source
- keeps metadata close to holdings
- minimal additional file management

Risks:

- mixes position quantities and descriptive metadata
- increases chance of manual portfolio source edits affecting holdings fields
- may blur portfolio state and metadata responsibilities
- may make metadata updates look like position updates

Governance fit: acceptable only with strict contract documentation, but not preferred.

### 8.2 Option B — Create Separate Portfolio Metadata Artifact

Description: create a dedicated manually maintained portfolio metadata source artifact.

Proposed artifact:

```text
data/portfolio/portfolio_metadata.csv
```

Candidate columns:

- `ticker`
- `sector`
- `industry`
- `asset_class`
- `currency`
- `metadata_source`
- `metadata_last_updated`

Advantages:

- clean separation between holdings and descriptive metadata
- audit-friendly
- can be updated without touching position quantities
- supports future provider-assisted metadata refresh without changing holdings contract
- easier to validate completeness and freshness independently

Risks:

- introduces another source file
- requires deterministic ticker matching
- requires clear behavior for missing metadata rows and raw-only metadata rows

Governance fit: strongest and recommended.

### 8.3 Option C — Reuse Scanner or Context Sector Metadata If Available

Description: reuse sector metadata from scanner, validation, or context outputs if such metadata exists upstream.

Advantages:

- may avoid a new source file if reliable sector fields already exist
- may align opportunity metadata with portfolio metadata

Risks:

- scanner/context artifacts may not cover portfolio-only holdings
- upstream opportunity metadata may be absent or partial
- reuse could create hidden dependency between opportunity scanning and portfolio source metadata
- sector metadata origin and freshness may be unclear

Governance fit: possible only if repository evidence shows stable source-supported sector fields and a documented freshness contract.

### 8.4 Option D — Hybrid Approach

Description: use a separate portfolio metadata artifact as the authoritative metadata source, while optionally allowing scanner/context sector fields to fill opportunity-only metadata after explicit governance approval.

Advantages:

- keeps portfolio metadata authoritative and auditable
- allows future enrichment from upstream opportunity metadata if governed
- supports staged implementation

Risks:

- requires clear precedence rules
- requires careful prevention of hidden metadata inference

Governance fit: viable later, after the separate artifact contract is established.

## 9. Recommended Direction

Recommended direction: create a governed Level 2 design for a separate portfolio metadata artifact.

Initial proposed artifact:

```text
data/portfolio/portfolio_metadata.csv
```

Candidate columns:

- `ticker`
- `sector`
- `industry`
- `asset_class`
- `currency`
- `metadata_source`
- `metadata_last_updated`

This preserves separation of concerns:

- `portfolio_positions.csv` remains current holdings and position state
- `portfolio_review.csv` remains portfolio review and risk observations
- `portfolio_metadata.csv` becomes descriptive metadata source only

The future design should define:

1. row identity and duplicate handling
2. required columns
3. allowed sector taxonomy
4. freshness rules
5. missing metadata handling
6. metadata completeness status
7. interaction with Portfolio Intelligence
8. Decision Engine-safe metadata propagation
9. tests required before implementation
10. forbidden semantics and governance checks

## 10. Implementation Authorization Boundary

This document does not authorize implementation.

Future work requires Level 2 design before implementation.

No sprint is closed or certified complete by this document.

The Decision Engine must not be loosened.

Reporting and Telegram must remain communication-only.

Portfolio metadata must remain descriptive input, not allocation authority.

No provider/API integration is authorized by this document.

No generated artifact, processed CSV, runtime file, test file, script, workflow, or report change is authorized by this document.

## 11. Backlog Impact Assessment

New backlog item identified and added to `docs/sprints/project_backlog.md`:

- BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract

Backlog impact assessment:
- New backlog items identified and added to project_backlog.md

## 12. Recommended Next Step

Create a governed Level 2 design for portfolio metadata and sector exposure source integration.

The Level 2 design should prefer a separate `data/portfolio/portfolio_metadata.csv` artifact unless repository evidence strongly supports another source.

Implementation must not begin until the Level 2 design is reviewed and approved.