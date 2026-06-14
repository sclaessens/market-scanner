# Backlog and Sprint Operating Model

Status: ACTIVE

## 1. Purpose

This document defines how the market-scanner project chooses, sizes, executes, closes, and reviews future work.

The goal is to prevent ad hoc sprint selection, protect useful logic, remove obsolete complexity deliberately, and keep documentation, calculations, code placement, and backlog priorities aligned over time.

This document does not authorize implementation, code changes, test changes, runtime behavior changes, source-data changes, generated artifact changes, provider/API usage, scraping, or pipeline execution.

## 2. Operating Principle

Future work must follow this loop:

```text
backlog review
-> triage and scoring
-> sprint selection within capacity
-> implementation or documentation sprint
-> sprint closeout review
-> backlog update
-> logic/document/code placement review
```

No sprint should start only because it feels like the next available task. The backlog must be reviewed before sprint selection.

## 3. Backlog Status Model

Backlog items should remain concise and operational. They do not authorize implementation.

Allowed status values remain those defined in `docs/sprints/project_backlog.md`, but the following working interpretation applies during triage:

| Status | Sprint-planning meaning |
|---|---|
| `CAPTURED` | Recorded but not ready for implementation. |
| `TRIAGED` | Reviewed for relevance, effort, risk, and dependencies. |
| `ANALYSIS REQUIRED` | Needs logic, contract, source-data, or architecture review before planning. |
| `CANDIDATE SPRINT` | Can be proposed for an upcoming sprint. |
| `APPROVED FOR PLANNING` | May be converted into a sprint plan or developer specification. |
| `ACTIVE SPRINT` | Included in an approved sprint scope. |
| `IMPLEMENTED` | Completed and validated through approved sprint execution. |
| `DEFERRED` | Still valid but intentionally postponed. |
| `REJECTED` | No longer desired or no longer aligned with project direction. |

## 4. Backlog Triage Cadence

The backlog must be reviewed:

1. before selecting any new sprint;
2. during sprint closeout;
3. when a sprint changes architecture, calculations, source-data contracts, or runtime responsibilities;
4. when new user insight changes priorities;
5. after any major cleanup, refactor, or implementation sprint.

The review should answer:

- Is this item still relevant?
- Is it now obsolete?
- Has it been implemented?
- Has it been superseded by a better architecture?
- Is it a duplicate or subset of another item?
- Does it need analysis before implementation?
- Does it require source-data evidence?
- Does it affect calculations or ticker-category logic?
- Does it touch Decision Engine authority?
- Does it need to be split into smaller items?

## 5. Effort Points

Each backlog item should receive an effort estimate before it enters sprint planning.

| Points | Meaning |
|---:|---|
| 1 | Small documentation update, reference cleanup, or narrow analysis note. |
| 2 | Small isolated implementation or cleanup with limited tests and no contract change. |
| 3 | Normal feature/refactor/spec with focused tests or moderate documentation impact. |
| 5 | Larger layer, contract, schema, or orchestration change requiring careful validation. |
| 8 | Complex architecture or multi-layer change. Must usually be split before implementation. |

Effort points measure implementation and validation effort, not importance.

## 6. Governance Risk Rating

Each backlog item should also receive a governance risk rating.

| Risk | Meaning |
|---|---|
| `LOW` | Documentation, reference, or isolated cleanup. No runtime authority boundary risk. |
| `MEDIUM` | Runtime code, tests, or operational behavior change with clear boundaries. |
| `HIGH` | Data contracts, pipeline sequencing, generated artifact contracts, or cross-layer dependencies. |
| `CRITICAL` | Decision Engine authority, allocation semantics, hidden filtering risk, or reporting decision semantics. |

High or critical items require explicit architecture review or developer specification before implementation.

## 7. Sprint Capacity Rule

Default sprint capacity:

```text
maximum sprint capacity = 5 effort points
maximum main themes per sprint = 1
```

A sprint may include:

- one 5-point item; or
- one 3-point item plus one or two small supporting items; or
- several 1-point documentation/cleanup items if they share the same theme.

Do not combine unrelated major themes in one sprint.

Any 8-point item should normally be split before implementation.

## 8. Sprint Selection Rule

A sprint may be selected only when:

1. the backlog has been reviewed;
2. candidate items have effort points;
3. candidate items have governance risk ratings;
4. dependencies are known;
5. the sprint fits within capacity;
6. the sprint has one primary theme;
7. the sprint has explicit non-scope;
8. required docs or specifications exist;
9. the user has approved the sprint direction.

## 9. Sprint Closeout Review

Every sprint closeout must answer these questions:

1. Was the sprint scope completed?
2. Did any backlog item become implemented, obsolete, split, deferred, or rejected?
3. Were new backlog items identified?
4. Does the active logic still make sense?
5. Are calculations still documented in the calculation registry?
6. Are any calculations duplicated or in the wrong layer?
7. Does any code now live in the wrong file or module?
8. Did any useful logic become stranded in a legacy file?
9. Do active documents still reflect current truth?
10. Should any document be replaced rather than patched?
11. Is the Decision Engine authority boundary intact?
12. Is Reporting still communication-only?
13. Is the next sprint still the right priority based on the backlog?

## 10. Documentation Replacement Policy

The project must avoid unlimited document growth.

When an active document no longer fits its intended structure, the preferred process is:

1. design the better replacement structure;
2. create the replacement document only if the existing document cannot be cleanly updated;
3. migrate the useful content into the replacement;
4. mark or move the old document to archive;
5. update active references;
6. avoid keeping two active documents that claim the same authority.

Do not create new active documents just to add a small note. Update the existing active document when the structure still fits.

## 11. Calculation Governance

All meaningful calculations must be documented in the calculation registry before or during implementation.

A calculation registry entry should define:

- calculation name;
- owning layer;
- purpose;
- inputs;
- formula or algorithm;
- output artifact and output field;
- whether it is descriptive or decision-authoritative;
- ticker-category or sector relevance;
- edge cases;
- review triggers;
- downstream consumers;
- tests required.

A calculation may not silently become allocation logic outside the Decision Engine.

## 12. Logic Review Cadence

Logic must be reviewed regularly, not only when something breaks.

A logic review is required:

- after every implementation sprint;
- before adding a new calculation family;
- before adding sector/category-specific behavior;
- after a meaningful failed or misleading signal is observed;
- when the backlog shows repeated pressure around the same area;
- before deleting or migrating code that contains analytical logic.

The review should classify logic as:

| Classification | Meaning |
|---|---|
| `KEEP` | Still correct and well placed. |
| `KEEP_BUT_DOCUMENT` | Still useful but insufficiently documented. |
| `MOVE` | Useful but belongs in another layer or module. |
| `SPLIT` | Too broad and should become smaller calculations or states. |
| `REPLACE` | Still needed but the current implementation is too complex or flawed. |
| `REMOVE_AFTER_MIGRATION` | Obsolete after useful logic is migrated or preserved elsewhere. |
| `REQUIRES_REVIEW` | Not enough evidence to decide. |

## 13. Ticker Category and Sector Logic Principle

The project may evolve toward ticker-category-aware logic because sectors and business types do not react to the same signals in the same way.

Potential future categories include, but are not limited to:

- semiconductors;
- software;
- retail;
- energy;
- industrials;
- financials;
- biotech or healthcare innovation;
- defensive compounders;
- cyclical growth;
- commodity-sensitive businesses.

Ticker/category classification is descriptive upstream. It may inform which calculations or context states are relevant, but it must not create hidden allocation authority outside the Decision Engine.

Any future ticker-category model requires:

- explicit category definitions;
- source-supported ticker assignment rules;
- calculation relevance mapping;
- tests;
- Decision Engine boundary review before downstream usage.

## 14. Role Integration

The active role matrix in `docs/active/roles_and_responsibilities.md` defines role ownership.

For this operating model:

- PM / Scrum Master owns backlog hygiene, sprint capacity, and sprint selection discipline.
- Strategy and Logic Steward owns logic review, ticker-category logic, and calculation registry hygiene.
- Financial Analyst owns financial metric meaning and financial interpretation.
- Functional Analyst owns workflow and acceptance criteria.
- Technical Analyst / Architect owns layer placement, contracts, and implementation boundaries.
- Governance Auditor owns doctrine compliance and authority-boundary review.
- Developer / Codex implements only after approved scope.

## 15. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

This document changes the operating model for backlog usage and sprint selection. It does not add a new feature request or implementation item by itself.

## 16. Validation

Documentation-only validation for this change should confirm:

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
- no tests run unless explicitly needed.