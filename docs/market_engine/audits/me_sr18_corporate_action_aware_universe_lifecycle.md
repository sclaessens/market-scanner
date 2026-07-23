# ME-SR18 Corporate-Action-Aware Universe Lifecycle Audit

Status: `implementation_complete_pending_post_merge_canary`

## Executive Summary

ME-SR18 corrects a domain-model gap exposed by the first ME-SR17 production
canary. ME-SR17 successfully refreshed the full 952-instrument universe,
published a checksum-valid exact fileset to `market-data`, and then correctly
prevented Daily Market Scan because the run was degraded. The degradation
combined three different conditions: completed listings, current recent
listings with short history, and a genuinely stale active listing.

ME-SR18 adds one generic, evidence-bound lifecycle projection over the existing
ME-DATA04 canonical universe. It does not replace that universe. At the
2026-07-23 evaluation date, the projection contains 949 active instruments and
three retained-inactive instruments. It also separates operational price
freshness from analytical history coverage. Current recent listings may now be
`limited_history` without making the refresh unhealthy; analyses still receive
that limitation and apply their existing minimum-history block.

Manifest v2 binds the canonical universe, active universe, retained history,
lifecycle registry, provenance, freshness, history coverage, and persisted
file bytes. The trusted consumer independently reconstructs those bindings and
passes only active instruments to ME-RUN30. True stale data, provider errors,
unexplained short history, malformed evidence, unknown schemas, checksum
tampering, or incomplete publication sets remain fail closed.

## ME-SR17 Canary Evidence

Input production run:

```text
https://github.com/sclaessens/market-scanner/actions/runs/29911859221
```

The run proved:

- the refresh covered all 952 canonical instruments;
- 945 instruments were updated;
- six were classified `insufficient`;
- one was classified `stale`;
- no provider result was `failed` or `unsupported`;
- `publication_set_valid` was true;
- the normal, non-force publication to `market-data` succeeded;
- the parent workflow remained degraded;
- Daily Market Scan did not start because its `workflow_run` gate requires
  upstream success.

This is not an ME-SR17 architecture failure. Refresh, atomic publication,
checksum validation, workflow permissions, and the safety gate all behaved as
designed. The canary revealed that the v1 status axis could not distinguish
lifecycle, freshness, and analytical-history sufficiency.

## Structural Root Cause

The ME-DATA04 snapshot recorded all 952 members as active. It had stable
instrument IDs and provider symbols but no effective-dated listing or
delisting contract. ME-SR17 therefore:

1. refreshed every active snapshot member;
2. converted every valid history shorter than 252 rows to freshness
   `insufficient`;
3. treated every `insufficient` or `stale` ticker as run-degrading;
4. supplied ME-RUN30 a directory containing every canonical CSV.

That behavior was safe but structurally too coarse. A completed listing could
remain stale forever. A current new listing could never have 252 observations,
yet appeared operationally unhealthy. Retained inactive files could not be
kept in the exact publication set without a consumer-side active-universe
projection.

## Official Lifecycle Evidence

Evidence was retrieved on 2026-07-23 and is persisted with source authority,
source type, publication date, retrieval timestamp, URL, and record checksum.

| Ticker | Verified lifecycle | Listing/start | Regular way | Final expected session | Effective status date | Official evidence |
|---|---|---:|---:|---:|---:|---|
| BLD | TopBuild acquired by QXO; NYSE listing terminated | n/a | n/a | 2026-06-30 | 2026-07-01 inactive | [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/1633931/000110465926079876/tm2618991d10_8k.htm), [QXO completion release](https://investors.qxo.com/news/news-details/2026/QXO-Completes-Acquisition-of-TopBuild/default.aspx) |
| JHG | Janus Henderson acquired and taken private; NYSE trading halted/delisted | n/a | n/a | 2026-06-30 | 2026-07-01 inactive | [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/1274173/000110465926079401/tm2619303d2_8k.htm) |
| GTLS | Chart Industries acquired by Baker Hughes; NYSE shares cancelled/delisted | n/a | n/a | 2026-07-16 | 2026-07-17 inactive | [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/892553/000119312526305482/d10289d8k.htm), [Baker Hughes completion release](https://investors.bakerhughes.com/news/press-releases/news-details/2026/Baker-Hughes-Completes-Acquisition-of-Chart-Industries/default.aspx) |
| FDXF | FedEx Freight Holding Company spin-off; NYSE when-issued listing | 2026-05-27 | 2026-06-01 | n/a | 2026-05-27 active | [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/2082247/000110465926068521/tm2615735d2_8k.htm), [FedEx completion release](https://investors.fedex.com/news-and-events/investor-news/investor-news-details/2026/FedEx-Completes-Spin-Off-of-FedEx-Freight/default.aspx) |
| HONA | Honeywell Aerospace spin-off; Nasdaq when-issued listing | 2026-06-15 | 2026-06-29 | n/a | 2026-06-15 active | [Honeywell distribution release](https://www.honeywell.com/us/en/news/press-releases/2026/06/honeywell-board-of-directors-sets-record-date-and-announces-expected-timing-for-spin-off-of-honeywell-aerospace-and-honeywell-reverse-stock-split), [Honeywell completion release](https://www.honeywell.com/us/en/news/press-releases/2026/06/honeywell-aerospace-completes-spin-off-from-honeywell-technologies-and-begins-trading-on-nasdaq) |
| Q | Qnity spin-off; NYSE when-issued listing | 2025-10-27 | 2025-11-03 | n/a | 2025-10-27 active | [DuPont distribution release](https://www.investors.dupont.com/news-and-media/press-release-details/2025/DuPont-Board-of-Directors-Approves-Qnity-Distribution/default.aspx), [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/2058873/000119312525261603/d65598d8k.htm) |
| SOLS | Solstice Advanced Materials spin-off; Nasdaq when-issued listing | 2025-10-20 | 2025-10-30 | n/a | 2025-10-20 active | [Honeywell distribution release](https://www.honeywell.com/us/en/news/press-releases/2025/10/honeywell-board-of-directors-sets-record-date-and-announces-expected-timing-for-spin-off-of-solstice-advanced-materials), [Honeywell completion release](https://investor.honeywell.com/news-releases/news-release-details/honeywell-completes-spin-solstice-advanced-materials) |

The earliest official when-issued session is the governed `listing_start_date`.
The separate `regular_way_listing_date` prevents a false assertion that valid
when-issued observations predate the listing. Tickers appear only in governed
records, regression fixtures, and this audit; runtime behavior branches only
on validated lifecycle fields.

## Lifecycle Contract

Registry:

```text
config/market_engine/universes/instrument_lifecycle.json
```

Schema:

```text
market-engine-instrument-lifecycle-registry-v1
```

Each governed record binds:

- canonical instrument ID, ticker, issuer, and exchange;
- record status and status effective date;
- when-issued/start and regular-way dates for recent listings;
- final expected trading date for terminated listings;
- reason, corporate-action type, and reliable successor/acquirer identity;
- one or more primary evidence records;
- a SHA-256 provenance checksum over canonical record bytes.

Records absent from the registry inherit the canonical snapshot status. A
governed active record is `pending` before its listing start and `active` from
that date. A governed inactive record remains active before the inactive
effective date and becomes inactive on that date. A future event is therefore
never applied early.

The loader rejects malformed JSON, unsupported schemas/statuses, duplicate or
unknown instrument IDs, ticker/exchange mismatches, contradictory dates,
missing evidence, non-HTTPS evidence URLs, future publication dates, and
missing or mismatched provenance checksums.

## Active and Retained Data

The canonical ME-DATA04 source remains 952 instruments. The 2026-07-23
lifecycle projection is:

```text
active:             949
retained inactive:    3
pending:              0
```

Inactive instruments are never submitted to the provider after their
effective date. Their current `market-data` CSVs are copied into isolated
staging and remain part of the checksum-bound publication. They are not
deleted, truncated, rewritten, or treated as stale active instruments.

The inherited BLD and JHG CSVs contain provider-dated observations through
2026-07-02 even though the official final expected trading session is
2026-06-30. ME-SR18 does not silently delete or rewrite those already
published rows. It records the authoritative end date separately, freezes the
files after the effective date, and preserves their exact bytes for audit.

The publication exact-fileset contract now binds both active and retained
inactive paths. A retained file is not orphaned: its manifest row contains
`lifecycle_status=inactive`, its dates, lifecycle provenance checksum,
`freshness_status=not_expected`, and
`history_coverage_status=retained_inactive`.

## Freshness and History Coverage

Freshness answers whether the latest expected completed session is present:

```text
updated
already_current
not_expected
stale
failed
unsupported
```

History coverage answers whether analytical minimum history is available:

```text
sufficient
limited_history
insufficient_unexplained
retained_inactive
not_applicable
```

An active short history is `limited_history` only when its first persisted
observation is on or after a checksum-valid listing start. If observations
precede that start, the row fails with
`LISTING_START_AFTER_FIRST_OBSERVATION`. Without listing evidence, a short
history is `insufficient_unexplained` and degrades the run.

`limited_history` is not promoted to analytical sufficiency. ME-RUN30 receives
the status, then its existing price-history inspection continues to report the
instrument as analytically insufficient where the 252-row minimum is required.

Key reason codes include:

```text
LIMITED_HISTORY_SINCE_LISTING
INSUFFICIENT_HISTORY_WITHOUT_LISTING_EVIDENCE
INACTIVE_AFTER_COMPLETED_CORPORATE_ACTION
RETAINED_INACTIVE_HISTORY
PRE_LISTING_NOT_EXPECTED
EXPECTED_SESSION_NOT_AVAILABLE
LISTING_START_AFTER_FIRST_OBSERVATION
PROVIDER_MAPPING_MISSING
PROVIDER_TIMEOUT
PROVIDER_ERROR
PRICE_VALIDATION_FAILED
```

## Manifest and Publication Contract

Manifest schema:

```text
market-engine-me-sr18-canonical-price-freshness-manifest-v2
```

Validation schema:

```text
market-engine-me-sr18-published-price-dataset-validation-v2
```

V2 adds canonical-, active-, and governed-universe checksums and sizes,
lifecycle schema/registry checksum, retained/pending counts, per-entry lifecycle
and provenance fields, separate history-coverage statuses/reasons and totals,
and a manifest-migration decision.

V1 is never silently accepted by the v2 consumer. The first v2 run requires a
manifest publication even if zero price files changed, because the schema and
lifecycle binding are a meaningful data change. Subsequent identical v2 runs
do not request publication and therefore cannot create an empty data commit.

The existing security model is unchanged:

- executable workflow code is checked out from `main`;
- only `data/processed/*.csv` and the manifest are published to `market-data`;
- feature-branch runs cannot publish;
- the refresh job has read permission and only the isolated publisher has
  write permission;
- all pushes are normal non-force pushes;
- publication is revalidated with trusted `main` code before and after
  materialization;
- no code is imported or executed from `market-data`.

## Consumption and Workflow Handoff

The consumer independently loads the ME-DATA04 snapshot and lifecycle registry
from trusted source, evaluates them at validation time, and reconciles:

- both schema versions;
- lifecycle, canonical, active, and governed checksums;
- ordered instrument identities;
- lifecycle dates and provenance per entry;
- freshness and history-coverage totals;
- every declared persisted file checksum and actual CSV date range;
- the exact fileset and absence of executable data-branch content;
- overall run status.

It then builds an explicit active-only universe snapshot. This prevents a
directory glob or legacy universe config from reintroducing BLD, JHG, or GTLS
into analysis. ME-RUN30 accepts this already validated snapshot and includes
lifecycle/freshness/history coverage in each analysis row.

Daily Market Scan retains its existing `workflow_run` success gate. A run
containing only current active histories, valid retained-inactive history, and
explained limited history can complete successfully. A real stale active
ticker, unexplained insufficiency, provider/validation failure, malformed
manifest, or checksum mismatch still prevents automatic analysis. There is no
new `always()` bypass for Daily Market Scan.

## Fundamental and Governance Boundaries

ME-SR18 does not generate or change DATA09/DATA10 approval authority. Every
manifest retains:

```json
{
  "approval_generated": false,
  "approval_required": false,
  "reason_code": "NO_RELIABLE_AUTOMATED_FUNDAMENTAL_FRESHNESS_CONTRACT",
  "status": "not_evaluated"
}
```

Lifecycle evidence is identity and market-state evidence, not investment
approval. The sprint adds no recommendation, ranking, conviction, urgency,
tradeability, portfolio, allocation, Telegram, broker, order, or Decision
Engine behavior.

## Offline Validation

Deterministic tests cover effective dates before/on transition, future
inactive events, pre-listing pending state, provider non-invocation, byte-for-
byte inactive retention, exact active/retained filesets, active-only consumer
input, listing-bound limited history, unexplained insufficiency, stale/provider
failures, listing-date contradictions, malformed evidence, missing provenance,
unknown lifecycle/manifest schemas, deterministic checksums and ordering,
lifecycle/file tampering, partial-success isolation, schema migration,
idempotent no-commit behavior, workflow security, and approval non-generation.

Final local command results:

- targeted lifecycle-aware refresh and existing SR17 contract tests:
  59 passed;
- complete Market Engine data suite: 329 passed;
- complete Market Engine suite: 1,354 passed;
- complete repository suite: 2,021 passed;
- Ruby standard-library parsing: both changed workflow YAML files valid;
- `actionlint`: not installed locally and not added as an uncontrolled
  dependency;
- `git diff --check`: passed.

An additional offline projection of current `market-data` commit `0558b5cd`
produced:

```text
canonical universe:          952
active universe:             949
retained inactive:             3
already_current:             948
not_expected:                  3
stale:                         1 (NSA)
sufficient history:          945
limited_history:               4
insufficient_unexplained:      0
retained_inactive history:     3
```

The projected v2 publication passed trusted validation with `allow_degraded`.
It remained degraded only because NSA still lacked the expected 2026-07-22
session in that fixed snapshot. This proves that lifecycle cases are corrected
without hiding a genuine active-instrument stale result.

## Local Proof and Live Proof

Locally proven:

- generic effective-date transitions;
- 952 canonical to 949 active plus three retained-inactive projection;
- no provider calls for inactive or pending instruments;
- retained history byte preservation;
- current recent listings as freshness-healthy and analytically limited;
- unexplained short history and true freshness/provider failures remain red;
- manifest v2 canonical serialization, checksums, lifecycle binding, and exact
  fileset reconciliation;
- active-only downstream analysis input;
- v1-to-v2 meaningful manifest publication and subsequent no-empty-commit
  behavior;
- unchanged workflow permission and trusted-code boundaries.

Only a post-merge canary can prove:

- current real provider availability and session coverage;
- a normal remote `market-data` v2 publication;
- the real run's active/retained/limited counts;
- a green workflow when no genuine stale/provider failure exists;
- automatic Daily Market Scan activation and acceptance;
- a second identical run creates no empty `market-data` commit.

## Post-Merge Canary

1. Merge ME-SR18 to `main`.
2. Open **Actions → Canonical Price Refresh → Run workflow**.
3. Select `main`, set `publish=true`, and start the run.
4. Confirm canonical size 952, active size 949, retained inactive count 3.
5. Confirm BLD, JHG, and GTLS are `inactive` / `not_expected` and no provider
   request was made for them.
6. Confirm their persisted file checksums still match the prior publication.
7. Confirm FDXF, HONA, Q, and SOLS are current and `limited_history`.
8. Confirm any `stale`, `failed`, `unsupported`, or
   `insufficient_unexplained` count is zero; if a real one occurs, accept the
   degraded result and investigate it rather than bypassing it.
9. Confirm manifest v2, lifecycle registry checksum, active/governed universe
   checksums, exact fileset, and all file checksums validate.
10. Confirm `market-data` advanced through one normal non-force commit
    containing data and the v2 manifest only.
11. Confirm Canonical Price Refresh concludes success.
12. Confirm Daily Market Scan starts automatically through `workflow_run`.
13. Confirm it validates the published boundary and analyzes 949 active
    instruments, with recent-listing limitations visible.
14. Inspect compact artifacts and logs for unexpected provider calls or
    secrets.
15. Run the same workflow again after no newly completed session.
16. Confirm the second run creates no empty `market-data` commit.

Production correction is operational only after those live checks pass.

## Locked Baseline Order

```text
ME-DATA10 -> ME-SR17 -> ME-SR18 -> ME-DATA11
```
