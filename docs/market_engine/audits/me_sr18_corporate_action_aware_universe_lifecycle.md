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

Manifest v3 binds the canonical universe, active universe, retained history,
lifecycle registry, provenance, freshness, history coverage, and persisted
file bytes. The trusted consumer independently reconstructs those bindings and
passes only active instruments to ME-RUN30. True stale data, provider errors,
unexplained short history, malformed evidence, unknown schemas, checksum
tampering, or incomplete publication sets remain fail closed.

The PR review found three additional integrity gaps: post-delisting rows could
be called healthy retained history, short history needed no session-level
proof, and checksum-valid evidence was not semantically bound to its authority
or transition. The review hardening introduces lifecycle registry v2 and
manifest/consumer v3. It requires an exact inactive end-date boundary,
exchange-calendar-complete listing coverage after a bounded one-session start
tolerance, and governed evidence identity, host, enum, and
announcement/completion support. The historical BLD and JHG tails are
preserved byte for byte but now block publication instead of being declared
healthy.

The third review found a publication-control gap independent of data-byte
validity. A mixed refresh could update one instrument, preserve a technically
valid stale or failed peer file, end `degraded`, and still set
`publication_required=true`. The publisher also used `always()` and two
`--allow-degraded` validations, so it could run after the upstream job was
marked red. The hardened contract now requires a completed runtime decision,
a successful upstream job, and strict non-degraded validation before any
`market-data` publication can start.

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
| SOLS | Solstice Advanced Materials spin-off; Nasdaq when-issued listing | 2025-10-20 | 2025-10-30 | n/a | 2025-10-20 active | [Honeywell distribution release](https://www.honeywell.com/us/en/news/press-releases/2025/10/honeywell-board-of-directors-sets-record-date-and-announces-expected-timing-for-spin-off-of-solstice-advanced-materials), [SEC Form 8-K](https://www.sec.gov/Archives/edgar/data/2064953/000162828025047305/sols-20251029.htm) |

The earliest official when-issued session is the governed `listing_start_date`.
The separate `regular_way_listing_date` prevents a false assertion that valid
when-issued observations predate the listing. These dates have separate
validation responsibilities: history coverage begins at `listing_start_date`,
while evidence claiming `listing_completion` cannot be published before
`regular_way_listing_date`. Publication on that regular-way date is valid at
the registry's day-level granularity. Tickers appear only in governed records,
regression fixtures, and this audit; runtime behavior branches only on
validated lifecycle fields.

## Lifecycle Contract

Registry:

```text
config/market_engine/universes/instrument_lifecycle.json
```

Schema:

```text
market-engine-instrument-lifecycle-registry-v2
```

Each governed record binds:

- canonical instrument ID, ticker, issuer, and exchange;
- record status and status effective date;
- when-issued/start and regular-way dates for recent listings;
- final expected trading date for terminated listings;
- reason, corporate-action type, and reliable successor/acquirer identity;
- controlled status, reason, action, exchange, authority, source-type, and
  transition-support enums;
- one or more primary evidence records bound to canonical instrument ID,
  ticker, exchange, HTTPS host, publication date, and retrieval timestamp;
- fixed SEC/exchange host policy and checksum-bound issuer/acquirer official
  host metadata;
- a SHA-256 provenance checksum over canonical record bytes.

Records absent from the registry inherit the canonical snapshot status. A
governed active record is `pending` before its listing start and `active` from
that date. A governed inactive record remains active before the inactive
effective date and becomes inactive on that date. A future event is therefore
never applied early.

The loader rejects malformed JSON, unsupported schema versions or enums,
duplicate or unknown instrument IDs, ticker/exchange mismatches,
authority/source-type/host mismatches, unbound issuer/acquirer hosts,
contradictory dates, evidence identity mismatches, future publication dates,
announcement-only completed transitions, and missing or mismatched provenance
checksums. Registry v1 is explicitly rejected; it is not silently reinterpreted
under v2 semantics.

For an active listing, `listing_schedule` and `listing_completion` remain
distinct evidence capabilities. A schedule may be published before trading
starts. Completion evidence is rejected with
`LISTING_COMPLETION_BEFORE_REGULAR_WAY` when its publication date is earlier
than the governed `regular_way_listing_date`, including when the date falls
inside a valid when-issued trading period. Missing or contradictory listing
metadata continues to fail closed before this evidence check.

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
staging for validation. They are not deleted, truncated, rewritten, or treated
as stale active instruments.

The inherited BLD and JHG CSVs contain flat, zero-volume provider-dated
observations on 2026-07-01 and 2026-07-02 even though the official final
expected trading session is 2026-06-30. The review reproduction proved that
the original comparison checked only `actual_end < expected_end`, so these
tails were accepted. The hardened contract requires equality. It classifies
these two files as `extends_after` with
`RETAINED_HISTORY_EXTENDS_AFTER_DELISTING`, preserves the exact bytes for
audit, sets their row validation to blocked, and makes
`publication_set_valid=false`. It never auto-deletes or rewrites historical
data. GTLS ends exactly on 2026-07-16 and is the real aligned control case.

For an aligned retained file, the exact-fileset contract binds the inactive
path, file checksum, expected and actual end date, lifecycle provenance,
`freshness_status=not_expected`,
`history_coverage_status=retained_inactive`, and
`retained_history_boundary_status=aligned`. Before/after anomalies remain in
the run evidence but cannot cross the publication boundary. The trusted
consumer independently rereads the CSV and recomputes the same exact boundary,
so a forged manifest checksum cannot use the prior inactive-record `continue`
to bypass date reconciliation.

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

The history coverage boundary is the governed `listing_start_date`. When this
precedes `regular_way_listing_date`, it is explicitly a when-issued boundary;
regular-way is not substituted for it. The existing exchange profile and
holiday calendar produce every expected completed session through the same
run boundary used for freshness. Weekends and exchange holidays are excluded.

An active short history is `limited_history` only when:

- no observation predates the evidence-bound listing start;
- the first observed session is the first expected session or at most one
  expected session later;
- the latest expected completed session is present;
- every expected session from the accepted start through that boundary is
  present;
- the bounded-session coverage ratio is exactly `1.0` (the manifest also
  exposes the raw ratio including any tolerated first-session lag);
- the base CSV validator has already rejected duplicates and out-of-order
  dates.

The one-session start tolerance is explicit and does not excuse an internal
gap. Any later start, missing internal session, or missing end session is
`insufficient_unexplained` and degrades the run. A listing date after the first
observation remains the controlled
`LISTING_START_AFTER_FIRST_OBSERVATION` failure. Future listings remain
`pending` and are neither refreshed nor called limited. Without lifecycle
evidence, short history remains `INSUFFICIENT_HISTORY_WITHOUT_LISTING_EVIDENCE`.

`limited_history` is not promoted to analytical sufficiency. ME-RUN30 receives
the status, then its existing price-history inspection continues to report the
instrument as analytically insufficient where the 252-row minimum is required.

Key reason codes include:

```text
LIMITED_HISTORY_SINCE_LISTING
INSUFFICIENT_HISTORY_WITHOUT_LISTING_EVIDENCE
HISTORY_START_TOO_LATE_AFTER_LISTING
HISTORY_SESSION_GAPS_AFTER_LISTING
HISTORY_END_BEFORE_EXPECTED_SESSION
INACTIVE_AFTER_COMPLETED_CORPORATE_ACTION
RETAINED_INACTIVE_HISTORY
RETAINED_HISTORY_ENDS_ON_EXPECTED_SESSION
RETAINED_HISTORY_ENDS_BEFORE_EXPECTED_SESSION
RETAINED_HISTORY_EXTENDS_AFTER_DELISTING
RETAINED_HISTORY_DATE_BOUNDARY_INVALID
PRE_LISTING_NOT_EXPECTED
EXPECTED_SESSION_NOT_AVAILABLE
LISTING_START_AFTER_FIRST_OBSERVATION
LISTING_COMPLETION_BEFORE_REGULAR_WAY
PROVIDER_MAPPING_MISSING
PROVIDER_TIMEOUT
PROVIDER_ERROR
PRICE_VALIDATION_FAILED
```

## Manifest and Publication Contract

Manifest schema:

```text
market-engine-me-sr18-canonical-price-freshness-manifest-v3
```

Validation schema:

```text
market-engine-me-sr18-published-price-dataset-validation-v3
```

V3 retains the v2 canonical-, active-, and governed-universe bindings and adds
per-entry retained-history expected/actual end dates and boundary status, plus
listing boundary type, coverage boundary, expected/observed/missing session
counts, initial-session lag, and coverage ratio.

Prior manifest versions are never silently accepted by the v3 consumer. The
first valid v3 run requires a manifest publication even if zero price files
changed, because the schema and lifecycle binding are a meaningful data
change. Subsequent identical v3 runs do not request publication and therefore
cannot create an empty data commit. A run containing the current BLD/JHG
boundary anomalies is not a valid first v3 publication and therefore creates
neither a manifest nor an empty data commit.

`publication_set_valid` answers only whether the staged bytes satisfy their
structural and checksum contract. It does not authorize publication. The
central decision is now equivalent to:

```text
run_status == completed
AND publication_set_valid
AND changed price bytes or a meaningful manifest migration exists
```

Any `degraded` or `failed` run therefore reports
`publication_required=false` and does not materialize a publishable manifest,
even when one instrument was successfully updated and the remaining staged
CSV bytes are technically valid. A completed changed run remains publishable;
a completed identical v3 run remains non-publishable and cannot create an
empty commit.

The existing security model is unchanged:

- executable workflow code is checked out from `main`;
- only `data/processed/*.csv` and the manifest are published to `market-data`;
- feature-branch runs cannot publish;
- the refresh job has read permission and only the isolated publisher has
  write permission;
- the publisher requires the upstream job result `success`, output
  `run-status=completed`, a valid publication set, a required publication, and
  a trusted-main invocation;
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
- exact retained-history expected/actual end dates before inactive records are
  excluded from analysis;
- exchange-calendar listing-session sets and coverage metrics;
- the exact fileset and absence of executable data-branch content;
- overall run status.

It then builds an explicit active-only universe snapshot. This prevents a
directory glob or legacy universe config from reintroducing BLD, JHG, or GTLS
into analysis. ME-RUN30 accepts this already validated snapshot and includes
lifecycle/freshness/history coverage in each analysis row.

Daily Market Scan retains its existing `workflow_run` success gate. A run
containing only current active histories, aligned retained-inactive history,
and fully proven limited history can complete successfully. A retained
boundary anomaly, real stale active ticker, unexplained insufficiency,
provider/validation failure, malformed manifest, or checksum mismatch still
prevents automatic analysis. The refresh workflow's existing
step-level `continue-on-error` remains necessary to read the compact report
and upload diagnostic evidence. The result-reader and evidence-upload steps
retain their diagnostic `always()` conditions, while the final status step
still turns degraded or failed results red. The publisher itself no longer
uses `always()`: it requires an actually successful upstream job and exact
`run-status=completed`. Its two trusted-main validations no longer pass
`--allow-degraded`. Daily Market Scan has no bypass and starts automatically
only after real upstream success.

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

## Review Hardening

The three review findings were reproduced before implementation:

1. An inactive file ending after `delisting_end_date` returned
   `not_expected` / `retained_inactive`; the comparison had no greater-than
   branch.
2. A current ten-row file beginning weeks after an older listing returned
   `limited_history`; the classifier checked only the first date and row
   count.
3. A provenance-checksum-valid record declared `source_authority=sec` while
   using `example.com`; the loader checked only the authority string and HTTPS
   syntax.

The fixes remain generic. Runtime code contains no governed ticker branch.
Retained history uses one exact boundary function in refresh and consumer.
Listing coverage uses the existing market-profile calendar and one explicit
session-set algorithm. Evidence uses one versioned registry with controlled
enums and transition requirements rather than a parallel evidence store.

Lifecycle registry v2 distinguishes evidence capabilities:

```text
listing_schedule
listing_completion
corporate_action_completion
trading_termination
```

`distribution_timing_release` can support only a schedule.
`completion_release` and `form_8_k` may support appropriate completion facts;
an exchange notice may support a listing schedule or trading termination.
Inactive transitions require both completion and trading-termination support
with publication dates between the final session and effective date. A mature
active listing requires schedule and completion support. A future scheduled
listing can remain pending with schedule evidence, but it cannot be projected
active early.

SEC evidence is restricted to `sec.gov`; exchange evidence is restricted to
the governed exchange's domains. Issuer/acquirer hosts are explicit per-record
metadata covered by the record checksum and must correspond exactly to used
authorities. Every evidence item repeats and validates canonical instrument
ID, ticker, and exchange. Checksum-valid but semantically invalid combinations
therefore fail before lifecycle projection.

The second review pass reproduced a further date-boundary defect. The active
evidence validator compared `listing_completion` publication against
`listing_start_date`. For a generic listing with a 2026-06-15 when-issued
start and a 2026-06-29 regular-way start, checksum-valid completion evidence
published on 2026-06-16 was therefore accepted. The regression failed before
the fix because the loader raised no exception. The validator now compares
every evidence entry carrying `listing_completion` against
`regular_way_listing_date` and raises
`LISTING_COMPLETION_BEFORE_REGULAR_WAY` when it is early.

Regression coverage proves rejection both before the when-issued start and
between when-issued and regular-way, acceptance on and after the regular-way
date, separation of announcement-only schedule evidence, and the same generic
boundary for FDXF, HONA, Q, SOLS, and a fictitious ticker. The
exchange-calendar history test separately proves that expected sessions still
begin at the governed `listing_start_date`; the evidence fix does not move
coverage to regular-way.

The third review pass reproduced the publication bypass with two fictitious
active instruments. One received and persisted the expected session; the
other raised a provider timeout while its existing CSV remained structurally
valid. The run measured one changed price file,
`publication_set_valid=true`, and `run_status=degraded`, but the old central
decision returned `publication_required=true`. Workflow contract tests then
confirmed that `publish-market-data` used `always()` without checking
`needs.refresh-and-validate.result` or `run-status`, and both pre-push
validators passed `--allow-degraded`.

The runtime now includes `run_status == "completed"` in the central
`publication_required` expression. Regressions cover a valid update paired
with a provider failure, a stale peer, and an invalid provider payload. Each
case retains the valid staged bytes for diagnosis but returns
`publication_required=false` and writes no publishable manifest. A separate
synthetic checksum-valid degraded manifest proves that diagnostic validation
may still opt into `allow_degraded`, while the default validation used by the
publisher rejects it with `PUBLISHED_DATASET_DEGRADED` and
`PUBLISHED_DATASET_STALE`.

The publisher job gate now requires:

```text
needs.refresh-and-validate.result == success
AND needs.refresh-and-validate.outputs.run-status == completed
AND trusted-publish == true
AND publication-required == true
AND publication-set-valid == true
```

Contract tests cover upstream failure, cancellation, or skipping; degraded,
failed, unknown, empty, or missing run statuses; and an untrusted
feature-branch invocation. Only an upstream-successful completed trusted run
with a valid, required publication passes. `always()` is absent from the
publisher job, and
`--allow-degraded` is absent from both publisher validations. The diagnostic
step behavior, final red status, permissions, trusted-code checkout,
data-only branch, and Daily Market Scan success gate remain unchanged.

The seven records were reverified from official SEC, issuer, or acquirer
sources. The inaccessible Honeywell IR completion URL for SOLS was replaced by
Solstice's official SEC Form 8-K; its dates and lifecycle decision are
unchanged. BLD/JHG real tails and GTLS exact end were inspected from current
`origin/market-data`. No price CSV in this repository or the data branch was
edited by this review fix.

## Offline Validation

Deterministic tests cover effective dates before/on transition, future
inactive events, pre-listing pending state, provider non-invocation, byte-for-
byte inactive retention, exact active/retained filesets, active-only consumer
input, listing-bound limited history, unexplained insufficiency, stale/provider
failures, listing-date contradictions, malformed evidence, missing provenance,
unknown lifecycle/manifest schemas, deterministic checksums and ordering,
lifecycle/file tampering, partial-success isolation, schema migration,
idempotent no-commit behavior, workflow security, and approval non-generation.
Active evidence tests additionally cover completion before when-issued,
between when-issued and regular-way, on regular-way, after regular-way,
announcement-only schedules, and all four governed recent listings.
Publication tests additionally cover mixed update/provider-failure,
update/stale, and update/validation-failure runs; a technically valid degraded
fileset; completed changed and completed identical runs; strict publisher
validation; the upstream result/status truth table; feature-branch denial; and
the Daily Market Scan success-only handoff.

Final local command results:

- targeted lifecycle-aware refresh and existing SR17 contract tests:
  107 passed;
- complete Market Engine data suite: 377 passed;
- complete Market Engine suite: 1,402 passed;
- complete repository suite: 2,069 passed;
- Ruby standard-library parsing: both relevant workflow YAML files valid;
- `actionlint`: not installed locally and not added as an uncontrolled
  dependency;
- `git diff --check`: passed.

An additional read-only inspection of current `origin/market-data` commit
`525ef93fcc6612726ab65d0b996d8cf5fc56e5db` produced:

```text
BLD:  3 rows, 2026-06-30 through 2026-07-02; expected end 2026-06-30
JHG:  3 rows, 2026-06-30 through 2026-07-02; expected end 2026-06-30
GTLS: 384 rows, final date 2026-07-16; expected end 2026-07-16
FDXF: 40/40 expected sessions through 2026-07-23
HONA: 27/27 expected sessions through 2026-07-23
Q:   185/185 expected sessions through 2026-07-23
SOLS: 190/190 expected sessions through 2026-07-23
```

The four recent listings therefore satisfy the v3 session contract on the
current data tip. GTLS satisfies the retained boundary. BLD and JHG
deterministically fail the retained boundary, so the hardened runtime must
block publication before any live provider outcome can make the overall run
green.

An offline run against those exact bytes, evaluated through the completed
2026-07-23 US session, measured:

```text
run_status:                    failed
publication_set_valid:         false
publication_required:          false
empty_commit_required:         false
active / inactive / pending:   949 / 3 / 0
already_current / stale:       948 / 1
not_expected / failed:         1 / 2
sufficient / limited:          945 / 4
retained / not_applicable:     1 / 2
```

BLD and JHG were the two failed boundary rows; GTLS was the one aligned
retained row. The one real stale active history remained stale with the
injected offline-empty provider, proving that lifecycle hardening does not
hide a freshness failure. Source and staged SHA-256 checksums matched exactly
for all three inactive files:

```text
BLD  f5503fa97cc9bfe651728dd27419249a1f329c68bf619f9126c6d1a36fb0f0cb
JHG  05f370cf792ea44a360cf61501b564054167fa28c72d038f16844a6caa7be5fa
GTLS 1b59ccd05928b59b797379300f145e81b59f55cf1de5dc675cf1ad6284f0accb
```

## Local Proof and Live Proof

Locally proven:

- generic effective-date transitions;
- 952 canonical to 949 active plus three retained-inactive projection;
- no provider calls for inactive or pending instruments;
- retained history byte preservation plus exact before/aligned/after boundary
  classification;
- current recent listings as freshness-healthy and analytically limited;
- full calendar-session proof for FDXF, HONA, Q, SOLS, and arbitrary listings;
- unexplained short history and true freshness/provider failures remain red;
- every non-completed runtime returns `publication_required=false`, including
  mixed partial-success runs with otherwise valid staged bytes;
- the publisher requires upstream success plus exact `run-status=completed`
  and performs two strict validations without `--allow-degraded`;
- failed, cancelled, skipped, degraded, unknown, empty, untrusted, or
  non-required publication states cannot enter the publisher;
- manifest v3 canonical serialization, checksums, lifecycle binding, retained
  boundaries, session coverage, and exact
  fileset reconciliation;
- active-only downstream analysis input;
- registry v1 and prior manifest rejection, meaningful v3 publication, and
  subsequent no-empty-commit
  behavior;
- unchanged workflow permission and trusted-code boundaries.

Only a post-merge canary can prove:

- current real provider availability and session coverage;
- live rejection of the inherited BLD/JHG tails without rewriting them;
- the first red workflow skips the publisher job, creates no `market-data`
  commit, and does not start Daily Market Scan;
- after separately reviewed data remediation, a normal remote `market-data`
  v3 publication;
- the real run's active/retained/limited counts;
- a green workflow when no genuine stale/provider failure exists;
- automatic Daily Market Scan activation and acceptance;
- a second identical run creates no empty `market-data` commit.

## Post-Merge Canary

1. Merge ME-SR18 to `main`.
2. Open **Actions → Canonical Price Refresh → Run workflow**.
3. Select `main`, set `publish=true`, and start the run.
4. Confirm canonical size 952, active size 949, retained inactive count 3.
5. Confirm BLD and JHG are inactive, are not sent to the provider, retain their
   original checksums, and fail with
   `RETAINED_HISTORY_EXTENDS_AFTER_DELISTING`.
6. Confirm GTLS is inactive, is not sent to the provider, and reports
   `RETAINED_HISTORY_ENDS_ON_EXPECTED_SESSION`.
7. Confirm the current inherited anomaly makes
   `publication_set_valid=false`, the workflow red, creates no `market-data`
   commit, and does not start Daily Market Scan.
8. Treat that first run as the required live proof of the fail-closed review
   fix. Do not bypass it and do not auto-truncate either historical file.
9. Resolve BLD/JHG only through a separate, explicitly reviewed data-remediation
   decision that establishes the authoritative retained bytes.
10. After such remediation, rerun from `main` and confirm FDXF, HONA, Q, and
    SOLS are `limited_history` only when their manifest v3 session counts show
    complete coverage from `listing_start_date`, and their
    `listing_completion` evidence is on or after `regular_way_listing_date`.
11. Confirm any `stale`, `failed`, `unsupported`, or
    `insufficient_unexplained` active count is zero; investigate any genuine
    failure rather than bypassing it.
12. Confirm manifest v3, lifecycle registry v2 checksum, active/governed
    universe checksums, exact fileset, retained boundaries, session coverage,
    and all file checksums validate.
13. Confirm `market-data` advances through one normal non-force commit
    containing data and the v3 manifest only.
14. Confirm Canonical Price Refresh concludes success and Daily Market Scan
    starts automatically through `workflow_run`.
15. Confirm the consumer validates the boundary and analyzes 949 active
    instruments, with recent-listing limitations visible.
16. Run the same workflow again after no newly completed session and confirm
    it creates no empty `market-data` commit.

Production correction is operational only after those live checks pass.

## Locked Baseline Order

```text
ME-DATA10 -> ME-SR17 -> ME-SR18 -> post-merge canary -> ME-DATA11
```
