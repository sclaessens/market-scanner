# ME-SR23 Corporate-Action Lifecycle Cutoff Remediation Audit

Status: `completed_with_blockers`

## Acquisition Identity and Absence Consumer Binding Remediation

This section is the current security-review conclusion. It supersedes the
earlier statement that an internally self-consistent provider envelope alone
was an adequate code-path identity boundary.

The remaining acquisition bypass existed because the former capture helper
accepted provider, adapter, instrument, ticker, provider symbol, exchange, and
request identity as independent caller strings. The scheduled production
acquisition path returns `yfinance` DataFrames and did not invoke that helper.
A downstream actor could therefore relabel the generic bars payload and rebuild
every artifact, reference, receipt, and publication checksum.

Envelope v2 removes the free production builder. A
`RegisteredMarketPriceAdapter` is constructed from the selected source-policy
route and authoritative instrument record. It derives adapter identity,
instrument ID, canonical ticker, provider symbol, exchange, currency, parser,
and source-policy identity. Its request object binds the registered provider
symbol, inclusive/exclusive window, timezone, endpoint method, parameters, and
pagination before response bytes are accepted. Unmapped aliases and symbol
overrides fail before storage.

Freshness manifest v10 records these contracts. Every capture now writes both
a content-addressed response envelope and a
content-addressed acquisition-run manifest. The run manifest binds run ID,
adapter and provider route, instrument identity, provider symbol, exchange,
request digest, artifact and raw-response digests, retrieval time, policy ID,
locator, producer component, and schema. Artifact references carry the
independent run-manifest locator and digest. Replay reloads and hashes both
objects and requires exact manifest/envelope/reference equality. Publication
also requires the exact declared artifact and acquisition-manifest fileset.
Consequently, rebuilding all downstream envelopes, digests, filenames,
locators, receipts, and publication checksums cannot replace the original
acquisition-run identity.

This is not a cryptographic provider attestation. The trusted components are
the repository-governed instrument and source-policy registries, the registered
adapter request/response boundary, trusted acquisition-run storage, and trusted
publisher code. An actor able to rewrite all of those components, their
storage, and the executing code is not cryptographically stopped. The
repository has no signing-key, HMAC-secret, remote-attestation, or external
append-only-ledger architecture, and this remediation does not invent one.
Downstream publication and receipt code cannot create acquisition identity;
only the registered adapter boundary writes acquisition-run records.

The absence root cause was independent: self-valid attestations were reduced
to session dates before they were reconciled with the freshness consumer. A
fully valid B chain placed under A could therefore explain A when both used the
same date. Session resolution now requires an identity-bound consumer and
retains the validated evidence object until consumer reconciliation completes.
Instrument ID, ticker, provider, provider symbol, exchange, policy, route,
timezone, request window, window semantics, expected session, lifecycle cutoff,
calendar expectation, and terminal-only reason must all match. Observation and
absence evidence for one identity/session remain mutually exclusive.

Negative tests now cover complete downstream relabelling with newly calculated
request, envelope, artifact, filename, locator, reference, receipt-facing, and
publication-facing values; missing trusted runmetadata; unapproved aliases;
request-symbol override; provider-symbol, exchange, provider, policy, route,
timezone, expected-session, and cutoff substitution; and a fully valid B
absence chain placed beneath both A and TMHC. The latter consumers remain
unresolved with `ABSENCE_EVIDENCE_CONSUMER_IDENTITY_MISMATCH`. The positive path
captures and replays bytes through the registered adapter itself.

The production source policy remains empty. The existing `yfinance` DataFrame
route is not silently promoted to trusted raw-response acquisition, and no new
provider is approved. EA and TMHC therefore remain unresolved; additions still
lack approved adapter evidence; historical precision rewrites still lack a
correction contract. The single post-remediation `publish=false` canary on
head `448e460a57d31f470fd0a542fc97eb1a15edd72a` subsequently confirmed this
safe fail-closed state; it did not establish publication readiness.

### Current local validation

| Command | Result | Duration |
|---|---:|---:|
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_mutation_evidence.py tests/market_engine/data/test_observation_receipts.py tests/market_engine/data/test_me_sr18_lifecycle_aware_freshness.py tests/market_engine/data/test_scheduled_canonical_price_refresh.py tests/market_engine/data/test_scheduled_canonical_price_refresh_workflow.py -q --tb=short` | 257 passed, 0 failed, 0 skipped | 3.93 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/run -q --tb=short` | 197 passed, 0 failed, 0 skipped | 3.02 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q --tb=short` | 1,551 passed, 1 failed, 0 skipped | 9.01 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q --tb=short` | 2,218 passed, 1 failed, 0 skipped | 9.98 s |

The sole broad-suite failure is the unchanged missing historical compact
evidence artifact
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`.
The exact failure was previously reproduced on PR base
`a0409a49e8f8f3ef9dce352c22b039ce4387faab`; this remediation does not touch
that contract or artifact path.

### Post-remediation non-publishing canary

Exactly one authorized post-remediation canary ran with `publish=false`:

| Evidence | Result |
|---|---|
| Workflow | [31580777980](https://github.com/sclaessens/market-scanner/actions/runs/31580777980) |
| Run attempt | 1 |
| Branch / executable head | `me-sr23-corporate-action-lifecycle-cutoff-remediation` / `448e460a57d31f470fd0a542fc97eb1a15edd72a` |
| Run identity | `me-sr23-canonical-price-refresh-20260812T085847Z` |
| Input | `publish=false` |
| Workflow result | expected fail-closed `failure`; refresh step succeeded and the explicit degraded/failed gate failed |
| Universe result | 952 total; 873 updated; 3 already current; 5 not expected; 71 failed |
| Coverage | 942 sufficient; 3 limited history; 5 retained inactive; 1 not applicable; 1 insufficient unexplained (`FDXF`) |
| Mutation evidence | 876 instruments; 17,684 mutations without receipts; 876 `MUTATION_EVIDENCE_MISSING` failures |
| Additions | 11,388 unproven additions; no approved artifact-bound receipts |
| Precision rewrites | 6,296 rows across 484 instruments; 7,712 field changes; maximum absolute delta about `1.14e-13`; no correction contract |
| Session reconciliation | 948 instruments; 12,307 unresolved instrument sessions |
| Freshness | 70 `EXPECTED_SESSION_COVERAGE_INCOMPLETE`; EA also `RETAINED_HISTORY_ENDS_BEFORE_EXPECTED_SESSION` |
| Adapter envelopes / replayed artifacts / accepted receipts | 0 / 0 / 0 |
| Identity mismatches / replay failures | 0 / 0 |
| Publish job | skipped |
| Publication bundle | not produced |
| Freshness artifact | `canonical-price-freshness-me-sr23-canonical-price-refresh-20260812T085847Z` |
| Artifact ID / size | `9135168244` / 9,749,555 bytes |
| Artifact digest | `sha256:ecabdc5306ce598ff3d74c48a4e07ee28171a3db5ae5135c83f9556c6f62abda` |
| Freshness manifest | 76,655,347 bytes; SHA-256 `2c61d58aa7bcac0cc024f89eed658fe38d5cc0750267725edd7a7f3a377a76ca` |
| Mutation diagnostics | 26,014,677 bytes; SHA-256 `8b9d84d35e08b3e5adf8f01b60e65469940ebf4adec107a844a45d19de12af53` |
| `market-data` before / after | `95c88276763b1762cbbfbccc402ec8535268127b` / unchanged |

EA remains at 389 canonical rows through July 23, 2026. Its eight expected
sessions through the August 4 lifecycle cutoff remain unresolved because no
artifact, receipt, or absence attestation was produced. TMHC remains
`not_expected` after its completed corporate action, but its July 24 session
remains unresolved because the lifecycle state is not accepted as a replayable
absence attestation.

The canary proved that missing trusted evidence is rejected, lifecycle state is
not substituted for absence evidence, unresolved sessions remain blocked, and
publication is skipped. It did not exercise the positive production adapter
path: zero adapter envelopes, replayed artifacts, and accepted receipts were
available. Therefore the registered-adapter capture, full relabelling, and
cross-consumer protections remain operationally demonstrated by local tests,
not by this production canary. Counts of zero identity mismatches do not prove
that a positive production identity chain was reconciled.

The freshness manifest records `source_main_sha =
a0409a49e8f8f3ef9dce352c22b039ce4387faab`, which identifies the trusted
`main` baseline rather than the executable branch head. GitHub workflow and
artifact metadata independently bind this run to `448e460a...`. A future
contract should record an explicit `executing_code_sha` or
`workflow_head_sha` so a detached manifest can identify both values without
external workflow metadata.

No retry, second canary, runtime change, merge, publication, or `market-data`
mutation occurred. The result is safe fail-closed but not publication-ready.
Another full-universe canary is not useful until a governed production provider
route emits trusted raw-response envelopes and artifact-bound receipts.
Historical precision rewrites, EA, and TMHC must remain separately blocked.

## Trusted Provider Identity and Diagnostic Retention Remediation

This section is the current conclusion for review remediation implemented in
code commit `ed2e2964566a7c5b9b3599ecad732b600c91cd27`. It supersedes the v8
receipt interpretation below while retaining the earlier history as an audit
record.

The review identified three generic defects. The former raw artifact was only
`{"bars":[...]}` and replay injected provider, instrument, symbol, exchange,
and request-window identity from the receipt. A downstream producer could
therefore relabel a payload and recalculate every non-trusted checksum. The
publisher also built receipts for every row in an overlapping response while
mutation equality accepted only added or modified rows. Finally, a failed
reconciliation cleared `canonical_mutations`, losing row counts, sessions,
digests, and field differences.

Source policy v3 and manifest v9 now use a content-addressed adapter envelope.
The envelope binds provider and adapter identity, adapter and parser versions,
instrument, canonical ticker, provider symbol, exchange, currency, route,
credential-free request method and parameters, inclusive/exclusive window
semantics, timezone, pagination, retrieval time, HTTP status and content type,
optional safe provider request ID, raw response bytes and digest, policy ID,
and envelope digest. Acquisition, retention, replay, and canonical-publication
rights are independent booleans and all are required. The production policy
remains empty; no provider or route was approved by this remediation.

The code-level trust boundary is explicit rather than cryptographically
overstated. Only the adapter API creates envelopes, artifacts are immutable and
content-addressed, and the publisher reloads the artifact and independently
validates artifact, envelope, and raw-response digests, source policy, adapter,
parser, identity, request context, credential scan, and parser output. The
repository has no signing key or remote attestation, so this proves integrity
and code-path binding inside the repository, not cryptographic provider
authorship.

Replay is now deliberately two-stage. The publisher first replays every bar in
every trusted artifact, including unchanged overlap bars. It then derives the
baseline/staged mutation ledger and selects the exact replayed observations
that prove added or permitted modified mutations. Receipts are emitted only
for that publisher-selected subset. An unchanged overlap row remains in the
immutable artifact and replay set but requires no mutation receipt. Duplicate
or conflicting sessions within or across paginated artifacts, observations
outside the bound window, missing staged rows, unexplained staged additions,
and any identity or canonical-digest mismatch fail closed. Historical
modifications and deletions remain blocked because no correction or deletion
contract exists.

Absence attestations use the same envelope identity. The publisher reconstructs
the attestation from the referenced envelope and proves instrument, symbol,
exchange, request window, cutoff, calendar expectation, terminal-only reason,
and actual absence after full replay. A status string, internal gap, wrong
instrument, wrong exchange, wrong cutoff, mutated artifact, or response that
contains the session cannot explain absence.

Mutation ledger v2 is retained before evidence equality is evaluated. Every
diagnostic row includes previous and new canonical digests, previous and new
values, field-level differences, receipt status, artifact status, evidence
failure, correction-policy status, and publication blocker. The report binds a
separate deterministic diagnostic artifact by file name and SHA-256. Failures
block publication but no longer erase the ledger or imply that no mutations
were detected.

### Local validation

| Command | Result | Duration |
|---|---:|---:|
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_mutation_evidence.py tests/market_engine/data/test_observation_receipts.py tests/market_engine/data/test_me_sr18_lifecycle_aware_freshness.py tests/market_engine/data/test_scheduled_canonical_price_refresh.py tests/market_engine/data/test_scheduled_canonical_price_refresh_workflow.py -q --tb=short` | 241 passed, 0 failed, 0 skipped | 3.62 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/run -q --tb=short` | 197 passed, 0 failed, 0 skipped | 2.56 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q --tb=short` | 1535 passed, 1 failed, 0 skipped | 7.76 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q --tb=short` | 2202 passed, 1 failed, 0 skipped | 8.94 s |
| Exact failing test on PR base `a0409a49e8f8f3ef9dce352c22b039ce4387faab` | 0 passed, 1 identical failure | 0.02 s |
| Mandatory BUY, SELL, and tradeable greps | only pre-existing portfolio command parsing; no tradeable result | <0.1 s |
| `git diff --check` | passed | <0.1 s |

The sole broad-suite failure is still the pre-existing missing compact-evidence
artifact documented below and was reproduced identically on the PR base.

### Single non-publishing canary

Exactly one workflow was dispatched after pushing the tested code commit:

| Evidence | Result |
|---|---|
| Workflow | [31483637994](https://github.com/sclaessens/market-scanner/actions/runs/31483637994) |
| Input / conclusion | `publish=false` / expected fail-closed failure |
| Code head | `ed2e2964566a7c5b9b3599ecad732b600c91cd27` |
| Run identity | `me-sr23-canonical-price-refresh-20260811T104611Z` |
| Duration | 6 minutes 9 seconds |
| Universe / status | 952 instruments; 942 updated, 4 already current, 5 not expected, 1 failed |
| Provider artifacts | 0 envelopes, 0 valid envelope digests, 0 replayed artifacts, 0 accepted receipts, 0 approved policy IDs |
| Mutations | 946 affected instruments; 11,352 added rows; 6,569 modified rows across 520 instruments; 0 deleted rows |
| Reconciliation | 17,921 mutations without receipts; 0 receipts without mutations; 0 identity mismatches; 6,569 correction-contract blockers |
| Receipt roots | mutation `3f8dda447bbd1d833d3a91f0edbea18aa12898d3bf043ba0b112a832c623a95e`; receipt `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| Diagnostic artifact | `me-sr23-canonical-price-refresh-20260811T104611Z-mutation-diagnostics.json`; SHA-256 `29bb11990b8b8af57606dc0233845a6d15b9462fec1b0638a5b0f8dd0002740f` |
| Report | SHA-256 `3ae7c6fc63febfe47f0ddbe6a2b7b03438f087626b38b18ac0cc0d06a4658ade`; manifest checksum `cef8f8877d700bca3bd55368d345238e38d20c54a1ac665ad161c7bfe1b738c7` |
| GitHub artifact | `canonical-price-freshness-me-sr23-canonical-price-refresh-20260811T104611Z`; digest `0259a3275a0c72ec9ca5ffc9cd02016173a40ba8ab6fa7f3c7d43850a7e8df74` |
| Publication | publication set invalid; publication bundle skipped; publish job skipped |
| `market-data` before / after | `95c88276763b1762cbbfbccc402ec8535268127b` / unchanged |

The canary artifact labelled 946 missing-evidence selection failures as
`artifact_replay_failure_count=946`, even though it also correctly reported
zero envelopes. This was a diagnostic classification defect, not an accepted
publication. The post-canary fix separates artifact replay from evidence
selection: the correct interpretation is zero artifact replay failures and 946
evidence-reconciliation failures. The post-canary fix also materializes every
unresolved expected session in the failed session partition instead of keeping
only the unresolved list. No second canary was dispatched. Consequently the
canary code head and final audit/runtime head are intentionally reported
separately, and the final post-canary runtime head is not claimed as canaried.

### Historical mutation analysis

The retained diagnostics establish 6,569 modified rows across 520 instruments,
covering 389 distinct sessions from 2025-01-02 through 2026-07-23. There are
8,063 field-string differences: Close 1,702; Low 1,602; High 1,587; Open 1,561;
Adjusted Close 1,469; and Volume 142. Of the rows, 5,315 change one field and
1,254 change multiple fields.

Every non-volume numeric delta is at most `1e-13` relative and absolute; the
largest examples change a final binary-float decimal digit such as
`963.5999755859375` to `963.5999755859376`. All 142 Volume string differences
are numerically equal and co-occur with a price-field microdifference. The
diagnostics therefore classify all 6,569 rows as float/CSV round-trip
normalization candidates. They provide no positive evidence of corporate
action adjustment, adjusted/unadjusted source switching, provider-history
revision, timezone/session drift, symbol/exchange substitution, parser change,
or erroneous merge. Those categories each have zero supported rows in this
run; they are not silently accepted or ruled out beyond the available
evidence. All 6,569 rows remain blocked pending an explicit correction and
numeric-normalization contract.

EA remains at July 23 with eight unresolved sessions through the formal August
4 cutoff. TMHC remains at July 23 with July 24 unresolved. Both have zero
envelopes, receipts, or absence attestations. No manual price, synthetic
production artifact, new provider approval, cutoff change, retry, merge, or
publication occurred.

The final status remains `COMPLETED WITH BLOCKERS`: EA lacks an approved raw
adapter route, TMHC lacks replayable absence evidence, 11,352 added rows lack
approved receipts, 6,569 historical micro-rewrites lack a correction contract,
and the final post-canary diagnostic fix has not itself been canaried.

## Final Generic Evidence and Session Reconciliation Remediation

This section is the current audit conclusion for commit
`06ac5536769d3414e38f59d1aa3a83b155153695` and supersedes the historical
implementation and canary details retained later in this document.

The final review found two generic code defects. First, the publisher treated
`primary_observed_sessions` as an alternative to replayable receipts, so a
manifest could label an added row as primary and bypass provider evidence.
Second, the acquisition path accumulated a fallback-required set before all
primary, replay, fallback, and terminal-absence outcomes were known. That set
could remain stale in a mixed historical-gap and terminal-absence case.

Manifest classifications are no longer evidence. The v8 publisher derives a
canonical mutation set directly from the trusted baseline and each staged CSV.
Every added row must equal exactly one uniquely replayed observation receipt,
regardless of whether its acquisition route is primary, primary replay, or
fallback. Historical modifications and deletions fail closed because this
repository has no approved correction or deletion contract. Modified rows
retain previous and new canonical row digests plus a field-level diff for
diagnosis, but cannot be published.

The source-policy and receipt contracts are now v2. Provider approval is
separated into acquisition, raw storage, replay, and canonical publication;
all four approvals and the exact exchange and acquisition route are required.
The uniform receipt binds the provider symbol, source-policy ID, route,
request window, retrieval time, content-addressed raw artifact, parser,
currency, normalized OHLCV and volume, canonical-row digest, and receipt
digest. The publisher replays the raw artifacts and reconstructs the receipts
without trusting producer labels. Per-instrument and publication-wide roots
sort by exchange, instrument, session, canonical-row digest, and receipt
digest, so ordering is irrelevant while missing, extra, duplicate, copied, or
mutated leaves fail reconciliation.

The new Mutation Evidence Ledger classifies baseline/staged differences as
`row_added`, `row_modified`, `row_deleted`, or `row_unchanged`. Its central
invariant is that publisher-derived added or modified rows equal the unique
replayed receipt rows and exact staged canonical rows. The new Session
Resolution Ledger assigns each considered session exactly one of
`observed_primary`, `observed_fallback`, `explained_absent`, `unresolved`, or
`not_expected`. Observed sessions come only from valid replayed receipts;
explained absences come only from valid replayed absence attestations; and
fallback candidates are rederived from the remaining unresolved sessions
after every merge. The legacy `fallback_required_sessions` publication
contract was removed.

An absence attestation is session-specific and binds the instrument, exchange,
formal lifecycle cutoff, terminal-only reason code, calendar expectation,
approved acquisition route, request window, content-addressed provider
artifact, parser result, and attestation digest. The publisher proves that the
artifact does not contain the terminal session, that no receipt or staged row
exists for the same session, and that the session equals the lifecycle cutoff.
An internal gap, a status string, a non-terminal session, an unsuccessful
response, or an unapproved source cannot explain absence.

Gap-directed primary replay is ticker-agnostic. It derives a buffered bounded
window from the earliest and latest unresolved exchange sessions and clamps it
to the original request. A second completeness acquisition is permitted only
when the provider exposes an identity already approved by the machine-readable
policy for `primary_replay`; otherwise the path stops and remains unresolved.
The production policy intentionally contains no providers, so the current
Yahoo Finance adapter is technically reachable but is not approved for raw
retention, replay, or canonical publication. No source approval was added and
no manual market data was introduced.

Deterministic combination tests cover arbitrary ticker shapes, multiple
exchanges, all receipt routes, reordered sessions and rows, unchanged, added,
modified, deleted and duplicate rows, mixed primary/fallback observations,
terminal absence, explicit not-expected sessions, stale fallback queues, and
root stability. Publisher regressions recompute CSV and manifest checksums and
still reject primary-label bypasses, removed or duplicated receipts, wrong
identity, exchange, session, parser, approval, artifact, checksum, OHLCV,
volume, adjusted close, and publication root. Product code contains no EA or
TMHC literal and no Investing.com reference.

### Final local validation

| Command | Result | Duration |
|---|---:|---:|
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_mutation_evidence.py tests/market_engine/data/test_observation_receipts.py tests/market_engine/data/test_me_sr18_lifecycle_aware_freshness.py tests/market_engine/data/test_scheduled_canonical_price_refresh.py -q --tb=short` | 227 passed, 0 failed, 0 skipped | 3.32 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/run -q --tb=short` | 197 passed, 0 failed, 0 skipped | 2.56 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q --tb=short` | 1530 passed, 1 failed, 0 skipped | 8.35 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q --tb=short` | 2197 passed, 1 failed, 0 skipped | 9.64 s |
| Exact broad-suite failure on PR base `a0409a49e8f8f3ef9dce352c22b039ce4387faab` | 0 passed, 1 identical failure | 0.02 s |
| Schema, receipt replay, lifecycle alias, mutation, session-partition, and publisher contract tests | passed | included above |
| `git diff --check` | passed | <0.1 s |

The sole broad-suite failure remains the pre-existing missing artifact
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`. The exact test fails identically
on the PR base. Mandatory governance greps still report only the pre-existing
BUY and SELL command parsing and portfolio transaction handling; `tradeable`
is absent. No allocation, Decision Engine, reporting, or market-data file was
changed.

### Final non-publishing canary

Exactly one new workflow was dispatched after pushing the implementation:

| Evidence | Result |
|---|---|
| Workflow | [31335952903](https://github.com/sclaessens/market-scanner/actions/runs/31335952903) |
| Input | `publish=false` |
| Branch / code head | `me-sr23-corporate-action-lifecycle-cutoff-remediation` / `06ac5536769d3414e38f59d1aa3a83b155153695` |
| Run identity | `me-sr23-canonical-price-refresh-20260809T210624Z` |
| Trusted source main | `a0409a49e8f8f3ef9dce352c22b039ce4387faab` |
| Duration / conclusion | 5 minutes 10 seconds / expected fail-closed failure |
| Universe | 952 total; 946 active; 6 retained inactive; 0 pending |
| Freshness status | 942 updated; 4 already current; 5 not expected; 1 failed |
| Mutation evidence | 946 invalid instruments; 6 valid retained instruments |
| Invalid mutation causes | 426 added-row/receipt equality failures; 520 unsupported historical modifications |
| Session resolution | 948 instruments and 10,415 expected sessions unresolved |
| Receipts / absence attestations | 0 / 0 |
| Changed files | 946 declared; 946 unique |
| Publication roots | mutation `3f8dda447bbd1d833d3a91f0edbea18aa12898d3bf043ba0b112a832c623a95e`; receipt `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| Publication | set valid `false`; required `false`; publication bundle skipped |
| Publish job | skipped |
| Freshness artifact | `canonical-price-freshness-me-sr23-canonical-price-refresh-20260809T210624Z`; GitHub digest `0288915d1bf3813f890e0cd62d928d8af2558abc1429992ab241d33ae6dfac6e` |
| Extracted report SHA-256 | `0cf0d86d1af78e1646c045f74a025a348c7ceb0720ca8ae038ab4bef47a01b12` |
| Manifest | v8; checksum `1e9bf8975b20cf96154bfa04b02f105796fdc456a9ee2b7fd47aad96c2d99777` |
| Source policy | v2; checksum `7bdc8454d4b766b9fdf2baae615846fcaff2324fe9cc3700afd3e4fce0e0d682`; zero providers |
| `market-data` before / after | `95c88276763b1762cbbfbccc402ec8535268127b` / unchanged |

The canary proves that primary labels no longer bypass evidence. EA's trusted
baseline remains 389 rows through July 23 with checksum
`758b5bd8ed67403eebc2ba1673e500ea8cc219ad708f4b0653ca0a180fb867a0`.
The batch returned four bars ending August 4, but only August 4 was in the
bounded lifecycle interval. Because the source has no approved replay/storage
policy and produced no receipt, even that primary-labeled row was rejected.
All eight expected sessions from July 24 through August 4 are unresolved;
`resulting_last_observation` remains July 23 and `rows_added=0`.

TMHC retains the formal July 24 cutoff and July 23 canonical observation end.
Its legacy observation record still contains only a status string and locator,
not a retained replayable provider artifact under the v2 source policy.
Therefore the final ledger correctly leaves July 24 unresolved and does not
accept the old status as an absence attestation. Local tests prove that a valid
terminal attestation resolves the terminal session, a mixed historical gap is
resolved independently, a later valid July 24 observation remains admissible,
and a post-cutoff bar is rejected.

The canary also exposed existing primary-path historical rewrites across 520
files. The new mutation ledger detected and blocked them because no correction
contract exists. This is an operational data-source blocker, not a reason to
weaken the publisher. No retry or second workflow was dispatched.

### Current blockers and rollback

EA has no approved replayable canonical source for the eight required rows;
TMHC has no replayable terminal absence attestation; and the primary refresh
attempts unsupported historical modifications in 520 files. The correct final
status is `COMPLETED WITH BLOCKERS`.

Rollback is a normal revert of the reviewed ME-SR23 commits. Do not rewrite
history, edit `market-data`, restore transcribed prices, approve a technically
reachable source without governance, or relax mutation, receipt, session, or
publisher equality checks.

## Historical Review Record

The remaining sections preserve earlier review and canary evidence. Where they
conflict with the final section above, they are historical and not current
operational claims.

## Third Review Findings

The third review of draft PR #474 identified three final merge blockers:

1. Seven EA gap-fill bars had been transcribed from Investing.com without a
   repository approval proving acquisition, raw storage, and canonical
   publication rights.
2. The trusted publisher validated a checksum-bound hand-authored evidence
   record, but could not replay a raw provider response and independently
   reconcile every OHLCV value with the staged canonical CSV row.
3. Legacy and canonical lifecycle observation aliases were coalesced without
   rejecting contradictory values.

These findings invalidate the earlier claim that the seven Investing.com bars
were governed canonical source evidence. Previous favorable EA canary results
remain historical diagnostics only and are not source-provenance or
merge-readiness evidence.

## Source-Governance Decision

The repository review covered the provider and source-approval contracts,
credential and acquisition boundaries, artifact retention patterns,
provenance fields, and existing market-data adapters. No repository contract
approves Investing.com, Alpha Vantage, Yahoo Finance, or another fallback
provider for all three required uses: daily-OHLCV acquisition, immutable raw
response storage, and canonical gap-fill publication.

The production policy is therefore explicit and empty:

`config/market_engine/source_policies/market_price_sources.json`

The policy distinguishes provider reachability from acquisition, raw-storage,
and canonical-publication approval. A fallback must have a non-empty approval
ID, daily-OHLCV scope, explicit exchange scope, retention and redistribution
classifications, and all three approval booleans. Unknown, partially approved,
wrong-exchange, or missing-approval providers fail closed.

The former `verified_daily_ohlcv_evidence.json` registry and
`verified_price_observations.py` injection path were deleted. No transcribed EA
OHLCV value remains in production code or configuration. In production, the
normal primary request still covers the complete bounded interval. Missing
expected sessions are recorded as fallback-required sessions, but remain
blocked because the source policy currently contains no approved fallback.

Consequently, EA cannot yet satisfy the eight-session interval from the
July 23 baseline through the August 4 lifecycle cutoff. This is the intended
fail-closed outcome, not a reason to recreate hand-authored evidence.

## Replayable Observation Receipt Contract

The v7 freshness manifest introduces a source-agnostic observation-receipt
ledger. A policy-approved fallback response is usable only through this chain:

`raw artifact -> deterministic parser -> receipt -> canonical row`

Raw JSON payloads are stored beneath a provider-bound
`evidence/market_price/<provider>/<sha256>.json` locator. Locators cannot be
absolute, traverse directories, select another provider directory, or disagree
with the payload checksum. Credential-like fields are rejected before storage
and are not included in diagnostics.

The `canonical-json-daily-ohlcv` parser version `v1` produces deterministic
decimal strings and integer volume. Every receipt binds:

- receipt and canonical-row schema versions;
- instrument, ticker, exchange, currency, and session;
- provider and approval IDs;
- UTC retrieval time and bounded request window;
- response status and content type;
- raw artifact locator and SHA-256;
- retention and redistribution classifications;
- parser name and version;
- normalized open, high, low, close, adjusted close, and volume;
- canonical row SHA-256 and receipt SHA-256.

The canonical row digest covers instrument, session, complete OHLCV, volume,
and currency. The observation receipt root is a SHA-256 digest over the sorted
set of canonical row digests, making ordering irrelevant while rejecting
missing, extra, or duplicate leaves.

## Trusted Publisher Reconciliation

The trusted publisher independently:

1. compares the staged dataset with the trusted `market-data` baseline;
2. rejects changes to existing historical rows;
3. derives every newly added session from the CSV diff;
4. reconciles added sessions with the primary acquisition journal and the
   fallback-required session set;
5. verifies the exact raw-artifact fileset;
6. reloads the source policy and approval;
7. reloads and hashes each raw artifact;
8. reruns the named parser version;
9. rebuilds every receipt and receipt checksum;
10. recalculates the receipt root;
11. enforces one receipt per fallback-required session; and
12. compares all replayed OHLCV fields and volume with the staged CSV row.

The publisher rejects absent or extra artifacts, absent or duplicate receipts,
empty evidence for required fallback sessions, wrong ticker, instrument,
exchange, session, approval, provider, request window, parser, raw checksum,
row value, volume, adjusted close, root, or lifecycle cutoff. Primary and
fallback observations cannot silently overwrite one another. Publication
remains atomic and gated by trusted main. The workflow now carries the governed
raw-evidence directory into `market-data` and stages additions and deletions
explicitly.

## Alias Boundary

Lifecycle parsing now uses one generic `resolve_semantic_alias` boundary for:

- `last_trading_session` and `delisting_end_date`;
- `canonical_ohlcv_last_observed_session` and
  `price_observation_end_session`; and
- `terminal_session_daily_ohlcv_status` and
  `final_session_observation_status`.

Dates are compared as semantic session dates, including equivalent midnight
ISO timestamps. Observation statuses use an explicit legacy-to-canonical map.
Legacy-only, canonical-only, and semantically equal dual inputs are accepted
and immediately projected to one canonical internal representation. Invalid,
unknown, or conflicting values fail closed with ticker, field names, raw and
normalized values, and contract version. Canonical v5 output does not emit the
legacy observation aliases.

TMHC remains separately governed: July 24 is its formal last trading session,
the provider-specific daily-OHLCV absence status remains temporary, a later
valid July 24 bar remains admissible, and any post-cutoff bar remains rejected.

## Regression Evidence

The regression set covers approved complete primary acquisition, exact
approved fallback supplementation with raw replay, unapproved and partially
approved providers, approval and exchange mismatch, missing or mutated raw
artifacts, parser mismatch, request-window and post-cutoff evidence, primary
and fallback conflict, secret rejection, deterministic numeric serialization,
receipt/root ordering, and trusted-publisher CSV and fileset mutations.

Alias coverage includes legacy-only, canonical-only, semantically equal,
conflicting, equivalent date notation, invalid date, mapped legacy status,
unknown status, and singleton/batch behavior.

### Local validation

| Command | Result | Duration |
|---|---:|---:|
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_observation_receipts.py tests/market_engine/data/test_me_sr18_lifecycle_aware_freshness.py tests/market_engine/data/test_scheduled_canonical_price_refresh.py tests/market_engine/data/test_scheduled_canonical_price_refresh_workflow.py -q --tb=short` | 194 passed, 0 failed, 0 skipped | 3.11 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/run -q --tb=short` | 197 passed, 0 failed, 0 skipped | 2.51 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q --tb=short` | 1488 passed, 1 failed, 0 skipped | 7.37 s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q --tb=short` | 2155 passed, 1 failed, 0 skipped | 8.53 s |
| Lifecycle v5 and market-price source-policy v1 loaders | passed | <0.1 s |
| `git diff --check` | passed | <0.1 s |

Both broad suites have only the known historical artifact failure:
`test_compact_checksums_match_committed_files_and_local_full_runs` cannot find
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`.
The exact test reproduces identically on clean base SHA
`a0409a49e8f8f3ef9dce352c22b039ce4387faab` (one failure in 0.03 seconds).
There are no new failures.

The lifecycle registry checksum is
`664416d7d81830b159a395ca4f00de5689b0180037ae5386cee75bfc1132a4db`.
The empty production source-policy checksum is
`55cd77a75c9e45c699d02e075b7c3ddf33bfb96d6ea5147ba9b7ad612c908d61`.

The mandatory repository greps still find pre-existing BUY and SELL command
parsing and transaction logging under `scripts/portfolio`; the current diff
does not touch those files. The `tradeable` grep is empty. No Decision Engine,
allocation, or reporting semantics changed in this remediation.

## Final Non-Publishing Canary

Exactly one post-remediation workflow was dispatched with `publish=false`:

| Evidence | Result |
|---|---|
| Workflow | [31323447253](https://github.com/sclaessens/market-scanner/actions/runs/31323447253) |
| Branch / canary head | `me-sr23-corporate-action-lifecycle-cutoff-remediation` / `43a5b646253fca1833d2882e41252cfaaa145203` |
| Run identity | `me-sr23-canonical-price-refresh-20260809T161908Z` |
| Input / job duration | `publish=false` / 5 minutes 48 seconds |
| Trusted source main | `a0409a49e8f8f3ef9dce352c22b039ce4387faab` |
| Universe | 952 total; 946 active; 6 retained inactive; 0 pending |
| Status | 942 updated; 4 already current; 5 not expected; 1 failed; 0 stale or unsupported |
| Coverage | 942 sufficient; 4 limited; 5 retained inactive; 1 not applicable; 0 insufficient unexplained |
| Source policy | v1; checksum `55cd77a75c9e45c699d02e075b7c3ddf33bfb96d6ea5147ba9b7ad612c908d61`; 0 approved fallbacks |
| Freshness artifact | `canonical-price-freshness-me-sr23-canonical-price-refresh-20260809T161908Z`; GitHub digest `8486953b76eece07306348187fa6e10f5d812f75f8ac255ab7017399a6fd130e` |
| Extracted report SHA-256 | `772a5ababbe6d06ed35d67008abdcf8941d54a525caa257c5338a6b8892a7cae` |
| Manifest | v7; checksum `99331b0cb70bba7d7a747e167c9e329b2cdcf25b6ef34707adab66f11f3634ce` |
| Changed files | 946 declared and 946 unique; EA not included |
| Receipts / roots | 0 / 0, because no fallback is approved or used |
| Publication | required `false`; set valid `false`; publication artifact skipped |
| Publish job | skipped |
| `market-data` before / after | `95c88276763b1762cbbfbccc402ec8535268127b` / unchanged |

EA is the sole failed instrument. Its trusted baseline remains 389 rows through
July 23 with checksum
`758b5bd8ed67403eebc2ba1673e500ea8cc219ad708f4b0653ca0a180fb867a0`.
The primary provider returned only August 4. The required interval contains
July 24, 27, 28, 29, 30, and 31 and August 3 and 4. The seven internal sessions
were classified as fallback-required, but the empty production policy yielded
no receipts, raw artifacts, or receipt root. The pipeline preserved the
baseline unchanged: `previous_last_observation=2026-07-23`,
`resulting_last_observation=2026-07-23`, and `rows_added=0`. It did not reuse
the former Investing.com values.

TMHC retained `last_trading_session=2026-07-24`, canonical observation end
July 23, and the precise temporary
`no_valid_daily_ohlcv_bar_from_provider_as_of` status. No legacy observation
aliases were emitted. Local regressions prove that a later complete July 24
bar remains admissible and a July 25 bar remains quarantined.

Artifact review found that the canary journal classified TMHC's already
explained July 24 no-bar session as fallback-required even though it added no
canonical row. The final implementation now excludes explicitly explained
no-observation sessions from gap-fill receipt requirements and requires every
remaining missing session to be explained before accepting the provider
result. A new local regression is green. Per the one-canary constraint, this
post-canary correction was not dispatched again and is not represented by a
second workflow run.

## Remaining Blocker and Rollback

The current production policy contains no approved canonical daily-OHLCV
fallback. EA therefore remains incomplete and the final status cannot be
`READY FOR RE-REVIEW`. The one non-publishing canary demonstrated this
fail-closed condition without modifying `market-data`.

Rollback is a normal revert of the reviewed ME-SR23 commits. Do not rewrite
branch history, manually edit `market-data`, restore the deleted hand-authored
EA registry, or weaken the receipt and alias gates.
