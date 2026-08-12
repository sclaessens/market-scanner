# ME-SR24 Approved Production Price-Evidence Route Audit

Status: `completed_with_blockers`

## Outcome

ME-SR24 did not activate a production price-evidence provider route. The
repository contains no approved market-price provider policy from which the
required acquisition, retention, replay, mutation-evidence, and canonical
publication permissions can be proven. Adding a provider entry would therefore
invent approval rather than implement an existing approval.

No runtime code, test code, provider configuration, canonical price data, or
publication workflow was changed. No provider request, canary, merge,
publication, workflow dispatch, retry, or `market-data` write was performed.
Automatic ticker publication remains disabled by the empty source policy and
the existing publisher gates.

## Safe Start

| Evidence | Result |
|---|---|
| Repository | `sclaessens/market-scanner` |
| Story | `ME-SR24` was unused in branches, backlog, roadmap, and current audit documents |
| Branch | `me-sr24-approved-production-price-evidence-route` |
| Start head | `00c1bbc792dc64bedd2cf9ac95fedbafe573f08b` |
| PR #474 | merged into `main` on 2026-08-12 |
| PR #474 merge commit | `00c1bbc792dc64bedd2cf9ac95fedbafe573f08b` |
| PR #474 final source head | `069d1b9692fff282367b8655c38265a87282542b` |
| `market-data` before documentation work | `95c88276763b1762cbbfbccc402ec8535268127b` |

The original local worktree contained unrelated untracked generated artifact
directories. They were preserved untouched. ME-SR24 was isolated in a clean
worktree created directly from `origin/main`.

`AGENTS.md` was reviewed. The repository references `docs/active/` as active
documentation, but that directory is absent from the merged tree. Current
Market Engine documentation under `docs/market_engine/`, including the ME-SR23
audit, backlog, and roadmap, was therefore inspected without treating archived
documents as new authority.

## Provider Approval Gate

The machine-readable production policy is
`config/market_engine/source_policies/market_price_sources.json`. It has schema
`market-engine-market-price-source-policy-v3` and an empty `providers` array.
Consequently there is no registered provider ID, approval ID, adapter version,
parser version, exchange scope, acquisition route, retention classification,
redistribution classification, or boolean approval for acquisition, retention,
replay, and canonical publication.

The archived provider audit-trail requirements explicitly state that raw
payload retention is `NOT_APPROVED`. Historical downloads, fixtures, provider
names, existing libraries, locally cached price files, credentials, and
publication manifests are not source approval.

The following required approval properties are absent:

* legal or contractual permission for automated production acquisition;
* permission to retain the necessary raw response and a defined retention
  duration;
* permission to replay the retained response through the registered parser;
* permission to use the response as mutation and publication evidence;
* approved daily-OHLCV instrument, exchange, and MIC coverage;
* approved provider-symbol mappings for the authoritative instrument registry;
* approved request frequency, request-window semantics, timezone, and
  pagination behavior;
* an approved adapter and parser release for the production route;
* an approved redaction contract for permitted response metadata.

No provider was selected. Yahoo Finance through `yfinance` was not approved:
the current production downloader returns normalized DataFrames, not retained
raw provider responses captured at the request/response boundary. Investing.com
was not used or introduced.

## Configuration and Secret Inspection

Configuration and secret presence were inspected by name only. The repository
has no price-provider variable and no price-provider secret. The only repository
secret names reported by GitHub were `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID`; neither is relevant to market-price acquisition. No secret,
token, cookie, credential value, or sensitive header was read, logged, stored,
or committed.

## Existing Technical Boundary

PR #474 already provides generic fail-closed infrastructure for a future
approved route:

```text
authoritative instrument registry
  -> registered source policy and adapter identity
  -> request identity
  -> immutable content-addressed raw-response envelope
  -> content-addressed acquisition-run manifest
  -> parser replay from storage
  -> observation receipt
  -> publisher-derived mutation reconciliation
```

The active policy gate prevents that infrastructure from claiming a production
route while the policy is empty. A `yfinance` DataFrame cannot be passed off as
a raw artifact, and downstream receipt data cannot establish acquisition
identity. Content addressing proves internal byte consistency; it is not an
external provider signature. An actor controlling trusted code and trusted
storage is outside the current cryptographic proof model.

ME-SR24 intentionally adds no generic production envelope builder, free
identity input, free request parameters, post-parse raw-artifact reconstruction,
DataFrame evidence path, alias, provider exception, or ticker exception.

## Required Route and Provenance — Blocked

Because no provider is approved, the following production capabilities were
not activated or falsely demonstrated:

* registered production transport and real request/response capture;
* immutable provider artifact retention under an approved license;
* trusted acquisition-run records for a real provider request;
* parser replay of a production response;
* artifact-bound receipts for new daily observations;
* publication-eligible daily-addition reconciliation;
* runtime provenance fields such as `executing_code_sha`,
  `workflow_head_sha`, `source_main_sha`, workflow/run ID, and local dirty
  status for a new acquisition run.

Those provenance fields remain required for the eventual approved route and
must originate from the trusted runtime or CI context. They must not be
accepted from request, artifact, or receipt input.

## Mutation Scope

No mutation class was newly admitted.

* A future approved route may admit only a replayed `add` for the next expected
  terminal trading session after the canonical dataset end, with exact policy,
  identity, symbol, exchange/MIC, request, window, timezone, pagination,
  lifecycle, absence, and diff reconciliation.
* Historical additions, internal gaps, lifecycle-related additions, unexpected
  additions, updates, deletes, and precision or serialization rewrites remain
  blocked.
* The 11,388 additions observed by canary run `31580777980` remain unproven and
  cannot be bulk-approved. They require per-session classification after a
  route has approved replayable evidence.
* The 6,296 precision/serialization rewrites remain outside scope and lack a
  correction contract.
* EA remains blocked with its canonical history ending on 2026-07-23 and no
  approved evidence for the expected later sessions.
* TMHC remains retained inactive with its governed 2026-07-24 lifecycle
  boundary unresolved by trusted observation or absence evidence.

There is no EA or TMHC hardcoding in this change because there is no runtime
change.

## Validation

The change is documentation-only. No positive production adapter test can be
truthfully added: such a test requires a registered approved route, while the
policy registry is empty. A controlled transport fixture paired with an
invented production provider entry would violate the approval gate rather than
prove it.

Existing runtime suites continue to provide the negative identity, artifact
replay, receipt, mutation, lifecycle, absence, overlap, session-ledger,
partition, schema, and publisher regressions introduced through ME-SR23.

| Command | Passed | Failed | Skipped | Duration | Result |
|---|---:|---:|---:|---:|---|
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data/test_observation_receipts.py tests/market_engine/data/test_mutation_evidence.py tests/market_engine/data/test_me_sr18_lifecycle_aware_freshness.py tests/market_engine/data/test_scheduled_canonical_price_refresh.py tests/market_engine/data/test_scheduled_canonical_price_refresh_workflow.py -q --tb=short` | 257 | 0 | 0 | 4.55 s | Passed |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/run -q --tb=short` | 197 | 0 | 0 | 2.69 s | Passed |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q --tb=short` | 1,551 | 1 | 0 | 8.43 s | One pre-existing missing-artifact failure |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q --tb=short` | 2,218 | 1 | 0 | 9.68 s | Same pre-existing missing-artifact failure |
| Single failing test on detached `origin/main` at `00c1bbc7` | 0 | 1 | 0 | 0.03 s | Identical failure reproduced on the correct base |
| `git diff --check` | n/a | 0 | n/a | <0.1 s | Passed |

The targeted 257-test command covers provider-policy validation, registered
adapter capture, trusted acquisition-run binding, immutable storage replay,
parser validation, mutation receipts, complete relabelling and substitution,
request-window/timezone/pagination binding, daily and historical additions,
precision modifications, lifecycle and absence reconciliation, overlap,
mutation diagnostics, session ledgers, partitions, schemas, contracts, and
workflow publication gates. It also proves the existing statements that a
receipt does not establish its own identity, an artifact for B cannot evidence
A, dictionary placement does not establish instrument binding, and only a
replayed observation from a trusted acquisition run can produce a receipt.

The sole broad-suite failure is:

```text
tests/market_engine/data/test_operator_pilot_compact_evidence.py::
test_compact_checksums_match_committed_files_and_local_full_runs
```

It expects the absent historical file
`artifacts/market_engine/fundamental_evidence_coverage_runs/me-data06-after-me-data09-aapl-20260719T155116Z/manifest.json`.
The identical assertion and path were reproduced in a clean detached worktree
at the current `origin/main` head, so ME-SR24 did not cause the failure.

The mandatory repository greps reported only the unchanged legacy trade-command
and portfolio transaction literals in `scripts/portfolio/`; no `tradeable`
match was reported. ME-SR24 changes no runtime file and introduces no allocation
or Decision Engine authority.

## Remaining Operational Blockers

Before a later story may activate the first production route, a human-approved
source policy or referenced contract must explicitly establish every missing
legal and technical property listed above. Any required credentials and
external storage must then be provisioned through approved secret and retention
boundaries. Only after that approval exists may implementation add the real
transport, provenance capture, positive end-to-end route test, and bounded
`publish=false` canary request.

No canary is authorized or requested by this blocker closeout. Automatic
publication remains disabled.
