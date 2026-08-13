# ME-SR25 — Advisory Price Evidence Artifact

Sprint ID: ME-SR25

Status: IMPLEMENTED BY ME-SR25 / POST-MERGE OPERATIONAL VALIDATION NOT EXECUTED

Job family: ME-SR / Source Refresh

## Purpose

Provide a daily best-effort price snapshot for ticker analysis and descriptive
portfolio enrichment without claiming canonical publication evidence.

The existing scheduled refresh can start and stage data, but the
`market-data` publication path is blocked by ME-SR24's production-evidence
requirements. ME-SR25 creates a separate advisory-only output boundary instead
of weakening or bypassing those gates.

## Product outcome

Each trading day, the project produces a retrievable advisory artifact that
states, per ticker:

- latest validated available session;
- normalized price observation;
- source identifier;
- retrieval timestamp;
- freshness state;
- validation state;
- failure or blocker code;
- explicit per-instrument failure evidence when current acquisition fails.

Candidate analysis may use only observations whose freshness and validation
states permit advisory consumption. Stale or missing data remains explicit.

## Required boundary

```text
scheduled source request
  -> existing supported acquisition path
  -> close-observation and identity validation
  -> advisory snapshot
  -> freshness manifest
  -> analysis handoff
```

The output is non-canonical, non-transactional, and non-executable. It is not a
mutation receipt, production price-evidence route, or substitute for the
`market-data` branch.

## Persistence and publication

The implementation should use a GitHub Actions artifact or another
reviewed, non-canonical storage target suitable for ChatGPT retrieval.

It must not write to or advance the canonical `market-data` branch.

The artifact must be clearly named and versioned as advisory data, include a
run manifest and per-ticker status, and define retention. Raw provider payloads
must not be persisted unless separately allowed; normalized observations and
minimal provenance are sufficient for this advisory boundary.

## Scheduling

Reuse the existing daily schedule where practical. The current workflow
schedule is 05:30 UTC. Exchange calendars and completed-session semantics must
prevent a pre-market run from claiming an unavailable session as current.

Retries must be bounded. Per-ticker failures must not be hidden. A degraded
artifact may be produced when successful and failed tickers are explicitly
partitioned, but failed or stale tickers must not be presented as fresh.

## Provider boundary

ME-SR25 does not require a new commercial provider contract as a roadmap
prerequisite. It may reuse the repository's existing supported acquisition
path for personal advisory analysis, subject to the provider's actual
availability and configured usage constraints.

It must not:

- promote `yfinance` or a DataFrame into canonical evidence;
- claim legal or cryptographic publication proof;
- store credentials, cookies, or sensitive headers;
- introduce Investing.com;
- silently switch providers;
- fabricate missing bars;
- hardcode ticker-specific exceptions.

## Consumer boundary

Advisory price output may enrich:

- ticker screening freshness;
- candidate review;
- descriptive last price;
- descriptive market value;
- unrealized profit/loss when combined downstream with ME-PR03 positions;
- exposure calculations in a later ME-PI01 story.

It must not alter the transaction ledger, average cost, realized profit/loss,
position quantity, recommendation authority, allocation, or execution.

## Allowed scope

- Source Refresh workflow and command for advisory snapshots;
- normalized advisory price and freshness contracts;
- bounded retry and per-ticker diagnostics;
- analysis-consumer handoff;
- artifact retention and retrieval documentation;
- focused tests, audit, backlog, and roadmap updates.

## Forbidden scope

- canonical `market-data` publication;
- bypassing ME-SR23/24 evidence gates;
- broker integration or order execution;
- portfolio mutation;
- provider-contract procurement;
- raw-response retention without explicit permission;
- historical precision rewrites;
- synthetic production prices;
- recommendation, ranking, sizing, or Decision Engine changes;
- automatic canary beyond the story's reviewed test plan.

## Test impact

Required coverage includes:

- successful fresh terminal session;
- pre-market/no-completed-session behavior;
- mixed success/failure partitioning;
- stale close observation;
- invalid close price;
- identity and symbol mismatch;
- bounded retry;
- deterministic manifest;
- no `market-data` branch write;
- no canonical receipt generation;
- no portfolio-ledger mutation;
- downstream consumer rejects stale/invalid observations;
- schedule and artifact-retention contract;
- secrets and sensitive headers absent.

## Acceptance criteria

ME-SR25 is complete only when:

1. the scheduled workflow produces a retrievable versioned advisory artifact;
2. freshness and validation are explicit per ticker;
3. analysis can consume the artifact without treating it as canonical;
4. mixed failures remain visible and cannot masquerade as current data;
5. no canonical branch, receipt, provider policy, portfolio ledger, broker, or
   order path is changed;
6. relevant tests pass;
7. one bounded non-publishing validation run is separately approved and
   documented before operational status is claimed.

## Dependencies

- ME-PR03 is implemented first because portfolio truth must not depend on price
  acquisition.
- ME-SR17 through ME-SR24 remain historical and canonical-publication context.
- ME-PI01 may later combine ME-PR03 positions with ME-SR25 price enrichment for
  exposure and concentration intelligence.

## Implementation outcome

ME-SR25 implements `me-sr25-advisory-price-evidence-v1` as a separate
non-canonical observations file and checksum-bound run manifest. The builder
uses the existing yfinance daily-history adapter, canonical instrument IDs,
canonical tickers, source mappings, currency, and the existing completed-
session resolver. It emits deterministic per-instrument `fresh`, `stale`,
`missing`, or `invalid` evidence and reconciled run totals over all 952
canonical instruments.

One semantic validator protects creation, load, integrity reconciliation, and
consumer trust boundaries. Prices are exact positive canonical decimal
strings. Identity, source, observation type, currency, timestamps, freshness,
full-universe membership, ordering, counts, policy, and digests fail closed.
The public consumer accepts an artifact path rather than caller-supplied price
objects. It preserves the immutable generation-time freshness evidence while
recomputing an effective freshness view at each trusted load or consumption
time, and exposes a current price only for a validated effectively fresh
observation. Retention never substitutes for this freshness decision.

The separate `Advisory Price Evidence` workflow runs at 05:30 UTC, uploads only
a 14-day GitHub Actions artifact, and has read-only repository permissions. It
contains no publication job or `market-data` write. No workflow run or canary
was executed by this sprint; operational status still requires separately
approved post-merge validation.

The accelerated product route after implementation is:

```text
ME-SR25 -> ME-DATA11 -> ME-RUN33 -> ME-CI12
```

ME-RUN33 is the first unreserved RUN identifier found in the authoritative
documentation and is reserved here for the first useful end-to-end candidate
analysis release. That release must produce 5–15 comparable candidates with
current price/freshness, technical setup, entry/stop/target context,
risk/reward, fundamental quality and gaps, risks and invalidation conditions,
explicit include/exclude reasons, and directly explainable ChatGPT input.
ME-PI01, further portfolio expansion, position sizing, notifications, broker
integration, cloud portfolio storage, and canonical-publication remediation
are deferred until after that release.
