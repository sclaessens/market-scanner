# ME-SA01 — Automated Cached-Source Acquisition Job Contract

Sprint ID: ME-SA01  
Status: Contract / docs-only  
Job family: ME-SA / Source Acquisition  
Date: 2026-06-26  

## 1. Purpose

ME-SA01 defines the contract for an automated cached-source acquisition job.

The job is responsible for retrieving approved real-world source data and writing cached-source snapshot packages that can be consumed by the existing downstream validation and dry-run flow.

The primary architectural route after ME-SA01 is:

```text
automated acquisition job
  -> cached-source snapshot package
  -> existing import/staging validation
  -> cached_source_snapshot dry-run
  -> operator preview
```

ME-SA01 does not implement the job. It defines the boundaries, request contract, output contract, safety rules, provenance requirements, freshness/staleness rules, failure model, and ME-SA02 acceptance criteria.

## 2. Background

Recent Market Engine work established a cached-source based dry-run path and validated that operator-supplied cached-source inputs can be imported and consumed safely.

Relevant prior work:

- ME-SR11 introduced the cached-source snapshot acquisition dry-run command.
- ME-SR12 introduced the operator-supplied cached-source snapshot import command.
- ME-RUN25 validated the operator-supplied fixture import, staging, and dry-run flow with a PASS result.
- ME-SR13 attempted real-world operator-supplied input validation but was blocked because the required operator input package was missing.
- ME-RM03 corrected the roadmap by making automated cached-source acquisition by an application job the primary direction.

Product-owner decision:

The application must be able to retrieve source data itself through an automated job. The system should not remain dependent on manually downloaded operator input files as the primary route.

ME-SR13A is therefore superseded as the primary route. It may remain only as a fallback or manual diagnostic candidate.

## 3. Architectural Chain

The approved chain is:

```text
automated acquisition job
  -> cached-source snapshot package
  -> existing import/staging validation
  -> cached_source_snapshot dry-run
  -> operator preview
```

Each stage has a strict responsibility boundary.

| Stage | Responsibility |
|---|---|
| Automated acquisition job | Retrieve approved source data through approved adapters and write cached-source snapshot packages with provenance. |
| Cached-source snapshot package | Persist payload files and manifest metadata in a format compatible with existing cached-source expectations. |
| Existing import/staging validation | Validate package structure, manifest fields, payload references, hashes, sizes, ticker identity, source identity, and safety constraints. |
| `cached_source_snapshot` dry-run | Consume staged cached-source snapshots without live provider calls. |
| Operator preview | Present terminal-visible or Telegram-style review output without sending Telegram messages in ME-SA01 or ME-SA02 unless separately approved later. |

The acquisition job must never bypass existing validation or dry-run controls.

## 4. Job Boundary

### 4.1 The job may do

The automated cached-source acquisition job may:

1. Read an approved ticker universe or explicit bounded ticker list.
2. Determine approved source families for each ticker.
3. Use approved provider/source adapters.
4. Retrieve source data.
5. Capture provenance.
6. Determine freshness and staleness state.
7. Write payload files.
8. Write `manifest.json`.
9. Record payload hashes and sizes.
10. Report fail-closed results.
11. Place output under an explicit `destination_root`.
12. Prepare output for the existing import/staging/dry-run flow.

### 4.2 The job must not do

The automated cached-source acquisition job must not:

1. Execute analysis.
2. Produce a recommendation.
3. Produce a decision.
4. Modify a portfolio.
5. Modify a watchlist.
6. Send Telegram messages.
7. Call a broker or execution system.
8. Create orders.
9. Introduce BUY / SELL / HOLD semantics.
10. Introduce target prices.
11. Introduce allocation or position sizing.
12. Introduce ranking, urgency, conviction, scoring, or tradeability authority.
13. Bypass existing validators.
14. Bypass existing staging.
15. Bypass the existing cached-source dry-run path.
16. Write production outputs.
17. Silently substitute one provider/source for another.
18. Fabricate payloads, timestamps, numeric values, or provenance.

## 5. Input Contract

Future request format:

```text
market-engine-automated-cached-source-acquisition-request-v1
```

### 5.1 Request payload shape

```json
{
  "request_format": "market-engine-automated-cached-source-acquisition-request-v1",
  "request_id": "me-sa02-local-20260626T120000Z",
  "requested_at": "2026-06-26T12:00:00Z",
  "run_mode": "dry_run",
  "ticker_source": {
    "mode": "explicit_list",
    "source_id": "operator_bounded_list"
  },
  "tickers": ["NVDA", "AMD", "ASML"],
  "source_families": ["company_profile"],
  "destination_root": "artifacts/market_engine/source_acquisition/me-sa02-local-20260626T120000Z",
  "freshness_policy": {
    "default_max_age_days": 7,
    "per_source_family": {
      "company_profile": {
        "max_age_days": 30,
        "source_timestamp_required": false
      }
    }
  },
  "provider_policy": {
    "approved_adapters": [
      {
        "adapter_id": "fake_company_profile_adapter",
        "adapter_version": "test-v1",
        "source_families": ["company_profile"],
        "allowed_run_modes": ["dry_run", "local_non_production"]
      }
    ],
    "allow_hidden_fallback": false,
    "allow_silent_substitution": false,
    "allow_fabricated_data": false
  },
  "safety_flags": {
    "allow_provider_calls": false,
    "allow_network": false,
    "allow_production_writes": false,
    "allow_telegram_send": false,
    "allow_portfolio_writes": false,
    "allow_watchlist_writes": false,
    "allow_broker_actions": false
  },
  "operator_context": {
    "requested_by": "operator",
    "purpose": "bounded local acquisition contract validation",
    "notes": []
  }
}
```

### 5.2 Field rules

| Field | Type | Required | Meaning | Validation rules |
|---|---:|---:|---|---|
| `request_format` | string | Yes | Request contract identifier. | Must equal `market-engine-automated-cached-source-acquisition-request-v1`. Unknown values must be rejected. |
| `request_id` | string | Yes | Stable request identifier supplied by caller or generated by command layer. | Must be non-empty, deterministic for the request, filesystem-safe if used in paths, and unique enough to avoid accidental overwrite. |
| `requested_at` | string timestamp | Yes | UTC timestamp when request was created. | Must be ISO-8601/RFC3339 compatible. Missing or invalid timestamp must reject request. |
| `run_mode` | enum string | Yes | Declares execution mode and allowed side effects. | Must be one of the approved run modes in this contract. Unknown modes must be rejected. |
| `ticker_source` | object | Yes | Declares where the ticker list came from. | Must include `mode`. If `mode=explicit_list`, `tickers` must be present and bounded. If `mode=approved_universe`, the universe identity must be present. |
| `tickers` | list[string] | Conditionally required | Explicit bounded ticker list. | Required when `ticker_source.mode=explicit_list`. Must be non-empty, bounded, deduplicated or reject duplicate entries, and validated per ticker format rules. |
| `source_families` | list[string] | Yes | Requested source families to acquire. | Must be non-empty. Every value must be contractually known and allowed for the selected run mode. Unsupported values must fail closed. |
| `destination_root` | string path | Yes | Explicit root under which snapshot packages and result files may be written. | Must be local, explicit, non-production by default, and must not resolve outside the allowed artifact area. |
| `freshness_policy` | object | Yes | Max-age and source timestamp rules. | Must define default or per-source-family rules. Invalid or missing rules must fail closed for affected entries. |
| `provider_policy` | object | Yes | Approved adapters and provider/source rules. | Must include approved adapters or a reference to an approved adapter policy. Hidden fallback, silent substitution, and fabricated data must be false. |
| `safety_flags` | object | Yes | Side-effect permissions for the run. | In ME-SA02 default implementation, provider/network/production/Telegram/portfolio/watchlist/broker actions must remain false unless separately approved by a later sprint. |
| `operator_context` | object | Optional | Human-readable context for auditability. | Must not be required for machine behavior. If present, must be preserved in result metadata where appropriate. |

## 6. Approved Run Modes

ME-SA01 approves the following run modes at contract level.

| Run mode | Status | Meaning |
|---|---|---|
| `dry_run` | Approved as default | Does not perform live provider calls. May use deterministic fake adapters and local fixtures to validate package writing and downstream compatibility. |
| `local_non_production` | Approved for bounded implementation planning | May write local non-production artifacts. Live provider calls are not approved by ME-SA01 itself. |
| `scheduled_non_production` | Contractually reserved | May be defined later for scheduled non-production acquisition. Not approved for implementation by ME-SA01. |

ME-SA01 does not approve production scheduled live behavior.

The following are not approved:

- `production`
- `scheduled_production`
- `live_trading`
- `telegram_delivery`
- `portfolio_update`
- `watchlist_update`

Unknown run modes must fail closed.

## 7. Ticker Input Rules

The job must acquire data only for an approved ticker universe or an explicit bounded ticker list.

### 7.1 Supported ticker sources

| Ticker source mode | Contract status | Rules |
|---|---|---|
| `explicit_list` | Approved for ME-SA02 | Caller supplies a small bounded ticker list. |
| `approved_universe` | Contractually reserved | Later sprint may define approved universe source, versioning, and persistence. |
| `portfolio_context` | Not approved | Acquisition must not infer ticker list from portfolio writes or live portfolio state in ME-SA01/ME-SA02. |
| `watchlist` | Not approved | Acquisition must not read or modify watchlists in ME-SA01/ME-SA02. |

### 7.2 Ticker validation

Ticker validation must:

1. Reject an empty ticker list.
2. Reject unsupported tickers.
3. Reject malformed tickers.
4. Reject duplicate tickers or normalize/deduplicate with an explicit warning in the result.
5. Preserve ticker identity exactly as validated.
6. Keep per-ticker acquisition isolated.

Suggested initial ticker format rules:

- Uppercase letters, digits, dot, or hyphen only.
- Minimum length: 1.
- Maximum length: 12.
- No whitespace.
- No path separators.
- No shell metacharacters.
- No URL-like strings.
- No empty strings.
- No null values.

### 7.3 Per-ticker isolation

A failure for one ticker must not silently contaminate another ticker.

Per-ticker result entries must preserve:

- ticker;
- requested source family;
- adapter identity;
- status;
- issues;
- paths written, if any;
- provenance, if retrieval occurred;
- validation state.

## 8. Source Family Rules

ME-SA01 distinguishes three levels:

1. Source family contractually provided by the architecture.
2. Source family allowed for ME-SA02 implementation.
3. Source family deferred to a later sprint.

| Source family | Contractually provided | Allowed in ME-SA02 | Notes |
|---|---:|---:|---|
| `sec_companyfacts` | Yes | No, unless a later sprint explicitly approves SEC/EDGAR access | Existing Market Engine semantics depend heavily on SEC Companyfacts, but ME-SA01 does not approve SEC/EDGAR provider calls. |
| `official_investor_relations_document` | Yes | No | Requires document discovery, canonical URL policy, parsing boundaries, and source timestamp rules in a later sprint. |
| `market_price_snapshot` | Yes | No | Requires provider policy, timestamp semantics, market data licensing review, and price freshness rules in a later sprint. |
| `earnings_release` | Yes | No | Requires canonical source identity, publication timestamp handling, and document hashing rules in a later sprint. |
| `company_profile` | Yes | Yes, with deterministic fake adapter first | Suitable for ME-SA02 package-writing implementation with deterministic test adapter and no real provider calls. |

ME-SA02 should implement at least one approved source family. The recommended first source family is `company_profile` using a deterministic fake adapter for tests and local package compatibility validation.

No source family may be implemented with live provider calls in ME-SA02 unless a later sprint explicitly updates the provider/source policy.

## 9. Provider / Source Adapter Policy

### 9.1 Approved adapter list

The acquisition request must include either:

1. An explicit `provider_policy.approved_adapters` list; or
2. A reference to a versioned approved adapter policy defined elsewhere.

Each adapter entry must define:

| Field | Required | Meaning |
|---|---:|---|
| `adapter_id` | Yes | Stable adapter identifier. |
| `adapter_version` | Yes | Adapter contract or implementation version. |
| `source_families` | Yes | Source families the adapter may provide. |
| `allowed_run_modes` | Yes | Run modes in which the adapter may be used. |
| `provider_name` | Required for real adapters | Provider or source name. |
| `canonical_source_identity` | Required for real adapters | Stable identity of the provider/source endpoint or document source. |
| `network_required` | Yes | Whether adapter requires network access. |
| `rate_limit_policy` | Required for real adapters | Rate limit and retry boundaries. |
| `error_policy` | Yes | How provider errors are surfaced. |

### 9.2 Adapter identity

Every result entry must record:

- adapter ID;
- adapter version;
- provider/source name if real;
- canonical source identity;
- request metadata;
- retrieval timestamp;
- source timestamp when available;
- payload hash;
- payload size.

### 9.3 No hidden fallback

The job must not:

1. Fall back to an unapproved provider.
2. Replace one source family with another source family.
3. Replace one ticker with another ticker.
4. Substitute stale local data while reporting it as freshly retrieved.
5. Fabricate missing payloads.
6. Convert missing numeric values to zero.
7. Hide adapter errors behind generic success states.

All fallback behavior must be explicitly approved, visible in the request/result, and compatible with fail-closed handling.

### 9.4 Provider errors and rate limits

Provider/source adapter failures must be represented as explicit entry statuses and issues.

The job must distinguish at least:

- provider unavailable;
- rate limited;
- timeout;
- authentication/configuration missing;
- unsupported ticker;
- unsupported source family;
- invalid provider response;
- missing required payload;
- stale payload;
- manifest write failure.

## 10. Output Contract

Future result format:

```text
market-engine-automated-cached-source-acquisition-result-v1
```

### 10.1 Result payload shape

```json
{
  "result_format": "market-engine-automated-cached-source-acquisition-result-v1",
  "request_id": "me-sa02-local-20260626T120000Z",
  "run_id": "me-sa02-local-20260626T120000Z",
  "generated_at": "2026-06-26T12:01:00Z",
  "destination_root": "artifacts/market_engine/source_acquisition/me-sa02-local-20260626T120000Z",
  "entries": [
    {
      "ticker": "NVDA",
      "source_family": "company_profile",
      "status": "completed",
      "snapshot_path": "artifacts/market_engine/source_acquisition/me-sa02-local-20260626T120000Z/NVDA/company_profile",
      "manifest_path": "artifacts/market_engine/source_acquisition/me-sa02-local-20260626T120000Z/NVDA/company_profile/manifest.json",
      "payload_paths": [
        "artifacts/market_engine/source_acquisition/me-sa02-local-20260626T120000Z/NVDA/company_profile/company_profile.json"
      ],
      "provenance": {
        "adapter_id": "fake_company_profile_adapter",
        "adapter_version": "test-v1",
        "provider_name": "deterministic_fake_provider",
        "canonical_source_identity": "fake://company_profile/NVDA",
        "retrieved_at": "2026-06-26T12:00:30Z",
        "source_timestamp": null,
        "request_metadata": {
          "network_used": false
        }
      },
      "freshness": {
        "retrieval_timestamp_required": true,
        "source_timestamp_required": false,
        "max_age_days": 30,
        "state": "current"
      },
      "issues": []
    }
  ],
  "summary": {
    "requested_ticker_count": 3,
    "requested_source_family_count": 1,
    "entry_count": 3,
    "completed_count": 3,
    "completed_with_limitations_count": 0,
    "blocked_count": 0,
    "rejected_count": 0,
    "provider_error_count": 0,
    "unsupported_count": 0,
    "stale_count": 0,
    "invalid_manifest_count": 0
  },
  "safety": {
    "provider_calls_performed": false,
    "network_used": false,
    "telegram_sent": false,
    "portfolio_written": false,
    "watchlist_written": false,
    "broker_action_performed": false,
    "production_write_performed": false
  },
  "next_step": {
    "recommended_action": "run_existing_import_staging_validation",
    "import_candidate_root": "artifacts/market_engine/source_acquisition/me-sa02-local-20260626T120000Z",
    "dry_run_candidate": true
  }
}
```

### 10.2 Result fields

| Field | Type | Required | Meaning |
|---|---:|---:|---|
| `result_format` | string | Yes | Must equal `market-engine-automated-cached-source-acquisition-result-v1`. |
| `request_id` | string | Yes | Echoes request ID. |
| `run_id` | string | Yes | Unique acquisition run ID. |
| `generated_at` | string timestamp | Yes | UTC timestamp when result was generated. |
| `destination_root` | string path | Yes | Root under which artifacts were written. |
| `entries` | list[object] | Yes | Per ticker/source-family acquisition entries. |
| `summary` | object | Yes | Aggregated counts by status and requested scope. |
| `safety` | object | Yes | Side-effect audit flags. |
| `next_step` | object | Yes | Machine-readable hint for downstream import/staging/dry-run flow. |

### 10.3 Entry fields

| Field | Type | Required | Meaning |
|---|---:|---:|---|
| `ticker` | string | Yes | Validated ticker. |
| `source_family` | string | Yes | Requested source family. |
| `status` | enum string | Yes | Entry acquisition status. |
| `snapshot_path` | string/null | Required | Snapshot directory if written, else null. |
| `manifest_path` | string/null | Required | Manifest path if written, else null. |
| `payload_paths` | list[string] | Yes | Payload files written for this entry. Empty if none. |
| `provenance` | object/null | Required | Provenance metadata if retrieval or deterministic adapter generation occurred. |
| `freshness` | object/null | Required | Freshness/staleness metadata if payload exists. |
| `issues` | list[object/string] | Yes | Explicit issues, warnings, or rejection reasons. |

## 11. Snapshot Package Compatibility

The job output must be compatible with existing cached-source snapshot manifest, staging, and import expectations established by the cached-source snapshot work.

At minimum, each accepted snapshot package must contain:

- `manifest.json`;
- one or more payload files when status allows a usable snapshot;
- payload hashes;
- payload sizes;
- ticker;
- `snapshot_id`;
- `batch_id`;
- `source_family`;
- provider/source identity;
- retrieval timestamp;
- source timestamp when available;
- freshness/stale state;
- validation state;
- usable flag.

The snapshot package must be suitable for existing import/staging validation and later `cached_source_snapshot` dry-run consumption.

A snapshot package must not be marked usable unless:

1. Required manifest fields are present.
2. Required payload files exist.
3. Payload file hash and size metadata match the actual files.
4. Ticker identity is valid.
5. Source family is approved.
6. Provider/source identity is known.
7. Retrieval timestamp is present.
8. Freshness/staleness state is explicit.
9. Missing-data markers are preserved.
10. The existing validator can accept or explicitly reject the package with documented reasons.

## 12. Provenance Requirements

Every payload must have non-anonymous provenance.

Required provenance:

- Where the data came from.
- When it was retrieved.
- Which provider/source supplied it.
- Which adapter retrieved it.
- Adapter version.
- Canonical source identity.
- Request metadata relevant to retrieval.
- Source timestamp if available.
- Payload hash.
- Payload size.
- Snapshot ID.
- Batch ID.

The job must not overwrite an existing snapshot in place without generating a new `snapshot_id`.

If the same ticker/source-family is acquired multiple times, each acquisition must be distinguishable by `snapshot_id`, `batch_id`, or another approved immutable identity.

## 13. Freshness / Staleness Policy

The request must define a freshness policy.

Policy fields:

| Field | Required | Meaning |
|---|---:|---|
| `default_max_age_days` | Required unless all families define explicit policy | Default max age for source data. |
| `per_source_family` | Optional | Overrides for specific source families. |
| `source_timestamp_required` | Required per family or default | Whether the source must provide its own timestamp. |
| `retrieval_timestamp_required` | Always true | Retrieval timestamp must always be recorded. |
| `stale_state_values` | Optional | Allowed stale state values if centrally defined. |

Freshness state must be explicit.

Suggested freshness states:

- `current`
- `stale`
- `unknown_source_timestamp`
- `not_applicable`
- `rejected_missing_timestamp`

Rules:

1. Retrieval timestamp is always required.
2. Source timestamp may be required or optional per source family.
3. Stale data must never be silently treated as current.
4. Stale entries must be visible in entry status, freshness metadata, or issues.
5. A stale entry may be written as a non-usable diagnostic artifact only if the manifest clearly marks it stale and not silently current.
6. Downstream dry-run and review layers must preserve stale-data markers.

## 14. Missing-Data Policy

Missing data must remain explicit.

Rules:

1. Missing required payload means the entry must be `blocked`, `rejected`, or another non-usable failure status.
2. Missing optional payload may result in `completed_with_limitations`.
3. Missing-data markers must be preserved for downstream validation and dry-run stages.
4. Missing numeric values must never be converted to zero.
5. Missing timestamps must never be replaced with retrieval timestamp unless explicitly represented as retrieval timestamp, not source timestamp.
6. Empty payloads must not be treated as valid payloads unless the source family contract explicitly allows empty payloads.
7. Optional payload absence must be recorded in `issues`.

## 15. Failure Model

ME-SA01 defines the following entry statuses.

| Status | Meaning | Usable as snapshot input |
|---|---|---:|
| `planned` | Entry was planned but not attempted. | No |
| `completed` | Required payloads and manifest were written successfully. | Yes, subject to validator acceptance |
| `completed_with_limitations` | Required payloads were written, but optional payloads or non-critical metadata were limited. | Yes, subject to validator acceptance and limitation markers |
| `blocked` | Entry could not proceed because request, safety, configuration, or policy blocked it. | No |
| `rejected` | Entry was rejected by validation rules. | No |
| `provider_error` | Approved adapter/provider failed. | No |
| `unsupported` | Ticker, source family, adapter, or mode is unsupported. | No |
| `stale` | Data was acquired but violates freshness policy. | No by default, unless a later source-family contract allows stale diagnostic artifacts |
| `invalid_manifest` | Manifest was missing, incomplete, inconsistent, or incompatible. | No |

A run-level result may contain mixed entry statuses. A run must fail closed if all entries are unusable.

## 16. Safety and Side Effects

The acquisition job is a source-data job only.

ME-SA01 explicitly requires:

- no Telegram send;
- no portfolio writes;
- no watchlist writes;
- no broker or execution action;
- no production writes;
- no downstream recommendation authority;
- no BUY / SELL / HOLD semantics;
- no target price;
- no allocation;
- no position sizing;
- no ranking;
- no urgency;
- no conviction;
- no tradeability authority.

Output may only be written to the explicit `destination_root`.

`dry_run` remains the default mode for the first implementation.

Any later mode that performs real network/provider calls must be explicitly approved by a later sprint with source-specific policy, rate-limit handling, secrets/configuration rules, and audit expectations.

## 17. Handoff to Existing Flow

ME-SA02 output must be shaped so that it can be handed off to the existing cached-source flow:

```text
ME-SA02 acquisition output
  -> ME-SR12 import command
  -> ME-SR10 validator/staging expectations
  -> ME-RUN25-proven fixture import/staging/dry-run path
  -> cached_source_snapshot dry-run
```

The acquisition job must produce snapshot packages that the existing import/staging validator can accept or reject with clear reasons.

If the local dry-run cannot consume an acquisition output package, the system must block with a documented reason rather than bypassing validation.

## 18. Acceptance Criteria for ME-SA02

ME-SA02 should implement the first bounded automated cached-source acquisition job.

Acceptance criteria:

1. Implementation uses a bounded ticker list of `NVDA`, `AMD`, `ASML`, or a smaller explicit list.
2. Implementation supports at least one approved source family.
3. Recommended first source family is `company_profile`.
4. Deterministic tests use a fake adapter.
5. Tests perform no real provider calls.
6. Tests perform no network calls.
7. Tests create no production writes.
8. Tests do not require yfinance.
9. Tests do not call SEC/EDGAR.
10. Runtime default is `dry_run`.
11. Any real fetch path, if later approved, must be opt-in through an explicit manual command or flag.
12. Snapshot package includes `manifest.json`.
13. Snapshot package includes payload files when entry status is usable.
14. Manifest includes ticker, snapshot ID, batch ID, source family, provider/source identity, retrieval timestamp, source timestamp when applicable, payload hashes, payload sizes, freshness/stale state, validation state, and usable flag.
15. Existing staging/import validator accepts the output or rejects it with documented reasons.
16. Local cached-source dry-run can consume the output or blocks with a documented reason.
17. No Telegram send is performed.
18. No portfolio/watchlist writes are performed.
19. No Decision Engine, Recommendation Review, Portfolio Review, or Delivery semantics are changed.
20. No BUY / SELL / HOLD, target price, allocation, position sizing, ranking, urgency, conviction, or tradeability authority is introduced.

## 19. Recommended Next Sprints

### ME-SA02 — Implement first bounded automated cached-source acquisition job

Implement the first local, bounded, deterministic acquisition job using the ME-SA01 contract.

Expected scope:

- bounded ticker list;
- at least one approved source family;
- deterministic fake adapter in tests;
- snapshot package writing;
- result payload writing;
- compatibility with existing import/staging expectations.

### ME-RUN26 — Run automated cached-source acquisition for NVDA/AMD/ASML through staging validation and local dry-run

Execute the ME-SA02 job through the existing cached-source import/staging/dry-run flow.

Expected scope:

- bounded acquisition run;
- staging/import validation;
- local dry-run attempt;
- audit of pass/block result.

### ME-TP01 — Produce terminal-visible operator preview from real cached-source dry-run artifacts

Produce operator-visible preview output from cached-source dry-run artifacts.

Expected scope:

- terminal-visible preview;
- Telegram-style formatting only;
- no Telegram send unless separately approved;
- no recommendation or execution authority.

## 20. ME-SA01 Non-Goals

ME-SA01 does not:

- implement runtime code;
- change tests;
- perform provider calls;
- fetch live data;
- use yfinance;
- call SEC/EDGAR;
- use internet access;
- create source files;
- create fake NVDA/AMD/ASML data;
- send Telegram messages;
- write portfolio state;
- write watchlist state;
- write production outputs;
- modify Decision Engine semantics;
- modify Recommendation Review semantics;
- modify Portfolio Review semantics;
- modify Delivery semantics;
- introduce BUY / SELL / HOLD;
- introduce target price;
- introduce allocation;
- introduce position sizing;
- introduce ranking;
- introduce urgency;
- introduce conviction;
- introduce tradeability authority.

## 21. Contract Conclusion

ME-SA01 establishes the automated cached-source acquisition job as the primary route for getting real-world source data into Market Engine while preserving the existing fail-closed cached-source validation and dry-run architecture.

The next implementation sprint should be ME-SA02.
