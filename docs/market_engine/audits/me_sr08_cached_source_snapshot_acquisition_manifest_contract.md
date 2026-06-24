# ME-SR08 — Cached-source snapshot acquisition manifest contract

## 1. Purpose

ME-SR08 defines the documentation-only contract for cached-source snapshot acquisition manifests.

An acquisition manifest records metadata about a locally acquired or staged cached-source snapshot. It is not the source payload itself, not a normalized source-context object, not an analysis result, and not a recommendation. Its purpose is to make future local cached-source dry-runs auditable, reproducible, and governance-safe.

The manifest answers:

- what was requested;
- what was collected;
- where the snapshot was stored locally;
- which source family it belongs to;
- which ticker/entity it belongs to;
- when it was acquired or staged;
- whether it passed validation;
- whether it can be used by later cached-source dry-runs.

ME-SR08 does not acquire data. It defines the contract a later implementation must follow.

## 2. Non-goals

ME-SR08 does not:

- implement snapshot acquisition;
- fetch live data;
- call SEC, EDGAR, yfinance, Alpha Vantage, brokers, Telegram, or external APIs;
- define source-specific parsers;
- modify cached-source dry-run execution;
- create source context, observations, analysis, recommendations, reports, or delivery output;
- create investment advice;
- create trade, allocation, order, execution, or broker authority;
- add broker integration;
- bypass source licensing or redistribution constraints;
- permit source payloads to be committed to git;
- mark missing, stale, partial, invalid, unsupported, or restricted source data as usable.

## 3. Manifest Scope

ME-SR08 defines two related manifest levels.

| Manifest | Required for future acquisition/staging? | Purpose |
|---|---:|---|
| Snapshot-level manifest | yes | One manifest per acquired or staged cached-source snapshot. It records exact payload, source, entity, validation, governance, and local path metadata. |
| Batch-level manifest | optional for a single snapshot, required for acquisition batches | One manifest per acquisition batch. It lists snapshot manifests, batch-level request metadata, and validation summary counts. |

The snapshot-level manifest is the authoritative record for whether a payload can be used by cached-source local execution.

The batch-level manifest must reference snapshot-level manifests. It must not replace them.

Manifest requirements:

- machine-readable JSON;
- deterministic key names;
- stable format version;
- no embedded restricted raw payload content;
- human-readable enough for audit review;
- local-first paths;
- explicit validation and governance fields;
- stable references from later dry-run artifacts.

Preferred future local path pattern:

```text
data/market_engine/cached_source_snapshots/<batch_id>/<ticker>/<snapshot_id>/payload.<ext>
data/market_engine/cached_source_snapshots/<batch_id>/<ticker>/<snapshot_id>/manifest.json
data/market_engine/cached_source_snapshots/<batch_id>/batch_manifest.json
```

Existing SEC CompanyFacts snapshots currently use:

```text
data/market_engine/source_snapshots/sec_companyfacts/<run_id>/raw/<snapshot_id>.json
```

A future implementation may bridge the existing path into this manifest contract, but must not silently move or rewrite existing snapshots.

## 4. Required Manifest Fields

Snapshot manifests must use this contract version:

```text
market-engine-cached-source-snapshot-acquisition-manifest-v1
```

Required fields:

- `manifest_format_version`
- `snapshot_id`
- `batch_id`
- `created_at_utc`
- `acquired_at_utc`
- `acquisition_mode`
- `source_family`
- `source_name`
- `source_url`
- `source_license_note`
- `redistribution_allowed`
- `local_use_allowed`
- `commit_allowed`
- `source_material_type`
- `ticker`
- `entity_name`
- `entity_country`
- `entity_exchange`
- `source_entity_identifier`
- `cik`
- `requested_document_type`
- `resolved_document_type`
- `requested_period`
- `resolved_period`
- `source_publication_date`
- `source_retrieved_at_utc`
- `local_snapshot_path`
- `local_manifest_path`
- `local_payload_sha256`
- `local_payload_size_bytes`
- `payload_mime_type`
- `payload_encoding`
- `normalization_status`
- `validation_status`
- `validation_errors`
- `validation_warnings`
- `staleness_status`
- `staleness_reason`
- `usable_for_cached_source_dry_run`
- `blocked_reason`
- `notes`

Example snapshot manifest:

```json
{
  "manifest_format_version": "market-engine-cached-source-snapshot-acquisition-manifest-v1",
  "snapshot_id": "example-asml-annual-report-2025",
  "batch_id": "example-me-sr11-manual-staging-20260701T090000Z",
  "created_at_utc": "2026-07-01T09:05:00Z",
  "acquired_at_utc": "2026-07-01T09:00:00Z",
  "acquisition_mode": "operator_supplied",
  "source_family": "company_investor_relations_document",
  "source_name": "ASML investor relations annual reports",
  "source_url": "https://www.example.invalid/asml/annual-report-2025.pdf",
  "source_license_note": "Illustrative example only; future operator must record actual source-use constraints before local use.",
  "redistribution_allowed": false,
  "local_use_allowed": true,
  "commit_allowed": false,
  "source_material_type": "company_published_material",
  "ticker": "ASML",
  "entity_name": "ASML Holding",
  "entity_country": "NL",
  "entity_exchange": "EURONEXT",
  "source_entity_identifier": "example-asml-ir",
  "cik": "0000937966",
  "requested_document_type": "annual_report",
  "resolved_document_type": "annual_report",
  "requested_period": "FY2025",
  "resolved_period": "FY2025",
  "source_publication_date": "2026-02-15",
  "source_retrieved_at_utc": "2026-07-01T09:00:00Z",
  "local_snapshot_path": "data/market_engine/cached_source_snapshots/example-me-sr11-manual-staging-20260701T090000Z/ASML/example-asml-annual-report-2025/payload.pdf",
  "local_manifest_path": "data/market_engine/cached_source_snapshots/example-me-sr11-manual-staging-20260701T090000Z/ASML/example-asml-annual-report-2025/manifest.json",
  "local_payload_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "local_payload_size_bytes": 1234567,
  "payload_mime_type": "application/pdf",
  "payload_encoding": "binary",
  "normalization_status": "raw_only",
  "validation_status": "not_validated",
  "validation_errors": [],
  "validation_warnings": [
    "Illustrative example; no payload was acquired by ME-SR08."
  ],
  "staleness_status": "unknown",
  "staleness_reason": "Illustrative example; future validator must evaluate staleness policy.",
  "usable_for_cached_source_dry_run": false,
  "blocked_reason": "illustrative_example_not_validated",
  "notes": "This is a contract example only. ME-SR08 did not fetch or stage this file."
}
```

## 5. Field Semantics

| Field | Type | Required | Allowed values | Meaning | Failure/null behavior |
|---|---|---:|---|---|---|
| `manifest_format_version` | string | yes | `market-engine-cached-source-snapshot-acquisition-manifest-v1` | Contract version for the snapshot manifest. | Unsupported or missing version blocks use. |
| `snapshot_id` | string | yes | stable path-safe identifier | Unique snapshot identifier within a batch. | Missing, empty, duplicate, or unsafe IDs block use. |
| `batch_id` | string | yes | stable path-safe identifier | Acquisition or staging batch identifier. | Missing or unsafe IDs block use. |
| `created_at_utc` | string | yes | UTC ISO-8601 timestamp | When the manifest was written. | Missing or non-UTC timestamp blocks use. |
| `acquired_at_utc` | string | yes | UTC ISO-8601 timestamp | When the payload was acquired or staged locally. | Missing timestamp blocks use. |
| `acquisition_mode` | string | yes | `manual_cached_download`, `scripted_local_acquisition`, `test_fixture`, `operator_supplied`, `approved_bundle_import` | How the payload entered local storage. | Unknown mode blocks use. `test_fixture` cannot be real coverage. |
| `source_family` | string | yes | approved source family id | Logical source family, for example `sec_companyfacts` or `company_investor_relations_document`. | Unknown family blocks use unless explicitly marked unsupported and not usable. |
| `source_name` | string | yes | source-specific text | Human-readable source name. | Missing value blocks use. |
| `source_url` | string or null | yes | URL or null | Original source URL when available and allowed to record. | Null requires note explaining absence. |
| `source_license_note` | string | yes | non-empty text | Source-use and licensing note. | Missing note blocks use. |
| `redistribution_allowed` | boolean | yes | true, false | Whether raw payload redistribution is allowed. | Missing value blocks use. False usually means raw payload must remain local-only. |
| `local_use_allowed` | boolean | yes | true, false | Whether local cached-source dry-run use is allowed. | False blocks use. |
| `commit_allowed` | boolean | yes | true, false | Whether raw payload may be committed to git. | Missing value blocks use. False means payload must not be committed. |
| `source_material_type` | string | yes | `official_regulatory_source`, `company_published_material`, `third_party_material`, `synthetic_fixture`, `operator_supplied_material` | Governance class for the source. | Unknown type blocks use. `synthetic_fixture` cannot count as real coverage. |
| `ticker` | string | yes | normalized ticker | Requested or resolved ticker. | Missing ticker blocks use. |
| `entity_name` | string | yes | non-empty text | Resolved company/entity name. | Missing value blocks use. |
| `entity_country` | string | yes | ISO-like country code or explicit text | Entity country or jurisdiction. | Missing value blocks use, especially for non-US rows. |
| `entity_exchange` | string | yes | exchange/listing context | Listing or exchange context. | Missing value blocks use for non-US/source-mapped rows. |
| `source_entity_identifier` | string or null | yes | source-specific id or null | Identifier used by the source family. | Null requires explicit note; ambiguous identity blocks use. |
| `cik` | string or null | yes | normalized CIK or null | SEC CIK when applicable. | Null is allowed only when source family does not use CIK. |
| `requested_document_type` | string | yes | contract-defined document type | Document type requested. | Missing value blocks use. |
| `resolved_document_type` | string or null | yes | document type or null | Document type actually found. | Null requires validation warning; may block depending on source family. |
| `requested_period` | string or null | yes | period label or null | Requested period, for example `FY2025`. | Null allowed only when not applicable. |
| `resolved_period` | string or null | yes | period label or null | Period resolved from source. | Null requires validation warning or error. |
| `source_publication_date` | string or null | yes | ISO date or null | Date the source was published, if known. | Null must be visible and may affect staleness. |
| `source_retrieved_at_utc` | string | yes | UTC ISO-8601 timestamp | When the operator/system retrieved the source. | Missing value blocks use. |
| `local_snapshot_path` | string | yes | local relative path | Local payload path. | Missing path or path outside allowed roots blocks use. |
| `local_manifest_path` | string | yes | local relative path | Local manifest path. | Missing or mismatched path blocks use. |
| `local_payload_sha256` | string | yes | 64 lowercase hex chars | Content hash. | Missing or mismatched hash blocks use. |
| `local_payload_size_bytes` | integer | yes | integer greater than 0 | Payload size. | Missing, non-integer, or zero blocks use. |
| `payload_mime_type` | string | yes | MIME type | Payload media type. | Missing or unsupported MIME type blocks use. |
| `payload_encoding` | string | yes | `utf-8`, `binary`, `base64`, `unknown` | Payload encoding. | `unknown` requires warning and may block parser use. |
| `normalization_status` | string | yes | `raw_only`, `normalized`, `normalization_failed`, `not_applicable` | Whether a normalized derivative exists. | `normalization_failed` blocks normalized use; may still allow raw-only storage if validation passes. |
| `validation_status` | string | yes | `passed`, `failed`, `warning`, `not_validated` | Snapshot validation state. | `failed` or `not_validated` requires `usable_for_cached_source_dry_run=false`. |
| `validation_errors` | array of strings | yes | deterministic strings | Blocking validation errors. | Non-empty errors require `usable_for_cached_source_dry_run=false`. |
| `validation_warnings` | array of strings | yes | deterministic strings | Non-blocking warnings. | Warnings may allow use only when no blocking conditions exist. |
| `staleness_status` | string | yes | `fresh`, `stale`, `unknown`, `not_applicable` | Staleness assessment. | `stale` may block by policy. `unknown` must remain visible. |
| `staleness_reason` | string | yes | deterministic text | Explanation for staleness state. | Missing reason blocks use. |
| `usable_for_cached_source_dry_run` | boolean | yes | true, false | Whether later cached-source dry-runs may consume the snapshot. | Must be false when validation failed, source constraints block usage, or required metadata is missing. |
| `blocked_reason` | string or null | yes | reason code or null | Blocking reason when not usable. | Required when usable is false. Must be null when usable is true. |
| `notes` | string | yes | text, may be empty only after validation | Operator/audit notes. | Missing field blocks use. Notes must not include action recommendations. |

## 6. Source-Family Coverage

The manifest contract can describe multiple source families without approving every family for runtime use.

| Source family | Manifest handling |
|---|---|
| SEC/companyfacts-style source payloads | Must record SEC-style entity identity, CIK when applicable, source name, payload format, source retrieval timestamp, local path, hash, and validation against expected ticker/CIK. Existing `sec-companyfacts-raw-v1` snapshots may be referenced by manifest path and payload hash. |
| Company investor relations documents | Must record source URL, publication date where available, retrieval timestamp, company-published material classification, country, exchange/listing context, source/license note, and whether redistribution is allowed. |
| Annual reports / interim reports | Must record requested and resolved document type, requested and resolved period, source publication date, and whether the document is official regulatory material or company-published material. |
| Exchange or regulator filings | Must record regulator/exchange source family, entity identifier, exchange/listing context, publication date, retrieval timestamp, URL, and source-use note. |
| Non-US issuer source material | Must record country, exchange/listing context, source entity identifier, CIK only when applicable, document type, publication date, source URL, source-use restrictions, and source-family governance status. |
| Operator-supplied local files | Must record acquisition mode, operator note, source origin, hash, size, local path, and whether source-use constraints permit local dry-run use. |
| Synthetic test fixtures | Must be marked `source_material_type=synthetic_fixture` and `acquisition_mode=test_fixture`; must never be treated as real source coverage. |

For non-US and investor-relations material, the manifest must clearly distinguish official filings from company marketing or investor-relations material. A current classifier successfully loading a snapshot is not enough to establish broad non-US source-family governance.

## 7. Governance and Source-Use Constraints

The manifest must record:

- whether redistribution is allowed;
- whether local use is allowed;
- whether raw payload commit to git is allowed;
- whether the snapshot is official regulatory source material, company-published material, third-party material, operator-supplied material, or a synthetic fixture;
- whether the snapshot must remain local-only;
- whether derived artifacts may reference the source metadata;
- any operator note needed for licensing/source restrictions.

Governance rules:

- Raw third-party payloads should not be committed unless explicitly allowed.
- Local-only snapshots may be used for local dry-runs only when `local_use_allowed=true` and validation passes.
- Manifests may be committed only if they do not expose restricted payload content.
- Source URL and metadata may be stored when legally and operationally acceptable.
- The manifest does not grant permission to use data beyond the recorded constraints.
- Source-use restrictions that block local use must force `usable_for_cached_source_dry_run=false`.

## 8. Local Path and Artifact Relationship

Preferred future acquisition path:

```text
data/market_engine/cached_source_snapshots/<batch_id>/<ticker>/<snapshot_id>/payload.<ext>
data/market_engine/cached_source_snapshots/<batch_id>/<ticker>/<snapshot_id>/manifest.json
data/market_engine/cached_source_snapshots/<batch_id>/batch_manifest.json
```

Later dry-run artifacts should reference validated snapshots by:

- `snapshot_id`;
- snapshot manifest path;
- payload hash;
- source family;
- ticker/entity;
- source publication date;
- staleness status.

Dry-run artifacts should not need to inspect the raw acquisition process. They should consume a validated local snapshot reference and preserve the manifest metadata needed for audit.

Existing dry-run artifact manifests remain separate from acquisition manifests. Acquisition manifests describe source payload acquisition; dry-run manifests describe dry-run artifact persistence.

## 9. Validation Contract

Minimum validation rules before `usable_for_cached_source_dry_run=true`:

1. All required manifest fields are present.
2. `manifest_format_version` is supported.
3. `snapshot_id` is unique within the batch.
4. Local payload exists.
5. Payload path is under an approved snapshot directory.
6. Payload size is greater than zero.
7. SHA-256 hash is computed and matches the local payload.
8. Source family is supported or explicitly marked unsupported.
9. Ticker/entity resolution is present.
10. Non-US rows include country and exchange/listing context.
11. Document type is resolved or explicitly unknown with warning/error.
12. Source-use constraints are recorded.
13. Local use is allowed.
14. Staleness is evaluated or explicitly unknown.
15. Validation errors and warnings are recorded deterministically.
16. Synthetic fixtures are not marked as real usable coverage.

Validation failure behavior:

- `validation_status=failed` requires `usable_for_cached_source_dry_run=false`.
- `validation_status=not_validated` requires `usable_for_cached_source_dry_run=false`.
- Non-empty `validation_errors` require `usable_for_cached_source_dry_run=false`.
- Warnings may allow usage only when no blocking conditions are present.
- `staleness_status=unknown` does not automatically block use, but must remain visible.
- `staleness_status=stale` may block use according to later source-family policy.

## 10. Example Manifest

The following example is illustrative only. ME-SR08 did not fetch, stage, or validate this payload.

```json
{
  "manifest_format_version": "market-engine-cached-source-snapshot-acquisition-manifest-v1",
  "snapshot_id": "example-amd-sec-companyfacts-2025",
  "batch_id": "example-me-sr11-scripted-local-acquisition-20260701T100000Z",
  "created_at_utc": "2026-07-01T10:02:00Z",
  "acquired_at_utc": "2026-07-01T10:00:00Z",
  "acquisition_mode": "scripted_local_acquisition",
  "source_family": "sec_companyfacts",
  "source_name": "SEC CompanyFacts",
  "source_url": "https://www.example.invalid/sec/companyfacts/AMD.json",
  "source_license_note": "Illustrative example only; future sprint must record actual SEC source-use constraints.",
  "redistribution_allowed": false,
  "local_use_allowed": true,
  "commit_allowed": false,
  "source_material_type": "official_regulatory_source",
  "ticker": "AMD",
  "entity_name": "Advanced Micro Devices",
  "entity_country": "US",
  "entity_exchange": "NASDAQ",
  "source_entity_identifier": "0000002488",
  "cik": "0000002488",
  "requested_document_type": "companyfacts",
  "resolved_document_type": "companyfacts",
  "requested_period": "latest",
  "resolved_period": "latest",
  "source_publication_date": null,
  "source_retrieved_at_utc": "2026-07-01T10:00:00Z",
  "local_snapshot_path": "data/market_engine/cached_source_snapshots/example-me-sr11-scripted-local-acquisition-20260701T100000Z/AMD/example-amd-sec-companyfacts-2025/payload.json",
  "local_manifest_path": "data/market_engine/cached_source_snapshots/example-me-sr11-scripted-local-acquisition-20260701T100000Z/AMD/example-amd-sec-companyfacts-2025/manifest.json",
  "local_payload_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "local_payload_size_bytes": 234567,
  "payload_mime_type": "application/json",
  "payload_encoding": "utf-8",
  "normalization_status": "raw_only",
  "validation_status": "not_validated",
  "validation_errors": [],
  "validation_warnings": [
    "Illustrative example only; no payload was acquired by ME-SR08."
  ],
  "staleness_status": "unknown",
  "staleness_reason": "SEC CompanyFacts freshness policy must be evaluated by a future validator.",
  "usable_for_cached_source_dry_run": false,
  "blocked_reason": "illustrative_example_not_real_payload",
  "notes": "Contract example only. A real future manifest must not mark example data usable."
}
```

## 11. Batch Manifest Example

Batch manifests use this contract version:

```text
market-engine-cached-source-snapshot-acquisition-batch-manifest-v1
```

Example:

```json
{
  "batch_manifest_format_version": "market-engine-cached-source-snapshot-acquisition-batch-manifest-v1",
  "batch_id": "example-me-sr11-scripted-local-acquisition-20260701T100000Z",
  "created_at_utc": "2026-07-01T10:05:00Z",
  "acquisition_mode": "scripted_local_acquisition",
  "source_families": [
    "sec_companyfacts"
  ],
  "snapshot_manifest_references": [
    {
      "ticker": "AMD",
      "snapshot_id": "example-amd-sec-companyfacts-2025",
      "manifest_path": "data/market_engine/cached_source_snapshots/example-me-sr11-scripted-local-acquisition-20260701T100000Z/AMD/example-amd-sec-companyfacts-2025/manifest.json",
      "payload_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "usable_for_cached_source_dry_run": false
    }
  ],
  "batch_validation_summary": {
    "requested_snapshot_count": 1,
    "manifest_count": 1,
    "usable_snapshot_count": 0,
    "warning_snapshot_count": 1,
    "failed_snapshot_count": 0,
    "blocked_snapshot_count": 1
  },
  "batch_blocked_reasons": [
    "illustrative_example_not_real_payload"
  ],
  "notes": "Illustrative example only. ME-SR08 did not acquire this snapshot."
}
```

## 12. Compatibility With Existing Cached-Source Dry-Runs

This contract supports later cached-source dry-runs by creating an auditable source-acquisition boundary before runtime selection.

Future cached-source dry-runs may consume only snapshots whose manifests mark `usable_for_cached_source_dry_run=true`.

Operator-readable reports may surface:

- source family;
- source name;
- source URL when allowed;
- source publication date;
- retrieved timestamp;
- staleness status;
- validation status;
- local manifest reference;
- blocked reason.

Source Context may include source family, source URL, publication date, retrieved timestamp, staleness status, manifest reference, and payload hash as provenance.

Decision and recommendation stages remain downstream. They do not receive acquisition authority and must not reinterpret source-use permissions.

## 13. Failure Modes

| Failure mode | Required manifest behavior |
|---|---|
| Missing payload | `validation_status=failed`, `usable_for_cached_source_dry_run=false`, blocking reason recorded. |
| Zero-byte payload | Failed validation and not usable. |
| Missing publication date | Warning or failure depending on source family; staleness must be `unknown` when not evaluated. |
| Missing entity mapping | Failed validation and not usable. |
| Ambiguous entity mapping | Failed validation and not usable. |
| Source-use restrictions block use | Not usable, even if payload exists. |
| Unsupported source family | Not usable unless a later governance sprint approves the family. |
| Hash mismatch | Failed validation and not usable. |
| Stale source | Must record `staleness_status=stale`; usability depends on later source-family policy. |
| Local path outside allowed snapshot directory | Failed validation and not usable. |
| Duplicate snapshot id | Failed validation for affected snapshots and batch. |
| Conflicting ticker/entity metadata | Failed validation and not usable. |
| Synthetic fixture mislabeled as real data | Failed validation and governance violation. |

## 14. Follow-Up Implementation Candidates

ME-SR08 keeps the ME-SR07 sequence coherent and does not replace existing candidates.

| Sprint | Candidate scope |
|---|---|
| ME-SR09 — Implement missing expanded-universe snapshot coverage inventory command | Produce deterministic missing coverage inventory. It should reference ME-SR08 manifest requirements but must not acquire data. |
| ME-SR10 — Implement manual cached-source snapshot staging validator | Validate staged payloads and snapshot manifests against this contract. |
| ME-SR11 — Implement approved bounded acquisition or import workflow | Write snapshot manifests and batch manifests while preserving no-overwrite and source-use constraints. |
| ME-SR12 — Define non-US ticker source-family and source-mapping governance contract | Define non-US source-family and source-identity governance before broad non-US admission. |
| ME-RUN25 — Rerun expanded cached-source coverage audit after staged snapshots exist | Run existing cached-source path after validated snapshots exist. |

Implementation must not start before this contract is merged. Later implementation must not bypass validation or governance constraints.

## Verification

ME-SR08 is docs-only. Verification should confirm:

- no runtime code changed;
- no snapshots were acquired or staged;
- JSON examples are syntactically valid;
- backlog and roadmap include ME-SR08 and future candidates;
- no source payload content is embedded in committed docs.
