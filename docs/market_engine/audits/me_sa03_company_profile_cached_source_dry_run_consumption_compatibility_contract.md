# ME-SA03 — Company Profile Cached-Source Dry-Run Consumption Compatibility Contract

Sprint ID: ME-SA03  
Status: COMPLETED BY ME-SA03  
Job family: ME-SA / Source Acquisition  
Date: 2026-06-26

## 1. Purpose

ME-SA03 defines the compatibility contract for consuming `company_profile` cached-source snapshots through local `cached_source_snapshot` dry-run flows.

ME-RUN26 proved that ME-SA02 `company_profile` packages pass acquisition and staging validation, but the current dry-run path still expects SEC CompanyFacts metadata and blocks with `SEC CompanyFacts snapshot metadata is missing`.

ME-SA03 is documentation-only. It does not implement provider access, acquisition behavior, staging/import runtime behavior, dry-run execution changes, downstream analysis behavior, or delivery behavior.

## 2. Scope

In scope:

- `company_profile` cached-source snapshot compatibility with local dry-run consumption;
- source-family identity and authority boundaries;
- snapshot payload expectations;
- manifest compatibility expectations;
- staging/import validation gate semantics;
- dry-run eligibility rules;
- required dry-run output behavior;
- compatibility matrix;
- failure states;
- auditability requirements;
- future implementation acceptance criteria.

Out of scope:

- live fetching;
- provider implementation;
- production writes;
- messaging delivery;
- portfolio or watchlist mutation;
- Decision Engine authority changes;
- Recommendation Review authority changes;
- runtime code changes in ME-SA03;
- test-code changes in ME-SA03.

## 3. Source family identity

Canonical source family:

```text
company_profile
```

Expected role:

```text
descriptive company metadata / profile context
```

`company_profile` may provide descriptive context such as company identity, name, sector, industry, exchange, country, description, website, currency, or similar profile-level metadata when present in an approved cached-source payload.

`company_profile` must remain contextual. It must not receive downstream decision authority.

Downstream use must remain supporting. It must not override source-readiness evidence, SEC CompanyFacts evidence, numeric evidence, missing-data evidence, stale-data evidence, analysis review, recommendation review, portfolio review, decision handoff, or delivery/reporting state.

## 4. Cached-source snapshot expectations

A `company_profile` cached-source snapshot is compatible only when the package can prove all of the following without network access:

- explicit source-family identity: `company_profile`;
- explicit ticker or symbol identity;
- provider/source provenance when available;
- generated, observed, retrieved, or acquired timestamp semantics, or an explicit audit note that timing is unknown;
- payload presence;
- payload shape compatible with the approved `company_profile` payload contract;
- deterministic local readability;
- stable local path references;
- no hidden live fallback;
- no side-effect intent.

Freshness may be `fresh`, `stale`, or `unknown` only when that state remains explicit in manifest, validation, dry-run summary, and audit output.

Missing payload fields must remain missing-data notes. Unknown or stale timestamps must remain stale-data or audit notes.

## 5. Manifest compatibility requirements

A compatible manifest must be able to prove at minimum:

- artifact or manifest format/version;
- input mode compatibility with `cached_source_snapshot`;
- ticker or symbol identity;
- source family includes exactly `company_profile` for the profile payload;
- payload location/reference;
- payload hash and size when available under the existing manifest contract;
- validation status;
- missing-data notes;
- stale-data notes;
- source provenance and audit notes;
- dry-run eligibility status;
- side-effect safety status.

ME-SA03 does not replace existing cached-source acquisition manifest contracts. It narrows how a `company_profile` package may be consumed after those contracts and the existing staging validator have accepted the package.

If existing manifest rules are stricter than ME-SA03, the stricter existing rules win. If ME-SA03 is stricter about `company_profile` semantic consumption, ME-SA03 wins for future `company_profile` dry-run compatibility work.

## 6. Staging/import validation contract

Staging/import compatibility must fail closed.

Blocking conditions:

- unknown source family;
- missing `company_profile` payload when `company_profile` is listed for consumption;
- malformed `company_profile` payload;
- ambiguous ticker identity;
- ticker mismatch between manifest and payload;
- inconsistent source-family identity between manifest and payload;
- payload path outside allowed local roots;
- missing or mismatched payload hash when hash is required by the active manifest contract;
- network dependency during validation or consumption;
- hidden provider fallback;
- live data fallback;
- side-effect intent.

Conditionally allowed states:

- missing provenance may be allowed only when explicitly represented as an audit note and not required by a stricter existing contract;
- unknown timestamp may be allowed only when explicitly represented as stale-data or audit evidence;
- stale timestamp may be allowed only when downstream output preserves stale-data evidence and the relevant stage contract permits stale contextual data.

A successful structural staging result does not automatically mean semantic dry-run consumption is allowed.

## 7. Dry-run consumption compatibility

A local dry-run may consume `company_profile` only when all conditions are true:

- input mode is `cached_source_snapshot`;
- staging/import validation has completed with a compatible result;
- source family is explicitly `company_profile`;
- payload and manifest identify the same ticker;
- consumption is deterministic and local-only;
- no network, provider, or delivery side effect is attempted;
- downstream stages treat the source family as contextual company profile input;
- missing fields are represented as missing-data notes, not fabricated values;
- stale or unknown freshness is represented in stale-data notes or audit notes;
- no live fallback is used;
- no downstream authority escalation occurs.

`company_profile` must not be coerced into SEC CompanyFacts Source Context. If an implementation needs a new profile-context object, it must define that object explicitly and preserve its lower authority relative to financial/fundamental source evidence.

## 8. Required dry-run output behavior

Future dry-run output must make `company_profile` state visible.

Required output semantics:

- dry-run summary indicates whether `company_profile` was present, absent, consumed, ignored, or blocked;
- missing company profile data remains visible as missing-data notes;
- stale or unknown company profile data remains visible as stale-data or audit notes;
- blocked `company_profile` consumption produces explicit blocked state/reason;
- consumed `company_profile` does not force downstream review or handoff decisions;
- existing SEC CompanyFacts and fundamental observation behavior must not regress;
- downstream stages must not treat profile metadata as numeric financial evidence;
- downstream authority must not escalate.

## 9. Compatibility matrix

| Scenario | Expected compatibility result | Required note/state |
|---|---|---|
| Valid `company_profile` snapshot plus matching manifest | Compatible for contextual dry-run consumption after validation | `company_profile_consumption_allowed` or equivalent success state |
| `company_profile` listed in manifest but payload missing | Blocked | `blocked_missing_company_profile_payload` |
| Payload present but manifest lacks `company_profile` | Blocked | `blocked_company_profile_manifest_mismatch` |
| Ticker mismatch between manifest and payload | Blocked | `blocked_ambiguous_company_profile_identity` or stricter ticker-mismatch state |
| Unknown freshness timestamp | Conditionally compatible only as contextual data | `allowed_with_company_profile_stale_data_note` or explicit unknown-freshness audit note |
| Stale timestamp | Conditionally compatible only if stale contextual data is allowed | `allowed_with_company_profile_stale_data_note` |
| Malformed payload shape | Blocked | `blocked_malformed_company_profile_payload` |
| Snapshot requires network fallback | Blocked | `blocked_company_profile_network_dependency` |
| Snapshot attempts side-effect or delivery action | Blocked | `blocked_company_profile_side_effect_intent` |
| Source family unknown or misspelled | Blocked | `blocked_company_profile_manifest_mismatch` or existing unknown-source-family state |

## 10. Failure states

Proposed canonical failure and allowed-with-note states for implementation:

```text
blocked_missing_company_profile_payload
blocked_malformed_company_profile_payload
blocked_company_profile_manifest_mismatch
blocked_ambiguous_company_profile_identity
blocked_company_profile_network_dependency
blocked_company_profile_side_effect_intent
allowed_with_company_profile_missing_data_note
allowed_with_company_profile_stale_data_note
```

If existing runtime failure-state naming conventions are stricter or already canonical, a future implementation may map these proposed states onto the existing naming style. The semantic distinction must remain intact.

## 11. Auditability requirements

Every future implementation must make the following auditable:

- ticker;
- source family;
- payload path/reference;
- manifest path/reference;
- payload hash and size when available;
- validator version or validation contract version;
- compatibility result;
- blocking reason when blocked;
- missing-data notes;
- stale-data notes;
- provenance notes;
- whether dry-run consumption was allowed or blocked;
- confirmation that no provider/network/live fallback occurred;
- confirmation that no production or delivery side effect occurred.

The audit trail must distinguish structural invalidity, structural validity without semantic dry-run compatibility, and full contextual `company_profile` compatibility.

## 12. Future implementation acceptance criteria

A later implementation sprint must include deterministic local tests for:

- valid compatible `company_profile` snapshot consumption;
- manifest/payload source-family mismatch;
- missing payload;
- malformed payload;
- ticker mismatch;
- stale timestamp handling;
- unknown timestamp handling;
- no-network/no-live-fallback enforcement;
- side-effect blocking;
- missing profile fields preserved as missing-data notes;
- stale or unknown profile data preserved as stale-data or audit notes;
- dry-run summary visibility for consumed/blocked/absent `company_profile` state;
- no regression to existing SEC CompanyFacts cached-source dry-run behavior;
- no downstream authority escalation.

The implementation must not silently reinterpret `company_profile` as `sec_companyfacts`. Any new internal profile-context payload must receive an explicit format identifier and must remain lower-authority contextual evidence.

## 13. Validation

ME-SA03 is docs-only.

Runtime tests were not run by ChatGPT because this GitHub-only execution changed no runtime or test files and does not have access to the user's local `.venv` test environment.

Required local post-merge validation for the operator:

```text
git diff --check
```

Optional local validation after pulling the PR:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
```

## 14. ME-SA03 outcome

```text
Contract defined.
No runtime changes.
No test-code changes.
No production side effects.
Ready for implementation sprint.
```

Follow-up candidate:

```text
ME-SA04 — Implement company_profile cached-source dry-run consumption compatibility gate
```

ME-SA04 may modify runtime code and tests only after ME-SA03 is accepted. It must preserve existing cached-source dry-run behavior while adding explicit `company_profile` compatibility gating.