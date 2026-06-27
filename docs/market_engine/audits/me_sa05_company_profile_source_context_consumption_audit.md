# ME-SA05 - Company Profile Source Context Consumption Audit

Sprint ID: ME-SA05
Status: COMPLETED BY ME-SA05
Job family: ME-SA / Source Acquisition and Source Context
Date: 2026-06-27
Branch: `me-sa05-consume-company-profile-into-source-context`

## Objective

ME-SA05 implements deterministic local consumption of compatible `company_profile`
cached-source snapshots into the Source Context output family.

The implementation follows the ME-SA03 compatibility contract and retains the
ME-SA04 compatibility gate as the only authority that allows or blocks profile
consumption.

## Implementation

ME-SA05:

* replaces the temporary `blocked_company_profile_consumption_not_implemented`
  outcome for valid packages with `company_profile_consumption_allowed`;
* introduces `market-engine-company-profile-source-context-v1`;
* records identity, descriptive profile fields, payload version, provenance,
  source and retrieval timestamps, snapshot and manifest paths, gate outcome,
  and consumption state;
* exposes the profile context in dry-run artifact provenance;
* marks SEC CompanyFacts-only execution as `company_profile_absent_optional`;
* represents rejected profile packages as `company_profile_present_but_blocked`
  without exposing their profile body as trusted context;
* preserves all existing dry-run stage families;
* blocks Fundamental Observations for profile-only input because descriptive
  profile context does not provide financial evidence;
* leaves later review, handoff, and reporting stages not started.

## Consumption States

```text
absent_optional
blocked
consumed
```

Primary reason codes:

```text
company_profile_absent_optional
company_profile_consumption_allowed
company_profile_consumed_into_source_context
company_profile_present_but_blocked
company_profile_consumption_blocked_by_compatibility_gate
allowed_with_company_profile_stale_data_note
```

ME-SA04 blocker codes remain stable for malformed payloads, manifest mismatch,
ambiguous identity, network dependency, side-effect intent, and missing
manifest evidence. ME-SA05 adds an explicit non-consumable timestamp blocker
for malformed or stale timestamp states.

## Fail-Closed Behavior

Profile data is consumed only after the gate validates:

* payload format and source family;
* ticker binding;
* descriptive profile field allowlist;
* provenance identity and timestamp shape;
* local-only, no-provider-call request metadata;
* sibling manifest identity, validation, local-use, dry-run eligibility,
  freshness, payload hash, and payload size.

Blocked profile payloads expose only identity and compatibility metadata. Their
profile body is not copied into trusted Source Context.

Unknown optional source timestamps remain consumable only with an explicit stale
data marker. Explicitly stale manifest state remains blocked.

## Backward Compatibility

SEC CompanyFacts cached-source loading and all existing downstream builders are
unchanged. SEC Source Context receives only an additive optional-profile state.

The end-to-end dry-run contract continues to accept
`sec-companyfacts-source-context-v1` and now also accepts the explicit
`market-engine-company-profile-source-context-v1` Source Context version.

## Tests

Coverage includes:

* valid consumed profile context;
* absent optional profile on SEC CompanyFacts input;
* invalid payload version;
* missing and malformed payload;
* missing manifest;
* ticker mismatch;
* missing provenance;
* provider-call provenance;
* unsupported profile fields;
* stale and unknown source timestamp behavior;
* local artifact visibility;
* unchanged SEC CompanyFacts execution;
* full run-layer regression.

Validation:

```text
21 passed - tests/market_engine/run/test_me_run10_cached_source_local_execution.py
112 passed - tests/market_engine/run
505 passed - tests/market_engine
1172 passed - full pytest
```

## Non-Goals

ME-SA05 does not derive fundamental observations from company-profile data. It
does not add ratings, recommendations, target prices, ranking, urgency,
conviction, allocation, position sizing, trade actions, or Decision Engine
authority.

ME-SA05 adds no provider, network, yfinance, SEC/EDGAR, Telegram, broker,
portfolio, watchlist, production-write, or delivery side effect.

## Final Status

```text
PASS
```

Valid `company_profile` packages are now consumed as lower-authority descriptive
Source Context. Invalid packages remain fail-closed and profile-only input cannot
advance into financial or investment interpretation.

## Follow-Up

```text
ME-SA06 - Derive basic company_profile observations from Source Context
```

ME-SA06 may define descriptive, non-investment observations only. It must not
introduce recommendations, targets, ranking, urgency, conviction, allocation,
or Decision Engine authority.
