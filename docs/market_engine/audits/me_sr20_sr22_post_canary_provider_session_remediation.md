# ME-SR20 through ME-SR22 Post-Canary Remediation Audit

Status: `implementation_complete_pending_review_with_data_gaps`

## Source Evidence and Fixed Boundary

Canonical Price Refresh run `30149436386` reconciles exactly to 44
`PROVIDER_OHLC_INVALID` and 26 `EXPECTED_SESSION_NOT_AVAILABLE` results. Its
manifest contains no rejected raw bar, violated relationship, deviation,
provider batch, or attempt evidence. A later response cannot reconstruct
those missing historical facts.

Research remained fixed to the completed 2026-07-24 US session. A bounded
Yahoo Finance re-fetch on 2026-07-27 returned July 23 and July 24 observations
for 68 of the 70 scoped instruments. NSA and TMHC remained empty. Research
did not mutate a canonical CSV.

## ME-SR20 — Provider OHLC Diagnostic Capture and Controlled Revalidation

Rejected bars now record ticker, session, provider, fetch time, raw OHLCV,
canonicalized OHLCV, numeric representation, each exact violated
relationship, absolute and relative deviation, retry number, and final reason
code. No unfiltered payload, header, cookie, credential, or token is retained.
No epsilon, rounding exception, or provider-specific tolerance was added.

All 44 later responses validated under the unchanged OHLC contract. Because
the original rejected values were not captured, every historical cause is
classified `INSUFFICIENT_EVIDENCE_ORIGINAL_REJECTED_BAR`; the later valid bar
supports controlled revalidation but does not prove a precision defect.

```json
{
  "original_reason": "PROVIDER_OHLC_INVALID",
  "original_count": 44,
  "original_last_date": "2026-07-23",
  "expected_session": "2026-07-24",
  "original_raw_bar": null,
  "original_violated_relationship": null,
  "refetch_result": "VALID_2026-07-24",
  "classification": "INSUFFICIENT_EVIDENCE_ORIGINAL_REJECTED_BAR",
  "runtime_retry_path_verified": true,
  "valid_bar_observed_in_bounded_dry_run": true,
  "canonical_history_updated": false,
  "historical_root_cause_proven": false,
  "still_blocked": true,
  "tickers": [
    "ABT", "ACM", "ADC", "ALL", "ATR", "AVNT", "BBY", "BJ", "BRO",
    "CLS", "CSL", "CUBE", "DAR", "DCI", "DOCN", "DTE", "EFX", "EGP",
    "ELS", "EPR", "GDDY", "GPC", "H", "HLT", "L", "LII", "LVS", "MS",
    "NVR", "ORI", "PAG", "PATH", "PKG", "RPM", "RVTY", "SSD", "TDG",
    "UDR", "UGI", "VICI", "WBS", "WLK", "WSM", "WST"
  ]
}
```

## ME-SR21 — Empty Provider-Batch Completeness and Retry Remediation

The normal provider path now classifies complete, fully empty, and partially
empty batches; retries each failing batch at most the configured limit;
splits unresolved symbols deterministically; and uses the existing
single-symbol adapter as the final bounded fallback. Returned frames still
traverse the unchanged schema, timestamp, numeric, OHLC, calendar, merge,
history-conflict, and freshness validation.

The manifest records request mode, exact batch membership, attempt, split
depth, classification, and received bar count. Persistent missingness remains
`EXPECTED_SESSION_NOT_AVAILABLE` and prevents partial publication.

```json
{
  "original_reason": "EXPECTED_SESSION_NOT_AVAILABLE",
  "expected_session": "2026-07-24",
  "classification": "TRANSIENT_EMPTY_PROVIDER_BATCH",
  "refetch_result": "VALID_2026-07-24",
  "added_sessions": [],
  "runtime_retry_path_verified": true,
  "valid_bar_observed_in_bounded_dry_run": true,
  "canonical_history_updated": false,
  "historical_root_cause_proven": false,
  "still_blocked": true,
  "tickers": [
    "MZTI", "NBIX", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NET",
    "NEU", "NFG", "NFLX", "NI", "NJR", "NKE", "NLY", "NNN", "NOC",
    "NOV", "NOVT", "NOW", "NRG", "NSC", "NTAP", "NTNX"
  ]
}
```

NSA had ordinary expected US sessions on July 22, 23, and 24. The approved
provider returned none in either the bounded batch or direct single-symbol
route, and no corporate-action evidence justified a lifecycle exception.

```json
{
  "ticker": "NSA",
  "original_reason": "EXPECTED_SESSION_NOT_AVAILABLE",
  "original_last_date": "2026-07-21",
  "expected_sessions": ["2026-07-22", "2026-07-23", "2026-07-24"],
  "classification": "PERSISTENT_PROVIDER_DATA_GAP",
  "refetch_result": "EMPTY",
  "added_sessions": [],
  "lifecycle_change": null,
  "end_status": "BLOCKED",
  "remaining_reason": "EXPECTED_SESSION_NOT_AVAILABLE"
}
```

No carry-forward, synthetic observation, or hand-authored value was created.

## ME-SR22 — TMHC Lifecycle and Final-Session Remediation

TMHC's July 24 Form 8-K proves that the acquisition completed on July 24,
public shares converted to USD 72.50 cash rights, and NYSE trading was to be
suspended only after the July 24 close. It separately states that Form 25
delisting becomes effective August 3. The generic lifecycle record therefore
uses July 24 as the final regular-way session and July 25 as the first inactive
date.

```json
{
  "ticker": "TMHC",
  "original_reason": "EXPECTED_SESSION_NOT_AVAILABLE",
  "original_last_date": "2026-07-23",
  "expected_session": "2026-07-24",
  "classification": "PROVEN_FINAL_SESSION_PROVIDER_BAR_UNAVAILABLE",
  "refetch_result": "EMPTY",
  "added_sessions": [],
  "lifecycle_change": {
    "delisting_end_date": "2026-07-24",
    "status_effective_date": "2026-07-25"
  },
  "end_status": "BLOCKED",
  "remaining_reason": "EXPECTED_SESSION_NOT_AVAILABLE"
}
```

TMHC remains provider-eligible through its final session and is excluded
afterward. Because Yahoo no longer reproduces its July 24 bar, no canonical
row or `market-data` pull request is justified. The retained history remains
fail-closed until a provider-derived, fully validated final bar is available.

## Governance and Dependency

The implementation adds no ticker-specific production branch, broad
tolerance, fail-open behavior, partial publication, synthetic OHLCV,
unbounded retry, bypass flag, or secret-bearing payload. Source, tests,
lifecycle configuration, and governance documentation belong on `main`.
There is no data branch in this remediation. Any later TMHC or NSA data-only
pull request must be supported by independently reproducible provider
evidence and must follow the reviewed runtime/lifecycle pull request.

## PR 472 Review-Blocker Follow-Up

The first draft captured an invalid batch bar but immediately propagated
`PROVIDER_OHLC_INVALID`; it did not perform the promised isolated
single-ticker revalidation. It also retained response and error state across
batch attempts and recursive ticker groups.

The follow-up isolates both boundaries:

- `original_invalid_bar` records the batch rejection before any re-fetch;
- each `single_ticker_revalidation` diagnostic is a bounded
  single-ticker re-fetch attempt;
- terminal outcomes are `single_ticker_refetch_valid`,
  `single_ticker_refetch_invalid`, `single_ticker_refetch_missing`, or
  `single_ticker_refetch_provider_failure`;
- the new frame traverses the complete unchanged provider-frame and
  downstream canonical-history contracts;
- every batch attempt creates a new response map and exception value;
- successful ticker frames become terminal and leave later retry requests;
- unresolved tickers recurse independently and each ticker owns its terminal
  provider failure.

The 24 batch tickers with a valid bar observed during bounded research are
not durably resolved: the runtime retry path and dry-run observation are
verified, but canonical history was not updated and the historical root cause
was not proven. NSA and TMHC remain blocked. No canonical data or publication
workflow is part of this follow-up.
