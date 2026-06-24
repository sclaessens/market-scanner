# ME-UNI09 Review Follow-up Candidates

Owner role: Scrum Master / Product Owner / Technical Architect / Governance Auditor

Job family: ME-RM - Roadmap / Governance

Status: DEFERRED / NON-BLOCKING

## Context

ME-UNI09 implemented `market-engine-professional-swing-universe-expansion-v1`, a deterministic, non-actionable Professional Swing Universe expansion builder from `market-engine-candidate-classification-v1`.

The ME-UNI09 review considered the implementation merge-ready. The items below are review attention points only. They are not blockers for ME-UNI09 and must not be inserted ahead of ME-SR06, ME-RUN23, or the next expanded-universe execution path unless later execution reveals a real blocker.

## Deferred Hardening Candidates

### 1. Replace reason-string grouping with explicit decision category metadata

ME-UNI09 currently derives `blocked_or_manual_review_entries` partly from controlled reason-string matching, for example by grouping entries where `requires_manual_review` is true or where the reason contains controlled tokens such as `blocked` or `manual_review`.

This is acceptable for ME-UNI09 because the reason values are internally controlled. A future hardening sprint may introduce explicit decision category metadata so audit grouping becomes schema-explicit rather than reason-string-based.

Potential future sprint candidate:

```text
ME-UNI09H - Harden Professional Swing Universe expansion audit semantics
```

### 2. Clarify duplicate semantics in Professional Swing Universe expansion results

ME-UNI09 detects duplicate candidate input by ticker/market. Because candidate keys are recorded at the candidate-input level, a first ineligible candidate for a ticker/market can cause a later candidate with the same ticker/market to be classified as duplicate even if the first candidate was excluded.

This is acceptable as input-level duplicate detection and is useful for auditability. A future refinement may split duplicate categories into more explicit states, such as:

* `duplicate_in_candidate_input`;
* `duplicate_after_eligible_inclusion`;
* `already_present_in_universe`.

This must remain a hardening refinement only unless a concrete expanded-universe run shows that the current behavior causes operator confusion or downstream ambiguity.

### 3. Consider extracting an in-memory Professional Swing Universe entry validator

ME-UNI09 validates proposed universe entries by writing a temporary CSV and reloading through the existing Professional Swing Universe loader/validator.

This is robust because it reuses the canonical validator. It is slightly heavy but acceptable for the current pure builder. If this validation pattern appears repeatedly in later universe tooling, a future sprint may extract a shared in-memory Professional Swing Universe entry validator to avoid temporary-file roundtrips while preserving identical validation semantics.

## Planning Rule

These follow-ups are deferred hardening candidates. They do not authorize runtime changes, tests, provider calls, source refresh, generated artifacts, portfolio/watchlist mutation, Decision Engine behavior, delivery/reporting behavior, or trading semantics.

The active planning direction remains expanded-universe source-support classification and execution before additional polish or hardening work, unless a concrete blocker is discovered.
