# RESET-10E0 — Current Telegram Output UX Review

## 1. Purpose

RESET-10E0 reviews the current legacy Telegram/reporting output before any v2 reporting or Telegram UX specification is written.

This review exists because the current Telegram output is not considered user-friendly enough to translate directly into v2. The goal is to identify what should be preserved, what should be redesigned, and what must not be copied into the new v2 reporting contract.

This is analysis and UX review only. It does not modify code, tests, data, reports, generated outputs, or workflows.

## 2. Scope

Reviewed surfaces:

- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `tests/reporting/test_build_reporting_layer.py`
- `tests/reporting/test_build_telegram_summary.py`
- `docs/archive/sprints/operational/operational_sprint_3_telegram_ux.md`
- attempted review of `reports/daily/telegram_message.txt`

No runtime was executed.
No Telegram message was sent.
No report artifact was generated.

## 3. Current Output Availability

`reports/daily/telegram_message.txt` was not found on `main` during GitHub inspection.

That means this review could not inspect a currently committed real Telegram artifact. Instead, it reviews the current message construction logic and tests that define the expected legacy Telegram text shape.

This is sufficient for a pre-spec UX review, but a future implementation should still require a before/after sample generated locally by Codex against synthetic or approved fixture data.

## 4. Current Message Construction

The current Telegram text is built by the legacy Reporting Layer in `build_telegram_message()`.

The current message starts with technical metadata:

```text
Daily Reporting Summary
Reporting contract: REPORTING_CONTRACT_V1
Source artifact: data/processed/final_decisions.csv
Dashboard artifact: data/processed/reporting_dashboard_data.csv
Source row count: ...
Represented row count: ...
omitted_row_count: 0
Input status: ...
Stability status: ...
```

Then it groups output by `source_final_action` and prints sections such as:

```text
Decision output: HOLD
Group count: 1
- AAA: action=HOLD; allocation=SOURCE_HOLD; execution=SOURCE_NONE
```

Finally, it ends with traceability metadata:

```text
Traceability
Grouping rule: GROUP_BY_SOURCE_FINAL_ACTION_THEN_SOURCE_ORDER
Truncation rule: TELEGRAM_GROUP_SUMMARY_WITH_SOURCE_ORDER_EXAMPLES
Ordering rule: SOURCE_ORDER_WITH_FIXED_SECTION_ORDER
All source rows are represented in the dashboard artifact.
```

## 5. What Works Today

The current design has important governance strengths:

- it is deterministic;
- it preserves row count visibility;
- it preserves source artifact references;
- it groups by Decision Engine output rather than inventing a new ranking;
- it keeps omitted row count explicit;
- it validates forbidden human terms;
- it avoids non-ASCII text;
- it avoids hidden source mutation;
- it is tested for wrapper delegation to the Reporting Layer;
- it protects traceability through `source_row_identity` and row-count preservation.

These qualities should be preserved in v2.

## 6. Main UX Problems

### 6.1 The message starts with implementation details

The first lines focus on reporting contract names, artifact paths, and row counts. This is useful for auditability but not the first thing the operator needs on a phone.

The operator first needs orientation:

- Did the run complete?
- Is there anything to review?
- Why are outputs review-only or limited?
- Are data problems blocking useful interpretation?
- Is anything portfolio-relevant?

### 6.2 The wording is too technical

Current labels such as these are governance-safe but not user-friendly:

- `Reporting contract: REPORTING_CONTRACT_V1`
- `Source artifact: data/processed/final_decisions.csv`
- `Dashboard artifact: data/processed/reporting_dashboard_data.csv`
- `omitted_row_count: 0`
- `Decision output: HOLD`
- `Grouping rule: GROUP_BY_SOURCE_FINAL_ACTION_THEN_SOURCE_ORDER`
- `Truncation rule: TELEGRAM_GROUP_SUMMARY_WITH_SOURCE_ORDER_EXAMPLES`

These are useful for logs, not for the top-level Telegram summary.

### 6.3 Row examples are not meaningful enough

The current row display text is:

```text
AAA: action=HOLD; allocation=SOURCE_HOLD; execution=SOURCE_NONE
```

This is technically faithful, but it does not explain why the ticker matters, what blocks interpretation, whether it is portfolio-related, or what the user should inspect next.

### 6.4 The message lacks a clear hierarchy

The current structure is a technical list. It does not clearly separate:

- run status;
- decision summary;
- review blockers;
- portfolio relevance;
- data quality warnings;
- examples;
- traceability details.

### 6.5 Audit details compete with useful summary

Traceability is essential, but it should not dominate the phone message. Detailed artifact paths and grouping rules should be moved to a short footer or a linked/full report reference.

### 6.6 Telegram has no operator-focused conclusion

The current output does not clearly answer:

- “What should I look at first?”
- “Why is everything review-only?”
- “Is the scan useful today?”
- “What data is missing?”
- “Are my holdings affected?”

Any future answer must remain communication-only and must not create a decision or recommendation.

## 7. What Must Not Be Copied Into V2 As-Is

Do not copy these current UX patterns directly into v2:

- contract/version labels as the headline;
- long artifact paths near the top;
- raw technical field labels such as `omitted_row_count`;
- row text limited to action/allocation/execution codes;
- traceability rules as the main closing section;
- grouping by final action without a user-readable explanation;
- Telegram text that reads like a debug artifact rather than a daily review message.

## 8. What Should Be Preserved

The following legacy reporting strengths should be preserved:

- Telegram/reporting communicates only;
- Decision Engine remains the only final-action authority;
- source row count and represented row count remain visible somewhere;
- no hidden omission;
- no hidden ranking;
- no urgency, conviction, or tradeability language;
- source artifact traceability remains available;
- deterministic ordering is maintained;
- truncation/summary behavior must be explicit;
- full-detail artifacts remain inspectable outside Telegram.

## 9. Recommended V2 Telegram UX Direction

The v2 Telegram message should be designed as a scan-friendly operator summary, not a raw technical dump.

Recommended top-level structure:

```text
Market Scanner — Daily Review
Run status: completed / review required / blocked
Summary: X rows checked, Y represented, Z review blockers

Main status
- Most outputs require REVIEW because ...
- Source-data status: ...
- Portfolio relevance: ...

Needs attention
- TICKER — short reason, source state, portfolio relevance if available
- TICKER — short reason, source state, portfolio relevance if available

Data warnings
- Fundamentals source missing for ...
- Missing/partial/stale data count ...

Details
Full dashboard: ...
Source artifact: ...
Reporting is communication-only; decisions come from the Decision Engine.
```

This is only a UX direction, not an implementation contract yet.

## 10. Required UX Questions Before Specification

Before RESET-10F writes the actual Telegram/reporting specification, the following questions should be answered:

1. Should Telegram show only a short summary, or also top examples?
2. How many ticker examples should fit in one Telegram message?
3. Should portfolio holdings be shown before non-portfolio opportunities?
4. How should “all REVIEW because source data is missing” be communicated?
5. Should data warnings be above or below ticker examples?
6. Which artifact references are useful on the phone?
7. Should the message include a short “not investment advice / communication-only” footer, or is that too noisy?
8. Should Telegram be English-only like current reporting, or can operator-facing messages be Dutch later?
9. Should the full report and Telegram message have different levels of detail?
10. How should truncation be worded so it is clear but not alarming?

## 11. Preliminary UX Principles

Future Telegram UX should follow these principles:

- Start with operator orientation, not implementation metadata.
- Use short, plain-language labels.
- Preserve auditability without overwhelming the first screen.
- Show the reason something needs review.
- Make missing data visible.
- Separate portfolio relevance from generic opportunity review.
- Keep detailed traceability available but secondary.
- Never introduce urgency, ranking, conviction, tradeability, or allocation advice.
- Never hide row representation/truncation behavior.
- Keep Telegram shorter than the full report.

## 12. Governance Assessment

A future Telegram/reporting UX implementation will likely be Level 2 if it changes:

- message structure;
- grouping;
- truncation wording;
- ordering;
- row representation text;
- artifact reference conventions;
- reporting contract fields or tests.

It must not become Level 3. Level 3 risks would include:

- new decision semantics outside the Decision Engine;
- hidden prioritization;
- urgency or conviction wording;
- tradeability or allocation guidance;
- report-side filtering that changes meaning;
- Telegram acting as a decision authority.

## 13. Recommendation

Do not proceed with direct translation of legacy Telegram tests into v2 yet.

Recommended next step:

```text
RESET-10F — User-Friendly Reporting and Telegram Output Specification
```

RESET-10F should use this review to define the desired v2 Telegram/reporting UX before Codex creates v2 reporting contracts and tests.

## 14. Scope Confirmation

RESET-10E0 is documentation-only.

No files under these paths were intentionally modified:

- `scripts/`
- `src/`
- `tests/`
- `data/`
- `reports/`
- `.github/workflows/`

No runtime was executed.
No Telegram message was sent.
No report artifact was generated.
No production pipeline, SEC/provider/network/broker/Telegram/live data call was made.
