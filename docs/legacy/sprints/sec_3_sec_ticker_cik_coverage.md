# SEC-3 — SEC Ticker/CIK Index and Coverage Report

Status: IMPLEMENTED
Backlog context: BL-0015 / BL-0017
Date: 2026-05-31

## Implemented Scope

SEC-3 implemented a narrow standalone SEC ticker-to-CIK index and project ticker coverage utility.

Implemented module:

```text
scripts/fundamentals/sec_ticker_cik_index.py
```

Implemented capabilities:

- read a local SEC-supported ticker/CIK source file;
- normalize project and SEC tickers;
- normalize CIK values to the SEC 10-digit padded form;
- build a descriptive SEC ticker/CIK index;
- build one coverage row per requested project ticker;
- preserve missing mappings in output;
- detect duplicate or conflicting ticker mappings as ambiguous;
- optionally write a generated coverage CSV only to an explicit output path;
- provide a small CLI for validation-only or coverage generation from local files.

SEC-3 does not call SEC endpoints, download SEC data, read SEC Company Facts files, extract XBRL facts, transform SEC data into internal fundamentals history, or integrate with the pipeline.

## Module Path

```text
scripts/fundamentals/sec_ticker_cik_index.py
```

The module is standalone and does not import or call:

- fundamental metrics builders;
- fundamental quality builders;
- fundamental analysis builders;
- pipeline orchestration;
- Decision Engine;
- Reporting;
- Telegram;
- portfolio logic.

## Input Expectations

Ticker/CIK source input is local only.

Supported local source shapes:

- SEC-style JSON object keyed by row number, where each row contains fields such as `ticker`, `cik_str`, and `title`;
- JSON list of row objects;
- JSON object with a `rows` list;
- CSV with ticker and CIK-like fields.

Project ticker input may be:

- CSV with a `ticker` column;
- JSON list;
- JSON object with `tickers` or `rows`;
- plain text with one ticker per line.

Tests use only synthetic local fixture data and pytest temporary paths.

## Mapping Model

The index model uses:

```text
ticker
cik
cik_padded
company_name
exchange
mapping_status
mapping_reason
source_reference
```

The coverage model uses:

```text
ticker
cik
cik_padded
company_name
mapping_status
mapping_reason
source_reference
review_required
```

CIK normalization rules:

- numeric or string CIK inputs are accepted;
- normalized SEC path CIKs use 10 digits with leading zeros;
- blank, non-numeric, zero, negative, or longer-than-10-digit values fail direct normalization;
- invalid source rows are marked `CIK_REVIEW_REQUIRED` in index construction;
- missing project ticker mappings are preserved as `CIK_MISSING`;
- duplicate matching ticker rows are preserved as `CIK_AMBIGUOUS`.

## Mapping Statuses

Descriptive statuses:

```text
CIK_MATCHED
CIK_MISSING
CIK_AMBIGUOUS
CIK_REVIEW_REQUIRED
CIK_NOT_SEC_REPORTER
```

These statuses are source-data readiness states only.

They do not imply:

- eligibility;
- ranking;
- scoring;
- urgency;
- conviction;
- tradeability;
- buy/sell;
- allocation;
- final action;
- hidden filtering.

## Coverage Report Behavior

Coverage output is row-preserving relative to the requested project ticker list.

Rules:

- one output row per requested ticker;
- ticker identity is normalized but not dropped;
- missing mappings remain represented;
- ambiguous mappings remain represented;
- coverage output does not determine source eligibility or pipeline inclusion;
- coverage output does not transform SEC facts into fundamentals history;
- coverage output does not generate `fundamental_metrics.csv`, `fundamental_quality.csv`, or `fundamental_analysis.csv`.

## Generated Output Policy

Real generated coverage reports should remain local/generated and should not be committed.

The module writes a coverage CSV only when an explicit `output_path` or CLI `--output` is supplied.

SEC-3 committed no generated operational coverage report, no SEC data, no downloaded SEC files, no extracted SEC files, no generated CSV/data/log/report files, and no real market data.

## Tests Added

Focused tests:

```text
tests/fundamentals/test_sec_ticker_cik_index.py
```

Tests cover:

- CIK integer normalization to 10 digits;
- CIK string normalization to 10 digits;
- invalid CIK handling;
- ticker normalization, including lowercase input;
- exact ticker match;
- missing ticker coverage as `CIK_MISSING`;
- duplicate/conflicting ticker mapping as `CIK_AMBIGUOUS`;
- one output row per requested ticker;
- preservation of missing mappings;
- generated coverage report writes only to a provided temporary path;
- no network call required on import;
- no SEC download helper exposure;
- no pipeline integration exposure.

Tests are fixture/temp-dir based. They do not call live SEC endpoints, require internet, download real SEC data, write to real `data/local/`, commit generated artifacts, or depend on current market data.

## No Runtime Downstream Change Confirmation

SEC-3 introduced no runtime downstream behavior changes.

It did not modify:

- Decision Engine logic;
- Reporting semantics;
- Telegram delivery or formatting;
- portfolio behavior;
- scanner behavior;
- validation/context/timing/portfolio intelligence behavior;
- fundamental metrics behavior;
- fundamental quality behavior;
- fundamental analysis behavior;
- full pipeline orchestration.

The new module is standalone and does not feed data into:

- `data/raw/fundamentals_history.csv`
- `data/processed/fundamental_metrics.csv`
- `data/processed/fundamental_quality.csv`
- `data/processed/fundamental_analysis.csv`

## Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

SEC-3 remains within the approved SEC source-data sequence. Future provider/API automation remains governed by BL-0017. Internal fundamentals transformation remains a later sprint.

## Recommended Next Sprint

Recommended next sprint:

```text
SEC-4 — SEC XBRL Mapping Investigation
```

Purpose:

- map SEC XBRL tags to internal fundamentals fields;
- identify primary and alternate tags;
- identify missing or inconsistent tags;
- determine which fields are reliable enough for internal raw history transformation.

SEC-4 should not implement SEC-to-internal transformation into `fundamentals_history.csv`; that remains later sprint scope.
