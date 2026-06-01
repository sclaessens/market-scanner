# SEC-2 — SEC Bulk Intake Implementation

Status: IMPLEMENTED
Backlog context: BL-0015 / BL-0017
Date: 2026-05-31

## Implemented Scope

SEC-2 implemented a narrow controlled local SEC EDGAR Company Facts bulk intake/cache utility.

Implemented module:

```text
scripts/fundamentals/sec_companyfacts_bulk_intake.py
```

The module supports:

- official SEC Company Facts bulk URL validation;
- HTTPS-only official SEC host allowlisting;
- explicit User-Agent requirement for download calls;
- safe local cache directory creation;
- controlled download to a local ignored cache path when explicitly invoked;
- local ZIP validation without network access;
- local manifest generation for validated or downloaded ZIP files;
- clear failure behavior for invalid URLs, unsupported hosts, missing User-Agent values, invalid ZIP files, and unsafe cache paths.

SEC-2 did not implement CIK/ticker coverage reporting, XBRL tag mapping, SEC-to-internal fundamentals transformation, pipeline integration, metrics changes, quality changes, analysis changes, Decision Engine changes, Reporting changes, Telegram changes, portfolio changes, or ticker-category runtime logic.

## Official Source References

SEC-2 preserves the approved official source direction:

- SEC EDGAR API documentation: `https://www.sec.gov/search-filings/edgar-application-programming-interfaces`
- SEC Company Facts bulk ZIP: `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip`
- SEC webmaster and developer FAQ: `https://www.sec.gov/about/webmaster-frequently-asked-questions`

Important technical facts preserved:

- Company Facts API shape: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`
- CIK values must be 10 digits with leading zeros for API paths.
- Company Facts bulk ZIP source: `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip`
- SEC bulk ZIP files are recompiled nightly.
- SEC automated access must comply with SEC privacy/security policy and Developer FAQ guidance.

SEC-2 does not scrape HTML pages for financial data.

## Cache Policy

Approved default local cache family:

```text
data/local/sec_edgar/companyfacts/
```

`.gitignore` now ignores:

```text
data/local/
```

The downloaded SEC ZIP, extracted SEC files, generated manifests, logs, generated CSVs, processed outputs, and reports must not be committed unless repository policy explicitly changes.

The utility writes only when explicitly called. It does not download on import and is not connected to normal pipeline execution.

## Validation Behavior

The utility validates:

- source URL uses HTTPS;
- source URL host is `www.sec.gov`;
- source URL path is the approved Company Facts bulk ZIP path;
- download calls provide an explicit descriptive User-Agent;
- cache path is a directory and not the filesystem root;
- local ZIP exists, is a file, is readable, contains at least one file, and contains JSON files;
- SHA-256 and file size are recorded in the manifest.

Manifest fields include:

- `status`
- `source_url`
- `downloaded_at`
- `extraction_date`
- `source_freshness_date`
- `file_size_bytes`
- `sha256`
- `local_zip_path`
- `file_count`
- `json_file_count`

## Tests Added

Focused tests:

```text
tests/fundamentals/test_sec_companyfacts_bulk_intake.py
```

Tests cover:

- no network call on import;
- official URL accepted;
- unsupported scheme rejected;
- unsupported host rejected;
- unsupported path rejected;
- missing User-Agent rejected for download;
- cache directory creation under a temp path;
- file cache path rejected;
- manifest generation from a local fixture ZIP;
- invalid ZIP failure;
- local fixture ZIP validation success;
- fixture-based download behavior writing only under the provided temp cache path;
- no pipeline integration hooks exposed.

Tests use only temporary directories and tiny fixture ZIP files. They do not call live SEC endpoints, require internet, download real SEC data, write to real `data/local/`, commit generated artifacts, or depend on current market data.

## No Runtime Downstream Change Confirmation

SEC-2 introduced no runtime downstream behavior changes.

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

## Generated Data Commit Policy

No live SEC data was downloaded or committed.

No generated SEC archives, extracted SEC files, generated CSVs, processed outputs, operational logs, or reports were committed.

Any future generated SEC artifacts must remain local or ignored unless repository policy explicitly allows committing them.

## Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

SEC-2 remains within the already approved SEC source-data implementation sequence. Future provider/API automation and broader ingestion strategy remain governed by BL-0017.

## Recommended Next Sprint

Recommended next sprint:

```text
SEC-3 — SEC Ticker/CIK Index and Coverage Report
```

Purpose:

- build or validate ticker-to-CIK mapping;
- determine SEC coverage for the relevant scanner and portfolio universe;
- identify missing tickers;
- produce coverage evidence without committing generated SEC datasets.

SEC-3 should not implement XBRL tag mapping or transformation into `fundamentals_history.csv`; those remain later sprint scopes.
