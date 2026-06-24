# ME-OUT02 - Readable operator report implementation

Sprint: ME-OUT02 - Implement readable operator report from dry-run artifacts

Status: Implemented

Job family: ME-OUT - Output / Operator Reporting

## Purpose

ME-OUT02 implements the `market-engine-readable-operator-report-v1` contract defined by ME-OUT01.

The implementation turns existing local Market Engine dry-run artifacts into a deterministic Markdown operator report and machine-readable companion summary.

## Implemented Runtime

Implemented package:

```text
src/market_engine/output_reports/
```

Public API:

```text
build_readable_operator_report(...)
ReadableOperatorReportResult
ReadableOperatorReportError
OperatorTickerInspection
```

CLI module:

```text
market_engine.output_reports.readable_operator_report_command
```

Example command:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.output_reports.readable_operator_report_command \
  --input-artifact-root artifacts/market_engine/me-run20-supported-universe-20260623T120000Z \
  --output-root artifacts/market_engine \
  --report-run-id me-out02-readable-operator-report-me-run20-supported-universe-20260623T120000Z \
  --generated-at 2026-06-24T12:00:00Z
```

## Output Contract

The report builder emits:

```text
operator_report.md
operator_report_summary.json
```

Report format version:

```text
market-engine-readable-operator-report-v1
```

Local output path pattern:

```text
artifacts/market_engine/<operator_report_run_id>/
```

The implementation refuses to overwrite an existing output directory.

## Input Behavior

The report consumes existing local artifact files only:

```text
<input_artifact_root>/<TICKER>/dry_run.json
<input_artifact_root>/<TICKER>/manifest.json
```

Supported input versions:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
market-engine-end-to-end-dry-run-v1
market-engine-interpretation-report-v1
```

Optional interpretation report roots are accepted when present and validated through the companion summary format version.

## Fail-Closed And Skip Behavior

The implementation fails closed for:

* missing input artifact root;
* input artifact root that is not a directory;
* unsafe report run id;
* path traversal;
* output directory overwrite.

The implementation skips individual ticker folders with explicit reasons for:

* missing `dry_run.json`;
* missing `manifest.json`;
* malformed JSON;
* unsupported dry-run artifact format version;
* unsupported manifest format version;
* unsupported dry-run payload format version;
* malformed dry-run payload.

## Preserved Evidence

The report preserves:

* ticker artifact paths;
* artifact format versions;
* dry-run format versions;
* manifest format versions;
* input mode;
* run state;
* completed stages;
* non-completed stages;
* output families;
* missing-data notes;
* stale-data notes;
* blocked reasons;
* blocked stage;
* provenance references;
* numeric-zero presence.

## Non-Scope

ME-OUT02 does not add provider calls, source refresh, live market data, broker integration, Telegram or email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, upstream dry-run mutation, Analysis Review changes, Recommendation Review changes, Portfolio Review changes, Decision Engine behavior changes, allocation authority, target prices, target weights, position sizing, order generation, execution guidance, ranking, scoring, urgency, conviction, or tradeability authority.

## Tests

Added focused tests under:

```text
tests/market_engine/output_reports/test_readable_operator_report.py
```

Coverage includes:

* multi-ticker happy path;
* required Markdown sections;
* required JSON metadata;
* deterministic ticker ordering;
* missing manifest;
* missing dry-run artifact;
* malformed JSON;
* unsupported format version;
* missing input root;
* unsafe report run id;
* overwrite refusal;
* missing/stale/blocked/provenance marker preservation;
* numeric-zero preservation;
* advisory-language guardrail;
* CLI report generation.

## Validation

Validation commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/output_reports -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run_reports -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
git diff --check
```

## Next Sprint

```text
ME-CANDIDATE01 - Define non-actionable candidate classification contract
```
