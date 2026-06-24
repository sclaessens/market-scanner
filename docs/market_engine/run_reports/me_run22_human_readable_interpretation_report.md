# ME-RUN22 - Human-readable Market Engine interpretation report

Sprint: ME-RUN22 - Produce first human-readable Market Engine interpretation report from cached-source supported-universe outputs

Status: Implemented

## Purpose

ME-RUN22 implements the first deterministic human-readable Market Engine interpretation report generator for cached-source supported-universe local dry-run artifacts.

The report converts existing local artifact wrappers and dry-run payloads into a Markdown summary plus a machine-readable JSON summary. It does not fetch data, alter upstream artifacts, change Market Engine review semantics, or create action authority.

## Runtime Module

```text
src/market_engine/run_reports/interpretation_report.py
src/market_engine/run_reports/interpretation_report_command.py
```

Public builder:

```text
build_market_engine_interpretation_report(...)
```

CLI:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run_reports.interpretation_report_command \
  --input-artifact-root artifacts/market_engine/me-run20-supported-universe-20260623T120000Z \
  --output-root artifacts/market_engine \
  --report-run-id me-run22-human-readable-report-me-run20-supported-universe-20260623T120000Z \
  --generated-at 2026-06-23T12:45:00Z
```

## Input

Expected input is a local supported-universe artifact root with one ticker directory per inspected ticker:

```text
artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/<TICKER>/dry_run.json
artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/<TICKER>/manifest.json
```

The generator parses local JSON only. It does not call providers or refresh artifacts.

## Output

Report format:

```text
market-engine-interpretation-report-v1
```

Generated local sample output:

```text
artifacts/market_engine/me-run22-human-readable-report-me-run20-supported-universe-20260623T120000Z/market_engine_interpretation_report.md
artifacts/market_engine/me-run22-human-readable-report-me-run20-supported-universe-20260623T120000Z/market_engine_interpretation_report_summary.json
```

The output directory is explicit and overwrite-protected.

Generated sample artifacts remain local and are not committed by default.

## Report Content

The Markdown report includes:

* title and report metadata;
* source artifact root;
* non-actionable boundary statement;
* scope and safety section;
* universe summary;
* per-ticker artifact presence table;
* per-ticker sections;
* output-family and stage-state summary;
* missing, stale, and blocked data notes;
* provenance references;
* cross-universe observations;
* readiness assessment;
* recommended next sprint.

The JSON summary includes:

* input root;
* output paths;
* ticker counts;
* included tickers;
* skipped tickers and reasons;
* report format version;
* output families present across the included universe;
* non-actionable boundary;
* advisory-language guardrail metadata.

## Fail-Closed Behavior

The generator:

* reports missing `dry_run.json` as a skipped ticker;
* reports missing `manifest.json` as a skipped ticker;
* reports malformed JSON as a skipped ticker;
* refuses to overwrite an existing output directory;
* rejects unsafe report run IDs and path traversal.

It does not infer missing values or repair malformed artifacts.

## Non-Scope

ME-RUN22 does not add provider calls, live data fetching, source refresh, Telegram or email delivery, broker integration, portfolio writes, watchlist writes, upstream review changes, Decision Engine behavior, action authority, capital guidance, ordered preference semantics, confidence labels, timing labels, market-participation guidance, or execution instructions.

## Tests

Test module:

```text
tests/market_engine/run_reports/test_interpretation_report.py
```

Coverage includes:

* happy path with multiple tickers;
* missing ticker manifest;
* missing ticker dry-run artifact;
* malformed JSON;
* deterministic ticker ordering;
* Markdown advisory-language avoidance with JSON guardrail metadata;
* CLI report generation.

## Next Sprint

ME-OUT01 - Define readable operator report contract from dry-run artifacts.
