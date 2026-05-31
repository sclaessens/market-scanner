 # SEC-7A — Controlled Review Ticker Input Preparation

Status: DOCUMENTATION-ONLY PREPARATION  
Backlog context: BL-0015 / BL-0017  
Date: 2026-05-31

## Purpose

SEC-7A prepares the controlled local review procedure for running the SEC transformation review tooling against a small explicitly selected review ticker set.

This sprint does not run a real SEC review, download SEC data, call SEC endpoints, create generated review outputs, or integrate SEC transformation into the full pipeline.

## Scope Boundary

SEC-7A is documentation-only.

SEC-7A does not modify code, tests, data, generated files, reports, CSV outputs, workflow files, runtime behavior, Decision Engine, Reporting, Telegram, portfolio, scanner, validation, context, timing, portfolio intelligence, or fundamentals runtime behavior.

## Review Ticker Set

The controlled SEC review ticker set is:

- NVDA
- TWLO
- ON
- AMD
- KEYS

These tickers are a controlled review sample only. They are not the full source-data strategy.

## Required Local Inputs

SEC-7B requires explicit local input files under:

```text
data/local/sec_edgar/review/sec_7b/

Required files:

project_tickers.csv
ticker_cik_source.json
companyfacts/

Required output target:

sec_transformation_review.csv

All input and output files in this local review folder are operational review artifacts and must not be committed.

SEC-7B Local Review Command

SEC-7B should run the existing local review runner with explicit local input paths:

.venv/bin/python scripts/fundamentals/run_sec_transformation_review.py \
  --project-tickers data/local/sec_edgar/review/sec_7b/project_tickers.csv \
  --ticker-cik-source data/local/sec_edgar/review/sec_7b/ticker_cik_source.json \
  --companyfacts-dir data/local/sec_edgar/review/sec_7b/companyfacts \
  --output data/local/sec_edgar/review/sec_7b/sec_transformation_review.csv \
  --source-freshness-date YYYY-MM-DD \
  --extraction-date YYYY-MM-DD

The operator must replace both dates explicitly. Generated output must remain local and uncommitted.

Review Checklist

SEC-7B should confirm:

Each requested ticker appears in the review output.
Missing CIK mappings remain represented.
Missing Company Facts files remain represented.
Direct fields populate when source facts are present.
Missing values are not inferred or treated as zero.
total_debt and free_cash_flow populate only under approved component conditions.
Review-required statuses remain descriptive only.
Generated output remains local and uncommitted.
Review Interpretation Rules

SEC-7B is a source-data quality review, not investment advice and not an allocation review.

It must not produce BUY, SELL, allocation, ranking, urgency, conviction, final-action, or tradeability decisions.

Recommended Next Sprint

SEC-7B — Controlled Local SEC Review Run

SEC-7B should run the existing local review runner on explicit local inputs and report findings. It should not implement pipeline integration.

Backlog Impact Assessment

No new backlog items identified.

No-Runtime-Change Confirmation

SEC-7A confirms no scripts, tests, data, reports, CSV files, generated files, workflow files, runtime behavior, SEC/network calls, SEC downloads, scraping, Decision Engine behavior, Reporting behavior, Telegram behavior, or portfolio behavior changed.
