# ME-CI11C - Configured Provider Invocation

## Objective

Run the corrected universal CI11 grounded advisory runtime with a local API key and record the first real provider invocation outcome.

ME-CI11C did not reach a real OpenAI network invocation because `OPENAI_API_KEY` was not visible as a non-empty value in the command process. The model env var was supplied explicitly as `MARKET_ENGINE_ADVISORY_MODEL=gpt-4.1-mini`, and the persisted invocation requests show that model value. The remaining blocker is process-environment propagation of the API key into Codex command execution.

## Source main SHA

`52c0b3729323dec6bfb7c4f45b8c836b715da661`

## Branch

`me-ci11c-run-configured-provider-invocation`

## Provider configuration

- Provider family: OpenAI Responses API.
- Model env var: `MARKET_ENGINE_ADVISORY_MODEL`.
- Model: `gpt-4.1-mini`.
- `OPENAI_API_KEY` present locally: no in the command process.
- `OPENAI_API_KEY` process check: `OPENAI_API_KEY nonempty: False`.
- Secrets persisted: no.
- Tools enabled: no.
- Browsing enabled: no.
- Non-streaming: yes.
- `max_output_tokens`: 1200.
- Real provider call attempted: no, blocked before network invocation because the process did not receive a non-empty API key.

## Primary ticker run - NVDA

- Source artifact: `artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/NVDA/dry_run.json`.
- Run ID: `me-ci11c-nvda-real-provider-grounded-advisory-20260711T000000Z`.
- Output directory: `artifacts/market_engine/grounded_advisory_outputs/me-ci11c-nvda-real-provider-grounded-advisory-20260711T000000Z/NVDA`.
- Invocation state: `request_blocked`.
- Provider request id present: no.
- Raw provider response hash present: no.
- Raw output hash: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Finish reason: null.
- Usage summary: null.
- Parser status: `invalid`.
- Parser state: `empty_response`.
- Validation status: `invalid`.
- Grounding status: null.
- Advisory status: `blocked_invocation_not_configured`.
- Validation issue codes: `invocation_not_completed`, `missing_parsed_response`.
- Report path: `artifacts/market_engine/grounded_advisory_outputs/me-ci11c-nvda-real-provider-grounded-advisory-20260711T000000Z/NVDA/advisory_report.md`.

## Smoke ticker run - AMD

- Source artifact: `artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/AMD/dry_run.json`.
- Run ID: `me-ci11c-amd-real-provider-grounded-advisory-smoke-20260711T000000Z`.
- Output directory: `artifacts/market_engine/grounded_advisory_outputs/me-ci11c-amd-real-provider-grounded-advisory-smoke-20260711T000000Z/AMD`.
- Invocation state: `request_blocked`.
- Provider request id present: no.
- Raw provider response hash present: no.
- Raw output hash: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Finish reason: null.
- Usage summary: null.
- Parser status: `invalid`.
- Parser state: `empty_response`.
- Validation status: `invalid`.
- Grounding status: null.
- Advisory status: `blocked_invocation_not_configured`.
- Validation issue codes: `invocation_not_completed`, `missing_parsed_response`.
- Report path: `artifacts/market_engine/grounded_advisory_outputs/me-ci11c-amd-real-provider-grounded-advisory-smoke-20260711T000000Z/AMD/advisory_report.md`.

## Report quality review

NVDA:

- Conclusion readable: yes, as a blocked/non-actionable result.
- Correctly non-actionable/action-limited: yes.
- Blockers and missing data visible: yes.
- Disclosures present: yes.
- Unsupported claims present: no.
- Overstates certainty: no.
- Broker/order/sizing/allocation authority: no.

AMD:

- Conclusion readable: yes, as a blocked/non-actionable result.
- Correctly non-actionable/action-limited: yes.
- Blockers and missing data visible: yes.
- Disclosures present: yes.
- Unsupported claims present: no.
- Overstates certainty: no.
- Broker/order/sizing/allocation authority: no.

## Universal ticker-agnostic check

- No ticker-specific code added: confirmed.
- No company-specific prompt tuning: confirmed.
- Same command path used for NVDA and AMD: confirmed.
- Same schema and validation path used: confirmed.
- Per-ticker blockers preserved: confirmed.
- No source/provider refresh: confirmed.
- No manual report edits: confirmed.

## Batch-readiness toward 500 tickers

Current state:

- One output directory per run/ticker: yes.
- Per-ticker manifest: yes.
- Per-ticker raw response: yes.
- Per-ticker parser result: yes.
- Per-ticker validation result: yes.
- Per-ticker advisory report: yes.

Missing for 500 tickers:

- batch command;
- artifact discovery;
- ticker universe input;
- resume mode;
- per-ticker status index;
- failure summary;
- rate limits;
- cost tracking;
- concurrency policy;
- quota handling.

## Outcome

`blocked_other`

The exact blocker is process-environment propagation: `OPENAI_API_KEY` is not visible as a non-empty value in the command process used by Codex. The command can receive `MARKET_ENGINE_ADVISORY_MODEL=gpt-4.1-mini`, and the invocation request persists that model, but the API key remains empty before runtime invocation.

## Next recommendation

ME-CI11D - Fix Codex command process provider environment propagation.

Before another provider run, verify in the exact command process:

```text
OPENAI_API_KEY nonempty: True
MARKET_ENGINE_ADVISORY_MODEL: gpt-4.1-mini
```

Then rerun the same universal CI11 command path for NVDA and AMD without code changes, prompt tuning, source refresh, provider fallback, delivery, broker behavior, portfolio mutation, watchlist mutation, allocation, sizing, or Decision Engine changes.
