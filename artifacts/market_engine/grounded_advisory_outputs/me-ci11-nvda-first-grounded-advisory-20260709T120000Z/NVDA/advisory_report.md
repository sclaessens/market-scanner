# Grounded Advisory Report - NVDA

This report is based only on the referenced Market Engine artifact. It does not use external model knowledge, live prices, news, broker data, portfolio mutation, or delivery authority.

## Instrument and context

- Ticker: NVDA
- Company/instrument: NVDA
- Source artifact: `artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/cached_source_batch/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/NVDA/dry_run.json`
- Source run: `me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z-nvda`
- Source generated at: 2026-07-02T11:56:52Z
- Input mode: cached_source_snapshot
- Readiness: partial_analysis
- Actionability allowed: False
- Advisory status: blocked_invocation_not_configured
- Missing data: portfolio_context, setup_price_market
- Stale data: None
- Blocked stage: portfolio_review

## Executive conclusion

No grounded advisory conclusion was generated because the model invocation boundary is blocked: OPENAI_API_KEY and MARKET_ENGINE_ADVISORY_MODEL or OPENAI_MODEL are required for real invocation.

## What supports this conclusion?

- None accepted in the grounded output.

## Main risks and limitations

- `blocked:reason:0` (blocked_state): Stage preserves an upstream blocked state.
- `blocked:reason:1` (blocked_state): missing_setup_or_price_context
- `blocked:stage` (blocked_state): Dry-run is blocked at portfolio_review.
- `missing:data:0` (missing_data): portfolio_context
- `missing:data:1` (missing_data): setup_price_market
- `readiness:analysis_context` (readiness): Readiness is partial_analysis with actionable_review_allowed=False and decision_engine_ready=False.

### Limitations

- Stage preserves an upstream blocked state.
- missing_setup_or_price_context
- portfolio_context
- setup_price_market

## Practical interpretation

Treat this as a blocked/non-actionable advisory generation result. Review the source artifact and invocation blocker before using any advisory output.

## Confidence / evidence quality

No successful grounded model response was accepted.

## Grounding and validation

- Parser status: invalid / empty_response
- Validation status: invalid
- Invocation state: request_blocked
- Provider/model: not_configured / not_configured
