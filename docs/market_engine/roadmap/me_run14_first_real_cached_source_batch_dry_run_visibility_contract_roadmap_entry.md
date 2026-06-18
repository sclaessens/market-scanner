# ME-RUN14 - First real cached-source batch dry-run execution and visibility contract roadmap entry

Owner role: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN14

## Placement

ME-RUN14 follows ME-RUN13.

ME-RUN13 implemented safe cached-source batch dry-run runtime behavior. ME-RUN14 defines the first real cached-source batch dry-run execution and visibility contract before any operator-facing implementation sprint broadens the runtime into a practical command flow.

## Roadmap purpose

ME-RUN14 prevents the project from jumping from runtime behavior directly into an opaque batch execution.

It makes the first real local batch dry-run reviewable by defining:

* what the operator must see in the terminal;
* what evidence must be captured;
* when artifacts may be written;
* how artifacts must be exposed;
* how blocked and failed tickers must be triaged;
* which side effects remain forbidden;
* what the next implementation sprint may build.

## Contract preserved

ME-RUN14 preserves:

```text
market-engine-cached-source-batch-dry-run-v1
market-engine-end-to-end-dry-run-v1
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

ME-RUN14 defines the operator execution visibility contract:

```text
market-engine-real-cached-source-batch-dry-run-visibility-v1
```

## Documentation added

```text
docs/market_engine/run/me_run14_first_real_cached_source_batch_dry_run_visibility_contract.md
docs/market_engine/audits/me_run14_first_real_cached_source_batch_dry_run_visibility_contract_audit.md
docs/market_engine/backlog/me_run14_first_real_cached_source_batch_dry_run_visibility_contract_backlog_entry.md
docs/market_engine/roadmap/me_run14_first_real_cached_source_batch_dry_run_visibility_contract_roadmap_entry.md
```

## Roadmap update

ME-RUN14 is complete as a documentation-only sprint.

Recommended next implementation sprint:

```text
ME-RUN15 - Implement first real cached-source batch dry-run command visibility
```

ME-RUN15 may implement only the narrow operator-facing command and visibility behavior defined by ME-RUN14.

ME-RUN15 must not introduce provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, external API calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, generated artifact commits, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Follow-up candidates after ME-RUN14

These remain candidates only and must not be inserted ahead of ME-RUN15 unless implementation uncovers a real blocker or governance gap:

* `ME-SR02 - Build bounded SEC CompanyFacts source refresh job runner`
* `ME-QA01 - Add cross-job dry-run contract regression suite`
