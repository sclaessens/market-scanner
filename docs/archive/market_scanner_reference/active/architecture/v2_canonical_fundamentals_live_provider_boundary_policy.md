# Canonical Fundamentals Live Provider Boundary Policy

## Status

Completed by RESET-10L-BL47.

## Reset stage

RESET-10L-BL47 - Govern Canonical Fundamentals Live Provider Boundary.

## Purpose

Define the governance policy for a future canonical v2 fundamentals live-provider boundary before any live SEC/EDGAR, yfinance, API, network, cache, raw-payload, or production persistence implementation is added.

BL47 does not approve implementation of live provider execution. It only governs the boundary required before such implementation.

This sprint is governance-only and documentation-only. It does not modify Python files, tests, workflows, data files, report files, archived scripts, script-era files, cache files, raw payloads, or production artifacts.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/active/v2_high_risk_script_era_test_execution_cleanup.md`
- `docs/active/v2_fundamentals_script_era_side_effect_migration_review.md`
- `docs/active/v2_free_cash_flow_derivation_policy.md`
- `docs/active/v2_free_cash_flow_derivation_implementation.md`
- `docs/active/v2_prior_year_growth_evidence_implementation.md`
- `docs/active/v2_nvda_real_source_persistence_smoke.md`
- `docs/active/v2_nvda_first_real_fundamental_analysis_review.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_derived_fcf.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_growth_evidence.md`
- `docs/active/v2_real_analysis_output_defect_review.md`
- Repository doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority.
- English-only repository content governance.

## Documents inspected

Governance and cleanup records:

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_high_risk_script_era_side_effect_cleanup_review.md`
- `docs/active/v2_high_risk_script_era_test_execution_cleanup.md`
- `docs/active/v2_fundamentals_script_era_side_effect_migration_review.md`
- `docs/active/backlog.md`

Fundamentals records:

- `docs/active/v2_free_cash_flow_derivation_policy.md`
- `docs/active/v2_free_cash_flow_derivation_implementation.md`
- `docs/active/v2_prior_year_growth_evidence_implementation.md`
- `docs/active/v2_nvda_real_source_persistence_smoke.md`
- `docs/active/v2_nvda_first_real_fundamental_analysis_review.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_derived_fcf.md`
- `docs/active/v2_nvda_real_analysis_rerun_with_growth_evidence.md`
- `docs/active/v2_real_analysis_output_defect_review.md`

Canonical fundamentals context:

- `src/market_scanner/fundamentals/`
- `src/market_scanner/fundamentals/fundamentals_provider_contracts.py`
- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `src/market_scanner/fundamentals/fundamentals_persistence.py`
- `src/market_scanner/fundamentals/fundamentals_real_source_smoke.py`
- `tests/unit/test_v2_fundamentals_provider_adapter.py`
- `tests/unit/test_v2_fundamentals_real_source_smoke.py`
- `tests/contract/test_v2_provider_to_persistence_integration_contracts.py`
- `tests/contract/test_v2_fundamentals_provider_contracts.py`

## Canonical owner

The canonical owner for future fundamentals live-provider policy and implementation is:

```text
src/market_scanner/fundamentals/
```

This boundary owns provider response contracts, normalization, provenance, explicit missingness, governed derivation, readiness, source-data evidence, and the persistence boundary.

Script-era fundamentals, provider, SEC, yfinance, and data-source files remain migration or retirement candidates. They are not approved live-provider runtime authorities.

## Approved source family

The approved first source family for a future canonical live-provider boundary is:

```text
SEC EDGAR / SEC CompanyFacts
```

Rationale:

- SEC EDGAR is an official public source.
- SEC CompanyFacts is fundamentals-specific.
- It aligns with the prior NVDA source-shaped smoke records.
- It can preserve ticker, CIK, fiscal period, concept, unit, accession, and retrieval provenance.
- It avoids broker, market-data, or API-key dependency for the first governed implementation.

`yfinance` is not approved for the first canonical fundamentals live-provider implementation. Any future yfinance use requires separate governance and must not be smuggled into the fundamentals live-provider boundary under this policy.

## Invocation mode policy

The default canonical application remains dry-run only.

BL47 approves no automatic live-provider execution.

Future live-provider implementation must obey:

- no automatic live-provider execution;
- no scheduled live-provider execution;
- no workflow live-provider execution;
- no scanner-triggered live-provider execution;
- no production pipeline live-provider execution;
- no multi-ticker live-provider run;
- explicit local operator invocation only after separate implementation approval;
- first implementation limited to a single-ticker smoke boundary.

The first implementation candidate, if approved later, should be a one-ticker SEC CompanyFacts canonical live-provider smoke boundary, with NVDA as the default validation ticker unless superseded by a later approval.

## Ticker and scope policy

The first approved scope for future implementation is one ticker only.

Policy:

- NVDA is the default first validation ticker because earlier controlled work used NVDA source-shaped SEC evidence.
- No portfolio-wide capture is approved.
- No watchlist-wide capture is approved.
- No scanner-produced universe capture is approved.
- No automatic expansion from one ticker to multiple tickers is approved.
- No broad historical capture is approved.
- No bulk SEC CompanyFacts ingestion is approved by BL47.

Any expansion beyond one ticker requires a separate governed sprint.

## Network and credential policy

BL47 performs no network calls and approves no implementation.

Future implementation must make network access explicit and fail closed by default.

Policy:

- SEC CompanyFacts is public-source and must not require API keys, broker credentials, provider tokens, or Telegram credentials.
- No API keys may be committed.
- No secrets may be committed.
- No credentials may be read unless separately approved.
- Environment-variable reads for credentials are not approved by BL47.
- Network behavior must be isolated from normalization, readiness, and persistence logic.
- Provider errors, request errors, malformed responses, rate-limit responses, and unavailable source responses must return explicit provider/source error states rather than partial hidden data.

## SEC User-Agent policy

Future SEC access must define a governed SEC User-Agent before any live request is made.

Policy:

- A User-Agent must be explicit, descriptive, non-secret, and suitable for SEC public-source access.
- A User-Agent must not embed tokens, API keys, passwords, or personal secrets.
- User-Agent configuration must be reviewed before live SEC calls.
- BL47 does not approve reading a User-Agent from environment variables unless a later sprint explicitly governs that behavior.
- Missing or invalid User-Agent configuration must fail closed before network access.

## Raw payload and cache policy

No raw live payloads are approved for commit or production retention by BL47.

Policy:

- no raw live payloads committed;
- no unredacted SEC payloads committed;
- no SEC cache files committed;
- no broad CompanyFacts cache committed;
- no production cache path approved in BL47;
- no raw payload write path approved in BL47;
- tests must remain fake, injected, or `tmp_path`-only;
- redacted summaries may be documented only when explicitly approved and must exclude raw unredacted values, request headers, credentials, and cache contents;
- raw payload retention requires separate approval.

## Persistence policy

No production persistence path is approved by BL47.

Policy:

- no production data writes by default;
- no writes under `data/` unless separately approved;
- no raw payload writes;
- no cache writes;
- no report writes;
- no Telegram artifacts;
- no portfolio or watchlist updates;
- first implementation should be in-memory or `tmp_path`-only smoke;
- production persistence requires a separate path-policy sprint.

The existing canonical persistence boundary remains useful for validation and synthetic/tmp-path-safe write tests. It does not authorize production persistence.

## Mandatory provenance requirements

All future live-provider facts must preserve explicit provenance.

At minimum, each fact or normalized record must carry:

- source family;
- provider name;
- ticker;
- CIK when SEC is used;
- accession or report identifier when available;
- company identity;
- fiscal year;
- fiscal period;
- fiscal quarter when applicable;
- period end date when available;
- source field or SEC concept;
- unit;
- currency where applicable;
- reported value;
- derived value when applicable;
- derivation formula when applicable;
- source timestamp when available;
- retrieval timestamp;
- original source reference;
- source record identity;
- missingness reason when missing;
- provider/source status;
- provider error status when applicable;
- quality or readiness state.

Derived facts must be distinguishable from source-reported facts and must preserve the source fields used in the derivation.

## Missingness and failure policy

Missing values must remain explicit.

Policy:

- missing values must never be converted to zero;
- provider errors fail closed;
- network errors fail closed;
- unavailable source responses fail closed;
- ambiguous SEC facts fail closed;
- unit mismatch fails closed;
- currency mismatch fails closed;
- period mismatch fails closed;
- fiscal context mismatch fails closed;
- missing provenance fails closed;
- stale or incomplete data must produce `review_required`, `limited_analysis`, `partial_data`, `stale_data`, `invalid_data`, `source_missing`, or `provider_error` style states rather than final decision behavior.

Failure states must remain source/data-focused and must not imply investment quality, tradeability, allocation, urgency, or recommendation.

## SEC fact-selection policy

SEC CompanyFacts may expose multiple concepts, units, frames, periods, forms, fiscal contexts, amended values, and company-specific reporting patterns.

Future fact selection must be governed before broad ingestion.

Policy:

- selection must be deterministic;
- selection must preserve concept, unit, fiscal period, frame, form, and retrieval provenance;
- selection must prefer comparable annual facts when configured for annual analysis;
- selection must reject ambiguous values rather than guessing;
- mapping from canonical fields to SEC concepts requires an explicit contract;
- skipped, rejected, ambiguous, amended, or conflicting facts must be visible as review evidence when relevant;
- fact selection must not silently backfill missing facts from unrelated periods, units, frames, or concepts.

## CIK/ticker mapping policy

Ticker-to-CIK mapping must be governed before live SEC implementation.

Policy:

- mapping must be deterministic;
- mapping source must be governed;
- ticker ambiguity must fail closed;
- company identity must be preserved;
- CIK must be included in SEC provenance;
- no broad CIK cache is approved for commit by BL47;
- no broad CIK cache write path is approved by BL47.

For the first one-ticker smoke, a deterministic NVDA-to-CIK mapping may be used only if the source and company identity are documented and the implementation remains explicitly local and single-ticker.

## Canonical field mapping policy

Canonical fundamentals field mapping must remain explicit.

Initial canonical fields may include:

- revenue;
- gross profit;
- operating income;
- net income;
- diluted EPS;
- total assets;
- total liabilities;
- shareholders equity;
- operating cash flow;
- capital expenditures;
- free cash flow.

Policy:

- each canonical field must map to one or more approved SEC concepts by contract;
- direct source-reported FreeCashFlow remains distinct from governed derived FreeCashFlow;
- derived FreeCashFlow may use only the governed formula and validation conditions already approved;
- prior-year growth evidence must preserve current and prior provenance;
- EPS growth evidence remains a separate governance gap unless explicitly approved later;
- no unmapped SEC concept may be treated as canonical evidence without review.

## Test and smoke policy

Default tests must use fake, injected, source-shaped responses and must not call live providers.

Policy:

- unit tests must remain network-free;
- contract tests must remain network-free unless a later sprint explicitly creates a manual smoke lane;
- workflow tests must not run live providers;
- future live SEC smoke must be explicit local operator invocation only;
- first live smoke, if approved, must be single-ticker and use NVDA unless superseded;
- live smoke output must be in-memory or `tmp_path`-only;
- no raw payloads, caches, production data, reports, Telegram artifacts, portfolio updates, or watchlist updates may be produced;
- provider failures must be testable through injected fake clients before any live path is approved.

## Forbidden behavior

BL47 explicitly forbids:

- implementation of live provider execution;
- multi-ticker live capture;
- portfolio-wide live capture;
- watchlist-wide live capture;
- scanner-triggered live provider execution;
- workflow live provider execution;
- scheduled live provider execution;
- production pipeline live provider execution;
- yfinance as the first canonical fundamentals live provider;
- API-key or credential dependency for the first implementation;
- production data writes;
- raw payload commits;
- cache commits;
- report generation;
- Telegram artifact creation;
- Telegram delivery;
- portfolio mutation;
- watchlist mutation;
- Decision Engine final behavior;
- BUY, SELL, or HOLD behavior;
- allocation behavior;
- conviction behavior;
- urgency behavior;
- scoring behavior;
- target-price behavior;
- tradeability behavior;
- recommendation behavior;
- missing-value-to-zero conversion.

No workflow, scanner, portfolio, or watchlist live-provider execution is approved by BL47.

## Script-era fundamentals relationship

Script-era fundamentals, data-source, provider, SEC, yfinance, and builder files remain migration or retirement candidates.

They must not be expanded as live-provider authorities.

Useful script-era logic may be migrated later only through controlled sprints that preserve canonical ownership, fail-closed missingness, provenance, no production writes by default, and no investment semantics.

High-risk script-era files such as SEC bulk intake, SEC transform, data-source prefill, old quality builders, and yfinance/scanner support must remain outside active live-provider execution until governed migration or retirement is approved.

## Implementation readiness conclusion

The canonical fundamentals live-provider boundary is governed but not implemented.

BL47 approves the policy conditions required before implementation. It does not approve live SEC/EDGAR calls, yfinance calls, external API calls, network calls, raw payload writes, cache writes, production data writes, report generation, Telegram artifacts, portfolio/watchlist updates, or Decision Engine behavior.

The first future implementation candidate is a one-ticker SEC CompanyFacts canonical live-provider smoke boundary, with NVDA as the default validation ticker unless superseded by later approval.

No production persistence path is approved by BL47.

## Required next implementation guardrails

Any future implementation sprint must:

- inspect the existing canonical fundamentals boundary first;
- avoid script-era runtime authority;
- use `src/market_scanner/fundamentals/` as the canonical owner;
- keep live provider execution separate from normalization, readiness, and persistence validation;
- require explicit local invocation;
- remain single-ticker for the first implementation;
- fail closed for provider/network/user-agent/ticker/CIK/fact-selection ambiguity;
- preserve mandatory provenance;
- keep raw payloads and caches out of committed files;
- keep tests fake/injected or `tmp_path`-only by default;
- avoid workflow integration;
- avoid production persistence;
- avoid reports, Telegram, portfolio/watchlist mutation, and Decision Engine behavior.

## Recommended next sprint

Proceed to:

```text
RESET-10L-BL48 — Implement Canonical Fundamentals SEC CompanyFacts Smoke Boundary
```

BL48 should implement only the separately approved one-ticker, explicit-local, no-production-write, fail-closed SEC CompanyFacts smoke boundary if the operator approves implementation scope.

## Guardrails confirmation

- No Python files changed.
- No tests changed.
- No workflows changed.
- No files moved.
- No files deleted.
- No files archived.
- No script-era files executed.
- No archived scripts executed.
- No SEC/EDGAR calls made.
- No yfinance/provider calls made.
- No network calls.
- No credentials read.
- No production data writes.
- No raw payloads written.
- No cache files written.
- No reports generated.
- No Telegram artifacts generated.
- No Telegram delivery.
- No portfolio/watchlist updates.
- No Decision Engine behavior changed.
- No BUY/SELL/HOLD/allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior added.

## Known limitations

BL47 does not implement the live-provider boundary.

It does not prove SEC fact-selection parity, ticker-to-CIK mapping behavior, live SEC request behavior, SEC User-Agent configuration, cache governance, raw-payload retention, production persistence, multi-period ingestion, or CompanyFacts bulk-intake retirement.

Those remain future controlled cleanup or implementation topics.
