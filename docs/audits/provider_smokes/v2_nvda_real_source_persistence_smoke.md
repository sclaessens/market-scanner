# V2 NVDA Real-Source Persistence Smoke

Status: ACTIVE
Reset stage: RESET-10L-BL19

## Ticker

NVDA

## Company

NVIDIA Corporation

## Execution Mode

Controlled local smoke execution with an explicitly supplied, redacted
source-shaped provider response.

The source-shaped response was manually constructed from public SEC EDGAR
metadata for NVIDIA Corporation's Form 10-K filing and then passed through the
existing v2 provider/source boundary, normalization boundary, neutral readiness
boundary, and controlled synthetic persistence boundary.

The local smoke used no committed credentials, no API keys, no provider tokens,
no automatic provider scheduling, no production pipeline execution, and no
production data writes.

## Source Family

SEC EDGAR Form 10-K.

Safe source reference summary:

- company: NVIDIA Corporation;
- ticker: NVDA;
- CIK: `0001045810`;
- accession: `0001045810-25-000023`;
- form type: 10-K;
- filing date: February 26, 2025;
- accepted timestamp: February 26, 2025 16:48:33;
- period of report: January 26, 2025;
- SEC filing index: `https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/0001045810-25-000023-index.htm`.

## Source Access Mode

Source access was public-source review plus local source-shaped handoff.

No raw SEC response body, inline XBRL document, unredacted payload, request
header, credential, cache file, or generated provider output was committed.

## Captured Or Manually Supplied

The smoke manually supplied a redacted `ProviderSourceResponse` for NVDA with
public SEC EDGAR provenance metadata and source-shaped XBRL field presence.

Observed source-shaped fields supplied to the provider boundary:

- `Revenues`;
- `GrossProfit`;
- `OperatingIncomeLoss`;
- `NetIncomeLoss`;
- `EarningsPerShareDiluted`;
- `Assets`;
- `Liabilities`;
- `StockholdersEquity`;
- `NetCashProvidedByUsedInOperatingActivities`;
- `PaymentsToAcquirePropertyPlantAndEquipment`.

All field values were redacted before handoff and represented as redacted public
value placeholders. This smoke tested boundary behavior, provenance linkage,
missingness, readiness, and persistence safety rather than preserving raw live
values.

## Intentionally Not Committed

The following were intentionally not committed:

- credentials;
- API keys;
- request headers;
- raw unredacted SEC payloads;
- inline XBRL source body;
- copied raw financial values;
- local cache files;
- provider temporary files;
- generated production data files;
- reports;
- Telegram artifacts;
- portfolio or watchlist updates;
- Decision Engine outputs.

## Readiness Result

Smoke status: `review_required`

Readiness record summary:

- readiness state: `partial`;
- source data status: `partial`;
- missing fundamentals count: `1`;
- partial data count: `1`;
- stale data count: `0`;
- provider status: `available`;
- provider error status: empty.

The readiness result remained source/data-focused and neutral. It did not imply
investment quality, valuation attractiveness, portfolio action, allocation,
conviction, urgency, target price, tradeability, or recommendation behavior.

## Missing Field Result

Explicit missing field:

- `FreeCashFlow`

Reason: the current provider mapping expects a direct `FreeCashFlow` field.
The controlled source-shaped SEC handoff did not supply a direct free-cash-flow
field. The value remained explicit as missing and was not derived, hidden,
backfilled, or converted to zero.

Provider-boundary warnings summary:

- `issue:metric_value:missing_required_value`;
- `issue:metric_unit:missing_required_value`;
- `issue:currency:missing_required_value`;
- `missing_fundamentals:1`.

Persistence-boundary validation summary:

- raw validation issues: `0`;
- normalized validation issues: `0`;
- readiness validation issues: `0`;
- persistence batch status: `valid`;
- persistence batch issue count: `0`.

## Persistence Result

The controlled persistence boundary accepted the redacted raw, normalized, and
readiness-shaped records.

The optional synthetic persistence write was executed only inside a temporary
directory created for the smoke. The temporary directory was removed after the
smoke.

Temporary write summary:

- write status: `written`;
- write record count: `13`;
- output families created in temporary root:
  - `raw_source_evidence`;
  - `normalized_fundamentals`;
  - `source_data_readiness`.

No persistence output was written under `data/`, `reports/`, or
`.github/workflows/`.

## Side-Effect Check

Post-smoke side-effect checks confirmed:

- `git status --short` showed no repository changes after execution;
- no leftover `/private/tmp/nvda-persistence-smoke-*` directory remained;
- no production data write was committed;
- no report was generated;
- no Telegram artifact was generated;
- no production pipeline was executed;
- no portfolio data was modified;
- no watchlist data was modified;
- no Decision Engine investment behavior was invoked.

The repository already contains historical `data/`, `reports`, and workflow
files. This smoke did not create or modify those files.

## Defects Or Blockers Discovered

The smoke did not fail closed.

Observed follow-up defect or limitation:

- The current provider mapping expects a direct `FreeCashFlow` source field.
  The NVDA SEC 10-K source-shaped handoff did not provide that direct field, so
  normalized `free_cash_flow` remained explicitly missing. A future review must
  decide whether free cash flow should remain source-only, be derived in a
  separately governed transformation, or be excluded from required readiness for
  official SEC annual filings.

Additional implementation observation:

- The provider-boundary normalization contract emits missing currency and unit
  warnings for the missing `free_cash_flow` normalized record. The persistence
  boundary can preserve explicit missingness safely, but future analysis should
  review whether missing metric value, missing currency, and missing unit should
  be grouped into one clearer source-data warning.

## Guardrail Confirmation

No credentials committed.
No raw unredacted live payload committed.
No production data writes committed.
No reports generated.
No Telegram artifacts generated.
No production pipeline executed.
No portfolio/watchlist data modified.
No Decision Engine investment behavior invoked.

No BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price,
tradeability, or recommendation behavior was added or invoked.

## Conclusion

The controlled NVDA one-ticker real-source persistence smoke succeeded with
`review_required` / `partial` readiness.

The smoke proved that a redacted, source-shaped NVDA SEC EDGAR 10-K response can
move through the approved v2 provider/source boundary, produce separate raw
evidence and normalized fundamentals, preserve explicit missing values, emit
neutral readiness, validate through the controlled persistence boundary, and
write only to temporary synthetic persistence directories without production
side effects.

The main real-data learning is that direct `FreeCashFlow` availability remains a
source/mapping limitation for this path and should be handled before first
analysis attempts depend on free-cash-flow completeness.

## Next Recommended Step

Proceed to:

```text
RESET-10L-BL20 — Run First One-Ticker Real Fundamental Analysis
```

Recommended BL20 constraint: the first analysis should remain one-ticker NVDA
only, should explicitly carry the `FreeCashFlow` missingness forward, and should
avoid BUY, SELL, HOLD, allocation, conviction, urgency, scoring, target-price,
tradeability, or recommendation behavior unless separately approved.
