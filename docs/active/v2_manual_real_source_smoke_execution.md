# V2 Manual Real-Source Smoke Execution

Status: ACTIVE
Reset stage: RESET-10L-BL8

## Purpose

This document describes how a developer/operator may manually execute a
controlled v2 real-source smoke review locally.

This is not production execution. It is not automated provider integration. It
is not investment analysis. It does not approve data writes, report generation,
Telegram delivery, production pipeline execution, Decision Engine behavior,
scoring, recommendations, BUY, SELL, HOLD, allocation, conviction, urgency, or
tradeability logic.

The allowed path is:

```text
manual local invocation
-> one ticker/source
-> injected source client or explicit ProviderSourceResponse
-> controlled smoke harness
-> in-memory result
-> terminal-only human review
-> no committed live output
-> no data writes
```

## Execution Modes

Two manual execution modes are allowed.

### Mode A — Explicit ProviderSourceResponse Review

Mode A is the safest first operator path. The operator manually supplies an
explicit provider/source-shaped response and reviews it through:

```text
review_injected_source_response(response)
```

This mode does not call a provider client. It uses only in-memory objects and is
appropriate for reviewing a copied or locally prepared response shape without
committing live output.

### Mode B — Injected Source Client Review

Mode B uses an injected local client with this shape:

```text
fetch_fundamentals(ticker) -> ProviderSourceResponse
```

The injected client must be manually supplied by the operator. Credentials,
source output, raw payloads, local scratch files, and terminal output must not be
committed. The client must return a `ProviderSourceResponse` and must not write
files, generate reports, send Telegram messages, run the production pipeline, or
touch Decision Engine behavior.

## Minimal Mode A Example

The following example uses only in-memory objects. It does not write files, call
providers, use credentials, import scripts, import reporting, import Telegram,
or import Decision Engine investment logic.

```python
from market_scanner.fundamentals.fundamentals_provider_contracts import (
    ProviderCategory,
    ProviderRawFieldEvidence,
    ProviderSourceResponse,
    ProviderSourceStatus,
)
from market_scanner.fundamentals.fundamentals_real_source_smoke import (
    review_injected_source_response,
)


response = ProviderSourceResponse(
    provider_name="Local Manual Source Review",
    provider_category=ProviderCategory.REGULATORY_FILING.value,
    provider_record_id="ASML-FY-2025-MANUAL-LOCAL",
    original_source_reference="local-manual-review://ASML/FY/2025",
    ticker="ASML",
    symbol="ASML",
    entity_identifier="ASML-HOLDING-NV-MANUAL",
    source_timestamp="2026-02-11T08:00:00Z",
    retrieval_timestamp="2026-06-03T00:00:00Z",
    reported_period="FY",
    fiscal_year="2025",
    fiscal_quarter="",
    raw_fields={
        "Revenues": ProviderRawFieldEvidence(
            original_field_name="Revenues",
            original_field_value="28000000000",
            original_currency="EUR",
            original_unit="EUR",
        ),
        "NetIncomeLoss": ProviderRawFieldEvidence(
            original_field_name="NetIncomeLoss",
            original_field_value="7600000000",
            original_currency="EUR",
            original_unit="EUR",
        ),
    },
    provider_status=ProviderSourceStatus.PARTIAL_DATA.value,
    provider_error_status="",
    missing_field_evidence=("GrossProfit", "FreeCashFlow"),
    provenance_metadata="local manual review metadata",
    raw_payload_hash="sha256:local-manual-review-placeholder",
    capture_version="v2-manual-smoke-v1",
)

result = review_injected_source_response(response)

print(
    {
        "ticker": result.ticker,
        "provider_name": result.provider_name,
        "smoke_status": result.smoke_status,
        "missing_fields": result.missing_field_summary,
        "warnings": result.warnings,
    }
)
```

The printed summary is for terminal-only human review. It must not be committed
if it contains live provider/source output.

## Injected Client Example

The following shape is allowed for a manually supplied local client:

```python
from market_scanner.fundamentals.fundamentals_provider_contracts import (
    ProviderSourceResponse,
)
from market_scanner.fundamentals.fundamentals_real_source_smoke import (
    run_controlled_real_source_smoke_test,
)


class LocalSourceClient:
    def fetch_fundamentals(self, ticker: str) -> ProviderSourceResponse:
        # Operator-supplied local source access goes here.
        # Do not commit credentials, live output, raw payloads, or scratch files.
        raise NotImplementedError("return a ProviderSourceResponse manually")


client = LocalSourceClient()
result = run_controlled_real_source_smoke_test(client, ticker="ASML")
print(result.smoke_status)
```

This example documents the shape only. The operator is responsible for supplying
the local client outside the committed repository diff. Output must remain
terminal-only or local scratch-only and must not be committed.

## Operator Commands

Before a manual review, confirm the repository is clean and tests for the smoke
boundary still pass:

```bash
git checkout main
git pull origin main
git status

.venv/bin/python -m pytest tests/unit/test_v2_fundamentals_real_source_smoke.py
.venv/bin/python -m pytest tests/unit/test_v2_fundamentals_provider_adapter.py
.venv/bin/python -m pytest tests/contract/test_v2_provider_dry_run_fixture_review.py

git status
```

No helper module or CLI command is added by RESET-10L-BL8. The controlled smoke
harness remains the existing importable v2 module:

```text
src/market_scanner/fundamentals/fundamentals_real_source_smoke.py
```

## Review Checklist

During review, check:

- ticker;
- provider/source name;
- provider category;
- source reference;
- source timestamp;
- retrieval timestamp;
- fiscal period;
- currency;
- unit;
- raw field preservation;
- normalized field mapping;
- explicit missing values;
- no missing-to-zero conversion;
- neutral readiness;
- no investment conclusions;
- no side effects;
- clean working tree after review.

## Post-Run Safety Checks

After a manual review, run:

```bash
git status
git diff --stat
git diff --check
```

Any live output or local scratch output must not be committed. If a future
summary is committed, it must be manually written, summary-only, and must
exclude credentials, secrets, API keys, raw live payloads, private data, large
provider output, and investment conclusions.

## Forbidden Outcomes

Manual smoke execution must not produce:

- committed credentials;
- committed live provider/source output;
- committed raw live payloads;
- generated data files;
- report files;
- Telegram artifacts;
- production pipeline runs;
- automated provider execution;
- Decision Engine investment behavior;
- BUY, SELL, HOLD, allocation, conviction, urgency, tradeability, scoring, or
  recommendation logic;
- missing values converted to zero.

## Next Step

The next candidate step is
`RESET-10L-BL9 — Local Real-Source Smoke Result Review`.

That future step should review the outcome of one local, manual, one-ticker
smoke execution. If anything is committed, it must be a governance-safe
summary-only review. It must not commit credentials, raw live payloads, data
files, reports, Telegram artifacts, production pipeline behavior, Decision
Engine investment logic, or BUY, SELL, HOLD, allocation, conviction, urgency, or
tradeability behavior.
