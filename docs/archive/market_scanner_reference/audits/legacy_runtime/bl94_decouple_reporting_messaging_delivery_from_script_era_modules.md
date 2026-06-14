# BL94 — Decouple reporting, messaging, and delivery from script-era modules

Status: COMPLETED

## Purpose

BL94 decouples active reporting, messaging, and delivery tests and canonical boundary metadata from script-era reporting and Telegram modules.

Targeted script-era modules:

```text
scripts/reporting/build_reporting_layer.py
scripts/reporting/build_telegram_summary.py
scripts/reporting/send_telegram.py
scripts/telegram/process_telegram_commands.py
```

BL94 is a decoupling sprint. It does not archive, delete, execute, or refactor the targeted script-era runtime modules.

## Scope

Changed active tests and canonical boundary metadata so that reporting, messaging, and delivery governance no longer depends on importing or statically reading the targeted script-era modules.

Updated files:

```text
src/market_scanner/delivery/delivery_boundary.py
src/market_scanner/messaging/message_boundary.py
src/market_scanner/reporting/report_boundary.py
tests/conftest.py
tests/reporting/test_build_reporting_layer.py
tests/reporting/test_build_telegram_summary.py
tests/test_operator_visibility.py
tests/unit/test_v2_canonical_delivery.py
tests/unit/test_v2_canonical_messaging.py
tests/unit/test_v2_canonical_reporting.py
```

Out of scope:

* archiving script-era reporting or Telegram modules;
* editing script-era reporting or Telegram modules;
* executing script-era reporting or Telegram modules;
* sending Telegram messages;
* reading credentials;
* writing production reports;
* writing production data;
* changing Decision Engine authority.

## What changed

### Reporting tests decoupled

The active reporting tests no longer import:

```text
scripts.reporting.build_reporting_layer
scripts.reporting.build_telegram_summary
```

They now validate canonical reporting, messaging, and delivery plans through:

```text
market_scanner.reporting.report_boundary
market_scanner.messaging.message_boundary
market_scanner.delivery.delivery_boundary
```

### Canonical boundary metadata decoupled

BL94 removed active script-era reporting and Telegram file-path authorities from canonical boundary metadata in:

```text
src/market_scanner/reporting/report_boundary.py
src/market_scanner/messaging/message_boundary.py
src/market_scanner/delivery/delivery_boundary.py
```

The boundaries no longer list the targeted active script-era paths as canonical legacy authorities.

### Canonical static tests decoupled

BL94 removed direct static file reads of the targeted script-era files from:

```text
tests/unit/test_v2_canonical_reporting.py
tests/unit/test_v2_canonical_messaging.py
tests/unit/test_v2_canonical_delivery.py
```

### Operator visibility blockers updated

Because the reporting tests are no longer high-risk script-era blocker tests, BL94 removed them from the high-risk blocker registry:

```text
tests/conftest.py
tests/test_operator_visibility.py
```

## Validation

Focused suite:

```bash
pytest tests/reporting/test_build_reporting_layer.py \
       tests/reporting/test_build_telegram_summary.py \
       tests/unit/test_v2_canonical_reporting.py \
       tests/unit/test_v2_canonical_messaging.py \
       tests/unit/test_v2_canonical_delivery.py \
       tests/test_operator_visibility.py -q
```

Result:

```text
45 passed in 0.05s
```

Full suite:

```bash
pytest -q
```

Result:

```text
560 passed in 0.57s
```

## Active import check

BL94 checked for active imports from script-era reporting and Telegram modules:

```bash
grep -RIn \
  "from scripts.reporting\|import scripts.reporting\|from scripts.telegram\|import scripts.telegram" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result:

```text
No output
```

Interpretation:

* no active positive imports remain from `scripts.reporting`;
* no active positive imports remain from `scripts.telegram`.

## Active path reference check

BL94 checked active path references for the targeted script-era files:

```bash
grep -RIn \
  "scripts/reporting/build_reporting_layer.py\|scripts/reporting/build_telegram_summary.py\|scripts/reporting/send_telegram.py\|scripts/telegram/process_telegram_commands.py" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Remaining output:

```text
tests/reporting/test_build_telegram_summary.py
tests/reporting/test_build_reporting_layer.py
```

Interpretation:

* remaining references are negative guardrail assertions only;
* they assert that canonical plans do not contain active script-era paths;
* they are not active imports, static file reads, runtime dependencies, or execution paths.

## Decision

BL94 decision:

```text
REPORTING_MESSAGING_DELIVERY_ACTIVE_DEPENDENCIES_DECOUPLED
```

The targeted script-era reporting and Telegram modules are not archived yet, but the active test and metadata dependency blockers have been removed.

## Remaining archive-readiness note

The targeted files still physically exist under `scripts/`:

```text
scripts/reporting/build_reporting_layer.py
scripts/reporting/build_telegram_summary.py
scripts/reporting/send_telegram.py
scripts/telegram/process_telegram_commands.py
```

They should remain in place until a dedicated archive-readiness sprint confirms:

* no active imports;
* no active static file reads;
* no active workflow references;
* only negative guardrail or historical governance references remain;
* focused and full tests pass.

## Recommended next sprint

Recommended next sprint:

```text
BL95 — Archive reporting, messaging, and delivery script-era modules after final no-active-reference check
```

Candidate archive targets:

```text
scripts/reporting/build_reporting_layer.py
scripts/reporting/build_telegram_summary.py
scripts/reporting/send_telegram.py
scripts/telegram/process_telegram_commands.py
```

BL95 must perform a final no-active-reference check before moving files to:

```text
archive/legacy_runtime/scripts/reporting/
archive/legacy_runtime/scripts/telegram/
```

## Guardrails

* No live provider calls were run.
* No yfinance calls were run.
* No SEC/EDGAR calls were run.
* No credentials were read.
* No production data was written.
* No production reports were generated.
* No Telegram messages were sent.
* No portfolio/watchlist state was modified.
* No Decision Engine authority was changed.
* No script-era reporting module was executed.
* No script-era Telegram module was executed.
* No script-era runtime module was archived.
* No script-era runtime behavior was modified.
