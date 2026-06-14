# BL95 — Archive reporting, messaging, and delivery script-era modules

Status: COMPLETED

## Purpose

BL95 archives the remaining script-era reporting, messaging, and delivery modules after BL94 decoupled active tests and canonical boundary metadata from them.

Targeted script-era modules:

```text
scripts/reporting/build_reporting_layer.py
scripts/reporting/build_telegram_summary.py
scripts/reporting/send_telegram.py
scripts/telegram/process_telegram_commands.py
```

BL95 is an archive sprint. It moves the targeted files into `archive/legacy_runtime/` and does not modify their runtime behavior.

## Pre-archive checks

### Target files before archive

BL95 confirmed the active script-era files before archive:

```text
scripts/reporting/build_reporting_layer.py
scripts/reporting/build_telegram_summary.py
scripts/reporting/send_telegram.py
scripts/telegram/process_telegram_commands.py
```

### Active import check

BL95 checked for active imports from script-era reporting and Telegram modules:

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

### Active path reference check

BL95 checked active path references for the targeted script-era files:

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
* they are not imports, static file reads, runtime dependencies, or execution paths.

## Archived files

BL95 moved the targeted files:

```text
scripts/reporting/build_reporting_layer.py -> archive/legacy_runtime/scripts/reporting/build_reporting_layer.py
scripts/reporting/build_telegram_summary.py -> archive/legacy_runtime/scripts/reporting/build_telegram_summary.py
scripts/reporting/send_telegram.py -> archive/legacy_runtime/scripts/reporting/send_telegram.py
scripts/telegram/process_telegram_commands.py -> archive/legacy_runtime/scripts/telegram/process_telegram_commands.py
```

The archive location now contains:

```text
archive/legacy_runtime/scripts/reporting/build_reporting_layer.py
archive/legacy_runtime/scripts/reporting/build_telegram_summary.py
archive/legacy_runtime/scripts/reporting/reporter.py
archive/legacy_runtime/scripts/reporting/send_telegram.py
archive/legacy_runtime/scripts/telegram/process_telegram_commands.py
```

Note:

* `archive/legacy_runtime/scripts/reporting/reporter.py` was already archived in an earlier sprint.

## Post-archive active folder check

BL95 confirmed that active `scripts/reporting` and `scripts/telegram` no longer contain Python files.

Result:

```text
No active scripts/reporting/*.py or scripts/telegram/*.py files remain.
```

## Operator note

During validation, the archived file paths were accidentally pasted into the terminal as shell commands. The shell produced `permission denied`, `command not found`, and syntax errors.

Interpretation:

* the files were not executed through Python;
* no Telegram delivery occurred;
* no credentials were read;
* no production data or reports were written;
* the validated test results were run after the archive move.

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
45 passed in 0.03s
```

Full suite:

```bash
pytest -q
```

Result:

```text
560 passed in 0.53s
```

## Decision

BL95 decision:

```text
ARCHIVED
```

The targeted script-era reporting, messaging, and delivery modules are now archived under:

```text
archive/legacy_runtime/scripts/reporting/
archive/legacy_runtime/scripts/telegram/
```

## Impact

After BL95:

* active `scripts/reporting/*.py` no longer exists;
* active `scripts/telegram/*.py` no longer exists;
* active tests no longer import `scripts.reporting` or `scripts.telegram`;
* canonical reporting, messaging, and delivery boundaries no longer list the targeted active script-era paths as authorities;
* historical script-era implementation remains preserved under `archive/legacy_runtime/`.

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
* No script-era reporting module was modified.
* No script-era Telegram module was modified.
* Files were archived, not deleted.

## Recommended next sprint

Recommended next sprint:

```text
BL96 — Review remaining active scripts tree after reporting and Telegram archive
```

Goal:

* inspect remaining active `scripts/**/*.py` files after BL92 and BL95;
* identify which domains remain coupled to active tests or canonical metadata;
* prioritize the next domain-specific decoupling sprint;
* avoid runtime execution;
* avoid production writes or provider calls.
