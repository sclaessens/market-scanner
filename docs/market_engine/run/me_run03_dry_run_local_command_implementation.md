# ME-RUN03 - Dry-run local command implementation

## Status

COMPLETED BY ME-RUN03

## Sprint

ME-RUN03 - Wire dry-run harness into runnable local command

## Job family

ME-RUN - Run / orchestration jobs

## Purpose

ME-RUN03 wires the ME-RUN02 deterministic end-to-end dry-run harness into a runnable local command.

The command is intended for local integration review only. It lets the operator run the approved dry-run summary boundary from the terminal and inspect the emitted `market-engine-end-to-end-dry-run-v1` JSON payload without provider access, broker access, Telegram/email delivery, production report writes, portfolio mutation, watchlist mutation, scheduler behavior, UI behavior, or trading/allocation authority.

## Implemented runtime

```text
src/market_engine/run/end_to_end_dry_run_command.py
pyproject.toml
```

`src/market_engine/run/end_to_end_dry_run_command.py` provides:

* `run_market_engine_end_to_end_dry_run_command(...)` for tested command execution;
* `main()` for console-script and `python -m` execution;
* `build_synthetic_dry_run_stage_payloads()` for the deterministic embedded synthetic fixture;
* local JSON loading for explicitly supplied stage payload mappings.

`pyproject.toml` now exposes:

```text
market-engine-dry-run = market_engine.run.end_to_end_dry_run_command:main
```

## Runnable local commands

From a local checkout with the package installed or `src` on `PYTHONPATH`, the deterministic synthetic command can be run as:

```bash
python -m market_engine.run.end_to_end_dry_run_command \
  --dry-run-id local-run-001 \
  --generated-at 2026-06-17T13:30:00Z
```

When installed as an editable package, the console script can be run as:

```bash
market-engine-dry-run \
  --dry-run-id local-run-001 \
  --generated-at 2026-06-17T13:30:00Z
```

For explicit local JSON stage payloads:

```bash
market-engine-dry-run \
  --input-mode explicit_in_memory_payload \
  --stage-payloads-json path/to/stage_payloads.json \
  --dry-run-id explicit-run-001
```

For compact single-line JSON output:

```bash
market-engine-dry-run --compact
```

## Input behavior

The command supports the approved ME-RUN01 / ME-RUN02 input modes:

* `synthetic_contract_fixture`;
* `local_snapshot_fixture`;
* `explicit_in_memory_payload`.

If no `--stage-payloads-json` path is supplied, only `synthetic_contract_fixture` is runnable. Non-synthetic modes require an explicit local JSON file so the command does not silently invent local snapshot or caller-supplied payloads.

Malformed JSON, unreadable JSON files, or non-object top-level JSON fail closed at command level with exit code `2` and no dry-run payload on stdout.

## Output behavior

Successful command execution prints the serialized `market-engine-end-to-end-dry-run-v1` payload to stdout as JSON.

The command does not persist artifacts. It does not write to production data folders, generated report folders, broker-connected folders, Telegram/email queues, portfolio state, watchlist state, scheduler state, or UI state.

## Implemented tests

```text
tests/market_engine/run/test_end_to_end_dry_run_command.py
```

Tests cover:

* synthetic fixture command output;
* explicit local JSON payload input;
* fail-closed non-synthetic mode without JSON payload file;
* fail-closed malformed JSON file;
* import guardrails against legacy runtime and side-effect dependencies.

## Forbidden behavior preserved

ME-RUN03 does not introduce provider calls, SEC/EDGAR calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine decisions, new financial analysis logic, BUY / SELL / HOLD semantics, allocation advice, target weights, target prices, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability.

## Outcome

ME-RUN03 makes the ME-RUN02 dry-run harness locally runnable from the terminal while preserving the approved dry-run boundary. The sprint converts the harness from a tested Python builder into an operator-facing local command that emits inspectable JSON and remains deterministic, non-live, non-delivering, non-mutating, and non-actionable.
