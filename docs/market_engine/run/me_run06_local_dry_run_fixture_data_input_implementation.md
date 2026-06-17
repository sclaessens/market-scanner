# ME-RUN06 - Local dry-run fixture/data input execution path

## Status

COMPLETED BY ME-RUN06

## Sprint

ME-RUN06 - Implement local dry-run fixture/data input execution path

## Job family

ME-RUN - Run / orchestration jobs

## Purpose

ME-RUN06 implements a controlled local JSON input path for the Market Engine end-to-end dry-run command.

The sprint makes `local_snapshot_fixture` usable through an explicit local non-production fixture wrapper while preserving the existing embedded synthetic fixture path and the existing raw `explicit_in_memory_payload` test path.

The implementation does not fetch, enrich, normalize, repair, reinterpret, or execute upstream Market Engine stages. It only loads approved local JSON and passes approved stage payload mappings into the existing `market-engine-end-to-end-dry-run-v1` builder.

## Implemented runtime

```text
src/market_engine/run/local_dry_run_inputs.py
src/market_engine/run/end_to_end_dry_run_command.py
src/market_engine/run/__init__.py
```

`src/market_engine/run/local_dry_run_inputs.py` provides:

* `MARKET_ENGINE_LOCAL_DRY_RUN_INPUT_FIXTURE_FORMAT_VERSION`;
* `LocalDryRunInputError`;
* `load_market_engine_local_dry_run_input(...)`.

## Approved local fixture wrapper

`local_snapshot_fixture` input now requires a local JSON wrapper with:

```text
market-engine-local-dry-run-input-fixture-v1
```

Required wrapper fields:

* `dry_run_input_fixture_format_version`;
* `non_production_fixture: true`;
* `stage_payloads` as an object.

Optional wrapper metadata:

* `fixture_id`;
* `input_mode: local_snapshot_fixture`.

If `input_mode` is present in the wrapper, it must equal `local_snapshot_fixture`.

## Command behavior

The command continues to support the embedded synthetic fixture by default:

```bash
python -m market_engine.run.end_to_end_dry_run_command
```

The command continues to support explicit raw stage payload JSON for test-oriented in-memory input:

```bash
python -m market_engine.run.end_to_end_dry_run_command \
  --input-mode explicit_in_memory_payload \
  --stage-payloads-json path/to/stage_payloads.json
```

The new approved local snapshot fixture path is:

```bash
python -m market_engine.run.end_to_end_dry_run_command \
  --input-mode local_snapshot_fixture \
  --stage-payloads-json path/to/local_snapshot_fixture.json
```

The command still prints the dry-run payload to stdout. Optional local artifact persistence remains controlled by ME-RUN05 through `--write-local-artifact`.

## Failure behavior

The input loader fails closed when:

* the JSON file cannot be read;
* the JSON is malformed;
* the top-level JSON value is not an object;
* `local_snapshot_fixture` input omits the fixture wrapper contract;
* the fixture wrapper does not explicitly set `non_production_fixture: true`;
* `stage_payloads` is missing or is not an object;
* the wrapper `input_mode`, when present, does not match `local_snapshot_fixture`;
* a caller tries to use local JSON input with an unsupported input mode.

Command failures return exit code `2` and do not emit stdout payload JSON.

## Implemented tests

```text
tests/market_engine/run/test_local_dry_run_inputs.py
tests/market_engine/run/test_end_to_end_dry_run_command.py
```

Tests cover:

* approved local snapshot fixture loading;
* preservation of numeric-zero fixture values;
* rejection of raw payloads in `local_snapshot_fixture` mode;
* rejection of fixture wrappers without the non-production marker;
* rejection of invalid `stage_payloads` wrappers;
* raw stage payload compatibility for `explicit_in_memory_payload`;
* fixture-wrapper compatibility for `explicit_in_memory_payload`;
* command execution with `local_snapshot_fixture`;
* command rejection of malformed JSON and invalid local snapshot payloads;
* import guardrails against legacy runtime and side-effect dependencies.

## Boundary preservation

ME-RUN06 does not introduce provider calls, SEC/EDGAR calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine decisions, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, BUY / SELL / HOLD semantics, allocation advice, target weights, target prices, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability.

## Outcome

ME-RUN06 gives the local dry-run command a controlled fixture/data input execution path. Operators and developers can now run the existing end-to-end dry-run builder against explicit local non-production snapshot fixtures without opening provider, broker, production-write, delivery, or action-authority boundaries.
