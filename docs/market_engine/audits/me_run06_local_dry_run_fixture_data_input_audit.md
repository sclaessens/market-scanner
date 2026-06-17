# ME-RUN06 - Local dry-run fixture/data input audit

## Status

COMPLETED BY ME-RUN06

## Scope audited

ME-RUN06 implemented a controlled local JSON fixture/data input path for the Market Engine end-to-end dry-run command.

Audited runtime files:

```text
src/market_engine/run/local_dry_run_inputs.py
src/market_engine/run/end_to_end_dry_run_command.py
src/market_engine/run/__init__.py
```

Audited tests:

```text
tests/market_engine/run/test_local_dry_run_inputs.py
tests/market_engine/run/test_end_to_end_dry_run_command.py
```

Audited documentation:

```text
docs/market_engine/run/me_run06_local_dry_run_fixture_data_input_implementation.md
```

## Implementation audit

ME-RUN06 introduced `market-engine-local-dry-run-input-fixture-v1` as the required local non-production wrapper for `local_snapshot_fixture` command input.

The loader:

* reads only caller-supplied local JSON;
* requires top-level JSON objects;
* requires `dry_run_input_fixture_format_version` for local snapshot fixtures;
* requires `non_production_fixture: true` for local snapshot fixtures;
* requires `stage_payloads` to be an object;
* allows raw stage payload mappings only for `explicit_in_memory_payload` compatibility;
* returns stage payload mappings to the existing end-to-end dry-run builder;
* raises `LocalDryRunInputError` for invalid local input.

The command:

* keeps embedded `synthetic_contract_fixture` as the default;
* routes `--stage-payloads-json` through the new loader;
* supports controlled `local_snapshot_fixture` execution;
* preserves raw `explicit_in_memory_payload` execution;
* returns exit code `2` for local input failures;
* continues to emit stdout JSON only after successful dry-run construction.

## Boundary audit

Confirmed preserved boundaries:

* no provider calls;
* no SEC/EDGAR calls;
* no live market data calls;
* no broker calls;
* no Telegram or email delivery;
* no production report generation;
* no portfolio writes;
* no watchlist writes;
* no scheduler behavior;
* no UI behavior;
* no Decision Engine decisions;
* no Recommendation Review or Portfolio Review behavior changes;
* no new financial analysis logic;
* no BUY / SELL / HOLD semantics;
* no allocation advice;
* no target weights or target prices;
* no position sizing;
* no order generation;
* no execution advice;
* no ranking, scoring, urgency, conviction, or tradeability authority.

## Numeric-zero and evidence audit

The loader does not filter, coerce, normalize, or drop values inside `stage_payloads`.

Numeric-zero values remain valid local fixture values and are passed through to the existing dry-run builder for contract inspection and summary behavior.

Missing-data markers, stale-data markers, blocked states, blocked reasons, provenance references, delivery report references, forbidden-side-effect confirmation, and authority-boundary confirmation remain governed by the existing `market-engine-end-to-end-dry-run-v1` builder.

## Test audit

Implemented tests verify:

* approved local snapshot fixture loading;
* numeric-zero preservation through local fixture loading;
* rejection of raw payloads in `local_snapshot_fixture` mode;
* rejection of missing non-production fixture markers;
* rejection of invalid `stage_payloads` wrappers;
* raw stage payload compatibility for `explicit_in_memory_payload`;
* fixture wrapper compatibility for `explicit_in_memory_payload`;
* command execution through `local_snapshot_fixture`;
* command failure on malformed JSON;
* command failure on invalid local snapshot fixture shape;
* import guardrails against legacy runtime and side-effect dependencies.

## Verification limitation

This audit was prepared from connector-backed repository changes. The execution environment used for this implementation could not clone the GitHub repository over the network, so the full local `pytest` suite was not executed here.

Recommended local verification:

```bash
python -m pytest tests/market_engine/run/test_local_dry_run_inputs.py tests/market_engine/run/test_end_to_end_dry_run_command.py | tee /dev/tty | pbcopy
```

## Outcome

ME-RUN06 is implementation-ready for review. It adds a controlled local fixture/data input execution path without opening any provider, broker, delivery, production-write, portfolio mutation, watchlist mutation, scheduling, Decision Engine action, allocation, ranking, scoring, urgency, conviction, or tradeability boundary.
