# ME-RUN25 - Operator-supplied cached-source snapshot import validation flow

Date: 2026-06-25

## Objective

ME-RUN25 validates whether the current Market Engine can accept an operator-supplied cached-source snapshot through the ME-SR12 import command, validate it with the ME-SR10 staging validator, and use the imported payload as a local `cached_source_snapshot` dry-run input.

This sprint is a run/validation sprint. It does not build a new ingestion system.

## Scope and Non-Goals

Scope:

* inspect the current ME-SR08 through ME-SR12 cached-source snapshot contracts;
* create a minimal explicitly non-production operator-supplied fixture under `/private/tmp`;
* run the ME-SR12 import command;
* run the ME-SR10 staging validator against the imported workspace;
* attempt the local `cached_source_snapshot` dry-run path;
* record pass/fail evidence, gaps, and next sprint recommendation.

Non-goals:

* no live provider calls;
* no yfinance;
* no SEC/EDGAR;
* no external network fetch;
* no Telegram send;
* no production writes;
* no production portfolio/watchlist mutation;
* no Decision Engine semantic changes;
* no real financial data was invented or presented as real.

## Base Commit

Current base/main commit inspected:

```text
f94f57c1c4411f5e9f145cb886b28ee704a7c63e
```

Recent merged context included:

```text
f94f57c Merge pull request #398 from sclaessens/me-sr12-operator-supplied-cached-source-snapshot-import-command
881f32f ME-SR12 add operator cached-source snapshot import command
fec54c1 Merge pull request #396 from sclaessens/me-sr11-cached-source-snapshot-acquisition-dry-run-command
477edef ME-SR11 add cached-source snapshot acquisition dry-run command
1cf42e4 Merge pull request #395 from sclaessens/me-sr10-manual-cached-source-snapshot-staging-validator
```

## Files and Commands Inspected

Relevant files inspected:

```text
docs/market_engine/audits/me_sr08_cached_source_snapshot_acquisition_manifest_contract.md
docs/market_engine/audits/me_sr10_manual_cached_source_snapshot_staging_validator_audit.md
docs/market_engine/audits/me_sr11_cached_source_snapshot_acquisition_dry_run_command_audit.md
docs/market_engine/audits/me_sr12_operator_supplied_cached_source_snapshot_import_command_audit.md
docs/market_engine/backlog/me_rm02_real_world_run_and_telegram_preview_backlog_lock.md
src/market_engine/source_refresh/cached_source_snapshot_import_command.py
src/market_engine/source_refresh/cached_source_snapshot_importer.py
src/market_engine/source_refresh/cached_source_snapshot_staging_validator_command.py
src/market_engine/source_refresh/cached_source_snapshot_staging_validator.py
src/market_engine/run/end_to_end_dry_run_command.py
src/market_engine/run/cached_source_execution.py
src/market_engine/source_refresh/sec_companyfacts_snapshots.py
src/market_engine/source_context/sec_companyfacts_context.py
tests/market_engine/source_refresh/test_cached_source_snapshot_importer.py
tests/market_engine/source_refresh/test_cached_source_snapshot_staging_validator.py
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
```

Discovery commands run:

```text
find docs/market_engine -maxdepth 4 -type f
find src tests data artifacts -maxdepth 5 -type f
rg -n "ME-SR12|manual cached-source|cached-source snapshot|cached_source_snapshot|snapshot staging|staging validator|operator-supplied|cached_source_snapshot_import" docs src scripts tests data
```

## Current Flow Identified

Expected operator-supplied import input:

```text
<operator-supplied-snapshot-dir>/manifest.json
<operator-supplied-snapshot-dir>/<payload file referenced by manifest local_snapshot_path>
```

The import command also accepts a direct `manifest.json` path.

Required manifest contract:

```text
market-engine-cached-source-snapshot-acquisition-manifest-v1
```

Destination layout:

```text
<destination-root>/<batch_id>/<ticker>/<snapshot_id>/
```

Default destination root:

```text
data/market_engine/cached_source_snapshots
```

Validation command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.source_refresh.cached_source_snapshot_staging_validator_command --staging-root <path>
```

Import command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.source_refresh.cached_source_snapshot_import_command --source-path <path>
```

Local dry-run input mode:

```text
cached_source_snapshot
```

Dry-run command shape used:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode cached_source_snapshot --source-snapshot-json <payload.json> --source-snapshot-root <root>
```

## Fixture Used

No existing real operator-supplied sample was available in the repo. ME-RUN25 created a temporary non-production fixture under:

```text
/private/tmp/me-run25-operator-supplied-flow/operator_supplied/snapshot
```

The fixture contained:

```text
manifest.json
payload.json
```

The manifest explicitly marked the material as non-production in `source_license_note`, `validation_warnings`, and `notes`. The payload used the SEC CompanyFacts raw snapshot envelope shape so the existing `cached_source_snapshot` dry-run path could load it:

```text
metadata
raw_payload
```

Payload SHA-256 observed:

```text
bee3800cf9d5b43a8744d93e7ffe3b27e589365b455f1f6d3bf44e4cf420a80d
```

The fixture is temporary run evidence only and was not committed.

## Commands Run and Results

### 1. Create non-production operator fixture

Command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c '<fixture creation script>'
```

Output summary:

```text
source=/private/tmp/me-run25-operator-supplied-flow/operator_supplied/snapshot
destination_root=/private/tmp/me-run25-operator-supplied-flow/cached_source_snapshots
dry_run_artifact_root=/private/tmp/me-run25-operator-supplied-flow/dry_run_artifacts
payload_sha256=bee3800cf9d5b43a8744d93e7ffe3b27e589365b455f1f6d3bf44e4cf420a80d
```

### 2. Import operator-supplied snapshot

Command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.source_refresh.cached_source_snapshot_import_command --source-path /private/tmp/me-run25-operator-supplied-flow/operator_supplied/snapshot --destination-root /private/tmp/me-run25-operator-supplied-flow/cached_source_snapshots --validated-at 2026-06-25T13:31:00Z
```

Result:

```text
exit code: 0
Snapshot ID: NVDA_companyfacts
Batch ID: me-run25-operator-supplied-fixture-20260625T133000Z
Source family: sec_companyfacts
Destination path: /private/tmp/me-run25-operator-supplied-flow/cached_source_snapshots/me-run25-operator-supplied-fixture-20260625T133000Z/NVDA/NVDA_companyfacts
Manifest: /private/tmp/me-run25-operator-supplied-flow/cached_source_snapshots/me-run25-operator-supplied-fixture-20260625T133000Z/NVDA/NVDA_companyfacts/manifest.json
Imported entities: NVDA
Validation: accepted
Warnings: none
```

### 3. Validate imported staging root

Command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.source_refresh.cached_source_snapshot_staging_validator_command --staging-root /private/tmp/me-run25-operator-supplied-flow/cached_source_snapshots --validated-at 2026-06-25T13:32:00Z --human
```

Result:

```text
exit code: 0
Report format: market-engine-cached-source-snapshot-staging-validation-v1
Counts: total=1 accepted=1 rejected=0 missing_manifest=0 malformed_manifest=0 unknown_format=0 missing_referenced_file=0 hash_mismatch=0 size_mismatch=0 stale=0 fixture_or_test=0 validation_status_blocked=0 usable_flag_conflict=0
Entry: NVDA | NVDA_companyfacts | sec_companyfacts | accepted | issues=none
```

### 4. Attempt local dry-run without portfolio context

Command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode cached_source_snapshot --source-snapshot-json /private/tmp/me-run25-operator-supplied-flow/cached_source_snapshots/me-run25-operator-supplied-fixture-20260625T133000Z/NVDA/NVDA_companyfacts/payload.json --source-snapshot-root /private/tmp/me-run25-operator-supplied-flow/cached_source_snapshots --dry-run-id me-run25-nvda-operator-fixture-dry-run --generated-at 2026-06-25T13:33:00Z --compact
```

Result:

```text
exit code: 0
input_mode: cached_source_snapshot
ticker: NVDA
cik: 0001045810
blocked_stage: portfolio_review
missing_data_summary: portfolio_context
```

Assessment: the imported payload fed the dry-run path, but the run was correctly blocked by missing local portfolio context.

### 5. Create non-production portfolio context

Command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c '<portfolio context fixture creation script>'
```

Output:

```text
/private/tmp/me-run25-operator-supplied-flow/portfolio_context.json
```

### 6. Run local dry-run with portfolio context

Command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.end_to_end_dry_run_command --input-mode cached_source_snapshot --source-snapshot-json /private/tmp/me-run25-operator-supplied-flow/cached_source_snapshots/me-run25-operator-supplied-fixture-20260625T133000Z/NVDA/NVDA_companyfacts/payload.json --source-snapshot-root /private/tmp/me-run25-operator-supplied-flow/cached_source_snapshots --portfolio-context-json /private/tmp/me-run25-operator-supplied-flow/portfolio_context.json --dry-run-id me-run25-nvda-operator-fixture-dry-run-with-context --generated-at 2026-06-25T13:35:00Z --compact
```

Result:

```text
exit code: 0
input_mode: cached_source_snapshot
ticker: NVDA
cik: 0001045810
blocked_stage: null
run_state: dry_run_completed
source_context_state: AVAILABLE
delivery_report_reference.source_handoff_run_id: me-run25-nvda-operator-fixture-dry-run-with-context-decision-engine-handoff
```

Assessment: with non-production portfolio context, the imported payload completed the existing local dry-run path.

## Artifacts Produced

Temporary local files were produced under:

```text
/private/tmp/me-run25-operator-supplied-flow
```

No repository-tracked artifacts were produced. No files under the existing untracked repository `artifacts/` directory were modified or committed.

No production data under `data/market_engine/cached_source_snapshots` was written because the run used an explicit `/private/tmp` destination root.

## Usability Answer

Can the current Market Engine accept an operator-supplied cached-source snapshot through the manual staging/import validation flow?

```text
YES, for one ME-SR08-compatible snapshot directory or direct manifest path.
```

Can the staged/imported snapshot be used as input for a local dry-run?

```text
YES, when the imported payload is a SEC CompanyFacts raw snapshot envelope and the dry-run is pointed at the imported payload path with source_snapshot_root set to the import root.
```

Can it complete the local dry-run path?

```text
YES, when local portfolio context is supplied. Without portfolio context, the run blocks at portfolio_review as expected.
```

## Contract Gaps Discovered

* ME-SR12 validates the ME-SR08 manifest and local payload integrity, but it does not independently validate that the payload content is a SEC CompanyFacts raw snapshot envelope before import. The dry-run path later validates that content when loading `payload.json`.
* The import command does not emit the next dry-run command or a cached-source local execution wrapper after import. The operator must manually derive `--source-snapshot-json` and `--source-snapshot-root`.
* Non-production fixture material can pass staging validation when `source_material_type=operator_supplied_material` and `validation_status=passed`. This is acceptable for ME-RUN25 because the audit marks the fixture as non-production, but real source validation should rely on real operator-supplied files in the next sprint.

## Runtime and Tooling Gaps Discovered

* The terminal `--compact` dry-run output is extremely large for operator review. A Telegram-style terminal preview remains needed to make real-world output inspectable quickly.
* The local dry-run path can consume an imported payload, but the import command and dry-run command are not yet connected by a single operator workflow.

## Documentation Gaps Discovered

* Existing docs describe ME-SR12 import and ME-SR10 validation separately. ME-RUN25 now records the exact command bridge from imported snapshot directory to `cached_source_snapshot` dry-run.
* Existing roadmap lock still names a real-world sample import sprint after ME-SR12. ME-RUN25 confirms that sprint remains necessary, but it should use real operator-supplied local files rather than another fixture.

## Safety Assessment

Safety result: PASS.

Observed boundaries:

* no provider calls;
* no network calls;
* no yfinance;
* no SEC/EDGAR calls;
* no Telegram sends;
* no broker calls;
* no production portfolio/watchlist writes;
* no production cached-source snapshot writes;
* no Decision Engine semantic changes;
* no BUY, SELL, HOLD, ranking, scoring, urgency, conviction, allocation, sizing, order, execution, or tradeability authority added.

## Conclusion

```text
PASS
```

ME-RUN25 proves that the current Market Engine can import a single ME-SR08-compatible operator-supplied cached-source snapshot into a local workspace, validate it with the ME-SR10 staging validator, and feed the imported SEC CompanyFacts raw payload into the existing `cached_source_snapshot` dry-run path.

The result is fixture-backed and non-production. It does not prove real-world source quality yet.

## Recommended Next Sprint

Recommended next sprint:

```text
ME-SR13 - Run real-world operator-supplied cached-source sample import for NVDA, AMD, ASML
```

ME-SR13 should use real local operator-supplied files, run the ME-SR12 import command, run ME-SR10 staging validation, and attempt the same dry-run bridge for accepted samples. It should preserve the roadmap trajectory toward:

* real-world cached-source validation;
* first real cached-source Market Engine analysis;
* Telegram-style terminal preview from real analysis output.
