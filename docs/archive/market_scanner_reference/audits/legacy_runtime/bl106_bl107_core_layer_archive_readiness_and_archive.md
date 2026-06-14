# BL106 / BL107 Core Layer Archive Readiness and Archive

## BL106 — Final archive-readiness review for fail-closed core layer modules

Scope:

* `scripts/core/build_context_layer.py`
* `scripts/core/build_validation_layer.py`
* `scripts/core/build_timing_state_layer.py`
* `scripts/core/build_stability_layer.py`

Out-of-scope areas were not modified: Decision Engine authority, scanner/provider runtime, SEC/EDGAR integrations, yfinance, live provider calls, credentials, production data writes, report generation, Telegram delivery, portfolio state, watchlist state, scan validation runtime, portfolio intelligence, and unrelated `scripts/` modules.

### Commands executed

```bash
grep -RInE \
  "build_context_layer|build_validation_layer|build_timing_state_layer|build_stability_layer" \
  src tests .github docs \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Relevant output:

```text
tests/core/test_build_stability_layer.py:5:LEGACY_STABILITY_LAYER_MODULE_PATH = Path("scripts/core/build_stability_layer.py")
tests/core/test_build_context_layer.py:5:LEGACY_CONTEXT_LAYER_MODULE_PATH = Path("scripts/core/build_context_layer.py")
tests/core/test_entry_quality.py:5:LEGACY_ENTRY_QUALITY_OWNER_PATH = Path("scripts/core/build_validation_layer.py")
tests/core/test_build_timing_state_layer.py:5:LEGACY_TIMING_LAYER_MODULE_PATH = Path("scripts/core/build_timing_state_layer.py")
tests/core/test_build_validation_layer.py:5:LEGACY_VALIDATION_LAYER_MODULE_PATH = Path("scripts/core/build_validation_layer.py")
docs/archive/... historical references to the four script-era module paths
docs/legacy/... historical references to the four script-era module paths
docs/resets/... historical references to the four script-era module paths
docs/active/project/backlog.md: existing BL102-BL105 backlog references
```

Conclusion: no active `src/`, `.github`, or active runtime path depends on the four target modules. Remaining references are static tests, negative import guardrails, backlog entries, audit notes, historical archive documents, and legacy documentation.

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core|import scripts\.core)" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Output:

```text
tests/core/test_build_entry_quality_backfill.py:5:from scripts.core.build_entry_quality_backfill import (
tests/core/test_build_context_backfill.py:9:from scripts.core import build_context_backfill as b
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
tests/core/test_build_portfolio_intelligence.py:9:from scripts.core import build_portfolio_intelligence as portfolio_module
tests/portfolio/test_portfolio_source_contract.py:8:from scripts.core import build_portfolio_intelligence
```

Conclusion: remaining active `scripts.core` imports do not reference the four BL106 target modules and are outside the sprint scope.

```bash
grep -RInE \
  "FAIL_CLOSED|if __name__|def main|to_csv|write_text|mkdir|read_csv|Path[(]" \
  scripts/core/build_context_layer.py \
  scripts/core/build_validation_layer.py \
  scripts/core/build_timing_state_layer.py \
  scripts/core/build_stability_layer.py
```

Relevant output:

```text
scripts/core/build_context_layer.py:9:SCANNER_PATH = Path("data/processed/scanner_ranked.csv")
scripts/core/build_context_layer.py:30:        df = pd.read_csv(path)
scripts/core/build_context_layer.py:116:    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
scripts/core/build_context_layer.py:121:    log_df.to_csv(LOG_PATH, index=False)
scripts/core/build_context_layer.py:155:if __name__ == "__main__":
scripts/core/build_context_layer.py:157:        "FAIL_CLOSED: scripts/core/build_context_layer.py is a legacy script-era module. "
scripts/core/build_validation_layer.py:8:INPUT_PATH = Path("data/processed/scanner_ranked.csv")
scripts/core/build_validation_layer.py:49:        df = pd.read_csv(INPUT_PATH)
scripts/core/build_validation_layer.py:215:    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
scripts/core/build_validation_layer.py:230:    validation_df.to_csv(OUTPUT_PATH, index=False)
scripts/core/build_validation_layer.py:239:if __name__ == "__main__":
scripts/core/build_validation_layer.py:241:        "FAIL_CLOSED: scripts/core/build_validation_layer.py is a legacy script-era module. "
scripts/core/build_timing_state_layer.py:9:INPUT_PATH = Path("data/processed/fundamental_quality.csv")
scripts/core/build_timing_state_layer.py:92:        df = pd.read_csv(path)
scripts/core/build_timing_state_layer.py:285:    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
scripts/core/build_timing_state_layer.py:304:    output_df.to_csv(OUTPUT_PATH, index=False)
scripts/core/build_timing_state_layer.py:311:if __name__ == "__main__":
scripts/core/build_timing_state_layer.py:313:        "FAIL_CLOSED: scripts/core/build_timing_state_layer.py is a legacy script-era module. "
scripts/core/build_stability_layer.py:10:PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
scripts/core/build_stability_layer.py:106:        df = pd.read_csv(path)
scripts/core/build_stability_layer.py:355:    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
scripts/core/build_stability_layer.py:357:    output_df.to_csv(OUTPUT_PATH, index=False)
scripts/core/build_stability_layer.py:362:def main() -> None:
scripts/core/build_stability_layer.py:364:        "FAIL_CLOSED: scripts/core/build_stability_layer.py is a legacy script-era module. "
scripts/core/build_stability_layer.py:369:if __name__ == "__main__":
```

Conclusion: historical read/write bodies remain preserved, while manual execution remains fail-closed.

```bash
pytest tests/core/test_build_context_layer.py \
       tests/core/test_build_validation_layer.py \
       tests/core/test_entry_quality.py \
       tests/core/test_build_timing_state_layer.py \
       tests/core/test_build_stability_layer.py \
       tests/test_operator_visibility.py -q
```

Output:

```text
zsh:1: command not found: pytest
```

Equivalent repository-virtualenv command executed:

```bash
.venv/bin/python -m pytest tests/core/test_build_context_layer.py tests/core/test_build_validation_layer.py tests/core/test_entry_quality.py tests/core/test_build_timing_state_layer.py tests/core/test_build_stability_layer.py tests/test_operator_visibility.py -q
```

Output:

```text
35 passed in 0.06s
```

```bash
pytest -q
```

Output:

```text
zsh:1: command not found: pytest
```

Equivalent repository-virtualenv command executed:

```bash
.venv/bin/python -m pytest -q
```

Output:

```text
610 passed in 0.62s
```

### BL106 decision

Decision: `BL107 archive sprint approved`

Rationale:

* no active imports from `src/`, `tests/`, or `.github` depend on the four target modules;
* grep results show no active runtime path depending on the four target modules;
* remaining references are acceptable documentation, audit, historical, backlog, static test, or negative import-guard references;
* manual execution remains fail-closed;
* focused suite passed;
* full suite passed;
* no unrelated changes were required.

## BL107 — Controlled archive of fail-closed core layer modules

Archived paths:

* `archive/legacy_runtime/scripts/core/build_context_layer.py`
* `archive/legacy_runtime/scripts/core/build_validation_layer.py`
* `archive/legacy_runtime/scripts/core/build_timing_state_layer.py`
* `archive/legacy_runtime/scripts/core/build_stability_layer.py`

The files were moved with `git mv` from `scripts/core/` to `archive/legacy_runtime/scripts/core/`, preserving historical source content.

### Post-archive commands executed

```bash
grep -RInE \
  "build_context_layer|build_validation_layer|build_timing_state_layer|build_stability_layer" \
  src tests .github docs scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Relevant output:

```text
tests/core/test_build_stability_layer.py:5:LEGACY_STABILITY_LAYER_MODULE_PATH = Path("scripts/core/build_stability_layer.py")
tests/core/test_build_context_layer.py:5:LEGACY_CONTEXT_LAYER_MODULE_PATH = Path("scripts/core/build_context_layer.py")
tests/core/test_entry_quality.py:5:LEGACY_ENTRY_QUALITY_OWNER_PATH = Path("scripts/core/build_validation_layer.py")
tests/core/test_build_timing_state_layer.py:5:LEGACY_TIMING_LAYER_MODULE_PATH = Path("scripts/core/build_timing_state_layer.py")
tests/core/test_build_validation_layer.py:5:LEGACY_VALIDATION_LAYER_MODULE_PATH = Path("scripts/core/build_validation_layer.py")
docs/... documentation, audit, backlog, historical, and legacy references
```

Conclusion: post-archive references are static tests, documentation, audit, backlog, historical, or legacy references. No active `scripts/` runtime module references the archived module names.

```bash
grep -RInE \
  "^[[:space:]]*(from scripts\.core|import scripts\.core)" \
  src tests .github scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Output:

```text
tests/core/test_build_entry_quality_backfill.py:5:from scripts.core.build_entry_quality_backfill import (
tests/core/test_build_context_backfill.py:9:from scripts.core import build_context_backfill as b
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
tests/core/test_build_portfolio_intelligence.py:9:from scripts.core import build_portfolio_intelligence as portfolio_module
tests/portfolio/test_portfolio_source_contract.py:8:from scripts.core import build_portfolio_intelligence
```

Conclusion: remaining active `scripts.core` imports do not reference the archived BL107 modules and are outside the controlled archive scope.

```bash
.venv/bin/python -m pytest tests/core/test_build_context_layer.py tests/core/test_build_validation_layer.py tests/core/test_entry_quality.py tests/core/test_build_timing_state_layer.py tests/core/test_build_stability_layer.py tests/test_operator_visibility.py -q
```

Output:

```text
35 passed in 0.05s
```

```bash
.venv/bin/python -m pytest -q
```

Output:

```text
610 passed in 0.64s
```

### BL107 decision

Decision: `CONTROLLED_CORE_LAYER_ARCHIVE_COMPLETED`

Canonical runtime confirmation:

* no files under `src/market_scanner/` were changed;
* Decision Engine authority was not modified;
* no live provider calls, yfinance calls, SEC/EDGAR calls, credential reads, production data writes, report generation, Telegram delivery, portfolio state changes, watchlist state changes, or scan validation runtime changes were performed.

