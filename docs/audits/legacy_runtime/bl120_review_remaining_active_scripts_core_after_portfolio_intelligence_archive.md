# BL120 — Review remaining active scripts/core tree after portfolio intelligence archive

Status: completed
Sprint type: review-only / inventory
Archive action performed: no

## Scope

This review inspected the remaining active `scripts/core/` tree after BL119 archived the fail-closed portfolio intelligence module.

BL120 did not archive, move, delete, or modify runtime code.

## Out of scope

This sprint did not touch:

* archive moves
* `scripts/core/decision_engine.py`
* Decision Engine authority
* scanner/provider runtime
* SEC/EDGAR integrations
* yfinance execution
* credentials
* production data writes
* report generation
* Telegram delivery
* portfolio state
* watchlist state
* scan validation runtime
* trade command parser
* portfolio command processing
* any other runtime behavior

## Background

BL119 moved:

* `scripts/core/build_portfolio_intelligence.py`

to:

* `archive/legacy_runtime/scripts/core/build_portfolio_intelligence.py`

using `git mv`.

BL120 reviewed the remaining active `scripts/core/` Python files and classified the next cleanup direction.

## Commands executed

### Branch setup

```bash id="tjv6it"
git checkout main
git pull origin main
git status
git checkout -b bl120-review-remaining-active-scripts-core-after-portfolio-intelligence-archive
git status
```

### Remaining active scripts/core inventory

```bash id="qwbekl"
find scripts/core -maxdepth 1 -type f -name "*.py" | sort
```

### Active scripts.core imports

```bash id="p5y3yo"
grep -RInE \
  "^[[:space:]]*(from scripts\.core|import scripts\.core)" \
  tests src .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

### Side-effect / runtime markers

```bash id="6uzt08"
grep -RInE \
  "FAIL_CLOSED|if __name__|def main|argparse|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs|yfinance|yf\.|requests|telegram|send_message|os\.environ|dotenv" \
  scripts/core \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

### Broad references

```bash id="m7fbqe"
grep -RInE \
  "scripts/core|scripts\.core" \
  src tests .github scripts docs/active docs/audits \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

### Operator visibility

```bash id="viiyzn"
.venv/bin/python -m pytest tests/test_operator_visibility.py -q
```

### Full suite

```bash id="refhdb"
.venv/bin/python -m pytest -q
```

## Remaining active scripts/core files

BL120 confirmed that the active `scripts/core/` tree now contains 7 Python files:

```text id="iw01aj"
scripts/core/data_fetcher.py
scripts/core/decision_engine.py
scripts/core/indicators.py
scripts/core/log_scans.py
scripts/core/scanner.py
scripts/core/validate_scans.py
scripts/core/validator.py
```

`build_portfolio_intelligence.py` is no longer present in the active `scripts/core/` tree.

## Active scripts.core imports

The active import scan returned only:

```text id="f384c8"
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
```

Conclusion:

* there is no active import from `src`, `tests`, or `.github` to `scripts.core.build_portfolio_intelligence`;
* the only remaining active positive `scripts.core` import is the Decision Engine test import.

## Side-effect and runtime marker findings

### `scripts/core/decision_engine.py`

Classification: `P0_DO_NOT_TOUCH_YET`

Findings:

* still actively imported by `tests/core/test_decision_engine.py`;
* statically referenced by canonical decision boundary metadata;
* contains `pd.read_csv(...)`;
* writes final decision output/log artifacts via `to_csv(...)`;
* creates output/log directories via `mkdir(...)`;
* defines `main()`;
* has direct execution via `if __name__ == "__main__"`.

Decision:

* do not archive;
* do not fail-close casually;
* reserve for dedicated Decision Engine authority migration/review.

### `scripts/core/data_fetcher.py`

Classification: `SCANNER_PROVIDER_REVIEW_REQUIRED`

Findings:

* contains `import yfinance as yf`;
* calls `yf.download(...)`;
* calls `yf.Ticker(...)`;
* statically referenced by canonical scanner boundary metadata.

Decision:

* do not execute;
* do not archive yet;
* review together with scanner/provider boundary.

### `scripts/core/scanner.py`

Classification: `SCANNER_PROVIDER_REVIEW_REQUIRED`

Findings:

* contains `import yfinance as yf`;
* calls `yf.Ticker(ticker).info`;
* statically referenced by canonical scanner boundary metadata;
* likely contains scanner/scoring/provider-adjacent behavior.

Decision:

* do not execute;
* do not archive yet;
* review together with scanner/provider boundary.

### `scripts/core/indicators.py`

Classification: `SCANNER_PURE_LOGIC_REVIEW_REQUIRED`

Findings:

* no active test import found;
* no write-risk marker found in the BL120 side-effect grep;
* likely belongs to scanner/provider or technical-indicator logic.

Decision:

* review together with scanner/provider boundary;
* determine whether it is pure logic, obsolete, or needs canonical parity.

### `scripts/core/log_scans.py`

Classification: `LOG_WRITE_REVIEW_REQUIRED`

Findings:

* fixed processed/log paths;
* `pd.read_csv(...)`;
* `mkdir(...)`;
* `to_csv(...)`;
* direct execution via `if __name__ == "__main__"`.

Decision:

* do not archive yet;
* handle in a later logging/validation/bootstrap cluster.

### `scripts/core/validate_scans.py`

Classification: `VALIDATION_WRITE_REVIEW_REQUIRED`

Findings:

* `read_csv_safe(...)`;
* `pd.read_csv(...)`;
* `mkdir(...)`;
* `to_csv(...)`;
* defines `main()`;
* direct execution via `if __name__ == "__main__"`.

Decision:

* do not archive yet;
* reconcile with validation/runtime ownership before fail-close/archive.

### `scripts/core/validator.py`

Classification: `BOOTSTRAP_WRITE_REVIEW_REQUIRED`

Findings:

* contains `path.mkdir(parents=True, exist_ok=True)`;
* no active test import found in BL120 output.

Decision:

* do not archive yet;
* review with logging/validation/bootstrap cluster.

## Validation

Operator visibility:

```text id="khhpkc"
8 passed
```

Full suite:

```text id="vf61fu"
667 passed
```

## Decision

BL120 does not approve an archive sprint.

Decision:

```text id="dr7aya"
BL121 archive sprint: NOT APPROVED
BL121 scanner/provider boundary review sprint: APPROVED
```

## Required next sprint

Approved next sprint:

```text id="5pd7w9"
BL121 — Scanner/provider boundary review for remaining script-era core scanner modules
```

BL121 must be review-only.

Recommended BL121 scope:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`
* `scripts/core/indicators.py`

BL121 must not execute yfinance, must not make provider calls, and must not modify runtime behavior.

## Final BL120 result

```text id="b9g568"
BL121 scanner/provider boundary review: APPROVED
```
