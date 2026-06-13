# BL122 — Archive-readiness review for script-era indicators helper

Status: completed
Sprint type: review-only / archive-readiness
Archive action performed: no

## Scope

BL122 reviewed only:

* `scripts/core/indicators.py`

BL122 did not archive, move, delete, fail-close, or modify runtime code.

## Out of scope

This sprint did not touch:

* archive moves
* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`
* scanner/provider runtime
* yfinance execution
* live provider calls
* SEC/EDGAR integrations
* credentials
* production data writes
* report generation
* Telegram delivery
* portfolio state
* watchlist state
* scan validation runtime
* Decision Engine authority
* trade command parser
* portfolio command processing
* any runtime behavior

## Background

BL121 reviewed the scanner/provider-adjacent script-era core modules:

* `scripts/core/data_fetcher.py`
* `scripts/core/scanner.py`
* `scripts/core/indicators.py`

BL121 found:

* `scripts/core/data_fetcher.py` contains yfinance provider calls and is not archive-ready;
* `scripts/core/scanner.py` contains yfinance sector lookup plus scanner/scoring/trade-plan semantics and is not archive-ready;
* `scripts/core/indicators.py` appears to be a pure pandas helper with no active import, no provider/network marker, and no write-risk marker.

BL122 performed a focused archive-readiness review for `scripts/core/indicators.py`.

## Commands executed

### Branch setup

```bash id="bkwlaf"
git checkout main
git pull origin main
git status
git checkout -b bl122-archive-readiness-review-script-era-indicators-helper
git status
```

### File existence and active import check

```bash id="dddvne"
test -f scripts/core/indicators.py

grep -RInE \
  "scripts\.core\.indicators|from scripts\.core import indicators|import scripts\.core\.indicators|from scripts\.core|import scripts\.core" \
  src tests .github scripts \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result summary:

* `scripts/core/indicators.py` exists.
* No active import references `scripts.core.indicators`.
* The only active positive `scripts.core` import remains:

```text id="z4k5ow"
tests/core/test_decision_engine.py:9:from scripts.core import decision_engine
```

Other matches were static string assertions preventing script-era imports.

### Broad references

```bash id="e3qcda"
grep -RInE \
  "scripts/core/indicators.py|scripts\.core\.indicators|add_indicators|MIN_REQUIRED_ROWS|MA20|MA50|MA200|ATR14|20D_HIGH|20D_LOW|AVG_VOL_20" \
  src tests .github scripts docs/active docs/audits \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result summary:

* `scripts/core/scanner.py` references indicator output columns such as:

  * `MA20`
  * `MA50`
  * `MA200`
  * `ATR14`
  * `20D_HIGH`
  * `20D_LOW`
  * `AVG_VOL_20`
* `scripts/core/scanner.py` does not import `scripts.core.indicators`.
* `scripts/core/indicators.py` defines the helper and generated indicator columns.
* Remaining references are in historical audits, backlog documentation, and separate watchlist/portfolio logic using similarly named columns.

Conclusion: indicator-column names are still part of scanner/watchlist vocabulary, but the script-era `scripts.core.indicators` helper is not actively imported.

### Provider/write/entrypoint markers

```bash id="qgybvw"
grep -nE \
  "yfinance|yf\.|Ticker|download|requests|urllib|http|session|provider|market data|os\.environ|dotenv|API|token|credential|if __name__|def main|argparse|to_csv|write_text|mkdir|read_csv|Path[(]|data/processed|data/logs|open[(]|json\.dump|pickle|csv\.writer" \
  scripts/core/indicators.py
```

Result:

```text id="molzbu"
No output.
```

Conclusion:

* no yfinance/provider/network markers;
* no credential markers;
* no runtime entrypoint;
* no production read/write markers;
* no CSV/log/data-path writes;
* no direct execution marker.

### Static source inspection

```bash id="i3037a"
sed -n '1,220p' scripts/core/indicators.py

grep -nE \
  "^(def |class |[A-Z_]+ = )" \
  scripts/core/indicators.py
```

Result summary:

`indicators.py` contains:

* `MIN_REQUIRED_ROWS = 20`
* `add_indicators(df: pd.DataFrame) -> pd.DataFrame`

The helper:

* imports pandas only;
* returns an empty DataFrame when input is missing, empty, or shorter than 20 rows;
* requires:

  * `Open`
  * `High`
  * `Low`
  * `Close`
  * `Volume`
* converts required columns to numeric floats;
* computes:

  * `MA20`
  * `MA50`
  * `MA200`
  * `ATR14`
  * `20D_HIGH`
  * `20D_LOW`
  * `AVG_VOL_20`
* returns the transformed DataFrame.

### Canonical scanner/tests inspection

```bash id="uhm9lq"
grep -RInE \
  "MA20|MA50|MA200|ATR14|20D_HIGH|20D_LOW|AVG_VOL_20|add_indicators|indicator|indicators" \
  src/market_scanner tests \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Result summary:

* no canonical `src/market_scanner` import of `scripts.core.indicators`;
* no active test import of `scripts.core.indicators`;
* the only test references are in the test-local point-in-time indicator contract in:

  * `tests/core/test_build_entry_quality_backfill.py`

Conclusion: active tests do not require the script-era indicators helper.

## Validation

Operator visibility:

```text id="k6olgc"
8 passed
```

Full suite:

```text id="ms2g90"
667 passed
```

## Archive-readiness classification

`scripts/core/indicators.py` is archive-ready.

Reasons:

* no active import from `src`, `tests`, `.github`, or `scripts`;
* no provider/network behavior;
* no yfinance usage;
* no runtime entrypoint;
* no write behavior;
* no production data/log path behavior;
* pure pandas helper only;
* full suite remains green.

## Decision

BL122 approves controlled archive.

Decision:

```text id="zfmky5"
BL123 controlled archive of script-era indicators helper: APPROVED
```

## Required next sprint

Approved next sprint:

```text id="nj6kdt"
BL123 — Controlled archive of script-era indicators helper
```

BL123 must move only:

* `scripts/core/indicators.py`

to:

* `archive/legacy_runtime/scripts/core/indicators.py`

using `git mv`.

BL123 must not modify the file contents.

## Final BL122 result

```text id="evb06d"
BL123 controlled archive indicators helper: APPROVED
```
