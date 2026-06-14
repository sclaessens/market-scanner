# BL80 — Remaining fundamentals migration blocker review

Status: COMPLETED

## Purpose

BL80 reviews the remaining `scripts/fundamentals/*.py` files after BL74 through BL79 reduced the script-era fundamentals coupling.

This sprint does not archive, delete, move, refactor, or execute runtime Python files.

BL80 exists to identify the exact blockers that prevent the remaining fundamentals scripts from being archived safely.

## Background

Recent cleanup sequence:

* BL74 removed active runtime imports from `scripts.fundamentals` in tests, source, and workflows.
* BL75 archived `scripts/fundamentals/__init__.py`.
* BL76 classified the remaining script-era Python files.
* BL77 archived low-risk non-fundamentals helper files.
* BL78 reviewed the remaining `scripts/fundamentals/*.py` files and concluded that none should be archived blindly.
* BL79 archived the `scripts/core/build_fundamental_*` compatibility wrappers.

After BL79, the remaining fundamentals risk is concentrated in the actual `scripts/fundamentals/*.py` modules.

## Scope

Reviewed files:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`
* `scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

Out of scope:

* `scripts/core/build_fundamental_*` wrappers, archived by BL79.
* Portfolio cleanup.
* Watchlist cleanup.
* Telegram cleanup.
* Reporting cleanup.
* Scanner cleanup.
* Decision Engine cleanup.
* Any runtime execution of fundamentals scripts.
* Any live SEC/EDGAR calls.
* Any production data writes.

## Remaining file inventory

Command:

```bash
find scripts/fundamentals -type f -name "*.py" | sort
```

Result:

```text
scripts/fundamentals/build_analysis.py
scripts/fundamentals/build_history_intake.py
scripts/fundamentals/build_metrics.py
scripts/fundamentals/build_quality.py
scripts/fundamentals/run_sec_transformation_review.py
scripts/fundamentals/sec_companyfacts_bulk_intake.py
scripts/fundamentals/sec_companyfacts_transform.py
scripts/fundamentals/sec_ticker_cik_index.py
```

Finding:

* Eight Python files remain under `scripts/fundamentals/`.
* None are declared archive-ready by BL80.
* The files are now isolated as the remaining fundamentals migration-risk group after the BL79 wrapper archive.

## Active import and dependency check

Command:

```bash
grep -R "from scripts.fundamentals\|import scripts.fundamentals" -n scripts/fundamentals tests src .github \
  --include="*.py" \
  --include="*.yml" \
  --include="*.yaml" \
  --exclude-dir=".git" \
  --exclude-dir=".venv" \
  --exclude-dir="archive"
```

Result:

```text
scripts/fundamentals/build_analysis.py:10:from scripts.fundamentals.build_metrics import HELPER_COLUMNS, IDENTITY_COLUMNS, METRIC_COLUMNS
scripts/fundamentals/run_sec_transformation_review.py:10:from scripts.fundamentals.build_history_intake import REQUIRED_COLUMNS
scripts/fundamentals/run_sec_transformation_review.py:11:from scripts.fundamentals.sec_companyfacts_transform import transform_companyfacts_file
scripts/fundamentals/run_sec_transformation_review.py:12:from scripts.fundamentals.sec_ticker_cik_index import (
scripts/fundamentals/build_metrics.py:10:from scripts.fundamentals.build_history_intake import validate_fundamentals_history
scripts/fundamentals/build_quality.py:11:from scripts.fundamentals.build_metrics import HELPER_COLUMNS, IDENTITY_COLUMNS, METRIC_COLUMNS
scripts/fundamentals/build_quality.py:12:from scripts.fundamentals.build_history_intake import validate_fundamentals_history
scripts/fundamentals/sec_companyfacts_transform.py:11:from scripts.fundamentals.build_history_intake import REQUIRED_COLUMNS
scripts/fundamentals/sec_companyfacts_transform.py:12:from scripts.fundamentals.sec_ticker_cik_index import normalize_cik, normalize_ticker
```

Result summary:

* No active test runtime imports from `scripts.fundamentals` were found.
* No `src/market_scanner` runtime imports from `scripts.fundamentals` were found.
* No `.github` workflow imports from `scripts.fundamentals` were found.
* Only internal imports inside `scripts/fundamentals/` were found.
* The remaining dependencies are local script-era dependencies inside the legacy fundamentals group.

Observed internal dependency cluster:

* `scripts/fundamentals/build_analysis.py`

  * depends on `scripts.fundamentals.build_metrics`
* `scripts/fundamentals/build_metrics.py`

  * depends on `scripts.fundamentals.build_history_intake`
* `scripts/fundamentals/build_quality.py`

  * depends on `scripts.fundamentals.build_metrics`
  * depends on `scripts.fundamentals.build_history_intake`
* `scripts/fundamentals/sec_companyfacts_transform.py`

  * depends on `scripts.fundamentals.build_history_intake`
  * depends on `scripts.fundamentals.sec_ticker_cik_index`
* `scripts/fundamentals/run_sec_transformation_review.py`

  * depends on `scripts.fundamentals.build_history_intake`
  * depends on `scripts.fundamentals.sec_companyfacts_transform`
  * depends on `scripts.fundamentals.sec_ticker_cik_index`

Interpretation:

The remaining fundamentals files form a coupled legacy module group. They should not be archived one by one unless dependency direction, canonical replacement, and retirement status are explicitly resolved.

## Side-effect and provider-risk check

Command:

```bash
grep -R "urlopen\|requests\|sec.gov\|to_csv\|to_json\|write_text\|open(.*w\|output_path\|data/processed\|data/raw\|data/local\|manifest" -n scripts/fundamentals \
  --include="*.py" \
  --exclude-dir=".git" \
  --exclude-dir=".venv" \
  --exclude-dir="archive"
```

Result:

```text
scripts/fundamentals/build_analysis.py:387:    output_path: str | Path | None = None,
scripts/fundamentals/build_analysis.py:393:    if output_path is not None:
scripts/fundamentals/build_analysis.py:394:        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
scripts/fundamentals/build_analysis.py:395:        analysis_df.to_csv(output_path, index=False)
scripts/fundamentals/build_analysis.py:410:    analysis_df = build_fundamental_analysis(args.quality_path, args.metrics_path, args.output_path)
scripts/fundamentals/build_analysis.py:414:        "output_path": args.output_path or "",
scripts/fundamentals/sec_ticker_cik_index.py:225:    output_path: str | Path | None = None,
scripts/fundamentals/sec_ticker_cik_index.py:230:    if output_path is not None:
scripts/fundamentals/sec_ticker_cik_index.py:231:        output = Path(output_path)
scripts/fundamentals/sec_ticker_cik_index.py:233:        coverage_df.to_csv(output, index=False)
scripts/fundamentals/sec_ticker_cik_index.py:273:        coverage_df.to_csv(args.output, index=False)
scripts/fundamentals/run_sec_transformation_review.py:125:    output_path: str | Path | None = None,
scripts/fundamentals/run_sec_transformation_review.py:223:    if output_path is not None:
scripts/fundamentals/run_sec_transformation_review.py:224:        output = Path(output_path)
scripts/fundamentals/run_sec_transformation_review.py:226:        review_df.to_csv(output, index=False)
scripts/fundamentals/run_sec_transformation_review.py:230:def summarize_review(review_df: pd.DataFrame, output_path: str | Path | None = None) -> dict[str, Any]:
scripts/fundamentals/run_sec_transformation_review.py:235:        "output_path": str(output_path) if output_path is not None else "",
scripts/fundamentals/run_sec_transformation_review.py:257:    output_path = None if args.validate_only else args.output
scripts/fundamentals/run_sec_transformation_review.py:264:        output_path=output_path,
scripts/fundamentals/run_sec_transformation_review.py:266:    print(json.dumps(summarize_review(review_df, output_path), sort_keys=True, indent=2))
scripts/fundamentals/build_history_intake.py:239:        Path(args.report_path).write_text(rendered + "\n", encoding="utf-8")
scripts/fundamentals/build_metrics.py:192:def build_fundamental_metrics(input_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
scripts/fundamentals/build_metrics.py:201:    if output_path is not None:
scripts/fundamentals/build_metrics.py:202:        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
scripts/fundamentals/build_metrics.py:203:        metrics_df.to_csv(output_path, index=False)
scripts/fundamentals/build_metrics.py:217:    metrics_df = build_fundamental_metrics(args.input_path, args.output_path)
scripts/fundamentals/build_metrics.py:221:        "output_path": args.output_path or "",
scripts/fundamentals/sec_companyfacts_bulk_intake.py:14:from urllib.request import Request, urlopen
scripts/fundamentals/sec_companyfacts_bulk_intake.py:16:SEC_COMPANYFACTS_BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
scripts/fundamentals/sec_companyfacts_bulk_intake.py:17:SEC_COMPANYFACTS_HOST = "www.sec.gov"
scripts/fundamentals/sec_companyfacts_bulk_intake.py:19:DEFAULT_CACHE_DIR = Path("data/local/sec_edgar/companyfacts")
scripts/fundamentals/sec_companyfacts_bulk_intake.py:21:MANIFEST_NAME = "companyfacts_manifest.json"
scripts/fundamentals/sec_companyfacts_bulk_intake.py:42:        raise ValueError("SEC requests require an explicit descriptive User-Agent.")
scripts/fundamentals/sec_companyfacts_bulk_intake.py:99:def build_companyfacts_manifest(
scripts/fundamentals/sec_companyfacts_bulk_intake.py:123:def write_manifest(manifest: dict[str, Any], cache_dir: str | Path) -> Path:
scripts/fundamentals/sec_companyfacts_bulk_intake.py:125:    manifest_path = directory / MANIFEST_NAME
scripts/fundamentals/sec_companyfacts_bulk_intake.py:126:    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
scripts/fundamentals/sec_companyfacts_bulk_intake.py:127:    return manifest_path
scripts/fundamentals/sec_companyfacts_bulk_intake.py:135:    write_manifest_file: bool = False,
scripts/fundamentals/sec_companyfacts_bulk_intake.py:137:    manifest = build_companyfacts_manifest(zip_path, source_url=source_url)
scripts/fundamentals/sec_companyfacts_bulk_intake.py:138:    if write_manifest_file:
scripts/fundamentals/sec_companyfacts_bulk_intake.py:140:            raise ValueError("cache_dir is required when write_manifest_file is true.")
scripts/fundamentals/sec_companyfacts_bulk_intake.py:141:        write_manifest(manifest, cache_dir)
scripts/fundamentals/sec_companyfacts_bulk_intake.py:142:    return manifest
scripts/fundamentals/sec_companyfacts_bulk_intake.py:150:        with os.fdopen(fd, "wb") as output:
scripts/fundamentals/sec_companyfacts_bulk_intake.py:151:            with urlopen(request, timeout=timeout_seconds) as response:
scripts/fundamentals/sec_companyfacts_bulk_intake.py:173:    write_manifest_file: bool = True,
scripts/fundamentals/sec_companyfacts_bulk_intake.py:186:    manifest = build_companyfacts_manifest(target_path, source_url=source_url, status="DOWNLOADED")
scripts/fundamentals/sec_companyfacts_bulk_intake.py:187:    if write_manifest_file:
scripts/fundamentals/sec_companyfacts_bulk_intake.py:188:        write_manifest(manifest, directory)
scripts/fundamentals/sec_companyfacts_bulk_intake.py:189:    return manifest
scripts/fundamentals/sec_companyfacts_bulk_intake.py:200:        "--write-manifest",
scripts/fundamentals/sec_companyfacts_bulk_intake.py:202:        help="Write a local manifest in the cache directory for --validate-local. Downloads write one by default.",
scripts/fundamentals/sec_companyfacts_bulk_intake.py:212:        manifest = download_companyfacts_bulk_zip(
scripts/fundamentals/sec_companyfacts_bulk_intake.py:218:        manifest = inspect_local_companyfacts_zip(
scripts/fundamentals/sec_companyfacts_bulk_intake.py:222:            write_manifest_file=args.write_manifest,
scripts/fundamentals/sec_companyfacts_bulk_intake.py:225:    print(json.dumps(manifest, indent=2, sort_keys=True))
scripts/fundamentals/build_quality.py:14:CONTEXT_PATH = Path("data/processed/context_strength.csv")
scripts/fundamentals/build_quality.py:15:RAW_FUNDAMENTALS_PATH = Path("data/raw/fundamentals.csv")
scripts/fundamentals/build_quality.py:16:OUTPUT_PATH = Path("data/processed/fundamental_quality.csv")
scripts/fundamentals/build_quality.py:673:    pd.DataFrame([log_row], columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)
scripts/fundamentals/build_quality.py:700:    output_df.to_csv(OUTPUT_PATH, index=False)
scripts/fundamentals/sec_companyfacts_transform.py:622:    output_path: str | Path | None = None,
scripts/fundamentals/sec_companyfacts_transform.py:631:    if output_path is not None:
scripts/fundamentals/sec_companyfacts_transform.py:632:        output = Path(output_path)
scripts/fundamentals/sec_companyfacts_transform.py:634:        df.to_csv(output, index=False)
scripts/fundamentals/sec_companyfacts_transform.py:638:def summarize_transform(df: pd.DataFrame, output_path: str | Path | None = None) -> dict[str, Any]:
scripts/fundamentals/sec_companyfacts_transform.py:642:        "output_path": str(output_path) if output_path is not None else "",
scripts/fundamentals/sec_companyfacts_transform.py:658:    output_path = None if args.validate_only else args.output
scripts/fundamentals/sec_companyfacts_transform.py:665:        output_path=output_path,
scripts/fundamentals/sec_companyfacts_transform.py:667:    print(json.dumps(summarize_transform(df, output_path), sort_keys=True, indent=2))
```

Result summary:

* `scripts/fundamentals/build_analysis.py` exposes optional CSV output behavior.
* `scripts/fundamentals/build_history_intake.py` exposes optional report write behavior.
* `scripts/fundamentals/build_metrics.py` exposes optional CSV output behavior.
* `scripts/fundamentals/build_quality.py` contains default `data/processed` and `data/raw` paths and writes `fundamental_quality.csv` and log CSV output.
* `scripts/fundamentals/sec_companyfacts_transform.py` exposes optional CSV output behavior.
* `scripts/fundamentals/sec_ticker_cik_index.py` exposes optional coverage CSV output behavior.
* `scripts/fundamentals/run_sec_transformation_review.py` exposes optional review CSV output behavior.
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py` is SEC/network/cache/manifest capable through `urlopen`, SEC URLs, local cache paths, and manifest writes.

Interpretation:

The remaining fundamentals files contain or expose write behavior, provider/source-data risk, or migration-relevant pure logic. They are not low-risk archive candidates as a single group.

## Blocker classification

| File                                                    | BL80 classification                        | Main blocker                                                                                                | Required next step                                                                                   |
| ------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `scripts/fundamentals/build_history_intake.py`          | BLOCKED_CONTRACT_EXTRACTION_REQUIRED       | History schema and validation knowledge may still be needed; dependency base for metrics and SEC transform. | Extract schema/validation contract into canonical fundamentals tests/docs before archive.            |
| `scripts/fundamentals/build_metrics.py`                 | BLOCKED_METRIC_PARITY_REQUIRED             | Metric formulas and missing-value behavior may not have complete canonical parity.                          | Compare formulas with canonical contracts; migrate approved formulas or declare obsolete.            |
| `scripts/fundamentals/build_analysis.py`                | BLOCKED_ANALYSIS_PARITY_REQUIRED           | Fundamental analysis state logic may overlap with canonical analysis boundary but parity is not proven.     | Define canonical analysis owner and migrate/retire review-safe analysis states.                      |
| `scripts/fundamentals/build_quality.py`                 | BLOCKED_QUALITY_READINESS_PARITY_REQUIRED  | Quality/readiness logic may combine source quality, freshness, sufficiency, and output writes.              | Split source-data readiness policy from legacy output generation before archive.                     |
| `scripts/fundamentals/sec_companyfacts_transform.py`    | BLOCKED_SEC_TRANSFORM_PARITY_REQUIRED      | SEC CompanyFacts transformation and fact-selection rules may still be useful.                               | Compare with canonical SEC/source-data boundary; migrate pure transform rules only.                  |
| `scripts/fundamentals/sec_ticker_cik_index.py`          | BLOCKED_IDENTIFIER_MAPPING_POLICY_REQUIRED | Ticker/CIK normalization and mapping ownership is not fully retired.                                        | Define canonical identifier/source-metadata owner before archive.                                    |
| `scripts/fundamentals/run_sec_transformation_review.py` | BLOCKED_REVIEW_RUNNER_RETIREMENT_REQUIRED  | Review runner may read local SEC-like inputs and write review outputs.                                      | Retire or replace with canonical static review policy after SEC transform parity decision.           |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py`  | BLOCKED_PROVIDER_GOVERNANCE_REQUIRED       | SEC/network/cache/manifest behavior is high-risk.                                                           | Keep blocked. Do not execute or migrate live behavior without explicit provider governance approval. |

## Dependency lanes

BL80 splits the remaining fundamentals cleanup into four follow-up lanes.

### Lane 1 — Pure contract extraction

Candidate files:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

Goal:

* Extract schema, validation, formula, and missing-value policy.
* Add or confirm canonical tests under `src/market_scanner/fundamentals/` or `src/market_scanner/analysis/`.
* Do not preserve legacy CSV write behavior unless explicitly approved.

### Lane 2 — Analysis and quality parity

Candidate files:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_quality.py`

Goal:

* Separate neutral evidence/readiness logic from legacy output generation.
* Preserve no-allocation/no-recommendation doctrine.
* Confirm whether canonical `src/market_scanner/analysis/` and `src/market_scanner/fundamentals/` own equivalent contracts.

### Lane 3 — SEC transform and identifier mapping

Candidate files:

* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

Goal:

* Extract local-only, pure transformation and mapping rules.
* Avoid live SEC behavior.
* Define canonical source-metadata ownership before archive.

### Lane 4 — Provider and review-runner retirement

Candidate files:

* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* `scripts/fundamentals/run_sec_transformation_review.py`

Goal:

* Keep provider/network/cache behavior blocked.
* Retire or replace local review-runner behavior only after transform parity and provider governance are decided.
* Do not execute either file during cleanup.

## Decision

No remaining `scripts/fundamentals/*.py` file is archive-ready as of BL80.

The remaining files are not protected by active runtime imports, but they still contain migration knowledge, internal dependencies, provider/data-write risk, or review-runner behavior.

The next cleanup must therefore be migration/parity-focused, not archive-focused.

## Recommended next sprint

Recommended next sprint:

```text
BL81 — Extract fundamentals history and metrics contracts from script-era modules
```

Proposed BL81 scope:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

BL81 should be a code-aware but no-side-effect migration sprint. It should inspect these two modules and translate only stable schema/formula contracts into canonical tests or docs. It should not execute the scripts and should not write production data.

Alternative if keeping the next sprint documentation-only:

```text
BL81 — Document canonical fundamentals history and metrics contract requirements
```

## Validation

Full test suite should be run for repository safety.

Command:

```bash
pytest -q
```

Result:

```text
522 passed in 0.57s
```

## Guardrails

* No live SEC/EDGAR calls were run.
* No yfinance calls were run.
* No credentials were read.
* No production data was written.
* No production reports were generated.
* No Telegram messages were sent.
* No portfolio/watchlist production state was modified.
* No Decision Engine authority was changed.
* No script-era Python runtime files were executed.

## Final status

BL80 completes the blocker review for the remaining `scripts/fundamentals/*.py` files.

The remaining fundamentals cleanup is now split into governed lanes:

1. pure contract extraction;
2. analysis and quality parity;
3. SEC transform and identifier mapping;
4. provider and review-runner retirement.

No fundamentals script is approved for direct archive in BL80.
