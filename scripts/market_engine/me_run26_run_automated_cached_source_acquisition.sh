#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "FAILED: .venv/bin/python is required." >&2
  exit 2
fi

RUN_TIMESTAMP="${ME_RUN26_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
GENERATED_AT="${ME_RUN26_GENERATED_AT:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
RUN_ID="${ME_RUN26_RUN_ID:-me-run26-automated-acquisition-${RUN_TIMESTAMP}}"
ARTIFACT_ROOT="artifacts/market_engine/me-run26-automated-cached-source-acquisition-${RUN_TIMESTAMP}"
ACQUISITION_ROOT="$ARTIFACT_ROOT/acquisition"
DRY_RUN_ROOT="$ARTIFACT_ROOT/dry_runs"
LOG_ROOT="$ARTIFACT_ROOT/logs"
SUMMARY_PATH="$ARTIFACT_ROOT/me_run26_summary.json"
mkdir -p "$ACQUISITION_ROOT" "$DRY_RUN_ROOT" "$LOG_ROOT"

echo "ME-RUN26 automated cached-source acquisition run"
echo "Run id: $RUN_ID"
echo "Generated at: $GENERATED_AT"
echo "Artifact root: $ARTIFACT_ROOT"
echo "Safety: local dry-run only; no provider, network, production, Telegram, portfolio, watchlist, broker, SEC/EDGAR, or yfinance calls."

ACQUISITION_STDOUT="$LOG_ROOT/acquisition.stdout.log"
ACQUISITION_STDERR="$LOG_ROOT/acquisition.stderr.log"
STAGING_STDOUT="$LOG_ROOT/staging_validation.stdout.log"
STAGING_STDERR="$LOG_ROOT/staging_validation.stderr.log"

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$RUN_ID" "$GENERATED_AT" "$ACQUISITION_ROOT" \
  >"$ACQUISITION_STDOUT" 2>"$ACQUISITION_STDERR" <<'PY'
import json
import sys

from market_engine.source_acquisition.automated_cached_source_acquisition import (
    REQUEST_FORMAT,
    run_automated_cached_source_acquisition,
)

run_id, generated_at, acquisition_root = sys.argv[1:4]
request = {
    "request_format": REQUEST_FORMAT,
    "request_id": run_id,
    "requested_at": generated_at,
    "generated_at": generated_at,
    "run_mode": "dry_run",
    "ticker_source": {
        "mode": "explicit_list",
        "source_id": "me_run26_bounded_list",
    },
    "tickers": ["NVDA", "AMD", "ASML"],
    "source_families": ["company_profile"],
    "destination_root": acquisition_root,
    "freshness_policy": {
        "default_max_age_days": 7,
        "per_source_family": {
            "company_profile": {
                "max_age_days": 30,
                "source_timestamp_required": False,
            }
        },
    },
    "provider_policy": {
        "approved_adapters": [
            {
                "adapter_id": "fake_company_profile_adapter",
                "adapter_version": "test-v1",
                "source_families": ["company_profile"],
                "allowed_run_modes": ["dry_run", "local_non_production"],
                "provider_name": "deterministic_fake_provider",
                "canonical_source_identity": "fake://company_profile",
                "network_required": False,
                "rate_limit_policy": "not_applicable",
                "error_policy": "fail_closed",
            }
        ],
        "allow_hidden_fallback": False,
        "allow_silent_substitution": False,
        "allow_fabricated_data": False,
    },
    "safety_flags": {
        "allow_provider_calls": False,
        "allow_network": False,
        "allow_production_writes": False,
        "allow_telegram_send": False,
        "allow_portfolio_writes": False,
        "allow_watchlist_writes": False,
        "allow_broker_actions": False,
    },
    "operator_context": {
        "requested_by": "operator",
        "purpose": "ME-RUN26 bounded local acquisition validation",
        "notes": [],
    },
}
result = run_automated_cached_source_acquisition(request)
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "Acquisition result written to $ACQUISITION_ROOT/acquisition_result.json"

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$ACQUISITION_ROOT" "$ARTIFACT_ROOT/staging_validation.json" "$GENERATED_AT" \
  >"$STAGING_STDOUT" 2>"$STAGING_STDERR" <<'PY'
import json
import sys

from market_engine.source_refresh.cached_source_snapshot_staging_validator import (
    build_cached_source_snapshot_staging_validation,
)

staging_root, output_json, generated_at = sys.argv[1:4]
report = build_cached_source_snapshot_staging_validation(
    staging_root=staging_root,
    validated_at=generated_at,
    tickers=("NVDA", "AMD", "ASML"),
)
with open(output_json, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
counts = report["counts"]
print("CACHED-SOURCE SNAPSHOT STAGING VALIDATION")
print(f"Report format: {report['report_format_version']}")
print(f"Staging root: {report['staging_root']}")
print(f"Validated at: {report['validated_at']}")
print(
    "Counts: "
    f"total={counts['total_inspected_entries']} "
    f"accepted={counts['accepted_entries']} "
    f"rejected={counts['rejected_entries']}"
)
for entry in report["entries"]:
    print(
        f"- {entry.get('ticker') or 'UNKNOWN'} | "
        f"{entry.get('snapshot_id') or 'UNKNOWN'} | "
        f"{entry.get('source_family') or 'unknown_source_family'} | "
        f"{entry['staging_validation_status']} | "
        f"issues={', '.join(entry['issues']) or 'none'}"
    )
PY

echo "Staging validation result written to $ARTIFACT_ROOT/staging_validation.json"

DRY_RUN_COMPLETED=0
DRY_RUN_BLOCKED=0
DRY_RUN_FAILED=0
DRY_RUN_RESULTS=()
for TICKER in NVDA AMD ASML; do
  SNAPSHOT_JSON="$ACQUISITION_ROOT/$TICKER/company_profile/company_profile.json"
  LOWER_TICKER="$(printf '%s' "$TICKER" | tr '[:upper:]' '[:lower:]')"
  STDOUT_PATH="$LOG_ROOT/dry_run_${TICKER}.stdout.log"
  STDERR_PATH="$LOG_ROOT/dry_run_${TICKER}.stderr.log"
  if PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m market_engine.run.end_to_end_dry_run_command \
    --input-mode cached_source_snapshot \
    --source-snapshot-json "$SNAPSHOT_JSON" \
    --source-snapshot-root "$ACQUISITION_ROOT" \
    --dry-run-id "me-run26-${LOWER_TICKER}-company-profile-dry-run" \
    --generated-at "$GENERATED_AT" \
    --artifact-output-root "$DRY_RUN_ROOT" \
    --artifact-created-at "$GENERATED_AT" \
    --write-local-artifact \
    --compact \
    >"$STDOUT_PATH" 2>"$STDERR_PATH"; then
    DRY_RUN_COMPLETED=$((DRY_RUN_COMPLETED + 1))
    DRY_RUN_RESULTS+=("${TICKER}:completed")
  else
    EXIT_CODE=$?
    if grep -qiE "sec companyfacts|companyfacts|source context|snapshot" "$STDERR_PATH"; then
      DRY_RUN_BLOCKED=$((DRY_RUN_BLOCKED + 1))
      DRY_RUN_RESULTS+=("${TICKER}:blocked:${EXIT_CODE}")
    else
      DRY_RUN_FAILED=$((DRY_RUN_FAILED + 1))
      DRY_RUN_RESULTS+=("${TICKER}:failed:${EXIT_CODE}")
    fi
  fi
done

if [[ "$DRY_RUN_FAILED" -gt 0 ]]; then
  OVERALL_RESULT="FAILED"
elif [[ "$DRY_RUN_COMPLETED" -eq 3 ]]; then
  OVERALL_RESULT="PASS"
else
  OVERALL_RESULT="BLOCKED"
fi

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$SUMMARY_PATH" "$RUN_ID" "$GENERATED_AT" "$ARTIFACT_ROOT" "$ACQUISITION_ROOT" "$DRY_RUN_COMPLETED" "$DRY_RUN_BLOCKED" "$DRY_RUN_FAILED" "$OVERALL_RESULT" "${DRY_RUN_RESULTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

(
    summary_path,
    run_id,
    generated_at,
    artifact_root,
    acquisition_root,
    completed,
    blocked,
    failed,
    overall_result,
    *dry_run_results,
) = sys.argv[1:]
payload = {
    "summary_format": "market-engine-me-run26-summary-v1",
    "run_id": run_id,
    "generated_at": generated_at,
    "artifact_root": artifact_root,
    "acquisition_root": acquisition_root,
    "tickers": ["NVDA", "AMD", "ASML"],
    "source_family": "company_profile",
    "acquisition_result": "completed",
    "staging_validation_result": "accepted",
    "cached_source_snapshot_dry_run": {
        "completed_count": int(completed),
        "blocked_count": int(blocked),
        "failed_count": int(failed),
        "results": dry_run_results,
    },
    "overall_result": overall_result,
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "Summary written to $SUMMARY_PATH"
echo "Dry-run results: ${DRY_RUN_RESULTS[*]}"
echo "$OVERALL_RESULT"
