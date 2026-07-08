from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from market_engine.advisory.advisory_prompt_package import (
    AdvisoryPromptPackageError,
    build_advisory_prompt_package,
)
from market_engine.advisory.advisory_response_grounding import (
    AdvisoryResponseGroundingResult,
    validate_advisory_response_grounding,
)


CONTROLLED_RESPONSE_DRY_RUN_MANIFEST_VERSION = (
    "market-engine-controlled-advisory-response-dry-run-manifest-v1"
)
CONTROLLED_RESPONSE_DRY_RUN_SUMMARY_VERSION = (
    "market-engine-controlled-advisory-response-dry-run-summary-v1"
)
CONTROLLED_RESPONSE_DRY_RUN_PATH_CATEGORY = (
    "artifacts/market_engine/controlled_advisory_response_dry_runs"
)

SUCCESS_STATUSES = {
    "grounded": "dry_run_completed_grounded",
    "grounded_with_mandatory_caveats": "dry_run_completed_with_caveats",
    "partially_grounded": "dry_run_completed_partial",
}
FAILED_STATUSES = {
    "ungrounded": "dry_run_failed_ungrounded",
    "blocked": "dry_run_blocked",
}


class AdvisoryResponseDryRunError(ValueError):
    """Raised when a controlled advisory response dry run cannot run safely."""


@dataclass(frozen=True)
class AdvisoryResponseDryRunResult:
    run_directory: Path
    prompt_package_path: Path
    synthetic_response_path: Path
    grounding_result_path: Path
    dry_run_summary_path: Path
    manifest_path: Path
    grounding_result: AdvisoryResponseGroundingResult
    summary: dict[str, Any]
    manifest: dict[str, Any]


def run_controlled_advisory_response_dry_run(
    *,
    advisory_artifact_path: Path | str,
    question: str,
    question_class: str,
    response_fixture_path: Path | str,
    run_id: str,
    artifact_root: Path | str = CONTROLLED_RESPONSE_DRY_RUN_PATH_CATEGORY,
    allow_overwrite: bool = False,
) -> AdvisoryResponseDryRunResult:
    source_artifact = _read_json_object(Path(advisory_artifact_path))
    response = _read_json_object(Path(response_fixture_path))
    if not isinstance(run_id, str) or not run_id:
        raise AdvisoryResponseDryRunError("run_id is required.")
    safe_run_id = _safe_path_segment(run_id, "run_id")
    ticker = _safe_path_segment(
        _nested_text(source_artifact, ("instrument_identity", "ticker")) or "UNKNOWN",
        "ticker",
    )
    prompt_package = build_advisory_prompt_package(
        advisory_artifact=source_artifact,
        question=question,
        question_class=question_class,
        package_id=safe_run_id,
    )
    grounding_result = validate_advisory_response_grounding(
        source_artifact=source_artifact,
        prompt_package=prompt_package,
        response=response,
    )
    root = _validated_output_root(artifact_root)
    root_resolved = root.resolve()
    run_directory = _resolved_child(_resolved_child(root_resolved, safe_run_id), ticker)
    if run_directory.exists() and not allow_overwrite:
        raise AdvisoryResponseDryRunError(
            f"Controlled advisory response dry-run directory already exists: {run_directory}"
        )
    if run_directory.exists() and allow_overwrite:
        shutil.rmtree(run_directory)
    run_directory.mkdir(parents=True, exist_ok=False)

    dry_run_state = (
        SUCCESS_STATUSES.get(grounding_result.status)
        or FAILED_STATUSES.get(grounding_result.status)
        or "dry_run_failed_ungrounded"
    )
    prompt_package_path = run_directory / "prompt_package.json"
    synthetic_response_path = run_directory / "synthetic_response.json"
    grounding_result_path = run_directory / "grounding_result.json"
    dry_run_summary_path = run_directory / "dry_run_summary.json"
    manifest_path = run_directory / "manifest.json"
    summary = {
        "schema_version": CONTROLLED_RESPONSE_DRY_RUN_SUMMARY_VERSION,
        "artifact_type": "market-engine-controlled-advisory-response-dry-run-summary",
        "run_id": safe_run_id,
        "ticker": ticker,
        "dry_run_state": dry_run_state,
        "grounding_status": grounding_result.status,
        "valid_grounded_response": grounding_result.valid,
        "response_mode": grounding_result.validated_response_mode,
        "issue_count": len(grounding_result.issues),
        "model_invocation_performed": False,
        "provider_call_performed": False,
        "delivery_performed": False,
        "portfolio_write_performed": False,
        "watchlist_write_performed": False,
    }
    manifest = {
        "manifest_format_version": CONTROLLED_RESPONSE_DRY_RUN_MANIFEST_VERSION,
        "artifact_count": 4,
        "run_id": safe_run_id,
        "ticker": ticker,
        "dry_run_state": dry_run_state,
        "grounding_status": grounding_result.status,
        "non_production_artifact": True,
        "local_only": True,
        "model_free": True,
        "artifact_path_category": CONTROLLED_RESPONSE_DRY_RUN_PATH_CATEGORY,
        "prompt_package_relative_path": _relative_posix(prompt_package_path, root_resolved),
        "synthetic_response_relative_path": _relative_posix(
            synthetic_response_path, root_resolved
        ),
        "grounding_result_relative_path": _relative_posix(
            grounding_result_path, root_resolved
        ),
        "dry_run_summary_relative_path": _relative_posix(
            dry_run_summary_path, root_resolved
        ),
    }
    _write_json(prompt_package_path, prompt_package)
    _write_json(synthetic_response_path, response)
    _write_json(grounding_result_path, grounding_result.to_payload())
    _write_json(dry_run_summary_path, summary)
    _write_json(manifest_path, manifest)
    return AdvisoryResponseDryRunResult(
        run_directory=run_directory,
        prompt_package_path=prompt_package_path,
        synthetic_response_path=synthetic_response_path,
        grounding_result_path=grounding_result_path,
        dry_run_summary_path=dry_run_summary_path,
        manifest_path=manifest_path,
        grounding_result=grounding_result,
        summary=summary,
        manifest=manifest,
    )


def run_controlled_advisory_response_dry_run_command(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    args = _argument_parser().parse_args(argv)
    try:
        result = run_controlled_advisory_response_dry_run(
            advisory_artifact_path=args.advisory_artifact,
            question=args.question,
            question_class=args.question_class,
            response_fixture_path=args.response_fixture,
            run_id=args.run_id,
            artifact_root=args.artifact_root,
            allow_overwrite=args.allow_overwrite,
        )
    except (AdvisoryPromptPackageError, AdvisoryResponseDryRunError) as exc:
        print(str(exc), file=error_stream)
        return 2
    if args.emit_json:
        json.dump(result.summary, output_stream, indent=2, sort_keys=True)
        output_stream.write("\n")
    else:
        output_stream.write(f"dry_run_state={result.summary['dry_run_state']}\n")
        output_stream.write(f"grounding_status={result.grounding_result.status}\n")
        output_stream.write(f"manifest_path={result.manifest_path.as_posix()}\n")
    return 0 if result.grounding_result.valid else 2


def main() -> None:
    raise SystemExit(run_controlled_advisory_response_dry_run_command())


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-engine-controlled-advisory-response-dry-run",
        description=(
            "Run a local deterministic ME-CI08 controlled advisory response dry run. "
            "The command loads a CI06-valid advisory artifact and an explicit "
            "synthetic response fixture; it performs no model, provider, delivery, "
            "broker, portfolio, or watchlist side effects."
        ),
    )
    parser.add_argument("--advisory-artifact", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--question-class", required=True)
    parser.add_argument("--response-fixture", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--artifact-root",
        default=CONTROLLED_RESPONSE_DRY_RUN_PATH_CATEGORY,
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--emit-json", action="store_true")
    return parser


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AdvisoryResponseDryRunError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AdvisoryResponseDryRunError(f"JSON file is malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise AdvisoryResponseDryRunError(f"JSON file must contain an object: {path}")
    return payload


def _validated_output_root(output_root: Path | str) -> Path:
    root = Path(output_root)
    if any(part == ".." for part in root.parts):
        raise AdvisoryResponseDryRunError(
            f"Output root must not contain parent traversal: {root}"
        )
    return root


def _resolved_child(parent: Path, child_name: str) -> Path:
    child = parent / child_name
    resolved = child.resolve()
    if parent not in resolved.parents and resolved != parent:
        raise AdvisoryResponseDryRunError(
            f"Resolved output path escapes artifact root: {child}"
        )
    return resolved


def _safe_path_segment(value: str, field_name: str) -> str:
    if not re_match_safe(value):
        raise AdvisoryResponseDryRunError(
            f"{field_name} must be a safe path segment: {value}"
        )
    return value


def re_match_safe(value: str) -> bool:
    return isinstance(value, str) and bool(value) and all(
        char.isalnum() or char in {"-", "_", "."} for char in value
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _nested_text(payload: Mapping[str, Any], path: tuple[str, ...]) -> str | None:
    current: Any = payload
    for segment in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current if isinstance(current, str) else None


if __name__ == "__main__":
    main()
