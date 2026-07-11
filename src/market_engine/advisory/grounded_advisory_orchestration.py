from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from market_engine.advisory import grounded_advisory_runtime as runtime

_PROVIDER_DISABLED_REASON = (
    "Provider invocation is disabled by default for the GitHub-first no-API "
    "baseline. The baseline path must prepare deterministic Market Engine "
    "artifacts for ChatGPT interpretation without calling the OpenAI API. Pass "
    "an explicit test invoker for fixture-based validation; do not use the "
    "default command path for real provider calls."
)


def generate_grounded_advisory_output(
    *,
    source_artifact_path: Path | str,
    output_root: Path | str = runtime.DEFAULT_OUTPUT_ROOT,
    run_id: str,
    generated_at: str,
    invoker: runtime.ModelInvoker | None = None,
    allow_overwrite: bool = False,
) -> runtime.GroundedAdvisoryGenerationResult:
    source_path = Path(source_artifact_path)
    source_artifact = runtime._read_json_object(source_path)
    source_validation = runtime._validate_source_artifact(source_artifact)
    source_summary = runtime._source_summary(source_artifact, source_path)
    ticker = runtime._safe_path_segment(source_summary["ticker"], "ticker")
    safe_run_id = runtime._safe_path_segment(run_id, "run_id")
    output_root_path = runtime._validated_output_root(output_root)
    output_directory = runtime._resolved_child(
        runtime._resolved_child(output_root_path.resolve(), safe_run_id), ticker
    )
    if output_directory.exists() and not allow_overwrite:
        raise runtime.GroundedAdvisoryOutputError(
            f"Grounded advisory output directory already exists: {output_directory}"
        )
    if output_directory.exists() and allow_overwrite:
        runtime._remove_tree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)

    invocation_request = runtime._build_invocation_request(
        source_artifact=source_artifact,
        source_path=source_path,
        source_validation=source_validation,
        source_summary=source_summary,
        run_id=safe_run_id,
        generated_at=generated_at,
    )
    request_validation = runtime._validate_invocation_request(invocation_request)
    if not source_validation["valid"]:
        invocation_result = runtime._blocked_invocation_result(
            "Source artifact failed pre-invocation validation."
        )
    elif not request_validation["valid"]:
        invocation_result = runtime._blocked_invocation_result(
            "Invocation request failed deterministic CI10 pre-invocation validation."
        )
    elif invoker is not None:
        invocation_result = invoker.invoke(invocation_request)
    else:
        invocation_result = runtime._blocked_invocation_result(_PROVIDER_DISABLED_REASON)

    parser_result = runtime._parse_model_response(invocation_result.raw_output)
    parsed_response = (
        parser_result.get("parsed_response")
        if parser_result.get("status") == "valid"
        else None
    )
    validation_result = runtime._validate_model_response(
        parsed_response=parsed_response,
        invocation_result=invocation_result,
        source_summary=source_summary,
        source_validation=source_validation,
        invocation_request=invocation_request,
        request_validation=request_validation,
        run_id=safe_run_id,
    )
    structured_output = runtime._structured_output(
        source_summary=source_summary,
        invocation_request=invocation_request,
        invocation_result=invocation_result,
        parser_result=parser_result,
        validation_result=validation_result,
        run_id=safe_run_id,
        generated_at=generated_at,
    )
    if not source_validation["valid"]:
        structured_output["advisory_status"] = "blocked_source_not_supported"
        structured_output["executive_conclusion"] = (
            "No grounded advisory conclusion was generated because the source artifact failed "
            "pre-invocation validation."
        )
    report = runtime._render_report(structured_output)
    manifest = runtime._manifest(
        output_directory=output_directory,
        structured_output=structured_output,
        run_id=safe_run_id,
        ticker=ticker,
    )

    invocation_request_path = output_directory / "invocation_request.json"
    raw_response_path = output_directory / "raw_model_response.json"
    parser_result_path = output_directory / "parser_result.json"
    validation_result_path = output_directory / "validation_result.json"
    structured_output_path = output_directory / "grounded_advisory_output.json"
    report_path = output_directory / "advisory_report.md"
    manifest_path = output_directory / "manifest.json"
    runtime._write_json(invocation_request_path, invocation_request)
    runtime._write_json(
        raw_response_path, runtime._raw_response_payload(invocation_result)
    )
    runtime._write_json(parser_result_path, parser_result)
    runtime._write_json(validation_result_path, validation_result)
    runtime._write_json(structured_output_path, structured_output)
    report_path.write_text(report, encoding="utf-8")
    runtime._write_json(manifest_path, manifest)

    return runtime.GroundedAdvisoryGenerationResult(
        output_directory=output_directory,
        structured_output_path=structured_output_path,
        report_path=report_path,
        invocation_request_path=invocation_request_path,
        raw_response_path=raw_response_path,
        parser_result_path=parser_result_path,
        validation_result_path=validation_result_path,
        manifest_path=manifest_path,
        summary={
            "run_id": safe_run_id,
            "ticker": ticker,
            "advisory_status": structured_output["advisory_status"],
            "validation_status": validation_result["status"],
            "grounding_status": validation_result.get("grounding_status"),
            "invocation_state": invocation_result.invocation_state,
            "structured_output_path": structured_output_path.as_posix(),
            "report_path": report_path.as_posix(),
        },
    )


def run_grounded_advisory_output_command(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    args = _argument_parser().parse_args(argv)
    try:
        result = generate_grounded_advisory_output(
            source_artifact_path=args.artifact,
            output_root=args.output_root,
            run_id=args.run_id,
            generated_at=args.generated_at,
            allow_overwrite=args.allow_overwrite,
        )
    except runtime.GroundedAdvisoryOutputError as exc:
        print(str(exc), file=error_stream)
        return 2
    json.dump(result.summary, output_stream, indent=2, sort_keys=True)
    output_stream.write("\n")
    return 0 if result.summary["validation_status"] == "valid" else 2


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-engine-grounded-advisory-output",
        description=(
            "Generate a local grounded advisory output from a supplied artifact. "
            "The GitHub-first baseline command path is provider-disabled by default "
            "and never calls the OpenAI API."
        ),
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-root", default=runtime.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(run_grounded_advisory_output_command())
