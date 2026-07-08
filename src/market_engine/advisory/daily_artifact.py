from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Sequence, TextIO

from market_engine.advisory.advisory_artifact import (
    CHATGPT_READY_ADVISORY_ARTIFACT_PATH_CATEGORY,
    ChatGPTReadyAdvisoryArtifactError,
    compose_chatgpt_ready_advisory_artifact_from_directory,
)


def run_chatgpt_ready_advisory_artifact_command(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    args = _argument_parser().parse_args(argv)
    try:
        result = compose_chatgpt_ready_advisory_artifact_from_directory(
            input_artifact_dir=args.input_artifact_dir,
            output_root=args.output_dir,
            generated_at=args.generated_at or _generated_at_utc(),
            allow_overwrite=args.allow_overwrite,
        )
    except ChatGPTReadyAdvisoryArtifactError as exc:
        print(str(exc), file=error_stream)
        return 2

    if args.emit_json:
        json.dump(result.manifest, output_stream, indent=2, sort_keys=True)
        output_stream.write("\n")
    else:
        output_stream.write(f"artifact_path={result.artifact_path.as_posix()}\n")
        output_stream.write(f"manifest_path={result.manifest_path.as_posix()}\n")

    return 0


def main() -> None:
    raise SystemExit(run_chatgpt_ready_advisory_artifact_command())


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-engine-chatgpt-ready-advisory-artifact",
        description=(
            "Compose a deterministic local ChatGPT-ready advisory artifact from "
            "explicit Market Engine JSON artifacts. The command performs no "
            "provider calls, LLM calls, delivery, broker actions, portfolio writes, "
            "or production writes."
        ),
    )
    parser.add_argument(
        "--input-artifact-dir",
        required=True,
        help=(
            "Directory containing structured_decision_output.json and optional "
            "ME-CI02/CI03/CI04, Governor, and Dispatch companion artifacts."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=CHATGPT_READY_ADVISORY_ARTIFACT_PATH_CATEGORY,
        help=(
            "Local non-production output root. Defaults to "
            "artifacts/market_engine/chatgpt_ready_advisory."
        ),
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Optional deterministic artifact generation timestamp.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing an existing local artifact directory.",
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help="Emit the persisted manifest as JSON instead of path lines.",
    )
    return parser


def _generated_at_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    main()
