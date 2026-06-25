from __future__ import annotations

import argparse
import sys
from typing import Mapping, Sequence, TextIO

from market_engine.source_refresh.cached_source_snapshot_importer import (
    CachedSourceSnapshotImportError,
    DEFAULT_CACHED_SOURCE_SNAPSHOT_IMPORT_ROOT,
    import_cached_source_snapshot,
)


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


def run_command(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        result = import_cached_source_snapshot(
            source_path=args.source_path,
            destination_root=args.destination_root,
            validated_at=args.validated_at,
        )
    except CachedSourceSnapshotImportError as exc:
        render_import_failure(exc, stdout=stderr)
        return 2

    render_import_success(result, stdout=stdout)
    return 0


def render_import_success(
    result: Mapping[str, object],
    *,
    stdout: TextIO,
) -> None:
    warnings = tuple(result["warnings"])
    imported_entities = ", ".join(result["imported_entities"]) or "none"
    print("Cached-source snapshot import completed", file=stdout)
    print("", file=stdout)
    print(f"Snapshot ID: {result['snapshot_id']}", file=stdout)
    print(f"Batch ID: {result['batch_id']}", file=stdout)
    print(f"Source family: {result['source_family']}", file=stdout)
    print(f"Source path: {result['source_path']}", file=stdout)
    print(f"Destination path: {result['destination_path']}", file=stdout)
    print(f"Manifest: {result['manifest_path']}", file=stdout)
    print(f"Imported entities: {imported_entities}", file=stdout)
    print(f"Validation: {result['validation_status']}", file=stdout)
    print(f"Warnings: {', '.join(warnings) if warnings else 'none'}", file=stdout)
    print(result["forbidden_side_effect_confirmation"], file=stdout)


def render_import_failure(
    error: CachedSourceSnapshotImportError,
    *,
    stdout: TextIO,
) -> None:
    print("Cached-source snapshot import failed", file=stdout)
    print("", file=stdout)
    print(f"Reason: {error.reason}", file=stdout)
    if error.source_path is not None:
        print(f"Source path: {error.source_path.as_posix()}", file=stdout)
    if error.expected_manifest is not None:
        print(f"Expected manifest: {error.expected_manifest.as_posix()}", file=stdout)
    if error.issues:
        print(f"Issues: {', '.join(error.issues)}", file=stdout)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-engine-import-cached-source-snapshot",
        description=(
            "Import one operator-supplied cached-source snapshot directory or "
            "manifest into the local cached-source snapshot workspace."
        ),
    )
    parser.add_argument(
        "--source-path",
        required=True,
        help="Operator-supplied snapshot directory or manifest.json file to import.",
    )
    parser.add_argument(
        "--destination-root",
        default=DEFAULT_CACHED_SOURCE_SNAPSHOT_IMPORT_ROOT.as_posix(),
        help=(
            "Local cached-source snapshot import root. Defaults to "
            "data/market_engine/cached_source_snapshots."
        ),
    )
    parser.add_argument(
        "--validated-at",
        default=None,
        help="Optional deterministic UTC timestamp for validation reports.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
