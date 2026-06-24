from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


MARKET_ENGINE_READABLE_OPERATOR_REPORT_FORMAT_VERSION = (
    "market-engine-readable-operator-report-v1"
)
DEFAULT_OPERATOR_REPORT_OUTPUT_ROOT = Path("artifacts/market_engine")
OPERATOR_REPORT_MARKDOWN_FILENAME = "operator_report.md"
OPERATOR_REPORT_SUMMARY_FILENAME = "operator_report_summary.json"
EXPECTED_LOCAL_ARTIFACT_FORMAT_VERSION = "market-engine-local-dry-run-artifact-v1"
EXPECTED_LOCAL_MANIFEST_FORMAT_VERSION = (
    "market-engine-local-dry-run-artifact-manifest-v1"
)
EXPECTED_DRY_RUN_FORMAT_VERSION = "market-engine-end-to-end-dry-run-v1"
EXPECTED_INTERPRETATION_REPORT_FORMAT_VERSION = (
    "market-engine-interpretation-report-v1"
)
NON_ACTIONABLE_BOUNDARY = (
    "This operator report is local, non-production, and non-actionable. It "
    "summarizes existing cached-source dry-run artifacts only."
)


class ReadableOperatorReportError(ValueError):
    pass


@dataclass(frozen=True)
class OperatorTickerInspection:
    ticker: str
    ticker_directory: str
    dry_run_path: str
    manifest_path: str
    included: bool
    skipped_reason: str | None
    dry_run_present: bool
    manifest_present: bool
    dry_run_json_status: str
    manifest_json_status: str
    artifact_format_version: str | None
    dry_run_format_version: str | None
    manifest_format_version: str | None
    input_mode: str | None
    run_state: str | None
    completed_stages: tuple[str, ...]
    non_completed_stages: tuple[str, ...]
    output_families_present: tuple[str, ...]
    missing_data_notes: tuple[str, ...]
    stale_data_notes: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    blocked_stage: str | None
    provenance_references: tuple[str, ...]
    numeric_zero_values_present: bool


@dataclass(frozen=True)
class ReadableOperatorReportResult:
    report_format_version: str
    report_run_id: str
    generated_at: str | None
    input_artifact_root: str
    interpretation_report_root: str | None
    output_directory: str
    markdown_report_path: str
    summary_json_path: str
    included_tickers: tuple[str, ...]
    skipped_tickers: tuple[dict[str, str], ...]
    blocked_tickers: tuple[dict[str, Any], ...]
    completed_tickers: tuple[str, ...]
    output_families_present: tuple[str, ...]
    missing_data_notes_present: bool
    stale_data_notes_present: bool
    blocked_notes_present: bool
    provenance_references_present: bool
    non_actionable_boundary: bool
    advisory_language_guardrail: dict[str, bool]
    discovered_ticker_directory_count: int
    artifact_present_tickers: tuple[str, ...]
    artifact_missing_tickers: tuple[dict[str, str], ...]
    malformed_artifact_tickers: tuple[dict[str, str], ...]

    def to_summary_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_readable_operator_report(
    *,
    input_artifact_root: str | Path,
    output_root: str | Path = DEFAULT_OPERATOR_REPORT_OUTPUT_ROOT,
    report_run_id: str,
    generated_at: str | None = None,
    interpretation_report_root: str | Path | None = None,
) -> ReadableOperatorReportResult:
    input_root = _validated_input_root(Path(input_artifact_root))
    safe_report_run_id = _safe_path_segment(report_run_id, field_name="report_run_id")
    interpretation_root = _validated_optional_interpretation_root(
        interpretation_report_root,
    )
    output_directory = _prepare_output_directory(Path(output_root), safe_report_run_id)
    inspections = _inspect_ticker_directories(input_root)
    included = tuple(item for item in inspections if item.included)
    skipped = tuple(
        {"ticker": item.ticker, "reason": item.skipped_reason or "unspecified"}
        for item in inspections
        if not item.included
    )
    blocked = tuple(
        {
            "ticker": item.ticker,
            "blocked_stage": item.blocked_stage,
            "blocked_reasons": list(item.blocked_reasons),
        }
        for item in included
        if item.blocked_stage or item.blocked_reasons or _is_blocked_state(item.run_state)
    )
    completed = tuple(
        item.ticker
        for item in included
        if item.run_state in {"dry_run_completed", "completed"}
        and not item.blocked_stage
        and not item.blocked_reasons
    )
    output_families = tuple(
        sorted(
            {
                family
                for item in included
                for family in item.output_families_present
            }
        )
    )
    artifact_missing = tuple(
        {"ticker": item.ticker, "reason": item.skipped_reason or "artifact missing"}
        for item in inspections
        if not item.dry_run_present or not item.manifest_present
    )
    malformed = tuple(
        {"ticker": item.ticker, "reason": item.skipped_reason or "malformed artifact"}
        for item in inspections
        if item.dry_run_json_status == "malformed"
        or item.manifest_json_status == "malformed"
    )
    markdown_path = output_directory / OPERATOR_REPORT_MARKDOWN_FILENAME
    summary_path = output_directory / OPERATOR_REPORT_SUMMARY_FILENAME
    result = ReadableOperatorReportResult(
        report_format_version=MARKET_ENGINE_READABLE_OPERATOR_REPORT_FORMAT_VERSION,
        report_run_id=safe_report_run_id,
        generated_at=generated_at,
        input_artifact_root=input_root.as_posix(),
        interpretation_report_root=(
            interpretation_root.as_posix() if interpretation_root is not None else None
        ),
        output_directory=output_directory.as_posix(),
        markdown_report_path=markdown_path.as_posix(),
        summary_json_path=summary_path.as_posix(),
        included_tickers=tuple(item.ticker for item in included),
        skipped_tickers=skipped,
        blocked_tickers=blocked,
        completed_tickers=completed,
        output_families_present=output_families,
        missing_data_notes_present=any(item.missing_data_notes for item in included),
        stale_data_notes_present=any(item.stale_data_notes for item in included),
        blocked_notes_present=bool(blocked),
        provenance_references_present=any(
            item.provenance_references for item in included
        ),
        non_actionable_boundary=True,
        advisory_language_guardrail={
            "forbidden_action_terms_checked": True,
            "operator_report_contains_trading_instruction": False,
        },
        discovered_ticker_directory_count=len(inspections),
        artifact_present_tickers=tuple(
            item.ticker
            for item in inspections
            if item.dry_run_present and item.manifest_present
        ),
        artifact_missing_tickers=artifact_missing,
        malformed_artifact_tickers=malformed,
    )
    markdown = _render_markdown_report(result=result, inspections=inspections)
    _write_text(markdown_path, markdown)
    _write_json(summary_path, result.to_summary_payload())
    return result


def _inspect_ticker_directories(input_root: Path) -> tuple[OperatorTickerInspection, ...]:
    return tuple(
        _inspect_ticker_directory(path)
        for path in sorted(
            (child for child in input_root.iterdir() if child.is_dir()),
            key=lambda child: child.name,
        )
    )


def _inspect_ticker_directory(path: Path) -> OperatorTickerInspection:
    ticker = path.name
    dry_run_path = path / "dry_run.json"
    manifest_path = path / "manifest.json"
    if not dry_run_path.exists():
        return _skipped_inspection(
            ticker=ticker,
            path=path,
            reason="dry_run.json is missing",
            dry_run_present=False,
            manifest_present=manifest_path.exists(),
        )
    if not manifest_path.exists():
        return _skipped_inspection(
            ticker=ticker,
            path=path,
            reason="manifest.json is missing",
            dry_run_present=True,
            manifest_present=False,
        )
    try:
        dry_run_artifact = _read_json_object(dry_run_path)
    except ReadableOperatorReportError as exc:
        return _skipped_inspection(
            ticker=ticker,
            path=path,
            reason=str(exc),
            dry_run_present=True,
            manifest_present=True,
            dry_run_json_status="malformed",
        )
    try:
        manifest = _read_json_object(manifest_path)
    except ReadableOperatorReportError as exc:
        return _skipped_inspection(
            ticker=ticker,
            path=path,
            reason=str(exc),
            dry_run_present=True,
            manifest_present=True,
            manifest_json_status="malformed",
        )

    artifact_format_version = _optional_text(
        dry_run_artifact.get("artifact_format_version")
    )
    if artifact_format_version != EXPECTED_LOCAL_ARTIFACT_FORMAT_VERSION:
        return _skipped_inspection(
            ticker=ticker,
            path=path,
            reason="unsupported dry_run artifact format version",
            dry_run_present=True,
            manifest_present=True,
        )
    manifest_format_version = _optional_text(manifest.get("manifest_format_version"))
    if manifest_format_version != EXPECTED_LOCAL_MANIFEST_FORMAT_VERSION:
        return _skipped_inspection(
            ticker=ticker,
            path=path,
            reason="unsupported manifest format version",
            dry_run_present=True,
            manifest_present=True,
        )

    payload = dry_run_artifact.get("payload")
    if not isinstance(payload, Mapping):
        return _skipped_inspection(
            ticker=ticker,
            path=path,
            reason="dry_run payload is missing or malformed",
            dry_run_present=True,
            manifest_present=True,
        )
    dry_run_format_version = _optional_text(
        dry_run_artifact.get("source_dry_run_format_version")
        or payload.get("dry_run_format_version")
    )
    if dry_run_format_version != EXPECTED_DRY_RUN_FORMAT_VERSION:
        return _skipped_inspection(
            ticker=ticker,
            path=path,
            reason="unsupported dry-run format version",
            dry_run_present=True,
            manifest_present=True,
        )

    stage_statuses = _stage_statuses(payload)
    completed_stages = tuple(
        stage for stage, status in stage_statuses if status == "completed"
    )
    non_completed = tuple(
        f"{stage}:{status}"
        for stage, status in stage_statuses
        if status != "completed"
    )
    output_families = tuple(stage for stage, _status in stage_statuses if stage)
    blocked_reasons = tuple(
        sorted(
            set(
                _text_tuple(payload.get("blocked_reasons"))
                + _text_tuple(payload.get("missing_data_summary"))
                if _is_blocked_state(_optional_text(payload.get("run_state")))
                else _text_tuple(payload.get("blocked_reasons"))
            )
        )
    )
    return OperatorTickerInspection(
        ticker=ticker,
        ticker_directory=path.as_posix(),
        dry_run_path=dry_run_path.as_posix(),
        manifest_path=manifest_path.as_posix(),
        included=True,
        skipped_reason=None,
        dry_run_present=True,
        manifest_present=True,
        dry_run_json_status="parsed",
        manifest_json_status="parsed",
        artifact_format_version=artifact_format_version,
        dry_run_format_version=dry_run_format_version,
        manifest_format_version=manifest_format_version,
        input_mode=_optional_text(
            dry_run_artifact.get("source_input_mode") or payload.get("input_mode")
        ),
        run_state=_optional_text(
            dry_run_artifact.get("source_run_state") or payload.get("run_state")
        ),
        completed_stages=completed_stages,
        non_completed_stages=non_completed,
        output_families_present=output_families,
        missing_data_notes=tuple(sorted(_text_tuple(payload.get("missing_data_summary")))),
        stale_data_notes=tuple(sorted(_text_tuple(payload.get("stale_data_summary")))),
        blocked_reasons=blocked_reasons,
        blocked_stage=_optional_text(payload.get("blocked_stage")),
        provenance_references=_provenance_references(payload),
        numeric_zero_values_present=_contains_numeric_zero(payload),
    )


def _skipped_inspection(
    *,
    ticker: str,
    path: Path,
    reason: str,
    dry_run_present: bool,
    manifest_present: bool,
    dry_run_json_status: str = "not_parsed",
    manifest_json_status: str = "not_parsed",
) -> OperatorTickerInspection:
    return OperatorTickerInspection(
        ticker=ticker,
        ticker_directory=path.as_posix(),
        dry_run_path=(path / "dry_run.json").as_posix(),
        manifest_path=(path / "manifest.json").as_posix(),
        included=False,
        skipped_reason=reason,
        dry_run_present=dry_run_present,
        manifest_present=manifest_present,
        dry_run_json_status=dry_run_json_status,
        manifest_json_status=manifest_json_status,
        artifact_format_version=None,
        dry_run_format_version=None,
        manifest_format_version=None,
        input_mode=None,
        run_state=None,
        completed_stages=(),
        non_completed_stages=(),
        output_families_present=(),
        missing_data_notes=(),
        stale_data_notes=(),
        blocked_reasons=(),
        blocked_stage=None,
        provenance_references=(),
        numeric_zero_values_present=False,
    )


def _render_markdown_report(
    *,
    result: ReadableOperatorReportResult,
    inspections: tuple[OperatorTickerInspection, ...],
) -> str:
    lines: list[str] = [
        "# Readable Operator Report",
        "",
        "## 1. Report Metadata",
        "",
        f"* Report format version: `{result.report_format_version}`",
        f"* Report run id: `{result.report_run_id}`",
        f"* Generated at: `{result.generated_at or 'not supplied'}`",
        f"* Included ticker count: `{len(result.included_tickers)}`",
        f"* Skipped ticker count: `{len(result.skipped_tickers)}`",
        f"* Blocked ticker count: `{len(result.blocked_tickers)}`",
        f"* Completed ticker count: `{len(result.completed_tickers)}`",
        "* Local-only marker: `true`",
        "",
        "## 2. Source Artifact Boundary",
        "",
        f"* Input artifact root: `{result.input_artifact_root}`",
        f"* Interpretation report root: `{result.interpretation_report_root or 'not supplied'}`",
        "* Approved source: existing local dry-run artifacts.",
        "* Upstream artifact mutation: `false`",
        "",
        "## 3. Non-Actionable Boundary",
        "",
        NON_ACTIONABLE_BOUNDARY,
        "",
        (
            "The report preserves local artifact states and does not create market "
            "participation guidance, capital guidance, ordered ticker preference, "
            "external-channel output, or broker-facing content."
        ),
        "",
        "## 4. Universe Coverage",
        "",
        f"* Ticker directories discovered: `{result.discovered_ticker_directory_count}`",
        f"* Included tickers: `{', '.join(result.included_tickers) or 'none'}`",
        f"* Skipped tickers: `{_skipped_text(result.skipped_tickers)}`",
        f"* Blocked tickers: `{_blocked_text(result.blocked_tickers)}`",
        "",
        "## 5. Artifact Integrity Summary",
        "",
        "| Ticker | dry_run.json | manifest.json | dry_run parse | manifest parse | Artifact format | Dry-run format | Manifest format |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for item in inspections:
        lines.append(
            "| "
            + " | ".join(
                (
                    item.ticker,
                    _yes_no(item.dry_run_present),
                    _yes_no(item.manifest_present),
                    item.dry_run_json_status,
                    item.manifest_json_status,
                    item.artifact_format_version or "not parsed",
                    item.dry_run_format_version or "not parsed",
                    item.manifest_format_version or "not parsed",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 6. Stage Completion Summary",
            "",
            f"* Completed tickers: `{', '.join(result.completed_tickers) or 'none'}`",
            f"* Output families present: `{', '.join(result.output_families_present) or 'none'}`",
            "",
            "| Ticker | Run state | Input mode | Completed stages | Non-completed stages |",
            "|---|---|---|---:|---|",
        ]
    )
    for item in inspections:
        lines.append(
            "| "
            + " | ".join(
                (
                    item.ticker,
                    item.run_state or "not parsed",
                    item.input_mode or "not parsed",
                    str(len(item.completed_stages)),
                    ", ".join(item.non_completed_stages) or "none",
                )
            )
            + " |"
        )
    lines.extend(["", "## 7. Per-Ticker Operator Summaries", ""])
    for item in inspections:
        lines.extend(_ticker_markdown(item))
    lines.extend(
        [
            "## 8. Missing-Data And Stale-Data Notes",
            "",
            f"* Missing-data notes present: `{_yes_no(result.missing_data_notes_present)}`",
            f"* Stale-data notes present: `{_yes_no(result.stale_data_notes_present)}`",
            f"* Tickers with numeric zero values preserved: `{', '.join(item.ticker for item in inspections if item.numeric_zero_values_present) or 'none'}`",
            "",
            "## 9. Blocked And Skipped Ticker Notes",
            "",
            f"* Blocked notes present: `{_yes_no(result.blocked_notes_present)}`",
            f"* Skipped ticker details: `{_skipped_text(result.skipped_tickers)}`",
            f"* Blocked ticker details: `{_blocked_text(result.blocked_tickers)}`",
            "",
            "## 10. Provenance Summary",
            "",
            f"* Provenance references present: `{_yes_no(result.provenance_references_present)}`",
            f"* Artifact-present tickers: `{', '.join(result.artifact_present_tickers) or 'none'}`",
            f"* Artifact-missing tickers: `{_skipped_text(result.artifact_missing_tickers)}`",
            f"* Malformed-artifact tickers: `{_skipped_text(result.malformed_artifact_tickers)}`",
            "",
            "## 11. Human-Review Checklist",
            "",
            "* Artifact root is readable.",
            "* Ticker directories were inspected.",
            "* Required local artifact files were checked.",
            "* JSON parse status was recorded.",
            "* Output families were summarized.",
            "* Missing-data notes were preserved.",
            "* Stale-data notes were preserved.",
            "* Blocked and skipped tickers were preserved.",
            "* Provenance references were preserved where available.",
            "* Non-actionable boundary is visible.",
            "",
            "## 12. Safe Next-Step Candidate",
            "",
            "ME-CANDIDATE01 - Define non-actionable candidate classification contract.",
            "",
            "## 13. Appendix: Machine-Readable Summary Reference",
            "",
            f"* Summary JSON: `{result.summary_json_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def _ticker_markdown(item: OperatorTickerInspection) -> list[str]:
    lines = [
        f"### {item.ticker}",
        "",
        f"* Ticker directory: `{item.ticker_directory}`",
        f"* Dry-run artifact: `{item.dry_run_path}`",
        f"* Manifest artifact: `{item.manifest_path}`",
    ]
    if not item.included:
        lines.extend(["", f"* Skipped reason: `{item.skipped_reason}`", ""])
        return lines
    lines.extend(
        [
            f"* Run state: `{item.run_state}`",
            f"* Input mode: `{item.input_mode}`",
            f"* Output families present: `{', '.join(item.output_families_present) or 'none'}`",
            f"* Completed stages: `{', '.join(item.completed_stages) or 'none'}`",
            f"* Non-completed stages: `{', '.join(item.non_completed_stages) or 'none'}`",
            f"* Missing-data notes: `{', '.join(item.missing_data_notes) or 'none'}`",
            f"* Stale-data notes: `{', '.join(item.stale_data_notes) or 'none'}`",
            f"* Blocked reasons: `{', '.join(item.blocked_reasons) or 'none'}`",
            f"* Blocked stage: `{item.blocked_stage or 'none'}`",
            f"* Numeric zero values preserved: `{_yes_no(item.numeric_zero_values_present)}`",
            f"* Provenance references: `{', '.join(item.provenance_references) or 'none'}`",
            "",
            "Operator summary: local artifacts are structurally available for human review, and observed stage states are preserved without new conclusions.",
            "",
        ]
    )
    return lines


def _stage_statuses(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    raw_stages = payload.get("stage_results")
    if not isinstance(raw_stages, Sequence) or isinstance(raw_stages, (str, bytes)):
        return ()
    return tuple(
        (str(stage.get("stage_name") or ""), str(stage.get("status") or ""))
        for stage in raw_stages
        if isinstance(stage, Mapping)
    )


def _provenance_references(payload: Mapping[str, Any]) -> tuple[str, ...]:
    references: set[str] = set()
    dry_run_id = payload.get("dry_run_id")
    ticker = payload.get("ticker")
    if dry_run_id:
        references.add(f"dry_run_id={dry_run_id}")
    if ticker:
        references.add(f"ticker={ticker}")
    delivery_reference = payload.get("delivery_report_reference")
    if isinstance(delivery_reference, Mapping):
        cached = delivery_reference.get("cached_source_reference")
        if isinstance(cached, Mapping):
            for key in (
                "source_snapshot_reference",
                "source_snapshot_path",
                "source_snapshot_root",
            ):
                value = cached.get(key)
                if value:
                    references.add(f"{key}={value}")
    return tuple(sorted(references))


def _validated_input_root(path: Path) -> Path:
    if any(part == ".." for part in path.parts):
        raise ReadableOperatorReportError(
            "Input artifact root may not contain parent traversal."
        )
    resolved = path.resolve()
    if not resolved.exists():
        raise ReadableOperatorReportError(f"Input artifact root does not exist: {path}")
    if not resolved.is_dir():
        raise ReadableOperatorReportError(
            f"Input artifact root is not a directory: {path}"
        )
    return resolved


def _validated_optional_interpretation_root(
    path: str | Path | None,
) -> Path | None:
    if path is None:
        return None
    root = _validated_input_root(Path(path))
    summary_path = root / "market_engine_interpretation_report_summary.json"
    if summary_path.exists():
        summary = _read_json_object(summary_path)
        if (
            summary.get("report_format_version")
            != EXPECTED_INTERPRETATION_REPORT_FORMAT_VERSION
        ):
            raise ReadableOperatorReportError(
                "Interpretation report summary uses an unsupported format version."
            )
    return root


def _prepare_output_directory(output_root: Path, report_run_id: str) -> Path:
    if any(part == ".." for part in output_root.parts):
        raise ReadableOperatorReportError("Output root may not contain parent traversal.")
    root = output_root.resolve()
    output_directory = (root / report_run_id).resolve()
    try:
        output_directory.relative_to(root)
    except ValueError as exc:
        raise ReadableOperatorReportError(
            "Operator report output path escaped the output root."
        ) from exc
    if output_directory.exists():
        raise ReadableOperatorReportError(
            f"Operator report output directory already exists: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    return output_directory


def _safe_path_segment(value: str, *, field_name: str) -> str:
    safe = value.strip()
    if not safe or safe in {".", ".."} or "/" in safe or "\\" in safe:
        raise ReadableOperatorReportError(f"{field_name} is not a safe path segment.")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(character not in allowed for character in safe):
        raise ReadableOperatorReportError(f"{field_name} contains unsafe characters.")
    return safe


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReadableOperatorReportError(f"{path.name} contains invalid JSON") from exc
    except OSError as exc:
        raise ReadableOperatorReportError(f"{path.name} cannot be read") from exc
    if not isinstance(payload, dict):
        raise ReadableOperatorReportError(f"{path.name} must contain a JSON object")
    return payload


def _write_text(path: Path, text: str) -> None:
    if path.exists():
        raise ReadableOperatorReportError(f"Operator report file already exists: {path}")
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ReadableOperatorReportError(f"Operator summary file already exists: {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _contains_numeric_zero(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)) and value == 0:
        return True
    if isinstance(value, Mapping):
        return any(_contains_numeric_zero(nested) for nested in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_numeric_zero(nested) for nested in value)
    return False


def _is_blocked_state(value: str | None) -> bool:
    return bool(value and "blocked" in value)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _skipped_text(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "none"
    return "; ".join(
        f"{item.get('ticker', 'unknown')}:{item.get('reason', 'unspecified')}"
        for item in items
    )


def _blocked_text(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "none"
    parts = []
    for item in items:
        reasons = item.get("blocked_reasons") or ()
        reason_text = ",".join(str(reason) for reason in reasons) or "unspecified"
        parts.append(
            f"{item.get('ticker', 'unknown')}:{item.get('blocked_stage') or reason_text}"
        )
    return "; ".join(parts)


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
        result = build_readable_operator_report(
            input_artifact_root=args.input_artifact_root,
            output_root=args.output_root,
            report_run_id=args.report_run_id,
            generated_at=args.generated_at,
            interpretation_report_root=args.interpretation_report_root,
        )
    except ReadableOperatorReportError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    print(f"operator_report_path={result.markdown_report_path}", file=stdout)
    print(f"summary_json_path={result.summary_json_path}", file=stdout)
    print(f"included_tickers={','.join(result.included_tickers)}", file=stdout)
    print(f"skipped_ticker_count={len(result.skipped_tickers)}", file=stdout)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a non-actionable readable operator report from local Market Engine dry-run artifacts."
    )
    parser.add_argument("--input-artifact-root", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OPERATOR_REPORT_OUTPUT_ROOT))
    parser.add_argument("--report-run-id", required=True)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--interpretation-report-root", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
