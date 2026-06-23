from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


MARKET_ENGINE_INTERPRETATION_REPORT_FORMAT_VERSION = (
    "market-engine-interpretation-report-v1"
)
DEFAULT_REPORT_OUTPUT_ROOT = Path("artifacts/market_engine")
REPORT_MARKDOWN_FILENAME = "market_engine_interpretation_report.md"
REPORT_SUMMARY_FILENAME = "market_engine_interpretation_report_summary.json"
EXPECTED_DRY_RUN_FORMAT_VERSION = "market-engine-end-to-end-dry-run-v1"
EXPECTED_LOCAL_ARTIFACT_FORMAT_VERSION = "market-engine-local-dry-run-artifact-v1"
NON_ACTIONABLE_BOUNDARY = (
    "This is a non-actionable interpretation report generated from cached-source "
    "local artifacts only."
)

FORBIDDEN_ADVISORY_TERMS = (
    "buy",
    "sell",
    "hold",
    "ranking",
    "target price",
    "stop-loss",
    "take-profit",
    "allocation",
    "urgency",
    "conviction",
    "score",
    "best stock",
    "top pick",
    "position sizing",
    "execution advice",
)


class MarketEngineInterpretationReportError(ValueError):
    pass


@dataclass(frozen=True)
class TickerArtifactInspection:
    ticker: str
    ticker_directory: str
    dry_run_path: str
    manifest_path: str
    included: bool
    skipped_reason: str | None
    artifact_format_version: str | None
    dry_run_format_version: str | None
    source_input_mode: str | None
    source_run_state: str | None
    blocked_stage: str | None
    stage_statuses: tuple[tuple[str, str], ...]
    missing_data_markers: tuple[str, ...]
    stale_data_markers: tuple[str, ...]
    provenance_references: dict[str, Any]
    output_families_present: tuple[str, ...]


@dataclass(frozen=True)
class MarketEngineInterpretationReportResult:
    report_format_version: str
    input_artifact_root: str
    report_run_id: str
    generated_at: str
    output_directory: str
    markdown_report_path: str
    summary_json_path: str
    discovered_ticker_directory_count: int
    parsed_ticker_count: int
    skipped_tickers: tuple[dict[str, str], ...]
    included_tickers: tuple[str, ...]
    output_families_present_across_universe: tuple[str, ...]
    non_actionable_boundary: str
    advisory_language_guardrail: tuple[str, ...]

    def to_summary_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_market_engine_interpretation_report(
    *,
    input_artifact_root: str | Path,
    output_root: str | Path = DEFAULT_REPORT_OUTPUT_ROOT,
    report_run_id: str | None = None,
    generated_at: str,
) -> MarketEngineInterpretationReportResult:
    input_root = _validated_input_root(Path(input_artifact_root))
    resolved_report_run_id = report_run_id or f"me-run22-human-readable-report-{input_root.name}"
    safe_report_run_id = _safe_path_segment(resolved_report_run_id, field_name="report_run_id")
    output_directory = _prepare_output_directory(Path(output_root), safe_report_run_id)
    inspections = _inspect_ticker_directories(input_root)
    included = tuple(inspection for inspection in inspections if inspection.included)
    skipped = tuple(
        {"ticker": inspection.ticker, "reason": inspection.skipped_reason or "unspecified"}
        for inspection in inspections
        if not inspection.included
    )
    output_families = tuple(
        sorted({family for inspection in included for family in inspection.output_families_present})
    )
    markdown_path = output_directory / REPORT_MARKDOWN_FILENAME
    summary_path = output_directory / REPORT_SUMMARY_FILENAME
    result = MarketEngineInterpretationReportResult(
        report_format_version=MARKET_ENGINE_INTERPRETATION_REPORT_FORMAT_VERSION,
        input_artifact_root=input_root.as_posix(),
        report_run_id=safe_report_run_id,
        generated_at=generated_at,
        output_directory=output_directory.as_posix(),
        markdown_report_path=markdown_path.as_posix(),
        summary_json_path=summary_path.as_posix(),
        discovered_ticker_directory_count=len(inspections),
        parsed_ticker_count=len(included),
        skipped_tickers=skipped,
        included_tickers=tuple(inspection.ticker for inspection in included),
        output_families_present_across_universe=output_families,
        non_actionable_boundary=NON_ACTIONABLE_BOUNDARY,
        advisory_language_guardrail=FORBIDDEN_ADVISORY_TERMS,
    )
    markdown = _render_markdown_report(result=result, inspections=inspections)
    _write_text(markdown_path, markdown)
    _write_json(summary_path, result.to_summary_payload())
    return result


def _inspect_ticker_directories(input_root: Path) -> tuple[TickerArtifactInspection, ...]:
    directories = tuple(sorted((path for path in input_root.iterdir() if path.is_dir()), key=lambda path: path.name))
    return tuple(_inspect_ticker_directory(path) for path in directories)


def _inspect_ticker_directory(path: Path) -> TickerArtifactInspection:
    ticker = path.name
    dry_run_path = path / "dry_run.json"
    manifest_path = path / "manifest.json"
    if not dry_run_path.exists():
        return _skipped_inspection(ticker=ticker, path=path, reason="dry_run.json is missing")
    if not manifest_path.exists():
        return _skipped_inspection(ticker=ticker, path=path, reason="manifest.json is missing")
    try:
        dry_run_artifact = _read_json_object(dry_run_path)
    except MarketEngineInterpretationReportError as exc:
        return _skipped_inspection(ticker=ticker, path=path, reason=str(exc))
    try:
        _read_json_object(manifest_path)
    except MarketEngineInterpretationReportError as exc:
        return _skipped_inspection(ticker=ticker, path=path, reason=str(exc))

    payload = dry_run_artifact.get("payload")
    if not isinstance(payload, Mapping):
        return _skipped_inspection(ticker=ticker, path=path, reason="dry_run payload is missing or malformed")

    stage_statuses = tuple(
        (str(stage.get("stage_name") or ""), str(stage.get("status") or ""))
        for stage in payload.get("stage_results") or ()
        if isinstance(stage, Mapping)
    )
    output_families = tuple(stage_name for stage_name, status in stage_statuses if stage_name and status)
    return TickerArtifactInspection(
        ticker=ticker,
        ticker_directory=path.as_posix(),
        dry_run_path=dry_run_path.as_posix(),
        manifest_path=manifest_path.as_posix(),
        included=True,
        skipped_reason=None,
        artifact_format_version=_optional_text(dry_run_artifact.get("artifact_format_version")),
        dry_run_format_version=_optional_text(
            dry_run_artifact.get("source_dry_run_format_version")
            or payload.get("dry_run_format_version")
        ),
        source_input_mode=_optional_text(dry_run_artifact.get("source_input_mode") or payload.get("input_mode")),
        source_run_state=_optional_text(dry_run_artifact.get("source_run_state") or payload.get("run_state")),
        blocked_stage=_optional_text(payload.get("blocked_stage")),
        stage_statuses=stage_statuses,
        missing_data_markers=_text_tuple(payload.get("missing_data_summary")),
        stale_data_markers=_text_tuple(payload.get("stale_data_summary")),
        provenance_references=_provenance_references(payload),
        output_families_present=output_families,
    )


def _skipped_inspection(*, ticker: str, path: Path, reason: str) -> TickerArtifactInspection:
    return TickerArtifactInspection(
        ticker=ticker,
        ticker_directory=path.as_posix(),
        dry_run_path=(path / "dry_run.json").as_posix(),
        manifest_path=(path / "manifest.json").as_posix(),
        included=False,
        skipped_reason=reason,
        artifact_format_version=None,
        dry_run_format_version=None,
        source_input_mode=None,
        source_run_state=None,
        blocked_stage=None,
        stage_statuses=(),
        missing_data_markers=(),
        stale_data_markers=(),
        provenance_references={},
        output_families_present=(),
    )


def _render_markdown_report(
    *,
    result: MarketEngineInterpretationReportResult,
    inspections: tuple[TickerArtifactInspection, ...],
) -> str:
    included = tuple(inspection for inspection in inspections if inspection.included)
    skipped = tuple(inspection for inspection in inspections if not inspection.included)
    lines: list[str] = [
        "# Market Engine Interpretation Report",
        "",
        f"Generated at: `{result.generated_at}`",
        "",
        f"Report run id: `{result.report_run_id}`",
        "",
        f"Input artifact root: `{result.input_artifact_root}`",
        "",
        f"Report format version: `{result.report_format_version}`",
        "",
        f"{NON_ACTIONABLE_BOUNDARY}",
        "",
        "## Scope And Safety",
        "",
        "This report is generated from cached-source local artifacts only.",
        "",
        "No provider calls were made while generating this report.",
        "",
        (
            "The report does not contain investment advice, external instructions, "
            "capital guidance, ordered ticker lists, confidence labels, timing labels, "
            "or market-participation guidance."
        ),
        "",
        "## Universe Summary",
        "",
        f"* Ticker directories discovered: `{result.discovered_ticker_directory_count}`",
        f"* Ticker artifacts parsed: `{result.parsed_ticker_count}`",
        f"* Ticker artifacts skipped: `{len(result.skipped_tickers)}`",
        f"* Included tickers: `{', '.join(result.included_tickers) or 'none'}`",
        f"* Skipped tickers: `{', '.join(item['ticker'] for item in result.skipped_tickers) or 'none'}`",
        "",
        "## Per-Ticker Artifact Presence",
        "",
        "| Ticker | dry_run.json | manifest.json | Parsed | Run state | Missing markers | Stale markers | Blocked stage |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for inspection in inspections:
        lines.append(
            "| "
            + " | ".join(
                (
                    inspection.ticker,
                    _yes_no(Path(inspection.dry_run_path).exists()),
                    _yes_no(Path(inspection.manifest_path).exists()),
                    _yes_no(inspection.included),
                    inspection.source_run_state or "not parsed",
                    str(len(inspection.missing_data_markers)),
                    str(len(inspection.stale_data_markers)),
                    inspection.blocked_stage or "none",
                )
            )
            + " |"
        )
    lines.extend(["", "## Per-Ticker Sections", ""])
    for inspection in inspections:
        lines.extend(_ticker_section(inspection))
    lines.extend(
        [
            "## Cross-Universe Observations",
            "",
            f"* Complete ticker artifacts: `{len(included)}`",
            f"* Skipped ticker artifacts: `{len(skipped)}`",
            f"* Tickers with missing-data markers: `{sum(1 for item in included if item.missing_data_markers)}`",
            f"* Tickers with stale-data markers: `{sum(1 for item in included if item.stale_data_markers)}`",
            f"* Tickers with blocked stages: `{sum(1 for item in included if item.blocked_stage)}`",
            f"* Output families present across included tickers: `{', '.join(result.output_families_present_across_universe) or 'none'}`",
            "",
            "No ordered comparison, preference, or action-oriented classification is produced by this report.",
            "",
            "## Readiness Assessment",
            "",
            "The ME-RUN20 artifacts are complete enough for human inspection because every included ticker has valid local wrapper artifacts and a completed end-to-end dry-run payload.",
            "",
            "The artifacts are structurally consistent enough to support the next interpretation/reporting sprint because the same wrapper keys, dry-run contract, input mode, stage families, and completion states appear across the included tickers.",
            "",
            "Actionability remains blocked by governance. These artifacts are local non-actionable dry-run outputs and must not be treated as investment instructions or market-participation analysis.",
            "",
            "## Recommended Next Sprint",
            "",
            "ME-OUT01 - Define readable operator report contract from dry-run artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def _ticker_section(inspection: TickerArtifactInspection) -> list[str]:
    lines = [
        f"### {inspection.ticker}",
        "",
        f"* Ticker directory: `{inspection.ticker_directory}`",
        f"* Dry-run artifact: `{inspection.dry_run_path}`",
        f"* Manifest artifact: `{inspection.manifest_path}`",
    ]
    if not inspection.included:
        lines.extend(["", f"Skipped reason: {inspection.skipped_reason}", ""])
        return lines

    completed_stages = tuple(stage for stage, status in inspection.stage_statuses if status == "completed")
    non_completed = tuple(
        f"{stage}:{status}"
        for stage, status in inspection.stage_statuses
        if status != "completed"
    )
    lines.extend(
        [
            f"* Artifact format: `{inspection.artifact_format_version}`",
            f"* Dry-run format: `{inspection.dry_run_format_version}`",
            f"* Input mode: `{inspection.source_input_mode}`",
            f"* Run state: `{inspection.source_run_state}`",
            f"* Output families present: `{', '.join(inspection.output_families_present) or 'none'}`",
            f"* Completed stages: `{len(completed_stages)}`",
            f"* Non-completed stages: `{', '.join(non_completed) or 'none'}`",
            f"* Missing-data notes: `{', '.join(inspection.missing_data_markers) or 'none'}`",
            f"* Stale-data notes: `{', '.join(inspection.stale_data_markers) or 'none'}`",
            f"* Blocked stage: `{inspection.blocked_stage or 'none'}`",
            f"* Provenance references: `{_short_provenance_text(inspection.provenance_references)}`",
            "",
            "Neutral interpretation: the artifact is structurally available for human review. The report preserves the observed review states without adding new conclusions.",
            "",
        ]
    )
    return lines


def _provenance_references(payload: Mapping[str, Any]) -> dict[str, Any]:
    delivery = payload.get("delivery_report_reference")
    if isinstance(delivery, Mapping):
        cached = delivery.get("cached_source_reference")
        if isinstance(cached, Mapping):
            return {
                "source_snapshot_reference": cached.get("source_snapshot_reference"),
                "source_snapshot_path": cached.get("source_snapshot_path"),
                "source_snapshot_root": cached.get("source_snapshot_root"),
            }
    return {"dry_run_id": payload.get("dry_run_id"), "ticker": payload.get("ticker")}


def _short_provenance_text(value: Mapping[str, Any]) -> str:
    parts = [f"{key}={nested}" for key, nested in sorted(value.items()) if nested]
    return "; ".join(parts) if parts else "none"


def _validated_input_root(path: Path) -> Path:
    if any(part == ".." for part in path.parts):
        raise MarketEngineInterpretationReportError("Input artifact root may not contain parent traversal.")
    resolved = path.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise MarketEngineInterpretationReportError(f"Input artifact root does not exist: {path}")
    return resolved


def _prepare_output_directory(output_root: Path, report_run_id: str) -> Path:
    if any(part == ".." for part in output_root.parts):
        raise MarketEngineInterpretationReportError("Output root may not contain parent traversal.")
    root = output_root.resolve()
    output_directory = (root / report_run_id).resolve()
    try:
        output_directory.relative_to(root)
    except ValueError as exc:
        raise MarketEngineInterpretationReportError("Output path escaped the output root.") from exc
    if output_directory.exists():
        raise MarketEngineInterpretationReportError(f"Report output directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    return output_directory


def _safe_path_segment(value: str, *, field_name: str) -> str:
    safe = value.strip()
    if not safe or safe in {".", ".."} or "/" in safe or "\\" in safe:
        raise MarketEngineInterpretationReportError(f"{field_name} is not a safe path segment.")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(character not in allowed for character in safe):
        raise MarketEngineInterpretationReportError(f"{field_name} contains unsafe characters.")
    return safe


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MarketEngineInterpretationReportError(f"{path.name} contains invalid JSON") from exc
    except OSError as exc:
        raise MarketEngineInterpretationReportError(f"{path.name} cannot be read") from exc
    if not isinstance(payload, dict):
        raise MarketEngineInterpretationReportError(f"{path.name} must contain a JSON object")
    return payload


def _write_text(path: Path, text: str) -> None:
    if path.exists():
        raise MarketEngineInterpretationReportError(f"Report file already exists: {path}")
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise MarketEngineInterpretationReportError(f"Summary file already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


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
        result = build_market_engine_interpretation_report(
            input_artifact_root=args.input_artifact_root,
            output_root=args.output_root,
            report_run_id=args.report_run_id,
            generated_at=args.generated_at,
        )
    except MarketEngineInterpretationReportError as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    print(f"markdown_report_path={result.markdown_report_path}", file=stdout)
    print(f"summary_json_path={result.summary_json_path}", file=stdout)
    print(f"included_tickers={','.join(result.included_tickers)}", file=stdout)
    print(f"skipped_ticker_count={len(result.skipped_tickers)}", file=stdout)
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a non-actionable Market Engine interpretation report from local dry-run artifacts."
    )
    parser.add_argument("--input-artifact-root", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_REPORT_OUTPUT_ROOT))
    parser.add_argument("--report-run-id", default=None)
    parser.add_argument("--generated-at", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
