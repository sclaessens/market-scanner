from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SUPPORTED_ARTIFACT_FORMAT_VERSION = "market-engine-local-dry-run-artifact-v1"
SUPPORTED_ARTIFACT_TYPE = "market_engine_end_to_end_dry_run"
CANONICAL_SELECTION_RULE = (
    "valid artifacts before invalid artifacts",
    "newest artifact_created_at",
    "newest file modified time",
    "lexicographically smallest path",
)


@dataclass(frozen=True)
class ArtifactCandidate:
    ticker: str | None
    artifact_path: str
    artifact_format_version: str | None
    artifact_type: str | None
    artifact_created_at: str | None
    dry_run_id: str | None
    input_mode: str | None
    valid: bool
    invalid_reasons: tuple[str, ...]
    sha256: str | None
    file_size_bytes: int | None
    file_modified_time: float | None
    payload: Mapping[str, Any]

    def to_discovery_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "artifact_path": self.artifact_path,
            "artifact_format_version": self.artifact_format_version,
            "artifact_type": self.artifact_type,
            "artifact_created_at": self.artifact_created_at,
            "dry_run_id": self.dry_run_id,
            "input_mode": self.input_mode,
            "valid": self.valid,
            "invalid_reasons": list(self.invalid_reasons),
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
        }


@dataclass(frozen=True)
class DiscoveryResult:
    artifact_root: str
    files_scanned: int
    json_files_seen: int
    candidates: tuple[ArtifactCandidate, ...]
    failures: tuple[dict[str, Any], ...]

    @property
    def valid_candidates(self) -> tuple[ArtifactCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.valid)

    @property
    def invalid_candidates(self) -> tuple[ArtifactCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if not candidate.valid)

    def summary_dict(self) -> dict[str, Any]:
        tickers_seen = {candidate.ticker for candidate in self.candidates if candidate.ticker}
        by_ticker: dict[str, int] = {}
        for candidate in self.candidates:
            if candidate.ticker:
                by_ticker[candidate.ticker] = by_ticker.get(candidate.ticker, 0) + 1
        duplicate_ticker_candidates = sum(max(0, count - 1) for count in by_ticker.values())
        return {
            "artifact_root": self.artifact_root,
            "files_scanned": self.files_scanned,
            "json_files_seen": self.json_files_seen,
            "dry_run_artifacts_seen": len(self.candidates),
            "valid_dry_run_artifacts": len(self.valid_candidates),
            "invalid_dry_run_artifacts": len(self.invalid_candidates),
            "tickers_seen": len(tickers_seen),
            "duplicate_ticker_candidates": duplicate_ticker_candidates,
            "selection_rule": list(CANONICAL_SELECTION_RULE),
        }


def discover_dry_run_artifacts(
    artifact_root: str | Path,
    *,
    tickers: Iterable[str] | None = None,
    max_artifacts: int | None = None,
    include_invalid: bool = True,
) -> DiscoveryResult:
    root = Path(artifact_root)
    ticker_filter = {ticker.upper() for ticker in tickers or ()}
    files_scanned = 0
    json_files_seen = 0
    candidates: list[ArtifactCandidate] = []
    failures: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        files_scanned += 1
        if path.suffix.lower() != ".json":
            continue
        json_files_seen += 1
        if path.name != "dry_run.json":
            continue
        candidate, failure = _candidate_from_path(path)
        if failure is not None:
            failures.append(failure)
        if ticker_filter and candidate.ticker and candidate.ticker.upper() not in ticker_filter:
            continue
        if candidate.valid or include_invalid:
            candidates.append(candidate)
        if max_artifacts is not None and len(candidates) >= max_artifacts:
            break

    return DiscoveryResult(
        artifact_root=root.as_posix(),
        files_scanned=files_scanned,
        json_files_seen=json_files_seen,
        candidates=tuple(candidates),
        failures=tuple(failures),
    )


def select_canonical_artifacts(
    candidates: Iterable[ArtifactCandidate],
) -> dict[str, ArtifactCandidate]:
    by_ticker: dict[str, list[ArtifactCandidate]] = {}
    for candidate in candidates:
        if not candidate.ticker:
            continue
        by_ticker.setdefault(candidate.ticker, []).append(candidate)
    return {
        ticker: sorted(ticker_candidates, key=_canonical_sort_key)[0]
        for ticker, ticker_candidates in by_ticker.items()
    }


def _candidate_from_path(path: Path) -> tuple[ArtifactCandidate, dict[str, Any] | None]:
    stat = path.stat()
    file_size = stat.st_size
    modified_time = stat.st_mtime
    sha256 = _sha256(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        candidate = ArtifactCandidate(
            ticker=None,
            artifact_path=path.as_posix(),
            artifact_format_version=None,
            artifact_type=None,
            artifact_created_at=None,
            dry_run_id=None,
            input_mode=None,
            valid=False,
            invalid_reasons=("invalid_json",),
            sha256=sha256,
            file_size_bytes=file_size,
            file_modified_time=modified_time,
            payload={},
        )
        return candidate, {
            "path": path.as_posix(),
            "failure_type": "invalid_json",
            "message": str(exc),
        }

    if not isinstance(data, dict):
        return _invalid_candidate(
            path,
            sha256,
            file_size,
            modified_time,
            None,
            None,
            None,
            ("json_root_not_object",),
        ), None

    payload = data.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    ticker = _string_or_none(payload.get("ticker"))
    artifact_format_version = _string_or_none(data.get("artifact_format_version"))
    artifact_type = _string_or_none(data.get("artifact_type"))
    artifact_created_at = _string_or_none(data.get("artifact_created_at"))
    dry_run_id = _string_or_none(payload.get("dry_run_id"))
    input_mode = _string_or_none(payload.get("input_mode"))

    invalid_reasons: list[str] = []
    if artifact_format_version != SUPPORTED_ARTIFACT_FORMAT_VERSION:
        invalid_reasons.append("unsupported_artifact_format_version")
    if artifact_type != SUPPORTED_ARTIFACT_TYPE:
        invalid_reasons.append("unsupported_artifact_type")
    if not ticker:
        invalid_reasons.append("missing_ticker")

    return ArtifactCandidate(
        ticker=ticker.upper() if ticker else None,
        artifact_path=path.as_posix(),
        artifact_format_version=artifact_format_version,
        artifact_type=artifact_type,
        artifact_created_at=artifact_created_at,
        dry_run_id=dry_run_id,
        input_mode=input_mode,
        valid=not invalid_reasons,
        invalid_reasons=tuple(invalid_reasons),
        sha256=sha256,
        file_size_bytes=file_size,
        file_modified_time=modified_time,
        payload=payload,
    ), None


def _invalid_candidate(
    path: Path,
    sha256: str,
    file_size: int,
    modified_time: float,
    ticker: str | None,
    artifact_format_version: str | None,
    artifact_type: str | None,
    invalid_reasons: tuple[str, ...],
) -> ArtifactCandidate:
    return ArtifactCandidate(
        ticker=ticker,
        artifact_path=path.as_posix(),
        artifact_format_version=artifact_format_version,
        artifact_type=artifact_type,
        artifact_created_at=None,
        dry_run_id=None,
        input_mode=None,
        valid=False,
        invalid_reasons=invalid_reasons,
        sha256=sha256,
        file_size_bytes=file_size,
        file_modified_time=modified_time,
        payload={},
    )


def _canonical_sort_key(candidate: ArtifactCandidate) -> tuple[int, int, str, float, str]:
    valid_order = 0 if candidate.valid else 1
    created_missing_order = 0 if candidate.artifact_created_at else 1
    created_at_order = _invert_text(candidate.artifact_created_at or "")
    modified_order = -(candidate.file_modified_time or 0.0)
    return (
        valid_order,
        created_missing_order,
        created_at_order,
        modified_order,
        candidate.artifact_path,
    )


def _invert_text(value: str) -> str:
    return "".join(chr(0x10FFFF - ord(char)) for char in value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
