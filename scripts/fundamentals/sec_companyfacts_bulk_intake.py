from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import ParseResult, urlparse
from urllib.request import Request, urlopen

SEC_COMPANYFACTS_BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
SEC_COMPANYFACTS_HOST = "www.sec.gov"
SEC_COMPANYFACTS_PATH = "/Archives/edgar/daily-index/xbrl/companyfacts.zip"
DEFAULT_CACHE_DIR = Path("data/local/sec_edgar/companyfacts")
COMPANYFACTS_ZIP_NAME = "companyfacts.zip"
MANIFEST_NAME = "companyfacts_manifest.json"


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_companyfacts_source_url(source_url: str = SEC_COMPANYFACTS_BULK_URL) -> ParseResult:
    parsed = urlparse(source_url)
    if parsed.scheme != "https":
        raise ValueError(f"SEC Company Facts bulk URL must use HTTPS: {source_url}")
    if parsed.hostname != SEC_COMPANYFACTS_HOST:
        raise ValueError(f"SEC Company Facts bulk URL host is not approved: {parsed.hostname or ''}")
    if parsed.path != SEC_COMPANYFACTS_PATH:
        raise ValueError(f"SEC Company Facts bulk URL path is not approved: {parsed.path}")
    return parsed


def require_user_agent(user_agent: str | None) -> str:
    value = (user_agent or "").strip()
    if not value:
        raise ValueError("SEC requests require an explicit descriptive User-Agent.")
    return value


def ensure_cache_dir(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    path = Path(cache_dir)
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError(f"cache directory is unsafe: {path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"cache path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_local_companyfacts_zip(zip_path: str | Path) -> dict[str, Any]:
    path = Path(zip_path)
    if not path.exists():
        raise FileNotFoundError(f"SEC Company Facts ZIP not found: {path}")
    if not path.is_file():
        raise ValueError(f"SEC Company Facts ZIP path is not a file: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"SEC Company Facts ZIP is unreadable or invalid: {path}")

    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"SEC Company Facts ZIP contains unreadable member: {bad_member}")
            members = [member for member in archive.namelist() if not member.endswith("/")]
    except zipfile.BadZipFile as exc:
        raise ValueError(f"SEC Company Facts ZIP is unreadable or invalid: {path}") from exc

    if not members:
        raise ValueError(f"SEC Company Facts ZIP contains no files: {path}")
    json_members = [member for member in members if member.lower().endswith(".json")]
    if not json_members:
        raise ValueError(f"SEC Company Facts ZIP contains no JSON company facts files: {path}")

    return {
        "status": "VALID",
        "local_zip_path": str(path),
        "file_count": len(members),
        "json_file_count": len(json_members),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_companyfacts_manifest(
    zip_path: str | Path,
    *,
    source_url: str = SEC_COMPANYFACTS_BULK_URL,
    downloaded_at: str | None = None,
    status: str = "VALIDATED",
) -> dict[str, Any]:
    validate_companyfacts_source_url(source_url)
    validation = validate_local_companyfacts_zip(zip_path)
    timestamp = downloaded_at or utc_timestamp()
    return {
        "status": status,
        "source_url": source_url,
        "downloaded_at": timestamp,
        "extraction_date": timestamp[:10],
        "source_freshness_date": timestamp[:10],
        "file_size_bytes": validation["file_size_bytes"],
        "sha256": validation["sha256"],
        "local_zip_path": validation["local_zip_path"],
        "file_count": validation["file_count"],
        "json_file_count": validation["json_file_count"],
    }


def write_manifest(manifest: dict[str, Any], cache_dir: str | Path) -> Path:
    directory = ensure_cache_dir(cache_dir)
    manifest_path = directory / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def inspect_local_companyfacts_zip(
    zip_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    source_url: str = SEC_COMPANYFACTS_BULK_URL,
    write_manifest_file: bool = False,
) -> dict[str, Any]:
    manifest = build_companyfacts_manifest(zip_path, source_url=source_url)
    if write_manifest_file:
        if cache_dir is None:
            raise ValueError("cache_dir is required when write_manifest_file is true.")
        write_manifest(manifest, cache_dir)
    return manifest


def _atomic_download_to_path(request: Request, target_path: Path, timeout_seconds: int) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target_path.name}.", suffix=".tmp", dir=str(target_path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as output:
            with urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise ValueError(f"SEC Company Facts download failed with HTTP status: {status}")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def download_companyfacts_bulk_zip(
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    user_agent: str | None,
    source_url: str = SEC_COMPANYFACTS_BULK_URL,
    timeout_seconds: int = 60,
    write_manifest_file: bool = True,
) -> dict[str, Any]:
    require_user_agent(user_agent)
    validate_companyfacts_source_url(source_url)
    directory = ensure_cache_dir(cache_dir)
    target_path = directory / COMPANYFACTS_ZIP_NAME
    request = Request(source_url, headers={"User-Agent": require_user_agent(user_agent)})

    try:
        _atomic_download_to_path(request, target_path, timeout_seconds)
    except URLError as exc:
        raise RuntimeError(f"SEC Company Facts download failed: {exc}") from exc

    manifest = build_companyfacts_manifest(target_path, source_url=source_url, status="DOWNLOADED")
    if write_manifest_file:
        write_manifest(manifest, directory)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled local SEC Company Facts bulk intake/cache utility.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Local ignored SEC cache directory.")
    parser.add_argument("--source-url", default=SEC_COMPANYFACTS_BULK_URL, help="Official SEC Company Facts ZIP URL.")
    parser.add_argument("--user-agent", help="Required descriptive User-Agent when --download is used.")
    parser.add_argument("--download", action="store_true", help="Download the SEC Company Facts ZIP into the cache.")
    parser.add_argument("--validate-local", type=Path, help="Validate an already-local SEC Company Facts ZIP.")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write a local manifest in the cache directory for --validate-local. Downloads write one by default.",
    )
    args = parser.parse_args()

    if args.download and args.validate_local:
        raise SystemExit("--download and --validate-local cannot be used together.")
    if not args.download and not args.validate_local:
        raise SystemExit("Choose --download or --validate-local.")

    if args.download:
        manifest = download_companyfacts_bulk_zip(
            cache_dir=args.cache_dir,
            user_agent=args.user_agent,
            source_url=args.source_url,
        )
    else:
        manifest = inspect_local_companyfacts_zip(
            args.validate_local,
            cache_dir=args.cache_dir,
            source_url=args.source_url,
            write_manifest_file=args.write_manifest,
        )

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
