from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path
from urllib.request import Request

import pytest

from scripts.fundamentals import sec_companyfacts_bulk_intake as intake


def _write_fixture_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cik": 320193,
        "entityName": "SYNTHETIC SAMPLE",
        "facts": {},
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("CIK0000320193.json", json.dumps(payload))


def test_import_does_not_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("import must not open network connections")

    monkeypatch.setattr(intake, "urlopen", fail_urlopen)

    importlib.reload(intake)


def test_official_companyfacts_url_is_accepted() -> None:
    parsed = intake.validate_companyfacts_source_url(intake.SEC_COMPANYFACTS_BULK_URL)

    assert parsed.scheme == "https"
    assert parsed.hostname == "www.sec.gov"
    assert parsed.path == "/Archives/edgar/daily-index/xbrl/companyfacts.zip"


@pytest.mark.parametrize(
    "source_url",
    [
        "http://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
        "https://example.com/Archives/edgar/daily-index/xbrl/companyfacts.zip",
        "https://www.sec.gov/Archives/edgar/daily-index/xbrl/not-companyfacts.zip",
    ],
)
def test_unsupported_source_urls_are_rejected(source_url: str) -> None:
    with pytest.raises(ValueError):
        intake.validate_companyfacts_source_url(source_url)


def test_missing_user_agent_is_rejected_when_download_is_requested(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="User-Agent"):
        intake.download_companyfacts_bulk_zip(cache_dir=tmp_path, user_agent="")


def test_cache_directory_creation_uses_temp_path(tmp_path: Path) -> None:
    cache_dir = tmp_path / "sec-cache" / "companyfacts"

    result = intake.ensure_cache_dir(cache_dir)

    assert result == cache_dir
    assert cache_dir.is_dir()


def test_file_cache_path_is_rejected(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache-file"
    cache_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        intake.ensure_cache_dir(cache_file)


def test_manifest_generation_from_local_fixture_zip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    zip_path = tmp_path / "fixture_companyfacts.zip"
    _write_fixture_zip(zip_path)

    manifest = intake.inspect_local_companyfacts_zip(
        zip_path,
        cache_dir=cache_dir,
        write_manifest_file=True,
    )

    manifest_path = cache_dir / intake.MANIFEST_NAME
    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "VALIDATED"
    assert manifest["source_url"] == intake.SEC_COMPANYFACTS_BULK_URL
    assert manifest["file_size_bytes"] == zip_path.stat().st_size
    assert manifest["sha256"] == intake.sha256_file(zip_path)
    assert manifest["local_zip_path"] == str(zip_path)
    assert manifest["json_file_count"] == 1
    assert written_manifest == manifest


def test_invalid_zip_fails_clearly(tmp_path: Path) -> None:
    invalid_zip = tmp_path / "companyfacts.zip"
    invalid_zip.write_text("not a zip", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        intake.validate_local_companyfacts_zip(invalid_zip)


def test_local_fixture_zip_validation_succeeds(tmp_path: Path) -> None:
    zip_path = tmp_path / "companyfacts.zip"
    _write_fixture_zip(zip_path)

    validation = intake.validate_local_companyfacts_zip(zip_path)

    assert validation["status"] == "VALID"
    assert validation["file_count"] == 1
    assert validation["json_file_count"] == 1
    assert validation["sha256"] == intake.sha256_file(zip_path)


def test_download_writes_only_under_provided_cache_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_zip = tmp_path / "fixture" / "companyfacts.zip"
    _write_fixture_zip(fixture_zip)
    cache_dir = tmp_path / "cache"
    requested_headers: dict[str, str] = {}

    class FixtureResponse:
        status = 200

        def __init__(self, payload: bytes):
            self._payload = payload
            self._offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size: int = -1) -> bytes:
            if self._offset >= len(self._payload):
                return b""
            if size < 0:
                size = len(self._payload) - self._offset
            chunk = self._payload[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

    def fake_urlopen(request: Request, timeout: int):
        assert timeout == 60
        requested_headers.update(dict(request.header_items()))
        return FixtureResponse(fixture_zip.read_bytes())

    monkeypatch.setattr(intake, "urlopen", fake_urlopen)

    manifest = intake.download_companyfacts_bulk_zip(
        cache_dir=cache_dir,
        user_agent="market-scanner-test contact@example.com",
    )

    cached_zip = cache_dir / intake.COMPANYFACTS_ZIP_NAME
    manifest_path = cache_dir / intake.MANIFEST_NAME
    assert cached_zip.exists()
    assert manifest_path.exists()
    assert manifest["status"] == "DOWNLOADED"
    assert manifest["local_zip_path"] == str(cached_zip)
    assert requested_headers["User-agent"] == "market-scanner-test contact@example.com"
    assert cached_zip.parent == cache_dir
    assert manifest_path.parent == cache_dir


def test_module_has_no_pipeline_integration() -> None:
    assert not hasattr(intake, "build_fundamental_metrics")
    assert not hasattr(intake, "build_fundamental_layer")
    assert not hasattr(intake, "build_fundamental_analysis")
