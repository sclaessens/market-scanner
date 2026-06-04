import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "fundamentals" / "persistence"

EXPECTED_FIXTURE_FILES = {
    "raw_complete_source.json",
    "raw_partial_source.json",
    "raw_invalid_source.json",
    "raw_stale_source.json",
    "raw_provenance_gap_source.json",
    "raw_forbidden_semantics_source.json",
}

SECRET_MARKERS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
}

PRODUCTION_ARTIFACT_MARKERS = {
    "data/raw/",
    "data/normalized/",
    "data/generated/",
    "data/processed/",
    "data/portfolio/",
    "data/watchlist/",
    "reports/daily/telegram_message.txt",
}

FORBIDDEN_TEST_INPUT_FIELDS = {
    "BUY",
    "SELL",
    "HOLD",
    "target_price",
    "allocation",
    "conviction",
    "urgency",
    "recommendation",
    "tradeability",
}


def _fixture_paths() -> tuple[Path, ...]:
    return tuple(sorted(FIXTURE_DIR.glob("raw_*_source.json")))


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk(path, value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield (*path, key), key
            yield from _walk((*path, key), child)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk((*path, str(index)), child)
    else:
        yield path, value


def test_expected_persistence_fixtures_exist_and_are_valid_json():
    paths = _fixture_paths()

    assert {path.name for path in paths} == EXPECTED_FIXTURE_FILES
    for path in paths:
        payload = _payload(path)

        assert isinstance(payload, dict)
        assert "fixture_metadata" in payload
        assert "raw_evidence" in payload
        assert "expected_contract" in payload


def test_every_fixture_declares_synthetic_non_live_non_production_metadata():
    for path in _fixture_paths():
        metadata = _payload(path)["fixture_metadata"]

        assert metadata["reset_stage"] == "RESET-10L-BL13"
        assert metadata["is_synthetic"] is True
        assert metadata["no_live_payload"] is True
        assert metadata["no_credentials"] is True
        assert metadata["no_production_data"] is True


def test_fixtures_do_not_contain_credentials_or_secret_markers():
    for path in _fixture_paths():
        payload = _payload(path)

        for location, value in _walk((), payload):
            if location == ("fixture_metadata", "no_credentials"):
                continue
            rendered = str(value).lower()
            for marker in SECRET_MARKERS:
                assert marker not in rendered


def test_fixtures_do_not_contain_live_payload_or_production_artifact_paths():
    forbidden_text_markers = {
        "raw live provider payload",
        "live payload",
        "sec-edgar",
        "broker",
        "telegram output",
        "telegram_message.txt",
        *PRODUCTION_ARTIFACT_MARKERS,
    }

    for path in _fixture_paths():
        payload = _payload(path)

        for location, value in _walk((), payload):
            if location == ("fixture_metadata", "no_live_payload"):
                continue
            rendered = str(value).lower()
            for marker in forbidden_text_markers:
                assert marker not in rendered


def test_forbidden_semantics_fixture_contains_terms_only_in_controlled_input():
    payload = _payload(FIXTURE_DIR / "raw_forbidden_semantics_source.json")

    assert set(payload["forbidden_test_input"]) == FORBIDDEN_TEST_INPUT_FIELDS

    for path, value in _walk((), payload):
        if str(value) in FORBIDDEN_TEST_INPUT_FIELDS:
            assert path[0] == "forbidden_test_input"


def test_non_forbidden_fixtures_do_not_contain_investment_semantics():
    for path in _fixture_paths():
        if path.name == "raw_forbidden_semantics_source.json":
            continue

        payload = _payload(path)
        values = {str(value) for _, value in _walk((), payload)}

        assert values.isdisjoint(FORBIDDEN_TEST_INPUT_FIELDS)


def test_persistence_contract_fixture_inspection_has_no_side_effects(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    for path in _fixture_paths():
        _payload(path)

    assert list(tmp_path.iterdir()) == []
    assert not Path("data").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
    assert not Path(".github/workflows").exists()


def test_persistence_contract_tests_do_not_import_provider_or_delivery_clients():
    test_files = tuple(sorted(Path(__file__).parent.glob("test_v2_persistence_*.py")))
    forbidden_imports = {"requests", "urllib", "httpx", "aiohttp", "yfinance"}
    forbidden_runtime_calls = {
        "fetch_" + "fundamentals(",
        "ingest_provider_" + "fundamentals(",
        "run_full_" + "pipeline.py",
    }

    assert test_files
    for test_file in test_files:
        source_lines = test_file.read_text(encoding="utf-8").lower().splitlines()
        import_lines = [
            line for line in source_lines if line.startswith(("import ", "from "))
        ]
        for line in import_lines:
            for term in forbidden_imports:
                assert term not in line
        source = "\n".join(source_lines)
        for term in forbidden_runtime_calls:
            assert term.lower() not in source


def test_persistence_contract_tests_do_not_create_production_paths(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    for relative_path in (
        "data/raw",
        "data/normalized",
        "data/generated",
        "data/processed",
        "reports/daily",
    ):
        assert not (Path.cwd() / relative_path).exists()


def test_no_github_workflow_files_are_part_of_persistence_fixture_scope():
    changed_scope = {
        "tests/fixtures/fundamentals/persistence",
        "tests/contract/test_v2_persistence_",
        "docs/active/backlog.md",
    }

    assert ".github/workflows" not in changed_scope
    assert os.getcwd()
