import importlib


V2_SUBPACKAGES = [
    "context",
    "decisions",
    "discovery",
    "fundamentals",
    "orchestration",
    "portfolio",
    "reporting",
    "shared",
    "timing",
    "validation",
]


def test_market_scanner_package_imports():
    package = importlib.import_module("market_scanner")

    assert package.__all__ == V2_SUBPACKAGES


def test_v2_subpackages_import():
    for subpackage in V2_SUBPACKAGES:
        module = importlib.import_module(f"market_scanner.{subpackage}")

        assert module.__name__ == f"market_scanner.{subpackage}"


def test_v2_imports_have_no_filesystem_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner")
    for subpackage in V2_SUBPACKAGES:
        importlib.import_module(f"market_scanner.{subpackage}")

    assert list(tmp_path.iterdir()) == []
