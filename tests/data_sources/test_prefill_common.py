from __future__ import annotations

from pathlib import Path


SCRIPT_PACKAGE_IMPORT_PATTERNS = tuple(
    " ".join(parts)
    for parts in (
        ("from", ".".join(("scripts", "data_sources"))),
        ("import", ".".join(("scripts", "data_sources"))),
    )
)

SCRIPT_ERA_DATA_SOURCE_PATHS = tuple(
    "/".join(parts)
    for parts in (
        ("scripts", "data_sources", "common.py"),
        ("scripts", "data_sources", "prefill_fundamentals.py"),
        ("scripts", "data_sources", "prefill_portfolio_metadata.py"),
    )
)

ACTIVE_SEARCH_ROOTS = (
    Path("src"),
    Path("tests"),
    Path(".github"),
)


def _python_and_workflow_files(root: Path) -> list[Path]:
    if not root.exists():
        return []

    allowed_suffixes = {".py", ".yml", ".yaml"}
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in allowed_suffixes
        and "__pycache__" not in path.parts
    ]


def test_active_code_no_longer_imports_script_era_data_source_modules():
    offenders: list[str] = []

    for root in ACTIVE_SEARCH_ROOTS:
        for path in _python_and_workflow_files(root):
            source = path.read_text(encoding="utf-8")
            if any(pattern in source for pattern in SCRIPT_PACKAGE_IMPORT_PATTERNS):
                offenders.append(str(path))

    assert offenders == []


def test_canonical_runtime_metadata_does_not_claim_script_era_data_source_authority():
    offenders: list[str] = []

    for root in (Path("src"), Path(".github")):
        for path in _python_and_workflow_files(root):
            source = path.read_text(encoding="utf-8")
            if any(script_path in source for script_path in SCRIPT_ERA_DATA_SOURCE_PATHS):
                offenders.append(str(path))

    assert offenders == []


def test_data_source_prefill_policy_remains_non_recommendation_authority():
    forbidden_terms = {
        "allocation",
        "ranking",
        "score",
        "tradeable",
        "urgency",
        "conviction",
        "final_action",
        "buy",
        "sell",
        "hold",
    }

    policy_terms = {
        "provider export",
        "source artifact",
        "dry run",
        "allow overwrite",
        "validation status",
        "missing required columns",
        "duplicate row identity",
        "credential safe audit",
    }

    policy_text = " ".join(sorted(policy_terms)).lower()

    for term in forbidden_terms:
        assert term not in policy_text
