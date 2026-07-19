from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("artifacts/market_engine/run_evidence/me-data09-aapl-bounded-operator-pilot-20260719T155116Z")


def _read(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_compact_manifest_records_run_identity_and_publication_boundary() -> None:
    manifest = _read("manifest.json")

    assert manifest["full_run_committed"] is False
    assert manifest["compact_evidence_committed"] is True
    assert manifest["approved_tickers"] == ["AAPL"]
    assert manifest["run_ids"] == {
        "data06": "me-data06-after-me-data09-aapl-20260719T155116Z",
        "data07": "me-data09-aapl-bounded-operator-pilot-20260719T155116Z",
        "run31": "me-run31-after-me-data09-aapl-20260719T155116Z",
    }


def test_compact_current_sprint_delta_is_zero_and_ticker_delta_is_traceable() -> None:
    coverage = _read("coverage_delta.json")
    current = coverage["current_sprint_comparison"]

    assert current["before"] == current["after"]
    assert set(current["absolute_delta"].values()) == {0}
    assert coverage["historical_origin_comparison"]["attributable_to_current_sprint"] is False
    assert "improvement_tickers" not in json.dumps(coverage)
    assert coverage["ticker_delta"] == {
        "ticker": "AAPL",
        "before_status": "partial",
        "after_status": "partial",
        "new_metrics": ["eps_growth_yoy", "revenue_growth_yoy"],
        "remaining_missing_metrics": ["debt_to_equity", "gross_margin", "operating_margin"],
        "advice_input_ready_before": False,
        "advice_input_ready_after": False,
    }
    report = (ROOT / "report.md").read_text(encoding="utf-8")
    assert json.dumps(current["absolute_delta"], sort_keys=True) in report
    assert "Historical DATA06 transitions are explicitly not attributable" in report


def test_compact_package_has_no_full_universe_or_absolute_source_root() -> None:
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in sorted(ROOT.iterdir()) if path.is_file())

    assert '"ticker_count": 952' not in serialized
    assert "/tmp/" not in serialized
    assert "/Users/" not in serialized
    assert "<html" not in serialized.lower()


def test_compact_checksums_match_committed_files_and_local_full_runs() -> None:
    top = _read("top_level_checksums.json")["checksums"]
    for name, expected in top.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected

    downstream = _read("downstream_run_index.json")
    for section, path_key, checksum_key in (
        ("data06", "manifest_path", "manifest_sha256"),
        ("run31", "compact_index_path", "compact_index_sha256"),
    ):
        path = Path(downstream[section][path_key])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == downstream[section][checksum_key]


def test_compact_counts_reconcile_and_source_documents_are_checksum_only() -> None:
    pilot = _read("data07_pilot_summary.json")
    counts = pilot["counts"]
    assert counts == {
        "selected_count": 12,
        "imported_count": 1,
        "normalized_count": 1,
        "success_count": 1,
        "blocked_count": 11,
        "failed_count": 0,
        "pending_count": 0,
        "not_selected_count": 940,
    }
    assert pilot["reconciliation"]["reconciled"] is True
    assert pilot["provider_calls_performed"] == 0
    documents = _read("source_document_checksums.json")["documents"]
    assert all("relative_path" in document and "local_path" not in document for document in documents)
