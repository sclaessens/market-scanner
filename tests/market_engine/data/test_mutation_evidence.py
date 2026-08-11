from __future__ import annotations

from pathlib import Path

import pytest

from market_engine.data.mutation_evidence import (
    MutationEvidenceError,
    derive_canonical_mutations,
    derive_session_resolution,
    mutation_evidence_diagnostics,
    mutation_root,
    reconcile_mutation_evidence,
)


HEADER = "Date,Open,High,Low,Close,Adj Close,Volume\n"


def _write(path: Path, rows: list[tuple[str, str, str, str, str, str, str]]) -> None:
    path.write_text(
        HEADER + "".join(",".join(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mutations(
    tmp_path: Path,
    baseline: list[tuple[str, str, str, str, str, str, str]],
    staged: list[tuple[str, str, str, str, str, str, str]],
    *,
    ticker: str = "RND",
    exchange: str = "NYSE",
):
    baseline_path = tmp_path / "baseline.csv"
    staged_path = tmp_path / "staged.csv"
    _write(baseline_path, baseline)
    _write(staged_path, staged)
    return derive_canonical_mutations(
        baseline_path=baseline_path,
        staged_path=staged_path,
        instrument_id=f"equity:{ticker.lower()}",
        ticker=ticker,
        exchange=exchange,
        currency="USD",
    )


def _receipt(mutation, *, route: str = "primary"):
    return {
        "instrument_id": mutation["instrument_id"],
        "ticker": mutation["ticker"],
        "exchange": mutation["exchange"],
        "session_date": mutation["session_date"],
        "acquisition_route": route,
        "canonical_row_sha256": mutation["new_canonical_row_sha256"],
        "receipt_sha256": "a" * 64,
    }


@pytest.mark.parametrize(
    ("ticker", "exchange"),
    [
        ("A", "NYSE"),
        ("BRK-B", "NYSE"),
        ("AB.C", "NASDAQ"),
        ("RANDOM12", "XAMS"),
    ],
)
@pytest.mark.parametrize("route", ["primary", "primary_replay", "fallback"])
def test_every_added_row_requires_evidence_independent_of_label(
    tmp_path: Path,
    ticker: str,
    exchange: str,
    route: str,
) -> None:
    mutations = _mutations(
        tmp_path,
        [("2026-07-23", "10", "11", "9", "10", "10", "100")],
        [
            ("2026-07-23", "10", "11", "9", "10", "10", "100"),
            ("2026-07-24", "11", "12", "10", "11", "11", "200"),
        ],
        ticker=ticker,
        exchange=exchange,
    )
    with pytest.raises(MutationEvidenceError, match="do not equal"):
        reconcile_mutation_evidence(mutations, [])

    summary = reconcile_mutation_evidence(
        mutations, [_receipt(mutations[0], route=route)]
    )

    assert summary["added_count"] == 1
    assert summary["evidence_required_mutation_count"] == 1


def test_modified_and_deleted_rows_block_while_unchanged_requires_no_receipt(
    tmp_path: Path,
) -> None:
    row = ("2026-07-23", "10", "11", "9", "10", "10", "100")
    assert reconcile_mutation_evidence(_mutations(tmp_path, [row], [row]), [])[
        "evidence_required_mutation_count"
    ] == 0
    modified = _mutations(
        tmp_path,
        [row],
        [("2026-07-23", "10", "11", "9", "10.5", "10", "100")],
    )
    assert modified[0]["field_diff"] == {
        "Close": {"previous": "10", "current": "10.5"}
    }
    with pytest.raises(MutationEvidenceError, match="correction contract"):
        reconcile_mutation_evidence(modified, [_receipt(modified[0])])
    deleted = _mutations(tmp_path, [row], [])
    with pytest.raises(MutationEvidenceError, match="deletion"):
        reconcile_mutation_evidence(deleted, [])


def test_failed_reconciliation_retains_exact_mutation_diagnostics(tmp_path: Path) -> None:
    baseline = [
        ("2026-07-22", "10", "11", "9", "10", "10", "100"),
        ("2026-07-23", "20", "21", "19", "20", "20", "200"),
    ]
    staged = [
        ("2026-07-22", "10", "11", "9", "10.5", "10", "101"),
        ("2026-07-23", "20", "21", "19", "20.5", "20", "200"),
        ("2026-07-24", "30", "31", "29", "30", "30", "300"),
    ]
    mutations = _mutations(tmp_path, baseline, staged)
    diagnostics = mutation_evidence_diagnostics(
        mutations,
        [],
        artifact_replay_failures=["ObservationReceiptError: artifact is missing"],
    )

    assert diagnostics["status"] == "invalid"
    assert diagnostics["modified_instrument_count"] == 1
    assert diagnostics["modified_row_count"] == 2
    assert diagnostics["added_row_count"] == 1
    assert diagnostics["mutations_without_receipt_count"] == 3
    assert diagnostics["artifact_replay_failure_count"] == 1
    assert [row["session_date"] for row in diagnostics["diagnostic_rows"]] == [
        "2026-07-22", "2026-07-23", "2026-07-24"
    ]
    assert diagnostics["diagnostic_rows"][0]["field_diff"] == {
        "Close": {"previous": "10", "current": "10.5"},
        "Volume": {"previous": "100", "current": "101"},
    }
    assert diagnostics["diagnostic_rows"][0]["previous_values"]["Close"] == "10"
    assert diagnostics["diagnostic_rows"][0]["new_values"]["Close"] == "10.5"
    assert diagnostics["diagnostic_rows"][0]["previous_canonical_row_sha256"]
    assert diagnostics["diagnostic_rows"][0]["new_canonical_row_sha256"]


def test_mutation_diagnostics_are_input_order_independent(tmp_path: Path) -> None:
    mutations = _mutations(
        tmp_path,
        [],
        [
            ("2026-07-24", "11", "12", "10", "11", "11", "200"),
            ("2026-07-25", "12", "13", "11", "12", "12", "300"),
        ],
    )
    receipts = [_receipt(row) for row in mutations]
    assert mutation_evidence_diagnostics(mutations, receipts) == mutation_evidence_diagnostics(
        list(reversed(mutations)), list(reversed(receipts))
    )


def test_row_order_does_not_change_mutations_or_root(tmp_path: Path) -> None:
    baseline = [("2026-07-23", "10", "11", "9", "10", "10", "100")]
    additions = [
        ("2026-07-24", "11", "12", "10", "11", "11", "200"),
        ("2026-07-27", "12", "13", "11", "12", "12", "300"),
    ]
    first = _mutations(tmp_path, baseline, baseline + additions)
    second = _mutations(tmp_path, baseline, list(reversed(baseline + additions)))
    first_receipts = [
        {**_receipt(row), "receipt_sha256": str(index + 1) * 64}
        for index, row in enumerate(first)
    ]
    second_receipts = list(reversed(first_receipts))
    assert first == second
    assert mutation_root(first, first_receipts) == mutation_root(
        second, second_receipts
    )


def test_duplicate_canonical_session_blocks_before_evidence(tmp_path: Path) -> None:
    row = ("2026-07-23", "10", "11", "9", "10", "10", "100")
    with pytest.raises(MutationEvidenceError, match="duplicate"):
        _mutations(tmp_path, [], [row, row])


@pytest.mark.parametrize(
    ("receipt_sessions", "absence_sessions", "expected_unresolved"),
    [
        (["2026-07-23", "2026-07-24"], [], []),
        (["2026-07-23"], [], ["2026-07-24"]),
        (["2026-07-23"], ["2026-07-24"], []),
        (["2026-07-22", "2026-07-23"], ["2026-07-24"], []),
    ],
)
def test_session_partition_is_rederived_from_final_evidence(
    receipt_sessions: list[str],
    absence_sessions: list[str],
    expected_unresolved: list[str],
) -> None:
    expected = ["2026-07-22", "2026-07-23", "2026-07-24"]
    expected = expected[-2:] if receipt_sessions[0] == "2026-07-23" else expected
    receipts = [
        {"session_date": session, "acquisition_route": "primary_replay"}
        for session in receipt_sessions
    ]
    attestations = [
        {"session_date": session, "instrument_id": "equity:aaa"}
        for session in absence_sessions
    ]

    ledger = derive_session_resolution(
        expected_sessions=list(reversed(expected)),
        receipts=list(reversed(receipts)),
        absence_attestations=attestations,
        canonical_mutation_sessions=receipt_sessions,
        consumer_instrument_id="equity:aaa",
    )

    assert ledger["unresolved_sessions"] == expected_unresolved
    assert ledger["fallback_candidates"] == expected_unresolved
    assert [row["session_date"] for row in ledger["partition"]] == sorted(expected)


def test_mixed_gap_and_terminal_absence_reconciles_disjointly() -> None:
    ledger = derive_session_resolution(
        expected_sessions=["2026-07-22", "2026-07-23", "2026-07-24"],
        receipts=[
            {"session_date": "2026-07-22", "acquisition_route": "fallback"},
            {"session_date": "2026-07-23", "acquisition_route": "primary_replay"},
        ],
        absence_attestations=[{"session_date": "2026-07-24", "instrument_id": "equity:aaa"}],
        canonical_mutation_sessions=["2026-07-23", "2026-07-22"],
        consumer_instrument_id="equity:aaa",
    )
    assert ledger["partition"] == [
        {"session_date": "2026-07-22", "state": "observed_fallback"},
        {"session_date": "2026-07-23", "state": "observed_primary"},
        {"session_date": "2026-07-24", "state": "explained_absent"},
    ]
    assert ledger["unresolved_sessions"] == []


def test_not_expected_sessions_are_explicit_and_disjoint() -> None:
    ledger = derive_session_resolution(
        expected_sessions=["2026-07-24"],
        not_expected_sessions=["2026-07-25", "2026-07-26"],
        receipts=[
            {"session_date": "2026-07-24", "acquisition_route": "primary"}
        ],
        absence_attestations=[],
        canonical_mutation_sessions=["2026-07-24"],
    )
    assert ledger["partition"][-2:] == [
        {"session_date": "2026-07-25", "state": "not_expected"},
        {"session_date": "2026-07-26", "state": "not_expected"},
    ]
    assert ledger["not_expected_sessions"] == ["2026-07-25", "2026-07-26"]

    with pytest.raises(MutationEvidenceError, match="expected and not expected"):
        derive_session_resolution(
            expected_sessions=["2026-07-24"],
            not_expected_sessions=["2026-07-24"],
            receipts=[],
            absence_attestations=[],
            canonical_mutation_sessions=[],
        )


@pytest.mark.parametrize(
    "mutation",
    ["overlap", "internal_absence", "not_expected_row", "duplicate_expected"],
)
def test_invalid_session_partitions_fail_closed(mutation: str) -> None:
    expected = ["2026-07-23", "2026-07-24"]
    receipts = [{"session_date": "2026-07-23", "acquisition_route": "primary"}]
    absences = [{"session_date": "2026-07-24"}]
    mutations = ["2026-07-23"]
    if mutation == "overlap":
        absences = [{"session_date": "2026-07-23"}]
    elif mutation == "internal_absence":
        absences = [{"session_date": "2026-07-22"}]
    elif mutation == "not_expected_row":
        mutations = ["2026-07-23", "2026-07-25"]
    else:
        expected.append("2026-07-24")
    with pytest.raises(MutationEvidenceError):
        derive_session_resolution(
            expected_sessions=expected,
            receipts=receipts,
            absence_attestations=absences,
            canonical_mutation_sessions=mutations,
        )
