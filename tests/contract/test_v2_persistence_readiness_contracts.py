import json
from pathlib import Path


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "fundamentals" / "persistence"
)

READINESS_REQUIRED_FIELDS = {
    "readiness_record_id",
    "raw_evidence_id",
    "normalized_record_set_id",
    "ticker",
    "provider_name",
    "readiness_state",
    "source_data_status",
    "missing_fundamentals_count",
    "partial_data_count",
    "stale_data_count",
    "invalid_data_count",
    "provenance_status",
    "parseability_status",
    "consistency_status",
    "freshness_status",
    "readiness_warnings",
    "readiness_version",
}

FORBIDDEN_READINESS_FIELDS = {
    "BUY",
    "SELL",
    "HOLD",
    "allocation",
    "conviction",
    "investment_quality",
    "recommendation",
    "recommendation_strength",
    "target_price",
    "tradeability",
    "urgency",
    "valuation_attractiveness",
    "valuation_score",
}


def _payload(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _fixture_paths() -> tuple[Path, ...]:
    return tuple(sorted(FIXTURE_DIR.glob("raw_*_source.json")))


def _readiness(payload: dict) -> dict:
    return payload["expected_contract"]["readiness"]


def test_readiness_records_require_neutral_source_data_fields():
    for path in _fixture_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        readiness = _readiness(payload)

        assert set(readiness) == READINESS_REQUIRED_FIELDS
        assert readiness["readiness_record_id"]
        assert readiness["raw_evidence_id"] == payload["raw_evidence"]["raw_evidence_id"]
        assert (
            readiness["normalized_record_set_id"]
            == payload["expected_contract"]["normalized_fundamentals"][
                "normalized_record_set_id"
            ]
        )
        assert readiness["ticker"] == payload["raw_evidence"]["ticker"]
        assert readiness["provider_name"] == payload["raw_evidence"]["provider_name"]
        assert isinstance(readiness["missing_fundamentals_count"], int)
        assert isinstance(readiness["partial_data_count"], int)
        assert isinstance(readiness["stale_data_count"], int)
        assert isinstance(readiness["invalid_data_count"], int)
        assert isinstance(readiness["readiness_warnings"], list)
        assert readiness["readiness_version"] == "v2-persistence-readiness-contract-v1"


def test_complete_fixture_produces_complete_neutral_readiness():
    readiness = _readiness(_payload("raw_complete_source.json"))

    assert readiness["readiness_state"] == "ready"
    assert readiness["source_data_status"] == "complete"
    assert readiness["missing_fundamentals_count"] == 0
    assert readiness["partial_data_count"] == 0
    assert readiness["stale_data_count"] == 0
    assert readiness["invalid_data_count"] == 0
    assert readiness["provenance_status"] == "complete"
    assert readiness["freshness_status"] == "fresh"
    assert readiness["readiness_warnings"] == []


def test_partial_fixture_produces_partial_insufficient_neutral_readiness():
    readiness = _readiness(_payload("raw_partial_source.json"))

    assert readiness["readiness_state"] == "partial"
    assert readiness["source_data_status"] == "partial"
    assert readiness["missing_fundamentals_count"] == 2
    assert readiness["partial_data_count"] == 1
    assert readiness["readiness_warnings"] == ["missing_source_fields"]


def test_invalid_fixture_produces_invalid_fail_closed_readiness():
    readiness = _readiness(_payload("raw_invalid_source.json"))

    assert readiness["readiness_state"] == "fail_closed"
    assert readiness["source_data_status"] == "invalid"
    assert readiness["invalid_data_count"] == 1
    assert readiness["parseability_status"] == "unparseable_values_present"
    assert readiness["readiness_warnings"] == ["invalid_source_value"]


def test_stale_fixture_produces_freshness_warning_readiness():
    readiness = _readiness(_payload("raw_stale_source.json"))

    assert readiness["readiness_state"] == "stale"
    assert readiness["source_data_status"] == "stale"
    assert readiness["stale_data_count"] == 1
    assert readiness["freshness_status"] == "stale"
    assert readiness["readiness_warnings"] == [
        "source_timestamp_outside_freshness_window"
    ]


def test_provenance_gap_fixture_produces_fail_closed_readiness():
    readiness = _readiness(_payload("raw_provenance_gap_source.json"))

    assert readiness["readiness_state"] == "fail_closed"
    assert readiness["source_data_status"] == "provenance_gap"
    assert readiness["provenance_status"] == "incomplete"
    assert readiness["freshness_status"] == "unknown"
    assert readiness["readiness_warnings"] == ["provenance_gap"]


def test_forbidden_semantics_fixture_stays_neutral_review_required():
    readiness = _readiness(_payload("raw_forbidden_semantics_source.json"))

    assert readiness["readiness_state"] == "neutral_review_required"
    assert readiness["source_data_status"] == "controlled_forbidden_input_present"
    assert readiness["readiness_warnings"] == ["controlled_forbidden_input_present"]


def test_readiness_never_contains_investment_quality_or_authority_fields():
    for path in _fixture_paths():
        readiness = _readiness(json.loads(path.read_text(encoding="utf-8")))

        assert set(readiness).isdisjoint(FORBIDDEN_READINESS_FIELDS)
        rendered_values = {str(value) for value in readiness.values()}
        assert rendered_values.isdisjoint(FORBIDDEN_READINESS_FIELDS)
