import json
from pathlib import Path


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "fundamentals" / "persistence"
)

RAW_REQUIRED_FIELDS = {
    "raw_evidence_id",
    "provider_name",
    "provider_category",
    "provider_record_id",
    "original_source_reference",
    "ticker",
    "symbol",
    "entity_identifier",
    "source_timestamp",
    "retrieval_timestamp",
    "reported_period",
    "fiscal_year",
    "fiscal_quarter",
    "currency",
    "unit",
    "raw_fields",
    "missing_field_evidence",
    "provenance_metadata",
    "raw_payload_hash",
    "capture_version",
    "validation_warnings",
}

FORBIDDEN_INVESTMENT_FIELDS = {
    "BUY",
    "SELL",
    "HOLD",
    "allocation",
    "conviction",
    "recommendation",
    "target_price",
    "tradeability",
    "urgency",
}

ZERO_LIKE_MISSING_VALUES = (0, 0.0, "0", False, "")


def _fixture_paths() -> tuple[Path, ...]:
    return tuple(sorted(FIXTURE_DIR.glob("raw_*_source.json")))


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_values(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)
    else:
        yield value


def test_raw_evidence_fixtures_include_required_provenance_fields():
    for path in _fixture_paths():
        raw = _payload(path)["raw_evidence"]

        assert set(raw) == RAW_REQUIRED_FIELDS
        assert raw["raw_evidence_id"]
        assert raw["provider_category"] == "regulatory_filing"
        assert raw["provider_record_id"]
        assert raw["ticker"] == "ASML"
        assert raw["symbol"] == "ASML"
        assert raw["entity_identifier"] == "ASML-HOLDING-NV-SYNTHETIC"
        assert raw["retrieval_timestamp"] == "2026-06-03T00:00:00Z"
        assert raw["reported_period"] == "FY"
        assert raw["fiscal_year"]
        assert raw["currency"] == "EUR"
        assert raw["unit"] == "EUR"
        assert isinstance(raw["raw_fields"], dict)
        assert isinstance(raw["missing_field_evidence"], list)
        assert isinstance(raw["provenance_metadata"], dict)
        assert raw["raw_payload_hash"].startswith("sha256:")
        assert raw["capture_version"] == "v2-persistence-contract-fixture-v1"


def test_raw_source_reference_and_timestamps_are_preserved():
    for path in _fixture_paths():
        raw = _payload(path)["raw_evidence"]
        readiness = _payload(path)["expected_contract"]["readiness"]

        assert readiness["raw_evidence_id"] == raw["raw_evidence_id"]
        assert raw["retrieval_timestamp"]
        if path.name != "raw_provenance_gap_source.json":
            assert raw["original_source_reference"]
            assert raw["source_timestamp"]
            assert raw["provider_name"]
        else:
            assert raw["original_source_reference"] == ""
            assert raw["source_timestamp"] == ""
            assert raw["provider_name"] == ""
            assert readiness["provenance_status"] == "incomplete"


def test_raw_field_names_remain_present_in_raw_layer():
    expected_raw_fields = {
        "raw_complete_source.json": {"Revenues", "GrossProfit", "NetIncomeLoss"},
        "raw_partial_source.json": {"Revenues", "GrossProfit", "FreeCashFlow"},
        "raw_invalid_source.json": {"Revenues", "NetIncomeLoss"},
        "raw_stale_source.json": {"Revenues", "NetIncomeLoss"},
        "raw_provenance_gap_source.json": {"Revenues", "NetIncomeLoss"},
        "raw_forbidden_semantics_source.json": {"Revenues", "NetIncomeLoss"},
    }

    for path in _fixture_paths():
        raw_fields = _payload(path)["raw_evidence"]["raw_fields"]

        assert set(raw_fields).issuperset(expected_raw_fields[path.name])


def test_partial_fixture_preserves_explicit_missing_field_evidence():
    payload = _payload(FIXTURE_DIR / "raw_partial_source.json")
    raw = payload["raw_evidence"]
    raw_contract = payload["expected_contract"]["raw_evidence"]

    assert raw["raw_fields"]["GrossProfit"] is None
    assert raw["raw_fields"]["FreeCashFlow"] is None
    assert len(raw["missing_field_evidence"]) == 2
    assert raw_contract["explicit_missing_state"] is True
    assert raw_contract["missing_count_expectation"] == 2


def test_raw_missing_values_are_never_converted_to_zero_like_values():
    for path in _fixture_paths():
        raw = _payload(path)["raw_evidence"]

        for field_name, field_value in raw["raw_fields"].items():
            if field_value is None:
                assert field_value not in ZERO_LIKE_MISSING_VALUES, field_name

        assert (
            _payload(path)["expected_contract"]["raw_evidence"][
                "missing_values_converted_to_zero"
            ]
            is False
        )


def test_raw_evidence_has_no_investment_conclusions_outside_controlled_fixture():
    for path in _fixture_paths():
        payload = _payload(path)
        raw = payload["raw_evidence"]

        assert set(raw).isdisjoint(FORBIDDEN_INVESTMENT_FIELDS)
        assert set(raw["raw_fields"]).isdisjoint(FORBIDDEN_INVESTMENT_FIELDS)
        assert (
            payload["expected_contract"]["raw_evidence"]["forbidden_semantics_present"]
            is False
        )


def test_forbidden_semantics_are_isolated_as_controlled_input_only():
    payload = _payload(FIXTURE_DIR / "raw_forbidden_semantics_source.json")

    assert set(payload["forbidden_test_input"]) == FORBIDDEN_INVESTMENT_FIELDS
    assert set(payload["raw_evidence"]).isdisjoint(FORBIDDEN_INVESTMENT_FIELDS)
    assert set(payload["expected_contract"]["raw_evidence"]).isdisjoint(
        FORBIDDEN_INVESTMENT_FIELDS
    )

    approved_outputs = {
        "raw_evidence": payload["raw_evidence"],
        "normalized_fundamentals": payload["expected_contract"][
            "normalized_fundamentals"
        ],
        "readiness": payload["expected_contract"]["readiness"],
    }
    rendered_outputs = {str(value) for value in _walk_values(approved_outputs)}

    assert rendered_outputs.isdisjoint(FORBIDDEN_INVESTMENT_FIELDS)
