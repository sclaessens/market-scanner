import json
from pathlib import Path


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "fundamentals" / "persistence"
)

NORMALIZED_REQUIRED_FIELDS = {
    "normalized_record_id",
    "raw_evidence_id",
    "provider_name",
    "original_source_reference",
    "ticker",
    "entity_identifier",
    "metric_name",
    "metric_value",
    "metric_value_status",
    "currency",
    "currency_status",
    "unit",
    "unit_status",
    "reported_period",
    "fiscal_year",
    "fiscal_quarter",
    "source_timestamp",
    "retrieval_timestamp",
    "normalization_version",
    "validation_warnings",
}

GOVERNED_METRIC_NAMES = {
    "revenue",
    "gross_profit",
    "net_income",
}

FORBIDDEN_OUTPUT_FIELDS = {
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


def _normalized_records(payload: dict) -> list[dict]:
    return payload["expected_contract"]["normalized_fundamentals"]["records"]


def test_normalized_records_require_raw_evidence_provenance_linkage():
    for path in _fixture_paths():
        payload = _payload(path)
        raw = payload["raw_evidence"]

        for record in _normalized_records(payload):
            assert set(record) == NORMALIZED_REQUIRED_FIELDS
            assert record["raw_evidence_id"] == raw["raw_evidence_id"]
            assert record["provider_name"] == raw["provider_name"]
            assert record["original_source_reference"] == raw["original_source_reference"]
            assert record["ticker"] == raw["ticker"]
            assert record["entity_identifier"] == raw["entity_identifier"]
            assert record["reported_period"] == raw["reported_period"]
            assert record["fiscal_year"] == raw["fiscal_year"]
            assert record["fiscal_quarter"] == raw["fiscal_quarter"]
            assert record["source_timestamp"] == raw["source_timestamp"]
            assert record["retrieval_timestamp"] == raw["retrieval_timestamp"]


def test_normalized_records_require_governed_metric_and_status_fields():
    for path in _fixture_paths():
        payload = _payload(path)
        normalized = payload["expected_contract"]["normalized_fundamentals"]

        assert normalized["normalized_record_set_id"]
        assert (
            normalized["normalization_version"]
            == "v2-persistence-normalization-contract-v1"
        )

        for record in normalized["records"]:
            assert record["metric_name"] in GOVERNED_METRIC_NAMES
            assert record["metric_value_status"]
            assert record["currency_status"]
            assert record["unit_status"]
            assert (
                record["normalization_version"]
                == "v2-persistence-normalization-contract-v1"
            )
            assert isinstance(record["validation_warnings"], list)


def test_normalized_fiscal_period_metadata_is_preserved():
    for path in _fixture_paths():
        payload = _payload(path)

        for record in _normalized_records(payload):
            assert record["reported_period"] == payload["raw_evidence"]["reported_period"]
            assert record["fiscal_year"] == payload["raw_evidence"]["fiscal_year"]
            assert record["fiscal_quarter"] == payload["raw_evidence"]["fiscal_quarter"]


def test_normalized_missing_values_remain_explicit_and_never_zero_like():
    missing_records = []
    for path in _fixture_paths():
        for record in _normalized_records(_payload(path)):
            if record["metric_value"] is None:
                missing_records.append(record)

    assert missing_records
    for record in missing_records:
        assert record["metric_value"] is None
        assert record["metric_value"] not in ZERO_LIKE_MISSING_VALUES
        assert record["metric_value_status"] in {
            "missing_source_field",
            "invalid_unparseable",
            "blocked_by_provenance_gap",
        }


def test_normalized_validation_warnings_remain_visible_for_problem_states():
    expected_warning_fixtures = {
        "raw_partial_source.json",
        "raw_invalid_source.json",
        "raw_stale_source.json",
        "raw_provenance_gap_source.json",
        "raw_forbidden_semantics_source.json",
    }

    for path in _fixture_paths():
        warnings = {
            warning
            for record in _normalized_records(_payload(path))
            for warning in record["validation_warnings"]
        }

        if path.name in expected_warning_fixtures:
            assert warnings
        else:
            assert warnings == set()


def test_normalized_output_expectations_exclude_investment_semantics():
    for path in _fixture_paths():
        normalized = _payload(path)["expected_contract"]["normalized_fundamentals"]

        assert set(normalized).isdisjoint(FORBIDDEN_OUTPUT_FIELDS)
        for record in normalized["records"]:
            assert set(record).isdisjoint(FORBIDDEN_OUTPUT_FIELDS)
            rendered_values = {str(value) for value in record.values()}
            assert rendered_values.isdisjoint(FORBIDDEN_OUTPUT_FIELDS)
