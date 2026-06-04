from importlib import reload
from pathlib import Path

import pytest

from market_scanner.fundamentals import (
    fundamentals_persistence,
    fundamentals_provider_adapter,
)
from market_scanner.fundamentals.fundamentals_persistence import (
    PersistenceIssueCode,
    prepare_persistence_batch,
    validate_normalized_fundamental_record,
    validate_raw_evidence_record,
    validate_readiness_record,
    write_synthetic_persistence_batch,
)
from market_scanner.fundamentals.fundamentals_provider_adapter import (
    ProviderFundamentalsIngestionResult,
    ingest_provider_fundamentals,
)
from market_scanner.fundamentals.fundamentals_provider_contracts import (
    ProviderCategory,
    ProviderContractIssueCode,
    ProviderRawEvidenceRecord,
    ProviderRawFieldEvidence,
    ProviderSourceDataReadinessRecord,
    ProviderSourceResponse,
    ProviderSourceStatus,
    validate_provider_source_response_shape,
)


ZERO_LIKE_MISSING_VALUES = (0, 0.0, "0", False, "")

FORBIDDEN_PROVIDER_FIELDS = {
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


def _field(
    name: str,
    value: object,
    *,
    currency: str = "USD",
    unit: str = "USD",
) -> ProviderRawFieldEvidence:
    return ProviderRawFieldEvidence(
        original_field_name=name,
        original_field_value=value,
        original_currency=currency,
        original_unit=unit,
    )


def _complete_raw_fields() -> dict[str, ProviderRawFieldEvidence]:
    return {
        "Revenues": _field("Revenues", "1000"),
        "GrossProfit": _field("GrossProfit", "600"),
        "OperatingIncomeLoss": _field("OperatingIncomeLoss", "240"),
        "NetIncomeLoss": _field("NetIncomeLoss", "155"),
        "EarningsPerShareDiluted": _field(
            "EarningsPerShareDiluted",
            "3.14",
            unit="USD per share",
        ),
        "Assets": _field("Assets", "5000"),
        "Liabilities": _field("Liabilities", "1700"),
        "StockholdersEquity": _field("StockholdersEquity", "3300"),
        "NetCashProvidedByUsedInOperatingActivities": _field(
            "NetCashProvidedByUsedInOperatingActivities",
            "222",
        ),
        "PaymentsToAcquirePropertyPlantAndEquipment": _field(
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "33",
        ),
        "FreeCashFlow": _field("FreeCashFlow", "189"),
    }


def _response(**overrides) -> ProviderSourceResponse:
    response = ProviderSourceResponse(
        provider_name="Synthetic Provider Integration Fixture",
        provider_category=ProviderCategory.REGULATORY_FILING.value,
        provider_record_id="SYNTH-PERSISTENCE-FY-2025",
        original_source_reference="synthetic-provider://AAPL/FY/2025",
        ticker="AAPL",
        symbol="AAPL",
        entity_identifier="AAPL-SYNTHETIC-ENTITY",
        source_timestamp="2026-02-15T00:00:00Z",
        retrieval_timestamp="2026-06-03T00:00:00Z",
        reported_period="FY",
        fiscal_year="2025",
        fiscal_quarter="",
        raw_fields=_complete_raw_fields(),
        provider_status=ProviderSourceStatus.AVAILABLE.value,
        provider_error_status="",
        missing_field_evidence=(),
        provenance_metadata="synthetic provider-to-persistence contract fixture",
        raw_payload_hash="sha256:synthetic-provider-to-persistence",
        capture_version="v2-provider-to-persistence-contract-v1",
    )
    values = {
        field_name: getattr(response, field_name)
        for field_name in response.__dataclass_fields__
    }
    values.update(overrides)
    return ProviderSourceResponse(**values)


def _persistence_raw(
    raw: ProviderRawEvidenceRecord,
    *,
    validation_warnings: tuple[str, ...] = (),
) -> dict[str, object]:
    raw_fields = {
        field.original_field_name: field.original_field_value
        for field in raw.raw_fields
    }
    missing_field_evidence = [
        {"field_name": field_name, "missing_state": "source_field_missing"}
        for field_name in raw.missing_field_evidence
    ]
    first_field = raw.raw_fields[0] if raw.raw_fields else None

    return {
        "raw_evidence_id": _raw_evidence_id(raw),
        "provider_name": raw.provider_name,
        "provider_category": raw.provider_category,
        "provider_record_id": raw.provider_record_id,
        "original_source_reference": raw.original_source_reference,
        "ticker": raw.ticker,
        "symbol": raw.symbol,
        "entity_identifier": raw.entity_identifier,
        "source_timestamp": raw.source_timestamp,
        "retrieval_timestamp": raw.retrieval_timestamp,
        "reported_period": raw.reported_period,
        "fiscal_year": raw.fiscal_year,
        "fiscal_quarter": raw.fiscal_quarter,
        "currency": first_field.original_currency if first_field else "USD",
        "unit": first_field.original_unit if first_field else "USD",
        "raw_fields": raw_fields,
        "missing_field_evidence": missing_field_evidence,
        "provenance_metadata": {
            "source_type": "synthetic_provider_boundary_output",
            "provider_status": raw.provider_status,
            "provider_error_status": raw.provider_error_status,
        },
        "raw_payload_hash": raw.raw_payload_hash,
        "capture_version": raw.capture_version,
        "validation_warnings": list(validation_warnings),
    }


def _persistence_normalized(
    result: ProviderFundamentalsIngestionResult,
    *,
    invalid_metric_names: tuple[str, ...] = (),
    validation_warnings: tuple[str, ...] = (),
) -> tuple[dict[str, object], ...]:
    default_currency = _default_currency(result.raw_evidence)
    default_unit = _default_unit(result.raw_evidence)

    records = []
    for record in result.normalized_records:
        is_invalid = record.metric_name in invalid_metric_names
        is_missing = record.metric_value is None or record.metric_value == ""
        metric_value = None if is_invalid else record.metric_value
        metric_value_status = _metric_value_status(record, is_invalid=is_invalid)
        records.append(
            {
                "normalized_record_id": (
                    f"norm-{result.raw_evidence.provider_record_id}-"
                    f"{record.metric_name}"
                ),
                "raw_evidence_id": _raw_evidence_id(result.raw_evidence),
                "provider_name": result.raw_evidence.provider_name,
                "original_source_reference": (
                    result.raw_evidence.original_source_reference
                ),
                "ticker": record.ticker,
                "entity_identifier": result.raw_evidence.entity_identifier,
                "metric_name": record.metric_name,
                "metric_value": metric_value,
                "metric_value_status": metric_value_status,
                "currency": record.currency if record.currency else default_currency,
                "currency_status": "reported" if record.currency else "inherited",
                "unit": record.metric_unit if record.metric_unit else default_unit,
                "unit_status": "reported" if record.metric_unit else "inherited",
                "reported_period": record.fiscal_period,
                "fiscal_year": record.fiscal_year,
                "fiscal_quarter": record.fiscal_quarter,
                "source_timestamp": result.raw_evidence.source_timestamp,
                "retrieval_timestamp": result.raw_evidence.retrieval_timestamp,
                "normalization_version": "v2-provider-to-persistence-contract-v1",
                "validation_warnings": list(validation_warnings)
                if is_invalid or is_missing
                else [],
            }
        )

    return tuple(records)


def _persistence_readiness(
    result: ProviderFundamentalsIngestionResult,
    *,
    source_data_status: str | None = None,
    readiness_state: str | None = None,
    provenance_status: str = "complete",
    parseability_status: str = "parseable",
    consistency_status: str = "consistent",
    freshness_status: str = "fresh",
    invalid_data_count: int = 0,
    readiness_warnings: tuple[str, ...] = (),
) -> dict[str, object]:
    readiness: ProviderSourceDataReadinessRecord = result.readiness_record

    return {
        "readiness_record_id": f"readiness-{result.raw_evidence.provider_record_id}",
        "raw_evidence_id": _raw_evidence_id(result.raw_evidence),
        "normalized_record_set_id": _normalized_record_set_id(result.raw_evidence),
        "ticker": readiness.ticker,
        "provider_name": result.raw_evidence.provider_name,
        "readiness_state": readiness_state or readiness.readiness_state,
        "source_data_status": source_data_status or readiness.source_data_status,
        "missing_fundamentals_count": readiness.missing_fundamentals_count,
        "partial_data_count": readiness.partial_data_count,
        "stale_data_count": readiness.stale_data_count,
        "invalid_data_count": invalid_data_count,
        "provenance_status": provenance_status,
        "parseability_status": parseability_status,
        "consistency_status": consistency_status,
        "freshness_status": freshness_status,
        "readiness_warnings": list(readiness_warnings),
        "readiness_version": "v2-provider-to-persistence-contract-v1",
    }


def _persistence_records(
    result: ProviderFundamentalsIngestionResult,
    *,
    raw_warnings: tuple[str, ...] = (),
    invalid_metric_names: tuple[str, ...] = (),
    normalized_warnings: tuple[str, ...] = (),
    readiness_overrides: dict[str, object] | None = None,
):
    readiness_overrides = readiness_overrides or {}
    return (
        _persistence_raw(result.raw_evidence, validation_warnings=raw_warnings),
        _persistence_normalized(
            result,
            invalid_metric_names=invalid_metric_names,
            validation_warnings=normalized_warnings,
        ),
        _persistence_readiness(result, **readiness_overrides),
    )


def _raw_evidence_id(raw: ProviderRawEvidenceRecord) -> str:
    return f"provider-raw-{raw.provider_record_id}"


def _normalized_record_set_id(raw: ProviderRawEvidenceRecord) -> str:
    return f"provider-normalized-{raw.provider_record_id}"


def _default_currency(raw: ProviderRawEvidenceRecord) -> str:
    for field in raw.raw_fields:
        if field.original_currency:
            return field.original_currency
    return "USD"


def _default_unit(raw: ProviderRawEvidenceRecord) -> str:
    for field in raw.raw_fields:
        if field.original_unit:
            return field.original_unit
    return "USD"


def _metric_value_status(record, *, is_invalid: bool) -> str:
    if is_invalid:
        return "invalid_unparseable"
    if record.metric_value is None or record.metric_value == "":
        return "missing_source_field"
    return "reported"


def _issue_codes(issues):
    return {issue.issue_code for issue in issues}


def _assert_no_production_outputs(tmp_path):
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "reports" / "daily" / "telegram_message.txt").exists()
    assert not (tmp_path / ".github" / "workflows").exists()


def test_complete_fake_provider_output_reaches_persistence_boundary(tmp_path):
    result = ingest_provider_fundamentals(_response())
    raw, normalized, readiness = _persistence_records(result)

    assert raw["provider_name"]
    assert raw["original_source_reference"]
    assert raw["source_timestamp"]
    assert all(record["raw_evidence_id"] == raw["raw_evidence_id"] for record in normalized)
    assert readiness["raw_evidence_id"] == raw["raw_evidence_id"]
    assert readiness["readiness_state"] == "available"
    assert readiness["source_data_status"] == "available"

    assert validate_raw_evidence_record(raw) == ()
    assert all(validate_normalized_fundamental_record(record) == () for record in normalized)
    assert validate_readiness_record(readiness) == ()

    batch = prepare_persistence_batch((raw,), normalized, (readiness,))
    write_result = write_synthetic_persistence_batch(batch, tmp_path / "persistence")

    assert batch.batch_status == "valid"
    assert write_result.batch_status == "written"
    assert write_result.output_root == str(tmp_path / "persistence")
    assert (tmp_path / "persistence" / "raw_source_evidence").is_dir()
    assert (tmp_path / "persistence" / "normalized_fundamentals").is_dir()
    assert (tmp_path / "persistence" / "source_data_readiness").is_dir()


def test_partial_fake_provider_output_remains_explicit_and_neutral():
    result = ingest_provider_fundamentals(
        _response(
            raw_fields={
                "Revenues": _field("Revenues", "1000"),
                "NetIncomeLoss": _field("NetIncomeLoss", "155"),
            },
            missing_field_evidence=("GrossProfit", "FreeCashFlow"),
        )
    )
    raw, normalized, readiness = _persistence_records(result)
    missing_records = [
        record for record in normalized if record["metric_value_status"] == "missing_source_field"
    ]

    assert raw["missing_field_evidence"] == [
        {"field_name": "GrossProfit", "missing_state": "source_field_missing"},
        {"field_name": "FreeCashFlow", "missing_state": "source_field_missing"},
    ]
    assert missing_records
    for record in missing_records:
        assert record["metric_value"] is None
        assert record["metric_value"] not in ZERO_LIKE_MISSING_VALUES
        assert validate_normalized_fundamental_record(record) == ()
    assert readiness["readiness_state"] == "partial"
    assert readiness["source_data_status"] == "partial"
    assert readiness["missing_fundamentals_count"] > 0
    assert set(readiness).isdisjoint(FORBIDDEN_OUTPUT_FIELDS)


def test_invalid_fake_provider_output_fails_closed_without_side_effects(tmp_path):
    result = ingest_provider_fundamentals(
        _response(raw_fields={**_complete_raw_fields(), "Revenues": _field("Revenues", "not-a-number")})
    )
    raw, normalized, readiness = _persistence_records(
        result,
        raw_warnings=("invalid_source_value",),
        invalid_metric_names=("revenue",),
        normalized_warnings=("unparseable_metric_value",),
        readiness_overrides={
            "source_data_status": "invalid",
            "readiness_state": "fail_closed",
            "parseability_status": "unparseable_values_present",
            "consistency_status": "invalid_source_value",
            "invalid_data_count": 1,
            "readiness_warnings": ("invalid_source_value",),
        },
    )
    revenue = next(record for record in normalized if record["metric_name"] == "revenue")
    batch = prepare_persistence_batch((raw,), normalized, (readiness,))
    write_result = write_synthetic_persistence_batch(batch, tmp_path / "persistence")

    assert revenue["metric_value"] is None
    assert revenue["metric_value_status"] == "invalid_unparseable"
    assert revenue["validation_warnings"] == ["unparseable_metric_value"]
    assert readiness["source_data_status"] == "invalid"
    assert batch.batch_status == "fail_closed"
    assert PersistenceIssueCode.FAIL_CLOSED_SOURCE_DATA_STATE in _issue_codes(batch.issues)
    assert write_result.write_records == ()
    _assert_no_production_outputs(tmp_path)


def test_provenance_gap_fake_provider_output_fails_closed():
    result = ingest_provider_fundamentals(
        _response(
            provider_name="",
            provider_record_id="",
            original_source_reference="",
            source_timestamp="",
        )
    )
    raw, normalized, readiness = _persistence_records(
        result,
        raw_warnings=("provenance_gap",),
        normalized_warnings=("provenance_gap",),
        readiness_overrides={
            "source_data_status": "provenance_gap",
            "readiness_state": "fail_closed",
            "provenance_status": "incomplete",
            "freshness_status": "unknown",
            "consistency_status": "blocked_by_provenance_gap",
            "readiness_warnings": ("provenance_gap",),
        },
    )
    batch = prepare_persistence_batch((raw,), normalized, (readiness,))

    assert validate_provider_source_response_shape(_response(provider_name=""))[0].issue_code == (
        ProviderContractIssueCode.MISSING_REQUIRED_VALUE
    )
    assert PersistenceIssueCode.MISSING_REQUIRED_VALUE in _issue_codes(batch.issues)
    assert PersistenceIssueCode.MISSING_PROVENANCE in _issue_codes(batch.issues)
    assert PersistenceIssueCode.FAIL_CLOSED_SOURCE_DATA_STATE in _issue_codes(batch.issues)
    assert batch.batch_status == "fail_closed"
    assert readiness["provenance_status"] == "incomplete"


def test_forbidden_investment_semantics_do_not_pass_as_approved_output():
    controlled_provider_input = {
        "provider_name": "Synthetic Provider Integration Fixture",
        "provider_category": ProviderCategory.REGULATORY_FILING.value,
        "provider_record_id": "SYNTH-PERSISTENCE-FORBIDDEN",
        "original_source_reference": "synthetic-provider://AAPL/FY/2025/forbidden",
        "ticker": "AAPL",
        "symbol": "AAPL",
        "source_timestamp": "2026-02-15T00:00:00Z",
        "retrieval_timestamp": "2026-06-03T00:00:00Z",
        "reported_period": "FY",
        "fiscal_year": "2025",
        "raw_fields": {"Revenues": _field("Revenues", "1000")},
        "provider_status": ProviderSourceStatus.AVAILABLE.value,
        "missing_field_evidence": (),
        "provenance_metadata": "synthetic controlled forbidden input",
        "raw_payload_hash": "sha256:synthetic-provider-forbidden",
        "capture_version": "v2-provider-to-persistence-contract-v1",
        **{field: "synthetic forbidden input" for field in FORBIDDEN_PROVIDER_FIELDS},
    }
    provider_issues = validate_provider_source_response_shape(controlled_provider_input)
    result = ingest_provider_fundamentals(_response())
    raw, normalized, readiness = _persistence_records(
        result,
        raw_warnings=("controlled_forbidden_input_present",),
        normalized_warnings=("controlled_forbidden_input_present",),
        readiness_overrides={
            "source_data_status": "controlled_forbidden_input_present",
            "readiness_state": "neutral_review_required",
            "consistency_status": "controlled_input_flagged",
            "readiness_warnings": ("controlled_forbidden_input_present",),
        },
    )
    batch = prepare_persistence_batch((raw,), normalized, (readiness,))

    assert {
        issue.field_name
        for issue in provider_issues
        if issue.issue_code == ProviderContractIssueCode.FORBIDDEN_FIELD
    } == {
        "target_price",
        "allocation",
        "conviction",
        "urgency",
        "recommendation",
        "tradeability",
    }
    assert FORBIDDEN_PROVIDER_FIELDS.isdisjoint(raw)
    assert all(FORBIDDEN_PROVIDER_FIELDS.isdisjoint(record) for record in normalized)
    assert FORBIDDEN_PROVIDER_FIELDS.isdisjoint(readiness)
    assert batch.batch_status == "fail_closed"
    assert PersistenceIssueCode.FAIL_CLOSED_SOURCE_DATA_STATE in _issue_codes(batch.issues)


def test_synthetic_write_stays_inside_tmp_path_with_deterministic_metadata(tmp_path):
    result = ingest_provider_fundamentals(_response())
    raw, normalized, readiness = _persistence_records(result)
    batch = prepare_persistence_batch((raw,), normalized, (readiness,))

    first = write_synthetic_persistence_batch(batch, tmp_path / "persistence")
    second = write_synthetic_persistence_batch(batch, tmp_path / "persistence")

    assert first == second
    assert first.batch_status == "written"
    assert tuple(record.record_family for record in first.write_records) == (
        "raw",
        *("normalized" for _ in normalized),
        "readiness",
    )
    assert all(str(tmp_path) in record.output_path for record in first.write_records)
    _assert_no_production_outputs(tmp_path)


@pytest.mark.parametrize(
    "forbidden_root",
    [
        Path("data"),
        Path("data/raw"),
        Path("data/processed"),
        Path("reports"),
        Path("reports/daily"),
        Path("reports/daily/telegram_message.txt"),
        Path(".github/workflows"),
    ],
)
def test_forbidden_output_roots_are_rejected_without_writes(forbidden_root):
    result = ingest_provider_fundamentals(_response())
    raw, normalized, readiness = _persistence_records(result)
    batch = prepare_persistence_batch((raw,), normalized, (readiness,))
    write_result = write_synthetic_persistence_batch(batch, forbidden_root)

    assert write_result.batch_status == "fail_closed"
    assert write_result.write_records == ()
    assert PersistenceIssueCode.FORBIDDEN_OUTPUT_PATH in _issue_codes(
        write_result.issues
    )


def test_provider_to_persistence_contracts_have_no_downstream_side_effects(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    reload(fundamentals_provider_adapter)
    reload(fundamentals_persistence)
    result = ingest_provider_fundamentals(_response())
    raw, normalized, readiness = _persistence_records(result)
    batch = prepare_persistence_batch((raw,), normalized, (readiness,))

    assert batch.raw_records
    assert batch.normalized_records
    assert batch.readiness_records
    assert list(tmp_path.iterdir()) == []
    assert not Path("data").exists()
    assert not Path("reports").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
    assert not Path(".github/workflows").exists()

    for module in (fundamentals_provider_adapter, fundamentals_persistence):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "requests",
            "urllib",
            "httpx",
            "aiohttp",
            "yfinance",
            "run_full_pipeline",
            "decision_engine",
        ):
            assert forbidden not in source
