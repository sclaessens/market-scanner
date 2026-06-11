from __future__ import annotations


FUNDAMENTALS_PREFILL_CONTRACT_COLUMNS = (
    "ticker",
    "as_of_date",
    "source_name",
    "source_last_updated",
    "report_period",
    "currency",
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "gross_margin",
    "operating_margin",
    "debt_to_equity",
    "free_cash_flow_positive",
)

FUNDAMENTALS_PREFILL_AUDIT_FIELDS = (
    "run_timestamp",
    "provider_source_label",
    "requested_ticker_count",
    "matched_ticker_count",
    "missing_ticker_count",
    "written_row_count",
    "stale_row_count",
    "invalid_row_count",
    "partial_row_count",
    "duplicate_detection_result",
    "artifact_write_path",
    "validation_status",
    "failure_reason",
    "refresh_mode",
    "source_artifact_target",
    "dry_run",
)


def test_fundamentals_prefill_contract_keeps_source_data_schema_explicit():
    assert FUNDAMENTALS_PREFILL_CONTRACT_COLUMNS == (
        "ticker",
        "as_of_date",
        "source_name",
        "source_last_updated",
        "report_period",
        "currency",
        "revenue_growth_yoy",
        "eps_growth_yoy",
        "gross_margin",
        "operating_margin",
        "debt_to_equity",
        "free_cash_flow_positive",
    )


def test_fundamentals_prefill_audit_contract_is_operational_not_recommendation_authority():
    audit_text = " ".join(FUNDAMENTALS_PREFILL_AUDIT_FIELDS).lower()

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

    for term in forbidden_terms:
        assert term not in audit_text


def test_fundamentals_prefill_contract_requires_dry_run_and_write_visibility():
    assert "dry_run" in FUNDAMENTALS_PREFILL_AUDIT_FIELDS
    assert "artifact_write_path" in FUNDAMENTALS_PREFILL_AUDIT_FIELDS
    assert "validation_status" in FUNDAMENTALS_PREFILL_AUDIT_FIELDS
    assert "failure_reason" in FUNDAMENTALS_PREFILL_AUDIT_FIELDS
