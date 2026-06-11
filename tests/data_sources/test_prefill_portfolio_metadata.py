from __future__ import annotations


PORTFOLIO_METADATA_PREFILL_CONTRACT_COLUMNS = (
    "ticker",
    "sector",
    "industry",
    "asset_class",
    "currency",
    "metadata_source",
    "metadata_last_updated",
    "sector_taxonomy",
    "industry_group",
    "country",
    "region",
    "exchange",
    "notes",
)

PORTFOLIO_METADATA_ALLOWED_ASSET_CLASSES = (
    "Equity",
    "ETF",
    "Fund",
    "Cash",
)

PORTFOLIO_METADATA_PREFILL_AUDIT_FIELDS = (
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


def test_portfolio_metadata_prefill_contract_keeps_metadata_schema_explicit():
    assert PORTFOLIO_METADATA_PREFILL_CONTRACT_COLUMNS == (
        "ticker",
        "sector",
        "industry",
        "asset_class",
        "currency",
        "metadata_source",
        "metadata_last_updated",
        "sector_taxonomy",
        "industry_group",
        "country",
        "region",
        "exchange",
        "notes",
    )


def test_portfolio_metadata_prefill_does_not_expand_to_portfolio_action_authority():
    contract_text = " ".join(
        PORTFOLIO_METADATA_PREFILL_CONTRACT_COLUMNS
        + PORTFOLIO_METADATA_PREFILL_AUDIT_FIELDS
        + PORTFOLIO_METADATA_ALLOWED_ASSET_CLASSES
    ).lower()

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
        "position_size",
    }

    for term in forbidden_terms:
        assert term not in contract_text


def test_portfolio_metadata_prefill_asset_classes_remain_bounded():
    assert PORTFOLIO_METADATA_ALLOWED_ASSET_CLASSES == (
        "Equity",
        "ETF",
        "Fund",
        "Cash",
    )
    assert "Crypto" not in PORTFOLIO_METADATA_ALLOWED_ASSET_CLASSES
