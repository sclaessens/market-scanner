from types import MappingProxyType

import pytest

from market_scanner.shared.records import PipelineRecord


def test_pipeline_record_preserves_fixture_identity():
    row = {"candidate_id": "candidate-001", "symbol": "ALFA"}

    record = PipelineRecord.from_fixture_row(
        source_name="synthetic_universe_candidates",
        row_id_column="candidate_id",
        row=row,
    )

    assert record.source_name == "synthetic_universe_candidates"
    assert record.row_id == "candidate-001"
    assert record.values["symbol"] == "ALFA"


def test_pipeline_record_values_are_read_only_copy():
    row = {"candidate_id": "candidate-001", "symbol": "ALFA"}

    record = PipelineRecord.from_fixture_row(
        source_name="synthetic_universe_candidates",
        row_id_column="candidate_id",
        row=row,
    )
    row["symbol"] = "CHANGED"

    assert isinstance(record.values, MappingProxyType)
    assert record.values["symbol"] == "ALFA"
    with pytest.raises(TypeError):
        record.values["symbol"] = "BRVO"
