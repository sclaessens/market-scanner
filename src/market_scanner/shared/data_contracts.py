"""Minimal v2 data contract metadata and fixture helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
V2_FIXTURE_ROOT = REPOSITORY_ROOT / "data" / "fixtures" / "v2"


class DataLifecycleStage(str, Enum):
    """Lifecycle stages defined by the v2 data lifecycle contract."""

    RAW_SOURCE = "raw_source"
    NORMALIZED_INPUT = "normalized_input"
    FIXTURE_INPUT = "fixture_input"
    GENERATED_OUTPUT = "generated_output"
    REPORTING_OUTPUT = "reporting_output"
    LOCAL_ONLY = "local_only"


@dataclass(frozen=True)
class FixtureContract:
    """Schema metadata for an approved RESET-4 fixture."""

    name: str
    relative_path: str
    required_columns: tuple[str, ...]
    lifecycle_stage: DataLifecycleStage = DataLifecycleStage.FIXTURE_INPUT
    classification: str = "approved_fixture"
    source: str = "synthetic"

    @property
    def path(self) -> Path:
        return REPOSITORY_ROOT / self.relative_path


APPROVED_FIXTURE_CONTRACTS: tuple[FixtureContract, ...] = (
    FixtureContract(
        name="synthetic_universe_candidates",
        relative_path="data/fixtures/v2/universe_candidates.csv",
        required_columns=(
            "candidate_id",
            "symbol",
            "company_name",
            "source_kind",
            "source_reference",
            "discovered_at",
            "row_identity",
            "inclusion_reason",
        ),
    ),
    FixtureContract(
        name="synthetic_portfolio_transactions",
        relative_path="data/fixtures/v2/portfolio_transactions.csv",
        required_columns=(
            "transaction_id",
            "portfolio_account",
            "symbol",
            "transaction_kind",
            "quantity_delta",
            "cash_amount",
            "currency",
            "occurred_at",
            "source_reference",
        ),
    ),
    FixtureContract(
        name="synthetic_source_data_readiness",
        relative_path="data/fixtures/v2/source_data_readiness.csv",
        required_columns=(
            "source_record_id",
            "symbol",
            "source_name",
            "metric_name",
            "metric_value",
            "metric_unit",
            "as_of_date",
            "readiness_state",
            "missing_value_policy",
            "review_required_reason",
        ),
    ),
)


FORBIDDEN_NON_DECISION_FIXTURE_VALUES = frozenset(
    {
        "buy",
        "sell",
        "conviction",
        "allocation",
        "urgency",
    }
)


def read_fixture_rows(contract: FixtureContract) -> list[dict[str, str]]:
    with contract.path.open(newline="", encoding="utf-8") as fixture_file:
        return list(csv.DictReader(fixture_file))
