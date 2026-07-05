from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from math import isfinite
from types import MappingProxyType
from typing import Any


GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION = (
    "market-engine-governor-factor-scoring-v1"
)
SCORE_SCALE = MappingProxyType(
    {
        "minimum": 0.0,
        "maximum": 100.0,
        "midpoint": 50.0,
        "contract_version": GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION,
    }
)
SCORE_PRECISION = 2


@dataclass(frozen=True)
class ComponentRule:
    component_id: str
    normalization_rule: str
    minimum: Decimal
    maximum: Decimal
    higher_is_favorable: bool = True


@dataclass(frozen=True)
class FactorScoreResult:
    score: float | None
    score_scale: Mapping[str, Any] | None
    score_components: tuple[Mapping[str, Any], ...]
    score_evidence_references: tuple[str, ...]
    score_limitations: tuple[str, ...]


_RULES: Mapping[str, tuple[ComponentRule, ...]] = MappingProxyType(
    {
        "fundamentals": (
            ComponentRule(
                "profitability_margin",
                "linear_clamp_negative_ten_to_thirty_percent_v1",
                Decimal("-0.10"),
                Decimal("0.30"),
            ),
            ComponentRule(
                "operating_cash_flow_margin",
                "linear_clamp_negative_ten_to_thirty_percent_v1",
                Decimal("-0.10"),
                Decimal("0.30"),
            ),
            ComponentRule(
                "return_on_assets",
                "linear_clamp_negative_five_to_twenty_percent_v1",
                Decimal("-0.05"),
                Decimal("0.20"),
            ),
        ),
        "growth": (
            ComponentRule(
                "revenue_growth_rate",
                "linear_clamp_negative_ten_to_thirty_percent_v1",
                Decimal("-0.10"),
                Decimal("0.30"),
            ),
            ComponentRule(
                "earnings_growth_rate",
                "linear_clamp_negative_twenty_to_forty_percent_v1",
                Decimal("-0.20"),
                Decimal("0.40"),
            ),
            ComponentRule(
                "cash_flow_growth_rate",
                "linear_clamp_negative_twenty_to_forty_percent_v1",
                Decimal("-0.20"),
                Decimal("0.40"),
            ),
        ),
        "risk": (
            ComponentRule(
                "debt_to_assets_ratio",
                "inverse_linear_clamp_twenty_to_eighty_percent_v1",
                Decimal("0.20"),
                Decimal("0.80"),
                higher_is_favorable=False,
            ),
            ComponentRule(
                "net_debt_to_cash_flow",
                "inverse_linear_clamp_zero_to_five_v1",
                Decimal("0"),
                Decimal("5"),
                higher_is_favorable=False,
            ),
            ComponentRule(
                "cash_coverage_ratio",
                "linear_clamp_zero_to_two_v1",
                Decimal("0"),
                Decimal("2"),
            ),
        ),
        "data_confidence": (
            ComponentRule(
                "source_support_ratio",
                "linear_clamp_zero_to_one_v1",
                Decimal("0"),
                Decimal("1"),
            ),
            ComponentRule(
                "provenance_completeness_ratio",
                "linear_clamp_zero_to_one_v1",
                Decimal("0"),
                Decimal("1"),
            ),
            ComponentRule(
                "evidence_completeness_ratio",
                "linear_clamp_zero_to_one_v1",
                Decimal("0"),
                Decimal("1"),
            ),
        ),
    }
)


def score_factor(
    *,
    factor: str,
    state: str,
    factor_evidence: Mapping[str, Any],
    evidence_references: Sequence[str],
    conflicting_evidence_references: Sequence[str] = (),
) -> FactorScoreResult:
    """Score one eligible factor from explicit approved component evidence."""
    if state != "evaluable":
        return _unscored(f"score_ineligible_factor_state:{state}")
    if conflicting_evidence_references:
        return _unscored("score_blocked_by_conflicting_evidence")
    rules = _RULES.get(factor)
    if rules is None:
        return _unscored("factor_scoring_rule_not_implemented")

    raw_inputs = factor_evidence.get("score_inputs")
    if not isinstance(raw_inputs, Mapping):
        return _unscored("approved_scoring_inputs_missing")
    if (
        raw_inputs.get("contract_version")
        != GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION
    ):
        return _unscored("score_input_contract_not_approved")
    if factor == "growth" and (
        raw_inputs.get("period_alignment") != "aligned_multi_period"
        or not _valid_period_count(raw_inputs.get("period_count"))
    ):
        return _unscored("growth_period_alignment_invalid")

    raw_components = raw_inputs.get("components")
    if not isinstance(raw_components, list):
        return _unscored("score_components_malformed")
    components_by_id: dict[str, Mapping[str, Any]] = {}
    for item in raw_components:
        if not isinstance(item, Mapping):
            return _unscored("score_components_malformed")
        component_id = item.get("component_id")
        if (
            not isinstance(component_id, str)
            or not component_id
            or component_id in components_by_id
        ):
            return _unscored("score_component_identity_invalid")
        components_by_id[component_id] = item

    expected_ids = {rule.component_id for rule in rules}
    if set(components_by_id) != expected_ids:
        return _unscored("required_score_components_incomplete")

    reference_set = set(evidence_references)
    normalized: list[tuple[ComponentRule, Mapping[str, Any], Decimal]] = []
    for rule in rules:
        item = components_by_id[rule.component_id]
        if item.get("normalization_rule") != rule.normalization_rule:
            return _unscored(
                f"normalization_rule_invalid:{rule.component_id}"
            )
        reference = item.get("evidence_reference")
        if (
            not isinstance(reference, str)
            or not reference
            or reference not in reference_set
        ):
            return _unscored(
                f"score_evidence_reference_invalid:{rule.component_id}"
            )
        value = _decimal_input(item.get("input_value"))
        if value is None:
            return _unscored(f"score_input_invalid:{rule.component_id}")
        if _validated_limitations(item.get("limitations", [])) is None:
            return _unscored(
                f"score_component_limitations_invalid:{rule.component_id}"
            )
        normalized.append((rule, item, _normalize(value, rule)))

    divisor = Decimal(len(normalized))
    score = _round_decimal(
        sum((value for _, _, value in normalized), Decimal("0")) / divisor
    )
    component_results = tuple(
        {
            "component_id": rule.component_id,
            "evidence_reference": item["evidence_reference"],
            "input_value": float(_decimal_input(item["input_value"])),
            "normalization_rule": rule.normalization_rule,
            "normalized_value": float(_round_decimal(value)),
            "normalized_contribution": float(
                _round_decimal(value / divisor)
            ),
            "limitations": _validated_limitations(item.get("limitations", [])),
        }
        for rule, item, value in normalized
    )
    return FactorScoreResult(
        score=float(score),
        score_scale=dict(SCORE_SCALE),
        score_components=component_results,
        score_evidence_references=tuple(
            item["evidence_reference"] for _, item, _ in normalized
        ),
        score_limitations=tuple(
            sorted(
                {
                    limitation
                    for component in component_results
                    for limitation in component["limitations"]
                }
            )
        ),
    )


def _normalize(value: Decimal, rule: ComponentRule) -> Decimal:
    bounded = min(max(value, rule.minimum), rule.maximum)
    ratio = (bounded - rule.minimum) / (rule.maximum - rule.minimum)
    if not rule.higher_is_favorable:
        ratio = Decimal("1") - ratio
    return ratio * Decimal("100")


def _decimal_input(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not isfinite(value):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _valid_period_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 3


def _validated_limitations(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item and item == item.strip()
        for item in value
    ):
        return None
    return tuple(sorted(set(value)))


def _round_decimal(value: Decimal) -> Decimal:
    quantum = Decimal("1").scaleb(-SCORE_PRECISION)
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def _unscored(reason: str) -> FactorScoreResult:
    return FactorScoreResult(
        score=None,
        score_scale=None,
        score_components=(),
        score_evidence_references=(),
        score_limitations=(reason,),
    )
