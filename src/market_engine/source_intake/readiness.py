from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

from market_engine.source_intake.provider_boundary import (
    InvalidTickerError,
    ProviderSourceResponse,
    ProviderUnavailableError,
    SourceProviderError,
    UnsupportedTickerError,
)


class SourceReadinessStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID_TICKER = "INVALID_TICKER"


def is_missing_value(value: Any) -> bool:
    return value is None


def split_available_and_missing_fields(
    fields: dict[str, Any],
    required_fields: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required = tuple(required_fields)
    available = tuple(field for field in required if field in fields and not is_missing_value(fields[field]))
    missing = tuple(field for field in required if field not in fields or is_missing_value(fields[field]))
    return available, missing


def readiness_from_response(
    response: ProviderSourceResponse | None,
    required_fields: Iterable[str],
    error: SourceProviderError | None = None,
) -> SourceReadinessStatus:
    if isinstance(error, UnsupportedTickerError):
        return SourceReadinessStatus.UNSUPPORTED
    if isinstance(error, InvalidTickerError):
        return SourceReadinessStatus.INVALID_TICKER
    if isinstance(error, ProviderUnavailableError):
        return SourceReadinessStatus.PROVIDER_ERROR
    if error is not None:
        return SourceReadinessStatus.PROVIDER_ERROR
    if response is None:
        return SourceReadinessStatus.MISSING

    required = tuple(required_fields)
    if not required:
        return SourceReadinessStatus.AVAILABLE

    available, missing = split_available_and_missing_fields(response.fields, required)
    if available and not missing:
        return SourceReadinessStatus.AVAILABLE
    if available and missing:
        return SourceReadinessStatus.PARTIAL
    return SourceReadinessStatus.MISSING
