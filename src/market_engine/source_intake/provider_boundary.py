from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class SourceProviderError(Exception):
    """Base error for controlled source provider failures."""


class ProviderUnavailableError(SourceProviderError):
    """Raised when a provider cannot return a controlled response."""


class UnsupportedTickerError(SourceProviderError):
    """Raised when a provider does not support a ticker."""


class InvalidTickerError(SourceProviderError):
    """Raised when a ticker is malformed or rejected by the provider."""


@dataclass(frozen=True)
class ProviderSourceResponse:
    ticker: str
    fields: dict[str, Any] = field(default_factory=dict)
    raw_evidence: Any | None = None
    raw_evidence_summary: str | None = None


class SourceProvider(Protocol):
    @property
    def name(self) -> str:
        """Stable provider name for source intake records."""

    def fetch_source(self, ticker: str) -> ProviderSourceResponse | None:
        """Return source facts for one ticker through an explicit boundary."""
