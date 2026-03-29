from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from perf_review.models import ArtifactRecord, RawDocument
from perf_review.utils.secrets import SecretStore


@dataclass(slots=True)
class FetchResult:
    raw_documents: list[RawDocument]
    artifacts: list[ArtifactRecord]
    cursor: str | None = None
    etag: str | None = None


class ConnectorError(RuntimeError):
    """Raised when a connector cannot fetch or parse data."""


class BaseConnector:
    source_type = "base"

    def __init__(self, alias: str, config: dict[str, Any], secret_store: SecretStore) -> None:
        self.alias = alias
        self.config = config
        self.secret_store = secret_store

    def supported_modes(self) -> list[str]:
        return ["direct"]

    def fetch(self, mode: str, sync_state: dict[str, Any] | None = None, import_path: str | None = None) -> FetchResult:
        raise NotImplementedError

    @staticmethod
    def _listify(records: Iterable[ArtifactRecord]) -> list[ArtifactRecord]:
        return list(records)

