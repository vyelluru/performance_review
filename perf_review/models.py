from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class SourceAccount:
    alias: str
    source_type: str
    config: dict[str, Any]
    ingest_modes: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass(slots=True)
class RawDocument:
    source_alias: str
    source_type: str
    external_id: str
    mode: str
    payload: dict[str, Any]


@dataclass(slots=True)
class ArtifactRecord:
    source_alias: str
    source_type: str
    artifact_type: str
    external_id: str
    title: str
    body_text: str
    author: str | None
    author_email: str | None
    occurred_at: str | None
    canonical_url: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskCandidate:
    key: str
    title: str
    artifact_ids: list[int]
    confidence: float
    reason: str
    start_at: str | None
    end_at: str | None
    primary_repo: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClaimRecord:
    section_id: str
    section_title: str
    claim_text: str
    artifact_ids: list[int]
    task_ids: list[int]


@dataclass(slots=True)
class ReviewRunArtifacts:
    self_review_markdown: str
    evidence_appendix_markdown: str
    gaps_markdown: str
    report_html: str
    claims: list[ClaimRecord]
    report_pdf: bytes = b""


def utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
