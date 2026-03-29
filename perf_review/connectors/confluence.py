from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from perf_review.connectors.base import BaseConnector, ConnectorError, FetchResult
from perf_review.connectors.http import append_query, auth_headers, http_get_json
from perf_review.models import ArtifactRecord, RawDocument
from perf_review.utils.text import chunk_document_text, extract_text_from_file, html_to_text


class ConfluenceConnector(BaseConnector):
    source_type = "confluence"

    def supported_modes(self) -> list[str]:
        return ["import", "direct"]

    def fetch(self, mode: str, sync_state: dict[str, object] | None = None, import_path: str | None = None) -> FetchResult:
        if mode == "import":
            return self._fetch_import(import_path or self.config.get("import_path"))
        if mode == "direct":
            return self._fetch_direct(sync_state or {})
        raise ConnectorError(f"Unsupported Confluence mode: {mode}")

    def _fetch_import(self, import_path: str | None) -> FetchResult:
        if not import_path:
            raise ConnectorError("Confluence import mode requires --import-path or source config import_path")
        root = Path(import_path).expanduser()
        if not root.exists():
            raise ConnectorError(f"Import path does not exist: {root}")
        files = [root] if root.is_file() else sorted(file for file in root.rglob("*") if file.is_file())
        raw_documents: list[RawDocument] = []
        artifacts: list[ArtifactRecord] = []
        for file_path in files:
            text = extract_text_from_file(file_path)
            external_id = str(file_path.resolve())
            payload = {"path": external_id, "title": file_path.stem, "suffix": file_path.suffix.lower(), "text": text}
            raw_documents.append(RawDocument(self.alias, self.source_type, external_id, "import", payload))
            artifacts.extend(
                self._document_artifacts(
                    external_id=external_id,
                    title=file_path.stem,
                    text=text,
                    author=None,
                    occurred_at=None,
                    canonical_url=None,
                    metadata={"path": external_id, "suffix": file_path.suffix.lower()},
                )
            )
        return FetchResult(raw_documents=raw_documents, artifacts=artifacts)

    def _fetch_direct(self, sync_state: dict[str, object]) -> FetchResult:
        token = self.secret_store.get_token(self.alias)
        if not token:
            raise ConnectorError(f"No token stored for source {self.alias}")
        username = self.config.get("auth_username")
        base_url = self.config["base_url"].rstrip("/")
        space = self.config["space"]
        headers = auth_headers(token, username=username)
        latest_cursor = sync_state.get("cursor")
        newest_seen = latest_cursor
        start = 0
        pages: list[dict[str, Any]] = []
        while True:
            url = append_query(
                f"{base_url}/wiki/rest/api/content/search",
                {
                    "cql": f"space={space} order by lastmodified desc",
                    "expand": "body.storage,version,history,metadata.labels",
                    "limit": 25,
                    "start": start,
                },
            )
            payload = http_get_json(url, headers=headers)
            results = payload.get("results", [])
            if not results:
                break
            stop = False
            for page in results:
                modified_at = ((page.get("version") or {}).get("when"))
                if latest_cursor and modified_at and modified_at <= str(latest_cursor):
                    stop = True
                    break
                if modified_at:
                    newest_seen = max(str(newest_seen or modified_at), str(modified_at))
                pages.append(page)
            if stop or not payload.get("_links", {}).get("next"):
                break
            start += len(results)
        raw_documents: list[RawDocument] = []
        artifacts: list[ArtifactRecord] = []
        for page in pages:
            page_id = page["id"]
            title = page.get("title", f"Confluence page {page_id}")
            body_text = html_to_text((((page.get("body") or {}).get("storage") or {}).get("value")) or "")
            external_id = f"page:{page_id}"
            raw_documents.append(RawDocument(self.alias, self.source_type, external_id, "direct", page))
            artifacts.extend(
                self._document_artifacts(
                    external_id=external_id,
                    title=title,
                    text=body_text,
                    author=((page.get("history") or {}).get("createdBy") or {}).get("displayName"),
                    occurred_at=((page.get("version") or {}).get("when")),
                    canonical_url=f"{base_url}{(page.get('_links') or {}).get('webui', '')}" if (page.get("_links") or {}).get("webui") else None,
                    metadata={"space": space, "labels": [label.get("name") for label in ((page.get("metadata") or {}).get("labels") or {}).get("results", [])]},
                )
            )
            comments = self._fetch_comments(base_url, page_id, headers)
            for comment in comments:
                comment_id = comment["id"]
                raw_documents.append(RawDocument(self.alias, self.source_type, f"comment:{comment_id}", "direct", comment))
                artifacts.append(
                    ArtifactRecord(
                        source_alias=self.alias,
                        source_type=self.source_type,
                        artifact_type="comment",
                        external_id=f"comment:{comment_id}",
                        title=f"Comment on {title}",
                        body_text=html_to_text((((comment.get("body") or {}).get("storage") or {}).get("value")) or ""),
                        author=((comment.get("history") or {}).get("createdBy") or {}).get("displayName"),
                        author_email=None,
                        occurred_at=((comment.get("version") or {}).get("when")),
                        canonical_url=None,
                        metadata={"page_id": page_id, "space": space},
                    )
                )
        return FetchResult(raw_documents=raw_documents, artifacts=artifacts, cursor=str(newest_seen) if newest_seen else None)

    def _document_artifacts(
        self,
        external_id: str,
        title: str,
        text: str,
        author: str | None,
        occurred_at: str | None,
        canonical_url: str | None,
        metadata: dict[str, Any],
    ) -> list[ArtifactRecord]:
        artifacts: list[ArtifactRecord] = []
        for chunk in chunk_document_text(title, text):
            artifacts.append(
                ArtifactRecord(
                    source_alias=self.alias,
                    source_type=self.source_type,
                    artifact_type="doc",
                    external_id=f"{external_id}#chunk-{chunk.chunk_index}",
                    title=chunk.title,
                    body_text=chunk.body_text,
                    author=author,
                    author_email=None,
                    occurred_at=occurred_at,
                    canonical_url=canonical_url,
                    metadata={
                        **metadata,
                        "document_external_id": external_id,
                        "chunk_index": chunk.chunk_index,
                        "section_title": chunk.section_title,
                        "heading_path": chunk.heading_path,
                    },
                )
            )
        return artifacts

    @staticmethod
    def _fetch_comments(base_url: str, page_id: str, headers: dict[str, str]) -> list[dict[str, Any]]:
        url = append_query(
            f"{base_url}/wiki/rest/api/content/{page_id}/child/comment",
            {"expand": "body.storage,version,history", "limit": 100},
        )
        payload = http_get_json(url, headers=headers)
        return payload.get("results", [])
