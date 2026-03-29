from __future__ import annotations

from typing import Any

from perf_review.connectors.base import BaseConnector, ConnectorError, FetchResult
from perf_review.connectors.http import append_query, auth_headers, http_get_json
from perf_review.models import ArtifactRecord, RawDocument
from perf_review.utils.text import flatten_atlassian_doc


class GitHubConnector(BaseConnector):
    source_type = "github"

    def fetch(self, mode: str, sync_state: dict[str, object] | None = None, import_path: str | None = None) -> FetchResult:
        if mode != "direct":
            raise ConnectorError("GitHub connector only supports direct mode")
        token = self.secret_store.get_token(self.alias)
        if not token:
            raise ConnectorError(f"No token stored for source {self.alias}")

        repo = self.config["repo"]
        api_base = self.config.get("api_base", "https://api.github.com").rstrip("/")
        headers = auth_headers(token)
        page = 1
        raw_documents: list[RawDocument] = []
        artifacts: list[ArtifactRecord] = []
        latest_cursor = sync_state.get("cursor") if sync_state else None
        newest_seen = latest_cursor

        while True:
            url = append_query(
                f"{api_base}/repos/{repo}/pulls",
                {"state": "all", "sort": "updated", "direction": "desc", "per_page": 100, "page": page},
            )
            payload = http_get_json(url, headers=headers)
            if not isinstance(payload, list) or not payload:
                break
            stop = False
            for pr in payload:
                updated_at = pr.get("updated_at")
                if latest_cursor and updated_at and updated_at <= str(latest_cursor):
                    stop = True
                    break
                newest_seen = max(str(newest_seen or updated_at), str(updated_at)) if updated_at else newest_seen
                pr_number = pr["number"]
                external_id = f"pr:{repo}#{pr_number}"
                raw_documents.append(RawDocument(self.alias, self.source_type, external_id, mode, pr))
                artifacts.append(
                    ArtifactRecord(
                        source_alias=self.alias,
                        source_type=self.source_type,
                        artifact_type="pr",
                        external_id=external_id,
                        title=pr.get("title", external_id),
                        body_text=pr.get("body") or "",
                        author=(pr.get("user") or {}).get("login"),
                        author_email=None,
                        occurred_at=pr.get("created_at"),
                        canonical_url=pr.get("html_url"),
                        metadata={"repo": repo, "number": pr_number, "state": pr.get("state"), "updated_at": updated_at},
                    )
                )
                for review in self._fetch_reviews(api_base, repo, pr_number, headers):
                    raw_documents.append(RawDocument(self.alias, self.source_type, f"review:{review['id']}", mode, review))
                    artifacts.append(
                        ArtifactRecord(
                            source_alias=self.alias,
                            source_type=self.source_type,
                            artifact_type="review",
                            external_id=f"review:{review['id']}",
                            title=f"Review on #{pr_number}",
                            body_text=review.get("body") or review.get("state") or "",
                            author=(review.get("user") or {}).get("login"),
                            author_email=None,
                            occurred_at=review.get("submitted_at"),
                            canonical_url=review.get("html_url"),
                            metadata={"repo": repo, "pr_number": pr_number, "state": review.get("state")},
                        )
                    )
                for comment in self._fetch_issue_comments(api_base, repo, pr_number, headers):
                    raw_documents.append(RawDocument(self.alias, self.source_type, f"comment:{comment['id']}", mode, comment))
                    artifacts.append(
                        ArtifactRecord(
                            source_alias=self.alias,
                            source_type=self.source_type,
                            artifact_type="comment",
                            external_id=f"comment:{comment['id']}",
                            title=f"Comment on #{pr_number}",
                            body_text=comment.get("body") or "",
                            author=(comment.get("user") or {}).get("login"),
                            author_email=None,
                            occurred_at=comment.get("created_at"),
                            canonical_url=comment.get("html_url"),
                            metadata={"repo": repo, "pr_number": pr_number},
                        )
                    )
            if stop or len(payload) < 100:
                break
            page += 1
        return FetchResult(raw_documents=raw_documents, artifacts=artifacts, cursor=str(newest_seen) if newest_seen else None)

    @staticmethod
    def _fetch_reviews(api_base: str, repo: str, pr_number: int, headers: dict[str, str]) -> list[dict[str, Any]]:
        url = append_query(f"{api_base}/repos/{repo}/pulls/{pr_number}/reviews", {"per_page": 100})
        payload = http_get_json(url, headers=headers)
        return payload if isinstance(payload, list) else []

    @staticmethod
    def _fetch_issue_comments(api_base: str, repo: str, pr_number: int, headers: dict[str, str]) -> list[dict[str, Any]]:
        url = append_query(f"{api_base}/repos/{repo}/issues/{pr_number}/comments", {"per_page": 100})
        payload = http_get_json(url, headers=headers)
        return payload if isinstance(payload, list) else []

