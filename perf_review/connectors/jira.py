from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from perf_review.connectors.base import BaseConnector, ConnectorError, FetchResult
from perf_review.connectors.http import append_query, auth_headers, http_get_json
from perf_review.models import ArtifactRecord, RawDocument
from perf_review.utils.text import flatten_atlassian_doc


class JiraConnector(BaseConnector):
    source_type = "jira"

    def supported_modes(self) -> list[str]:
        return ["import", "direct"]

    def fetch(self, mode: str, sync_state: dict[str, object] | None = None, import_path: str | None = None) -> FetchResult:
        if mode == "import":
            return self._fetch_import(import_path or self.config.get("import_path"))
        if mode == "direct":
            return self._fetch_direct(sync_state or {})
        raise ConnectorError(f"Unsupported Jira mode: {mode}")

    def _fetch_import(self, import_path: str | None) -> FetchResult:
        if not import_path:
            raise ConnectorError("Jira import mode requires --import-path or source config import_path")
        path = Path(import_path).expanduser()
        if not path.exists():
            raise ConnectorError(f"Import path does not exist: {path}")
        if path.suffix.lower() == ".csv":
            issues = list(self._load_csv(path))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                issues = payload
            elif isinstance(payload, dict):
                issues = payload.get("issues", [])
            else:
                issues = []
        return self._normalize_issues(issues, mode="import")

    def _fetch_direct(self, sync_state: dict[str, object]) -> FetchResult:
        token = self.secret_store.get_token(self.alias)
        if not token:
            raise ConnectorError(f"No token stored for source {self.alias}")
        base_url = self.config["base_url"].rstrip("/")
        project = self.config["project"]
        username = self.config.get("auth_username")
        headers = auth_headers(token, username=username)
        start_at = 0
        latest_cursor = sync_state.get("cursor")
        newest_seen = latest_cursor
        collected: list[dict[str, Any]] = []
        while True:
            jql = f"project={project} ORDER BY updated DESC"
            url = append_query(
                f"{base_url}/rest/api/3/search/jql",
                {
                    "jql": jql,
                    "fields": (
                        "summary,description,status,labels,issuetype,comment,assignee,reporter,"
                        "updated,created,parent,priority,resolution,resolutiondate,components,"
                        "customfield_10016,customfield_10020"
                    ),
                    "maxResults": 50,
                    "startAt": start_at,
                },
            )
            payload = http_get_json(url, headers=headers)
            issues = payload.get("issues", [])
            if not issues:
                break
            stop = False
            for issue in issues:
                updated = ((issue.get("fields") or {}).get("updated"))
                if latest_cursor and updated and updated <= str(latest_cursor):
                    stop = True
                    break
                if updated:
                    newest_seen = max(str(newest_seen or updated), str(updated))
                collected.append(issue)
            if stop or start_at + len(issues) >= int(payload.get("total", 0)):
                break
            start_at += len(issues)
        return self._normalize_issues(collected, mode="direct", cursor=str(newest_seen) if newest_seen else None)

    def _normalize_issues(self, issues: list[dict[str, Any]], mode: str, cursor: str | None = None) -> FetchResult:
        raw_documents: list[RawDocument] = []
        artifacts: list[ArtifactRecord] = []
        base_url = str(self.config.get("base_url") or "").rstrip("/")
        for issue in issues:
            key = issue.get("key") or issue.get("Issue key") or issue.get("issueKey")
            fields = issue.get("fields", issue)
            if not key:
                continue
            description = flatten_atlassian_doc(fields.get("description") or issue.get("Description") or "")
            title = fields.get("summary") or issue.get("Summary") or key
            created_at = fields.get("created") or issue.get("Created")
            updated_at = fields.get("updated") or issue.get("Updated")
            reporter = ((fields.get("reporter") or {}).get("displayName") if isinstance(fields.get("reporter"), dict) else issue.get("Reporter"))
            assignee = self._field_value(fields.get("assignee") or issue.get("Assignee"))
            status = self._field_value(fields.get("status") or issue.get("Status"))
            priority = self._field_value(fields.get("priority") or issue.get("Priority"))
            issue_type = self._field_value(fields.get("issuetype") or issue.get("Issue Type") or issue.get("Issue type"))
            labels = fields.get("labels") or issue.get("Labels") or []
            if isinstance(labels, str):
                labels = [label.strip() for label in labels.split(",") if label.strip()]
            components = [self._field_value(component) for component in (fields.get("components") or issue.get("Components") or [])]
            parent_key = self._parent_key(fields.get("parent") or issue.get("Parent"))
            story_points = self._story_points(fields, issue)
            resolution = self._field_value(fields.get("resolution") or issue.get("Resolution"))
            resolved_at = fields.get("resolutiondate") or issue.get("Resolved")
            raw_documents.append(RawDocument(self.alias, self.source_type, key, mode, issue))
            artifacts.append(
                ArtifactRecord(
                    source_alias=self.alias,
                    source_type=self.source_type,
                    artifact_type="issue",
                    external_id=key,
                    title=title,
                    body_text=description,
                    author=reporter,
                    author_email=None,
                    occurred_at=updated_at or created_at,
                    canonical_url=f"{base_url}/browse/{key}" if base_url else None,
                    metadata={
                        "issue_key": key,
                        "assignee": assignee,
                        "reporter": reporter,
                        "status": status,
                        "priority": priority,
                        "issue_type": issue_type,
                        "story_points": story_points,
                        "labels": labels,
                        "components": components,
                        "parent_key": parent_key,
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "resolved_at": resolved_at,
                        "resolution": resolution,
                    },
                )
            )
            comments = ((fields.get("comment") or {}).get("comments")) if isinstance(fields.get("comment"), dict) else issue.get("comments", [])
            for comment in comments or []:
                comment_id = comment.get("id") or f"{key}:{comment.get('created')}"
                raw_documents.append(RawDocument(self.alias, self.source_type, f"{key}:comment:{comment_id}", mode, comment))
                artifacts.append(
                    ArtifactRecord(
                        source_alias=self.alias,
                        source_type=self.source_type,
                        artifact_type="comment",
                        external_id=f"{key}:comment:{comment_id}",
                        title=f"Comment on {key}",
                        body_text=flatten_atlassian_doc(comment.get("body") or ""),
                        author=self._field_value(comment.get("author")),
                        author_email=None,
                        occurred_at=comment.get("created"),
                        canonical_url=f"{base_url}/browse/{key}" if base_url else None,
                        metadata={"issue_key": key},
                    )
                )
        return FetchResult(raw_documents=raw_documents, artifacts=artifacts, cursor=cursor)

    @staticmethod
    def _load_csv(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _field_value(value: Any) -> Any:
        if isinstance(value, dict):
            return value.get("displayName") or value.get("name") or value.get("value") or value.get("key")
        return value

    @staticmethod
    def _parent_key(value: Any) -> str | None:
        if isinstance(value, dict):
            return value.get("key")
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def _story_points(fields: dict[str, Any], issue: dict[str, Any]) -> float | int | None:
        candidates = [
            fields.get("customfield_10016"),
            fields.get("customfield_10020"),
            fields.get("Story Points"),
            fields.get("Story points"),
            fields.get("Story point estimate"),
            issue.get("Story Points"),
            issue.get("Story points"),
            issue.get("Story point estimate"),
        ]
        for candidate in candidates:
            if candidate in (None, ""):
                continue
            if isinstance(candidate, (int, float)):
                return candidate
            if isinstance(candidate, str):
                try:
                    numeric = float(candidate)
                    return int(numeric) if numeric.is_integer() else numeric
                except ValueError:
                    continue
        return None
