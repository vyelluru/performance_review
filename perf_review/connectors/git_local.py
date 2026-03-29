from __future__ import annotations

import json
import subprocess
from pathlib import Path

from perf_review.connectors.base import BaseConnector, ConnectorError, FetchResult
from perf_review.models import ArtifactRecord, RawDocument


class GitLocalConnector(BaseConnector):
    source_type = "git"

    def supported_modes(self) -> list[str]:
        return ["local"]

    def fetch(self, mode: str, sync_state: dict[str, object] | None = None, import_path: str | None = None) -> FetchResult:
        if mode != "local":
            raise ConnectorError(f"{self.source_type} only supports local mode")
        repo_path = Path(self.config["path"]).expanduser()
        if not repo_path.exists():
            raise ConnectorError(f"Repository path does not exist: {repo_path}")

        repo_name = repo_path.name
        remote_url = self._run(repo_path, ["git", "remote", "get-url", "origin"], check=False).strip() or None
        raw_documents: list[RawDocument] = []
        artifacts: list[ArtifactRecord] = []

        commit_output = self._run(
            repo_path,
            [
                "git",
                "log",
                "--date=iso-strict",
                "--pretty=format:%H%x1f%an%x1f%ae%x1f%ad%x1f%s%x1f%b%x1e",
            ],
        )
        for record in commit_output.split("\x1e"):
            if not record.strip():
                continue
            commit_hash, author, email, occurred_at, subject, body = (record.split("\x1f") + ["", "", "", "", "", ""])[:6]
            payload = {
                "commit": commit_hash,
                "author": author,
                "email": email,
                "occurred_at": occurred_at,
                "subject": subject,
                "body": body.strip(),
                "repo_name": repo_name,
                "repo_path": str(repo_path),
                "remote_url": remote_url,
            }
            raw_documents.append(RawDocument(self.alias, self.source_type, commit_hash, mode, payload))
            artifacts.append(
                ArtifactRecord(
                    source_alias=self.alias,
                    source_type=self.source_type,
                    artifact_type="commit",
                    external_id=commit_hash,
                    title=subject or commit_hash[:12],
                    body_text=body.strip(),
                    author=author,
                    author_email=email,
                    occurred_at=occurred_at,
                    canonical_url=f"{remote_url.rstrip('/')}/commit/{commit_hash}" if remote_url and remote_url.startswith("http") else str(repo_path / ".git" / commit_hash),
                    metadata={"repo_name": repo_name, "repo_path": str(repo_path), "remote_url": remote_url},
                )
            )

        branch_output = self._run(
            repo_path,
            [
                "git",
                "for-each-ref",
                "refs/heads",
                "--format=%(refname:short)\t%(committerdate:iso-strict)\t%(objectname)\t%(subject)",
            ],
            check=False,
        )
        for record in branch_output.splitlines():
            if not record.strip():
                continue
            branch, occurred_at, commit_hash, subject = (record.split("\t") + ["", "", "", ""])[:4]
            external_id = f"branch:{branch}"
            payload = {
                "branch": branch,
                "occurred_at": occurred_at,
                "head_commit": commit_hash,
                "subject": subject,
                "repo_name": repo_name,
            }
            raw_documents.append(RawDocument(self.alias, self.source_type, external_id, mode, payload))
            artifacts.append(
                ArtifactRecord(
                    source_alias=self.alias,
                    source_type=self.source_type,
                    artifact_type="branch",
                    external_id=external_id,
                    title=branch,
                    body_text=subject or "",
                    author=None,
                    author_email=None,
                    occurred_at=occurred_at,
                    canonical_url=None,
                    metadata={"repo_name": repo_name, "head_commit": commit_hash},
                )
            )

        tags_output = self._run(
            repo_path,
            [
                "git",
                "for-each-ref",
                "refs/tags",
                "--format=%(refname:short)\t%(creatordate:iso-strict)\t%(objectname)\t%(subject)",
            ],
            check=False,
        )
        for record in tags_output.splitlines():
            if not record.strip():
                continue
            tag, occurred_at, object_hash, subject = (record.split("\t") + ["", "", "", ""])[:4]
            external_id = f"tag:{tag}"
            payload = {"tag": tag, "occurred_at": occurred_at, "object_hash": object_hash, "subject": subject, "repo_name": repo_name}
            raw_documents.append(RawDocument(self.alias, self.source_type, external_id, mode, payload))
            artifacts.append(
                ArtifactRecord(
                    source_alias=self.alias,
                    source_type=self.source_type,
                    artifact_type="tag",
                    external_id=external_id,
                    title=tag,
                    body_text=subject or "",
                    author=None,
                    author_email=None,
                    occurred_at=occurred_at,
                    canonical_url=None,
                    metadata={"repo_name": repo_name, "object_hash": object_hash},
                )
            )

        repo_payload = {"repo_name": repo_name, "repo_path": str(repo_path), "remote_url": remote_url}
        raw_documents.append(RawDocument(self.alias, self.source_type, f"repo:{repo_name}", mode, repo_payload))
        artifacts.append(
            ArtifactRecord(
                source_alias=self.alias,
                source_type=self.source_type,
                artifact_type="repo",
                external_id=f"repo:{repo_name}",
                title=repo_name,
                body_text=json.dumps(repo_payload, sort_keys=True),
                author=None,
                author_email=None,
                occurred_at=None,
                canonical_url=remote_url,
                metadata=repo_payload,
            )
        )
        return FetchResult(raw_documents=raw_documents, artifacts=artifacts)

    @staticmethod
    def _run(repo_path: Path, command: list[str], check: bool = True) -> str:
        result = subprocess.run(command, cwd=repo_path, capture_output=True, text=True, check=False)
        if check and result.returncode != 0:
            raise ConnectorError(result.stderr.strip() or f"Command failed: {' '.join(command)}")
        return result.stdout
