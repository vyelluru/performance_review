from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from perf_review.cli.app import main
from perf_review.connectors.confluence import ConfluenceConnector
from perf_review.connectors.git_local import GitLocalConnector
from perf_review.connectors.github import GitHubConnector
from perf_review.connectors.jira import JiraConnector
from perf_review.graph.builder import rebuild_graph
from perf_review.llm.provider import BaseLLMProvider
from perf_review.review.engine import generate_review
from perf_review.store.db import Database
from perf_review.utils.config import ConfigManager, add_source
from perf_review.utils.secrets import MacOSKeychainSecretStore, MemorySecretStore
from perf_review.utils.time import parse_period


class PerfReviewTests(unittest.TestCase):
    def test_git_local_connector_ingests_commits_and_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir) / "repo"
            self._create_repo(repo)
            connector = GitLocalConnector("repo", {"path": str(repo)}, MemorySecretStore())
            result = connector.fetch("local")
            artifact_types = {artifact.artifact_type for artifact in result.artifacts}
            self.assertIn("commit", artifact_types)
            self.assertIn("branch", artifact_types)
            self.assertIn("repo", artifact_types)

    def test_jira_import_and_direct_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            import_path = Path(temp_dir) / "jira.json"
            import_path.write_text(
                json.dumps(
                    [
                        {
                            "key": "ABC-123",
                            "fields": {
                                "summary": "Build onboarding flow",
                                "description": "Shipped onboarding flow to beta customers",
                                "updated": "2026-03-01T10:00:00Z",
                                "created": "2026-02-28T10:00:00Z",
                                "comment": {"comments": [{"id": "c1", "body": "Coordinated rollout", "created": "2026-03-02T10:00:00Z", "author": {"displayName": "Teammate"}}]},
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            connector = JiraConnector("jira-demo", {"import_path": str(import_path)}, MemorySecretStore())
            imported = connector.fetch("import")
            self.assertEqual(2, len(imported.artifacts))

            secret_store = MemorySecretStore()
            secret_store.save_token("jira-live", "token")
            live_connector = JiraConnector(
                "jira-live",
                {"base_url": "https://example.atlassian.net", "project": "ABC", "auth_username": "user@example.com"},
                secret_store,
            )
            payload = {
                "issues": [
                    {
                        "key": "ABC-456",
                        "fields": {
                            "summary": "Improve reliability",
                            "description": {"content": [{"text": "Reduced alert noise"}]},
                            "updated": "2026-03-10T10:00:00Z",
                            "created": "2026-03-08T10:00:00Z",
                            "comment": {"comments": []},
                        },
                    }
                ],
                "total": 1,
            }
            with mock.patch("perf_review.connectors.jira.http_get_json", return_value=payload):
                direct = live_connector.fetch("direct", sync_state={})
            self.assertEqual(1, len([artifact for artifact in direct.artifacts if artifact.artifact_type == "issue"]))

    def test_confluence_import_and_direct_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_root = Path(temp_dir) / "docs"
            docs_root.mkdir()
            (docs_root / "launch.md").write_text("# Launch\nPartnered with design and support.\n", encoding="utf-8")
            connector = ConfluenceConnector("docs", {"import_path": str(docs_root)}, MemorySecretStore())
            imported = connector.fetch("import")
            self.assertEqual(1, len(imported.artifacts))

            secret_store = MemorySecretStore()
            secret_store.save_token("conf-live", "token")
            live_connector = ConfluenceConnector(
                "conf-live",
                {"base_url": "https://example.atlassian.net", "space": "ENG", "auth_username": "user@example.com"},
                secret_store,
            )
            search_payload = {
                "results": [
                    {
                        "id": "42",
                        "title": "Launch Plan",
                        "body": {"storage": {"value": "<p>Worked cross-team on launch</p>"}},
                        "version": {"when": "2026-03-04T10:00:00Z"},
                        "history": {"createdBy": {"displayName": "Engineer"}},
                        "metadata": {"labels": {"results": [{"name": "launch"}]}},
                        "_links": {"webui": "/wiki/spaces/ENG/pages/42"},
                    }
                ],
                "_links": {},
            }
            comment_payload = {
                "results": [
                    {
                        "id": "99",
                        "body": {"storage": {"value": "<p>Shared with stakeholders</p>"}},
                        "version": {"when": "2026-03-05T10:00:00Z"},
                        "history": {"createdBy": {"displayName": "Reviewer"}},
                    }
                ]
            }
            with mock.patch("perf_review.connectors.confluence.http_get_json", side_effect=[search_payload, comment_payload]):
                direct = live_connector.fetch("direct", sync_state={})
            self.assertEqual(2, len(direct.artifacts))

    def test_github_direct_sync(self) -> None:
        secret_store = MemorySecretStore()
        secret_store.save_token("github-demo", "token")
        connector = GitHubConnector("github-demo", {"repo": "example/repo"}, secret_store)
        pulls = [
            {
                "number": 12,
                "title": "ABC-123 Improve onboarding",
                "body": "Ships the feature",
                "created_at": "2026-03-01T10:00:00Z",
                "updated_at": "2026-03-02T10:00:00Z",
                "state": "closed",
                "html_url": "https://github.com/example/repo/pull/12",
                "user": {"login": "engineer"},
            }
        ]
        reviews = [{"id": 501, "body": "Looks good", "submitted_at": "2026-03-02T12:00:00Z", "state": "APPROVED", "user": {"login": "reviewer"}}]
        comments = [{"id": 601, "body": "Can we add monitoring?", "created_at": "2026-03-02T13:00:00Z", "user": {"login": "peer"}}]
        with mock.patch("perf_review.connectors.github.http_get_json", side_effect=[pulls, reviews, comments]):
            result = connector.fetch("direct", sync_state={})
        artifact_types = [artifact.artifact_type for artifact in result.artifacts]
        self.assertIn("pr", artifact_types)
        self.assertIn("review", artifact_types)
        self.assertIn("comment", artifact_types)

    def test_cli_pipeline_generates_cited_outputs_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "repo"
            imports = root / "imports"
            docs_dir = root / "docs"
            imports.mkdir()
            docs_dir.mkdir()
            self._create_repo(repo)
            (imports / "jira.json").write_text(
                json.dumps(
                    [
                        {
                            "key": "ABC-123",
                            "fields": {
                                "summary": "Build onboarding flow",
                                "description": "Built the onboarding flow and coordinated launch",
                                "updated": "2026-03-10T10:00:00Z",
                                "created": "2026-03-05T10:00:00Z",
                                "labels": ["launch"],
                                "comment": {"comments": [{"id": "c1", "body": "Partnered with support", "created": "2026-03-11T10:00:00Z", "author": {"displayName": "Teammate"}}]},
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (docs_dir / "launch.md").write_text("# Launch Plan\n\nWorked cross-team on launch and rollout.\n", encoding="utf-8")
            current = Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(0, main(["init"]))
                self.assertEqual(0, main(["source", "add", "git", "--path", str(repo), "--alias", "repo"]))
                self.assertEqual(0, main(["source", "add", "jira", "--import-path", str(imports / "jira.json"), "--alias", "jira"]))
                self.assertEqual(0, main(["source", "add", "confluence", "--import-path", str(docs_dir), "--alias", "docs"]))
                self.assertEqual(0, main(["ingest"]))
                self.assertEqual(0, main(["ingest"]))
                self.assertEqual(0, main(["build", "--period", "2026-H1"]))
                config_manager = ConfigManager(root)
                config = config_manager.load_config()
                config["app"]["model"]["enabled"] = False
                config_manager.save_config(config)
                self.assertEqual(0, main(["review", "draft", "--period", "2026-H1"]))
            finally:
                os.chdir(current)

            db = Database(root / ".perf_review" / "perf_review.db")
            artifact_count = db.connection.execute("select count(*) as count from artifacts").fetchone()["count"]
            claim_count = db.connection.execute("select count(*) as count from claim_citations").fetchone()["count"]
            self.assertEqual(7, artifact_count)
            self.assertGreater(claim_count, 0)
            self_review = (root / ".perf_review" / "output" / "self_review.md").read_text(encoding="utf-8")
            html_report = (root / ".perf_review" / "output" / "review_report.html").read_text(encoding="utf-8")
            self.assertIn("[task:", self_review)
            self.assertIn("<html", html_report)
            db.close()

    def test_git_only_repo_builds_multiple_tasks_not_single_repo_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "repo"
            self._create_topic_repo(repo)
            current = Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(0, main(["init"]))
                self.assertEqual(0, main(["source", "add", "git", "--path", str(repo), "--alias", "topic-repo"]))
                self.assertEqual(0, main(["ingest"]))
                self.assertEqual(0, main(["build", "--period", "2026-H1"]))
            finally:
                os.chdir(current)

            db = Database(root / ".perf_review" / "perf_review.db")
            tasks = db.connection.execute(
                """
                select title, summary, implementation_summary, impact_summary, collaboration_summary, complexity_score, complexity_reasoning
                from tasks order by id
                """
            ).fetchall()
            titles = [row["title"] for row in tasks]
            self.assertGreater(len(titles), 1)
            self.assertNotIn("main", [title.lower() for title in titles])
            self.assertTrue(all(row["summary"] for row in tasks))
            self.assertTrue(all(row["implementation_summary"] for row in tasks))
            self.assertTrue(all(row["complexity_reasoning"] for row in tasks))
            self.assertTrue(all(row["complexity_score"] >= 0.0 for row in tasks))
            db.close()

    def test_cross_repo_task_merges_shared_workstream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_a = root / "service-a"
            repo_b = root / "service-b"
            docs_dir = root / "docs"
            docs_dir.mkdir()
            self._create_shared_issue_repo(
                repo_a,
                [
                    "ABC-789 Build shared ingestion pipeline",
                    "ABC-789 Add ingestion metrics",
                ],
            )
            self._create_shared_issue_repo(
                repo_b,
                [
                    "ABC-789 Wire shared ingestion hooks",
                    "ABC-789 Validate ingestion contract",
                ],
            )
            (docs_dir / "rollout.md").write_text(
                "# Shared ingestion rollout\n\nABC-789 rollout across service-a and service-b with platform coordination.\n",
                encoding="utf-8",
            )
            jira_import = root / "jira.json"
            jira_import.write_text(
                json.dumps(
                    [
                        {
                            "key": "ABC-789",
                            "fields": {
                                "summary": "Shared ingestion pipeline rollout",
                                "description": "Coordinate ingestion pipeline updates across service-a and service-b.",
                                "updated": "2026-03-15T10:00:00Z",
                                "created": "2026-03-12T10:00:00Z",
                                "comment": {"comments": []},
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            current = Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(0, main(["init"]))
                self.assertEqual(0, main(["source", "add", "git", "--path", str(repo_a), "--alias", "service-a"]))
                self.assertEqual(0, main(["source", "add", "git", "--path", str(repo_b), "--alias", "service-b"]))
                self.assertEqual(0, main(["source", "add", "jira", "--import-path", str(jira_import), "--alias", "jira"]))
                self.assertEqual(0, main(["source", "add", "confluence", "--import-path", str(docs_dir), "--alias", "docs"]))
                self.assertEqual(0, main(["ingest"]))
                self.assertEqual(0, main(["build", "--period", "2026-H1"]))
            finally:
                os.chdir(current)

            db = Database(root / ".perf_review" / "perf_review.db")
            shared_task = db.connection.execute(
                """
                select id, title, repo_count, is_cross_repo, repo_names_json, source_anchor
                from tasks
                where source_anchor = 'anchor:issue:ABC-789'
                """
            ).fetchone()
            self.assertIsNotNone(shared_task)
            self.assertEqual("Shared ingestion pipeline rollout", shared_task["title"])
            self.assertEqual(2, shared_task["repo_count"])
            self.assertEqual(1, shared_task["is_cross_repo"])
            self.assertEqual(["service-a", "service-b"], json.loads(shared_task["repo_names_json"]))
            memberships = db.fetch_task_memberships(int(shared_task["id"]))
            membership_sources = {row["source_alias"] for row in memberships}
            self.assertTrue({"service-a", "service-b", "jira", "docs"}.issubset(membership_sources))
            repo_names = db.fetch_task_repo_names(int(shared_task["id"]))
            self.assertEqual(["service-a", "service-b"], repo_names)
            db.close()

    def test_keychain_secret_store_shells_out(self) -> None:
        store = MacOSKeychainSecretStore()
        with mock.patch("perf_review.utils.secrets.subprocess.run") as run_mock:
            run_mock.side_effect = [
                subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="secret-token\n", stderr=""),
            ]
            store.save_token("jira-demo", "secret-token")
            token = store.get_token("jira-demo")
        self.assertEqual("secret-token", token)
        self.assertEqual(2, run_mock.call_count)

    def _create_repo(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True, capture_output=True)
        (path / "app.txt").write_text("hello\n", encoding="utf-8")
        subprocess.run(["git", "add", "app.txt"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "ABC-123 Build onboarding flow"], cwd=path, check=True, capture_output=True)
        (path / "app.txt").write_text("hello\nworld\n", encoding="utf-8")
        subprocess.run(["git", "add", "app.txt"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Improve onboarding docs and rollout"], cwd=path, check=True, capture_output=True)

    def _create_topic_repo(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True, capture_output=True)
        commits = [
            ("voice.txt", "voice controls", "voice controls"),
            ("voice.txt", "voice synthesis polish", "voice synthesis polish"),
            ("ext.txt", "extension overlay", "extension overlay"),
            ("ext.txt", "extension shortcuts", "extension shortcuts"),
            ("doc.txt", "transcript parser", "transcript parser"),
        ]
        for filename, content, message in commits:
            target = path / filename
            previous = target.read_text(encoding="utf-8") if target.exists() else ""
            target.write_text(previous + content + "\n", encoding="utf-8")
            subprocess.run(["git", "add", filename], cwd=path, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", message], cwd=path, check=True, capture_output=True)

    def _create_shared_issue_repo(self, path: Path, commit_messages: list[str]) -> None:
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True, capture_output=True)
        target = path / "work.txt"
        for index, message in enumerate(commit_messages, start=1):
            previous = target.read_text(encoding="utf-8") if target.exists() else ""
            target.write_text(previous + f"step {index}\n", encoding="utf-8")
            subprocess.run(["git", "add", "work.txt"], cwd=path, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", message], cwd=path, check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
