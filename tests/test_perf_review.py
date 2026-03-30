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
from perf_review.utils.text import flatten_atlassian_doc
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
                                "status": {"name": "Done"},
                                "priority": {"name": "High"},
                                "issuetype": {"name": "Story"},
                                "assignee": {"displayName": "Engineer"},
                                "reporter": {"displayName": "Manager"},
                                "labels": ["launch"],
                                "customfield_10016": 5,
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
            issue_artifact = next(artifact for artifact in imported.artifacts if artifact.artifact_type == "issue")
            self.assertEqual("Engineer", issue_artifact.metadata["assignee"])
            self.assertEqual("Done", issue_artifact.metadata["status"])
            self.assertEqual("High", issue_artifact.metadata["priority"])
            self.assertEqual("Story", issue_artifact.metadata["issue_type"])
            self.assertEqual(5, issue_artifact.metadata["story_points"])

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
                            "status": {"name": "In Progress"},
                            "priority": {"name": "Medium"},
                            "issuetype": {"name": "Bug"},
                            "assignee": {"displayName": "Engineer"},
                            "reporter": {"displayName": "Lead"},
                            "labels": ["reliability"],
                            "customfield_10016": 3,
                            "comment": {"comments": []},
                        },
                    },
                    {
                        "key": "ABC-457",
                        "fields": {
                            "summary": "Harden retries",
                            "description": {"content": [{"text": "Added guardrails"}]},
                            "updated": "2026-03-09T10:00:00Z",
                            "created": "2026-03-07T10:00:00Z",
                            "status": {"name": "Done"},
                            "priority": {"name": "Low"},
                            "issuetype": {"name": "Task"},
                            "assignee": {"displayName": "Engineer"},
                            "reporter": {"displayName": "Lead"},
                            "labels": ["stability"],
                            "customfield_10016": 1,
                            "comment": {"comments": []},
                        },
                    },
                ],
                "total": 2,
            }
            with mock.patch("perf_review.connectors.jira.http_get_json", return_value=payload):
                direct = live_connector.fetch("direct", sync_state={})
            direct_issues = [artifact for artifact in direct.artifacts if artifact.artifact_type == "issue"]
            self.assertEqual(2, len(direct_issues))
            direct_issue = direct_issues[0]
            self.assertEqual("Engineer", direct_issue.metadata["assignee"])
            self.assertEqual("In Progress", direct_issue.metadata["status"])
            self.assertEqual("Medium", direct_issue.metadata["priority"])
            self.assertEqual("Bug", direct_issue.metadata["issue_type"])
            self.assertEqual(3, direct_issue.metadata["story_points"])

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
                    },
                    {
                        "id": "43",
                        "title": "Follow-up Plan",
                        "body": {"storage": {"value": "<p>Ship supporting reliability work</p>"}},
                        "version": {"when": "2026-03-03T10:00:00Z"},
                        "history": {"createdBy": {"displayName": "Engineer"}},
                        "metadata": {"labels": {"results": [{"name": "reliability"}]}},
                        "_links": {"webui": "/wiki/spaces/ENG/pages/43"},
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
            empty_comment_payload = {"results": []}
            with mock.patch("perf_review.connectors.confluence.http_get_json", side_effect=[search_payload, comment_payload, empty_comment_payload]):
                direct = live_connector.fetch("direct", sync_state={})
            self.assertEqual(3, len(direct.artifacts))

    def test_document_import_is_chunked_by_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_root = Path(temp_dir) / "docs"
            docs_root.mkdir()
            (docs_root / "notes.md").write_text(
                "# Launch Plan\nCoordinate rollout with support.\n\n# Reliability Follow-up\nReduce alert fatigue and tune retries.\n",
                encoding="utf-8",
            )
            connector = ConfluenceConnector("docs", {"import_path": str(docs_root)}, MemorySecretStore())
            imported = connector.fetch("import")
            doc_artifacts = [artifact for artifact in imported.artifacts if artifact.artifact_type == "doc"]
            self.assertEqual(2, len(doc_artifacts))
            self.assertEqual(
                ["notes - Launch Plan", "notes - Reliability Follow-up"],
                [artifact.title for artifact in doc_artifacts],
            )
            self.assertTrue(all("document_external_id" in artifact.metadata for artifact in doc_artifacts))
            self.assertEqual([1, 2], [artifact.metadata["chunk_index"] for artifact in doc_artifacts])

    def test_flatten_atlassian_doc_ignores_adf_node_labels(self) -> None:
        payload = {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "blockquote",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {"type": "text", "text": "Priority on this"},
                                {"type": "hardBreak"},
                                {"type": "text", "text": "Ship blocker"},
                            ],
                        }
                    ],
                },
                {
                    "type": "bulletList",
                    "content": [
                        {
                            "type": "listItem",
                            "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Coordinate rollout"}]}],
                        }
                    ],
                },
            ],
        }
        flattened = flatten_atlassian_doc(payload)
        self.assertIn("Priority on this", flattened)
        self.assertIn("Ship blocker", flattened)
        self.assertIn("- Coordinate rollout", flattened)
        self.assertNotIn("blockquote", flattened)
        self.assertNotIn("paragraph", flattened)
        self.assertNotIn("doc", flattened)

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
                self.assertTrue((root / "review_template.md").exists())
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
            pdf_report = (root / ".perf_review" / "output" / "review_report.pdf").read_bytes()
            self.assertIn("Project summary:", self_review)
            self.assertIn("Total number of projects combined:", self_review)
            self.assertIn("[task:", self_review)
            self.assertTrue(pdf_report.startswith(b"%PDF-1.4"))
            db.close()

    def test_run_command_uses_editable_review_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "repo"
            imports = root / "imports"
            imports.mkdir()
            self._create_repo(repo)
            (imports / "jira.json").write_text(
                json.dumps(
                    [
                        {
                            "key": "ABC-123",
                            "fields": {
                                "summary": "Build onboarding flow",
                                "description": "Launch onboarding for new users",
                                "updated": "2026-03-10T10:00:00Z",
                                "created": "2026-03-05T10:00:00Z",
                                "labels": ["launch"],
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
                template_path = root / "review_template.md"
                template_path.write_text(
                    "# Performance Review Template\n\n"
                    "Use this exact structure for each project:\n"
                    "- Start with the project heading\n"
                    "- Include a line starting with 'Outcome:'\n",
                    encoding="utf-8",
                )
                config_manager = ConfigManager(root)
                config = config_manager.load_config()
                config["app"]["model"]["enabled"] = False
                config_manager.save_config(config)
                self.assertEqual(0, main(["source", "add", "git", "--path", str(repo)]))
                self.assertEqual(0, main(["source", "add", "jira", "--import-path", str(imports / "jira.json"), "--alias", "jira"]))
                self.assertEqual(0, main(["run", "--period", "2026-H1"]))
            finally:
                os.chdir(current)

            self_review = (root / ".perf_review" / "output" / "self_review.md").read_text(encoding="utf-8")
            self.assertIn("Project summary:", self_review)

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
                select id, title, description, artifact_count, repo_count, is_cross_repo, repo_names_json, jira_keys_json, source_anchor
                from tasks
                where source_anchor = 'anchor:issue:ABC-789'
                """
            ).fetchone()
            self.assertIsNotNone(shared_task)
            self.assertEqual("Shared ingestion pipeline rollout", shared_task["title"])
            self.assertEqual(2, shared_task["repo_count"])
            self.assertEqual(1, shared_task["is_cross_repo"])
            self.assertEqual(["service-a", "service-b"], json.loads(shared_task["repo_names_json"]))
            self.assertEqual(["ABC-789"], json.loads(shared_task["jira_keys_json"]))
            self.assertGreaterEqual(shared_task["artifact_count"], 6)
            self.assertIn("Coordinate ingestion pipeline updates", shared_task["description"])
            memberships = db.fetch_task_memberships(int(shared_task["id"]))
            membership_sources = {row["source_alias"] for row in memberships}
            self.assertTrue({"service-a", "service-b", "jira", "docs"}.issubset(membership_sources))
            repo_names = db.fetch_task_repo_names(int(shared_task["id"]))
            self.assertEqual(["service-a", "service-b"], repo_names)
            db.close()

    def test_related_issue_and_branch_clusters_consolidate_into_fewer_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "repo"
            docs_dir = root / "docs"
            docs_dir.mkdir()
            self._create_auth_repo(repo)
            (docs_dir / "auth.md").write_text(
                "# Auth hardening\n\nABC-200, ABC-201, and ABC-202 are part of the same auth stabilization push across login, token refresh, and env validation.\n",
                encoding="utf-8",
            )
            jira_import = root / "jira.json"
            jira_import.write_text(
                json.dumps(
                    [
                        {
                            "key": "ABC-200",
                            "fields": {
                                "summary": "Add auth login/logout/status commands to CLI",
                                "description": "Part of the auth stabilization push.",
                                "updated": "2026-03-15T10:00:00Z",
                                "created": "2026-03-12T10:00:00Z",
                                "comment": {"comments": []},
                            },
                        },
                        {
                            "key": "ABC-201",
                            "fields": {
                                "summary": "Fix token refresh race in auth flows",
                                "description": "Also part of the auth stabilization push.",
                                "updated": "2026-03-16T10:00:00Z",
                                "created": "2026-03-13T10:00:00Z",
                                "comment": {"comments": []},
                            },
                        },
                        {
                            "key": "ABC-202",
                            "fields": {
                                "summary": "Add env validation for auth config",
                                "description": "Completes the auth stabilization push.",
                                "updated": "2026-03-17T10:00:00Z",
                                "created": "2026-03-14T10:00:00Z",
                                "comment": {"comments": []},
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            current = Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(0, main(["init"]))
                self.assertEqual(0, main(["source", "add", "git", "--path", str(repo), "--alias", "repo"]))
                self.assertEqual(0, main(["source", "add", "jira", "--import-path", str(jira_import), "--alias", "jira"]))
                self.assertEqual(0, main(["source", "add", "confluence", "--import-path", str(docs_dir), "--alias", "docs"]))
                self.assertEqual(0, main(["ingest"]))
                self.assertEqual(0, main(["build", "--period", "2026-H1"]))
            finally:
                os.chdir(current)

            db = Database(root / ".perf_review" / "perf_review.db")
            task_count = db.connection.execute("select count(*) as count from tasks").fetchone()["count"]
            self.assertLessEqual(task_count, 3)
            auth_task = db.connection.execute(
                """
                select title, jira_keys_json, artifact_count
                from tasks
                order by artifact_count desc, title asc
                limit 1
                """
            ).fetchone()
            self.assertIn("ABC-200", json.loads(auth_task["jira_keys_json"]))
            self.assertIn("ABC-201", json.loads(auth_task["jira_keys_json"]))
            self.assertIn("ABC-202", json.loads(auth_task["jira_keys_json"]))
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

    def _create_auth_repo(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True, capture_output=True)
        target = path / "auth.txt"
        commits = [
            "ABC-200 add auth login/logout/status commands",
            "ABC-201 fix token refresh race condition",
            "ABC-202 add env validation for auth config",
            "ABC-201 add auth refresh concurrency test",
        ]
        for index, message in enumerate(commits, start=1):
            previous = target.read_text(encoding="utf-8") if target.exists() else ""
            target.write_text(previous + f"auth step {index}\n", encoding="utf-8")
            subprocess.run(["git", "add", "auth.txt"], cwd=path, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", message], cwd=path, check=True, capture_output=True)
        subprocess.run(["git", "checkout", "-b", "feat/ABC-202-env-validate"], cwd=path, check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
