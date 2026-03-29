from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from perf_review.models import ArtifactRecord, ClaimRecord, RawDocument, SourceAccount, utcnow_iso


SCHEMA_STATEMENTS = [
    """
    create table if not exists sources (
      id integer primary key,
      alias text not null unique,
      source_type text not null,
      enabled integer not null default 1,
      config_json text not null,
      ingest_modes_json text not null,
      created_at text not null,
      updated_at text not null
    )
    """,
    """
    create table if not exists source_accounts (
      id integer primary key,
      alias text not null unique,
      source_type text not null,
      enabled integer not null default 1,
      config_json text not null,
      ingest_modes_json text not null,
      created_at text not null,
      updated_at text not null
    )
    """,
    """
    create table if not exists sync_state (
      id integer primary key,
      source_alias text not null unique,
      mode text not null,
      cursor text,
      etag text,
      last_synced_at text,
      status text,
      updated_at text not null
    )
    """,
    """
    create table if not exists raw_documents (
      id integer primary key,
      source_alias text not null,
      source_type text not null,
      external_id text not null,
      mode text not null,
      payload_json text not null,
      fingerprint text not null,
      imported_at text not null,
      unique(source_alias, external_id, mode, fingerprint)
    )
    """,
    """
    create table if not exists artifacts (
      id integer primary key,
      source_alias text not null,
      source_type text not null,
      artifact_type text not null,
      external_id text not null,
      title text not null,
      body_text text not null,
      author text,
      author_email text,
      occurred_at text,
      canonical_url text,
      fingerprint text not null,
      metadata_json text not null,
      created_at text not null,
      updated_at text not null,
      unique(source_alias, external_id, artifact_type)
    )
    """,
    """
    create virtual table if not exists artifact_search using fts5(
      title,
      body_text
    )
    """,
    """
    create table if not exists entities (
      id integer primary key,
      entity_type text not null,
      value text not null,
      display_name text not null,
      metadata_json text not null,
      created_at text not null,
      updated_at text not null,
      unique(entity_type, value)
    )
    """,
    """
    create table if not exists tasks (
      id integer primary key,
      task_key text not null unique,
      title text not null,
      description text not null,
      summary text not null default '',
      implementation_summary text not null default '',
      impact_summary text not null default '',
      collaboration_summary text not null default '',
      complexity_score real not null default 0.0,
      complexity_reasoning text not null default '',
      status text not null default 'inferred',
      source_anchor text,
      confidence real not null,
      start_at text,
      end_at text,
      primary_repo text,
      repo_count integer not null default 0,
      is_cross_repo integer not null default 0,
      repo_names_json text not null default '[]',
      metadata_json text not null,
      created_at text not null,
      updated_at text not null
    )
    """,
    """
    create table if not exists task_memberships (
      task_id integer not null,
      artifact_id integer not null,
      membership_score real not null,
      membership_reason text not null,
      primary key(task_id, artifact_id)
    )
    """,
    """
    create table if not exists edges (
      id integer primary key,
      from_kind text not null,
      from_id integer not null,
      rel_type text not null,
      to_kind text not null,
      to_id integer not null,
      provenance_artifact_id integer not null default -1,
      confidence real not null default 1.0,
      valid_from text,
      valid_to text,
      metadata_json text not null,
      created_at text not null,
      unique(from_kind, from_id, rel_type, to_kind, to_id, provenance_artifact_id)
    )
    """,
    """
    create table if not exists competencies (
      id integer primary key,
      competency_id text not null unique,
      title text not null,
      description text not null,
      keywords_json text not null,
      created_at text not null,
      updated_at text not null
    )
    """,
    """
    create table if not exists review_runs (
      id integer primary key,
      period_label text not null,
      rubric_path text not null,
      model_provider text not null,
      self_review_markdown text not null,
      evidence_appendix_markdown text not null,
      gaps_markdown text not null,
      report_html text not null,
      created_at text not null
    )
    """,
    """
    create table if not exists claim_citations (
      id integer primary key,
      review_run_id integer not null,
      section_id text not null,
      section_title text not null,
      claim_text text not null,
      artifact_ids_json text not null,
      task_ids_json text not null,
      created_at text not null
    )
    """,
]


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("pragma journal_mode = wal")
        self.connection.execute("pragma foreign_keys = on")

    def init_schema(self) -> None:
        for statement in SCHEMA_STATEMENTS:
            self.connection.execute(statement)
        self._migrate_schema()
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def _migrate_schema(self) -> None:
        self._ensure_columns(
            "tasks",
            {
                "summary": "text not null default ''",
                "implementation_summary": "text not null default ''",
                "impact_summary": "text not null default ''",
                "collaboration_summary": "text not null default ''",
                "complexity_score": "real not null default 0.0",
                "complexity_reasoning": "text not null default ''",
                "status": "text not null default 'inferred'",
                "source_anchor": "text",
                "repo_count": "integer not null default 0",
                "is_cross_repo": "integer not null default 0",
                "repo_names_json": "text not null default '[]'",
            },
        )

    def _ensure_columns(self, table_name: str, columns: dict[str, str]) -> None:
        existing = {
            row["name"]
            for row in self.connection.execute(f"pragma table_info({table_name})").fetchall()
        }
        for column_name, column_def in columns.items():
            if column_name in existing:
                continue
            self.connection.execute(f"alter table {table_name} add column {column_name} {column_def}")

    def upsert_source(self, source: SourceAccount) -> None:
        now = utcnow_iso()
        payload = (source.alias, source.source_type, int(source.enabled), json.dumps(source.config, sort_keys=True), json.dumps(source.ingest_modes), now, now)
        self.connection.execute(
            """
            insert into source_accounts(alias, source_type, enabled, config_json, ingest_modes_json, created_at, updated_at)
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(alias) do update set
              source_type=excluded.source_type,
              enabled=excluded.enabled,
              config_json=excluded.config_json,
              ingest_modes_json=excluded.ingest_modes_json,
              updated_at=excluded.updated_at
            """,
            payload,
        )
        self.connection.execute(
            """
            insert into sources(alias, source_type, enabled, config_json, ingest_modes_json, created_at, updated_at)
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(alias) do update set
              source_type=excluded.source_type,
              enabled=excluded.enabled,
              config_json=excluded.config_json,
              ingest_modes_json=excluded.ingest_modes_json,
              updated_at=excluded.updated_at
            """,
            payload,
        )
        self.connection.commit()

    def list_sources(self, alias: str | None = None) -> list[sqlite3.Row]:
        if alias:
            rows = self.connection.execute("select * from source_accounts where alias = ? order by alias", (alias,)).fetchall()
        else:
            rows = self.connection.execute("select * from source_accounts order by alias").fetchall()
        return rows

    def set_sync_state(self, source_alias: str, mode: str, cursor: str | None, etag: str | None, status: str) -> None:
        now = utcnow_iso()
        self.connection.execute(
            """
            insert into sync_state(source_alias, mode, cursor, etag, last_synced_at, status, updated_at)
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(source_alias) do update set
              mode=excluded.mode,
              cursor=excluded.cursor,
              etag=excluded.etag,
              last_synced_at=excluded.last_synced_at,
              status=excluded.status,
              updated_at=excluded.updated_at
            """,
            (source_alias, mode, cursor, etag, now, status, now),
        )
        self.connection.commit()

    def get_sync_state(self, source_alias: str) -> dict[str, Any]:
        row = self.connection.execute("select * from sync_state where source_alias = ?", (source_alias,)).fetchone()
        return dict(row) if row else {}

    def insert_raw_documents(self, documents: Iterable[RawDocument]) -> int:
        inserted = 0
        for document in documents:
            fingerprint = self._fingerprint(document.payload)
            cursor = self.connection.execute(
                """
                insert or ignore into raw_documents(source_alias, source_type, external_id, mode, payload_json, fingerprint, imported_at)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.source_alias,
                    document.source_type,
                    document.external_id,
                    document.mode,
                    json.dumps(document.payload, sort_keys=True),
                    fingerprint,
                    utcnow_iso(),
                ),
            )
            inserted += cursor.rowcount
        self.connection.commit()
        return inserted

    def upsert_artifacts(self, artifacts: Iterable[ArtifactRecord]) -> list[int]:
        artifact_ids: list[int] = []
        for artifact in artifacts:
            now = utcnow_iso()
            fingerprint = self._fingerprint(
                {
                    "title": artifact.title,
                    "body_text": artifact.body_text,
                    "author": artifact.author,
                    "author_email": artifact.author_email,
                    "occurred_at": artifact.occurred_at,
                    "canonical_url": artifact.canonical_url,
                    "metadata": artifact.metadata,
                }
            )
            self.connection.execute(
                """
                insert into artifacts(
                  source_alias, source_type, artifact_type, external_id, title, body_text, author, author_email,
                  occurred_at, canonical_url, fingerprint, metadata_json, created_at, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(source_alias, external_id, artifact_type) do update set
                  title=excluded.title,
                  body_text=excluded.body_text,
                  author=excluded.author,
                  author_email=excluded.author_email,
                  occurred_at=excluded.occurred_at,
                  canonical_url=excluded.canonical_url,
                  fingerprint=excluded.fingerprint,
                  metadata_json=excluded.metadata_json,
                  updated_at=excluded.updated_at
                """,
                (
                    artifact.source_alias,
                    artifact.source_type,
                    artifact.artifact_type,
                    artifact.external_id,
                    artifact.title,
                    artifact.body_text,
                    artifact.author,
                    artifact.author_email,
                    artifact.occurred_at,
                    artifact.canonical_url,
                    fingerprint,
                    json.dumps(artifact.metadata, sort_keys=True),
                    now,
                    now,
                ),
            )
            row = self.connection.execute(
                "select id from artifacts where source_alias = ? and external_id = ? and artifact_type = ?",
                (artifact.source_alias, artifact.external_id, artifact.artifact_type),
            ).fetchone()
            if row:
                artifact_ids.append(int(row["id"]))
                self.connection.execute("delete from artifact_search where rowid = ?", (int(row["id"]),))
                self.connection.execute(
                    "insert into artifact_search(rowid, title, body_text) values (?, ?, ?)",
                    (int(row["id"]), artifact.title, artifact.body_text),
                )
        self.connection.commit()
        return artifact_ids

    def clear_graph(self) -> None:
        for table in ("entities", "tasks", "task_memberships", "edges"):
            self.connection.execute(f"delete from {table}")
        self.connection.commit()

    def set_competencies(self, competencies: list[dict[str, Any]]) -> None:
        self.connection.execute("delete from competencies")
        now = utcnow_iso()
        for competency in competencies:
            self.connection.execute(
                """
                insert into competencies(competency_id, title, description, keywords_json, created_at, updated_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    competency["id"],
                    competency["title"],
                    competency.get("description", ""),
                    json.dumps(competency.get("keywords", [])),
                    now,
                    now,
                ),
            )
        self.connection.commit()

    def insert_entity(self, entity_type: str, value: str, display_name: str, metadata: dict[str, Any] | None = None) -> int:
        now = utcnow_iso()
        self.connection.execute(
            """
            insert into entities(entity_type, value, display_name, metadata_json, created_at, updated_at)
            values (?, ?, ?, ?, ?, ?)
            on conflict(entity_type, value) do update set
              display_name=excluded.display_name,
              metadata_json=excluded.metadata_json,
              updated_at=excluded.updated_at
            """,
            (entity_type, value, display_name, json.dumps(metadata or {}, sort_keys=True), now, now),
        )
        row = self.connection.execute("select id from entities where entity_type = ? and value = ?", (entity_type, value)).fetchone()
        self.connection.commit()
        return int(row["id"])

    def insert_task(
        self,
        task_key: str,
        title: str,
        description: str,
        summary: str,
        implementation_summary: str,
        impact_summary: str,
        collaboration_summary: str,
        complexity_score: float,
        complexity_reasoning: str,
        status: str,
        source_anchor: str | None,
        confidence: float,
        start_at: str | None,
        end_at: str | None,
        primary_repo: str | None,
        repo_names: list[str],
        metadata: dict[str, Any],
    ) -> int:
        now = utcnow_iso()
        repo_names = sorted({repo_name for repo_name in repo_names if repo_name})
        repo_count = len(repo_names)
        is_cross_repo = int(repo_count > 1)
        self.connection.execute(
            """
            insert into tasks(
              task_key, title, description, summary, implementation_summary, impact_summary, collaboration_summary,
              complexity_score, complexity_reasoning, status, source_anchor, confidence, start_at, end_at, primary_repo,
              repo_count, is_cross_repo, repo_names_json, metadata_json, created_at, updated_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(task_key) do update set
              title=excluded.title,
              description=excluded.description,
              summary=excluded.summary,
              implementation_summary=excluded.implementation_summary,
              impact_summary=excluded.impact_summary,
              collaboration_summary=excluded.collaboration_summary,
              complexity_score=excluded.complexity_score,
              complexity_reasoning=excluded.complexity_reasoning,
              status=excluded.status,
              source_anchor=excluded.source_anchor,
              confidence=excluded.confidence,
              start_at=excluded.start_at,
              end_at=excluded.end_at,
              primary_repo=excluded.primary_repo,
              repo_count=excluded.repo_count,
              is_cross_repo=excluded.is_cross_repo,
              repo_names_json=excluded.repo_names_json,
              metadata_json=excluded.metadata_json,
              updated_at=excluded.updated_at
            """,
            (
                task_key,
                title,
                description,
                summary,
                implementation_summary,
                impact_summary,
                collaboration_summary,
                complexity_score,
                complexity_reasoning,
                status,
                source_anchor,
                confidence,
                start_at,
                end_at,
                primary_repo,
                repo_count,
                is_cross_repo,
                json.dumps(repo_names),
                json.dumps(metadata, sort_keys=True),
                now,
                now,
            ),
        )
        row = self.connection.execute("select id from tasks where task_key = ?", (task_key,)).fetchone()
        self.connection.commit()
        return int(row["id"])

    def insert_task_membership(self, task_id: int, artifact_id: int, score: float, reason: str) -> None:
        self.connection.execute(
            """
            insert or replace into task_memberships(task_id, artifact_id, membership_score, membership_reason)
            values (?, ?, ?, ?)
            """,
            (task_id, artifact_id, score, reason),
        )

    def insert_edge(
        self,
        from_kind: str,
        from_id: int,
        rel_type: str,
        to_kind: str,
        to_id: int,
        provenance_artifact_id: int | None = None,
        confidence: float = 1.0,
        valid_from: str | None = None,
        valid_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.connection.execute(
            """
            insert or ignore into edges(
              from_kind, from_id, rel_type, to_kind, to_id, provenance_artifact_id, confidence, valid_from, valid_to, metadata_json, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                from_kind,
                from_id,
                rel_type,
                to_kind,
                to_id,
                provenance_artifact_id if provenance_artifact_id is not None else -1,
                confidence,
                valid_from,
                valid_to,
                json.dumps(metadata or {}, sort_keys=True),
                utcnow_iso(),
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def fetch_artifacts(self) -> list[sqlite3.Row]:
        return self.connection.execute("select * from artifacts order by coalesce(occurred_at, created_at), id").fetchall()

    def fetch_artifacts_for_period(self, start_at: str, end_at: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            select * from artifacts
            where occurred_at is not null and occurred_at >= ? and occurred_at <= ?
            order by occurred_at, id
            """,
            (start_at, end_at),
        ).fetchall()

    def fetch_tasks_for_period(self, start_at: str, end_at: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            select distinct t.* from tasks t
            join task_memberships tm on tm.task_id = t.id
            join artifacts a on a.id = tm.artifact_id
            where a.occurred_at is not null and a.occurred_at >= ? and a.occurred_at <= ?
            order by t.confidence desc, t.title asc
            """,
            (start_at, end_at),
        ).fetchall()

    def fetch_task_memberships(self, task_id: int) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            select tm.*, a.title, a.body_text, a.artifact_type, a.occurred_at, a.source_alias,
                   json_extract(a.metadata_json, '$.repo_name') as repo_name,
                   json_extract(a.metadata_json, '$.repo') as metadata_repo
            from task_memberships tm
            join artifacts a on a.id = tm.artifact_id
            where tm.task_id = ?
            order by tm.membership_score desc, a.occurred_at asc
            """,
            (task_id,),
        ).fetchall()

    def fetch_task_repo_names(self, task_id: int) -> list[str]:
        rows = self.connection.execute(
            """
            select distinct e.display_name
            from edges rel
            join entities e on e.id = rel.to_id
            where rel.from_kind = 'task'
              and rel.from_id = ?
              and rel.rel_type = 'belongs_to'
              and rel.to_kind = 'entity'
              and e.entity_type = 'repo'
            order by e.display_name asc
            """,
            (task_id,),
        ).fetchall()
        return [str(row["display_name"]) for row in rows]

    def fetch_competencies(self) -> list[sqlite3.Row]:
        return self.connection.execute("select * from competencies order by competency_id").fetchall()

    def create_review_run(self, period_label: str, rubric_path: str, model_provider: str, self_review_markdown: str, evidence_appendix_markdown: str, gaps_markdown: str, report_html: str, claims: list[ClaimRecord]) -> int:
        now = utcnow_iso()
        self.connection.execute(
            """
            insert into review_runs(period_label, rubric_path, model_provider, self_review_markdown, evidence_appendix_markdown, gaps_markdown, report_html, created_at)
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (period_label, rubric_path, model_provider, self_review_markdown, evidence_appendix_markdown, gaps_markdown, report_html, now),
        )
        row = self.connection.execute("select last_insert_rowid() as id").fetchone()
        review_run_id = int(row["id"])
        for claim in claims:
            self.connection.execute(
                """
                insert into claim_citations(review_run_id, section_id, section_title, claim_text, artifact_ids_json, task_ids_json, created_at)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_run_id,
                    claim.section_id,
                    claim.section_title,
                    claim.claim_text,
                    json.dumps(claim.artifact_ids),
                    json.dumps(claim.task_ids),
                    now,
                ),
            )
        self.connection.commit()
        return review_run_id

    def fetch_task(self, task_id: int) -> sqlite3.Row | None:
        return self.connection.execute("select * from tasks where id = ?", (task_id,)).fetchone()

    @staticmethod
    def _fingerprint(payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
