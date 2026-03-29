from __future__ import annotations

import html
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from perf_review.llm.provider import BaseLLMProvider
from perf_review.models import ClaimRecord, ReviewRunArtifacts
from perf_review.store.db import Database
from perf_review.utils.text import summarize_text
from perf_review.utils.time import TimeWindow, parse_datetime


def generate_review(database: Database, llm_provider: BaseLLMProvider, rubric: dict[str, Any], period: TimeWindow, rubric_path: str) -> tuple[int, ReviewRunArtifacts]:
    task_rows = [dict(row) for row in database.fetch_tasks_for_period(period.start.isoformat(), period.end.isoformat())]
    competency_rows = [dict(row) for row in database.fetch_competencies()]
    competencies = [
        {
            "id": row["competency_id"],
            "title": row["title"],
            "description": row["description"],
            "keywords": json.loads(row["keywords_json"]),
        }
        for row in competency_rows
    ]
    section_order = rubric.get("section_order", [row["id"] for row in competencies])
    section_titles = {row["id"]: row["title"] for row in competencies}
    claims: list[ClaimRecord] = []
    weak_tasks: list[str] = []
    project_entries: list[dict[str, Any]] = []
    all_evidence: list[dict[str, Any]] = []

    for task in task_rows:
        memberships = [dict(row) for row in database.fetch_task_memberships(task["id"])]
        evidence = [
            {
                "artifact_id": membership["artifact_id"],
                "title": membership["title"],
                "body_text": summarize_text(membership["body_text"], 240),
                "artifact_type": membership["artifact_type"],
                "source_alias": membership["source_alias"],
                "membership_score": membership["membership_score"],
                "author": membership["author"],
                "repo_name": membership["repo_name"] or membership["metadata_repo"],
                **json.loads(membership["metadata_json"] or "{}"),
            }
            for membership in memberships
        ]
        all_evidence.extend(evidence)
        if task["confidence"] < 0.7:
            weak_tasks.append(task["title"])
        metadata = json.loads(task.get("metadata_json") or "{}")
        project_payload = {
            "title": task["title"],
            "description": task.get("description") or "",
            "summary": task.get("summary") or llm_provider.summarize_task(task["title"], evidence),
            "implementation_summary": task.get("implementation_summary") or task.get("summary") or task["title"],
            "impact_summary": task.get("impact_summary") or "",
            "collaboration_summary": task.get("collaboration_summary") or "",
            "challenge_summary": task.get("challenge_summary") or "",
            "complexity_reasoning": task.get("complexity_reasoning") or "",
            "complexity_score": task.get("complexity_score") or 0.0,
            "status": task.get("status") or "inferred",
            "repos": database.fetch_task_repo_names(task["id"]),
            "jira_keys": json.loads(task.get("jira_keys_json") or "[]"),
            "people": json.loads(task.get("people_json") or "[]"),
            "labels": json.loads(task.get("labels_json") or "[]"),
            "issue_types": json.loads(task.get("issue_types_json") or "[]"),
            "story_points": task.get("story_points"),
            "artifact_count": task.get("artifact_count") or len(evidence),
            "timeframe": _format_timeframe(task.get("start_at"), task.get("end_at")),
            "evidence_highlights": metadata.get("evidence_highlights", [item["title"] for item in evidence[:5]]),
            "design_docs": metadata.get("design_docs", []),
            "code_contributions": metadata.get("code_contributions", [item["title"] for item in evidence if item.get("artifact_type") in {"commit", "pr", "issue"}][:6]),
            "challenge_hints": metadata.get("challenge_hints", []),
        }
        drafted_entry = llm_provider.draft_project_entry(project_payload)
        citation = _citation_suffix(task["id"], [item["artifact_id"] for item in evidence[:5]])
        claims.append(
            ClaimRecord(
                section_id=f"task:{task['id']}",
                section_title=task["title"],
                claim_text=(drafted_entry["my_impact"] or drafted_entry["project_summary"]).rstrip(".") + citation,
                artifact_ids=[item["artifact_id"] for item in evidence[:5]],
                task_ids=[task["id"]],
            )
        )
        classifier_text = " ".join(
            part
            for part in [
                drafted_entry["project_summary"],
                drafted_entry["technical_leadership"],
                drafted_entry["my_impact"],
                drafted_entry["initiative_and_mentorship"],
                " ".join(item["body_text"] for item in evidence if item.get("body_text")),
            ]
            if part
        )
        assigned = llm_provider.classify_competencies(classifier_text, competencies)
        if not assigned and section_order:
            assigned = [section_order[0]]
        for section_id in assigned[:2]:
            competency_entity = database.insert_entity("competency", section_id, section_titles.get(section_id, section_id.title()), {})
            database.insert_edge("task", task["id"], "maps_to", "entity", competency_entity, confidence=0.8, metadata={"source": "review_generation"})
        project_entries.append(
            {
                "task_id": task["id"],
                "title": task["title"],
                "timeframe": project_payload["timeframe"],
                "repos": project_payload["repos"],
                "draft": drafted_entry,
                "citation": citation,
            }
        )
    database.commit()

    missing = _missing_competencies(database, section_order, section_titles)
    self_review_markdown = _render_self_assessment(period, llm_provider, project_entries, all_evidence)

    evidence_appendix_markdown = _render_evidence_appendix(task_rows, database)
    gaps_markdown = "# Evidence Gaps\n\n" + llm_provider.draft_gaps(missing, weak_tasks) + "\n"
    report_html = _render_html_report(period.label, self_review_markdown, evidence_appendix_markdown, gaps_markdown)
    artifacts = ReviewRunArtifacts(
        self_review_markdown=self_review_markdown,
        evidence_appendix_markdown=evidence_appendix_markdown,
        gaps_markdown=gaps_markdown,
        report_html=report_html,
        claims=claims,
    )
    review_run_id = database.create_review_run(
        period_label=period.label,
        rubric_path=rubric_path,
        model_provider=llm_provider.provider_name if llm_provider.available() else "deterministic",
        self_review_markdown=self_review_markdown,
        evidence_appendix_markdown=evidence_appendix_markdown,
        gaps_markdown=gaps_markdown,
        report_html=report_html,
        claims=claims,
    )
    return review_run_id, artifacts


def explain_task(database: Database, task_id: int) -> str:
    task = database.fetch_task(task_id)
    if not task:
        return f"Task {task_id} was not found."
    memberships = database.fetch_task_memberships(task_id)
    repo_names = database.fetch_task_repo_names(task_id)
    lines = [
        f"Task {task['id']}: {task['title']}",
        f"Confidence: {task['confidence']}",
        f"Repos: {', '.join(repo_names) if repo_names else 'None detected'}",
        f"Description: {task['description']}",
        f"Jira keys: {', '.join(json.loads(task['jira_keys_json'] or '[]')) or 'None'}",
        f"People: {', '.join(json.loads(task['people_json'] or '[]')) or 'None'}",
        f"Story points: {task['story_points'] if task['story_points'] is not None else 'None'}",
        f"Challenges: {task['challenge_summary'] or 'None captured'}",
        "Evidence:",
    ]
    for membership in memberships:
        repo_name = membership["repo_name"] or membership["metadata_repo"]
        repo_suffix = f", repo={repo_name}" if repo_name else ""
        lines.append(
            f"- [{membership['artifact_id']}] {membership['artifact_type']} from {membership['source_alias']} "
            f"(score={membership['membership_score']}, reason={membership['membership_reason']}{repo_suffix}): {membership['title']}"
        )
    return "\n".join(lines)


def write_review_outputs(output_dir: str | Path, artifacts: ReviewRunArtifacts) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "self_review.md": root / "self_review.md",
        "evidence_appendix.md": root / "evidence_appendix.md",
        "gaps.md": root / "gaps.md",
        "review_report.html": root / "review_report.html",
    }
    outputs["self_review.md"].write_text(artifacts.self_review_markdown, encoding="utf-8")
    outputs["evidence_appendix.md"].write_text(artifacts.evidence_appendix_markdown, encoding="utf-8")
    outputs["gaps.md"].write_text(artifacts.gaps_markdown, encoding="utf-8")
    outputs["review_report.html"].write_text(artifacts.report_html, encoding="utf-8")
    return outputs


def _render_evidence_appendix(task_rows: list[dict[str, Any]], database: Database) -> str:
    lines = ["# Evidence Appendix", ""]
    for task in task_rows:
        repo_names = database.fetch_task_repo_names(task["id"])
        lines.append(f"## Task {task['id']}: {task['title']}")
        if repo_names:
            lines.append(f"- Repos involved: {', '.join(repo_names)}")
        memberships = database.fetch_task_memberships(task["id"])
        for membership in memberships:
            repo_name = membership["repo_name"] or membership["metadata_repo"]
            repo_suffix = f", repo={repo_name}" if repo_name else ""
            lines.append(
                f"- Artifact {membership['artifact_id']} ({membership['artifact_type']} via {membership['source_alias']}): "
                f"{membership['title']} [score={membership['membership_score']}, reason={membership['membership_reason']}{repo_suffix}]"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _citation_suffix(task_id: int, artifact_ids: list[int]) -> str:
    joined = ", ".join(str(artifact_id) for artifact_id in artifact_ids)
    return f" [task:{task_id}; artifacts:{joined}]"


def _render_self_assessment(period: TimeWindow, llm_provider: BaseLLMProvider, project_entries: list[dict[str, Any]], all_evidence: list[dict[str, Any]]) -> str:
    generated_on = datetime.now(UTC).strftime("%Y-%b-%d")
    subject_name = _infer_subject_name(all_evidence)
    intro = llm_provider.draft_portfolio_intro(subject_name, generated_on, len(project_entries), [entry["title"] for entry in project_entries])
    lines = [f"# {intro}", ""]
    for entry in project_entries:
        draft = entry["draft"]
        lines.append(f"## {entry['title']}")
        if entry["timeframe"]:
            lines.append(f"Timeframe: {entry['timeframe']}")
        if entry["repos"]:
            lines.append(f"Repos involved: {', '.join(entry['repos'])}")
        lines.append("")
        lines.append(f"Project summary: {draft['project_summary']}{entry['citation']}")
        lines.append("")
        lines.append(f"Complexity & difficulty: {draft['complexity_and_difficulty']}")
        lines.append("")
        lines.append(f"My impact: {draft['my_impact']}")
        lines.append("")
        lines.append("My specific contributions:")
        for item in draft["specific_contributions"] or ["Contribution details are inferred from the attached evidence."]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Technical leadership and code contributions:")
        lines.append("")
        lines.append(f"{draft['technical_leadership']}")
        lines.append("")
        lines.append("Design docs authored/reviewed:")
        design_docs = draft["design_docs"] or ["No standalone design docs were attached to this task."]
        for item in design_docs:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Code & system contributions:")
        code_items = draft["code_contributions"] or ["Code and system contributions are represented in the attached evidence."]
        for item in code_items:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Initiative & mentorship:")
        lines.append("")
        lines.append(f"{draft['initiative_and_mentorship']}")
        lines.append("")
    lines.append(f"Total number of projects combined: {len(project_entries)}")
    return "\n".join(lines).strip() + "\n"


def _infer_subject_name(evidence: list[dict[str, Any]]) -> str:
    candidates: list[str] = []
    for item in evidence:
        for key in ("assignee", "author", "reporter"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    if not candidates:
        return "Engineer"
    return Counter(candidates).most_common(1)[0][0]


def _format_timeframe(start_at: str | None, end_at: str | None) -> str:
    start = parse_datetime(start_at)
    end = parse_datetime(end_at)
    if not start and not end:
        return ""
    if start and end:
        return f"{start.strftime('%B %Y')} – {end.strftime('%B %Y')}"
    value = start or end
    if value is None:
        return ""
    return value.strftime("%B %Y")


def _missing_competencies(database: Database, section_order: list[str], section_titles: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for section_id in section_order:
        row = database.connection.execute(
            """
            select 1
            from edges rel
            join entities e on e.id = rel.to_id
            where rel.from_kind = 'task'
              and rel.rel_type = 'maps_to'
              and e.entity_type = 'competency'
              and e.value = ?
            limit 1
            """,
            (section_id,),
        ).fetchone()
        if row is None:
            missing.append(section_titles.get(section_id, section_id.title()))
    return missing


def _render_html_report(period_label: str, self_review_markdown: str, evidence_appendix_markdown: str, gaps_markdown: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Performance Review Report - {html.escape(period_label)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe6;
      --ink: #1f2933;
      --muted: #52606d;
      --panel: #fffdf9;
      --accent: #b85c38;
      --border: #e5d8c2;
    }}
    body {{ font-family: Georgia, 'Times New Roman', serif; background: linear-gradient(180deg, #f4efe6 0%, #efe2c8 100%); color: var(--ink); margin: 0; padding: 2rem; }}
    main {{ max-width: 960px; margin: 0 auto; background: var(--panel); border: 1px solid var(--border); box-shadow: 0 12px 32px rgba(31, 41, 51, 0.08); padding: 2rem; }}
    h1, h2 {{ color: var(--accent); }}
    section {{ margin-bottom: 2rem; }}
    pre {{ white-space: pre-wrap; background: #fff7eb; border: 1px solid var(--border); padding: 1rem; border-radius: 12px; }}
    .meta {{ color: var(--muted); margin-bottom: 2rem; }}
  </style>
</head>
<body>
  <main>
    <h1>Performance Review Report</h1>
    <p class="meta">Generated for period {html.escape(period_label)}</p>
    <section>
      <h2>Self Review</h2>
      <pre>{html.escape(self_review_markdown)}</pre>
    </section>
    <section>
      <h2>Evidence Appendix</h2>
      <pre>{html.escape(evidence_appendix_markdown)}</pre>
    </section>
    <section>
      <h2>Gaps</h2>
      <pre>{html.escape(gaps_markdown)}</pre>
    </section>
  </main>
</body>
</html>
"""
