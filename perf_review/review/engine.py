from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from perf_review.llm.provider import BaseLLMProvider
from perf_review.models import ClaimRecord, ReviewRunArtifacts
from perf_review.store.db import Database
from perf_review.utils.text import summarize_text
from perf_review.utils.time import TimeWindow


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
    sections: dict[str, list[str]] = {section_id: [] for section_id in section_order}
    claims: list[ClaimRecord] = []
    weak_tasks: list[str] = []

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
            }
            for membership in memberships
        ]
        if task["confidence"] < 0.7:
            weak_tasks.append(task["title"])
        summary = task.get("summary") or llm_provider.summarize_task(task["title"], evidence)
        implementation_summary = task.get("implementation_summary") or summary
        impact_summary = task.get("impact_summary") or ""
        collaboration_summary = task.get("collaboration_summary") or ""
        classifier_text = " ".join(
            part for part in [
                summary,
                implementation_summary,
                impact_summary,
                collaboration_summary,
                " ".join(item["body_text"] for item in evidence if item.get("body_text")),
            ] if part
        )
        assigned = llm_provider.classify_competencies(classifier_text, competencies)
        if not assigned:
            assigned = [section_order[0]] if section_order else ["uncategorized"]
        citation = _citation_suffix(task["id"], [item["artifact_id"] for item in evidence[:5]])
        claim_core = impact_summary or implementation_summary or summary
        claim_text = claim_core.rstrip(".") + citation
        for section_id in assigned[:2]:
            sections.setdefault(section_id, []).append(claim_text)
            claims.append(
                ClaimRecord(
                    section_id=section_id,
                    section_title=section_titles.get(section_id, section_id.title()),
                    claim_text=claim_text,
                    artifact_ids=[item["artifact_id"] for item in evidence[:5]],
                    task_ids=[task["id"]],
                )
            )
            competency_entity = database.insert_entity("competency", section_id, section_titles.get(section_id, section_id.title()), {})
            database.insert_edge("task", task["id"], "maps_to", "entity", competency_entity, confidence=0.8, metadata={"source": "review_generation"})
    database.commit()

    missing = [section_titles.get(section_id, section_id.title()) for section_id in section_order if not sections.get(section_id)]
    intro = f"# Self Review ({period.label})\n\n"
    intro += "This draft is grounded in locally collected evidence and every bullet includes citations.\n\n"
    body_parts: list[str] = [intro]
    for section_id in section_order:
        title = section_titles.get(section_id, section_id.title())
        summaries = sections.get(section_id, [])
        rendered = llm_provider.draft_section(title, summaries) if summaries else "- Evidence was sparse for this competency during the selected period."
        body_parts.append(f"## {title}\n\n{rendered}\n")
    self_review_markdown = "\n".join(body_parts).strip() + "\n"

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
