from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import request


@dataclass(slots=True)
class LLMResult:
    text: str
    provider_name: str


class BaseLLMProvider:
    provider_name = "deterministic"

    def available(self) -> bool:
        return True

    def summarize_task(self, task_title: str, evidence: list[dict[str, Any]]) -> str:
        snippets = []
        for item in evidence[:3]:
            title = item.get("title") or "Untitled artifact"
            body = (item.get("body_text") or "").strip()
            detail = f"{title} ({body[:80]})" if body else title
            snippets.append(detail)
        joined = "; ".join(snippets)
        return f"{task_title}: supported by {joined}" if joined else task_title

    def classify_competencies(self, task_summary: str, competencies: list[dict[str, Any]]) -> list[str]:
        lowered = task_summary.lower()
        scored: list[tuple[int, str]] = []
        for competency in competencies:
            score = sum(1 for keyword in competency.get("keywords", []) if keyword.lower() in lowered)
            if score:
                scored.append((score, competency["id"]))
        scored.sort(reverse=True)
        ids = [competency_id for _, competency_id in scored[:2]]
        if ids:
            return ids
        return [competencies[0]["id"]] if competencies else []

    def draft_section(self, section_title: str, summaries: list[str]) -> str:
        return "\n".join(f"- {summary}" for summary in summaries if summary)

    def draft_gaps(self, missing: list[str], weak: list[str]) -> str:
        lines: list[str] = []
        if missing:
            lines.append(f"- Sparse competencies: {', '.join(missing)}")
        if weak:
            lines.append(f"- Low-confidence tasks: {', '.join(weak)}")
        if not lines:
            lines.append("- Evidence coverage is balanced across configured competencies.")
        return "\n".join(lines)

    def draft_portfolio_intro(
        self,
        subject_name: str,
        generated_on: str,
        total_projects: int,
        task_titles: list[str],
        template_instructions: str = "",
    ) -> str:
        if task_titles:
            highlighted = ", ".join(task_titles[:3])
            return (
                f"Self assessment for {subject_name} - {generated_on}\n"
                f"This review summarizes {total_projects} projects, with a focus on {highlighted}."
            )
        return f"Self assessment for {subject_name} - {generated_on}"

    def draft_project_entry(self, project: dict[str, Any], template_instructions: str = "") -> dict[str, Any]:
        summary = project.get("summary") or project.get("description") or project["title"]
        implementation_summary = project.get("implementation_summary") or project.get("description") or summary
        impact_summary = project.get("impact_summary") or "Impact is inferred from the available implementation evidence."
        collaboration_summary = project.get("collaboration_summary") or "Collaboration signals were limited in the attached evidence."
        challenge_summary = project.get("challenge_summary") or "Challenges were inferred from implementation complexity and supporting evidence."
        doc_titles = [str(item) for item in project.get("design_docs", [])]
        code_titles = [str(item) for item in project.get("code_contributions", [])]
        support_titles = [str(item) for item in project.get("evidence_highlights", [])]
        specific = _dedupe_preserve_order(code_titles + doc_titles + support_titles)[:5]
        code_contributions = _dedupe_preserve_order(code_titles)[:4]
        return {
            "project_summary": summary,
            "complexity_and_difficulty": project.get("complexity_reasoning") or challenge_summary,
            "my_impact": impact_summary,
            "specific_contributions": specific,
            "technical_leadership": implementation_summary,
            "design_docs": _dedupe_preserve_order(doc_titles)[:3],
            "code_contributions": code_contributions,
            "initiative_and_mentorship": collaboration_summary,
        }

    def render_project_markdown(self, project: dict[str, Any], template_instructions: str = "") -> str:
        draft = project["draft"]
        lines = [f"## {project['title']}"]
        if project.get("timeframe"):
            lines.append(f"Timeframe: {project['timeframe']}")
        if project.get("repos"):
            lines.append(f"Repos involved: {', '.join(project['repos'])}")
        lines.append("")
        lines.append(f"Project summary: {draft['project_summary']}{project['citation']}")
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
        lines.append(draft["technical_leadership"])
        lines.append("")
        lines.append("Design docs authored/reviewed:")
        for item in draft["design_docs"] or ["No standalone design docs were attached to this task."]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Code & system contributions:")
        for item in draft["code_contributions"] or ["Code and system contributions are represented in the attached evidence."]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Initiative & mentorship:")
        lines.append("")
        lines.append(draft["initiative_and_mentorship"])
        return "\n".join(lines)

    def enrich_task(self, task_context: dict[str, Any]) -> dict[str, Any]:
        title = str(task_context.get("title") or "Task")
        description = str(task_context.get("description") or "").strip()
        highlights = [str(item) for item in task_context.get("evidence_highlights", [])[:5]]
        summary = description or (f"{title}: " + "; ".join(highlights[:2]) if highlights else title)
        implementation_summary = "; ".join(highlights[:3]) if highlights else (description or title)
        impact_summary = _infer_impact_summary(task_context)
        collaboration_summary = _infer_collaboration_summary(task_context)
        challenge_summary = _infer_challenge_summary(task_context)
        complexity_score = _infer_complexity_score(task_context)
        complexity_reasoning = _infer_complexity_reasoning(task_context, complexity_score)
        return {
            "summary": summary,
            "implementation_summary": implementation_summary,
            "impact_summary": impact_summary,
            "collaboration_summary": collaboration_summary,
            "challenge_summary": challenge_summary,
            "complexity_score": complexity_score,
            "complexity_reasoning": complexity_reasoning,
            "status": str(task_context.get("status") or "inferred"),
        }

    @staticmethod
    def _artifact_summary(item: dict[str, Any]) -> str:
        title = item.get("title") or "Untitled artifact"
        body = (item.get("body_text") or "").strip()
        return f"{title} ({body[:80]})" if body else title


class OllamaProvider(BaseLLMProvider):
    provider_name = "ollama"

    def __init__(self, base_url: str, model: str, enabled: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled

    def available(self) -> bool:
        if not self.enabled:
            return False
        req = request.Request(f"{self.base_url}/api/tags")
        try:
            with request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def summarize_task(self, task_title: str, evidence: list[dict[str, Any]]) -> str:
        if not self.available():
            return super().summarize_task(task_title, evidence)
        prompt = (
            "Summarize this engineering task in one performance-review-ready bullet. "
            "Focus on action and outcome. Evidence:\n"
            f"{json.dumps(evidence[:5], indent=2)}"
        )
        try:
            return self._generate(prompt)
        except Exception:
            return super().summarize_task(task_title, evidence)

    def classify_competencies(self, task_summary: str, competencies: list[dict[str, Any]]) -> list[str]:
        if not self.available():
            return super().classify_competencies(task_summary, competencies)
        prompt = (
            "Choose up to two competency ids that best match this task summary. "
            "Return only a comma-separated list of ids.\n"
            f"Competencies: {json.dumps(competencies)}\nSummary: {task_summary}"
        )
        try:
            raw = self._generate(prompt)
            return [part.strip() for part in raw.split(",") if part.strip()]
        except Exception:
            return super().classify_competencies(task_summary, competencies)

    def draft_section(self, section_title: str, summaries: list[str]) -> str:
        if not self.available():
            return super().draft_section(section_title, summaries)
        prompt = (
            f"Write a concise performance review section for {section_title}. "
            "Return markdown bullets only.\n"
            f"{json.dumps(summaries, indent=2)}"
        )
        try:
            return self._generate(prompt)
        except Exception:
            return super().draft_section(section_title, summaries)

    def draft_gaps(self, missing: list[str], weak: list[str]) -> str:
        if not self.available():
            return super().draft_gaps(missing, weak)
        prompt = (
            "Write markdown bullets summarizing evidence gaps for a performance review. "
            f"Missing competencies: {missing}. Weak tasks: {weak}."
        )
        try:
            return self._generate(prompt)
        except Exception:
            return super().draft_gaps(missing, weak)

    def draft_portfolio_intro(
        self,
        subject_name: str,
        generated_on: str,
        total_projects: int,
        task_titles: list[str],
        template_instructions: str = "",
    ) -> str:
        if not self.available():
            return super().draft_portfolio_intro(subject_name, generated_on, total_projects, task_titles, template_instructions)
        prompt = (
            "Write a concise 2-3 sentence header for a software engineer self assessment. "
            "Output plain text only. No preamble, no commentary, no 'here is', and no closing sentence. "
            "Keep the exact first line format 'Self assessment for <name> - <date>'. "
            "Then add one short summary sentence about the portfolio.\n"
            f"Template instructions:\n{template_instructions or 'Use the default project-review structure.'}\n"
            f"Name: {subject_name}\n"
            f"Date: {generated_on}\n"
            f"Total projects: {total_projects}\n"
            f"Project titles: {json.dumps(task_titles[:8])}"
        )
        try:
            return _clean_intro(self._generate(prompt), subject_name, generated_on, total_projects, task_titles)
        except Exception:
            return super().draft_portfolio_intro(subject_name, generated_on, total_projects, task_titles, template_instructions)

    def draft_project_entry(self, project: dict[str, Any], template_instructions: str = "") -> dict[str, Any]:
        if not self.available():
            return super().draft_project_entry(project, template_instructions)
        prompt = (
            "You are writing one project entry for a performance review self assessment. "
            "Return strict JSON with keys project_summary, complexity_and_difficulty, my_impact, "
            "specific_contributions, technical_leadership, design_docs, code_contributions, initiative_and_mentorship. "
            "specific_contributions, design_docs, and code_contributions must be arrays of short strings. "
            "Focus on concrete work and outcomes. Use the provided task rollup as the primary source of truth. "
            "Do not include chatty framing.\n"
            f"Template instructions:\n{template_instructions or 'Use the default project-review structure.'}\n"
            f"Project: {json.dumps(project, indent=2)}"
        )
        try:
            raw = self._generate(prompt)
            parsed = json.loads(raw)
            return {
                "project_summary": str(parsed.get("project_summary", project["title"])),
                "complexity_and_difficulty": str(parsed.get("complexity_and_difficulty", project.get("complexity_reasoning", ""))),
                "my_impact": str(parsed.get("my_impact", project.get("impact_summary", ""))),
                "specific_contributions": [str(item) for item in parsed.get("specific_contributions", [])][:6],
                "technical_leadership": str(parsed.get("technical_leadership", project.get("implementation_summary", ""))),
                "design_docs": [str(item) for item in parsed.get("design_docs", [])][:4],
                "code_contributions": [str(item) for item in parsed.get("code_contributions", [])][:5],
                "initiative_and_mentorship": str(parsed.get("initiative_and_mentorship", project.get("collaboration_summary", ""))),
            }
        except Exception:
            return super().draft_project_entry(project, template_instructions)

    def render_project_markdown(self, project: dict[str, Any], template_instructions: str = "") -> str:
        if not self.available() or not template_instructions.strip():
            return super().render_project_markdown(project, template_instructions)
        prompt = (
            "Write one project section in markdown for a performance review. "
            "Follow the user's template instructions as closely as possible. "
            "Use the provided draft fields as the source of truth. "
            "Preserve the exact citation string verbatim somewhere in the project summary line. "
            "Do not add commentary outside the project section.\n"
            f"Template instructions:\n{template_instructions}\n"
            f"Project payload: {json.dumps(project, indent=2)}"
        )
        try:
            rendered = self._generate(prompt).strip()
            if rendered:
                return rendered
        except Exception:
            pass
        return super().render_project_markdown(project, template_instructions)

    def enrich_task(self, task_context: dict[str, Any]) -> dict[str, Any]:
        if not self.available():
            return super().enrich_task(task_context)
        prompt = (
            "You are enriching a software engineering task for performance review preparation. "
            "Return strict JSON with keys summary, implementation_summary, impact_summary, "
            "collaboration_summary, challenge_summary, complexity_score, complexity_reasoning, status. "
            "complexity_score must be a number from 0.0 to 1.0.\n"
            "Use the compact task context below rather than reconstructing the task from raw evidence.\n"
            f"Task context: {json.dumps(task_context, indent=2)}"
        )
        try:
            raw = self._generate(prompt)
            parsed = json.loads(raw)
            return {
                "summary": str(parsed.get("summary", task_context.get("title", "Task"))),
                "implementation_summary": str(parsed.get("implementation_summary", task_context.get("title", "Task"))),
                "impact_summary": str(parsed.get("impact_summary", "")),
                "collaboration_summary": str(parsed.get("collaboration_summary", "")),
                "challenge_summary": str(parsed.get("challenge_summary", "")),
                "complexity_score": max(0.0, min(1.0, float(parsed.get("complexity_score", 0.5)))),
                "complexity_reasoning": str(parsed.get("complexity_reasoning", "")),
                "status": str(parsed.get("status", task_context.get("status", "inferred"))),
            }
        except Exception:
            return super().enrich_task(task_context)

    def _generate(self, prompt: str) -> str:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
            return str(data.get("response", "")).strip()


def build_llm_provider(config: dict[str, Any]) -> BaseLLMProvider:
    model_config = (config.get("app") or {}).get("model", {})
    if model_config.get("provider") == "ollama":
        return OllamaProvider(
            base_url=model_config.get("base_url", "http://localhost:11434"),
            model=model_config.get("model", "llama3.2:latest"),
            enabled=bool(model_config.get("enabled", True)),
        )
    return BaseLLMProvider()


def _infer_impact_summary(task_context: dict[str, Any]) -> str:
    highlights = " ".join(str(item) for item in task_context.get("evidence_highlights", [])[:5]).lower()
    if not highlights:
        return ""
    for keyword in ("launch", "customer", "support", "ship", "release", "user", "beta", "publish", "auth"):
        if keyword in highlights:
            return f"Evidence suggests user-facing or delivery impact related to {keyword}."
    return "Impact is inferred from implementation progress and supporting documentation."


def _infer_collaboration_summary(task_context: dict[str, Any]) -> str:
    text = " ".join(str(item) for item in task_context.get("evidence_highlights", [])[:6]).lower()
    people = task_context.get("people", [])
    if len(people) >= 2:
        return "Evidence includes collaboration across multiple contributors or stakeholders."
    for keyword in ("partner", "team", "review", "shared", "stakeholder", "collaborat"):
        if keyword in text:
            return "Evidence includes cross-team or collaborative signals."
    return "Collaboration signals are limited in the currently attached evidence."


def _infer_challenge_summary(task_context: dict[str, Any]) -> str:
    challenge_hints = [str(item) for item in task_context.get("challenge_hints", []) if item]
    if challenge_hints:
        return f"Key challenges included {challenge_hints[0].lower()}."
    if task_context.get("story_points"):
        return f"The task carried an estimated effort of {task_context['story_points']} story points and required coordination across multiple systems."
    return "Challenges are inferred from the attached implementation and coordination evidence."


def _infer_complexity_score(task_context: dict[str, Any]) -> float:
    issue_types = set(str(item) for item in task_context.get("issue_types", []) if item)
    unique_repos = set(str(item) for item in task_context.get("repos", []) if item)
    artifact_count = int(task_context.get("artifact_count") or 0)
    score = 0.25 + min(0.4, 0.08 * max(0, artifact_count - 1))
    score += min(0.2, 0.08 * max(0, len(issue_types) - 1))
    score += min(0.15, 0.08 * max(0, len(unique_repos) - 1))
    max_story_points = float(task_context.get("story_points") or 0.0)
    score += min(0.15, max_story_points / 40.0)
    if str(task_context.get("source_anchor") or "").startswith("anchor:issue:"):
        score += 0.1
    return round(min(0.95, score), 2)


def _infer_complexity_reasoning(task_context: dict[str, Any], complexity_score: float) -> str:
    issue_types = sorted({str(item) for item in task_context.get("issue_types", []) if item})
    unique_repos = sorted({str(item) for item in task_context.get("repos", []) if item})
    max_story_points = float(task_context.get("story_points") or 0.0)
    anchor_text = str(task_context.get("source_anchor") or "semantic clustering")
    return (
        f"Complexity score {complexity_score:.2f} derived from {task_context.get('artifact_count', 0)} evidence items, "
        f"issue types {', '.join(issue_types) or 'unknown'}, repos {', '.join(unique_repos) or 'none'}, "
        f"max story points {max_story_points:.0f}, anchored by {anchor_text}."
    )


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _clean_intro(raw: str, subject_name: str, generated_on: str, total_projects: int, task_titles: list[str]) -> str:
    lines = [line.rstrip() for line in raw.splitlines()]
    start_index = 0
    for index, line in enumerate(lines):
        if line.strip().startswith("Self assessment for "):
            start_index = index
            break
    cleaned = "\n".join(
        line
        for line in lines[start_index:]
        if line.strip() and "let me know" not in line.lower() and "here is" not in line.lower()
    ).strip()
    if cleaned.startswith("Self assessment for "):
        return cleaned
    return BaseLLMProvider().draft_portfolio_intro(subject_name, generated_on, total_projects, task_titles, "")
