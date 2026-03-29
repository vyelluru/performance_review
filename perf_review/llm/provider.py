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

    def enrich_task(self, task_title: str, evidence: list[dict[str, Any]], source_anchor: str | None = None) -> dict[str, Any]:
        summaries = [self._artifact_summary(item) for item in evidence[:5]]
        summary = f"{task_title}: " + "; ".join(summaries[:2]) if summaries else task_title
        implementation_summary = "; ".join(summaries[:3]) if summaries else task_title
        impact_summary = _infer_impact_summary(evidence)
        collaboration_summary = _infer_collaboration_summary(evidence)
        complexity_score = _infer_complexity_score(evidence, source_anchor)
        complexity_reasoning = _infer_complexity_reasoning(evidence, source_anchor, complexity_score)
        return {
            "summary": summary,
            "implementation_summary": implementation_summary,
            "impact_summary": impact_summary,
            "collaboration_summary": collaboration_summary,
            "complexity_score": complexity_score,
            "complexity_reasoning": complexity_reasoning,
            "status": "inferred",
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
        return self._generate(prompt)

    def classify_competencies(self, task_summary: str, competencies: list[dict[str, Any]]) -> list[str]:
        if not self.available():
            return super().classify_competencies(task_summary, competencies)
        prompt = (
            "Choose up to two competency ids that best match this task summary. "
            "Return only a comma-separated list of ids.\n"
            f"Competencies: {json.dumps(competencies)}\nSummary: {task_summary}"
        )
        raw = self._generate(prompt)
        return [part.strip() for part in raw.split(",") if part.strip()]

    def draft_section(self, section_title: str, summaries: list[str]) -> str:
        if not self.available():
            return super().draft_section(section_title, summaries)
        prompt = (
            f"Write a concise performance review section for {section_title}. "
            "Return markdown bullets only.\n"
            f"{json.dumps(summaries, indent=2)}"
        )
        return self._generate(prompt)

    def draft_gaps(self, missing: list[str], weak: list[str]) -> str:
        if not self.available():
            return super().draft_gaps(missing, weak)
        prompt = (
            "Write markdown bullets summarizing evidence gaps for a performance review. "
            f"Missing competencies: {missing}. Weak tasks: {weak}."
        )
        return self._generate(prompt)

    def enrich_task(self, task_title: str, evidence: list[dict[str, Any]], source_anchor: str | None = None) -> dict[str, Any]:
        if not self.available():
            return super().enrich_task(task_title, evidence, source_anchor)
        prompt = (
            "You are enriching a software engineering task for performance review preparation. "
            "Return strict JSON with keys summary, implementation_summary, impact_summary, "
            "collaboration_summary, complexity_score, complexity_reasoning, status. "
            "complexity_score must be a number from 0.0 to 1.0.\n"
            f"Source anchor: {source_anchor}\n"
            f"Task title: {task_title}\n"
            f"Evidence: {json.dumps(evidence[:8], indent=2)}"
        )
        raw = self._generate(prompt)
        try:
            parsed = json.loads(raw)
            return {
                "summary": str(parsed.get("summary", task_title)),
                "implementation_summary": str(parsed.get("implementation_summary", task_title)),
                "impact_summary": str(parsed.get("impact_summary", "")),
                "collaboration_summary": str(parsed.get("collaboration_summary", "")),
                "complexity_score": max(0.0, min(1.0, float(parsed.get("complexity_score", 0.5)))),
                "complexity_reasoning": str(parsed.get("complexity_reasoning", "")),
                "status": str(parsed.get("status", "inferred")),
            }
        except (ValueError, TypeError, json.JSONDecodeError):
            return super().enrich_task(task_title, evidence, source_anchor)

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


def _infer_impact_summary(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return ""
    text = " ".join((item.get("body_text") or item.get("title") or "") for item in evidence[:5]).lower()
    for keyword in ("launch", "customer", "support", "ship", "release", "user", "beta"):
        if keyword in text:
            return f"Evidence suggests user-facing or delivery impact related to {keyword}."
    return "Impact is inferred from implementation progress and supporting documentation."


def _infer_collaboration_summary(evidence: list[dict[str, Any]]) -> str:
    text = " ".join((item.get("body_text") or item.get("title") or "") for item in evidence[:6]).lower()
    for keyword in ("partner", "team", "review", "shared", "stakeholder", "collaborat"):
        if keyword in text:
            return "Evidence includes cross-team or collaborative signals."
    return "Collaboration signals are limited in the currently attached evidence."


def _infer_complexity_score(evidence: list[dict[str, Any]], source_anchor: str | None) -> float:
    unique_types = {item.get("artifact_type") for item in evidence}
    unique_repos = {item.get("repo_name") for item in evidence if item.get("repo_name")}
    score = 0.25 + min(0.4, 0.08 * max(0, len(evidence) - 1))
    score += min(0.2, 0.08 * max(0, len(unique_types) - 1))
    score += min(0.15, 0.08 * max(0, len(unique_repos) - 1))
    max_story_points = max(
        (
            float(item.get("story_points"))
            for item in evidence
            if item.get("story_points") not in (None, "")
        ),
        default=0.0,
    )
    score += min(0.15, max_story_points / 40.0)
    if source_anchor and source_anchor.startswith("anchor:issue:"):
        score += 0.1
    return round(min(0.95, score), 2)


def _infer_complexity_reasoning(evidence: list[dict[str, Any]], source_anchor: str | None, complexity_score: float) -> str:
    unique_types = sorted({item.get("artifact_type") for item in evidence if item.get("artifact_type")})
    unique_repos = sorted({item.get("repo_name") for item in evidence if item.get("repo_name")})
    max_story_points = max(
        (
            float(item.get("story_points"))
            for item in evidence
            if item.get("story_points") not in (None, "")
        ),
        default=0.0,
    )
    anchor_text = source_anchor or "semantic clustering"
    return (
        f"Complexity score {complexity_score:.2f} derived from {len(evidence)} evidence items, "
        f"artifact types {', '.join(unique_types) or 'unknown'}, repos {', '.join(unique_repos) or 'none'}, "
        f"max story points {max_story_points:.0f}, anchored by {anchor_text}."
    )
