from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from perf_review.utils.datafile import dump_structured_file, load_structured_file


DEFAULT_CONFIG = {
  "app": {
    "db_path": ".perf_review/perf_review.db",
    "output_dir": ".perf_review/output",
    "review_template_path": "review_template.md",
    "tasking": {
      "target_task_count": 4,
      "consolidation_threshold": 0.82
    },
    "model": {
      "provider": "ollama",
      "base_url": "http://localhost:11434",
      "model": "llama3.2:latest",
      "enabled": True
    },
    "redactions": []
  },
  "sources": []
}


DEFAULT_REVIEW_TEMPLATE = """# Performance Review Template

Use this file to describe how the final self-review should be structured.
These are instructions for the model, not a strict schema.

Preferred voice:
- First person
- Concise and confident
- Outcome-focused

Global instructions:
- Prefer polished prose over raw evidence snippets
- Do not start summaries with "Comment 1", raw ticket formatting, branch names, or commit hashes
- Synthesize the work into project language and write like a performance review, not an evidence dump
- Mention concrete outcomes, technical depth, and collaboration when supported
- Keep citations intact when provided
- Avoid repeating the same evidence in multiple sections
- If the source text is low quality, rewrite it into clean professional prose
- Use Jira descriptions/comments and notes as background context, not as literal first sentences
- Keep each project section readable and manager-friendly

Per-project structure:
1. Project heading
2. Timeframe
3. Repos involved if relevant
4. Project summary: one polished paragraph that explains the work in plain language
5. Complexity & difficulty: one concise explanation of technical or coordination difficulty
6. My impact: one concise explanation of the result or value
7. My specific contributions: short bullet list
8. Technical leadership and code contributions
9. Design docs authored/reviewed
10. Code & system contributions
11. Initiative & mentorship
"""


DEFAULT_RUBRIC = {
  "competencies": [
    {
      "id": "impact",
      "title": "Impact",
      "description": "Deliver meaningful outcomes with visible customer or business value.",
      "keywords": ["launch", "customer", "metric", "impact", "improved", "delivered"]
    },
    {
      "id": "execution",
      "title": "Execution",
      "description": "Drive projects to completion with strong ownership and follow-through.",
      "keywords": ["implemented", "fixed", "shipped", "closed", "resolved", "completed"]
    },
    {
      "id": "collaboration",
      "title": "Collaboration",
      "description": "Work effectively across teams through communication and partnership.",
      "keywords": ["partnered", "collaborated", "cross-team", "review", "shared", "aligned"]
    },
    {
      "id": "leadership",
      "title": "Leadership",
      "description": "Shape direction, quality, and team effectiveness beyond individual execution.",
      "keywords": ["mentored", "led", "designed", "strategy", "proposed", "unblocked"]
    }
  ],
  "section_order": ["impact", "execution", "collaboration", "leadership"],
  "prompts": {
    "self_review_intro": "Write a concise self-review grounded in the cited evidence.",
    "gap_summary": "Identify sparse competencies and unsupported claims."
  }
}


class ConfigManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.config_path = self.root / "perf.yaml"
        self.rubric_path = self.root / "rubric.yml"
        self.review_template_path = self.root / "review_template.md"

    def ensure_project(self) -> tuple[Path, Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        app_dir = self.root / ".perf_review"
        app_dir.mkdir(exist_ok=True)
        if not self.config_path.exists():
            dump_structured_file(self.config_path, copy.deepcopy(DEFAULT_CONFIG))
        if not self.rubric_path.exists():
            dump_structured_file(self.rubric_path, copy.deepcopy(DEFAULT_RUBRIC))
        if not self.review_template_path.exists():
            self.review_template_path.write_text(DEFAULT_REVIEW_TEMPLATE, encoding="utf-8")
        output_dir = self.root / ".perf_review" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return self.config_path, self.rubric_path

    def load_config(self) -> dict[str, Any]:
        data = load_structured_file(self.config_path, copy.deepcopy(DEFAULT_CONFIG))
        return self._merge_defaults(DEFAULT_CONFIG, data or {})

    def save_config(self, config: dict[str, Any]) -> None:
        dump_structured_file(self.config_path, config)

    def load_rubric(self, path: str | None = None) -> dict[str, Any]:
        rubric_path = Path(path) if path else self.rubric_path
        data = load_structured_file(rubric_path, copy.deepcopy(DEFAULT_RUBRIC))
        return self._merge_defaults(DEFAULT_RUBRIC, data or {})

    def load_review_template(self, path: str | None = None, config: dict[str, Any] | None = None) -> str:
        configured_path = None
        if path:
            configured_path = Path(path)
        elif config:
            configured_path = Path((config.get("app") or {}).get("review_template_path", self.review_template_path))
        else:
            configured_path = self.review_template_path
        template_path = configured_path if configured_path.is_absolute() else self.root / configured_path
        if not template_path.exists():
            return ""
        return template_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _merge_defaults(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(default)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager._merge_defaults(merged[key], value)
            else:
                merged[key] = value
        return merged


def add_source(config: dict[str, Any], source_type: str, alias: str, ingest_modes: list[str], source_config: dict[str, Any]) -> None:
    sources = config.setdefault("sources", [])
    for source in sources:
        if source["alias"] == alias:
            source.update({"source_type": source_type, "ingest_modes": ingest_modes, "config": source_config, "enabled": True})
            return
    sources.append(
        {
            "alias": alias,
            "source_type": source_type,
            "ingest_modes": ingest_modes,
            "config": source_config,
            "enabled": True,
        }
    )
