from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from perf_review.utils.datafile import dump_structured_file, load_structured_file


DEFAULT_CONFIG = {
  "app": {
    "db_path": ".perf_review/perf_review.db",
    "output_dir": ".perf_review/output",
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

    def ensure_project(self) -> tuple[Path, Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        app_dir = self.root / ".perf_review"
        app_dir.mkdir(exist_ok=True)
        if not self.config_path.exists():
            dump_structured_file(self.config_path, copy.deepcopy(DEFAULT_CONFIG))
        if not self.rubric_path.exists():
            dump_structured_file(self.rubric_path, copy.deepcopy(DEFAULT_RUBRIC))
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

