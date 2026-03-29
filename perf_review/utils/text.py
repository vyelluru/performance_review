from __future__ import annotations

import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ISSUE_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
URL_RE = re.compile(r"https?://[^\s)>]+")
PR_REF_RE = re.compile(r"(?<!\w)#(\d+)\b")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def html_to_text(value: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(value)
    return "\n".join(parser.parts)


def extract_text_from_file(path: str | Path) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        return html_to_text(file_path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".json":
        return json.dumps(json.loads(file_path.read_text(encoding="utf-8", errors="ignore")), indent=2)
    if suffix == ".pdf":
        pdftotext = _command_exists("pdftotext")
        if pdftotext:
            result = subprocess.run([pdftotext, str(file_path), "-"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout
        return ""
    return file_path.read_text(encoding="utf-8", errors="ignore")


def flatten_atlassian_doc(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text_parts = [flatten_atlassian_doc(item) for item in value.values()]
        return " ".join(part for part in text_parts if part)
    if isinstance(value, list):
        return " ".join(flatten_atlassian_doc(item) for item in value if item is not None)
    return str(value)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9_./-]{3,}", text)]


def extract_issue_keys(text: str) -> list[str]:
    return sorted(set(ISSUE_KEY_RE.findall(text)))


def extract_urls(text: str) -> list[str]:
    return sorted(set(URL_RE.findall(text)))


def extract_pr_refs(text: str) -> list[str]:
    return sorted(set(PR_REF_RE.findall(text)))


def summarize_text(text: str, limit: int = 320) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def slugify(value: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return lowered.strip("-") or "task"


def _command_exists(name: str) -> str | None:
    result = subprocess.run(["/usr/bin/which", name], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()

