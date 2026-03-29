from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ISSUE_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
URL_RE = re.compile(r"https?://[^\s)>]+")
PR_REF_RE = re.compile(r"(?<!\w)#(\d+)\b")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(slots=True)
class DocumentChunk:
    title: str
    body_text: str
    section_title: str | None
    heading_path: list[str]
    chunk_index: int


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
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        node_type = value.get("type")
        if node_type == "hardBreak":
            return "\n"
        if "content" in value and isinstance(value["content"], list):
            child_texts = [_flatten_atlassian_nodes(value["content"])]
            joined = child_texts[0]
            if node_type in {"paragraph", "heading", "blockquote", "listItem", "tableCell", "tableHeader"}:
                return joined.strip()
            if node_type in {"bulletList", "orderedList", "table", "tableRow", "doc"}:
                return joined
            return joined.strip()
        text_parts = [
            flatten_atlassian_doc(item)
            for key, item in value.items()
            if key not in {"type", "version", "attrs", "marks"}
        ]
        return _normalize_whitespace(" ".join(part for part in text_parts if part))
    if isinstance(value, list):
        return _flatten_atlassian_nodes(value)
    return str(value)


def _flatten_atlassian_nodes(nodes: list[Any]) -> str:
    rendered: list[str] = []
    for node in nodes:
        text = flatten_atlassian_doc(node)
        if not text:
            continue
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "listItem":
                text = f"- {_normalize_whitespace(text)}"
            elif node_type in {"paragraph", "heading", "blockquote", "tableRow"}:
                text = _normalize_whitespace(text)
        rendered.append(text)
    return "\n".join(part for part in rendered if part).strip()


def _normalize_whitespace(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


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


def chunk_document_text(title: str, text: str, max_chars: int = 1200) -> list[DocumentChunk]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return [
            DocumentChunk(
                title=title,
                body_text="",
                section_title=None,
                heading_path=[],
                chunk_index=1,
            )
        ]

    blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]
    if not blocks:
        return [
            DocumentChunk(
                title=title,
                body_text=normalized,
                section_title=None,
                heading_path=[],
                chunk_index=1,
            )
        ]

    chunks: list[DocumentChunk] = []
    heading_path: list[str] = []
    section_blocks: list[str] = []
    section_title: str | None = None

    def flush_section() -> None:
        nonlocal section_blocks
        if not section_blocks:
            return
        body = "\n\n".join(section_blocks).strip()
        if not body:
            section_blocks = []
            return
        for part in _split_long_text(body, max_chars=max_chars):
            chunks.append(
                DocumentChunk(
                    title=_chunk_title(title, section_title, len(chunks) + 1),
                    body_text=part,
                    section_title=section_title,
                    heading_path=list(heading_path),
                    chunk_index=len(chunks) + 1,
                )
            )
        section_blocks = []

    for block in blocks:
        heading_match = HEADING_RE.match(block.splitlines()[0].strip())
        if heading_match:
            flush_section()
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            heading_path[:] = heading_path[: max(0, level - 1)]
            heading_path.append(heading)
            section_title = heading
            remainder = block.splitlines()[1:]
            if remainder:
                remainder_text = "\n".join(line for line in remainder if line.strip()).strip()
                if remainder_text:
                    section_blocks.append(remainder_text)
            continue

        if _looks_like_heading(block):
            flush_section()
            heading = block.strip()
            heading_path[:] = [heading]
            section_title = heading
            continue

        section_blocks.append(block)

    flush_section()

    if not chunks:
        chunks.append(
            DocumentChunk(
                title=title,
                body_text=normalized,
                section_title=None,
                heading_path=[],
                chunk_index=1,
            )
        )
    return chunks


def _split_long_text(text: str, max_chars: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    if not paragraphs:
        return [text.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        separator_len = 2 if current else 0
        if current and current_len + separator_len + paragraph_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = paragraph_len
            continue
        current.append(paragraph)
        current_len += separator_len + paragraph_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text.strip()]


def _looks_like_heading(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    line = lines[0]
    if len(line) > 80:
        return False
    if line.endswith(".") or line.endswith("!") or line.endswith("?"):
        return False
    if line.endswith(":"):
        return True
    words = line.split()
    if not words or len(words) > 8:
        return False
    alpha_words = [word for word in words if any(character.isalpha() for character in word)]
    if not alpha_words:
        return False
    title_case = sum(1 for word in alpha_words if word[:1].isupper()) >= max(1, len(alpha_words) - 1)
    uppercase = line.isupper()
    return title_case or uppercase


def _chunk_title(document_title: str, section_title: str | None, chunk_index: int) -> str:
    if not section_title:
        return document_title
    if section_title.lower() == document_title.lower():
        return section_title
    return f"{document_title} - {section_title}"


def _command_exists(name: str) -> str | None:
    result = subprocess.run(["/usr/bin/which", name], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()
