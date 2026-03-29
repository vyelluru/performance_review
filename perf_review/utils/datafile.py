from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_structured_file(path: str | Path, default: Any | None = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return default

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _load_simple_yaml(text)


def dump_structured_file(path: str | Path, data: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_simple_yaml(text: str) -> Any:
    """Parse a tiny YAML subset used by default config files.

    This supports mappings, lists, strings, booleans, numbers, and null values.
    """

    lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return None

    index = 0

    def parse_block(indent: int) -> tuple[Any, int]:
        nonlocal index
        items: list[Any] = []
        mapping: dict[str, Any] = {}
        mode: str | None = None

        while index < len(lines):
            line = lines[index]
            current_indent = len(line) - len(line.lstrip(" "))
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation in structured file near line: {line}")

            stripped = line.strip()
            if stripped.startswith("- "):
                if mode is None:
                    mode = "list"
                elif mode != "list":
                    raise ValueError("Mixed YAML structures are not supported.")
                value = stripped[2:]
                index += 1
                if not value:
                    nested, _ = parse_block(indent + 2)
                    items.append(nested)
                elif ":" in value and not value.startswith('"') and not value.startswith("'"):
                    key, raw = value.split(":", 1)
                    nested_map = {key.strip(): _parse_scalar(raw.strip())}
                    if index < len(lines):
                        next_line = lines[index] if index < len(lines) else ""
                        next_indent = len(next_line) - len(next_line.lstrip(" ")) if next_line else 0
                        if next_line and next_indent >= indent + 2:
                            nested, _ = parse_block(indent + 2)
                            if isinstance(nested, dict):
                                nested_map.update(nested)
                    items.append(nested_map)
                else:
                    items.append(_parse_scalar(value))
            else:
                if mode is None:
                    mode = "dict"
                elif mode != "dict":
                    raise ValueError("Mixed YAML structures are not supported.")
                if ":" not in stripped:
                    raise ValueError(f"Expected key/value pair, got: {stripped}")
                key, raw_value = stripped.split(":", 1)
                key = key.strip()
                raw_value = raw_value.strip()
                index += 1
                if raw_value:
                    mapping[key] = _parse_scalar(raw_value)
                else:
                    nested, _ = parse_block(indent + 2)
                    mapping[key] = nested
        return (items if mode == "list" else mapping), index

    parsed, _ = parse_block(0)
    return parsed


def _parse_scalar(raw: str) -> Any:
    if not raw:
        return ""
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw

