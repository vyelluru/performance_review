from __future__ import annotations

import base64
import json
from typing import Any
from urllib import parse, request

from perf_review.connectors.base import ConnectorError


def http_get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = request.Request(url, headers=headers or {})
    try:
        with request.urlopen(req, timeout=30) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            payload = response.read().decode(charset)
            return json.loads(payload)
    except Exception as exc:  # pragma: no cover - exercised through connector tests via fakes
        raise ConnectorError(f"Failed to fetch {url}: {exc}") from exc


def auth_headers(token: str, username: str | None = None) -> dict[str, str]:
    if username:
        basic = base64.b64encode(f"{username}:{token}".encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {basic}", "Accept": "application/json"}
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def append_query(url: str, params: dict[str, Any]) -> str:
    return f"{url}?{parse.urlencode(params, doseq=True)}"
