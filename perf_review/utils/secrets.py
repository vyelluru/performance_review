from __future__ import annotations

import getpass
import subprocess
from dataclasses import dataclass


SERVICE_PREFIX = "perf-review"


@dataclass(slots=True)
class SecretRecord:
    account: str
    token: str


class SecretStore:
    def save_token(self, account: str, token: str) -> None:
        raise NotImplementedError

    def get_token(self, account: str) -> str | None:
        raise NotImplementedError


class MacOSKeychainSecretStore(SecretStore):
    def save_token(self, account: str, token: str) -> None:
        service = f"{SERVICE_PREFIX}:{account}"
        subprocess.run(
            ["security", "add-generic-password", "-U", "-a", account, "-s", service, "-w", token],
            check=True,
            capture_output=True,
            text=True,
        )

    def get_token(self, account: str) -> str | None:
        service = f"{SERVICE_PREFIX}:{account}"
        result = subprocess.run(
            ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()


class MemorySecretStore(SecretStore):
    def __init__(self) -> None:
        self._tokens: dict[str, str] = {}

    def save_token(self, account: str, token: str) -> None:
        self._tokens[account] = token

    def get_token(self, account: str) -> str | None:
        return self._tokens.get(account)


def prompt_for_token(prompt: str = "API token: ") -> str:
    return getpass.getpass(prompt)
