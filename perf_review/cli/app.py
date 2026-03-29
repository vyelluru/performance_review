from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from perf_review.connectors import CONNECTOR_TYPES
from perf_review.connectors.base import BaseConnector, ConnectorError
from perf_review.graph.builder import rebuild_graph_with_enrichment
from perf_review.llm.provider import build_llm_provider
from perf_review.models import SourceAccount
from perf_review.review.engine import explain_task, generate_review, write_review_outputs
from perf_review.store.db import Database
from perf_review.utils.config import ConfigManager, add_source
from perf_review.utils.secrets import MacOSKeychainSecretStore, prompt_for_token
from perf_review.utils.time import parse_period


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        return int(args.func(args))
    except ConnectorError as exc:
        print(f"Connector error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perf", description="Local-first performance review CLI")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize the project config and local storage")
    init_parser.set_defaults(func=cmd_init)

    source_parser = subparsers.add_parser("source", help="Manage sources")
    source_sub = source_parser.add_subparsers(dest="source_command")
    source_add = source_sub.add_parser("add", help="Add or update a source")
    source_add.add_argument("source_type", choices=sorted(CONNECTOR_TYPES))
    source_add.add_argument("--alias")
    source_add.add_argument("--path")
    source_add.add_argument("--repo")
    source_add.add_argument("--base-url")
    source_add.add_argument("--project")
    source_add.add_argument("--space")
    source_add.add_argument("--import-path")
    source_add.add_argument("--auth-username")
    source_add.set_defaults(func=cmd_source_add)
    source_list = source_sub.add_parser("list", help="List configured sources")
    source_list.set_defaults(func=cmd_source_list)

    auth_parser = subparsers.add_parser("auth", help="Manage source authentication")
    auth_sub = auth_parser.add_subparsers(dest="auth_command")
    auth_login = auth_sub.add_parser("login", help="Store a token for a source in the macOS keychain")
    auth_login.add_argument("alias")
    auth_login.add_argument("--token")
    auth_login.add_argument("--username")
    auth_login.set_defaults(func=cmd_auth_login)

    ingest_parser = subparsers.add_parser("ingest", help="Fetch/import and normalize source data")
    ingest_parser.add_argument("alias", nargs="?")
    ingest_parser.add_argument("--mode", choices=["local", "import", "direct"])
    ingest_parser.add_argument("--import-path")
    ingest_parser.set_defaults(func=cmd_ingest)

    build_parser_cmd = subparsers.add_parser("build", help="Construct the local graph and tasks")
    build_parser_cmd.add_argument("--period", required=True)
    build_parser_cmd.add_argument("--rubric")
    build_parser_cmd.set_defaults(func=cmd_build)

    review_parser = subparsers.add_parser("review", help="Generate or inspect review output")
    review_sub = review_parser.add_subparsers(dest="review_command")
    review_draft = review_sub.add_parser("draft", help="Generate review outputs")
    review_draft.add_argument("--period", required=True)
    review_draft.add_argument("--rubric")
    review_draft.set_defaults(func=cmd_review_draft)
    review_report = review_sub.add_parser("report", help="Generate HTML report and markdown outputs")
    review_report.add_argument("--period", required=True)
    review_report.add_argument("--rubric")
    review_report.set_defaults(func=cmd_review_draft)
    review_explain = review_sub.add_parser("explain", help="Explain how a task was built")
    review_explain.add_argument("--task", type=int, required=True)
    review_explain.set_defaults(func=cmd_review_explain)

    return parser


def cmd_init(args: argparse.Namespace) -> int:
    root = Path.cwd()
    config_manager = ConfigManager(root)
    config_path, rubric_path = config_manager.ensure_project()
    config = config_manager.load_config()
    database = Database(root / config["app"]["db_path"])
    try:
        database.init_schema()
        print(f"Initialized project in {root}")
        print(f"- Config: {config_path}")
        print(f"- Rubric: {rubric_path}")
        print(f"- DB: {root / config['app']['db_path']}")
        return 0
    finally:
        database.close()


def cmd_source_add(args: argparse.Namespace) -> int:
    root, config_manager, config, database = _load_project()
    try:
        source_type = args.source_type
        alias = args.alias or _default_alias(source_type, args)
        ingest_modes, source_config = _source_config_from_args(source_type, args)
        add_source(config, source_type, alias, ingest_modes, source_config)
        config_manager.save_config(config)
        _sync_sources_to_db(database, config)
        print(f"Configured source '{alias}' ({source_type}) with modes: {', '.join(ingest_modes)}")
        return 0
    finally:
        database.close()


def cmd_source_list(args: argparse.Namespace) -> int:
    _, _, config, database = _load_project()
    try:
        for source in config.get("sources", []):
            print(f"{source['alias']}: {source['source_type']} [{', '.join(source.get('ingest_modes', []))}]")
        return 0
    finally:
        database.close()


def cmd_auth_login(args: argparse.Namespace) -> int:
    _, config_manager, config, database = _load_project()
    try:
        source = _find_source(config, args.alias)
        if not source:
            raise ValueError(f"Unknown source alias: {args.alias}")
        token = args.token or prompt_for_token()
        secret_store = MacOSKeychainSecretStore()
        secret_store.save_token(args.alias, token)
        if args.username:
            source["config"]["auth_username"] = args.username
            config_manager.save_config(config)
        print(f"Stored token for {args.alias} in the macOS keychain.")
        return 0
    finally:
        database.close()


def cmd_ingest(args: argparse.Namespace) -> int:
    _, _, config, database = _load_project()
    try:
        secret_store = MacOSKeychainSecretStore()
        sources = [_find_source(config, args.alias)] if args.alias else config.get("sources", [])
        sources = [source for source in sources if source]
        if not sources:
            raise ValueError("No sources configured.")
        total_artifacts = 0
        for source in sources:
            connector = _instantiate_connector(source, secret_store)
            modes = [args.mode] if args.mode else source.get("ingest_modes", connector.supported_modes())
            sync_state = database.get_sync_state(source["alias"])
            for mode in modes:
                if mode not in connector.supported_modes():
                    continue
                result = connector.fetch(mode=mode, sync_state=sync_state, import_path=args.import_path)
                database.insert_raw_documents(result.raw_documents)
                artifact_ids = database.upsert_artifacts(result.artifacts)
                database.set_sync_state(source["alias"], mode, result.cursor, result.etag, "ok")
                total_artifacts += len(artifact_ids)
                print(f"Ingested {len(artifact_ids)} artifacts from {source['alias']} ({mode})")
        return 0 if total_artifacts >= 0 else 1
    finally:
        database.close()


def cmd_build(args: argparse.Namespace) -> int:
    _, config_manager, config, database = _load_project()
    try:
        period = parse_period(args.period)
        rubric = config_manager.load_rubric(args.rubric)
        llm_provider = build_llm_provider(config)
        rebuild_graph_with_enrichment(database, rubric, llm_provider)
        task_count = len(database.fetch_tasks_for_period(period.start.isoformat(), period.end.isoformat()))
        print(f"Built graph and task clusters for {args.period}. {task_count} tasks overlap the selected period.")
        return 0
    finally:
        database.close()


def cmd_review_draft(args: argparse.Namespace) -> int:
    root, config_manager, config, database = _load_project()
    try:
        period = parse_period(args.period)
        rubric = config_manager.load_rubric(args.rubric)
        llm_provider = build_llm_provider(config)
        rebuild_graph_with_enrichment(database, rubric, llm_provider)
        rubric_path = str(Path(args.rubric) if args.rubric else config_manager.rubric_path)
        review_run_id, artifacts = generate_review(database, llm_provider, rubric, period, rubric_path)
        outputs = write_review_outputs(root / config["app"]["output_dir"], artifacts)
        print(f"Generated review run {review_run_id} for {args.period}")
        for name, path in outputs.items():
            print(f"- {name}: {path}")
        return 0
    finally:
        database.close()


def cmd_review_explain(args: argparse.Namespace) -> int:
    _, _, _, database = _load_project()
    try:
        print(explain_task(database, args.task))
        return 0
    finally:
        database.close()


def _load_project() -> tuple[Path, ConfigManager, dict[str, Any], Database]:
    root = Path.cwd()
    config_manager = ConfigManager(root)
    config_manager.ensure_project()
    config = config_manager.load_config()
    database = Database(root / config["app"]["db_path"])
    database.init_schema()
    _sync_sources_to_db(database, config)
    return root, config_manager, config, database


def _sync_sources_to_db(database: Database, config: dict[str, Any]) -> None:
    for source in config.get("sources", []):
        database.upsert_source(
            SourceAccount(
                alias=source["alias"],
                source_type=source["source_type"],
                config=source.get("config", {}),
                ingest_modes=source.get("ingest_modes", []),
                enabled=source.get("enabled", True),
            )
        )


def _default_alias(source_type: str, args: argparse.Namespace) -> str:
    if source_type == "git" and args.path:
        return Path(args.path).expanduser().name
    if source_type == "github" and args.repo:
        return args.repo.replace("/", "-")
    if source_type == "jira" and args.project:
        return f"jira-{args.project.lower()}"
    if source_type == "confluence" and args.space:
        return f"confluence-{args.space.lower()}"
    return source_type


def _source_config_from_args(source_type: str, args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    if source_type == "git":
        if not args.path:
            raise ValueError("--path is required for git sources")
        return ["local"], {"path": str(Path(args.path).expanduser())}
    if source_type == "github":
        if not args.repo:
            raise ValueError("--repo is required for github sources")
        return ["direct"], {"repo": args.repo, "api_base": args.base_url or "https://api.github.com"}
    if source_type == "jira":
        if not args.base_url and not args.import_path:
            raise ValueError("Jira requires --base-url and/or --import-path")
        modes = _modes_for_optional_direct_import(args.base_url, args.import_path)
        return modes, {"base_url": args.base_url, "project": args.project, "import_path": args.import_path, "auth_username": args.auth_username}
    if source_type == "confluence":
        if not args.base_url and not args.import_path:
            raise ValueError("Confluence requires --base-url and/or --import-path")
        modes = _modes_for_optional_direct_import(args.base_url, args.import_path)
        return modes, {"base_url": args.base_url, "space": args.space, "import_path": args.import_path, "auth_username": args.auth_username}
    raise ValueError(f"Unsupported source type: {source_type}")


def _modes_for_optional_direct_import(base_url: str | None, import_path: str | None) -> list[str]:
    modes: list[str] = []
    if import_path:
        modes.append("import")
    if base_url:
        modes.append("direct")
    return modes


def _find_source(config: dict[str, Any], alias: str | None) -> dict[str, Any] | None:
    if alias is None:
        return None
    for source in config.get("sources", []):
        if source["alias"] == alias:
            return source
    return None


def _instantiate_connector(source: dict[str, Any], secret_store: MacOSKeychainSecretStore) -> BaseConnector:
    connector_class = CONNECTOR_TYPES[source["source_type"]]
    return connector_class(source["alias"], source.get("config", {}), secret_store)
