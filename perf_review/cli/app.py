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

    run_parser = subparsers.add_parser("run", help="Ingest, build, and generate a review in one command")
    run_parser.add_argument("--period", required=True)
    run_parser.add_argument("--rubric")
    run_parser.add_argument("--template")
    run_parser.set_defaults(func=cmd_run)

    source_parser = subparsers.add_parser("source", help="Manage sources")
    source_sub = source_parser.add_subparsers(dest="source_command")
    source_add = source_sub.add_parser("add", help="Add or update a source")
    source_add.add_argument("source_type", choices=sorted(CONNECTOR_TYPES))
    source_add.add_argument("--alias")
    source_add.add_argument("--path", nargs="+")
    source_add.add_argument("--repo")
    source_add.add_argument("--base-url")
    source_add.add_argument("--project")
    source_add.add_argument("--space")
    source_add.add_argument("--import-path", nargs="+")
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
    review_draft.add_argument("--template")
    review_draft.set_defaults(func=cmd_review_draft)
    review_report = review_sub.add_parser("report", help="Generate HTML report and markdown outputs")
    review_report.add_argument("--period", required=True)
    review_report.add_argument("--rubric")
    review_report.add_argument("--template")
    review_report.set_defaults(func=cmd_review_draft)
    review_explain = review_sub.add_parser("explain", help="Explain how a task was built")
    review_explain.add_argument("--task", type=int, required=True)
    review_explain.set_defaults(func=cmd_review_explain)
    review_template = review_sub.add_parser("template", help="Create or inspect the editable review template")
    review_template_sub = review_template.add_subparsers(dest="review_template_command")
    review_template_init = review_template_sub.add_parser("init", help="Create the default review template if needed")
    review_template_init.add_argument("--path")
    review_template_init.set_defaults(func=cmd_review_template_init)
    review_template_show = review_template_sub.add_parser("show", help="Print the current review template")
    review_template_show.add_argument("--path")
    review_template_show.set_defaults(func=cmd_review_template_show)

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
        print(f"- Review template: {config_manager.review_template_path}")
        print(f"- DB: {root / config['app']['db_path']}")
        return 0
    finally:
        database.close()


def cmd_source_add(args: argparse.Namespace) -> int:
    root, config_manager, config, database = _load_project()
    try:
        source_type = args.source_type
        source_entries = _source_entries_from_args(source_type, args)
        for alias, ingest_modes, source_config in source_entries:
            add_source(config, source_type, alias, ingest_modes, source_config)
            print(f"Configured source '{alias}' ({source_type}) with modes: {', '.join(ingest_modes)}")
        config_manager.save_config(config)
        _sync_sources_to_db(database, config)
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
                import_path = args.import_path[0] if isinstance(args.import_path, list) and args.import_path else args.import_path
                result = connector.fetch(mode=mode, sync_state=sync_state, import_path=import_path)
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
        rebuild_graph_with_enrichment(database, rubric, llm_provider, (config.get("app") or {}).get("tasking"))
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
        review_template = config_manager.load_review_template(args.template, config)
        llm_provider = build_llm_provider(config)
        rebuild_graph_with_enrichment(database, rubric, llm_provider, (config.get("app") or {}).get("tasking"))
        rubric_path = str(Path(args.rubric) if args.rubric else config_manager.rubric_path)
        review_run_id, artifacts = generate_review(database, llm_provider, rubric, period, rubric_path, review_template)
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


def cmd_review_template_init(args: argparse.Namespace) -> int:
    root = Path.cwd()
    config_manager = ConfigManager(root)
    config_manager.ensure_project()
    template_path = Path(args.path) if args.path else config_manager.review_template_path
    if not template_path.is_absolute():
        template_path = root / template_path
    if template_path.exists():
        print(f"Review template already exists at {template_path}")
        return 0
    from perf_review.utils.config import DEFAULT_REVIEW_TEMPLATE

    template_path.write_text(DEFAULT_REVIEW_TEMPLATE, encoding="utf-8")
    print(f"Created review template at {template_path}")
    return 0


def cmd_review_template_show(args: argparse.Namespace) -> int:
    root = Path.cwd()
    config_manager = ConfigManager(root)
    config_manager.ensure_project()
    config = config_manager.load_config()
    template = config_manager.load_review_template(args.path, config)
    print(template)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root, config_manager, config, database = _load_project()
    try:
        period = parse_period(args.period)
        rubric = config_manager.load_rubric(args.rubric)
        review_template = config_manager.load_review_template(args.template, config)
        secret_store = MacOSKeychainSecretStore()
        sources = config.get("sources", [])
        if not sources:
            raise ValueError("No sources configured.")
        for source in sources:
            connector = _instantiate_connector(source, secret_store)
            sync_state = database.get_sync_state(source["alias"])
            for mode in source.get("ingest_modes", connector.supported_modes()):
                if mode not in connector.supported_modes():
                    continue
                result = connector.fetch(mode=mode, sync_state=sync_state, import_path=None)
                artifact_ids = database.upsert_artifacts(result.artifacts)
                database.insert_raw_documents(result.raw_documents)
                database.set_sync_state(source["alias"], mode, result.cursor, result.etag, "ok")
                print(f"Ingested {len(artifact_ids)} artifacts from {source['alias']} ({mode})")
        llm_provider = build_llm_provider(config)
        rebuild_graph_with_enrichment(database, rubric, llm_provider, (config.get("app") or {}).get("tasking"))
        task_count = len(database.fetch_tasks_for_period(period.start.isoformat(), period.end.isoformat()))
        print(f"Built graph and task clusters for {args.period}. {task_count} tasks overlap the selected period.")
        rubric_path = str(Path(args.rubric) if args.rubric else config_manager.rubric_path)
        review_run_id, artifacts = generate_review(database, llm_provider, rubric, period, rubric_path, review_template)
        outputs = write_review_outputs(root / config["app"]["output_dir"], artifacts)
        print(f"Generated review run {review_run_id} for {args.period}")
        for name, path in outputs.items():
            print(f"- {name}: {path}")
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
    first_path = args.path[0] if isinstance(args.path, list) and args.path else args.path
    first_import_path = args.import_path[0] if isinstance(args.import_path, list) and args.import_path else args.import_path
    if source_type == "git" and first_path:
        return Path(first_path).expanduser().name
    if source_type == "github" and args.repo:
        return args.repo.replace("/", "-")
    if source_type == "jira" and args.project:
        return f"jira-{args.project.lower()}"
    if source_type == "confluence" and args.space:
        return f"confluence-{args.space.lower()}"
    if source_type == "confluence" and first_import_path:
        return Path(first_import_path).expanduser().stem.replace("_", "-")
    return source_type


def _source_entries_from_args(source_type: str, args: argparse.Namespace) -> list[tuple[str, list[str], dict[str, Any]]]:
    if source_type == "git":
        if not args.path:
            raise ValueError("--path is required for git sources")
        if args.alias and len(args.path) > 1:
            raise ValueError("--alias can only be used with a single git path")
        return [
            (args.alias or Path(path).expanduser().name, ["local"], {"path": str(Path(path).expanduser())})
            for path in args.path
        ]
    if source_type == "github":
        if not args.repo:
            raise ValueError("--repo is required for github sources")
        return [(args.alias or _default_alias(source_type, args), ["direct"], {"repo": args.repo, "api_base": args.base_url or "https://api.github.com"})]
    if source_type == "jira":
        import_paths = args.import_path or []
        if not args.base_url and not import_paths:
            raise ValueError("Jira requires --base-url and/or --import-path")
        if len(import_paths) > 1:
            raise ValueError("Jira accepts only one --import-path at a time")
        modes = _modes_for_optional_direct_import(args.base_url, import_paths[0] if import_paths else None)
        return [(
            args.alias or _default_alias(source_type, args),
            modes,
            {"base_url": args.base_url, "project": args.project, "import_path": import_paths[0] if import_paths else None, "auth_username": args.auth_username},
        )]
    if source_type == "confluence":
        import_paths = args.import_path or []
        if not args.base_url and not import_paths:
            raise ValueError("Confluence requires --base-url and/or --import-path")
        if args.base_url or args.alias:
            if len(import_paths) > 1:
                raise ValueError("Use one confluence source per alias when combining direct sync or custom alias with imports")
            modes = _modes_for_optional_direct_import(args.base_url, import_paths[0] if import_paths else None)
            return [(
                args.alias or _default_alias(source_type, args),
                modes,
                {"base_url": args.base_url, "space": args.space, "import_path": import_paths[0] if import_paths else None, "auth_username": args.auth_username},
            )]
        return [
            (
                Path(import_path).expanduser().stem.replace("_", "-"),
                ["import"],
                {"base_url": None, "space": None, "import_path": str(Path(import_path).expanduser()), "auth_username": None},
            )
            for import_path in import_paths
        ]
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
