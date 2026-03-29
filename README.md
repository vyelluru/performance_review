# perf-review

`perf-review` is a local-first CLI for collecting engineering work evidence, grouping it into major tasks, and generating performance review drafts with citations.

## Highlights

- Local SQLite storage with a graph-shaped data model
- Explicit repo, Jira, GitHub, and Confluence connectors
- Import and direct-sync modes where appropriate
- Optional Ollama synthesis with deterministic fallback
- Markdown and static HTML review outputs

## Quickstart

```bash
python3 -m perf_review init
python3 -m perf_review source add git --path ~/code/example
python3 -m perf_review build --period 2026-H1
python3 -m perf_review review draft --period 2026-H1 --rubric rubric.yml
```

## Notes

- Config and output paths default to files inside the current working directory.
- Config files are written as JSON-formatted YAML for portability without extra dependencies.
- Jira, Confluence, and GitHub credentials are stored in the macOS keychain by default.

