# Architecture

`perf-review` is a local-first CLI that turns engineering evidence into a small set of canonical work items, then uses those task records to draft a performance review.

## System Overview

The system has four main stages:

1. Source setup
2. Ingestion and normalization
3. Task formation and graph building
4. Review generation

At a high level:

```text
configured sources
  -> raw source payloads
  -> normalized artifacts
  -> canonical tasks + graph edges
  -> review outputs
```

## Sources

The CLI supports a mix of direct-sync and import-mode sources:

- `git`
- `jira`
- `github`
- `confluence`

Examples:

- Local git repos contribute commits, branches, tags, and repo metadata.
- Jira contributes issues and comments.
- Confluence import mode contributes parsed document chunks from PDFs, markdown, or text docs.

Source configuration is stored in `perf.yaml`, while sync cursors are tracked in SQLite.

## Storage Model

The SQLite database is split into layers.

### 1. Source and sync state

- `source_accounts`
- `sync_state`

These tables answer:

- what sources are connected
- how each source is configured
- where the next incremental sync should resume

### 2. Raw ingestion layer

- `raw_documents`

This is the traceability layer. It stores source-native payloads close to the original format returned by git/Jira/GitHub/Confluence/imports.

### 3. Normalized evidence layer

- `artifacts`
- `artifact_search`

Every source record is normalized into a shared artifact shape. Common artifact types include:

- `commit`
- `branch`
- `tag`
- `repo`
- `issue`
- `comment`
- `pr`
- `review`
- `doc`

This is the main evidence layer used by the graph builder.

### 4. Graph and task layer

- `tasks`
- `task_memberships`
- `entities`
- `edges`
- `competencies`

This is the most important layer for review generation.

Each row in `tasks` is intended to represent one meaningful workstream, not one commit or one ticket. A task can span:

- multiple Jira issues
- multiple repos
- multiple commits
- multiple document chunks

The `task_memberships` table explains which artifacts support each task and why.

The `entities` and `edges` tables provide a graph-shaped representation for repos, people, issues, and relationships.

### 5. Review output layer

- `review_runs`
- `claim_citations`

Each draft run stores the generated review text and the citations backing each claim.

## Ingestion Flow

During `ingest`, the system:

1. Reads configured sources from `perf.yaml`
2. Fetches source data
3. Stores source-native payloads in `raw_documents`
4. Normalizes them into `artifacts`
5. Updates `sync_state` for incremental sources

For documents, ingestion also includes parsing and chunking:

- large notes/PDFs are split into section-level `doc` artifacts
- chunk metadata keeps provenance like section title and chunk index

## Task Formation Flow

During `build`, the system:

1. Clears the previous graph/task layer
2. Loads all normalized artifacts
3. Extracts anchors such as:
   - Jira keys
   - PR references
   - branch names
   - URLs
4. Builds initial anchor-based clusters
5. Builds semantic clusters for orphaned artifacts
6. Merges related clusters
7. Runs a softer consolidation pass to absorb thin supporting clusters into stronger tasks
8. Writes canonical task rows plus memberships and edges

The clustering logic is intentionally not a hard cap. There is a soft target for task count, but merges only happen when the evidence looks strong enough.

## What a Task Row Contains

A task row is the compact context that the review generator uses. Current fields include:

- `title`
- `description`
- `summary`
- `implementation_summary`
- `impact_summary`
- `collaboration_summary`
- `challenge_summary`
- `complexity_score`
- `complexity_reasoning`
- `status`
- `source_anchor`
- `story_points`
- `artifact_count`
- `repo_count`
- `people_json`
- `jira_keys_json`
- `labels_json`
- `issue_types_json`
- `repo_names_json`

This means the database itself stores the canonical work items before the LLM writes anything.

## Ollama's Role

Ollama does not read the database directly.

Instead, Python:

1. Reads the task rows and supporting evidence from SQLite
2. Builds a compact task context
3. Passes that context, plus optional review template instructions, into Ollama
4. Receives drafted review prose back

If Ollama is unavailable, the system falls back to deterministic text generation so the pipeline still completes.

## Output Flow

During `review draft` or `run`, the system:

1. Loads the selected review period
2. Reads the canonical tasks for that period
3. Applies the optional `review_template.md` instructions
4. Drafts:
   - self review
   - evidence appendix
   - gaps report
5. Writes final output files:
   - `.perf_review/output/self_review.md`
   - `.perf_review/output/evidence_appendix.md`
   - `.perf_review/output/gaps.md`
   - `.perf_review/output/review_report.pdf`

## Typical CLI Flow

```bash
python3 -m perf_review init
python3 -m perf_review source add git --path ~/repo-a ~/repo-b
python3 -m perf_review source add jira --base-url https://example.atlassian.net --project ENG --alias jira-eng
python3 -m perf_review source add confluence --import-path ~/notes.pdf ~/design.pdf
python3 -m perf_review run --period 2026-H1
```

## Design Intent

The core design goal is:

- keep ingestion deterministic
- keep the database task-first
- use the LLM for synthesis and narration, not raw parsing

That separation makes the system easier to debug, easier to trust, and more stable as new sources are added.
