from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from perf_review.llm.provider import BaseLLMProvider
from perf_review.store.db import Database
from perf_review.utils.text import extract_issue_keys, extract_pr_refs, extract_urls, slugify, summarize_text, tokenize
from perf_review.utils.time import earliest, latest

DEFAULT_BRANCH_NAMES = {"main", "master", "trunk", "develop", "development", "dev"}
TOPIC_STOPWORDS = {
    "added",
    "adding",
    "adjust",
    "and",
    "branch",
    "bump",
    "change",
    "cleanup",
    "default",
    "docs",
    "feature",
    "final",
    "first",
    "fix",
    "for",
    "from",
    "hot",
    "into",
    "merge",
    "new",
    "of",
    "play",
    "readme",
    "the",
    "update",
    "usage",
    "use",
    "with",
}


@dataclass(slots=True)
class Cluster:
    key: str
    artifact_ids: list[int]
    reason: str
    primary_repo: str | None
    title_hint: str | None = None


def rebuild_graph(database: Database, rubric: dict[str, Any]) -> None:
    rebuild_graph_with_enrichment(database, rubric, BaseLLMProvider())


def rebuild_graph_with_enrichment(database: Database, rubric: dict[str, Any], llm_provider: BaseLLMProvider) -> None:
    database.clear_graph()
    database.set_competencies(rubric.get("competencies", []))
    artifacts = [dict(row) for row in database.fetch_artifacts()]
    entity_ids: dict[tuple[str, str], int] = {}
    anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    orphan_artifacts: list[dict[str, Any]] = []

    for artifact in artifacts:
        metadata = _load_json(artifact["metadata_json"])
        combined_text = f"{artifact['title']} {artifact['body_text']}"
        repo_name = metadata.get("repo_name") or metadata.get("repo")
        if repo_name:
            entity_ids.setdefault(("repo", repo_name), database.insert_entity("repo", repo_name, repo_name, {"source_alias": artifact["source_alias"]}))
            database.insert_edge("artifact", artifact["id"], "implemented_in", "entity", entity_ids[("repo", repo_name)], artifact["id"], metadata={"repo_name": repo_name})
        if artifact["artifact_type"] == "repo":
            continue
        if artifact.get("author"):
            author_key = artifact["author"].strip().lower()
            entity_ids.setdefault(("person", author_key), database.insert_entity("person", author_key, artifact["author"], {}))
            database.insert_edge("artifact", artifact["id"], "reviewed_by" if artifact["artifact_type"] == "review" else "mentions", "entity", entity_ids[("person", author_key)], artifact["id"], metadata={"kind": "author"})
        issue_keys, urls, pr_refs, branch_keys = _artifact_anchor_parts(artifact, combined_text)
        strong_anchor_found = False
        for issue_key in issue_keys:
            entity_ids.setdefault(("issue", issue_key), database.insert_entity("issue", issue_key, issue_key, {}))
            database.insert_edge("artifact", artifact["id"], "mentions", "entity", entity_ids[("issue", issue_key)], artifact["id"])
            anchors[f"issue:{issue_key}"].append(artifact)
            strong_anchor_found = True
        for pr_ref in pr_refs:
            value = f"{repo_name or artifact['source_alias']}#{pr_ref}"
            entity_ids.setdefault(("pr", value), database.insert_entity("pr", value, value, {}))
            database.insert_edge("artifact", artifact["id"], "mentions", "entity", entity_ids[("pr", value)], artifact["id"])
            anchors[f"pr:{value}"].append(artifact)
            strong_anchor_found = True
        for url in urls:
            entity_ids.setdefault(("doc", url), database.insert_entity("doc", url, summarize_text(url, 80), {}))
            database.insert_edge("artifact", artifact["id"], "mentions", "entity", entity_ids[("doc", url)], artifact["id"])
            strong_anchor_found = True
        for branch_key in branch_keys:
            anchors[f"branch:{branch_key}"].append(artifact)
            strong_anchor_found = True
        if not strong_anchor_found and artifact["artifact_type"] != "branch":
            orphan_artifacts.append(artifact)

    initial_clusters = _build_anchor_clusters(anchors)
    artifact_lookup = {artifact["id"]: artifact for artifact in artifacts}
    semantic_clusters = _build_semantic_clusters(orphan_artifacts)
    merged_clusters = _merge_clusters(initial_clusters + semantic_clusters, artifact_lookup)

    for cluster in merged_clusters:
        cluster_artifacts = [artifact for artifact in artifacts if artifact["id"] in cluster.artifact_ids]
        evidence = _task_evidence_rows(cluster_artifacts)
        title = _derive_task_title(cluster, cluster_artifacts)
        enrichment = llm_provider.enrich_task(title, evidence, cluster.reason)
        description = summarize_text(enrichment["implementation_summary"] or enrichment["summary"] or " ".join(artifact["title"] for artifact in cluster_artifacts), 400)
        start_at = earliest(artifact.get("occurred_at") for artifact in cluster_artifacts)
        end_at = latest(artifact.get("occurred_at") for artifact in cluster_artifacts)
        task_key = cluster.key
        task_id = database.insert_task(
            task_key=task_key,
            title=title,
            description=description,
            summary=enrichment["summary"],
            implementation_summary=enrichment["implementation_summary"],
            impact_summary=enrichment["impact_summary"],
            collaboration_summary=enrichment["collaboration_summary"],
            complexity_score=float(enrichment["complexity_score"]),
            complexity_reasoning=enrichment["complexity_reasoning"],
            status=enrichment["status"],
            source_anchor=cluster.reason,
            confidence=_cluster_confidence(cluster, cluster_artifacts),
            start_at=start_at,
            end_at=end_at,
            primary_repo=cluster.primary_repo,
            metadata={"reason": cluster.reason, "artifact_count": len(cluster.artifact_ids), "title_hint": cluster.title_hint},
        )
        for artifact in cluster_artifacts:
            reason = cluster.reason
            score = 1.0 if cluster.reason.startswith("anchor") else 0.65
            database.insert_task_membership(task_id, artifact["id"], score, reason)
            database.insert_edge("artifact", artifact["id"], "supports", "task", task_id, artifact["id"], confidence=score)
        if cluster.primary_repo and ("repo", cluster.primary_repo) in entity_ids:
            database.insert_edge("task", task_id, "belongs_to", "entity", entity_ids[("repo", cluster.primary_repo)], confidence=1.0)
    database.commit()


def _load_json(raw: str) -> dict[str, Any]:
    return json.loads(raw) if raw else {}


def _build_anchor_clusters(anchors: dict[str, list[dict[str, Any]]]) -> list[Cluster]:
    clusters: list[Cluster] = []
    for key, items in anchors.items():
        unique_ids = sorted({artifact["id"] for artifact in items})
        if not unique_ids:
            continue
        repo_names = [json.loads(artifact["metadata_json"]).get("repo_name") or json.loads(artifact["metadata_json"]).get("repo") for artifact in items]
        primary_repo = next((repo for repo in repo_names if repo), None)
        clusters.append(Cluster(key=slugify(key), artifact_ids=unique_ids, reason=f"anchor:{key}", primary_repo=primary_repo, title_hint=_anchor_title_hint(key, items)))
    return clusters


def _build_semantic_clusters(orphan_artifacts: list[dict[str, Any]]) -> list[Cluster]:
    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in orphan_artifacts:
        metadata = json.loads(artifact["metadata_json"])
        repo_name = metadata.get("repo_name") or metadata.get("repo")
        scope = repo_name or "global"
        by_scope[scope].append(artifact)

    clusters: list[Cluster] = []
    for scope, scoped_artifacts in by_scope.items():
        token_to_ids: dict[str, list[int]] = defaultdict(list)
        artifact_tokens: dict[int, list[str]] = {}
        for artifact in scoped_artifacts:
            tokens = _topic_tokens(artifact, None if scope == "global" else scope)
            artifact_tokens[artifact["id"]] = tokens
            for token in set(tokens):
                token_to_ids[token].append(artifact["id"])

        used: set[int] = set()
        prioritized_tokens = sorted(
            (
                (token, len(ids))
                for token, ids in token_to_ids.items()
                if len(ids) >= 2
            ),
            key=lambda item: (-item[1], item[0]),
        )
        for token, _ in prioritized_tokens:
            candidate_ids = sorted(artifact_id for artifact_id in token_to_ids[token] if artifact_id not in used)
            if len(candidate_ids) < 2:
                continue
            used.update(candidate_ids)
            title_hint = _semantic_title_hint(token, scoped_artifacts, candidate_ids)
            clusters.append(
                Cluster(
                    key=slugify(f"topic:{scope}:{token}"),
                    artifact_ids=candidate_ids,
                    reason=f"semantic:token:{token}",
                    primary_repo=None if scope == "global" else scope,
                    title_hint=title_hint,
                )
            )

        for artifact in scoped_artifacts:
            if artifact["id"] in used:
                continue
            title_hint = summarize_text(artifact["title"], 80)
            clusters.append(
                Cluster(
                    key=slugify(f"artifact:{scope}:{artifact['id']}"),
                    artifact_ids=[artifact["id"]],
                    reason="semantic:singleton",
                    primary_repo=None if scope == "global" else scope,
                    title_hint=title_hint,
                )
            )
    return clusters


def _merge_clusters(clusters: list[Cluster], artifact_lookup: dict[int, dict[str, Any]]) -> list[Cluster]:
    merged: list[Cluster] = []
    consumed: set[int] = set()
    for index, cluster in enumerate(clusters):
        if index in consumed:
            continue
        current_ids = set(cluster.artifact_ids)
        current_reason = cluster.reason
        primary_repo = cluster.primary_repo
        title_hint = cluster.title_hint
        signature = _cluster_signature(cluster, artifact_lookup)
        for other_index in range(index + 1, len(clusters)):
            if other_index in consumed:
                continue
            other = clusters[other_index]
            other_signature = _cluster_signature(other, artifact_lookup)
            similarity = _cosine(signature, other_signature)
            if similarity >= 0.58 and (primary_repo is None or other.primary_repo is None or primary_repo == other.primary_repo):
                current_ids.update(other.artifact_ids)
                consumed.add(other_index)
                if current_reason.startswith("anchor"):
                    current_reason = "merged:" + current_reason
                else:
                    current_reason = "merged:semantic"
                primary_repo = primary_repo or other.primary_repo
                title_hint = title_hint or other.title_hint
        merged.append(Cluster(key=cluster.key, artifact_ids=sorted(current_ids), reason=current_reason, primary_repo=primary_repo, title_hint=title_hint))
    return merged


def _cluster_signature(cluster: Cluster, artifact_lookup: dict[int, dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for artifact_id in cluster.artifact_ids:
        artifact = artifact_lookup.get(artifact_id)
        if not artifact:
            continue
        tokens = tokenize(f"{artifact['title']} {artifact['body_text']}")
        counter.update(tokens)
    return counter


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right[token] for token in set(left) & set(right))
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _derive_task_title(cluster: Cluster, artifacts: list[dict[str, Any]]) -> str:
    if cluster.title_hint:
        return cluster.title_hint
    titles = [artifact["title"] for artifact in artifacts if artifact.get("title")]
    issue_titles = [title for title in titles if any(char.isdigit() for char in title[:10])]
    if issue_titles:
        return issue_titles[0]
    if titles:
        return sorted(titles, key=len)[0]
    return cluster.key.replace("-", " ").title()


def _cluster_confidence(cluster: Cluster, artifacts: list[dict[str, Any]]) -> float:
    base = 0.55 if cluster.reason.startswith("merged") else 0.85
    size_bonus = min(0.15, 0.03 * max(0, len(artifacts) - 1))
    return round(min(0.99, base + size_bonus), 2)


def _artifact_anchor_parts(artifact: dict[str, Any], combined_text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    issue_keys = extract_issue_keys(combined_text)
    urls = extract_urls(combined_text)
    pr_refs = extract_pr_refs(combined_text)
    branch_keys: list[str] = []
    if artifact["artifact_type"] == "branch":
        branch_name = artifact["title"].strip()
        if branch_name and branch_name.lower() not in DEFAULT_BRANCH_NAMES:
            branch_keys.append(branch_name)
    return issue_keys, urls, pr_refs, branch_keys


def _anchor_title_hint(key: str, items: list[dict[str, Any]]) -> str | None:
    kind, _, raw = key.partition(":")
    if kind == "issue":
        issue_titles = [artifact["title"] for artifact in items if artifact["artifact_type"] == "issue" and artifact["title"]]
        return issue_titles[0] if issue_titles else raw
    if kind == "branch":
        return raw
    if kind == "pr":
        pr_titles = [artifact["title"] for artifact in items if artifact["artifact_type"] == "pr" and artifact["title"]]
        return pr_titles[0] if pr_titles else raw
    return None


def _topic_tokens(artifact: dict[str, Any], repo_name: str | None) -> list[str]:
    repo_tokens = set(tokenize(repo_name or ""))
    raw_tokens = tokenize(f"{artifact['title']} {artifact['body_text']}")
    topic_tokens: list[str] = []
    for token in raw_tokens:
        if token in repo_tokens or token in TOPIC_STOPWORDS:
            continue
        if token.startswith("http") or token in DEFAULT_BRANCH_NAMES:
            continue
        topic_tokens.append(token)
    return topic_tokens


def _semantic_title_hint(token: str, repo_artifacts: list[dict[str, Any]], candidate_ids: list[int]) -> str:
    matching_titles = [artifact["title"] for artifact in repo_artifacts if artifact["id"] in candidate_ids and token in tokenize(artifact["title"])]
    if matching_titles:
        return sorted(matching_titles, key=len)[0]
    return token.replace("-", " ").title()


def _task_evidence_rows(cluster_artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for artifact in cluster_artifacts:
        metadata = _load_json(artifact["metadata_json"])
        evidence.append(
            {
                "artifact_id": artifact["id"],
                "artifact_type": artifact["artifact_type"],
                "title": artifact["title"],
                "body_text": summarize_text(artifact["body_text"], 240),
                "source_alias": artifact["source_alias"],
                "occurred_at": artifact["occurred_at"],
                "repo_name": metadata.get("repo_name") or metadata.get("repo"),
            }
        )
    return evidence
