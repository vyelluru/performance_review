from perf_review.connectors.base import BaseConnector
from perf_review.connectors.confluence import ConfluenceConnector
from perf_review.connectors.git_local import GitLocalConnector
from perf_review.connectors.github import GitHubConnector
from perf_review.connectors.jira import JiraConnector


CONNECTOR_TYPES: dict[str, type[BaseConnector]] = {
    "git": GitLocalConnector,
    "github": GitHubConnector,
    "jira": JiraConnector,
    "confluence": ConfluenceConnector,
}

