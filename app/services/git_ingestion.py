"""Index a public Git repository into Qdrant."""

import logging
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from git import GitCommandError, Repo

from app.services.ingestion import (
    CODE_EXTENSIONS,
    CODE_FILENAMES,
    IngestionError,
    IngestionResult,
    collect_documents,
    index_chunks,
    split_documents,
)

logger = logging.getLogger(__name__)


def validate_repo_url(repo_url: str) -> str:
    """Accept only https URLs with a host.

    Rejecting file://, ssh and bare paths stops the clone endpoint from being used to
    read files on the server or reach internal services.
    """
    url = (repo_url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.path.strip("/"):
        raise IngestionError(f"Repository URL must be an https URL, got: {repo_url!r}")
    return url


def ingest_github_repository(repo_url: str, branch: str = "main") -> IngestionResult:
    """Shallow-clone a repository into a temp dir, index its source files, then clean up."""
    url = validate_repo_url(repo_url)
    branch = (branch or "main").strip()

    with tempfile.TemporaryDirectory(prefix="context_engine_") as tmp:
        logger.info("Cloning %s (branch %s)", url, branch)
        try:
            Repo.clone_from(url, tmp, branch=branch, depth=1)
        except GitCommandError as err:
            raise IngestionError(
                f"Could not clone {url} on branch '{branch}'. "
                "Check the URL, the branch name and that the repository is public."
            ) from err

        documents = collect_documents(
            Path(tmp), CODE_EXTENSIONS, filenames=CODE_FILENAMES, extra_metadata={"repo_url": url}
        )
        if not documents:
            raise IngestionError(f"No supported source files found in {url}")

        chunks = split_documents(documents, chunk_overlap=150)
        logger.info("Indexing %d chunks from %d files in %s", len(chunks), len(documents), url)
        index_chunks(chunks, recreate=False)

    return IngestionResult(source=url, files=len(documents), chunks=len(chunks))
