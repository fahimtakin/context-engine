"""Loading, chunking and indexing documents into Qdrant."""

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.llm import get_embeddings

logger = logging.getLogger(__name__)

DOC_EXTENSIONS = (".md", ".txt", ".py")
CODE_EXTENSIONS = (
    ".py",
    ".md",
    ".txt",
    ".json",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".go",
    ".rs",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".yml",
    ".yaml",
    ".sh",
    ".sql",
    ".tf",
    ".dockerfile",
)
CODE_FILENAMES = ("Dockerfile",)
IGNORED_DIRS = frozenset(
    {
        "node_modules",
        ".git",
        "venv",
        ".venv",
        "env",
        "__pycache__",
        "build",
        "dist",
        "target",
        ".next",
        ".nuxt",
        "out",
        "pods",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)
MAX_FILE_BYTES = 500 * 1024  # skips lockfiles, minified bundles and data dumps


class IngestionError(Exception):
    """Raised when a source cannot be ingested (bad path, nothing to index, clone failure)."""


@dataclass(frozen=True)
class IngestionResult:
    source: str
    files: int
    chunks: int

    @property
    def message(self) -> str:
        return f"Indexed {self.chunks} chunks from {self.files} files in {self.source}."


def collect_documents(
    root: Path,
    extensions: Iterable[str],
    *,
    filenames: Iterable[str] = (),
    ignored_dirs: Iterable[str] = IGNORED_DIRS,
    extra_metadata: dict | None = None,
) -> list[Document]:
    """Walk `root` and load matching text files as Documents.

    `source` metadata is the path relative to `root` (POSIX style), so citations stay
    stable no matter where the files were cloned or mounted.
    """
    root = Path(root)
    exts = tuple(e.lower() for e in extensions)
    names = set(filenames)
    ignored = set(ignored_dirs)
    documents: list[Document] = []

    for dirpath, dirnames, files in os.walk(root):
        # Prune in place so os.walk never descends into ignored directories.
        # Virtual environments can have any name, so also skip any folder with a pyvenv.cfg.
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in ignored and not (Path(dirpath) / d / "pyvenv.cfg").exists()
        )
        for name in sorted(files):
            if not (name.lower().endswith(exts) or name in names):
                continue
            path = Path(dirpath) / name
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                logger.warning("Could not read %s", path, exc_info=True)
                continue
            if not text.strip():
                continue
            relative = path.relative_to(root).as_posix()
            metadata = {"source": relative, "file_name": name, **(extra_metadata or {})}
            documents.append(Document(page_content=text, metadata=metadata))

    return documents


def split_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def index_chunks(chunks: list[Document], *, recreate: bool) -> None:
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        url=settings.qdrant_url,
        collection_name=settings.QDRANT_COLLECTION,
        force_recreate=recreate,
        batch_size=64,
    )


def ingest_directory(
    directory_path: str | Path,
    *,
    recreate: bool = True,
    ignored_dirs: Iterable[str] = IGNORED_DIRS,
) -> IngestionResult:
    """Index .md, .txt and .py files under a local folder.

    With recreate=True (the default) the collection is rebuilt from scratch.
    """
    path = Path(directory_path).expanduser().resolve()
    if not path.is_dir():
        raise IngestionError(f"Directory not found: {path}")

    documents = collect_documents(path, DOC_EXTENSIONS, ignored_dirs=ignored_dirs)
    if not documents:
        raise IngestionError(f"No .md, .txt or .py files found in {path}")

    chunks = split_documents(documents, chunk_overlap=200)
    logger.info("Indexing %d chunks from %d files in %s", len(chunks), len(documents), path)
    index_chunks(chunks, recreate=recreate)
    return IngestionResult(source=str(path), files=len(documents), chunks=len(chunks))
