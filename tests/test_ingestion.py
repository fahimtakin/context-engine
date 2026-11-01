from pathlib import Path

import pytest

from app.services import git_ingestion, ingestion
from app.services.git_ingestion import ingest_github_repository, validate_repo_url
from app.services.ingestion import (
    CODE_EXTENSIONS,
    DOC_EXTENSIONS,
    MAX_FILE_BYTES,
    IngestionError,
    collect_documents,
    ingest_directory,
)


def write(root: Path, rel: str, text: str = "content") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sources(docs) -> list[str]:
    return sorted(d.metadata["source"] for d in docs)


def test_collects_matching_files_with_relative_posix_paths(tmp_path):
    write(tmp_path, "README.md")
    write(tmp_path, "app/main.py")
    write(tmp_path, "app/data.csv")  # wrong extension

    docs = collect_documents(tmp_path, DOC_EXTENSIONS)

    assert sources(docs) == ["README.md", "app/main.py"]


def test_skips_ignored_dirs_by_exact_name_only(tmp_path):
    write(tmp_path, "node_modules/pkg/index.md")
    write(tmp_path, "build/out.md")
    write(tmp_path, "rebuild_tools/notes.md")  # 'build' substring must not exclude this

    docs = collect_documents(tmp_path, DOC_EXTENSIONS)

    assert sources(docs) == ["rebuild_tools/notes.md"]


def test_skips_virtualenvs_with_any_name(tmp_path):
    write(tmp_path, "myenv/pyvenv.cfg", "home = /usr/bin")
    write(tmp_path, "myenv/lib/site.py")
    write(tmp_path, "main.py")

    assert sources(collect_documents(tmp_path, DOC_EXTENSIONS)) == ["main.py"]


def test_skips_empty_and_oversized_files(tmp_path):
    write(tmp_path, "empty.md", "   \n")
    write(tmp_path, "huge.md", "x" * (MAX_FILE_BYTES + 1))
    write(tmp_path, "ok.md")

    assert sources(collect_documents(tmp_path, DOC_EXTENSIONS)) == ["ok.md"]


def test_matches_extensionless_filenames(tmp_path):
    write(tmp_path, "Dockerfile", "FROM python")
    docs = collect_documents(tmp_path, CODE_EXTENSIONS, filenames=("Dockerfile",))
    assert sources(docs) == ["Dockerfile"]


def test_extra_metadata_is_attached(tmp_path):
    write(tmp_path, "a.py")
    [d] = collect_documents(tmp_path, DOC_EXTENSIONS, extra_metadata={"repo_url": "https://x/y"})
    assert d.metadata["repo_url"] == "https://x/y"


def test_ingest_directory_errors_clearly(tmp_path):
    with pytest.raises(IngestionError, match="not found"):
        ingest_directory(tmp_path / "missing")
    with pytest.raises(IngestionError, match="No .md"):
        ingest_directory(tmp_path)


def test_ingest_directory_indexes_chunks(tmp_path, monkeypatch):
    write(tmp_path, "a.md", "hello " * 400)
    calls = []
    monkeypatch.setattr(
        ingestion, "index_chunks", lambda chunks, recreate: calls.append((len(chunks), recreate))
    )

    result = ingest_directory(tmp_path)

    assert result.files == 1 and result.chunks >= 2
    assert calls == [(result.chunks, True)]


@pytest.mark.parametrize(
    "url", ["https://github.com/fahimtakin/context-engine", " https://gitlab.com/a/b.git "]
)
def test_accepts_https_repo_urls(url):
    assert validate_repo_url(url) == url.strip()


@pytest.mark.parametrize(
    "url",
    [
        "",
        "file:///etc",
        "/etc/passwd",
        "git@github.com:a/b.git",
        "http://github.com/a/b",
        "https://github.com",
        "ssh://host/repo",
    ],
)
def test_rejects_unsafe_or_malformed_repo_urls(url):
    with pytest.raises(IngestionError):
        validate_repo_url(url)


def test_ingest_github_repository_indexes_clone(monkeypatch):
    def fake_clone(url, target, branch, depth):
        write(Path(target), "src/app.py", "print('hi')")
        write(Path(target), "Dockerfile", "FROM python")
        write(Path(target), "node_modules/x/index.js", "ignored")

    indexed = []
    monkeypatch.setattr(git_ingestion.Repo, "clone_from", fake_clone)
    monkeypatch.setattr(
        git_ingestion, "index_chunks", lambda chunks, recreate: indexed.append((chunks, recreate))
    )

    result = ingest_github_repository("https://github.com/o/r", "main")

    assert result.files == 2
    chunks, recreate = indexed[0]
    assert recreate is False
    assert {c.metadata["repo_url"] for c in chunks} == {"https://github.com/o/r"}


def test_clone_failure_becomes_ingestion_error(monkeypatch):
    from git import GitCommandError

    def boom(*args, **kwargs):
        raise GitCommandError("clone", 128)

    monkeypatch.setattr(git_ingestion.Repo, "clone_from", boom)
    with pytest.raises(IngestionError, match="Could not clone"):
        ingest_github_repository("https://github.com/o/missing")
