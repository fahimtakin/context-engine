# app/services/git_ingestion.py
import os
import sys
import shutil
from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from app.db.qdrant_client import COLLECTION_NAME
from app.core.config import settings

def ingest_github_repository(repo_url: str, branch: str = "main"):
    """
    Dynamically clones any target GitHub repository into a temporary sandbox volume,
    safely filters and parses code modules, and writes chunks directly to Qdrant.
    """
    # Create a unique path using the repo name to prevent folder collision on simultaneous requests
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    temp_clone_dir = os.path.abspath(f"./tmp/context_engine_{repo_name}")
    
    if os.path.exists(temp_clone_dir):
        try:
            shutil.rmtree(temp_clone_dir)
        except Exception:
            pass
        
    print(f"Cloning remote workspace target pipeline: {repo_url} [{branch}]...", file=sys.stderr, flush=True)
    
    try:
        # Clones only the latest commit layer (depth=1) to keep memory footprint incredibly light
        Repo.clone_from(repo_url, temp_clone_dir, branch=branch, depth=1)
        
        # Broad target matrix to ensure full coverage of modern code architectures
        valid_extensions = (
            '.py', '.md', '.txt', '.json', '.js', '.jsx', '.ts', '.tsx', 
            '.html', '.css', '.go', '.rs', '.java', '.cpp', '.c', '.h', 
            '.cs', '.yml', '.yaml', '.sh', '.sql', '.tf', '.dockerfile', 'Dockerfile'
        )
        
        # Massive production directories to skip to prevent vector DB pollution
        ignored_directories = [
            "node_modules", ".git", "venv", ".venv", "__pycache__", 
            "build", "dist", "target", ".next", ".nuxt", "out", "pods"
        ]
        
        documents = []
        
        for root, _, files in os.walk(temp_clone_dir):
            if any(ignored in root.split(os.sep) for ignored in ignored_directories):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip files larger than 500KB (lockfiles/minified scripts)
                try:
                    if os.path.getsize(file_path) > 500 * 1024:
                        continue
                except Exception:
                    continue

                if file.lower().endswith(valid_extensions) or file == 'Dockerfile':
                    relative_path = os.path.relpath(file_path, temp_clone_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                            if not text.strip():
                                continue
                                
                            # Enriched metadata pattern: giving the LLM deep awareness of origin structures
                            metadata = {
                                "source": relative_path, 
                                "repo_url": repo_url,
                                "file_name": file,
                                "file_path": relative_path
                            }
                            documents.append(Document(page_content=text, metadata=metadata))
                    except Exception as parse_err:
                        print(f"Skipping trace item '{relative_path}': {str(parse_err)}", file=sys.stderr, flush=True)

        if not documents:
            return f"Error: No valid codebase source files discovered matching supported extensions inside {repo_url}."

        # Optimized split overlap for code architectures (Code benefits from smaller chunk sizes for context purity)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = text_splitter.split_documents(documents)
        print(f"Codebase split successfully into {len(split_docs)} vector tokens.", file=sys.stderr, flush=True)

        # Connect to Embeddings engine
        from app.services.ingestion import resolve_ollama_endpoint
        resolved_url = resolve_ollama_endpoint()
        embeddings = OllamaEmbeddings(base_url=resolved_url, model="mxbai-embed-large")
        
        host_target = "qdrant" if os.path.exists("/.dockerenv") else settings.QDRANT_HOST
        qdrant_url = f"http://{host_target}:{settings.QDRANT_PORT}"
        
        # Write chunks incrementally to prevent Qdrant request payload dropouts
        QdrantVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings,
            url=qdrant_url,
            collection_name=COLLECTION_NAME,
            batch_size=64 # Limits memory spikes when embedding massive external repositories
        )
        return f"Success! Vectorized {len(split_docs)} code segments directly from GitHub: {repo_name}."
        
    finally:
        # Wipes scratchpad completely to secure host disk space footprints
        if os.path.exists(temp_clone_dir):
            try:
                shutil.rmtree(temp_clone_dir)
                print("Scratchpad directory wiped clean.", file=sys.stderr, flush=True)
            except Exception:
                pass
