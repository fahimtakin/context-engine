# app/services/ingestion.py
import os
import sys
import socket
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from app.db.qdrant_client import COLLECTION_NAME
from app.core.config import settings

def resolve_ollama_endpoint() -> str:
    """
    Dynamically cross-references internal container network adapters to locate
    the host's active Ollama socket binding, resolving firewall lockouts automatically.
    """
    base_url = settings.OLLAMA_BASE_URL
    
    # If we are not running inside a Docker sandbox container, use local paths instantly
    if not os.path.exists("/.dockerenv") and os.environ.get("DOCKER_CONTAINER") != "true":
        return base_url

    # Matrix array of every possible bridge gateway IP route back to a Windows/Mac host machine
    network_gateways = ["host.docker.internal", "172.17.0.1", "172.18.0.1", "10.0.75.2", "192.168.65.2"]
    
    for gateway in network_gateways:
        try:
            # Performs a low-level socket handshake check to verify if the port is awake
            host_ip = socket.gethostbyname(gateway)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5) # Fast test loop
            s.connect((host_ip, 11434))
            s.close()
            print(f"Docker network bridge successfully mapped back to host card via: http://{gateway}:11434", file=sys.stderr, flush=True)
            return f"http://{gateway}:11434"
        except Exception:
            continue
            
    return base_url

def ingest_directory(directory_path: str):
    # 1. Force paths to resolve correctly back to your true workspace root
    if not os.path.isabs(directory_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        directory_path = os.path.join(base_dir, directory_path)

    if not os.path.exists(directory_path):
        print(f"Error: Target path does not exist: {directory_path}", file=sys.stderr)
        return

    documents = []
    print(f"Scanning files inside absolute path: {directory_path}", file=sys.stderr)

    for root, _, files in os.walk(directory_path):
        # 2. CRITICAL: Skip node_modules, virtual environments, and caches entirely
        if any(ignored in root for ignored in ["node_modules", "venv", ".git", "__pycache__"]):
            continue

        for file in files:
            if file.endswith(('.txt', '.md', '.py')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        metadata = {"source": file, "path": file_path}
                        documents.append(Document(page_content=text, metadata=metadata))
                        print(f"Found & loaded: {file}", file=sys.stderr)
                except Exception as e:
                    print(f"Failed to read file {file_path}: {e}", file=sys.stderr)

    if not documents:
        print("Error: No valid files were successfully read!", file=sys.stderr)
        return

    # 3. Cleanly slice up your discovered files
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    
    print(f"Split {len(documents)} source files into {len(split_docs)} text chunks.", file=sys.stderr)

    # 4. Resolve the network endpoint (Restored!)
    resolved_url = resolve_ollama_endpoint()
    embeddings = OllamaEmbeddings(
        base_url=resolved_url,
        model="mxbai-embed-large"
    )
  
    host_target = "qdrant" if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER") == "true" else settings.QDRANT_HOST
    qdrant_url = f"http://{host_target}:{settings.QDRANT_PORT}"
    
    # 5. Overwrite the Qdrant database index fresh
    print(f"Overwriting Qdrant collection '{COLLECTION_NAME}' with real data...", file=sys.stderr)
    QdrantVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        url=qdrant_url,
        collection_name=COLLECTION_NAME,
        force_recreate=True
    )
    print(f"Success! Vector database completely re-indexed.", file=sys.stderr)
