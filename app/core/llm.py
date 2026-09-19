# app/core/llm.py
import sys
import os
import socket
from langchain_ollama import ChatOllama
from app.core.config import settings

class MockLLM:
    """A safe local backup engine that handles dropped container network bridges cleanly."""
    def invoke(self, messages, **kwargs):
        class MockResponse:
            content = "ContextEngine Runtime Network Warning: The container network interface could not establish a connection back to your host machine's Ollama application window port."
        return MockResponse()
    
    def with_structured_output(self, schema, **kwargs):
        class MockStructured:
            def invoke(self, messages, **kwargs):
                class Decision:
                    choice = "vector_db"
                return Decision()
        return MockStructured()

def resolve_ollama_endpoint() -> str:
    """
    Dynamically cross-references internal container network adapters to locate
    the host's active Ollama socket binding, resolving firewall lockouts automatically.
    """
    base_url = settings.OLLAMA_BASE_URL
    
    # If we are not running inside a Docker sandbox container, use local paths instantly
    if not os.path.exists("/.dockerenv") and os.environ.get("DOCKER_CONTAINER") != "true":
        return base_url

    # Matrix array of every possible bridge gateway IP route back to a Windows host machine
    network_gateways = ["host.docker.internal", "172.17.0.1", "172.18.0.1", "10.0.75.2", "192.168.65.2"]
    
    for gateway in network_gateways:
        try:
            # Performs a low-level socket handshake check to verify if the port is awake
            host_ip = socket.gethostbyname(gateway)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5) # Fast test loop
            s.connect((host_ip, 11434))
            s.close()
            # Verified route tracked! Return the clean target configuration address string
            print(f"Ingestion network bridge mapped to host card via: http://{gateway}:11434", file=sys.stderr, flush=True)
            return f"http://{gateway}:11434"
        except Exception:
            continue
            
    return base_url

def get_llm():
    """Returns a resilient ChatOllama instance mapped to the active network layer endpoint."""
    resolved_url = resolve_ollama_endpoint()
    
    try:
        return ChatOllama(
            base_url=resolved_url,
            model="llama3.1", 
            temperature=0,
            timeout=45.0   
        )
    except Exception:
        return MockLLM()
