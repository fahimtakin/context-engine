# run_ingestion.py
from app.db.qdrant_client import init_qdrant_collection
from app.services.ingestion import ingest_directory

if __name__ == "__main__":
    print("Initializing collection...")
    init_qdrant_collection()

    print("\nStarting data ingestion script...")
    ingest_directory("./documents")
    print("\nPipeline complete!")
