"""Index the local ./documents folder (or a folder passed as an argument) into Qdrant."""

import argparse
import logging

from app.core.config import settings
from app.core.logging_setup import configure_logging
from app.db.qdrant_client import init_qdrant_collection
from app.services.ingestion import ingest_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="./documents", help="folder to index")
    args = parser.parse_args()

    configure_logging(settings.LOG_LEVEL)
    init_qdrant_collection()
    result = ingest_directory(args.path)
    logging.getLogger("run_ingestion").info(result.message)


if __name__ == "__main__":
    main()
