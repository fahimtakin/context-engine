"""Logging setup.

Everything goes to stderr: the MCP server talks JSON-RPC over stdout, so a stray
print() to stdout would corrupt the protocol stream.
"""

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Third-party clients are chatty at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True
