"""One-line logger factory shared by the whole system and the runners."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "mmdataselect", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(level)
        _CONFIGURED = True
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
