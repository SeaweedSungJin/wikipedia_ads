"""Minimal logging setup for consistent messages across modules."""
from __future__ import annotations

import logging
import os


def get_logger(name: str) -> logging.Logger:
    level = os.environ.get("WIKI_LOGLEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

