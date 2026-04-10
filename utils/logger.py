"""
utils/logger.py
===============
Centralised logging configuration.
All modules call get_logger(__name__) to get a consistent logger.
"""

import logging
import os
import sys

_LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results")
_LOG_FILE = os.path.join(_LOG_DIR, "run.log")

os.makedirs(_LOG_DIR, exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_FMT,
    datefmt=_DATE_FMT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8"),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the project-wide configuration."""
    return logging.getLogger(name)
