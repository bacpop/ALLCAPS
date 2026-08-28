"""Central logging configuration for the pneumococcal-serotyping pipeline.

Call :func:`get_logger` from any module to obtain a configured logger. The first
call configures the root handler with a timestamped formatter; subsequent calls
return per-module loggers that inherit from the root.
"""

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def configure_logging(level: int | str | None = None) -> None:
    """Configure the root logger with a timestamped stream handler.

    Idempotent: repeated calls are no-ops unless the level changes. The level
    can also be set via the ``PNEUMO_LOG_LEVEL`` environment variable.
    """
    global _CONFIGURED

    if level is None:
        level = os.environ.get("PNEUMO_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    root = logging.getLogger()
    root.setLevel(level)

    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name``, configuring the root handler on first use."""
    configure_logging()
    return logging.getLogger(name)
