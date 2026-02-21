"""Shared logging setup for scripts and training."""

import logging
import os


def setup_logging(
    level: str | None = None,
    format_string: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """Configure root logger. Level can be overridden via LOG_LEVEL env var."""
    log_level = level or os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
