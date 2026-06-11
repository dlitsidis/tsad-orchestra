"""Centralized logging configuration for TSAD Orchestra.

Call ``setup_logging()`` once at process startup — both the main CLI process
and the MCP server subprocess use this so that every log line ends up in a
predictable place.

Logs are written to:
    logs/tsad_orchestra.log   — main / agent process
    logs/mcp_server.log       — MCP server subprocess

Both sinks also mirror to stderr so you see output in the terminal when
running interactively.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False

# Repository root (two levels up from src/logging_config.py)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOG_DIR = _REPO_ROOT / "logs"


def setup_logging(
    *,
    process_name: str = "tsad_orchestra",
    level: str | None = None,
    log_dir: Path | None = None,
) -> Path:
    """Configure loguru sinks for the current process.

    Args:
        process_name: Base name for the log file (without extension).
            Use ``"tsad_orchestra"`` for the main process and
            ``"mcp_server"`` for the MCP subprocess.
        level: Minimum log level.  Falls back to the ``TSAD_LOG_LEVEL``
            environment variable, then to ``"DEBUG"``.
        log_dir: Directory for log files.  Defaults to ``<repo>/logs/``.

    Returns:
        Path to the log file that was created.
    """
    global _CONFIGURED  # noqa: PLW0603

    if _CONFIGURED:
        return (log_dir or _LOG_DIR) / f"{process_name}.log"

    resolved_level = (level or os.getenv("TSAD_LOG_LEVEL", "DEBUG")).upper()
    resolved_dir = log_dir or _LOG_DIR
    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_dir / f"{process_name}.log"

    # Remove loguru's default stderr handler so we can replace it with our
    # own formatted version.
    logger.remove()


    logger.add(
        sys.stderr,
        level=resolved_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )


    logger.add(
        str(log_path),
        level="DEBUG",           # always capture everything in the file
        rotation="5 MB",
        retention=5,
        compression="gz",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} — "
            "{message}"
        ),
        backtrace=True,
        diagnose=True,
    )

    logger.info(
        "Logging initialised — process={}, level={}, file={}",
        process_name,
        resolved_level,
        log_path,
    )

    _CONFIGURED = True
    return log_path
