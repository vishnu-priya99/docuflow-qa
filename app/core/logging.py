"""Application-wide logging configuration."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Under data/ (already gitignored - see .gitignore) so log files never get
# committed. Rotated at 20MB x 5 backups so a long dev/demo session with
# full-chunk-text logging (see ingestion_service.py etc.) can't silently
# fill the disk.
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"
_LOG_FILE = _LOG_DIR / "app.log"


def configure_logging(level: str = "INFO") -> None:
    # Under `uvicorn --reload` on Windows, the actual app runs in a child
    # subprocess whose stdout is piped back to the parent terminal.
    # logging.StreamHandler already flushes Python's own buffer after every
    # record, but that only guarantees the write leaves Python - it does
    # not guarantee the OS pipe/console between the child and parent
    # delivers it to the screen immediately. Forcing the underlying stream
    # into line-buffered mode closes that gap for the console handler.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. re-imported under uvicorn's reloader).
        root.setLevel(level.upper())
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # Second, independent sink straight to a file - bypasses whatever is
    # eating console output (reloader subprocess pipes, terminal quirks,
    # etc. - see the investigation this was added for). A file write is
    # reliable regardless of what's happening upstream of the terminal;
    # reading it afterward with `Get-Content -Wait` (or `tail -f`) or just
    # reopening it is the fallback source of truth for these logs.
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    root.setLevel(level.upper())

    # Keep noisy third-party loggers at a sane level.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging to console and to %s", _LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
