from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import tempfile


def configure_logging(root: Path) -> Path:
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    stream_handler = _ensure_stream_handler(root_logger, formatter)

    log_path = _select_log_path(root)
    if log_path is not None and not _already_configured(root_logger, log_path):
        try:
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
        except OSError as exc:
            root_logger.warning("[logging] file logging disabled path=%s err=%s", log_path, exc)
        else:
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
    if stream_handler is not None and log_path is not None:
        root_logger.info("[logging] using log file path=%s", log_path)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

    return log_path or Path("-")


def _already_configured(root_logger: logging.Logger, log_path: Path) -> bool:
    target = str(log_path.resolve())
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            if str(Path(handler.baseFilename).resolve()) == target:
                return True
    return False


def _ensure_stream_handler(root_logger: logging.Logger, formatter: logging.Formatter) -> logging.Handler | None:
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            return handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)
    return stream_handler


def _select_log_path(root: Path) -> Path | None:
    for directory in _candidate_log_dirs(root):
        if directory is None:
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if not os.access(directory, os.W_OK):
            continue
        return directory / "claude-sdk-agent.log"
    return None


def _candidate_log_dirs(root: Path) -> list[Path | None]:
    tmp_root = Path(tempfile.gettempdir()) / "claude-sdk-agent"
    user = os.getenv("USER") or os.getenv("LOGNAME") or "unknown"
    return [
        root / "log",
        tmp_root / user,
        tmp_root,
    ]
