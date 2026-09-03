"""Structured logging tối thiểu, không ghi PDF/câu hỏi/câu trả lời."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


_SAFE_FIELDS = (
    "event",
    "request_id",
    "stage",
    "route",
    "round",
    "duration_ms",
    "candidate_count",
    "source_count",
    "confidence_bucket",
    "reason_code",
)


class PrivacyJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
        }
        for field in _SAFE_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info and "reason_code" not in payload:
            payload["reason_code"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("rag_app")
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = PrivacyJsonFormatter()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "rag.jsonl", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
