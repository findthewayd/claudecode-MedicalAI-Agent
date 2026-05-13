"""Structured JSON logging with patient ID masking."""
import logging
import json
import re
import sys
import hashlib
from datetime import datetime, timezone


class MaskFormatter(logging.Formatter):
    """JSON formatter that masks PII like patient names."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }

        # Mask patient_id: keep first 2 chars, hash the rest
        msg = record.getMessage()
        msg = re.sub(r'(patient_id[\'"]?\s*[:=]\s*[\'"]?)([^\s,\'"]+)',
                     lambda m: m.group(1) + mask_id(m.group(2)), msg)
        log_entry["message"] = msg

        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        if record.exc_info and record.exc_info[1]:
            log_entry["error"] = str(record.exc_info[1])
            log_entry["error_type"] = type(record.exc_info[1]).__name__

        return json.dumps(log_entry, ensure_ascii=False)


def mask_id(value: str) -> str:
    if len(value) <= 2:
        return value[:1] + "*"
    return value[:2] + hashlib.md5(value.encode()).hexdigest()[:6]


def setup_logging(level: str = "INFO"):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(MaskFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Silence noisy libs
    for name in ["httpx", "httpcore", "urllib3", "chromadb", "neo4j"]:
        logging.getLogger(name).setLevel(logging.WARNING)
