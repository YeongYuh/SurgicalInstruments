from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

import app.config as config

_HISTORY_FILE = Path(config.OUTPUT_DIR) / "history.json"
_MAX_RECORDS = 500
_lock = threading.Lock()


def load_history() -> list[dict]:
    if not _HISTORY_FILE.exists():
        return []
    try:
        return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def append_record(record: dict) -> None:
    with _lock:
        records = load_history()
        records.append(record)
        if len(records) > _MAX_RECORDS:
            records = records[-_MAX_RECORDS:]
        _atomic_write(_HISTORY_FILE, json.dumps(records, ensure_ascii=False, indent=2))


def make_record(
    source: str,
    counts: dict,
    weight: float | None,
    standards_snapshot: dict,
) -> dict:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uid = uuid.uuid4().hex[:6]
    return {
        "id": f"{ts.replace(' ', '_').replace(':', '-')}_{uid}",
        "timestamp": ts,
        "source": source,
        "counts": counts,
        "weight": weight,
        "standards_snapshot": standards_snapshot,
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
