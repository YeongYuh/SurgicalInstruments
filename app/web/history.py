from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import app.config as config

# Module-level so tests can redirect it without touching the real output dir.
HISTORY_FILE = Path(config.OUTPUT_DIR) / "history.json"
MAX_RECORDS = 500
_lock = threading.Lock()

# Kept for callers that imported the private names before packages existed.
_HISTORY_FILE = HISTORY_FILE
_MAX_RECORDS = MAX_RECORDS


def load_history() -> List[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def append_record(record: dict) -> None:
    with _lock:
        records = load_history()
        records.append(record)
        if len(records) > MAX_RECORDS:
            records = records[-MAX_RECORDS:]
        _atomic_write(HISTORY_FILE, json.dumps(records, ensure_ascii=False, indent=2))


def make_record(
    source: str,
    counts: dict,
    weight: Optional[float],
    standards_snapshot: dict,
    *,
    package_id: Optional[str] = None,
    package_display_name: Optional[str] = None,
    model_identity: Optional[Mapping[str, Any]] = None,
    class_weights_snapshot: Optional[Mapping[str, float]] = None,
    weight_verification: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build one inventory record.

    The package fields are what make a stored count interpretable later: after
    the active package changes, the only correct way to read an old record is
    with the standards and weights that were in force when it was taken.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    uid = uuid.uuid4().hex[:6]
    record: Dict[str, Any] = {
        "id": f"{ts.replace(' ', '_').replace(':', '-')}_{uid}",
        "timestamp": ts,
        "source": source,
        "counts": counts,
        "weight": weight,
        "standards_snapshot": standards_snapshot,
        "package_id": package_id,
        "package_display_name": package_display_name,
    }
    if model_identity:
        record["model"] = dict(model_identity)
    if class_weights_snapshot is not None:
        record["class_weights_snapshot"] = dict(class_weights_snapshot)
    if weight_verification is not None:
        record["weight_verification"] = dict(weight_verification)
    return record


def record_standards(record: Mapping[str, Any],
                     fallback: Optional[Mapping[str, int]] = None) -> Dict[str, int]:
    """Standards that apply to ``record``.

    Always prefers the snapshot stored with the record.  The fallback is only
    for records written before snapshots existed — using the *current* active
    standards for a record taken under a different package would report an
    obstetric tray against orthopaedic expectations.
    """
    snapshot = record.get("standards_snapshot")
    if isinstance(snapshot, dict):
        return {str(k): v for k, v in snapshot.items()}
    return dict(fallback or {})


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
