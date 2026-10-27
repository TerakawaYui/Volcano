"""Stable serialization, signatures and atomic status writes."""
import csv
import hashlib
import json
import math
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def dump_json(path, value):
    """Atomic JSON writes tolerate transient Windows reader/antivirus locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="."+path.name+".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        for attempt in range(16):
            try:
                os.replace(temporary, path)
                return True
            except PermissionError:
                if attempt == 15:
                    if path.name != "progress.json":
                        raise
                    warnings.warn("Progress JSON is locked; skipped this update. Completed tiles remain in SQLite.",
                                  RuntimeWarning, stacklevel=1)
                    return False
                time.sleep(min(.05 * 2**attempt, .5))
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def signature(fits_path, config, files):
    stat = Path(fits_path).stat()
    digest = hashlib.sha256()
    digest.update(f"{Path(fits_path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    digest.update(json.dumps(config, sort_keys=True).encode())
    for path in sorted(map(Path, files)):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
