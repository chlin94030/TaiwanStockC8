"""Production-safe operational helpers for Alpha Radar V16."""
from __future__ import annotations

from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import tempfile
import time


class ScanBusyError(RuntimeError):
    """Raised when another market scan already owns the cross-session lock."""


def safe_error_text(exc: Exception | str, max_len: int = 300, mask_tokens: tuple[str, ...] = ()) -> str:
    msg = str(exc)
    for token in mask_tokens:
        if token and isinstance(token, str):
            msg = msg.replace(token, "*****")
    return msg[:max_len]


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write a text file atomically in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _lock_age_seconds(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except Exception:
        return None


@contextmanager
def scan_lock(lock_path: Path, stale_after_seconds: int = 45 * 60):
    """Cross-session file lock with stale-lock recovery.

    Streamlit can serve multiple browser sessions concurrently.  Without a
    process-level guard, two users clicking refresh can write the same SQLite
    database and dashboard snapshot at once.  O_EXCL gives us a lightweight
    lock that also works without external dependencies.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    age = _lock_age_seconds(lock_path)
    if age is not None and age > stale_after_seconds:
        try:
            lock_path.unlink()
        except Exception:
            pass

    fd = None
    try:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            age = _lock_age_seconds(lock_path)
            suffix = f"（已持續約 {int(age)} 秒）" if age is not None else ""
            raise ScanBusyError("另一個更新正在執行" + suffix) from exc

        payload = {
            "pid": os.getpid(),
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        os.fsync(fd)
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                lock_path.unlink()
            except Exception:
                pass
